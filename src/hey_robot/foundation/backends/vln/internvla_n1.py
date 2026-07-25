from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from hey_robot.foundation.backends.vln.models import (
    VLNPlannerInput,
    VLNPlannerResult,
    VLNPlanningError,
)

DEFAULT_PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. "
    "Your task is to <instruction>. "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)

_MODEL_LOAD_LOCK = threading.Lock()


class InternVLAN1Runtime:
    """Own only the third-party InternNav model lifecycle and inference."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self._model: Any | None = None
        self._current_policy_session_id: str | None = None
        self._pending_actions: list[int] = []
        self._last_llm_output: str | None = None
        # InternVLA keeps history on the policy object.  Serialize load, reset,
        # and inference so concurrent gRPC calls cannot corrupt that state.
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self._model is not None:
            return
        repo_path = self._internnav_repo_path()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from internnav.model import get_config, get_policy
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
        )

        policy_name = str(self.settings.get("policy_name") or "InternVLAN1_Policy")
        policy_cls = get_policy(policy_name)
        config_cls = get_config(policy_name)
        model_settings = self._model_settings(policy_name)

        with _MODEL_LOAD_LOCK:
            had_local_loader = "from_pretrained" in InternVLAN1ForCausalLM.__dict__
            original_descriptor = InternVLAN1ForCausalLM.__dict__.get("from_pretrained")
            original_loader = InternVLAN1ForCausalLM.from_pretrained
            attention = str(self.settings.get("attn_implementation") or "sdpa")

            @classmethod  # type: ignore[misc]
            def patched_loader(cls: Any, *args: Any, **kwargs: Any) -> Any:
                del cls
                if kwargs.get("attn_implementation") == "flash_attention_2":
                    kwargs["attn_implementation"] = attention
                return original_loader(*args, **kwargs)

            InternVLAN1ForCausalLM.from_pretrained = patched_loader
            try:
                model = policy_cls(
                    config=config_cls(model_cfg={"model": model_settings})
                )
            finally:
                if had_local_loader:
                    InternVLAN1ForCausalLM.from_pretrained = original_descriptor
                else:
                    del InternVLAN1ForCausalLM.from_pretrained

        self._ensure_latent_queries(model)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        self._model = model

    def plan(
        self,
        planner_input: VLNPlannerInput,
        *,
        policy_session_id: str | None,
        reset_policy: bool,
    ) -> VLNPlannerResult:
        with self._lock:
            self._load_unlocked()
            model = self._model
            assert model is not None
            self._reset_policy_session(
                model,
                policy_session_id=policy_session_id,
                reset_policy=reset_policy,
            )
            self._apply_prompt_override(model)
            if self._pending_actions and not planner_input.look_down:
                step_no_infer = getattr(model, "step_no_infer", None)
                if callable(step_no_infer):
                    step_no_infer(
                        planner_input.rgb,
                        planner_input.depth,
                        planner_input.pose,
                    )
                current_action = self._pending_actions.pop(0)
                return planner_result_from_output(
                    None,
                    image_width=int(self.settings.get("resize_w", 384)),
                    image_height=int(self.settings.get("resize_h", 384)),
                    image_source=planner_input.image_source,
                    policy_session_id=policy_session_id,
                    action_sequence=[current_action],
                    remaining_action_count=len(self._pending_actions),
                    raw_output=self._last_llm_output,
                    turn_angle_deg=float(self.settings.get("discrete_turn_deg", 15.0)),
                    forward_distance_cm=float(
                        self.settings.get("discrete_forward_cm", 25.0)
                    ),
                )
            output = model.s2_step(
                planner_input.rgb,
                planner_input.depth,
                planner_input.pose,
                planner_input.instruction,
                planner_input.intrinsic,
                planner_input.look_down,
            )
            self._last_llm_output = str(getattr(model, "llm_output", "") or "") or None
            action_sequence = action_codes_from_output(output)
            self._pending_actions = action_sequence[1:]
            if action_sequence and action_sequence[0] in {0, 5}:
                self._pending_actions.clear()
        return planner_result_from_output(
            output,
            image_width=int(self.settings.get("resize_w", 384)),
            image_height=int(self.settings.get("resize_h", 384)),
            image_source=planner_input.image_source,
            policy_session_id=policy_session_id,
            action_sequence=action_sequence or None,
            remaining_action_count=len(self._pending_actions),
            raw_output=self._last_llm_output,
            turn_angle_deg=float(self.settings.get("discrete_turn_deg", 15.0)),
            forward_distance_cm=float(self.settings.get("discrete_forward_cm", 25.0)),
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._current_policy_session_id = None
            self._pending_actions.clear()
            self._last_llm_output = None

    def _internnav_repo_path(self) -> Path:
        value = str(self.settings.get("internnav_repo") or "").strip()
        if not value:
            raise RuntimeError("InternVLA-N1 requires model setting internnav_repo")
        path = Path(value).expanduser().resolve()
        if not (path / "internnav" / "__init__.py").is_file():
            raise RuntimeError(f"invalid InternNav repository: {path}")
        return path

    def _model_settings(self, policy_name: str) -> dict[str, Any]:
        model_path = str(self.settings.get("model_path") or "").strip()
        if not model_path:
            raise RuntimeError("InternVLA-N1 requires model setting model_path")
        return {
            "policy_name": policy_name,
            "state_encoder": None,
            "mode": "system2",
            "model_path": model_path,
            "device": self.settings.get("device", "cuda"),
            "dtype": self.settings.get(
                "dtype", self.settings.get("torch_dtype", "auto")
            ),
            "torch_dtype": self.settings.get(
                "torch_dtype", self.settings.get("dtype", "auto")
            ),
            "attn_implementation": self.settings.get("attn_implementation", "sdpa"),
            "num_history": int(self.settings.get("num_history", 8)),
            "resize_w": int(
                self.settings.get("resize_w", self.settings.get("image_width", 384))
            ),
            "resize_h": int(
                self.settings.get("resize_h", self.settings.get("image_height", 384))
            ),
            "max_new_tokens": int(self.settings.get("max_new_tokens", 128)),
            "num_frames": int(self.settings.get("num_frames", 8)),
            "num_future_steps": int(self.settings.get("num_future_steps", 0)),
            "continuous_traj": bool(self.settings.get("continuous_traj", False)),
            "n_query": int(self.settings.get("n_query", 4)),
            "vis_debug": bool(self.settings.get("vis_debug", False)),
            "vis_debug_path": self.settings.get("vis_debug_path", "./logs/vln_debug"),
        }

    def _ensure_latent_queries(self, model: Any) -> None:
        import torch

        n_query = int(self.settings.get("n_query", 4))
        inner = model.model
        if not hasattr(inner.config, "n_query"):
            inner.config.n_query = n_query
        vlm = inner.get_model()
        if not hasattr(vlm, "latent_queries") or vlm.latent_queries is None:
            vlm.latent_queries = torch.nn.Parameter(
                torch.randn(1, n_query, vlm.config.hidden_size)
            )

    def _reset_policy_session(
        self,
        model: Any,
        *,
        policy_session_id: str | None,
        reset_policy: bool,
    ) -> None:
        should_reset = reset_policy or bool(
            policy_session_id and policy_session_id != self._current_policy_session_id
        )
        if should_reset:
            self._pending_actions.clear()
            self._last_llm_output = None
            reset = getattr(model, "reset", None)
            if callable(reset):
                reset()
        if policy_session_id:
            self._current_policy_session_id = policy_session_id

    def _apply_prompt_override(self, model: Any) -> None:
        prompt = str(
            self.settings.get("vln_prompt_template") or DEFAULT_PROMPT_TEMPLATE
        )
        conversation = getattr(model, "conversation", None)
        if (
            isinstance(conversation, list)
            and conversation
            and isinstance(conversation[0], dict)
        ):
            conversation[0]["value"] = prompt


def planner_result_from_output(
    output: Any,
    *,
    image_width: int,
    image_height: int,
    image_source: str | None = None,
    policy_session_id: str | None = None,
    action_sequence: list[int] | None = None,
    remaining_action_count: int | None = None,
    raw_output: str | None = None,
    turn_angle_deg: float = 15.0,
    forward_distance_cm: float = 25.0,
) -> VLNPlannerResult:
    raw_output = raw_output or _public_raw_output(output)
    output_latent = getattr(output, "output_latent", None)

    def result(
        mode: str,
        reason: str,
        *,
        pixel_goal: list[int] | None = None,
        heading_deg: float | None = None,
        action_code: int | None = None,
        stop: bool = False,
        requires_secondary_observation: bool = False,
    ) -> VLNPlannerResult:
        return VLNPlannerResult(
            mode=mode,
            pixel_goal=pixel_goal,
            heading_deg=heading_deg,
            action_code=action_code,
            action_sequence=action_sequence,
            remaining_action_count=(
                remaining_action_count
                if remaining_action_count is not None
                else max(len(action_sequence or []) - 1, 0)
            ),
            forward_distance_cm=(forward_distance_cm if action_code == 1 else None),
            stop=stop,
            reason=reason,
            raw_output=raw_output,
            image_source=image_source,
            image_width=image_width,
            image_height=image_height,
            output_latent=output_latent,
            requires_secondary_observation=requires_secondary_observation,
            policy_session_id=policy_session_id,
        )

    pixel = getattr(output, "output_pixel", None)
    if pixel is not None:
        parsed = _parse_pixel_goal(pixel)
        if parsed is None:
            raise VLNPlanningError(
                "vln_parse_failed",
                "InternVLA-N1 System 2 returned an invalid pixel goal",
            )
        row, col = parsed
        bounded = (
            min(max(row, 0), max(image_height - 1, 0)),
            min(max(col, 0), max(image_width - 1, 0)),
        )
        reason = "InternVLA-N1 System 2 returned output_pixel"
        if bounded != parsed:
            reason = "InternVLA-N1 System 2 output_pixel was clamped to image bounds"
        return result("pixel_goal", reason, pixel_goal=list(bounded))

    if action_sequence is None:
        action_sequence = action_codes_from_output(output)
    current_action = action_sequence[0] if action_sequence else None
    if current_action == 0:
        return result(
            "stop",
            "InternVLA-N1 System 2 returned STOP",
            action_code=0,
            stop=True,
        )
    if current_action == 5:
        return result(
            "look_down_required",
            "InternVLA-N1 System 2 requested a look-down secondary observation",
            action_code=5,
            requires_secondary_observation=True,
        )
    heading = action_to_heading(current_action, turn_angle_deg=turn_angle_deg)
    if heading is not None:
        return result(
            "heading",
            "InternVLA-N1 System 2 returned direction action",
            heading_deg=heading,
            action_code=current_action,
        )
    raise VLNPlanningError(
        "vln_no_valid_goal",
        "InternVLA-N1 System 2 did not return output_pixel or STOP",
    )


def action_to_heading(action: Any, *, turn_angle_deg: float = 15.0) -> float | None:
    current = _current_action_code(action)
    if current == 1:
        return 0.0
    if current == 2:
        return -abs(turn_angle_deg)
    if current == 3:
        return abs(turn_angle_deg)
    return None


def action_codes_from_output(output: Any) -> list[int]:
    action = getattr(output, "output_action", None)
    if action is None:
        return []
    array = np.asarray(action).reshape(-1)
    codes: list[int] = []
    for item in array:
        if str(item).strip().upper() == "STOP":
            codes.append(0)
            continue
        try:
            code = int(item)
        except (TypeError, ValueError):
            continue
        if code in {0, 1, 2, 3, 5}:
            codes.append(code)
    return codes


def _parse_pixel_goal(value: Any) -> tuple[int, int] | None:
    array = np.asarray(value).reshape(-1)
    if array.size < 2:
        return None
    return (int(array[0]), int(array[1]))


def _current_action_code(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    item = array[0]
    if str(item).strip().upper() == "STOP":
        return 0
    try:
        return int(item)
    except (TypeError, ValueError):
        return None


def _public_raw_output(output: Any) -> str | None:
    text = getattr(output, "llm_output", None) or getattr(output, "raw_output", None)
    if text:
        return str(text)
    pixel = getattr(output, "output_pixel", None)
    if pixel is not None:
        return str(np.asarray(pixel).reshape(-1).tolist())
    action = getattr(output, "output_action", None)
    if action is not None:
        return str(np.asarray(action).reshape(-1).tolist())
    return None
