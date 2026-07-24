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

_ACTION_HEADING: dict[int, float] = {
    1: 0.0,
    2: -90.0,
    3: 90.0,
}
_MODEL_LOAD_LOCK = threading.Lock()


class InternVLAN1Runtime:
    """Own only the third-party InternNav model lifecycle and inference."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self._model: Any | None = None
        self._current_policy_session_id: str | None = None
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
            output = model.s2_step(
                planner_input.rgb,
                planner_input.depth,
                planner_input.pose,
                planner_input.instruction,
                planner_input.intrinsic,
                planner_input.look_down,
            )
        return planner_result_from_output(
            output,
            image_width=int(planner_input.rgb.shape[1]),
            image_height=int(planner_input.rgb.shape[0]),
            image_source=planner_input.image_source,
            policy_session_id=policy_session_id,
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._current_policy_session_id = None

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
) -> VLNPlannerResult:
    raw_output = _public_raw_output(output)
    output_latent = getattr(output, "output_latent", None)

    def result(
        mode: str,
        reason: str,
        *,
        pixel_goal: list[int] | None = None,
        heading_deg: float | None = None,
        stop: bool = False,
        requires_secondary_observation: bool = False,
    ) -> VLNPlannerResult:
        return VLNPlannerResult(
            mode=mode,
            pixel_goal=pixel_goal,
            heading_deg=heading_deg,
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

    action = getattr(output, "output_action", None)
    if _is_stop_action(action):
        return result(
            "stop",
            "InternVLA-N1 System 2 returned STOP",
            stop=True,
        )
    if _is_look_down_action(action):
        return result(
            "look_down_required",
            "InternVLA-N1 System 2 requested a look-down secondary observation",
            requires_secondary_observation=True,
        )
    heading = action_to_heading(action)
    if heading is not None:
        return result(
            "heading",
            "InternVLA-N1 System 2 returned direction action",
            heading_deg=heading,
        )
    raise VLNPlanningError(
        "vln_no_valid_goal",
        "InternVLA-N1 System 2 did not return output_pixel or STOP",
    )


def action_to_heading(action: Any) -> float | None:
    current = _current_action_code(action)
    return _ACTION_HEADING.get(current) if current is not None else None


def _parse_pixel_goal(value: Any) -> tuple[int, int] | None:
    array = np.asarray(value).reshape(-1)
    if array.size < 2:
        return None
    return (int(array[0]), int(array[1]))


def _is_stop_action(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() == "STOP"
    return _current_action_code(value) == 0


def _is_look_down_action(value: Any) -> bool:
    return _current_action_code(value) == 5


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
