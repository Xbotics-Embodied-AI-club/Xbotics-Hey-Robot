from __future__ import annotations

import re
import threading
import time
from typing import Any, Protocol

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.backends.vln.control import build_base_action_chunk
from hey_robot.foundation.backends.vln.input import planner_input_from_payload
from hey_robot.foundation.backends.vln.models import (
    VLNPlannerInput,
    VLNPlannerResult,
    VLNPlanningError,
)


class VLNRuntime(Protocol):
    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def plan(
        self,
        planner_input: VLNPlannerInput,
        *,
        policy_session_id: str | None,
        reset_policy: bool,
    ) -> VLNPlannerResult: ...

    def close(self) -> None: ...


class VLNPlannerExecutor:
    """Model-service adapter for the InternNav VLN planning contract."""

    def __init__(
        self,
        service_id: str,
        spec: ModelServiceSpec,
        *,
        runtime: VLNRuntime | None = None,
    ) -> None:
        self.service_id = service_id
        self.spec = spec
        self.settings = dict(spec.settings)
        self.backend = str(self.settings.get("backend") or "internvla_n1_dualvln")
        self.control_mode = str(
            self.settings.get("control_mode") or "base_action_chunk"
        )
        self.camera = str(self.settings.get("camera") or "front")
        self._cancelled = threading.Event()
        self._runtime = runtime
        self._load_error: str | None = None

    def load(self) -> None:
        if self._mock_mode():
            return
        runtime = self._get_runtime()
        try:
            runtime.load()
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"failed to load VLN backend {self.backend}: {self._load_error}"
            ) from exc
        self._load_error = None

    def health(self) -> dict[str, Any]:
        missing = self._missing_config()
        mock_mode = self._mock_mode()
        loaded = mock_mode or bool(self._runtime and self._runtime.loaded)
        error = self._load_error
        if missing:
            error = f"missing VLN configuration: {', '.join(missing)}"
        return {
            "name": self.service_id,
            "online": not bool(missing or error),
            "loaded": loaded,
            "robot_id": self.spec.robot_id,
            "error": error,
            "metrics": {
                "type": self.spec.type,
                "backend": self.backend,
                "runtime": self.backend,
                "model_path": self.settings.get("model_path"),
                "mock_mode": mock_mode,
                "control_mode": self.control_mode,
                "camera": self.camera,
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._cancelled.clear()
        started_at = time.time()
        if self.control_mode != "base_action_chunk":
            return self._failure(
                "unsupported_control_mode",
                f"VLN control_mode={self.control_mode!r} is not supported",
            )
        missing = self._missing_config()
        if missing:
            return self._failure(
                "invalid_configuration",
                f"missing VLN configuration: {', '.join(missing)}",
            )
        try:
            result = (
                self._mock_plan(payload)
                if self._mock_mode()
                else self._real_plan(payload)
            )
        except VLNPlanningError as exc:
            return self._failure(exc.failure_mode, str(exc))
        except (ImportError, ModuleNotFoundError) as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return self._failure(
                "missing_dependency",
                f"VLN dependencies are unavailable: {exc}",
            )
        except Exception as exc:
            return self._failure(
                "vln_inference_failed",
                f"VLN inference failed: {type(exc).__name__}: {exc}",
            )
        metrics: dict[str, Any] = {
            "duration_sec": round(time.time() - started_at, 3),
            "vln": result.to_metrics(
                backend=self.backend,
                camera=self.camera,
                control_mode=self.control_mode,
            ),
        }
        metrics["vln"]["base_control"] = {
            "linear_speed": float(self.settings["base_linear_speed"]),
            "angular_speed": float(self.settings["base_angular_speed"]),
            "forward_distance_cm": float(self.settings["discrete_forward_cm"]),
            "turn_angle_deg": float(self.settings["discrete_turn_deg"]),
            "max_chunk_steps": int(self.settings["max_action_chunk_steps"]),
        }
        metrics["vln"]["control_chunk"] = build_base_action_chunk(result, self.settings)
        if self._cancelled.is_set():
            return {
                "success": False,
                "status": "cancelled",
                "failure_mode": "cancelled",
                "summary": "VLN planning cancelled",
                "metrics": metrics,
            }
        return {
            "success": True,
            "status": "completed",
            "summary": f"VLN planner produced {result.mode}",
            "metrics": metrics,
        }

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()

    def _real_plan(self, payload: dict[str, Any]) -> VLNPlannerResult:
        self.load()
        runtime = self._get_runtime()
        planner_input = planner_input_from_payload(
            payload,
            camera=self.camera,
            media_root=str(self.settings.get("media_root") or "") or None,
            hfov=float(self.settings.get("hfov", 90.0)),
        )
        return runtime.plan(
            planner_input,
            policy_session_id=_policy_session_id_from_payload(payload),
            reset_policy=_reset_policy_from_payload(payload),
        )

    def _get_runtime(self) -> VLNRuntime:
        if self._runtime is None:
            if self.backend != "internvla_n1_dualvln":
                raise RuntimeError(f"unsupported VLN backend: {self.backend}")
            from hey_robot.foundation.backends.vln.internvla_n1 import (
                InternVLAN1Runtime,
            )

            self._runtime = InternVLAN1Runtime(self.settings)
        return self._runtime

    def _mock_mode(self) -> bool:
        if "mock_mode" in self.settings:
            return bool(self.settings.get("mock_mode"))
        return not bool(self.settings.get("model_path"))

    def _missing_config(self) -> list[str]:
        if self._mock_mode():
            return []
        required = ("model_path", "internnav_repo", "media_root")
        return [key for key in required if not self.settings.get(key)]

    def _mock_plan(self, payload: dict[str, Any]) -> VLNPlannerResult:
        arguments = dict(payload.get("arguments", {}) or {})
        text = " ".join(
            str(value)
            for value in (
                payload.get("objective"),
                arguments.get("target"),
                arguments.get("instruction"),
            )
            if value
        ).lower()
        if any(token in text for token in ("stop", "停止", "停下", "done", "完成")):
            return VLNPlannerResult(
                mode="stop",
                stop=True,
                confidence=1.0,
                reason="mock planner matched stop-like instruction",
                raw_output="STOP",
            )
        heading = _number_arg(arguments, "heading_deg")
        if heading is None:
            heading = _number_arg(self.settings, "mock_heading_deg")
        if heading is not None:
            return VLNPlannerResult(
                mode="heading",
                heading_deg=heading,
                confidence=0.5,
                reason="mock planner returned configured heading",
                raw_output=f"HEADING {heading:.1f}",
            )
        width = int(self.settings.get("image_width", 640))
        height = int(self.settings.get("image_height", 480))
        row = int(self.settings.get("mock_pixel_y", height // 2))
        col = int(self.settings.get("mock_pixel_x", width // 2))
        return VLNPlannerResult(
            mode="pixel_goal",
            pixel_goal=[row, col],
            action_code=1,
            action_sequence=[1],
            forward_distance_cm=float(self.settings["discrete_forward_cm"]),
            confidence=0.5,
            reason="mock planner returned center pixel goal",
            raw_output=f"({row}, {col})",
            image_width=width,
            image_height=height,
            policy_session_id=_policy_session_id_from_payload(payload),
        )

    def _failure(self, failure_mode: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "status": "failed",
            "failure_mode": failure_mode,
            "summary": message,
            "error": message,
            "metrics": {
                "vln": {
                    "backend": self.backend,
                    "control_mode": self.control_mode,
                    "camera": self.camera,
                }
            },
        }


def build_vln_executor(service_id: str, spec: ModelServiceSpec) -> VLNPlannerExecutor:
    backend = str(spec.settings.get("backend") or "internvla_n1_dualvln")
    if backend != "internvla_n1_dualvln":
        raise ValueError(f"unsupported VLN backend: {backend}")
    return VLNPlannerExecutor(service_id, spec)


def _number_arg(source: dict[str, Any], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _policy_session_id_from_payload(payload: dict[str, Any]) -> str | None:
    arguments = dict(payload.get("arguments", {}) or {})
    metadata = dict(payload.get("metadata", {}) or {})
    value = (
        arguments.get("policy_session_id")
        or metadata.get("policy_session_id")
        or payload.get("policy_session_id")
        or payload.get("skill_id")
    )
    return str(value) if value else None


def _reset_policy_from_payload(payload: dict[str, Any]) -> bool:
    arguments = dict(payload.get("arguments", {}) or {})
    return bool(arguments.get("reset_policy") or payload.get("reset_policy"))
