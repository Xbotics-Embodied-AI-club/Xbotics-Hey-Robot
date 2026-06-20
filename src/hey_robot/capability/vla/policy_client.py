from __future__ import annotations

import contextlib
import io
from typing import Any, Protocol

import numpy as np

from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="capability.vla.policy")


class VLAPolicyClient(Protocol):
    """Interface for VLA policy inference backends."""

    def ping(self) -> bool: ...

    def reset(self) -> None: ...

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Fake client (integration testing)
# ---------------------------------------------------------------------------


class FakePolicyClient:
    """Returns a gentle sinusoidal motion chunk in GR00T format for testing.

    The chunk moves shoulder_lift and elbow_flex by a small amount so the
    simulated arm visibly moves without hitting joint limits.
    """

    def __init__(
        self,
        *,
        action_horizon: int = 16,
        amplitude: float = 0.02,
    ) -> None:
        self._horizon = action_horizon
        self._amplitude = amplitude
        self._t = 0

    def ping(self) -> bool:
        return True

    def reset(self) -> None:
        self._t = 0

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        _ = observation
        phase = self._t * 0.15
        # GR00T format: (B=1, T, D)
        single_arm = np.zeros((1, self._horizon, 5), dtype=np.float32)
        single_arm[0, :, 1] = self._amplitude * np.sin(phase)  # shoulder_lift
        single_arm[0, :, 2] = self._amplitude * 0.5 * np.cos(phase * 1.3)  # elbow_flex
        gripper = np.zeros((1, self._horizon, 1), dtype=np.float32)
        self._t += 1
        return {"single_arm": single_arm, "gripper": gripper}


# ---------------------------------------------------------------------------
# GR00T ZMQ client
# ---------------------------------------------------------------------------


class _MsgSerializer:
    """msgpack serialization matching the GR00T policy server format.

    Mirrors RoboCrew's ``_MsgSerializer`` in ``groot_client.py``.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        import msgpack

        return msgpack.packb(data, default=_MsgSerializer._encode)  # type: ignore[no-any-return]

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        import msgpack

        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)

    @staticmethod
    def _decode(obj: dict) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        return obj

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Cannot serialize type {type(obj)}")


# ---------------------------------------------------------------------------
# LeRobot in-process policy client (pi0, pi05, smolvla, act, etc.)
# ---------------------------------------------------------------------------


class LerobotPolicyClient:
    """In-process policy client that loads a LeRobot VLA policy directly.

    Bypasses the need for a separate gRPC/ZMQ policy server by loading
    the policy via ``get_policy_class`` + ``from_pretrained`` and running
    inference in the same process.

    Converts between GR00T-format observations/actions and the format
    expected by the LeRobot pre/post-processor pipeline.

    Supported policy types: act, pi0, pi05, smolvla, diffusion, vqbet, xvla, groot.
    """

    def __init__(
        self,
        *,
        policy_type: str,
        model_path: str,
        device: str = "cpu",
        action_horizon: int = 16,
        camera_key_map: dict[str, str] | None = None,
    ) -> None:
        import torch
        import torch.nn.functional as F  # noqa: N812
        from lerobot.configs.types import FeatureType
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._device = device
        self._action_horizon = action_horizon
        self._camera_map = dict(camera_key_map or {})
        self._torch = torch
        self._F = F

        logger.info(
            f"Loading LeRobot policy type={policy_type} from {model_path} on {device}"
        )

        policy_class = get_policy_class(policy_type)
        self._policy = policy_class.from_pretrained(model_path)
        self._policy.to(device)
        self._policy.eval()

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=model_path,
        )

        self._policy_type = policy_type
        self._model_path = model_path
        self._config = self._policy.config

        # Discover image and state keys from policy input features.
        self._image_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, ...]] = {}
        self._state_key: str = "observation.state"
        self._has_language: bool = False

        for key, feat in self._policy.config.input_features.items():
            if feat.type == FeatureType.VISUAL:
                self._image_keys.append(key)
                self._image_shapes[key] = feat.shape  # (C, H, W)
            elif feat.type == FeatureType.STATE:
                self._state_key = key

        # VLA policies (pi0, pi05, smolvla, xvla) expect a language prompt.
        self._has_language = policy_type in (
            "pi0",
            "pi05",
            "smolvla",
            "xvla",
            "groot",
            "wall_x",
        )

        logger.info(
            f"LeRobot policy loaded: state_key={self._state_key}, "
            f"image_keys={self._image_keys}, has_language={self._has_language}"
        )

    # ------------------------------------------------------------------
    # VLAPolicyClient protocol
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        return True

    def reset(self) -> None:
        pass

    def get_action(self, observation: dict[str, object]) -> dict[str, object]:
        """Convert GR00T observation → LeRobot inference → GR00T action."""
        import numpy as np

        lerobot_obs = self._groot_obs_to_lerobot_obs(observation)

        # Preprocessor: normalize, add batch dim, move to device
        processed = self._preprocessor(lerobot_obs)

        # Inference
        with self._torch.no_grad():
            chunk = self._policy.predict_action_chunk(processed)
        # chunk: (B, T, action_dim) or (T, action_dim)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)  # add batch dim
        chunk = chunk[:, : self._action_horizon, :]

        # Postprocessor: un-normalize per step, then re-stack
        _, n_steps, _ = chunk.shape
        processed_steps = []
        for i in range(n_steps):
            single = self._postprocessor(chunk[:, i, :])
            processed_steps.append(single)
        chunk = self._torch.stack(processed_steps, dim=1).squeeze(
            0
        )  # (n_steps, n_dims)

        chunk_np = chunk.detach().cpu().numpy().astype(np.float64)

        # Convert LeRobot action (T, D) → GR00T format
        return self._lerobot_action_to_groot(chunk_np)

    # ------------------------------------------------------------------
    # Format conversion helpers
    # ------------------------------------------------------------------

    def _groot_obs_to_lerobot_obs(
        self, observation: dict[str, object]
    ) -> dict[str, object]:
        """Convert GR00T-format observation to LeRobot preprocessor input.

        GR00T format::

            {"video": {"camera1": (1,1,H,W,3) uint8, ...},
             "state": {"single_arm": (1,1,5), "gripper": (1,1,1)},
             "language": {"annotation.human.task_description": "..."}}

        LeRobot preprocessor expects::

            {"observation.state": (state_dim,) float tensor,
             "observation.images.<cam>": (C, H_policy, W_policy) float [0,1] tensor,
             "task": "language instruction"}
        """
        import numpy as np

        obs: dict[str, object] = {}

        # ── State ──────────────────────────────────────────────────────
        state_dict = observation.get("state", {})
        if isinstance(state_dict, dict):
            arm = np.asarray(state_dict.get("single_arm", np.zeros(5)))
            gripper = np.asarray(state_dict.get("gripper", np.zeros(1)))
            joint_vec = np.concatenate([arm.ravel(), gripper.ravel()], axis=0).astype(
                np.float32
            )
        else:
            joint_vec = np.asarray(state_dict, dtype=np.float32).ravel()

        obs[self._state_key] = self._torch.from_numpy(joint_vec)

        # ── Images ─────────────────────────────────────────────────────
        video = observation.get("video", {})
        if isinstance(video, dict):
            for i, (cam_name, frame) in enumerate(video.items()):
                policy_key = self._resolve_image_key(cam_name, i)
                if policy_key is None:
                    continue

                img = np.asarray(frame, dtype=np.float32)
                # Squeeze leading batch/time dims: (1,1,H,W,3) → (H,W,3)
                while img.ndim > 3:
                    img = img.squeeze(0)

                # (H, W, 3) → (3, H, W) and normalize to [0, 1]
                img_t = self._torch.from_numpy(img).float() / 255.0
                if img_t.ndim == 3 and img_t.shape[-1] == 3:
                    img_t = img_t.permute(2, 0, 1)  # (H,W,3) → (3,H,W)

                # Resize to policy expected resolution
                target_shape = self._image_shapes.get(policy_key)
                if target_shape is not None:
                    _, target_h, target_w = target_shape
                    if img_t.shape[1] != target_h or img_t.shape[2] != target_w:
                        img_t = self._F.interpolate(
                            img_t.unsqueeze(0),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                img_t = img_t.contiguous()
                obs[policy_key] = img_t

        # ── Language ───────────────────────────────────────────────────
        if self._has_language:
            task_text = self._extract_task(observation)
            if task_text:
                obs["task"] = task_text

        return obs

    def _resolve_image_key(self, cam_name: str, index: int) -> str | None:
        """Map a GR00T camera name to a policy image key."""
        # Explicit mapping takes priority
        if cam_name in self._camera_map:
            return self._camera_map[cam_name]
        # Fallback: match by position order
        if index < len(self._image_keys):
            return self._image_keys[index]
        return None

    @staticmethod
    def _extract_task(observation: dict[str, object]) -> str:
        """Extract the task/language prompt from a GR00T observation.

        Handles nested formats:
        - ``{"language": "pick up"}``
        - ``{"language": {"task": "pick up"}}``
        - ``{"language": {"task": ["pick up"]}}``
        - ``{"language": {"task": [["pick up"]]}}`` (nested lists from batch/time dims)
        """
        for key in ("language", "task", "prompt", "annotation.human.task_description"):
            val = observation.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                for v in val.values():
                    result = LerobotPolicyClient._extract_scalar_str(v)
                    if result:
                        return result
            result = LerobotPolicyClient._extract_scalar_str(val)
            if result:
                return result
        return ""

    @staticmethod
    def _extract_scalar_str(val: object) -> str:
        """Recursively unwrap nested lists to find a string."""
        if isinstance(val, str):
            return val
        if isinstance(val, (list, tuple)) and len(val) > 0:  # type: ignore[arg-type]
            return LerobotPolicyClient._extract_scalar_str(val[0])  # type: ignore[arg-type]
        return ""

    @staticmethod
    def _lerobot_action_to_groot(
        action: np.ndarray,  # (T, action_dim)
    ) -> dict[str, object]:
        """Convert LeRobot action tensor to GR00T format.

        LeRobot: (T, 6) where columns are [arm_joint_0..4, gripper]
        GR00T:   {"single_arm": (1, T, 5), "gripper": (1, T, 1)}
        """
        import numpy as np

        n_steps = action.shape[0]
        single_arm = action[:, :5].reshape(1, n_steps, 5).astype(np.float32)
        gripper = action[:, 5:6].reshape(1, n_steps, 1).astype(np.float32)
        return {"single_arm": single_arm, "gripper": gripper}


class GrootZmqPolicyClient:
    """Lightweight ZMQ client for the GR00T inference server.

    Uses msgpack serialization and endpoint-based RPC, matching RoboCrew's
    ``PolicyClient`` in ``groot_client.py``.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._socket: Any = None
        self._context: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def reset(self) -> None:
        self._call_endpoint("reset", {"options": None})

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send observation and return the action_chunk dict.

        The server returns ``(action_chunk, info)``; we return only the
        action_chunk to keep the interface consistent with FakePolicyClient.
        """
        response = self._call_endpoint("get_action", {"observation": observation})
        if isinstance(response, (list, tuple)):
            return response[0]  # type: ignore[no-any-return]
        return response  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:

        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}

        self._ensure_socket()
        self._socket.send(_MsgSerializer.to_bytes(request))

        if self._socket.poll(self._timeout_ms):
            raw = self._socket.recv()
        else:
            raise TimeoutError(
                f"GR00T policy did not respond within {self._timeout_ms}ms"
            )

        if raw == b"ERROR":
            raise RuntimeError("GR00T server returned a generic ERROR.")

        response = _MsgSerializer.from_bytes(raw)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")

        return response

    def _ensure_socket(self) -> None:
        if self._socket is not None:
            return
        self._init_socket()

    def _init_socket(self) -> None:
        try:
            import zmq
        except ImportError as err:
            raise ImportError(
                "pyzmq is required for GR00T ZMQ policy client. "
                "Install it with: pip install pyzmq"
            ) from err
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        if self._timeout_ms:
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
