from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from threading import Event, Lock
from typing import Any

DEFAULT_POLICY = "lerobot/smolvla_robocasa"
DEFAULT_TASK = "CloseFridge"
DEFAULT_SPLIT = "target"
DEFAULT_REGISTRIES = ("lightwheel",)
ALLOWED_TASKS = frozenset(
    {
        "CloseFridge",
        "OpenDrawer",
        "OpenCabinet",
        "TurnOnMicrowave",
        "TurnOffStove",
    }
)
CAMERA_RENAME_MAP = {
    "observation.images.robot0_agentview_left": "observation.images.camera1",
    "observation.images.robot0_agentview_right": "observation.images.camera2",
    "observation.images.robot0_eye_in_hand": "observation.images.camera3",
}


def evaluation_rename_map(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return an optional environment-to-policy feature rename map.

    The eval wrapper automatically restores a checkpoint's saved processor
    mapping. This environment value is only an explicit override for custom
    checkpoints or diagnostics.
    """
    raw = (environ or os.environ).get("ROBOCASA_RENAME_MAP", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROBOCASA_RENAME_MAP is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(target, str)
        for key, target in value.items()
    ):
        raise ValueError("ROBOCASA_RENAME_MAP must be a JSON object of string pairs")
    return value


class RolloutError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class RolloutRequest:
    skill_id: str
    objective: str
    task: str
    seed: int = 1000
    n_episodes: int = 1
    policy_path: str = DEFAULT_POLICY
    obj_registries: tuple[str, ...] = DEFAULT_REGISTRIES
    record_video: bool = True
    timeout_sec: float = 1800.0


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    status: str
    summary: str
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


class RoboCasaRolloutRunner:
    """A safe subprocess boundary around the official ``lerobot-eval`` CLI."""

    def __init__(
        self,
        *,
        output_root: Path | str | None = None,
        eval_binary: str | None = None,
        environ: dict[str, str] | None = None,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.output_root = Path(
            output_root or os.environ.get("ROBOCASA_OUTPUT_ROOT", "/outputs")
        ).resolve()
        self.environ = dict(environ or os.environ)
        self.eval_binary = eval_binary or self.environ.get(
            "ROBOCASA_EVAL_BINARY", "lerobot-eval"
        )
        self.default_policy = self.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
        self.rename_map = evaluation_rename_map(self.environ)
        self._popen_factory = popen_factory
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancelled = Event()
        self._current_skill_id: str | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._process is not None

    @property
    def current_skill_id(self) -> str | None:
        with self._lock:
            return self._current_skill_id

    def health(self) -> dict[str, Any]:
        imports_ok, import_error = self._imports_available()
        assets_ok = self._assets_available()
        policy_cached = self._policy_cached(self.default_policy)
        loaded = imports_ok and assets_ok and policy_cached
        return {
            "online": True,
            # Do not report loaded solely because a gRPC port is listening. A
            # cached checkpoint plus the lightwheel asset profile is required.
            "loaded": loaded,
            "error": import_error
            or (
                None
                if loaded
                else "RoboCasa worker is waiting for checkpoint cache or lightwheel assets"
            ),
            "metrics": {
                "benchmark": "robocasa365",
                "asset_profile": "lightwheel",
                "policy_path": self.default_policy,
                "policy_revision": self._policy_revision(self.default_policy),
                "policy_cached": policy_cached,
                "imports_available": imports_ok,
                "assets_available": assets_ok,
                "busy": self.busy,
                "current_skill_id": self.current_skill_id,
                "versions": _versions(),
            },
        }

    def cancel(self, skill_id: str | None = None) -> bool:
        with self._lock:
            if self._process is None:
                return False
            if skill_id and self._current_skill_id != skill_id:
                return False
            self._cancelled.set()
            process = self._process
        _terminate_process(process)
        return True

    def run(self, request: RolloutRequest) -> RolloutResult:
        self._validate_request(request)
        output_dir = self._output_dir(request.skill_id)
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "request.json", _request_log(request))
        _write_json(output_dir / "versions.json", _versions())

        command = self._command(request, output_dir)
        started_at = time.monotonic()
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                with self._lock:
                    if self._process is not None:
                        raise RolloutError(
                            "model_service_busy", "RoboCasa worker is busy"
                        )
                    self._cancelled.clear()
                    self._current_skill_id = request.skill_id
                    self._process = self._popen_factory(
                        command,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        env=self.environ,
                        start_new_session=True,
                    )
                    process = self._process
                return_code = self._wait_for_process(process, request.timeout_sec)
        except RolloutError as exc:
            return RolloutResult(
                success=False,
                status=(
                    "cancelled" if exc.failure_mode == "rollout_cancelled" else "failed"
                ),
                summary=f"RoboCasa {request.task} did not run: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
                metrics=self._base_metrics(
                    request, output_dir, time.monotonic() - started_at
                ),
            )
        except OSError as exc:
            return RolloutResult(
                success=False,
                status="failed",
                summary=f"RoboCasa evaluator could not start: {exc}",
                failure_mode="environment_reset_failed",
                error=str(exc),
                metrics=self._base_metrics(
                    request, output_dir, time.monotonic() - started_at
                ),
            )
        finally:
            with self._lock:
                self._process = None
                self._current_skill_id = None

        duration_sec = time.monotonic() - started_at
        metrics = self._base_metrics(request, output_dir, duration_sec)
        if self._cancelled.is_set():
            return RolloutResult(
                success=False,
                status="cancelled",
                summary=f"RoboCasa {request.task} rollout was cancelled",
                failure_mode="rollout_cancelled",
                metrics=metrics,
            )
        if return_code != 0:
            failure_mode = _failure_mode_from_logs(stderr_path)
            return RolloutResult(
                success=False,
                status="failed",
                summary=f"RoboCasa evaluator exited with code {return_code}",
                failure_mode=failure_mode,
                error=_tail(stderr_path),
                metrics=metrics,
            )

        try:
            eval_metrics = parse_eval_info(
                output_dir / "eval_info.json", request.n_episodes
            )
        except RolloutError as exc:
            return RolloutResult(
                success=False,
                status="failed",
                summary=f"RoboCasa evaluator completed but result parsing failed: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
                metrics=metrics,
            )

        metrics.update(eval_metrics)
        metrics["video_paths"] = [
            str(path.relative_to(self.output_root))
            for path in sorted((output_dir / "videos").rglob("*.mp4"))
            if path.is_file()
        ]
        success = int(eval_metrics["success_count"]) == request.n_episodes
        return RolloutResult(
            success=success,
            status="completed" if success else "failed",
            summary=(
                f"RoboCasa {request.task}: {eval_metrics['success_count']}/"
                f"{request.n_episodes} successful"
            ),
            failure_mode=None if success else "task_unsuccessful",
            metrics=metrics,
        )

    def _validate_request(self, request: RolloutRequest) -> None:
        if request.task not in ALLOWED_TASKS:
            raise RolloutError("invalid_task", f"task {request.task!r} is not allowed")
        if request.n_episodes != 1:
            raise RolloutError(
                "invalid_task", "agent-facing robocasa_rollout requires n_episodes=1"
            )
        if not request.skill_id or not _safe_component(request.skill_id):
            raise RolloutError(
                "invalid_task", "skill_id contains unsupported characters"
            )
        if not request.policy_path.strip():
            raise RolloutError("checkpoint_unavailable", "policy_path is required")
        if tuple(request.obj_registries) != DEFAULT_REGISTRIES:
            raise RolloutError(
                "asset_unavailable",
                "only the lightwheel asset profile is enabled in the first release",
            )
        if not request.record_video:
            raise RolloutError(
                "invalid_task",
                "record_video=false is unsupported by the pinned evaluator",
            )

    def _output_dir(self, skill_id: str) -> Path:
        candidate = (self.output_root / _safe_component(skill_id)).resolve()
        if candidate.parent != self.output_root:
            raise RolloutError("invalid_task", "skill_id resolves outside output root")
        return candidate

    def _command(self, request: RolloutRequest, output_dir: Path) -> list[str]:
        command = [
            self.eval_binary,
            f"--policy.path={request.policy_path}",
            "--env.type=robocasa",
            f"--env.task={request.task}",
            f"--env.split={DEFAULT_SPLIT}",
            f"--env.obj_registries=[{','.join(request.obj_registries)}]",
            "--eval.batch_size=1",
            "--eval.n_episodes=1",
            "--eval.use_async_envs=false",
            f"--seed={request.seed}",
            "--policy.device=cuda",
            f"--output_dir={output_dir}",
        ]
        if self.rename_map:
            command.append(
                f"--rename_map={json.dumps(self.rename_map, separators=(',', ':'))}"
            )
        # The pinned RoboCasa evaluation command records its video artifacts in
        # the output directory. Do not add an unverified "disable video" flag:
        # unknown LeRobot CLI options abort an otherwise valid rollout.
        return command

    def _wait_for_process(
        self, process: subprocess.Popen[str], timeout_sec: float
    ) -> int:
        deadline = time.monotonic() + max(1.0, timeout_sec)
        while True:
            code = process.poll()
            if code is not None:
                return code
            if self._cancelled.is_set():
                _terminate_process(process)
                raise RolloutError(
                    "rollout_cancelled", "rollout cancellation requested"
                )
            if time.monotonic() >= deadline:
                _terminate_process(process)
                raise RolloutError("rollout_timeout", "rollout timed out")
            time.sleep(0.1)

    def _base_metrics(
        self, request: RolloutRequest, output_dir: Path, duration_sec: float
    ) -> dict[str, Any]:
        return {
            "benchmark": "robocasa365",
            "task": request.task,
            "split": DEFAULT_SPLIT,
            "policy_path": request.policy_path,
            "n_episodes": request.n_episodes,
            "seeds": [request.seed],
            "duration_sec": round(duration_sec, 3),
            "output_dir": str(output_dir.relative_to(self.output_root)),
        }

    def _imports_available(self) -> tuple[bool, str | None]:
        try:
            __import__("robocasa")
            __import__("robosuite")
            __import__("mujoco")
            __import__("lerobot")
        except Exception as exc:
            return False, f"dependency unavailable: {type(exc).__name__}: {exc}"
        return True, None

    def _assets_available(self) -> bool:
        return _assets_available(self.environ)

    def _policy_cached(self, policy_path: str) -> bool:
        cache_root = Path(self.environ.get("HF_HOME", "/cache/huggingface")) / "hub"
        repo_name = policy_path.replace("/", "--")
        snapshots = cache_root.glob(f"models--{repo_name}/snapshots/*")
        return any(
            (snapshot / "model.safetensors").is_file()
            or (snapshot / "model.safetensors.index.json").is_file()
            or any(snapshot.glob("model-*.safetensors"))
            for snapshot in snapshots
        )

    def _policy_revision(self, policy_path: str) -> str | None:
        cache_root = Path(self.environ.get("HF_HOME", "/cache/huggingface")) / "hub"
        repo_name = policy_path.replace("/", "--")
        snapshots = sorted(cache_root.glob(f"models--{repo_name}/snapshots/*"))
        return snapshots[-1].name if snapshots else None


def request_from_payload(payload: dict[str, Any]) -> RolloutRequest:
    arguments = dict(payload.get("arguments", {}) or {})
    registries = arguments.get("obj_registries", DEFAULT_REGISTRIES)
    if not isinstance(registries, list | tuple):
        registries = DEFAULT_REGISTRIES
    return RolloutRequest(
        skill_id=str(payload.get("skill_id") or ""),
        objective=str(payload.get("objective") or ""),
        task=str(
            arguments.get("task")
            or os.environ.get("ROBOCASA_DEFAULT_TASK", DEFAULT_TASK)
        ),
        seed=int(arguments.get("seed", 1000)),
        n_episodes=int(arguments.get("n_episodes", 1)),
        policy_path=str(
            arguments.get("policy_path")
            or os.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
        ),
        obj_registries=tuple(str(item) for item in registries),
        record_video=bool(arguments.get("record_video", True)),
        timeout_sec=float(payload.get("timeout_sec", 1800.0)),
    )


def parse_eval_info(path: Path, n_episodes: int) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RolloutError(
            "result_parse_failed", "eval_info.json was not written"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RolloutError(
            "result_parse_failed", f"invalid eval_info.json: {exc}"
        ) from exc

    successes = _find_successes(data)
    if successes:
        success_count = sum(successes)
        return {
            "success_count": success_count,
            "success_rate": success_count / len(successes),
            "episode_successes": successes,
        }
    pc_success = _find_number(data, "pc_success")
    if pc_success is not None:
        success_rate = pc_success / 100 if pc_success > 1 else pc_success
        success_count = round(success_rate * n_episodes)
        return {
            "success_count": success_count,
            "success_rate": success_rate,
            "episode_successes": [],
        }
    raise RolloutError("result_parse_failed", "eval_info.json has no success result")


def _find_successes(value: Any) -> list[int]:
    if isinstance(value, dict):
        for key in ("success", "successes"):
            direct = value.get(key)
            if isinstance(direct, list) and all(
                isinstance(item, bool | int | float) for item in direct
            ):
                return [int(bool(item)) for item in direct]
        for key in ("per_episode", "episodes", "episode_results"):
            items = value.get(key)
            if isinstance(items, list):
                extracted = [
                    int(bool(item.get("success")))
                    for item in items
                    if isinstance(item, dict) and "success" in item
                ]
                if extracted:
                    return extracted
        for nested in value.values():
            found = _find_successes(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_successes(nested)
            if found:
                return found
    return []


def _find_number(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, int | float):
            return float(candidate)
        for nested in value.values():
            found = _find_number(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_number(nested, key)
            if found is not None:
                return found
    return None


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")


def _request_log(request: RolloutRequest) -> dict[str, Any]:
    # The request is built exclusively from RPC fields. HF_TOKEN is intentionally
    # never read or copied into this artifact.
    return {
        "skill_id": request.skill_id,
        "objective": request.objective,
        "task": request.task,
        "split": DEFAULT_SPLIT,
        "seed": request.seed,
        "n_episodes": request.n_episodes,
        "policy_path": request.policy_path,
        "obj_registries": list(request.obj_registries),
        "record_video": request.record_video,
        "timeout_sec": request.timeout_sec,
    }


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "lerobot": _distribution_version("lerobot"),
        "robocasa": _distribution_version("robocasa"),
        "robosuite": _distribution_version("robosuite"),
        "mujoco": _distribution_version("mujoco"),
    }
    version_file = Path(
        os.environ.get(
            "ROBOCASA_BRIDGE_VERSION_FILE", "/opt/bridge/bridge_versions.json"
        )
    )
    try:
        bridge_versions = json.loads(version_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        bridge_versions = {}
    return {
        **versions,
        **{
            key: value
            for key, value in bridge_versions.items()
            if isinstance(key, str) and isinstance(value, str)
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError):
            process.kill()


def _tail(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except FileNotFoundError:
        return ""


def _failure_mode_from_logs(stderr_path: Path) -> str:
    text = _tail(stderr_path).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "cuda_out_of_memory"
    if "asset" in text or "texture" in text or "mesh" in text:
        return "asset_unavailable"
    if "checkpoint" in text or "huggingface" in text or "snapshot" in text:
        return "checkpoint_unavailable"
    if "policy" in text and ("load" in text or "deserialize" in text):
        return "policy_load_failed"
    return "environment_reset_failed"


def _assets_available(environ: dict[str, str] | None = None) -> bool:
    environ = environ or os.environ
    explicit_root = Path(
        environ.get("ROBOCASA_MODEL_ASSET_ROOT", "/opt/robocasa/robocasa/models/assets")
    )
    roots = [explicit_root]
    with suppress(Exception):
        import robocasa

        roots.append(Path(robocasa.__file__).resolve().parent / "models" / "assets")

    marker_override = environ.get("ROBOCASA_ASSET_READY_FILE")
    for root in dict.fromkeys(roots):
        required = (
            root / "textures",
            root / "generative_textures",
            root / "fixtures",
            root / "objects" / "lightwheel",
        )
        if not all(path.is_dir() for path in required):
            continue
        marker = (
            Path(marker_override)
            if marker_override
            else root / ".robocasa-assets-ready"
        )
        if marker.is_file() or root != explicit_root:
            return True
    return False
