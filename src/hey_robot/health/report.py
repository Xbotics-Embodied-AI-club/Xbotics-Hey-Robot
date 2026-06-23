from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.skills.registry import registry_from_config


@dataclass(frozen=True)
class HealthReport:
    component: str
    status: str
    severity: str
    evidence: str
    impacted_skills: tuple[str, ...] = ()
    fix_hint: str | None = None
    source: str = "health_report"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["impacted_skills"] = list(self.impacted_skills)
        return data


class HealthReportService:
    """Builds user-facing health findings from existing read-only state."""

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        episode_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        live: bool = False,
    ) -> None:
        self.config = config
        self.task_runs = _task_run_store(episode_dir or config.resources.episodes_root)
        self.config_path = Path(config_path) if config_path is not None else None
        self.live = live

    def reports(
        self, *, robot_id: str | None = None, full: bool = False
    ) -> list[HealthReport]:
        findings: list[HealthReport] = []
        findings.extend(self._configuration_reports(robot_id=robot_id))
        findings.extend(self._skill_readiness_reports(robot_id=robot_id))
        if full:
            findings.extend(self._platform_reports(robot_id=robot_id))
            findings.extend(self._robot_component_reports(robot_id=robot_id))
            findings.extend(self._diagnostic_script_reports(robot_id=robot_id))
            findings.extend(self._audio_reports())
            findings.extend(self._recent_task_reports(robot_id=robot_id))
        if not findings:
            component = f"robot.{robot_id}" if robot_id else "deployment"
            findings.append(
                HealthReport(
                    component=component,
                    status="ok",
                    severity="info",
                    evidence="No configuration or recent task health issues found.",
                    fix_hint="Run the hardware-specific diagnose scripts for live device checks before operating a real robot.",
                    source="health_report.summary",
                )
            )
        return findings

    def payload(
        self, *, robot_id: str | None = None, full: bool = False
    ) -> dict[str, Any]:
        reports = self.reports(robot_id=robot_id, full=full)
        return {
            "robot_id": robot_id,
            "status": _overall_status(reports),
            "reports": [report.to_dict() for report in reports],
        }

    def _configuration_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        reports: list[HealthReport] = []
        for issue in validate_deployment(self.config):
            component = _component_from_issue(issue.message, robot_id=robot_id)
            if component is None:
                continue
            reports.append(
                HealthReport(
                    component=component,
                    status="failed" if issue.level == "error" else "degraded",
                    severity=issue.level,
                    evidence=issue.message,
                    impacted_skills=_skills_from_text(issue.message),
                    fix_hint=_fix_hint(issue.message),
                    source="config.validation",
                )
            )
        return reports

    def _skill_readiness_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        registry = registry_from_config(self.config)
        reports: list[HealthReport] = []
        for name in registry.names(enabled_only=True):
            try:
                spec = registry.get(name).spec
            except KeyError:
                continue
            resources = tuple(spec.required_resources)
            if not resources:
                continue
            if robot_id and not _skill_matches_robot(
                self.config, robot_id, spec.supported_robots
            ):
                continue
            reports.append(
                HealthReport(
                    component=f"skill.{name}",
                    status="ready_check_required",
                    severity="info",
                    evidence=(
                        f"Skill {name} requires resources: {', '.join(resources)}."
                    ),
                    impacted_skills=(name,),
                    fix_hint=_resource_fix_hint(resources),
                    source="skill.catalog",
                    metadata={
                        "resources": list(resources),
                        "driver_primitives": list(spec.driver_primitives),
                        "safety_level": spec.safety_level,
                    },
                )
            )
        return reports

    def _recent_task_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        reports: list[HealthReport] = []
        for task in self.task_runs.list_recent(50):
            if robot_id and task.robot_id != robot_id:
                continue
            if task.status not in {"failed", "recovering"} and not task.failure_reason:
                continue
            reports.append(
                HealthReport(
                    component=f"task.{task.task_id}",
                    status="failed" if task.status == "failed" else "degraded",
                    severity="error" if task.status == "failed" else "warning",
                    evidence=task.failure_reason
                    or (task.recovery or {}).get("summary")
                    or f"Task {task.root_task} is {task.status}.",
                    impacted_skills=tuple(
                        skill_id for skill_id in task.skill_ids if skill_id
                    ),
                    fix_hint=_task_fix_hint(task.failure_reason),
                    source="task_run",
                    metadata={
                        "episode_id": task.episode_id,
                        "root_task": task.root_task,
                        "status": task.status,
                        "recovery": dict(task.recovery or {}),
                    },
                )
            )
        return reports

    def _platform_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        del robot_id
        script = _repo_root() / "scripts" / "ops" / "check_platform.py"
        build_report = _load_function(script, "build_report")
        if build_report is None:
            return [
                HealthReport(
                    component="diagnostics.check_platform",
                    status="failed",
                    severity="error",
                    evidence=f"Cannot load platform check script: {script}",
                    fix_hint="Restore scripts/ops/check_platform.py.",
                    source="diagnostic.script",
                )
            ]
        report = build_report(
            config_path=str(self.config_path) if self.config_path else None
        )
        findings: list[HealthReport] = []
        for item in report.get("checks", []) or []:
            if not isinstance(item, dict):
                continue
            ok = bool(item.get("ok"))
            findings.append(
                HealthReport(
                    component=f"platform.{item.get('name')}",
                    status="ok" if ok else "failed",
                    severity="info" if ok else "warning",
                    evidence=str(item.get("detail") or ""),
                    fix_hint=None
                    if ok
                    else _platform_fix_hint(str(item.get("name") or "")),
                    source="check_platform",
                    metadata={"ready": bool(report.get("ready"))},
                )
            )
        config_report = report.get("config")
        if isinstance(config_report, dict):
            for item in config_report.get("checks", []) or []:
                if not isinstance(item, dict):
                    continue
                ok = bool(item.get("ok"))
                findings.append(
                    HealthReport(
                        component=f"config.{item.get('name')}",
                        status="ok" if ok else "failed",
                        severity="info" if ok else "error",
                        evidence=str(item.get("detail") or ""),
                        fix_hint=None
                        if ok
                        else _fix_hint(str(item.get("detail") or "")),
                        source="check_platform.config",
                    )
                )
        return findings

    def _robot_component_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        reports: list[HealthReport] = []
        for rid, robot in self.config.robots.items():
            if robot_id and rid != robot_id:
                continue
            components = robot.settings.get("components")
            if not isinstance(components, dict):
                reports.append(
                    HealthReport(
                        component=f"robot.{rid}.components",
                        status="degraded",
                        severity="warning",
                        evidence="Robot has no structured components config.",
                        impacted_skills=_skills_for_resources(
                            ("camera", "base", "arm")
                        ),
                        fix_hint="Add components.camera/base/arm configuration for product-grade diagnostics.",
                        source="health.config",
                    )
                )
                continue
            reports.extend(_component_reports_for_robot(rid, components))
        return reports

    def _diagnostic_script_reports(self, *, robot_id: str | None) -> list[HealthReport]:
        robot_types = {
            robot.type
            for rid, robot in self.config.robots.items()
            if not robot_id or rid == robot_id
        }
        scripts = [
            (
                "diagnostics.check_platform",
                "scripts/ops/check_platform.py",
                (),
            ),
            (
                "diagnostics.audio_devices",
                "scripts/audio/list_devices.py",
                ("voice",),
            ),
        ]
        if "xlerobot" in robot_types or "mock" in robot_types:
            scripts.extend(
                [
                    (
                        "diagnostics.xlerobot.full",
                        "scripts/robots/xlerobot/diagnose.py",
                        ("inspect_scene", "human_follow", "move_base", "set_arm_pose"),
                    ),
                    (
                        "diagnostics.xlerobot.servos",
                        "scripts/robots/xlerobot/scan_servos.py",
                        (
                            "move_base",
                            "base_velocity_step",
                            "set_arm_pose",
                            "set_gripper",
                        ),
                    ),
                    (
                        "diagnostics.xlerobot.camera",
                        "scripts/robots/xlerobot/scan_cameras.py",
                        ("inspect_scene", "human_follow"),
                    ),
                ]
            )
        reports = []
        root = _repo_root()
        for component, relative, impacted in scripts:
            path = root / relative
            ok = path.exists()
            reports.append(
                HealthReport(
                    component=component,
                    status="available" if ok else "missing",
                    severity="info" if ok else "error",
                    evidence=str(path),
                    impacted_skills=tuple(impacted),
                    fix_hint=None if ok else f"Restore {relative}.",
                    source="diagnostic.script_inventory",
                    metadata={"command": f"uv run python {relative} --json"},
                )
            )
        return reports

    def _audio_reports(self) -> list[HealthReport]:
        if "voice" not in {
            channel.type for channel in self.config.channels.values() if channel.enabled
        }:
            return []
        script = _repo_root() / "scripts" / "audio" / "list_devices.py"
        list_devices = _load_function(script, "list_devices")
        if list_devices is None:
            return []
        devices = list_devices()
        if devices and isinstance(devices[0], dict) and devices[0].get("ok") is False:
            return [
                HealthReport(
                    component="audio.devices",
                    status="failed",
                    severity="warning",
                    evidence=str(devices[0].get("error")),
                    fix_hint="Install sounddevice or configure voice input/output devices.",
                    source="audio.list_devices",
                )
            ]
        inputs = [item for item in devices if item.get("input")]
        outputs = [item for item in devices if item.get("output")]
        reports = [
            HealthReport(
                component="audio.input",
                status="ok" if inputs else "failed",
                severity="info" if inputs else "warning",
                evidence=f"{len(inputs)} input devices detected.",
                fix_hint=None
                if inputs
                else "Connect or configure a microphone device.",
                source="audio.list_devices",
                metadata={"devices": inputs[:5]},
            ),
            HealthReport(
                component="audio.output",
                status="ok" if outputs else "failed",
                severity="info" if outputs else "warning",
                evidence=f"{len(outputs)} output devices detected.",
                fix_hint=None
                if outputs
                else "Connect or configure a speaker output device.",
                source="audio.list_devices",
                metadata={"devices": outputs[:5]},
            ),
        ]
        return reports


def _overall_status(reports: list[HealthReport]) -> str:
    severities = {
        report.severity
        for report in reports
        if report.status not in {"ok", "available", "ready_check_required"}
    }
    if "error" in severities:
        return "failed"
    if "warning" in severities:
        return "degraded"
    return "ok"


def _component_from_issue(message: str, *, robot_id: str | None) -> str | None:
    if robot_id and robot_id not in message:
        return None
    for token in message.replace(",", " ").split():
        if token.startswith("skill"):
            continue
        if robot_id and token == robot_id:
            return f"robot.{robot_id}"
    if "skill" in message:
        return "skill.config"
    if "resource path" in message:
        return "resources"
    return "deployment"


def _skills_from_text(message: str) -> tuple[str, ...]:
    words = [word.strip(",.:;") for word in message.split()]
    if "skill" not in words:
        return ()
    index = words.index("skill")
    if index + 1 >= len(words):
        return ()
    name = words[index + 1]
    return (name,) if name else ()


def _fix_hint(message: str) -> str | None:
    lower = message.lower()
    if "camera" in lower:
        return "Run camera scan and update the camera device mapping in the deployment config."
    if "primitive" in lower or "does not support" in lower:
        return "Check robot driver primitives and keep only skills supported by this embodiment."
    if "capability" in lower:
        return "Start or configure the required capability service before enabling the skill."
    if "resource path" in lower:
        return (
            "Fix the runtime/media/episode path permissions or choose writable paths."
        )
    if "skills.enabled" in lower:
        return "Update skills.enabled so the production surface only contains semantic user-facing skills."
    return None


def _resource_fix_hint(resources: tuple[str, ...]) -> str | None:
    hints: list[str] = []
    if "camera" in resources:
        hints.append("camera scan")
    if "base" in resources:
        hints.append("base motion diagnose")
    if "arm" in resources or "gripper" in resources:
        hints.append("servo scan")
    if not hints:
        return None
    return "Before running on hardware, verify " + ", ".join(hints) + "."


def _task_fix_hint(reason: str | None) -> str | None:
    if not reason:
        return "Open the task cockpit and inspect the latest evidence and recovery strategy."
    lower = reason.lower()
    if "camera" in lower or "image" in lower or "observation" in lower:
        return "Check camera availability, then re-run inspect_scene before retrying."
    if "lost" in lower or "target" in lower:
        return "Bring the target back into view and resume from the cockpit or voice channel."
    return "Use the cockpit recovery actions to inspect, retry, or abort the task."


def _skill_matches_robot(
    config: DeploymentConfig,
    robot_id: str,
    supported_robots: tuple[str, ...],
) -> bool:
    if not supported_robots:
        return True
    robot = config.robots.get(robot_id)
    return robot is not None and robot.robot_family in set(supported_robots)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _task_run_store(root: str | Path):
    from hey_robot.agents.task_run import TaskRunStore

    return TaskRunStore(root)


def _load_function(path: Path, name: str):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        f"hey_robot_diagnostic_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        return None
    fn = getattr(module, name, None)
    return fn if callable(fn) else None


def _platform_fix_hint(name: str) -> str | None:
    if name == "nats_server":
        return "Install nats-server and ensure it is available on PATH."
    if name.startswith("import:"):
        return "Run uv sync with the required extras for this deployment."
    if name == "python":
        return "Use Python 3.12 for this project."
    return None


def _component_reports_for_robot(
    robot_id: str, components: dict[str, Any]
) -> list[HealthReport]:
    reports: list[HealthReport] = []
    camera = components.get("camera")
    if isinstance(camera, dict) and bool(camera.get("enabled", True)):
        device_id = camera.get("device_id")
        backend = str(camera.get("backend", "auto"))
        ok = isinstance(device_id, int) and device_id >= 0
        reports.append(
            HealthReport(
                component=f"robot.{robot_id}.camera",
                status="configured" if ok else "missing",
                severity="info" if ok else "warning",
                evidence=f"device_id={device_id} backend={backend}",
                impacted_skills=("inspect_scene", "human_follow"),
                fix_hint=None
                if ok
                else "Run camera scan and set components.camera.device_id.",
                source="robot.component_config",
                metadata=dict(camera),
            )
        )
    base = components.get("base")
    if isinstance(base, dict) and bool(base.get("enabled", True)):
        reports.append(
            HealthReport(
                component=f"robot.{robot_id}.base",
                status="configured",
                severity="info",
                evidence=f"type={base.get('type', 'unknown')}",
                impacted_skills=(
                    "move_base",
                    "turn_base",
                    "base_velocity_step",
                    "human_follow",
                ),
                fix_hint="Run xlerobot diagnose or servo scan before live motion.",
                source="robot.component_config",
                metadata=dict(base),
            )
        )
    arm = components.get("arm")
    if isinstance(arm, dict) and bool(arm.get("enabled", True)):
        joint_ids = (
            arm.get("joint_ids") if isinstance(arm.get("joint_ids"), dict) else {}
        )
        missing = not bool(joint_ids)
        reports.append(
            HealthReport(
                component=f"robot.{robot_id}.arm",
                status="missing" if missing else "configured",
                severity="warning" if missing else "info",
                evidence=f"type={arm.get('type', 'unknown')} joints={joint_ids}",
                impacted_skills=("set_arm_pose", "set_gripper"),
                fix_hint=None
                if not missing
                else "Configure arm joint_ids and run servo scan.",
                source="robot.component_config",
                metadata=dict(arm),
            )
        )
    return reports


def _skills_for_resources(resources: tuple[str, ...]) -> tuple[str, ...]:
    skills: list[str] = []
    if "camera" in resources:
        skills.extend(["inspect_scene", "human_follow"])
    if "base" in resources:
        skills.extend(["move_base", "turn_base", "base_velocity_step", "human_follow"])
    if "arm" in resources:
        skills.extend(["set_arm_pose", "set_gripper"])
    return tuple(dict.fromkeys(skills))
