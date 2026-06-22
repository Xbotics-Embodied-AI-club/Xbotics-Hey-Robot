"""检查 hey-robot 运行所需的平台环境（Python / NATS / 依赖 / 配置）。

用法:
    uv run python scripts/ops/check_platform.py                                    # 基础检查
    uv run python scripts/ops/check_platform.py --config configs/xlerobot.real.ubuntu.yaml
    uv run python scripts/ops/check_platform.py --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hey_robot.config.model import DeploymentConfig

ROOT = Path(__file__).resolve().parents[2]
_SEPARATOR = "─" * 60


def _display_width(text: str) -> int:
    w = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1100 <= cp <= 0x115F
            or 0x2E80 <= cp <= 0xA4CF
            or 0xAC00 <= cp <= 0xD7A3
            or 0xF900 <= cp <= 0xFAFF
            or 0xFF01 <= cp <= 0xFF60
            or 0xFFE0 <= cp <= 0xFFE6
        ):
            w += 2
        else:
            w += 1
    return w


def _pad_cjk(text: str, width: int) -> str:
    dw = _display_width(text)
    return text + " " * (width - dw) if dw < width else text


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


class PlatformReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        sec: list[str] = []
        sec.append(self._title())
        sec.append(self._checks())
        if self._r.get("config"):
            sec.append(self._config_checks())
        sec.append(self._recommendation())
        return "\n".join(sec)

    def _title(self) -> str:
        r = self._r
        ready = r["ready"]
        status = "✓ 就绪" if ready else "✗ 异常"
        return (
            f"\n  平台环境检查\n  {'=' * 12}\n\n"
            f"  整体: {status}\n"
            f"  系统: {r['platform']['system']} {r['platform']['release']}"
            f"  Python {r['platform']['python']}\n"
        )

    def _checks(self) -> str:
        lines = ["\n  ▸ 基础检查\n"]
        kw = 16
        for item in self._r["checks"]:
            icon = "✓" if item["ok"] else "✗"
            lines.append(f"    {icon} {_pad_cjk(item['name'], kw)}{item['detail']}")
        return "\n".join(lines)

    def _config_checks(self) -> str:
        config = self._r["config"]
        lines: list[str] = ["\n  ▸ 配置检查\n"]
        kw = 20
        for item in config["checks"]:
            icon = "✓" if item["ok"] else "✗"
            lines.append(f"    {icon} {_pad_cjk(item['name'], kw)}{item['detail']}")
        channels = config.get("enabled_channels", [])
        if channels:
            lines.append(f"\n    启用通道: {', '.join(channels)}")
        return "\n".join(lines)

    def _recommendation(self) -> str:
        return (
            f"\n  {_SEPARATOR}\n"
            f"\n"
            f"  推荐配置文件:\n"
            f"    Ubuntu 真机    configs/xlerobot.real.ubuntu.yaml\n"
            f"    Ubuntu 仿真    configs/xlerobot.sim.ubuntu.yaml\n"
            f"    Windows 真机   configs/xlerobot.real.windows.yaml\n"
            f"    Windows 仿真   configs/xlerobot.sim.windows.yaml\n"
            f"    开发 (mock)    configs/mock.dev.yaml\n"
            f"\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 hey-robot 运行所需的平台环境。")
    parser.add_argument("--config", default=None, help="顺带校验的部署配置 YAML 路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    report = build_report(config_path=args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(PlatformReport(report).render())
    raise SystemExit(0 if report["ready"] else 2)


def build_report(*, config_path: str | None) -> dict[str, Any]:
    config_report = inspect_config(config_path) if config_path else None
    enabled_channels = (
        set(config_report.get("enabled_channels", [])) if config_report else set()
    )
    enabled_robot_types = (
        set(config_report.get("enabled_robot_types", [])) if config_report else set()
    )
    checks: list[CheckResult] = [check_python(), check_nats_server()]
    checks.extend(
        build_import_checks(
            enabled_channels=enabled_channels, enabled_robot_types=enabled_robot_types
        )
    )
    ready = all(item.ok for item in checks)
    if config_report is not None:
        ready = ready and config_report["ready"]
    return {
        "ready": ready,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "checks": [asdict(item) for item in checks],
        "config": config_report,
    }


def check_python() -> CheckResult:
    ok = sys.version_info[:2] == (3, 12)
    return CheckResult(
        name="python", ok=ok, detail=f"python={sys.version.split()[0]}  required=3.12.x"
    )


def check_nats_server() -> CheckResult:
    resolved = shutil.which("nats-server")
    return CheckResult(
        name="nats_server",
        ok=resolved is not None,
        detail=resolved or "nats-server 未安装",
    )


def check_optional_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(
            name=f"import:{module_name}",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(name=f"import:{module_name}", ok=True, detail="已安装")


def build_import_checks(
    *, enabled_channels: set[str], enabled_robot_types: set[str]
) -> list[CheckResult]:
    modules: set[str] = set()
    if not enabled_channels and not enabled_robot_types:
        modules.update({"cv2", "serial"})
    if enabled_robot_types:
        modules.update({"cv2", "serial"})
    if "web" in enabled_channels:
        modules.update({"fastapi", "uvicorn"})
    if "voice" in enabled_channels:
        modules.add("sounddevice")
    return [check_optional_import(m) for m in sorted(modules)]


def inspect_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {
            "ready": False,
            "path": str(path),
            "checks": [
                asdict(CheckResult(name="config_path", ok=False, detail="文件不存在"))
            ],
        }
    config = DeploymentConfig.from_yaml(path)
    checks: list[CheckResult] = [
        CheckResult(name="config_path", ok=True, detail=str(path))
    ]
    checks.extend(check_robot_platform_constraints(config))
    return {
        "ready": all(item.ok for item in checks),
        "path": str(path),
        "deployment_id": config.deployment.id,
        "enabled_channels": [cid for cid, ch in config.channels.items() if ch.enabled],
        "enabled_robot_types": [r.type for r in config.robots.values() if r.enabled],
        "checks": [asdict(item) for item in checks],
    }


def check_robot_platform_constraints(config: DeploymentConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    is_windows = platform.system().lower().startswith("win")
    expected_backends = {"dshow", "msmf", "auto"} if is_windows else {"v4l2", "auto"}
    for robot_id, robot in config.robots.items():
        settings = robot.settings
        serial_bus = settings.get("serial_bus")
        if isinstance(serial_bus, dict):
            port = str(serial_bus.get("port", "")).strip()
            results.append(
                CheckResult(
                    name=f"{robot_id}:serial_port",
                    ok=is_valid_serial_port(port, is_windows=is_windows),
                    detail=port or "missing serial_bus.port",
                )
            )
        components = settings.get("components")
        if not isinstance(components, dict):
            continue
        camera = components.get("camera")
        if isinstance(camera, dict) and bool(camera.get("enabled", True)):
            backend = str(camera.get("backend", "auto")).strip().lower()
            device_id = camera.get("device_id")
            results.append(
                CheckResult(
                    name=f"{robot_id}:camera_backend",
                    ok=backend in expected_backends,
                    detail=f"backend={backend}  expected={sorted(expected_backends)}",
                )
            )
            results.append(
                CheckResult(
                    name=f"{robot_id}:camera_device",
                    ok=isinstance(device_id, int) and device_id >= 0,
                    detail=f"device_id={device_id}",
                )
            )
    return results


def is_valid_serial_port(port: str, *, is_windows: bool) -> bool:
    if not port:
        return False
    if is_windows:
        normalized = port.upper()
        return normalized.startswith("COM") and normalized[3:].isdigit()
    return port.startswith("/dev/")


if __name__ == "__main__":
    main()
