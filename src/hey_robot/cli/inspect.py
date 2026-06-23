from __future__ import annotations

import argparse
import json
import sys

from hey_robot.capability.catalog import CapabilityLoader
from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.skills.registry import registry_from_config


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


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Hey Robot 部署配置。")
    parser.add_argument(
        "section",
        nargs="?",
        choices=["deployment", "capabilities"],
        default="deployment",
    )
    parser.add_argument("--config", required=True, help="部署配置 YAML 路径")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    if args.section == "capabilities":
        manifest = CapabilityLoader(robot_skills=registry_from_config(config)).build()
        sys.stdout.write(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        return

    issues = validate_deployment(config)
    out: list[str] = []

    title = f"  部署配置检查: {config.deployment.id}"
    out.append(title)
    out.append(f"  {'=' * (len(config.deployment.id) + 11)}")
    out.append("")

    kw = 10
    robots = ", ".join(sorted(config.robots)) or "(无)"
    agents = ", ".join(sorted(config.agents)) or "(无)"
    channels = ", ".join(sorted(config.channels)) or "(无)"
    out.append(f"  {_pad_cjk('机器人', kw)}{robots}")
    out.append(f"  {_pad_cjk('智能体', kw)}{agents}")
    out.append(f"  {_pad_cjk('通道', kw)}{channels}")
    out.append(f"  {_pad_cjk('运行时', kw)}{config.resources.runtime_dir}")
    out.append(f"  {_pad_cjk('媒体', kw)}{config.resources.media_root}")
    out.append(f"  {_pad_cjk('会话', kw)}{config.resources.episodes_root}")

    if issues:
        out.append(f"\n  ✗ 配置问题 ({len(issues)} 个):")
        for issue in issues:
            tag = "✗" if issue.level == "error" else "\u26a0"
            out.append(f"    {tag} [{issue.level}] {issue.message}")
    else:
        out.append("\n  \u2713 配置有效，可以启动 runtime。")
    out.append("")
    sys.stdout.write("\n".join(out) + "\n")
