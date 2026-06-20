from __future__ import annotations

import argparse
import json
import sys

from hey_robot.capability.catalog import CapabilityLoader
from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.skills.registry import registry_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Hey Robot deployment config"
    )
    parser.add_argument(
        "section",
        nargs="?",
        choices=["deployment", "capabilities"],
        default="deployment",
    )
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    if args.section == "capabilities":
        manifest = CapabilityLoader(
            robot_skills=registry_from_config(config),
        ).build()
        sys.stdout.write(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        return

    issues = validate_deployment(config)
    lines = [
        f"deployment: {config.deployment.id} mode={config.deployment.mode}",
        f"robots: {', '.join(sorted(config.robots)) or 'none'}",
        f"agents: {', '.join(sorted(config.agents)) or 'none'}",
        f"channels: {', '.join(sorted(config.channels)) or 'none'}",
        (
            f"resources: runtime={config.resources.runtime_dir} "
            f"media={config.resources.media_root} episodes={config.resources.episodes_root}"
        ),
    ]
    if issues:
        lines.append("issues:")
        lines.extend(f"- {issue.level}: {issue.message}" for issue in issues)
    sys.stdout.write("\n".join(lines) + "\n")
