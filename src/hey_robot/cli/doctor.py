from __future__ import annotations

import argparse
import json
import sys

from hey_robot.config import DeploymentConfig
from hey_robot.health import HealthReportService


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except OSError:
        return


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Build a Hey Robot health report")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument("--robot", default=None, help="Robot id to inspect")
    parser.add_argument("--episode-dir", default=None, help="Episode store directory")
    parser.add_argument(
        "--full", action="store_true", help="Include recent task failures"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow live hardware checks when supported by a diagnostic provider",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    payload = HealthReportService(
        config,
        episode_dir=args.episode_dir,
        config_path=args.config,
        live=args.live,
    ).payload(
        robot_id=args.robot,
        full=args.full,
    )
    if args.json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    lines = [f"health: {payload['status']}"]
    if args.robot:
        lines.append(f"robot: {args.robot}")
    for report in payload["reports"]:
        impacted = ", ".join(report["impacted_skills"]) or "none"
        lines.append(
            f"- {report['severity']} {report['component']} "
            f"[{report['status']}]: {report['evidence']}"
        )
        lines.append(f"  impacted_skills: {impacted}")
        if report.get("fix_hint"):
            lines.append(f"  fix_hint: {report['fix_hint']}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
