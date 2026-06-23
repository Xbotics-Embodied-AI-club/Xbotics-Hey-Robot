"""Unified Hey Robot CLI entry point."""

from __future__ import annotations

import sys
from importlib import import_module

CLI_ACTIONS: dict[str, str] = {
    "agent": "hey_robot.cli.agent:main",
    "capability-service": "hey_robot.cli.capability_service:main",
    "doctor": "hey_robot.cli.doctor:main",
    "gateway": "hey_robot.cli.gateway:main",
    "human-follow": "hey_robot.cli.human_follow:main",
    "robot": "hey_robot.cli.robot:main",
    "task-supervisor": "hey_robot.cli.task_supervisor:main",
    "run": "hey_robot.cli.run:main",
    "inspect": "hey_robot.cli.inspect:main",
    "inspect-episode": "hey_robot.cli.episode:main",
    "export-run": "hey_robot.cli.episode:main",
}


def _print_help() -> None:
    actions = ", ".join(sorted(CLI_ACTIONS))
    sys.stdout.write(
        f"usage: hey-robot <action> [options]\n\nHey Robot agent system CLI\n\nactions: {actions}\n"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return

    action_name = sys.argv[1]
    target = CLI_ACTIONS.get(action_name)
    if target is None:
        _print_help()
        raise SystemExit(f"unknown CLI action: {action_name}")

    forwarded = sys.argv[2:]
    if action_name == "inspect-episode":
        forwarded = ["inspect", *forwarded]
    elif action_name == "export-run":
        forwarded = ["export", *forwarded]
    sys.argv = [f"hey-robot {action_name}", *forwarded]
    module_name, func_name = target.split(":", 1)
    runner = getattr(import_module(module_name), func_name)
    runner()


if __name__ == "__main__":
    main()
