"""校验项目配置文件中的引用是否过期。

用途：
  - 开发期工具：检测 ruff.toml 和 poe_tasks.toml 里引用的脚本/文档路径是否还存在。
  - 防止重命名/删除文件后忘记更新 poe 任务或 lint 配置。
  - CI 流水线会自动跑这个；本地改完文件结构后也建议跑一次。

常见用法：
  # 跑校验，没有输出就是通过
  uv run python scripts/dev/validate_config.py

输出说明：
  - 没输出（退出码 0）：所有引用都有效。
  - stderr 列出失效的引用：哪个配置文件、哪条规则、引用了哪个不存在的路径。

退出码：
  - 0：校验通过
  - 1：发现失效引用

注意：
  这是开发期工具，普通用户不需要跑。文件改名/删除后建议跑一次。
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_ruff_per_file_ignores())
    errors.extend(_check_poe_task_refs())
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    return 0


def _check_ruff_per_file_ignores() -> list[str]:
    errors: list[str] = []
    ruff_toml = ROOT / "ruff.toml"
    if not ruff_toml.exists():
        return errors
    data = tomllib.loads(ruff_toml.read_text(encoding="utf-8"))
    ignores: dict[str, list[str]] = data.get("lint", {}).get("per-file-ignores", {})
    for pattern, rules in ignores.items():
        if not any(ROOT.glob(pattern)):
            errors.append(
                f"ruff.toml per-file-ignore glob '{pattern}' "
                f"matches no files (rules: {rules})"
            )
    return errors


def _check_poe_task_refs() -> list[str]:
    errors: list[str] = []
    task_sources: list[tuple[str, dict]] = []

    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        tasks = data.get("tool", {}).get("poe", {}).get("tasks", {})
        if tasks:
            task_sources.append(("pyproject.toml", tasks))

    poe_config = ROOT / "poe_tasks.toml"
    if poe_config.exists():
        data = tomllib.loads(poe_config.read_text(encoding="utf-8"))
        tasks = data.get("tasks", {})
        if tasks:
            task_sources.append(("poe_tasks.toml", tasks))

    if len(task_sources) > 1:
        errors.append(
            "Poe tasks are defined in both pyproject.toml and poe_tasks.toml; "
            "keep one configuration source"
        )

    for filename, tasks in task_sources:
        for name, task in tasks.items():
            if not isinstance(task, dict):
                continue
            commands = []
            if "cmd" in task:
                commands.append((str(task["cmd"]), f"{filename} tasks.{name}"))
            commands.extend(
                (str(step["cmd"]), f"{filename} tasks.{name}[{index}]")
                for index, step in enumerate(task.get("sequence", ()))
                if isinstance(step, dict) and "cmd" in step
            )
            for command, source in commands:
                errors.extend(_extract_stale_paths(command, source))
    return errors


def _extract_stale_paths(cmd: str, source: str) -> list[str]:
    errors: list[str] = []
    for match in re.finditer(r"(?:scripts|docs)/[^\s;|><]+", cmd):
        path_str = match.group(0)
        if not (ROOT / path_str).exists():
            errors.append(f"{source}: referenced path '{path_str}' does not exist")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
