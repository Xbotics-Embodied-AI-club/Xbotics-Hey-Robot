"""清理 hey-robot 项目中的临时/构建产物。

用途：
  - 跨平台的清理工具，按类别删除指定类型的中间文件。
  - 想从干净状态重新跑测试 / 构建 / lint 时用。
  - 通过 poe 任务封装为：poe clean / poe clean-pyc 等。

常见用法：
  # 清理 Python 字节码缓存
  uv run python scripts/dev/clean.py pyc

  # 清理构建产物
  uv run python scripts/dev/clean.py build

  # 清理测试覆盖率产物
  uv run python scripts/dev/clean.py test

  # 清理 lint 缓存
  uv run python scripts/dev/clean.py lint

  # 一把梭：所有类别都清
  uv run python scripts/dev/clean.py all

输出说明：
  - 删除的每个文件/目录路径（相对仓库根）。
  - 没有可清理内容时打印：nothing to clean

退出码：
  - 0：清理完成（即使没东西可清也算成功）

更多选项：uv run python scripts/dev/clean.py --help
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "pyc": {
        "files": ("*.pyc", "*.pyo", "*~"),
        "dirs": ("__pycache__",),
    },
    "build": {
        "files": (),
        "dirs": ("build", "dist", ".eggs"),
        "dir_globs": ("*.egg-info",),
    },
    "test": {
        "files": (".coverage", "report.xml", "cov.xml"),
        "dirs": ("htmlcov", ".pytest_cache", "tests/api/cassettes"),
    },
    "lint": {
        "files": (),
        "dirs": (".mypy_cache", ".ruff_cache"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="清理项目临时/构建产物，可选范围：pyc/build/test/lint/all。"
    )
    parser.add_argument(
        "target",
        choices=[*TARGETS.keys(), "all"],
        help="清理范围：pyc 字节码 / build 构建产物 / test 测试产物 / lint 缓存 / all 全部",
    )
    args = parser.parse_args()

    targets = list(TARGETS) if args.target == "all" else [args.target]
    removed: list[Path] = []
    for target in targets:
        removed.extend(clean_target(target))

    if removed:
        for path in removed:
            print(path.relative_to(ROOT).as_posix())
    else:
        print("没有需要清理的内容")


def clean_target(name: str) -> list[Path]:
    spec = TARGETS[name]
    removed: list[Path] = []

    for relative in spec.get("dirs", ()):
        path = ROOT / relative
        if path.exists():
            remove_path(path)
            removed.append(path)

    for pattern in spec.get("dir_globs", ()):
        for path in iter_matches(pattern, include_dirs=True, include_files=False):
            remove_path(path)
            removed.append(path)

    for pattern in spec.get("files", ()):
        for path in iter_matches(pattern, include_dirs=False, include_files=True):
            remove_path(path)
            removed.append(path)

    return sorted(set(removed))


def iter_matches(
    pattern: str, *, include_dirs: bool, include_files: bool
) -> Iterable[Path]:
    for path in ROOT.rglob(pattern):
        if path == ROOT:
            continue
        if (path.is_dir() and include_dirs) or (path.is_file() and include_files):
            yield path


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
