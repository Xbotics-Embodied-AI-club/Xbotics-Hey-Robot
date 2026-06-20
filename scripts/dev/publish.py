"""把 dist/ 下的构建产物发布到自定义 PyPI 仓库。

用途：
  - 维护者工具：CI 流水线或本地手动发包时使用。
  - 读 dist/ 下所有 wheel/sdist 文件，调用 uv publish 推送到 custom-repo 索引。
  - 需要 PYPI_PASS 环境变量；没有会直接报错。

常见用法：
  # 先构建（在仓库根）
  uv build

  # 发布
  uv run python scripts/dev/publish.py

输出说明：
  - dist/ 为空时打印：未找到发布产物，跳过发布
  - 缺 PYPI_PASS 时报错退出
  - 正常情况下由 uv publish 接管输出

退出码：
  - 0：发布成功
  - 非 0：PYPI_PASS 缺失，或 uv publish 失败

注意：
  仅维护者使用，普通用户不需要跑这个脚本。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "dist"


def main() -> None:
    artifacts = list(DIST_DIR.iterdir()) if DIST_DIR.exists() else []
    if not artifacts:
        print("未找到发布产物（dist/ 为空），跳过发布")
        return

    password = os.environ.get("PYPI_PASS")
    if not password:
        raise SystemExit("环境变量 PYPI_PASS 未设置")

    command = [
        "uv",
        "publish",
        "-u",
        "admin",
        "-p",
        password,
        "--index",
        "custom-repo",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
