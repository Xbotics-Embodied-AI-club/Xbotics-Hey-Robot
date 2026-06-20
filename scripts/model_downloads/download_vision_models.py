"""下载本地视觉检测模型（YOLO 等）。

用途：
  - 第一次配置 detect_marker 等需要视觉检测的技能时跑这个工具下载模型权重。
  - 模型放在 models/ 下，被技能运行时按文件名引用。
  - 已经存在的文件默认跳过；用 --force 强制重下。

常见用法：
  # 默认下载 yolo26n（体积最小、速度最快）
  uv run python scripts/model_downloads/download_vision_models.py

  # 列出所有可用模型
  uv run python scripts/model_downloads/download_vision_models.py --list

  # 下载全部可用模型
  uv run python scripts/model_downloads/download_vision_models.py --all

  # 指定模型名（不带 .pt 后缀）
  uv run python scripts/model_downloads/download_vision_models.py yolo26n

  # 强制重新下载
  uv run python scripts/model_downloads/download_vision_models.py --force

输出说明：
  - --list 时打印每个模型文件名和描述
  - 下载进度和目标路径

退出码：
  - 0：所选模型下载完成（或已存在跳过）

更多选项：uv run python scripts/model_downloads/download_vision_models.py --help
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MODELS: dict[str, dict[str, str]] = {
    "yolo26n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "description": "YOLO26 Nano",
    },
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"unsupported download scheme for {url}")

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = int(downloaded * 100 / total_size)
        sys.stdout.write(
            f"\rDownloading {destination.name}: {percent:3d}% "
            f"({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)"
        )
        sys.stdout.flush()

    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:  # noqa: S310
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        block_size = 1024 * 64
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            reporthook(downloaded // block_size, block_size, total_size)
    sys.stdout.write("\n")


def install_model(name: str, *, force: bool) -> None:
    config = MODELS[name]
    destination = MODELS_DIR / name
    if destination.exists() and not force:
        print(f"{name} already installed: {destination}")
        return
    print(f"Installing {name} ({config['description']})")
    download_file(config["url"], destination)
    print(f"Installed {name}: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载本地视觉检测模型（YOLO 等）。")
    parser.add_argument(
        "model",
        nargs="?",
        default="yolo26n",
        help="模型名（不带 .pt 后缀）或完整文件名，默认 yolo26n",
    )
    parser.add_argument("--all", action="store_true", help="下载所有列出的模型")
    parser.add_argument(
        "--force", action="store_true", help="即使文件已存在也强制重新下载"
    )
    parser.add_argument("--list", action="store_true", help="只列出可用模型，不下载")
    args = parser.parse_args()

    if args.list:
        for name, config in MODELS.items():
            print(f"{name}  {config['description']}")
        return 0

    selected = list(MODELS)
    if not args.all:
        selected_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
        if selected_name not in MODELS:
            raise KeyError(f"unknown model: {selected_name}")
        selected = [selected_name]

    for name in selected:
        install_model(name, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
