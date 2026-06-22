"""下载视觉检测模型（YOLO 等）。

用法:
    uv run python scripts/model_downloads/download_vision_models.py           # yolo26n
    uv run python scripts/model_downloads/download_vision_models.py --list    # 列出可用
    uv run python scripts/model_downloads/download_vision_models.py --all     # 全部
    uv run python scripts/model_downloads/download_vision_models.py --force   # 强制重下
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

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
            f"\r  Downloading {destination.name}: {percent:3d}% "
            f"({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)"
        )
        sys.stdout.flush()

    urls = [url]
    mirror = os.environ.get("GH_PROXY", "").strip()
    if mirror:
        mirror = mirror.rstrip("/")
        urls.append(f"{mirror}/{url}")
    else:
        urls.append(f"https://ghproxy.com/{url}")

    def _try(attempt_url: str) -> bool:
        try:
            with urllib.request.urlopen(attempt_url, timeout=30) as resp:  # noqa: S310
                total_size = int(resp.headers.get("Content-Length", "0"))
                downloaded = 0
                block_size = 1024 * 64
                with destination.open("wb") as handle:
                    while True:
                        chunk = resp.read(block_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        reporthook(downloaded // block_size, block_size, total_size)
            sys.stdout.write("\n")
            return True
        except Exception:
            return False

    for candidate in urls:
        label = "镜像" if "ghproxy" in candidate.lower() else "直连"
        if _try(candidate):
            return
        if candidate != urls[-1]:
            print(f"  {label} 失败，尝试下一个源...")
    raise RuntimeError(f"所有下载源均失败: {url}")


def install_model(name: str, *, force: bool) -> None:
    config = MODELS[name]
    destination = MODELS_DIR / name
    if destination.exists() and not force:
        print(f"  ✓ {name} 已安装: {destination}")
        return
    print(f"  ▸ {name} ({config['description']})")
    download_file(config["url"], destination)
    print(f"  ✓ 已安装: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载视觉检测模型（YOLO 等）。")
    parser.add_argument(
        "model", nargs="?", default="yolo26n", help="模型名，默认 yolo26n"
    )
    parser.add_argument("--all", action="store_true", help="下载所有列出的模型")
    parser.add_argument(
        "--force", action="store_true", help="即使文件已存在也强制重新下载"
    )
    parser.add_argument("--list", action="store_true", help="只列出可用模型，不下载")
    args = parser.parse_args()

    if args.list:
        print("\n  可用视觉模型\n  " + "=" * 12 + "\n")
        for name, config in MODELS.items():
            print(f"    {name:<20s} {config['description']}")
        print()
        return 0

    selected = list(MODELS)
    if not args.all:
        selected_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
        if selected_name not in MODELS:
            print(f"  未知模型: {selected_name}", file=sys.stderr)
            print(f"  可用: {', '.join(MODELS)}", file=sys.stderr)
            return 1
        selected = [selected_name]

    print()
    for name in selected:
        install_model(name, force=args.force)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
