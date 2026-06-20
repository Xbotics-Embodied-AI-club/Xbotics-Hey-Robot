"""下载本地语音识别（ASR）模型。

用途：
  - 第一次配置 voice channel（语音对话）时跑这个工具下载 sherpa-onnx 模型。
  - 模型放在 models/asr 下，被 channels.voice.asr.sherpa_model_dir 引用。
  - 已经存在的文件默认跳过；用 --force 强制重下。

常见用法：
  # 下载 ASR 模型
  uv run python scripts/model_downloads/download_speech_models.py

  # 强制重新下载
  uv run python scripts/model_downloads/download_speech_models.py --force

输出说明：
  - 下载进度和目标路径
  - 解压完成的模型文件清单

退出码：
  - 0：所选模型下载完成（或已存在跳过）

更多选项：uv run python scripts/model_downloads/download_speech_models.py --help
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache" / "downloads"

MODELS: dict[str, dict[str, object]] = {
    "asr": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
        "target_dir": MODELS_DIR / "asr",
        "files": [
            "encoder.int8.onnx",
            "decoder.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ],
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


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.isdir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue
        if member.issym() or member.islnk():
            raise ValueError(f"unsupported archive link: {member.name}")
        member_path.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"unable to extract archive member: {member.name}")
        with source, member_path.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def is_installed(target_dir: Path, files: list[str]) -> bool:
    return target_dir.exists() and all((target_dir / name).exists() for name in files)


def find_file(root: Path, filename: str) -> Path | None:
    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def install_model(model_name: str, *, force: bool) -> None:
    config = MODELS[model_name]
    url = str(config["url"])
    target_dir = Path(config["target_dir"])
    files = list(config["files"])
    archive_name = url.rsplit("/", 1)[-1]
    archive_path = CACHE_DIR / archive_name
    extract_root = CACHE_DIR / "extracted" / model_name

    if not force and is_installed(target_dir, files):
        print(f"{model_name} already installed: {target_dir}")
        return

    print(f"Installing {model_name} into {target_dir}")
    download_file(url, archive_path)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:bz2") as archive:
        safe_extract(archive, extract_root)

    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        source = find_file(extract_root, filename)
        if source is None:
            raise FileNotFoundError(f"missing {filename} in archive {archive_name}")
        shutil.copy2(source, target_dir / filename)

    shutil.rmtree(extract_root, ignore_errors=True)
    archive_path.unlink(missing_ok=True)
    print(f"Installed {model_name}: {target_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载本地语音识别（ASR）模型。")
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使文件已存在也强制重新下载",
    )
    args = parser.parse_args()

    install_model("asr", force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
