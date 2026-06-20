"""扫描本机摄像头设备，列出每个设备的可用分辨率并保存视角截图。

用途：
  - 不知道接了几个摄像头、device_id 怎么对应时，用这个工具快速枚举。
  - 默认会把每个能打开的摄像头当前画面截图保存，打开图片就能判断 device_id 0/1/...
    分别对应物理上哪个摄像头（头、手腕、侧视等）。
  - diagnose.py 在摄像头异常时也会自动调用本模块的 scan_cameras()。

常见用法：
  # 最简单：扫描 device 0..4，截图默认保存到 outputs/diagnostic/cameras/
  uv run python scripts/robots/xlerobot/scan_cameras.py

  # 自定义扫描范围和保存目录
  uv run python scripts/robots/xlerobot/scan_cameras.py --limit 8 --samples-dir ./my_cam_check

  # 不保存截图（只看分辨率列表）
  uv run python scripts/robots/xlerobot/scan_cameras.py --no-samples

  # 机器可读 JSON 输出
  uv run python scripts/robots/xlerobot/scan_cameras.py --json

输出说明：
  - 设备 N：可用/不可用  分辨率=WxH  截图已保存=<路径>
  - "不可用" 通常表示该 device_id 没有对应物理设备，或被其他程序占用。

退出码：
  - 0：至少一个摄像头可用
  - 2：全部不可用（请检查 USB 连接和驱动）

更多选项：uv run python scripts/robots/xlerobot/scan_cameras.py --help
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES_DIR = REPO_ROOT / "outputs" / "diagnostic" / "cameras"


def _silence_opencv_warnings() -> Any:
    """Best-effort: silence OpenCV warnings during device probing."""
    import cv2

    previous = None
    with contextlib.suppress(Exception):
        previous = cv2.getLogLevel()
    with contextlib.suppress(Exception):
        cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 0))
    return previous


def _restore_opencv_log_level(previous: Any) -> None:
    if previous is None:
        return
    with contextlib.suppress(Exception):
        import cv2

        cv2.setLogLevel(previous)


def scan_cameras(
    *, limit: int, sample_dir: Path | None = None, backend: str = "auto"
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        import cv2
    except ImportError as exc:
        return [
            {"device_id": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ]

    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
    backend_id = opencv_backend(cv2, backend)
    previous_log_level = _silence_opencv_warnings()
    results = []
    try:
        for device_id in range(limit):
            capture = (
                cv2.VideoCapture(device_id, backend_id)
                if backend_id is not None
                else cv2.VideoCapture(device_id)
            )
            try:
                if not capture.isOpened():
                    results.append(
                        {"device_id": device_id, "ok": False, "error": "not opened"}
                    )
                    continue
                ok, frame = capture.read()
                sample_path = None
                if ok and frame is not None and sample_dir is not None:
                    sample_path = sample_dir / f"device_{device_id}.jpg"
                    cv2.imwrite(str(sample_path), frame)
                results.append(
                    {
                        "device_id": device_id,
                        "ok": bool(ok and frame is not None),
                        "shape": list(frame.shape)
                        if ok and frame is not None
                        else None,
                        "sample_path": str(sample_path) if sample_path else None,
                        "error": None
                        if ok and frame is not None
                        else "opened but no frame",
                    }
                )
            finally:
                capture.release()
    finally:
        _restore_opencv_log_level(previous_log_level)
    return results


def opencv_backend(cv2: Any, backend: str) -> int | None:
    normalized = backend.lower().strip()
    if normalized in {"", "auto", "default"}:
        return cv2.CAP_DSHOW if sys.platform.startswith("win") else None
    if normalized == "dshow":
        return cv2.CAP_DSHOW
    if normalized == "msmf":
        return cv2.CAP_MSMF
    if normalized == "v4l2":
        return cv2.CAP_V4L2
    return None


_ERROR_HINTS = {
    "not opened": "无法打开（可能被其他程序占用，或设备号无效）",
    "opened but no frame": "已打开但读不到画面（USB 带宽不足或驱动异常）",
}


def _explain_error(error: str | None) -> str:
    if not error:
        return "未知原因"
    return _ERROR_HINTS.get(error, error)


def format_camera_scan(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in results:
        ok = bool(item["ok"])
        status = "可用" if ok else "不可用"
        shape = item.get("shape")
        if shape:
            detail = f"分辨率={shape[1]}x{shape[0]}"
        else:
            detail = f"原因={_explain_error(item.get('error'))}"
        sample = (
            f"  截图已保存={item['sample_path']}" if item.get("sample_path") else ""
        )
        lines.append(f"  - 设备 {item['device_id']}：{status}  {detail}{sample}")
    usable = [item["device_id"] for item in results if item["ok"]]
    if usable:
        lines.append(f"建议使用 device_id = {usable[0]}（第一个可用的摄像头）")
    else:
        lines.append("未检测到任何可用摄像头，请检查 USB 连接和驱动")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描本机摄像头设备，列出每个设备的分辨率并保存视角截图。"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="扫描 device 0..N-1，默认 5"
    )
    parser.add_argument(
        "--backend",
        default="dshow",
        help="OpenCV 后端：auto/dshow/msmf/v4l2，Windows 默认 dshow",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help=f"截图保存目录，默认 {DEFAULT_SAMPLES_DIR}",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="不保存截图（只看分辨率列表）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    if args.no_samples:
        sample_dir = None
    else:
        sample_dir = Path(args.samples_dir) if args.samples_dir else DEFAULT_SAMPLES_DIR
        sample_dir.mkdir(parents=True, exist_ok=True)

    results = scan_cameras(
        limit=max(args.limit, 0),
        sample_dir=sample_dir,
        backend=args.backend,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "cameras": results,
                    "samples_dir": str(sample_dir) if sample_dir else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("摄像头扫描：")
        for line in format_camera_scan(results):
            print(line)
        if sample_dir:
            print(f"\n视角截图已保存到：{sample_dir}")
            print("打开图片即可确认每个 device_id 对应哪个物理摄像头。")

    raise SystemExit(0 if any(item["ok"] for item in results) else 2)


if __name__ == "__main__":
    main()
