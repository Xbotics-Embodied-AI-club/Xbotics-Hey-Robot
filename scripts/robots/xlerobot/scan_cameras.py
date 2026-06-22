"""扫描本机摄像头并保存视角截图，帮助识别每个 device_id 对应的物理摄像头。

用法:
    uv run python scripts/robots/xlerobot/scan_cameras.py               # 默认扫描 0..4
    uv run python scripts/robots/xlerobot/scan_cameras.py --limit 8     # 扩大范围
    uv run python scripts/robots/xlerobot/scan_cameras.py --json        # 机器可读
    uv run python scripts/robots/xlerobot/scan_cameras.py --no-samples  # 不存截图
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES_DIR = ROOT / "outputs" / "diagnostic" / "cameras"

_SEPARATOR = "─" * 60

_ERROR_HINTS: dict[str, str] = {
    "not opened": "无法打开（被其他程序占用或设备号无效）",
    "opened but no frame": "已打开但读不到画面（USB 带宽不足或驱动异常）",
}


def _default_backend() -> str:
    return "dshow" if sys.platform.startswith("win") else "v4l2"


# ── 数据模型 ──


@dataclass(frozen=True)
class CameraInfo:
    device_id: int
    ok: bool
    width: int | None = None
    height: int | None = None
    sample_path: str | None = None
    error: str | None = None
    note: str = ""

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "—"

    @property
    def error_hint(self) -> str:
        if self.ok or not self.error:
            return ""
        return _ERROR_HINTS.get(self.error, self.error)

    @property
    def status_icon(self) -> str:
        return "✓" if self.ok else "✗"


# ── 扫描逻辑 ──


class CameraScanner:
    def __init__(self, *, backend: str = "") -> None:
        self._backend = backend or _default_backend()

    def scan(self, *, limit: int, save_dir: Path | None = None) -> list[CameraInfo]:
        if limit <= 0:
            return []
        try:
            import cv2
        except ImportError as exc:
            logger.error("OpenCV 未安装: %s", exc)
            return []

        backend_id = _resolve_backend(cv2, self._backend)
        previous_log = _silence_opencv()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        try:
            results: list[CameraInfo] = [
                self._probe(cv2, i, backend_id, save_dir) for i in range(limit)
            ]
        finally:
            _restore_opencv(cv2, previous_log)
        return results

    @staticmethod
    def _probe(
        cv2: Any, device_id: int, backend_id: int | None, save_dir: Path | None
    ) -> CameraInfo:
        cap = (
            cv2.VideoCapture(device_id, backend_id)
            if backend_id is not None
            else cv2.VideoCapture(device_id)
        )
        try:
            if not cap.isOpened():
                return CameraInfo(device_id=device_id, ok=False, error="not opened")

            ok, frame = cap.read()
            if not ok or frame is None:
                return CameraInfo(
                    device_id=device_id, ok=False, error="opened but no frame"
                )

            sample: str | None = None
            if save_dir is not None:
                path = save_dir / f"device_{device_id}.jpg"
                cv2.imwrite(str(path), frame)
                sample = str(path)

            h, w = frame.shape[:2]
            note = _guess_camera_role(device_id, w, h)
            return CameraInfo(
                device_id=device_id,
                ok=True,
                width=w,
                height=h,
                sample_path=sample,
                note=note,
            )
        finally:
            cap.release()


# ── 输出格式化 ──


class CameraReport:
    def __init__(
        self, cameras: list[CameraInfo], samples_dir: Path | None = None
    ) -> None:
        self._cameras = cameras
        self._usable = [c for c in cameras if c.ok]
        self._samples_dir = samples_dir

    def render(self) -> str:
        sec: list[str] = []
        sec.append(self._title())
        sec.append(self._results())
        sec.append(self._advice())
        if self._samples_dir:
            sec.append(self._footer())
        return "\n".join(sec)

    def _title(self) -> str:
        return f"\n  摄像头扫描\n  {'=' * 12}\n"

    def _results(self) -> str:
        backend = _default_backend()
        summary = f"  扫描 0..{len(self._cameras) - 1}  后端 {backend}  可用 {len(self._usable)}/{len(self._cameras)}\n"
        rows: list[str] = []
        for c in self._cameras:
            mark = " ✓" if c.ok else " ✗"
            if c.ok:
                info = f"{c.resolution}"
                if c.sample_path:
                    info += "  [已截图]"
            else:
                info = c.error_hint or "未知错误"
            # 第一列：ID + 状态，固定宽度；第二列：信息；第三列：备注
            head = f"    {c.device_id}{mark}"
            hint = f" ← {c.note}" if c.note else ""
            rows.append(f"{head:<8s}{info:<30s}{hint}")
        return summary + "\n".join(rows)

    def _advice(self) -> str:
        sep = f"  {_SEPARATOR}"
        if not self._usable:
            steps = (
                "  未检测到可用摄像头。\n"
                "\n"
                "  排查步骤：\n"
                "    1. ls /dev/video* 检查设备是否存在\n"
                "    2. sudo apt install v4l-utils && v4l2-ctl --list-devices\n"
                "    3. 尝试其他后端: --backend auto\n"
                "    4. 确认摄像头未被其他程序占用"
            )
            return f"\n{sep}\n\n{steps}"

        ids = [c.device_id for c in self._usable]
        lines: list[str] = [
            "",
            sep,
            "",
            "  怎么填配置",
            "",
            "    打开 configs/xlerobot.<env>.<os>.yaml，找到 robots 下的 cameras：",
            "",
            "      robots:",
            "        <robot_name>:",
            "          components:",
            "            cameras:",
            "              front:",
            "                device_id:  <头部 ID>    # 前方/头部摄像头",
            "              wrist:",
            "                device_id:  <腕部 ID>    # 腕部摄像头",
            "",
            f"  本机可用摄像头: {ids}",
        ]
        if self._samples_dir:
            lines.append(f"  截图位置: {self._samples_dir}")
            lines.append("")
            lines.append("  打开截图，确认每个 device_id 对应哪个物理摄像头后填入。")
        lines.append("")
        return "\n".join(lines)

    def _footer(self) -> str:
        return (
            f"\n  {_SEPARATOR}\n"
            f"\n"
            f"  视角截图已保存到: {self._samples_dir}\n"
            f"  打开图片即可确认每个 device_id 对应的物理摄像头。\n"
        )


# ── 辅助函数 ──


def _resolve_backend(cv2: Any, backend: str) -> int | None:
    name = backend.lower().strip()
    if name in {"", "auto", "default"}:
        return cv2.CAP_DSHOW if sys.platform.startswith("win") else None
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "v4l2":
        return cv2.CAP_V4L2
    logger.warning("未识别的后端 %r，使用默认", backend)
    return None


def _guess_camera_role(device_id: int, width: int, _height: int) -> str:
    """根据分辨率和设备号给一个模糊的摄像头角色提示。"""
    if width >= 1280:
        return "可能是头部摄像头（高分辨率）"
    if width <= 640 and device_id > 0:
        return "可能是腕部摄像头（低分辨率，非首位）"
    if device_id == 0:
        return "可能是头部摄像头（首个设备）"
    return ""


def _silence_opencv() -> Any:
    with contextlib.suppress(Exception):
        import cv2

        previous = cv2.getLogLevel()
        cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 0))
        return previous
    return None


def _restore_opencv(cv2: Any, previous: Any) -> None:
    if previous is None:
        return
    with contextlib.suppress(Exception):
        cv2.setLogLevel(previous)


# ── 兼容 diagnose.py 的旧接口 ──


def scan_cameras(
    *, limit: int, sample_dir: Path | None = None, backend: str = "auto"
) -> list[dict[str, Any]]:
    """扫描摄像头，返回 dict 列表。保留给 diagnose.py 调用。"""
    scanner = CameraScanner(backend=backend)
    cameras = scanner.scan(limit=limit, save_dir=sample_dir)
    return [
        {
            "device_id": c.device_id,
            "ok": c.ok,
            "shape": [c.height, c.width] if c.ok and c.width and c.height else None,
            "sample_path": c.sample_path,
            "error": c.error,
        }
        for c in cameras
    ]


def format_camera_scan(results: list[dict[str, Any]]) -> list[str]:
    """将扫描结果格式化为文本行。保留给 diagnose.py 调用。"""
    lines: list[str] = []
    for item in results:
        ok = bool(item["ok"])
        status = "可用" if ok else "不可用"
        shape = item.get("shape")
        if shape and isinstance(shape, list) and len(shape) >= 2:
            detail = f"分辨率={shape[1]}x{shape[0]}"
        else:
            detail = f"原因={_ERROR_HINTS.get(item.get('error') or '', item.get('error') or '未知')}"
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


# ── 主入口 ──


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描本机摄像头并保存视角截图。")
    parser.add_argument(
        "--limit", type=int, default=5, help="扫描 device 0..N-1，默认 5"
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="OpenCV 后端: auto/dshow/msmf/v4l2；不指定则自动选择",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="截图保存目录，默认 outputs/diagnostic/cameras/",
    )
    parser.add_argument("--no-samples", action="store_true", help="不保存截图")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    save_dir = (
        None
        if args.no_samples
        else (Path(args.samples_dir) if args.samples_dir else DEFAULT_SAMPLES_DIR)
    )

    scanner = CameraScanner(backend=args.backend or "")
    cameras = scanner.scan(limit=max(args.limit, 0), save_dir=save_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "cameras": [
                        {
                            "device_id": c.device_id,
                            "ok": c.ok,
                            "width": c.width,
                            "height": c.height,
                            "sample_path": c.sample_path,
                            "error": c.error,
                        }
                        for c in cameras
                    ],
                    "samples_dir": str(save_dir) if save_dir else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    report = CameraReport(cameras, samples_dir=save_dir)
    print(report.render())

    raise SystemExit(0 if any(c.ok for c in cameras) else 2)


if __name__ == "__main__":
    main()
