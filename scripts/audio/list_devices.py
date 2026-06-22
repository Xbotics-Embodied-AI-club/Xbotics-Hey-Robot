"""列出本机音频设备并给出配置建议。

用法:
    uv run python scripts/audio/list_devices.py             # 精简报告
    uv run python scripts/audio/list_devices.py --verbose   # 显示全部设备
    uv run python scripts/audio/list_devices.py --json      # 机器可读
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import shutil
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WIDTH = shutil.get_terminal_size((80, 24)).columns
SEP = "─" * min(WIDTH, 80)

# ── 设备分类 ──

_HDMI_KW = ("hdmi", "nvida", "nvidia")
_MIC_KW = ("dmic", "digital", "mic")


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int = -1
    is_default_input: bool = False
    is_default_output: bool = False
    _error_message: str | None = field(default=None, repr=False)

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def kind(self) -> str:
        lower = self.name.lower()
        hi, ho = self.max_input_channels > 0, self.max_output_channels > 0
        if not hi and ho:
            return "hdmi" if any(k in lower for k in _HDMI_KW) else "output"
        if hi and not ho:
            return "mic" if any(k in lower for k in _MIC_KW) else "input"
        if hi and ho:
            return "duplex"
        return "none"

    @property
    def short_name(self) -> str:
        """短名称，去掉冗余前缀。"""
        n = self.name
        for prefix in ("HDA NVidia: ", "sof-hda-dsp: "):
            n = n.removeprefix(prefix)
        # USB 摄像头自带麦克风，名字太长，简化
        if "USB Camera" in n and "Audio" in n:
            n = n.replace("1080P ", "").replace(": Audio", "")
        return n

    @property
    def tag(self) -> str:
        """中文设备类型标签。"""
        k = self.kind
        lower = self.name.lower()
        if k == "hdmi":
            return "HDMI 显卡音频"
        if k == "duplex":
            if any(x in lower for x in ("pulse", "default", "pipewire")):
                return "虚拟设备"
            return "声卡"
        if k == "mic":
            if "16k" in lower:
                return "数字麦克风 16kHz"
            if "dmic" in lower:
                return "数字麦克风"
            return "麦克风"
        if k == "input":
            return "输入设备"
        if k == "output":
            return "输出设备"
        return "—"

    @property
    def detail(self) -> str:
        parts: list[str] = []
        if self.is_input:
            parts.append(f"{self.max_input_channels}ch 输入")
        if self.is_output:
            parts.append(f"{self.max_output_channels}ch 输出")
        parts.append(f"{self.default_samplerate:.0f}Hz")
        return "  ".join(parts)

    @property
    def warning(self) -> str:
        if self.kind == "hdmi":
            return "不适合语音"
        return ""

    @property
    def star(self) -> str:
        return ""


def _display_width(text: str) -> int:
    """计算字符串在终端中的显示宽度（CJK 字符占 2）。"""
    w = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1100 <= cp <= 0x115F  # Hangul Jamo
            or 0x2E80 <= cp <= 0xA4CF  # CJK Radicals .. Yi
            or 0xAC00 <= cp <= 0xD7A3  # Hangul Syllables
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            or 0xFF01 <= cp <= 0xFF60  # Fullwidth Forms
            or 0xFFE0 <= cp <= 0xFFE6  # Fullwidth Signs
            or 0x1F300 <= cp <= 0x1F64F  # Emoticons
            or cp == 0x200D  # ZWJ
        ):
            w += 2
        else:
            w += 1
    return w


def _pad_cjk(text: str, width: int) -> str:
    """按显示宽度用空格补齐到 width。超出则截断末尾加 …。"""
    dw = _display_width(text)
    if dw <= width:
        return text + " " * (width - dw)
    # 逐字截断，预留 1 个字符宽度给 "…"
    result = ""
    cur = 0
    limit = max(1, width - 1)
    for ch in text:
        cw = _display_width(ch)
        if cur + cw > limit:
            break
        result += ch
        cur += cw
    return result + "…" + " " * max(0, width - cur - 1)


class DeviceList:
    def __init__(self, devices: list[AudioDevice]) -> None:
        self._all = devices
        self.inputs = [d for d in devices if d.is_input]
        self.outputs = [d for d in devices if d.is_output]
        self.noisy_outputs = [d for d in self.outputs if d.kind == "output"]
        self._has_virtual = any(
            d.kind == "duplex" and "pulse" in d.name.lower() for d in devices
        )

    @property
    def recommended_input(self) -> AudioDevice | None:
        for d in self.inputs:
            if d.kind == "duplex" and any(
                x in d.name.lower() for x in ("pulse", "default")
            ):
                return d
        for d in self.inputs:
            if d.is_default_input:
                return d
        return self.inputs[0] if self.inputs else None

    @property
    def recommended_output(self) -> AudioDevice | None:
        for d in self.outputs:
            if d.kind == "duplex" and any(
                x in d.name.lower() for x in ("pulse", "default")
            ):
                return d
        for d in self.outputs:
            if d.is_default_output and d.kind != "hdmi":
                return d
        # 找到第一个非 HDMI 的输出
        for d in self.outputs:
            if d.kind != "hdmi":
                return d
        return None


class Report:
    def __init__(self, devices: DeviceList, verbose: bool = False) -> None:
        self._d = devices
        self._verbose = verbose

    def render(self) -> str:
        sec: list[str] = []
        sec.append(self._title("音频设备检测"))
        sec.append(self._input_block())
        sec.append(self._output_block())
        sec.append(self._advice())
        if self._verbose:
            sec.append(self._all_devices())
        return "\n".join(sec)

    def _title(self, text: str) -> str:
        return f"\n  {text}\n  {'=' * (len(text) * 2)}"

    def _input_block(self) -> str:
        header = f"\n  ▸ 麦克风（共 {len(self._d.inputs)} 个可用）\n"
        rows = [self._row(d, role="input") for d in self._d.inputs]
        return header + "".join(rows)

    def _output_block(self) -> str:
        shown = [d for d in self._d.outputs if d.kind != "output"]
        header = f"\n  ▸ 扬声器（共 {len(self._d.outputs)} 个可用，已隐藏 {len(self._d.noisy_outputs)} 个无标签接口）\n"
        rows = [self._row(d, role="output") for d in shown]
        return header + "".join(rows)

    def _row(self, d: AudioDevice, *, role: str) -> str:
        rec = (
            self._d.recommended_input if role == "input" else self._d.recommended_output
        )
        is_rec = rec is not None and d.index == rec.index

        mark = " ★" if is_rec else "  "
        warn = f"  ⚠ {d.warning}" if d.warning else ""
        star_label = d.star
        flag = f"  [{star_label}]" if star_label else ""

        name_col = _pad_cjk(d.short_name, 28)
        tag_col = _pad_cjk(d.tag, 14)
        detail_col = _pad_cjk(d.detail, 30)

        return f"    {d.index:>2}{mark}  {name_col}{tag_col}{detail_col}{flag}{warn}\n"

    def _advice(self) -> str:
        rec_in = self._d.recommended_input
        rec_out = self._d.recommended_output
        has_pulse = self._d._has_virtual

        lines: list[str] = [
            "",
            f"  {SEP}",
            "",
            "  怎么填配置",
            "",
            "    打开 configs/xlerobot.<env>.<os>.yaml，找到 channels.voice：",
            "",
            "      channels:",
            "        voice:",
            "          recorder:",
            "            input_device:  null      # 麦克风，null = 系统默认",
            "          tts:",
            "            output_device:  null      # 扬声器，null = 系统默认",
            "",
            "    绝大多数情况用 null 即可，不需要改。",
            "",
        ]
        if has_pulse:
            lines.append("    ✓ PulseAudio 已检测到，null 会自动路由到真实硬件。")
        else:
            lines.append(
                "    ⚠ 未检测到 PulseAudio。语音异常？sudo apt install pulseaudio"
            )
        if rec_in is rec_out and rec_in is not None:
            lines.append(
                f"    推荐设备: [{rec_in.index}] {rec_in.short_name}（全双工，输入输出皆可）"
            )
        else:
            if rec_in:
                lines.append(f"    推荐输入: [{rec_in.index}] {rec_in.short_name}")
            if rec_out:
                lines.append(f"    推荐输出: [{rec_out.index}] {rec_out.short_name}")
        lines.extend(
            [
                "",
                "    如果必须手动指定设备号，把上面 ★ 标记的序号填入。",
                "",
            ]
        )
        return "\n".join(lines)

    def _all_devices(self) -> str:
        lines: list[str] = [
            "",
            f"  {SEP}",
            "",
            "  全部设备（含不适合语音的）",
            "",
        ]
        rows = [
            f"    [{d.index:>2}]  {_pad_cjk(d.short_name, 28)}{_pad_cjk(d.tag, 14)}{d.detail}"
            for d in self._d._all
        ]
        lines.extend(rows)
        return "\n".join(lines) + "\n"


# ── 主入口 ──


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="列出本机音频设备并给出配置建议。")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示全部设备（含不适合语音的）"
    )
    args = parser.parse_args()

    devices = _query_devices()
    if not devices:
        logger.error("未检测到任何音频设备。")
        raise SystemExit(1)
    if devices[0]._error_message:
        logger.error("音频设备查询失败: %s", devices[0]._error_message)
        raise SystemExit(1)

    defaults = _query_defaults()
    for d in devices:
        object.__setattr__(d, "is_default_input", d.index == defaults["input"])
        object.__setattr__(d, "is_default_output", d.index == defaults["output"])

    if args.json:
        _dump_json(devices, defaults)
        return

    report = Report(DeviceList(devices), verbose=args.verbose)
    print(report.render())


def _query_devices() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError:
        return [_error_device("sounddevice 未安装。pip install sounddevice")]

    try:
        raw = sd.query_devices()
    except Exception as exc:
        return [_error_device(f"查询失败: {type(exc).__name__}: {exc}")]

    return [
        AudioDevice(
            index=i,
            name=str(item.get("name", "")),
            hostapi=int(item.get("hostapi", -1)),
            max_input_channels=int(item.get("max_input_channels") or 0),
            max_output_channels=int(item.get("max_output_channels") or 0),
            default_samplerate=float(item.get("default_samplerate") or 0),
        )
        for i, item in enumerate(raw)
    ]


def _query_defaults() -> dict[str, int | None]:
    try:
        import sounddevice as sd
    except ImportError:
        return {"input": None, "output": None}
    inp = sd.default.device[0]
    out = sd.default.device[1]
    return {
        "input": int(inp) if isinstance(inp, int) else None,
        "output": int(out) if isinstance(out, int) else None,
    }


def _error_device(message: str) -> list[AudioDevice]:
    return [
        AudioDevice(
            index=-1,
            name="_error",
            max_input_channels=0,
            max_output_channels=0,
            default_samplerate=0.0,
            _error_message=message,
        )
    ]


def _dump_json(devices: list[AudioDevice], defaults: dict[str, int | None]) -> None:
    import json

    print(
        json.dumps(
            {
                "devices": [
                    {
                        "index": d.index,
                        "name": d.name,
                        "kind": d.kind,
                        "max_input_channels": d.max_input_channels,
                        "max_output_channels": d.max_output_channels,
                        "default_samplerate": d.default_samplerate,
                        "is_default_input": d.is_default_input,
                        "is_default_output": d.is_default_output,
                    }
                    for d in devices
                    if d.index >= 0
                ],
                "defaults": defaults,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _configure_stdout() -> None:
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
