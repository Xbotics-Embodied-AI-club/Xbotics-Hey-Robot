"""列出本机所有音频输入/输出设备。

用途：
  - 配置 voice channel 时不知道麦克风/扬声器该填哪个 index，跑这个工具列出所有设备。
  - 找出可用的输入设备（麦克风）和输出设备（扬声器/耳机）。
  - 看每个设备的通道数和默认采样率。

常见用法：
  # 列出所有音频设备
  uv run python scripts/audio/list_devices.py

  # 机器可读 JSON 输出
  uv run python scripts/audio/list_devices.py --json

输出说明：
  - 每行一个设备：序号 / 名称 / 角色（输入/输出）/ 输入通道数 / 输出通道数 / 默认采样率
  - 拿到序号后填到 configs/xlerobot.real.windows.yaml 的 channels.voice.recorder.input_device
    或 channels.voice.tts.output_device 字段。

退出码：
  - 0：成功列出（即使为空也算成功）
  - 非 0：sounddevice 未安装或查询失败

更多选项：uv run python scripts/audio/list_devices.py --help
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="列出本机所有音频输入/输出设备，便于配置 voice channel。"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()
    devices = list_devices()
    if args.json:
        import json

        print(json.dumps({"devices": devices}, ensure_ascii=False, indent=2))
    else:
        print(format_devices(devices))


def list_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        return [{"ok": False, "error": f"{type(exc).__name__}: {exc}"}]
    results = []
    for index, item in enumerate(sd.query_devices()):
        max_input = int(item.get("max_input_channels") or 0)
        max_output = int(item.get("max_output_channels") or 0)
        results.append(
            {
                "ok": True,
                "index": index,
                "name": str(item.get("name", "")),
                "hostapi": int(item.get("hostapi", -1)),
                "max_input_channels": max_input,
                "max_output_channels": max_output,
                "default_samplerate": float(item.get("default_samplerate") or 0),
                "input": max_input > 0,
                "output": max_output > 0,
            }
        )
    return results


def format_devices(devices: list[dict]) -> str:
    if devices and devices[0].get("ok") is False:
        return f"音频设备列举失败：{devices[0]['error']}"
    lines = ["音频设备列表："]
    for item in devices:
        role = []
        if item["input"]:
            role.append("输入")
        if item["output"]:
            role.append("输出")
        lines.append(
            f"  - 序号 {item['index']:2d}：{item['name']} "
            f"角色={'+'.join(role) or '无'} "
            f"输入通道={item['max_input_channels']} 输出通道={item['max_output_channels']} "
            f"采样率={item['default_samplerate']:.0f}Hz"
        )
    return "\n".join(lines)


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        return


if __name__ == "__main__":
    main()
