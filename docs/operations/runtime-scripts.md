# 运行脚本索引

部署验证、硬件诊断、模型下载和维护脚本。

## 平台与配置

验证本地平台和部署配置：

```bash
uv run python scripts/ops/check_platform.py --config configs/xlerobot.real.ubuntu.yaml
```

检查配置有效性：

```bash
uv run hey-robot inspect --config configs/xlerobot.real.ubuntu.yaml
```

## XLeRobot 诊断

完整一键诊断：

```bash
uv run python scripts/robots/xlerobot/diagnose.py
```

带摄像头视角截图（推荐第一次配置时用）：

```bash
uv run python scripts/robots/xlerobot/diagnose.py --scan-cameras
```

指定串口：

```bash
uv run python scripts/robots/xlerobot/diagnose.py --serial-port /dev/ttyUSB0
```

### 子系统检查

扫描舵机：

```bash
uv run python scripts/robots/xlerobot/scan_servos.py
```

机械臂关节检查：

```bash
uv run python scripts/robots/xlerobot/check_arm.py
```

### 摄像头扫描

枚举所有摄像头并保存视角截图到 `outputs/diagnostic/cameras/`：

```bash
uv run python scripts/robots/xlerobot/scan_cameras.py
```

打开截图确认每个 device_id 对应哪个物理摄像头，然后更新配置文件中的 `cameras.<name>.device_id`。

## 模型下载

语音模型（sherpa-onnx，约 126 MB）：

```bash
uv run python scripts/model_downloads/download_speech_models.py
```

视觉模型（YOLO，用于 human follow）：

```bash
uv run python scripts/model_downloads/download_vision_models.py
```

强制重新下载：

```bash
uv run python scripts/model_downloads/download_speech_models.py --force
```

国内网络 GitHub 直连可能失败，脚本会自动走 ghproxy 镜像。可通过 `GH_PROXY` 环境变量指定自定义镜像。

## 音频

列出本机所有音频设备：

```bash
uv run python scripts/audio/list_devices.py
```

机器可读 JSON：

```bash
uv run python scripts/audio/list_devices.py --json
```

## 开发维护

```bash
uv run python scripts/dev/clean.py pyc     # 清理 .pyc
uv run python scripts/dev/clean.py build   # 清理构建产物
uv run python scripts/dev/clean.py test    # 清理测试产物
```

也可以通过 Poe 执行：

```bash
poe clean
```
