# 运行脚本索引

本页列出部署验证、硬件诊断、模型下载和开发维护脚本。

## 平台检查

验证本地平台和部署配置：

```powershell
uv run python scripts\ops\check_platform.py --config configs\xlerobot.real.windows.yaml
```

## XLeRobot 诊断

完整诊断流程：

```powershell
uv run python scripts\robots\xlerobot\diagnose.py --config configs\xlerobot.real.windows.yaml
```

扫描舵机：

```powershell
uv run python scripts\robots\xlerobot\scan_servos.py --config configs\xlerobot.real.windows.yaml
```

机械臂硬件检查：

```powershell
uv run python scripts\robots\xlerobot\check_arm.py --config configs\xlerobot.real.windows.yaml
```

相机扫描（默认会保存每个摄像头的视角截图到 outputs/diagnostic/cameras/）：

```powershell
uv run python scripts\robots\xlerobot\scan_cameras.py
```

### 配置多摄像头

当前默认使用单摄像头（`left_wrist`，device 1）。如果后续接入第二个摄像头，按以下步骤配置：

**1. 扫描确认设备号**

先运行 `scan_cameras.py`，打开 `outputs/diagnostic/cameras/` 下的截图，确认每个 device_id 对应哪个物理摄像头。

**2. 修改 `configs/xlerobot.real.windows.yaml`**

在 `robots.xlerobot.components.cameras` 下添加第二个摄像头，并更新 `default_camera`：

```yaml
default_camera: left_wrist
cameras:
  left_wrist:
    type: opencv_camera
    enabled: true
    owner: robot_driver
    device_id: 1
    backend: dshow
    width: 640
    height: 480
    fps: 30
  right_wrist:          # 新增第二个摄像头
    type: opencv_camera
    enabled: true
    owner: robot_driver
    device_id: 2
    backend: dshow
    width: 640
    height: 480
    fps: 30
```

**3. 更新 VLA 相机映射**

如果启用了 VLA capability service，需要同步更新 `capability_services.arm_vla.settings` 中的摄像头列表和 `camera_key_map`：

```yaml
cameras:
  - left_wrist
  - right_wrist
camera_devices:
  left_wrist: 1
  right_wrist: 2
camera_key_map:
  left_wrist: camera1
  right_wrist: camera2
```

`camera_key_map` 中的 key（如 `camera1`、`camera2`）必须和模型训练时的 observation key 保持一致。

**4. 验证**

重新运行诊断确认两个摄像头都正常：

```powershell
uv run python scripts\robots\xlerobot\diagnose.py --scan-cameras
```

诊断报告会列出每路摄像头的设备号、分辨率和归属，异常时附排查建议。

## 模型下载

下载本地语音模型，包含 `sherpa_onnx` ASR 和唤醒词模型：

```powershell
uv run python scripts\model_downloads\download_speech_models.py
```

只下载其中一类语音模型：

```powershell
uv run python scripts\model_downloads\download_speech_models.py --model asr
uv run python scripts\model_downloads\download_speech_models.py --model wakeup
```

下载视觉检测模型，默认下载 `yolo26n.pt`，用于 human follow / YOLO detector 相关能力：

```powershell
uv run python scripts\model_downloads\download_vision_models.py
```

列出可下载的视觉模型：

```powershell
uv run python scripts\model_downloads\download_vision_models.py --list
```

## 音频

列出音频设备：

```powershell
uv run python scripts\audio\list_devices.py
```

## 开发维护

清理生成文件：

```powershell
uv run python scripts\dev\clean.py pyc
uv run python scripts\dev\clean.py build
uv run python scripts\dev\clean.py test
uv run python scripts\dev\clean.py lint
```

也可以通过 Poe 执行：

```powershell
poe clean
```
