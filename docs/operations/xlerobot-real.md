# XLeRobot 真机部署

XLeRobot 是 Hey Robot 的组合式真实机器人 embodiment：

- **SO101**：六自由度机械臂 + 夹爪（Feetech 舵机 ID 1-6）
- **LeKiwi**：三轮全向移动底盘（Feetech 舵机 ID 7-9）
- **OpenCVCamera**：双路摄像头（头部 front + 腕部 wrist）
- **ServoBusBattery**：通过舵机总线读取电池电压

Agent 和 Skill 层不绑定具体机器人形态，上层发送 `SkillAction`，robot driver 判断目标 embodiment 是否能执行。

## 配置文件

| OS | 配置 |
|---|---|
| Windows | `configs/xlerobot.real.windows.yaml` |
| Ubuntu | `configs/xlerobot.real.ubuntu.yaml` |

## 平台差异

| 配置项 | Windows | Ubuntu |
|---|---|---|
| 串口 | `COM5` | `/dev/ttyUSB0` |
| 摄像头后端 | `dshow` | `v4l2` |
| 摄像头 device_id | `1` | `0`（头）、`1`（腕） |
| 音频设备 | 设备索引号 | `null`（PulseAudio 默认） |
| 路径分隔 | `\` | `/` |
| 命令前缀 | `uv run python scripts\...` | `uv run python scripts/...` |

## 环境变量

`.env` 文件中配置：

- `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` — Agent 推理
- `DASHSCOPE_API_KEY`、`DASHSCOPE_MODEL` — Vision / 场景理解
- `ARK_API_KEY` — 语音 TTS（Doubao）
- `FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_ENCRYPT_KEY`、`FEISHU_VERIFICATION_TOKEN` — 飞书通道

## 部署流程

### 1. 安装依赖

```bash
uv sync --dev
```

### 2. 下载模型

```bash
uv run python scripts/model_downloads/download_speech_models.py
uv run python scripts/model_downloads/download_vision_models.py
```

国内网络 GitHub 直连可能失败，脚本会自动走 ghproxy 镜像。也可通过 `GH_PROXY` 环境变量指定自定义镜像。

### 3. 音频设备检查

```bash
uv run python scripts/audio/list_devices.py
```

确认默认麦克风和扬声器可用。Ubuntu 配置已设 `input_device: null` / `output_device: null`，一般不需要修改。

### 4. 启动 NATS

`hey-robot run` 不自动启动 NATS broker：

```bash
nats-server
```

### 5. 摄像头扫描

连接机器人前先确认摄像头 device_id：

```bash
uv run python scripts/robots/xlerobot/scan_cameras.py
```

截图保存到 `outputs/diagnostic/cameras/`，打开确认：
- 哪个 `/dev/videoN` 是头部（front）
- 哪个 `/dev/videoN` 是腕部（wrist）

如果和配置不一致，修改 `cameras.front.device_id` 和 `cameras.wrist.device_id`。

### 6. 连接机器人，运行诊断

插上机器人 USB，确认串口出现后：

```bash
uv run python scripts/robots/xlerobot/diagnose.py
```

一键检查：串口总线 → 底盘舵机 → 机械臂舵机 → 摄像头 → 电池。

如果串口不是配置文件中的默认值，临时指定：

```bash
uv run python scripts/robots/xlerobot/diagnose.py --serial-port /dev/ttyACM0
```

单独检查子系统：

```bash
uv run python scripts/robots/xlerobot/scan_servos.py     # 扫描在线舵机 ID
uv run python scripts/robots/xlerobot/check_arm.py       # 检查机械臂关节角度
```

### 7. 验证配置

```bash
uv run hey-robot inspect --config configs/xlerobot.real.ubuntu.yaml
```

确认 services 列表、robot/agent/channel 配置、skills 清单符合预期。

### 8. 诊断后修正配置

| 配置项 | 根据诊断调整 |
|---|---|
| `serial_bus.port` | 按实际串口修改 |
| `cameras.front.device_id` | 按摄像头扫描结果 |
| `cameras.wrist.device_id` | 按摄像头扫描结果 |
| `base.*_id` | 按舵机扫描结果 |
| `arm.joint_ids.*` | 按舵机扫描结果 |

### 9. 启动系统

```bash
hey-robot run --config configs/xlerobot.real.ubuntu.yaml
```

> **Linux 用户注意**：串口需要 `dialout` 组权限。如果遇到 `Permission denied: '/dev/ttyACM0'`：
>
> **一次性生效**（不用登出）：
> ```bash
> sg dialout -c "hey-robot run --config configs/xlerobot.real.ubuntu.yaml"
> ```
>
> **永久修复**：
> ```bash
> sudo usermod -a -G dialout $USER
> newgrp dialout     # 当前终端立即生效，或重新登录
> ```

所有服务在单进程中启动：robot service → skill controller → task supervisor → agent → gateway。

## 建议验证顺序

先验证 11 个非 VLA skill 稳定，再单独调试 VLA：

1. 摄像头稳定发布 frame，心跳日志正常
2. `inspect_scene` 和 `look_around` 返回观测
3. `detect_marker` 在 marker 可见时返回检测结果
4. `stop_motion`、`move_base`、`turn_base` 可用
5. `set_arm_pose`、`move_arm_joints` 可用
6. `set_gripper` 可用
7. readiness gate 阻止不安全的动作
8. failure 进入 recovery flow
9. VLA 稳定后把 `vla_manipulation` 加入 `skills.enabled`

## 启用的 Skills（11 个）

默认 `mode: bringup`，启用 11 个非 VLA skill：

| 类别 | Skill | 说明 |
|---|---|---|
| 感知 | `inspect_scene` | 获取当前场景观察和摘要 |
| 感知 | `look_around` | 转动/扫描视野并观察 |
| 感知 | `detect_marker` | 检测可见 marker |
| 导航 | `move_base` | 底盘前进/后退 |
| 导航 | `turn_base` | 底盘左转/右转 |
| 导航 | `human_follow` | 基于视觉的人体跟随 |
| 安全 | `stop_motion` | 停止所有运动 |
| 安全 | `reset_posture` | 回到安全姿态 |
| 操作 | `set_arm_pose` | 设置机械臂命名姿态 |
| 操作 | `move_arm_joints` | 控制机械臂关节 |
| 操作 | `set_gripper` | 控制夹爪开合 |

`vla_manipulation` 已注册 capability service，默认不加入 `skills.enabled`。VLA 稳定后手动添加。

## 双摄像头配置

```yaml
cameras:
  front:        # 头部，1280x720
  wrist:        # 腕部，640x480
```

## 语音配置

- **ASR**：本地 sherpa-onnx（离线，模型在 `models/asr/`）
- **TTS**：云端 Doubao（火山引擎），需 `ARK_API_KEY`
- **唤醒词**：`小白`、`机器人`、`robot`

如需切换云端 ASR，将 `channels.voice.asr.provider` 从 `sherpa_onnx` 改为 `doubao`。

---

# VLA Capability Service

VLA 在系统中不属于 robot driver，而是独立的 `capability_service`。Agent 只请求 skill，是否走 VLA 由 `SkillControllerService` 和 `CapabilityRuntime` 决定。

## 链路

```text
Agent
  -> vla_manipulation
  -> SkillControllerService
  -> CapabilityRuntime
  -> VLACapabilityService
  -> LeRobot RobotClient
  -> LeRobot policy_server
  -> SO101 arm + cameras
```

## 先关闭 VLA 验证 native skills

```yaml
capability_services:
  arm_vla:
    enabled: false
```

关闭后普通 robot skills 不受影响。

## 开启 VLA

配置位置：`capability_services.arm_vla.settings`

Ubuntu 配置使用 `policy_runtime: groot_zmq`，通过 ZMQ 连接 policy server。Windows 配置使用 `policy_runtime: lerobot_single_arm`。

关键参数：

| 参数 | 说明 |
|---|---|
| `policy_host` / `policy_port` | Policy server 地址 |
| `task_prompt` | 任务提示词 |
| `arm` | 机械臂标识 |
| `cameras` | 摄像头列表，`camera_key_map` 对齐 model observation keys |
| `execution_time` | 单次执行时长 |
| `action_horizon` | 动作 chunk 长度 |

## 启动顺序

先启动 LeRobot policy server：

```bash
uv run python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

再启动 VLA capability service：

```bash
uv run hey-robot capability-service --config configs/xlerobot.real.ubuntu.yaml --service-id arm_vla
```

## 更换模型

通过配置切换，不需要改代码：

```yaml
policy_type: pi05
policy_name: Grigorij/pi05_collect_tissue_23_02
task_prompt: "Pick up tissue."
```

需确认：

- `policy_type` 和 checkpoint 类型匹配
- `camera_key_map` 的 key 与模型训练时 observation key 对齐
- `task_prompt` 与训练任务接近

## 常见问题

| 问题 | 排查 |
|---|---|
| 串口未识别 | `ls /dev/ttyUSB* /dev/ttyACM*`，检查 USB 连接和驱动 |
| 串口 Permission denied | `sudo usermod -a -G dialout $USER` 再 `newgrp dialout`（或重新登录） |
| 舵机无响应 | 跑 `scan_servos.py` 确认 ID，检查供电 |
| 摄像头打不开 | 检查 `device_id` 和 `backend`（v4l2/dshow），确认未被占用 |
| 飞书消息收不到 | 检查 `allow_from` 是否包含 `"*"` 或你的 open_id |
| 语音识别为空 | 检查麦克风，确认 `models/asr/` 四个模型文件完整 |
| VLA health.loaded=false | 检查 `policy_host/port`、`policy_type`、`task_prompt` |
| 动作不稳定 | 降低 `execution_time`，确认 calibration 和 camera key 对齐训练配置 |
