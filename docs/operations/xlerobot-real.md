# XLeRobot 真机部署

XLeRobot 是 Hey Robot 的组合式真实机器人 embodiment：

- **SO101**：六自由度机械臂 + 夹爪（Feetech 舵机 ID 1-6）
- **LeKiwi**：三轮全向移动底盘（Feetech 舵机 ID 7-9）
- **OpenCVCamera**：Ubuntu/S600 默认双路（front + wrist），Windows 默认一路 front
- **ServoBusBattery**：通过舵机总线读取电池电压

Agent 和 Skill 层不绑定具体机器人形态。Agent 提交 `SkillIntent`，Skill OS 将实际
primitive 编码为 `RobotAction`，robot driver 最后判断目标 embodiment 是否能执行。

## 配置文件

| OS | 配置 |
|---|---|
| Windows | `configs/xlerobot.real.windows.yaml` |
| Ubuntu | `configs/xlerobot.real.ubuntu.yaml` |
| RDK S600 | `configs/xlerobot.real.s600.yaml` |

## 平台差异

| 配置项 | Windows | Ubuntu |
|---|---|---|
| 串口 | `COM5` | `/dev/ttyACM0` |
| 摄像头后端 | `dshow` | `v4l2` |
| 摄像头 device_id | `1`（front） | `4`（front）、`2`（wrist） |
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

如果和配置不一致，修改 `components.cameras.front.device_id`，以及存在时的
`components.cameras.wrist.device_id`。

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
| `components.cameras.front.device_id` | 按摄像头扫描结果 |
| `components.cameras.wrist.device_id` | 按摄像头扫描结果；Windows 默认没有该项 |
| `components.base.*_id` | 按舵机扫描结果 |
| `components.arm.joint_ids.*` | 按舵机扫描结果 |

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

主服务在同一 asyncio 进程中并发启动：robot service、skill controller、task
supervisor、agent 和 gateway。它们各自连接 NATS，不要依赖列表顺序作为 readiness
顺序；应观察各服务的 ready/health 日志。

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
9. 只有完成独立 ModelService、observation 和真机安全验证后，才设计 VLA deployment

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

`vla_manipulation` Skill 已注册，但当前三份真机配置的 `model_services` 都为空，也没有把
VLA 加入 `skills.enabled`。因此默认真机系统不具备可启动的 VLA 服务。

## 摄像头配置

Ubuntu/S600 默认双摄像头：

```yaml
components:
  cameras:
    front:        # 头部，1280x720
    wrist:        # 腕部，640x480
```

Windows 默认只配置 `front`。新增 wrist 前先扫描设备，并确认相机不会被其他进程占用。

## 语音配置

- **ASR**：Windows 为 Doubao，Ubuntu 为本地 sherpa-onnx，S600 为 OpenAI-compatible
- **TTS**：云端 Doubao（火山引擎），需 `ARK_API_KEY`
- **唤醒词**：`小白`、`机器人`、`robot`

具体以所选 YAML 的 `channels.voice.asr.provider` 为准。使用 sherpa-onnx 时模型位于
`models/asr/`；使用云端 provider 时确认对应 API key 和 endpoint。

---

# VLA ModelService 状态

VLA 的架构位置已经确定，但当前真机 deployment 尚未交付：

```text
Agent request_skill
  -> SkillControllerService
  -> ModelServiceRegistry
  -> gRPC VLAPolicyService
  -> one-step policy result
  -> Skill OS converts result to guarded primitives
  -> RobotRuntime / XLeRobotDriver
```

当前代码和配置的事实边界：

- 真机 YAML 中没有 ModelService entry，不能直接运行 `--service-id arm_vla`；
- 默认真机 Skill surface 只有前述 11 个 native skill；
- VLA executor 的内部测试路径只能用于协议测试；
- real path 仍依赖 LeRobot RobotClient 的 arm/camera config，尚未完全成为只消费系统注入
  observation 的纯推理服务；
- `pick_object` / `place_object` 还需要对齐 Skill 调用名与 ModelService `provides`。

启用真机 VLA 前至少需要：

1. 新建独立 deployment profile，不直接修改已验证的 native profile；
2. 配置 gRPC target、policy server、checkpoint、arm 和 camera mapping；
3. 验证 `GetHealth` 的 online/loaded/busy；
4. 添加 observation -> inference -> primitive -> RobotStatus 的端到端测试；
5. 在仿真和空载机械臂上验证动作范围、取消和超时；
6. 最后才把 semantic VLA skill 加入 `skills.enabled`。

在这些条件完成前，不应把当前真机配置描述为支持 VLA 抓取。

## 常见问题

| 问题 | 排查 |
|---|---|
| 串口未识别 | `ls /dev/ttyUSB* /dev/ttyACM*`，检查 USB 连接和驱动 |
| 串口 Permission denied | `sudo usermod -a -G dialout $USER` 再 `newgrp dialout`（或重新登录） |
| 舵机无响应 | 跑 `scan_servos.py` 确认 ID，检查供电 |
| 摄像头打不开 | 检查 `device_id` 和 `backend`（v4l2/dshow），确认未被占用 |
| 飞书消息收不到 | 检查 `allow_from` 是否包含 `"*"` 或你的 open_id |
| 语音识别为空 | 检查麦克风，确认 `models/asr/` 四个模型文件完整 |
| VLA service 不存在 | 当前真机配置未声明 ModelService；使用单独实验 profile |
| VLA health.loaded=false | 检查 gRPC service 配置、policy server 和 checkpoint |
| VLA 动作不稳定 | 停止真机测试，先回到仿真和空载校准 |
