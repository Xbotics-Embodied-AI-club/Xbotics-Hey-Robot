# XLeRobot 真机部署

XLeRobot 是 Hey Robot 的组合式真实机器人 embodiment，当前硬件构成：

- `SO101`：六自由度机械臂 + 夹爪
- `LeKiwi`：三轮全向移动底盘
- `OpenCVCamera`：本地相机组件
- `ServoBusBattery`：通过 Feetech 舵机总线读取电池电压
- `XLeRobot`：上述组件的组合体

Agent 和 Skill 层不绑定具体机器人形态，上层发送 `SkillAction`，robot driver 判断目标 embodiment 是否能执行。

## 配置文件

```
configs/xlerobot.real.windows.yaml
```

## 平台要求

- Windows，串口示例 `COM5`
- 相机后端 `dshow`

## 部署步骤

### 1. 安装依赖

```powershell
uv sync --dev
```

如果启用本地语音唤醒和 `sherpa_onnx` ASR，先下载语音模型：

```powershell
uv run python scripts\model_downloads\download_speech_models.py
```

### 2. 启动 NATS

`hey-robot run` 不启动 NATS broker，需要先单独运行：

```powershell
nats-server
```

### 3. 验证硬件映射

启动前确认 `configs/xlerobot.real.windows.yaml` 中的配置：

- `robots.xlerobot.serial_bus.port`
- `robots.xlerobot.components.camera.device_id`
- `robots.xlerobot.components.camera.backend`
- 与实际接线不一致的 arm/base ID

然后检查环境：

```powershell
uv run python scripts\ops\check_platform.py --config configs\xlerobot.real.windows.yaml
uv run hey-robot inspect --config configs\xlerobot.real.windows.yaml
```

如果 `check_platform` 报错，先解决平台问题再启动 runtime。

### 4. 启动系统

```powershell
uv run hey-robot run --config configs\xlerobot.real.windows.yaml
```

这条命令在单进程中启动所有启用的服务：robot service、skill controller、task supervisor、agent、gateway，以及可选的 human-follow service。

## 环境变量

默认 agent provider 需要的变量：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`（例如 `deepseek-chat`）
- `DASHSCOPE_API_KEY`（DashScope 视觉模型）
- `DASHSCOPE_MODEL`（例如 `qwen-vl-plus`）

可选（启用语音时需要）：

- `ARK_API_KEY`

可选（启用飞书时需要）：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_ENCRYPT_KEY`
- `FEISHU_VERIFICATION_TOKEN`

## 运行时诊断

```powershell
uv run python scripts\robots\xlerobot\diagnose.py --config configs\xlerobot.real.windows.yaml
```

## 建议验证顺序

当前 `configs/xlerobot.real.windows.yaml` 默认不把 `vla_manipulation` 加入 `skills.enabled`。建议先验证 11 个非 VLA skill 稳定，再单独调试 VLA capability service。

1. camera 稳定发布 frame
2. `inspect_scene` 和 `look_around` 返回 observation
3. `detect_marker` 能在有 marker 时返回检测结果
4. `stop_motion`、`move_base`、`turn_base` 可用
5. `set_arm_pose`、`move_arm_joints` 可用
6. `set_gripper` 可用
7. readiness gate 阻止不安全或不可执行动作
8. failure 进入 recovery flow
9. VLA 稳定后再把 `vla_manipulation` 加入 `skills.enabled`

## 支持的 Robot Skills

当前 `configs/xlerobot.real.windows.yaml` 启用 11 个非 VLA skill：

- `inspect_scene`
- `look_around`
- `detect_marker`
- `move_base`
- `turn_base`
- `human_follow`
- `stop_motion`
- `reset_posture`
- `set_arm_pose`
- `move_arm_joints`
- `set_gripper`

`vla_manipulation` 已注册，并且可以配置独立 VLA capability service；但当前不加入 `skills.enabled`。只有显式加入后，Agent 才能请求它。

SO101 只支持 arm 和 observation skills，LeKiwi 只支持 base 和 observation skills。

## 测试

```powershell
uv run pytest tests\robots\test_xlerobot.py tests\robots\test_xlerobot_battery.py -q --no-cov
```

# VLA Capability Service

VLA 在系统中不属于 robot driver，而是独立的 `capability_service`。Agent 只请求 skill，是否走 VLA 由 `SkillControllerService` 和 `CapabilityRuntime` 决定。

## 当前链路

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

如果当前目标是先验证 IK、joint、named pose、gripper、camera、base 等 native skills：

```yaml
capability_services:
  arm_vla:
    enabled: false
```

关闭后普通 robot skills 不受影响。只有在把 `vla_manipulation` 加入 `skills.enabled` 后，Agent 才能请求 VLA；如果 capability service 不可用，请求会被拒绝。

## 开启 VLA

配置位置：`configs/xlerobot.real.windows.yaml`

```yaml
capability_services:
  arm_vla:
    type: vla_service
    enabled: true
    robot_id: xlerobot
    target: "127.0.0.1:9090"
    skill_names: [vla_manipulation]
    resources: [arm, gripper, camera]
    timeout_sec: 30

    runtime: lerobot_single_arm
    task_prompt: "Pick up tissue."
    policy_type: pi05
    policy_name: Grigorij/pi05_collect_tissue_23_02
    server_address: "127.0.0.1:8080"
    arm_port: "COM5"
    policy_device: "cuda"
    fps: 30
    actions_per_chunk: 50
    execution_time: 30
```

- `target`：capability service 的 gRPC 地址
- `server_address`：LeRobot policy server 地址

## 启动顺序

先启动 LeRobot policy server：

```powershell
uv run python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

再启动 VLA capability service：

```powershell
uv run hey-robot capability-service --config configs\xlerobot.real.windows.yaml --service-id arm_vla
```

## 更换模型

通过配置切换，不需要改代码：

```yaml
policy_type: pi05
policy_name: Grigorij/pi05_collect_tissue_23_02
task_prompt: "Pick up tissue."
```

可替换为其他 LeRobot 支持的 policy type（`smolvla`、`act` 等）。需确认：

- `policy_type` 和 checkpoint 类型匹配
- `camera_config` 的 camera names 与训练时 observation keys 匹配
- `task_prompt` 与训练任务接近
- `arm_port` 指向正确的 SO101 arm serial port
- `calibration_dir` 下有可用校准文件

## 相机配置

当前单摄像头配置（`left_wrist`，device 1）：

```yaml
camera_source: opencv
camera_config:
  camera1:
    index_or_path: 1
    width: 640
    height: 480
    fps: 30
```

如果需要接入第二个摄像头，先运行 `scan_cameras.py` 确认设备号，再添加 `camera2` 配置块。`camera1` / `camera2` 这种 key 名必须和模型训练时的 observation key 对齐。详细步骤见 [运行时脚本索引](runtime-scripts.md#配置多摄像头)。

## 常见问题

- `health.loaded=false`：通常缺少 `server_address`、`policy_name`、`policy_type`、`arm_port` 或 `task_prompt`
- `policy_server_unavailable`：LeRobot policy server 未启动或 `server_address` 不正确
- 相机打不开：检查 `camera_config.index_or_path`、Windows camera index 和 backend 占用
- 动作不稳定：先降低 `execution_time`，确认 calibration、camera key、task prompt 和模型训练条件一致
