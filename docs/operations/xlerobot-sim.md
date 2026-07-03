# XLeRobot 仿真部署

在本地 MuJoCo 中运行 XLeRobot 仿真，验证仿真驱动、场景文件、相机观测和系统配置。

## 配置文件

| OS | 配置 |
|---|---|
| Windows | `configs/xlerobot.sim.windows.yaml` |
| Ubuntu | `configs/xlerobot.sim.ubuntu.yaml` |
| 实验 VLA + VLN | `configs/xlerobot.sim.vla_vln.yaml` |

Windows/Ubuntu 主配置只运行 11 个 native/sim skill，不包含 ModelService。只有实验配置
声明 `vln_nav` 和 `manipulate`。

## 平台差异

| 配置项 | Windows | Ubuntu |
|---|---|---|
| 音频设备 | 设备索引号 | `null`（系统默认） |
| ASR provider | `sherpa_onnx`（本地离线） | `doubao` |
| viewer.enabled | `false` | `false` |

## 系统执行架构

```
┌─────────────────────────────────────────────────────────┐
│  主进程 (.venv)    →  gateway / agent / Skill OS / robot  │
│  端口 8080 (web)   →  DeepSeek / DashScope               │
│  端口 4222 (nats)  →  NATS 消息总线                       │
│  端口 9090 (grpc)  →  VLA 服务（ACT 模型，进程内加载）     │
└──────────────────┬──────────────────────────────────────┘
                   │ gRPC :9091
┌──────────────────▼──────────────────────┐
│ VLN 服务                                 │
│ (.vln-venv)                             │
│ InternVLA-N1-System2                    │
│ GPU 1                                   │
└──────────────────────────────────────────┘
```

该图只适用于 `xlerobot.sim.vla_vln.yaml`。普通 sim 配置没有模型服务。VLA 服务现在直接在
主进程中加载 ACT 模型（不再需要独立 `.vla-venv`），通过 gRPC 端口 9090 暴露。
VLN 仍使用独立 venv（huggingface-hub 版本冲突），通过 gRPC 端口 9091 暴露。
Skill OS 调用模型服务并把输出转换为 Robot Runtime primitive；Agent 不直接加载模型。

## 依赖分组

`pyproject.toml` 使用 `[dependency-groups]` 将不同服务的依赖分开管理，
避免 huggingface-hub、transformers 等包的版本冲突：

| 命令 | 用途 |
|------|------|
| `uv sync` | 主运行时（Agent / Skill OS / Robot Runtime / VLA ACT policy） |
| `uv sync --group sim` | 主运行时 + MuJoCo 仿真 |
| `uv sync --group vln` | VLN 模型服务（InternVLA-N1 planner） |
| `uv sync --group dev` | 开发工具链（lint / type-check / test） |

VLA（ACT policy）的依赖（torch、PIL、safetensors、lerobot）已纳入主依赖组，
不再需要独立 `.vla-venv`。VLN 仍需独立 venv（huggingface-hub 版本冲突）。

### 创建主环境 (.venv)

```bash
uv sync --group sim --group dev
```

### 创建 VLN 环境 (.vln-venv)

```bash
uv sync --group vln
```

如果需要把环境安装到独立路径（而非当前 `.venv`），先创建 venv 再指定路径：

```bash
python3.12 -m venv .vln-venv
uv sync --group vln --python .vln-venv/bin/python
```

## 生成仿真模型

`assets/robots/xlerobot/xlerobot.xml` 是生成文件，不建议手工编辑。

```bash
python scripts/robots/xlerobot/generate_mjcf.py
```

## 快速验证

运行仿真测试：

```bash
pytest tests/robot_runtime/test_simulation.py -q --no-cov
```

## 启动步骤

### 普通仿真

普通仿真不需要 ModelService：

```bash
export MUJOCO_GL=egl
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

### 实验 VLN：1. 初始化依赖和模型

当前 checkout 必须先初始化 InternNav submodule，并准备配置所指向的模型目录：

```bash
git submodule update --init --recursive third_party/InternNav
test -d models/InternVLA-N1-System2
```

### 实验 VLN：2. 启动 VLN 服务

```bash
.vln-venv/bin/python -m hey_robot.cli.model_service \
  --config configs/xlerobot.sim.vla_vln.yaml \
  --service-id vln_nav
```

日志输出 `listening on grpc://127.0.0.1:9091` 只表示 gRPC server 已启动。还应通过
`GetHealth` 确认 `online=true`、`loaded=true`，不能依赖固定加载时间或显存数字判断。

### 实验 VLA：3. 启动 VLA 服务（可选）

VLA 服务直接在主进程中加载 ACT 模型，通过 gRPC 暴露：

```bash
.venv/bin/python -m hey_robot.cli.model_service \
  --config configs/xlerobot.sim.vla_vln.yaml \
  --service-id manipulate
```

日志输出 `listening on grpc://127.0.0.1:9090` 表示 gRPC server 已启动。模型通过
配置中的 `model_path` 自动延迟加载，无需单独启动 HTTP 推理服务器。
如果只验证 VLN，可以不启动这个服务。

### 实验：4. 启动主进程

headless 环境（无 X11）必须设置 EGL：

```bash
export MUJOCO_GL=egl

.venv/bin/python -m hey_robot.cli.main run \
  --config configs/xlerobot.sim.vla_vln.yaml
```

日志看到 `MuJoCo sim ready state=idle` + `Web channel started` 表示就绪。

### 5. 发送测试任务

```bash
curl -s http://localhost:8080/turn -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"走到桌子旁边","sender_id":"web-user","chat_id":"sim-dev-web"}'

# 响应: {"accepted":true,"trace_id":"tr_..."}
```

### 6. 停止

```bash
kill $(lsof -t -i:8080)   # 主进程（含 VLA 服务）
kill $(lsof -t -i:9091)   # VLN 服务
```

## VLN 配置

```yaml
model_services:
  vln_nav:
    type: vln_planner
    target: grpc://127.0.0.1:9091
    settings:
      backend: internvla_n1_system2
      mock_mode: false              # 内部测试开关；true 时跳过模型
      model_path: models/InternVLA-N1-System2
      device: cuda:1                # 使用 GPU 1，留 GPU 0 给其他任务
      attn_implementation: sdpa     # 用 PyTorch 内置 SDPA 代替 flash_attn
      internnav_repo: third_party/InternNav
      control_mode: planner_only    # 当前仅支持 planner_only
      camera: front
      image_width: 640
      image_height: 480
      resize_w: 384
      resize_h: 384
      num_history: 8
      max_new_tokens: 128
      hfov: 90
```

### 内部测试开关

| 值 | 行为 | 适用场景 |
|---|---|---|
| `true` | 返回测试替身结果（屏幕中心点/配置的 heading） | 调试 gRPC 管道、skill 调度 |
| `false` | 加载真实 InternVLA-N1 模型推理 | 实际导航验证 |

`mock_mode` 是现有配置字段名，仅用于内部测试，不表示项目对外提供独立的 Mock 机器人环境。

## 仿真配置项

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mjcf_path` | `assets/robots/xlerobot/scene.xml` | MuJoCo 场景文件 |
| `render_width` | `640` | 渲染宽度 |
| `render_height` | `480` | 渲染高度 |
| `control_hz` | `2.0` | 控制频率 |
| `linear_speed` | `0.2` | 默认线速度 (m/s) |
| `angular_speed` | `0.45` | 默认角速度 (rad/s) |
| `viewer.enabled` | `false` | 是否打开 MuJoCo 交互窗口 |

仿真摄像头（3 路，固定视角）：

| 摄像头 | 说明 |
|---|---|
| `front` | 前方视角 |
| `left_wrist` | 左腕视角 |
| `right_wrist` | 右腕视角 |

## 启用的 Skills

普通 Windows/Ubuntu 仿真配置启用 11 个非 VLA skill，和真机保持一致：

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

实验 `xlerobot.sim.vla_vln.yaml` 在上述能力之外声明：

| 类别 | Skill | 当前状态 |
|---|---|---|
| 导航 | `navigate_to` | 需要 `vln_nav` gRPC 服务 |
| 导航 | `approach_object` | 需要 `vln_nav` gRPC 服务 |
| 操作 | `manipulate` | 需要 `manipulate` gRPC 服务，ACT 模型直接加载 |
| 操作 | `human_follow` | 基于视觉的人体跟随（使用 YOLO 检测器） |

VLA 操作已合并为单一 `manipulate` skill，由 ACT policy 直接推理，模型在进程中加载
（不再需要独立 HTTP 推理服务器或 `.vla-venv`）。已通过 MuJoCo 仿真端到端验证：
Skill OS → gRPC → ACT 推理 → 解析原语 → 仿真执行，全部链路正常。

## 常见问题

### MuJoCo 启动报 OpenGL/GLFW 错误

headless 环境没有 X11 显示，需要改用 EGL 渲染：

```bash
export MUJOCO_GL=egl
```

确认 `libEGL.so` 已安装：`ldconfig -p | grep libEGL`

### VLN 报 huggingface-hub 版本冲突

说明 VLN 服务用了主环境。VLN 需要 `huggingface-hub==0.33.4`，主运行时不依赖
huggingface-hub。必须用 `.vln-venv` 启动 VLN 服务。详见上方「依赖分组」。

### flash_attn 编译/加载失败

配置已设 `attn_implementation: sdpa`，InternNav executor 会自动将 `flash_attention_2` patch 为 PyTorch 内置 SDPA（要求 torch >= 2.0）。不需要 `pip install flash-attn`。

### 看不到 LeKiwi 底盘

重新生成模型：

```bash
python scripts/robots/xlerobot/generate_mjcf.py
```

生成后的 MJCF 应包含 `lekiwi_chassis_visual`、`base_plate`、`Omni-Directional-Wheel` 等几何体。

### 中间出现黑色实体块

通常是 collision box 被渲染出来。不要手工修改 `xlerobot.xml`，应修改 `scripts/robots/xlerobot/generate_mjcf.py` 后重新生成。

### 修改 `xlerobot.xml` 后被覆盖

这是预期行为。请修改生成器脚本后重新运行。

### MuJoCo viewer 窗口不显示

Windows 和 Ubuntu 当前配置都设置 `viewer.enabled: false`。需要交互窗口时显式改为
`true`，并确保存在图形环境（X11/Wayland/Windows desktop）；headless 环境保持关闭并
使用 EGL。

### 麦克风/语音不工作

Ubuntu 使用系统默认音频设备（`input_device: null`），Windows 根据
`scripts/audio/list_devices.py` 的输出调整设备索引。注意当前 Ubuntu 仿真配置使用云端
Doubao ASR，需要有效的 `ARK_API_KEY`；Windows 仿真配置使用本地 sherpa-onnx。
### gRPC 请求超时 (DEADLINE_EXCEEDED)

配置中 `target: grpc://127.0.0.1:9091` 的 `grpc://` 前缀是项目内部格式，`GrpcModelServiceClient` 会自动去掉。如果直接使用 gRPC 客户端工具测试，目标地址应为 `127.0.0.1:9091`（不带 scheme）。
