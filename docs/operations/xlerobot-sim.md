# XLeRobot 仿真部署

在本地 MuJoCo 中运行 XLeRobot 仿真，验证仿真驱动、场景文件、相机观测和系统配置。

## 配置文件

| OS | 配置 |
|---|---|
| Windows | `configs/xlerobot.sim.windows.yaml` |
| Ubuntu | `configs/xlerobot.sim.ubuntu.yaml` |
| 实验 VLA + VLN | `configs/xlerobot.sim.vla_vln.yaml` |

Windows/Ubuntu 主配置只运行 11 个 native/sim skill，不包含 ModelService。只有实验配置
声明 `vln_nav` 和 `vla_manipulation`。

## 平台差异

| 配置项 | Windows | Ubuntu |
|---|---|---|
| 音频设备 | 设备索引号 | `null`（系统默认） |
| ASR provider | `sherpa_onnx`（本地离线） | `doubao` |
| viewer.enabled | `false` | `false` |

## 系统执行架构

```
┌─────────────────────────────────────────────┐
│  主进程 (.venv)    →  gateway / agent / Skill OS / robot │
│  端口 8080 (web)   →  DeepSeek / DashScope    │
│  端口 4222 (nats)  →  NATS 消息总线            │
└──────────┬──────────────────────────────────┘
           │ gRPC (127.0.0.1:9091)
┌──────────▼──────────────────────────────────┐
│  VLN 服务 (.vln-venv)  →  InternVLA-N1       │
│  GPU 1              →  真实模型推理            │
└─────────────────────────────────────────────┘
```

该图只适用于 `xlerobot.sim.vla_vln.yaml`。普通 sim 配置没有 VLN 服务。实验配置中
VLN 以独立 gRPC ModelService 运行，Skill OS 调用模型并把输出转换为 Robot Runtime
primitive；Agent 不直接加载模型。

## 双环境设置

只有启用实验 VLN 时才需要双环境。主进程和 VLN 服务需要**独立的 Python 虚拟环境**，
原因是 InternNav 依赖 `huggingface-hub>=0.30.0,<1.0`，而主项目声明
`huggingface-hub>=1.0.0,<2.0.0`，无法在同一环境共存。

| 环境 | huggingface-hub | 其他依赖 | 用途 |
|------|:---:|---|------|
| `.venv/` | `>=1.0,<2.0` | 由项目 lockfile 管理 | 主进程：Agent / Skill OS / Robot Runtime |
| `.vln-venv/` | `>=0.30,<1.0` | 按 InternNav 要求安装 | VLN 服务：InternVLA-N1 模型推理 |

### 创建主环境 (.venv)

```bash
uv sync --dev --extra sim
```

### 创建 VLN 环境 (.vln-venv)

```bash
# 1. 创建独立 venv（用系统 Python 3.12）
python3.12 -m venv .vln-venv
source .vln-venv/bin/activate

# 2. 安装 huggingface-hub（必须在 0.30~1.0 之间）
pip install "huggingface-hub>=0.30.0,<1.0"

# 3. 安装 InternNav 及其依赖
cd third_party/InternNav
pip install -e .
cd ../..

# 4. 安装本项目代码，但不要解析主项目的冲突依赖
pip install --no-deps -e .
pip install grpcio protobuf pyyaml python-dotenv imageio pillow numpy

# 5. 验证版本
python -c "
import huggingface_hub
v = huggingface_hub.__version__
assert v < '1.0', f'huggingface-hub 版本 {v} 冲突，需要 <1.0'
print(f'✓ huggingface-hub {v} OK')
"
```

## 依赖

```bash
uv sync --extra sim
```

如果只需要补 MuJoCo：

```bash
uv pip install "mujoco>=3.3.0"
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

### 实验 VLA：3. 可选启动接口测试服务

实验配置同时声明了 VLA service。因为 `model_path` 为空，当前只会启用内部测试路径，用于
验证 gRPC、Skill OS 和 primitive 链路：

```bash
.venv/bin/python -m hey_robot.cli.model_service \
  --config configs/xlerobot.sim.vla_vln.yaml \
  --service-id vla_manipulation
```

如果只验证 VLN，可以不启动这个服务，但不要提交 VLA Skill。

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
kill $(lsof -t -i:8080)   # 主进程
kill $(lsof -t -i:9091)   # VLN 服务
kill $(lsof -t -i:9090)   # 可选 VLA 服务
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
| 操作 | `vla_manipulation` | 当前空 `model_path` 只启用接口测试路径 |
| 操作 | `pick_object` | 实验；当前 service `provides` 与系统调用名尚未对齐 |
| 操作 | `place_object` | 实验；当前 service `provides` 与系统调用名尚未对齐 |

因此该实验配置适合验证分层和消息链路，不应直接作为真实 VLA 抓取成功的证明。真实 VLA
还需要完成 policy checkpoint、observation 输入和 service routing 的端到端验证。

## 常见问题

### MuJoCo 启动报 OpenGL/GLFW 错误

headless 环境没有 X11 显示，需要改用 EGL 渲染：

```bash
export MUJOCO_GL=egl
```

确认 `libEGL.so` 已安装：`ldconfig -p | grep libEGL`

### VLN 报 huggingface-hub 版本冲突

```
huggingface-hub>=0.30.0,<1.0 is required, but found huggingface-hub==1.x
```

说明 VLN 服务用了主环境。必须用 `.vln-venv/bin/python` 启动 VLN 服务。详见上方「双环境设置」。

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
