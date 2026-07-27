# XLeRobot 仿真部署

在本地 MuJoCo 中运行 XLeRobot 仿真，验证仿真驱动、场景文件、相机观测和系统配置。

## 配置文件

| OS | 配置 |
|---|---|
| Windows | `configs/xlerobot.sim.windows.yaml` |
| Ubuntu | `configs/xlerobot.sim.ubuntu.yaml` |
| 实验 VLN | `configs/xlerobot.sim.vln.yaml` |

Windows/Ubuntu主配置只向Agent开放`inspect_scene`、`move_base`和`turn_base`，不包含
ModelService。实验 VLN 配置额外开放`navigate_to`和`approach_object`，由独立
InternNav ModelService 提供导航能力。

## 平台差异

| 配置项 | Windows | Ubuntu |
|---|---|---|
| 音频设备 | 设备索引号 | `null`（系统默认） |
| ASR provider | `sherpa_onnx`（本地离线） | `doubao` |
| viewer.enabled | `true` | `true` |

## 系统执行架构

```
┌─────────────────────────────────────────────────────┐
│  主进程 (.venv)  →  gateway / agent / skills / robot   │
│  端口 8080 (web)  →  DeepSeek / DashScope             │
│  端口 4222 (nats) →  NATS 消息总线                     │
└───────────────────────┬─────────────────────────────┘
                        │ gRPC :9091
                        ▼
              ┌─────────────────────────┐
              │ VLN 导航服务              │
              │ InternVLA-N1-DualVLN   │
              │ (transformers 4.x)      │
              └─────────────────────────┘
```

该图只适用于 `xlerobot.sim.vln.yaml`。普通 sim 配置没有模型服务。
VLN 是独立的 gRPC 模型服务，主进程通过 skill 调用它完成导航
（navigate_to / approach_object）。

## 依赖分组

`pyproject.toml` 使用运行时 extras 和开发 dependency groups 分离可部署服务，
避免 Gateway/Agent 携带 Robot、语音或 GPU 依赖，也避免
huggingface-hub、transformers 等包的版本冲突：

| 命令 | 用途 |
|------|------|
| `uv sync --extra gateway --extra agent --extra robot --group sim --group dev` | Web + Agent + Robot + MuJoCo 开发环境 |
| `uv sync --extra model-service --group lerobot-policy` | 通用 LeRobot Policy gRPC 服务 |
| `uv sync --extra model-service --group vln` | VLN 模型服务——必须使用独立 venv |
| `uv sync --extra voice` | 可选语音通道 |
| `uv sync --extra human-follow` | 可选 Torch/Ultralytics 人体跟随 |

**版本约束（Python 3.12）：**

| 包 | 主环境 (.venv) | VLN 环境 (.venv-vln) | 冲突原因 |
|---|---|---|---|
| lerobot | >= 0.6.0 | — | 0.4.x/0.5.x dataclass bug 在 3.12 上无法 import |
| transformers | >= 5.4.0, < 5.6.0 | == 4.51.0 | lerobot 0.6.0 要求 5.x，InternNav 要求 4.x |
| huggingface-hub | >= 1.0.0 | == 0.33.4 | lerobot >= 0.5 要求 >= 1.0，transformers 4.x 要求 < 1.0 |

> **关键约束**：lerobot 0.6.0 要求 `transformers >= 5.4.0, < 5.6.0`。
> 不要使用 transformers 5.13.0+（`create_causal_mask` API 不兼容）。

LeRobot Policy 使用独立的 `lerobot-policy` 依赖组和
`hey-robot model-service` 进程，不在 `hey-robot run` 主进程中加载。
该组包含当前实际使用的 SmolVLA、PI0/PI0.5、FastWAM 推理依赖；ACT 使用
LeRobot 核心依赖。以后增加新的 Policy family 时，应把它需要的 LeRobot extra
或等价的显式依赖加入该组，而不是装进 Gateway、Agent 或 Robot。
VLN 使用独立 venv（transformers / huggingface-hub 版本冲突）。

## 环境创建

### 主环境（.venv）：运行时 + 仿真

```bash
uv sync --extra gateway --extra agent --extra robot --group sim --group dev
```

之后主进程命令（`hey-robot run` 和测试脚本）使用 `.venv/bin/python`。

### VLN 环境（.venv-vln）：导航模型服务

由于 VLN（InternVLA-N1）依赖 `transformers==4.51.0` + `huggingface-hub==0.33.4`，
与主环境的 `transformers>=5.4` + `huggingface-hub>=1.0` 不兼容，必须创建独立 venv：

```bash
scripts/dev/setup_vln_env.sh
```

### LeRobot Policy 环境（.policy-venv，可选）

如需在容器外测试 LeRobot 策略，可创建独立 Policy 环境：

```bash
python3.12 -m venv .policy-venv
uv sync --extra model-service --group lerobot-policy \
  --python .policy-venv/bin/python
```

日常使用不需要此环境；Compose 会使用独立的 Policy 镜像。

## 生成仿真模型

`assets/robots/xlerobot/xlerobot.xml` 是生成文件，不建议手工编辑。

```bash
uv run python scripts/robots/xlerobot/generate_mjcf.py
```

## 场景布局俯视图

当前 XLeRobot home 仿真配置加载 `assets/scenes/home_scene.xml`，该文件组合了
`home_environment.xml` 中的 20m x 14m 家庭环境和 XLeRobot 本体。下图是从当前
MJCF 渲染得到的正俯视图，用于人工检查房间、地毯、家具和机器人出生点的相对位置。

![XLeRobot home scene top-down layout](../images/xlerobot-home-scene-topdown.png)

布局维护时注意：

- 家具 mesh 的原点不一定在中心，移动物体后应以编译后的 MuJoCo 世界边界为准。
- 地毯应保持在对应房间边界内；大厅地毯中心应接近 `(10, 5)`。
- 餐厅四把椅子应朝向餐桌，椅背朝外。
- 可见资产不应低于地面；相关回归测试在 `tests/assets/test_xlerobot_home_scene.py`。

## 快速验证

运行仿真测试：

```bash
uv run pytest tests/robot_backends/simulation/test_xlerobot_sim.py -q --no-cov
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
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  .venv-vln/bin/huggingface-cli download \
  InternRobotics/InternVLA-N1-DualVLN \
  --local-dir models/InternVLA-N1-DualVLN
test -f models/InternVLA-N1-DualVLN/model.safetensors.index.json
```

### 实验 VLN：2. 启动 VLN 服务

```bash
.venv-vln/bin/python -m hey_robot.cli.model_service \
  --config configs/xlerobot.sim.vln.yaml \
  --service-id vln_nav
```

日志输出 `listening on grpc://127.0.0.1:9091` 只表示 gRPC server 已启动。还应通过
`GetHealth` 确认 `online=true`、`loaded=true`，不能依赖固定加载时间或显存数字判断。

### 实验 VLN：3. 启动主进程

headless 环境（无 X11）必须设置 EGL：

```bash
export MUJOCO_GL=egl

.venv/bin/python -m hey_robot.cli.main run \
  --config configs/xlerobot.sim.vln.yaml
```

日志看到 `MuJoCo sim ready state=idle` + `Web channel started` 表示就绪。

### 4. 发送测试任务

```bash
curl -s http://localhost:8080/turn -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"走到桌子旁边","sender_id":"web-user","chat_id":"sim-dev-web"}'

# 响应: {"accepted":true,"trace_id":"tr_..."}
```

### 5. 停止

在两个启动终端中分别按 `Ctrl+C`，让主进程和 VLN 服务执行各自的清理流程。

## VLN 配置

```yaml
model_services:
  vln_nav:
    type: vln_planner
    target: grpc://127.0.0.1:9091
    settings:
      backend: internvla_n1_dualvln
      mock_mode: false              # 内部测试开关；true 时跳过模型
      model_path: models/InternVLA-N1-DualVLN
      device: cuda                  # 物理 GPU 由 CUDA_VISIBLE_DEVICES / 容器映射选择
      attn_implementation: sdpa     # 用 PyTorch 内置 SDPA 代替 flash_attn
      internnav_repo: third_party/InternNav
      control_mode: base_action_chunk
      base_linear_speed: 0.25
      base_angular_speed: 0.30
      max_action_chunk_steps: 4
      system1_replans_per_waypoint: 4
      camera: front
      image_width: 640
      image_height: 480
      resize_w: 384
      resize_h: 384
      discrete_turn_deg: 15       # InternNav 离散转向 token 的官方步长
      discrete_forward_cm: 25     # InternNav 离散前进 token 的官方步长
      num_history: 8
      max_new_tokens: 128
      hfov: 90
```

### 内部测试开关

| 值 | 行为 | 适用场景 |
|---|---|---|
| `true` | 返回测试替身结果（屏幕中心点/配置的 heading） | 调试 gRPC 管道、skill 调度 |
| `false` | 加载真实 InternVLA-N1 DualVLN 模型推理 | 实际导航验证 |

`mock_mode` 是现有配置字段名，仅用于内部测试，不表示项目对外提供独立的 Mock 机器人环境。

## 仿真配置项

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mjcf_path` | `assets/scenes/home_scene.xml` | MuJoCo 场景文件 |
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

## Agent 可见 Skills

普通Windows/Ubuntu仿真配置只开放：

| 类别 | Skill | 说明 |
|---|---|---|
| 感知 | `inspect_scene` | 获取当前场景观察和摘要 |
| 导航 | `move_base` | 底盘前进/后退 |
| 导航 | `turn_base` | 底盘左转/右转 |

实验配置中的ModelService声明可支持：

| 类别 | Skill | 当前状态 |
|---|---|---|
| 导航 | `navigate_to` | 需要 `vln_nav` gRPC 服务，输出受限底盘控制周期 |
| 导航 | `approach_object` | 需要 `vln_nav` gRPC 服务，输出受限底盘控制周期 |

实验 VLN 配置已将这两个 Skill 暴露给 Agent。`base_action_chunk` 模式保留
DualVLN 的原生双系统边界：System 2 输出转向、像素 waypoint 或 STOP；System 1
消费 latent waypoint 和连续 RGB 观测，生成最多4步局部轨迹动作。ModelService 将原生
动作按 `15°` / `25 cm` 语义校准为 `base_velocity_chunk`，Robot Runtime 逐个执行其中的
`base_velocity_step`，每步之间获取 fresh observation，整个 chunk 完成后再规划。一个
`navigate_to` run 内保持同一 policy session 和 waypoint latent；模型不直接访问串口或
仿真驱动。纯 `InternVLA-N1-System2` checkpoint 缺少 System 1 权重，配置会在加载阶段
明确失败，不能以固定前进或随机 latent 代替。启用时必须同时验证 ModelService health、capability name、observation
mapping、fresh frame 和 budget termination。仓库测试覆盖接口链路，不代表指定
checkpoint 已经完成真实任务效果验证。

## Docker 状态

仓库的基础 Compose 拓扑把应用拆为 `frontend`、`gateway`、`agent`、`robot`
四个容器，并使用独立的 NATS 容器通信。`robot` 同时承载本机 RobotService 与
SkillWorker；`agent` 通过 NATS 提交 SkillCommand，不再依赖单体 `runtime` 进程。
Policy、VLN 与 RoboCasa365 是可选 profile，仍依赖模型、GPU 和外部凭据。

### 服务架构（Docker）

```
浏览器 :8080
      │
      ▼
 frontend (Nginx) ──HTTP/WebSocket──▶ gateway
                                      │
                                      ▼
                                  NATS :4222
                                   │       │
                                   ▼       ▼
                                agent    robot + SkillWorker
                                             │
                                             ▼
                                         VLN :9091
                                          (可选)
```

Frontend 只提供静态资源和反向代理；Gateway 只处理 HTTP/WebSocket 与消息路由；
Agent 负责推理和任务状态；Robot 负责设备、安全边界和 Skill 执行。
Compose 中的 `base` 仅用于构建上下文，副本数固定为 0，不会成为第六个运行容器。

### 构建镜像

```bash
# 先构建公共 core，再构建三个独立服务镜像
docker build -f docker/Dockerfile.base -t hey-robot-base:latest .
docker build -f docker/Dockerfile.gateway -t hey-robot-gateway:latest .
docker build -f docker/Dockerfile.agent -t hey-robot-agent:latest .
docker build -f docker/Dockerfile.robot -t hey-robot-robot:latest .

# 轻量前端镜像
docker build -f docker/Dockerfile.frontend -t hey-robot-frontend:latest .

# 通用 LeRobot Policy 服务镜像（策略类型由 checkpoint 决定）
docker build -f docker/Dockerfile.policy -t hey-robot-policy:latest .

# VLN 导航服务镜像（InternVLA-N1，GPU 1）
docker build -f docker/Dockerfile.vln -t hey-robot-vln:latest .
```

### 启动

启动基础拓扑：

```bash
docker-compose up -d nats robot agent gateway frontend
docker-compose ps
curl --fail http://127.0.0.1:8080/health
```

单 GPU 机器应分别启动 VLN 或 RoboCasa，避免模型争用同一张显卡：

```bash
docker-compose --profile vln up -d vln
docker-compose --profile robocasa up -d robocasa365 robocasa-policy
```

当前不提供独立的通用 Policy Compose 服务；需要 LeRobot Policy 的 RoboCasa
评测通过 `robocasa-policy` 启动。`gpu` profile 当前只启动 VLN。
`robocasa` profile 会启动环境容器 `robocasa365` 和策略容器
`robocasa-policy`；后者使用 `hey-robot-policy` 镜像，分别监听 9092 和
9091。VLN 与 RoboCasa Policy 默认占用相同宿主机端口 9091，不能同时启动，
除非调整 `HEY_ROBOT_ROBOCASA_POLICY_PORT`。RoboCasa365 启动前必须设置
`ROBOCASA_EVALUATOR_TOKEN` 和 `ROBOCASA_DATA_TOKEN`。安装了 Docker Compose
CLI plugin 的机器也可以把 `docker-compose` 替换为 `docker compose`。

GPU 服务显式使用 `runtime: nvidia` 和 `NVIDIA_VISIBLE_DEVICES`。这既兼容标准
NVIDIA Container Toolkit，也避免 Snap Docker 29 把 Compose device reservation
错误解析到宿主 `/var/run/cdi`；Snap Docker 会改用自己生成的 hostfs CDI 路径。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HEY_ROBOT_NATS_PORT` | 4222 | NATS 客户端端口 |
| `HEY_ROBOT_NATS_MONITOR_PORT` | 8222 | NATS 监控端口 |
| `HEY_ROBOT_FRONTEND_PORT` | 8080 | 前端入口；动态请求反代至 Gateway |
| `HEY_ROBOT_APP_UID` / `HEY_ROBOT_APP_GID` | 1000 / 1000 | 应用容器写入 runtime 时使用的身份 |
| `HEY_ROBOT_BASE_IMAGE` | `hey-robot-base:latest` | 三个 Python 服务的公共 core 镜像 |
| `HEY_ROBOT_GATEWAY_IMAGE` | `hey-robot-gateway:latest` | Gateway 镜像 |
| `HEY_ROBOT_AGENT_IMAGE` | `hey-robot-agent:latest` | Agent 镜像 |
| `HEY_ROBOT_ROBOT_IMAGE` | `hey-robot-robot:latest` | Robot/SkillWorker 镜像 |
| `HEY_ROBOT_POLICY_IMAGE` | `hey-robot-policy:latest` | RoboCasa Policy 使用的 LeRobot 推理镜像 |
| `HEY_ROBOT_ROBOCASA_POLICY_PORT` | 9091 | RoboCasa Policy 的宿主机端口 |
| `HEY_ROBOT_ROBOCASA_POLICY_GPU` | 0 | RoboCasa Policy 使用的 GPU 编号 |
| `HEY_ROBOT_ROBOCASA_POLICY_CONFIG` | `configs/evaluation/robocasa365.yaml` | RoboCasa Policy 配置文件 |
| `HEY_ROBOT_ROBOCASA_POLICY_SERVICE_ID` | `robocasa365` | RoboCasa Policy 服务 ID |
| `HEY_ROBOT_VLN_PORT` | 9091 | VLN gRPC 端口 |
| `HEY_ROBOT_VLN_GPU` | 0 | VLN 使用的 GPU 编号；单 GPU 机器不要与 RoboCasa 同时运行 |
| `HEY_ROBOT_VLN_CONFIG` | `configs/xlerobot.sim.vln.yaml` | VLN 配置文件 |
| `HEY_ROBOT_VLN_SERVICE_ID` | `vln_nav` | VLN 服务 ID |

### Dockerfile 说明

**`docker/Dockerfile.base`** — Gateway / Agent / Robot 公共基础：
- 基础镜像：`python:3.12-slim`
- 公共层只包含 NumPy、Pillow、YAML、dotenv 与 NATS

**`docker/Dockerfile.gateway`**、**`Dockerfile.agent`**、**`Dockerfile.robot`**：
- 分别从 Compose 提供的 `hey_robot_base` 构建上下文继承
- Gateway 安装 FastAPI/Uvicorn/渠道依赖
- Agent 安装 OpenAI/HTTP/模板依赖
- Robot 安装 OpenCV、串口、gRPC 与机器人 SDK
- Torch/Ultralytics 不进入这三个服务镜像

**`docker/Dockerfile.frontend`** — 前端镜像：
- 基础镜像：`nginx:1.27-alpine`
- 静态页面由 Nginx 直接提供，API 与 WebSocket 反代至 Gateway

**`docker/Dockerfile.policy`** — 通用 LeRobot Policy 服务：
- 基础镜像：`python:3.12-slim-bookworm`
- 依赖：`uv sync --extra model-service --group lerobot-policy`
- 入口：`python -m hey_robot.cli.model_service`
- Policy 类型由 checkpoint 和 LeRobot factory 解析，不固化为 VLA
- 当前仅由 `robocasa-policy` Compose 服务使用
- CUDA/PyTorch 用户态库来自锁文件，NVIDIA runtime 只挂载宿主驱动，避免与
  `nvidia/cuda` 基础镜像重复保存 CUDA/cuDNN

**`docker/Dockerfile.vln`** — VLN 导航服务：
- 基础镜像：`python:3.12-slim-bookworm`
- 依赖：通过 `uv.lock` 同步独立 `vln` group，不与主运行时或 Policy 环境混装
- CUDA/PyTorch 用户态库同样只来自锁文件，避免基础镜像再保存一套 CUDA/cuDNN
- 入口：`python -m hey_robot.cli.model_service --service-id vln_nav`
- GPU 1，gRPC :9091

## 常见问题

### MuJoCo 启动报 OpenGL/GLFW 错误

headless 环境没有 X11 显示，需要改用 EGL 渲染：

```bash
export MUJOCO_GL=egl
```

确认 `libEGL.so` 已安装：`ldconfig -p | grep libEGL`

### VLN 报 huggingface-hub 版本冲突

说明 VLN 服务用了主环境或 `.venv-vln` 没有按锁文件同步。VLN 需要
`huggingface-hub==0.33.4`。运行 `scripts/dev/setup_vln_env.sh` 重建独立环境，
不要使用 `uv sync --group vln --python .venv-vln/bin/python`；后者缺少
`model-service` extra，并且仍可能选择项目
默认 `.venv`。

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

Windows 和 Ubuntu 当前配置都设置 `viewer.enabled: true`，需要可用的图形桌面。
headless 环境应先改为 `false` 并使用 EGL。

### 麦克风/语音不工作

Ubuntu 使用系统默认音频设备（`input_device: null`），Windows 根据
`scripts/audio/list_devices.py` 的输出调整设备索引。注意当前 Ubuntu 仿真配置使用云端
Doubao ASR，需要有效的 `ARK_API_KEY`；Windows 仿真配置使用本地 sherpa-onnx。
### gRPC 请求超时 (DEADLINE_EXCEEDED)

配置中 `target: grpc://127.0.0.1:9091` 的 `grpc://` 前缀是项目内部格式，`GrpcModelServiceClient` 会自动去掉。如果直接使用 gRPC 客户端工具测试，目标地址应为 `127.0.0.1:9091`（不带 scheme）。
