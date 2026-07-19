# RoboCasa365 独立容器接入 Hey Robot

本文给出 RoboCasa365 接入 Hey Robot 的架构决策、实现细节和分阶段验收记录。
目标是先验证完整的 Agent 调度链路，不训练模型，也不把 RoboCasa、robosuite 或其资产
安装进 Hey Robot 主环境。

本文基于 2026-07-19 时的以下上游接口：

- [LeRobot RoboCasa365 文档](https://huggingface.co/docs/lerobot/en/robocasa)
- [LeRobot RoboCasa benchmark Dockerfile](https://github.com/huggingface/lerobot/blob/main/docker/Dockerfile.benchmark.robocasa)
- [RoboCasa 仓库](https://github.com/robocasa/robocasa)
- [预训练 `lerobot/smolvla_robocasa`](https://huggingface.co/lerobot/smolvla_robocasa)

> 状态：首版任务级接入、离线资产挂载、独立 GPU 镜像、官方 checkpoint smoke test 以及
> ModelService v1 gRPC 调度均已完成。帧级 Runtime 的协议、Hey Robot `RobotDriver` 适配与
> 容器服务实现也已完成。Gate 0–5 均已执行；其中 Gate 4 的任务成功率属于 checkpoint
> 质量问题，不影响系统接入验收。本文将任务级控制面、评测结果和逐帧 Runtime 分开标注。

## 1. 结论

首版采用“容器内闭环 rollout”，而不是把 RoboCasa 伪装成现有 XLeRobot Driver：

```text
Hey Robot Agent / Task / Memory
             |
             | robocasa_rollout(task, seed, n_episodes)
             v
Skill OS -> ModelService gRPC v1 client
             |
             | ExecuteSkill / CancelSkill / GetHealth
             v
RoboCasa365 独立 GPU 容器
  RoboCasa + robosuite + LeRobot + SmolVLA
  reset -> observe -> policy -> 12D action -> step -> success
             |
             v
eval_info.json + video + 结构化执行结果
```

这个边界有三个重要含义：

1. RoboCasa 容器同时拥有仿真环境、VLA checkpoint 和控制循环。
2. Hey Robot 只下发语义任务并接收任务级反馈，不在首版逐帧执行 PandaOmron 动作。
3. 首版验证的是 `Agent -> Skill -> 外部具身执行 -> 反馈 -> Task` 闭环；它不是
   XLeRobot 真机的 sim-to-real 验证。

这是默认且风险最低的接法。帧级 `observe/reset/step` Driver 已作为可选通路实现，供需要
让 Hey Robot 直接编排 VLA action loop 的实验使用；它不替代官方 benchmark 的任务级闭环。

## 2. 为什么不能直接复用现有 VLA 控制循环

### 2.1 Hey Robot 当前的有效边界

Hey Robot 已经有适合承载外部仿真服务的分层：

- `RobotDriver` 负责本地硬件或仿真的 `observe/apply_action/reset` 生命周期；
- `RobotRuntime` 负责观测实体化、安全检查、状态和控制权；
- Skill OS 负责任务级控制循环和失败反馈；
- Foundation ModelService 通过 gRPC 隔离 VLA/VLN 的模型环境；
- `ModelServiceRegistry` 根据 `provides + robot_id` 路由远程服务。

相关实现：

- `src/hey_robot/robot_runtime/base.py`
- `src/hey_robot/robot_runtime/runtime.py`
- `src/hey_robot/foundation/transport/grpc/client.py`
- `proto/hey_robot/model_service/v1/model_service.proto`

`RobotManager` 现在构造以下组合：

```text
mock
xlerobot + mujoco -> XLeRobotSimDriver
xlerobot + native -> XLeRobotDriver
robocasa + remote + grpc -> RoboCasaRemoteDriver
```

只有以上完整的 `family/environment/driver` 组合受支持；任意 RoboCasa 配置仍不可隐式回退到
XLeRobot 或 SO101 driver。

### 2.2 当前 LeRobot VLA executor 是 SO101 单臂语义

现有 `LeRobotVLAPolicyExecutor` 和 `manipulation_adapter` 的动作约定是：

```text
SO101 单臂关节目标 + gripper
    -> move_arm_joints
    -> set_gripper
```

而 LeRobot 的 RoboCasa 环境约定是：

- 机器人：PandaOmron，Franka 机械臂 + 全向移动底盘；
- 状态：16 维；
- 动作：`Box(-1, 1, shape=(12,))`；
- 相机：左侧第三人称、腕部、右侧第三人称三路 256×256 RGB；
- 默认控制频率：20 FPS；
- 默认 episode 上限：1000 steps。

12D 动作包含底盘、控制模式、末端位置、末端旋转和夹爪，不能安全转换为当前
SO101 的 `move_arm_joints/set_gripper` primitive。若直接把 checkpoint 填进
`configs/xlerobot.sim.vla_vln.yaml`，即使模型能加载，动作语义仍然错误。

### 2.3 不采用逐帧跨容器控制的原因

逐帧方案需要在两个进程之间持续传输三路图像、状态和 12D action chunk，并处理：

- 20Hz 时序和超时；
- episode reset 与 seed；
- action chunk 缓存和 policy reset；
- MuJoCo step 所有权；
- 中断时的安全停止；
- 图像编码开销和背压；
- 仿真 success/done 与 Hey Robot Task 状态的一致性。

这些问题对最终的通用 Runtime 很重要，但不是验证第三方 VLA 是否能被 Agent 调度的
最短路径。首版把高频闭环保留在 RoboCasa 容器内，只跨边界传任务级 RPC。

## 3. RoboCasa365 与预训练策略基线

首个 checkpoint 固定为：

```text
lerobot/smolvla_robocasa
```

该模型使用 `lerobot/robocasa_target_human_unified` 数据训练，约 0.5B 参数，许可证为
Apache-2.0。首个任务选择 `CloseFridge`，原因是：

- LeRobot 文档提供了完整评测命令；
- 只需 lightwheel 轻量资产，不依赖约 30GB 的完整 objaverse/aigen 包；
- fixture-centric 任务更适合作为安装和控制链路 smoke test；
- 官方另有 `pepijn223/robocasa_CloseFridge` 单任务数据集，后续容易做针对性微调。

三路相机必须按 checkpoint 训练时的名称重命名：

```json
{
  "observation.images.robot0_agentview_left": "observation.images.camera1",
  "observation.images.robot0_agentview_right": "observation.images.camera2",
  "observation.images.robot0_eye_in_hand": "observation.images.camera3"
}
```

这里采用 checkpoint 的 `train_config.json` 和自带 preprocessor 所记录的训练顺序。当前
LeRobot 单任务文档示例曾将右侧相机和腕部相机对调，不能用该示例覆盖 checkpoint 的训练
映射。正式 target 评测还必须显式设置 `--env.split=target`；省略时 wrapper 会使用 `all`，
从全部布局、风格和物体实例采样，结果不再是 target benchmark。

首轮只启用：

```text
--env.obj_registries='[lightwheel]'
```

不要在未下载完整资产时启用 `objaverse` 或 `aigen`。缺失 registry 可能导致对象采样
概率异常，或在 reset 时因纹理/mesh 不存在而失败。

## 4. 独立镜像设计

### 4.1 隔离要求

RoboCasa 必须使用独立镜像和独立 Python 环境，不能加入 Hey Robot 的 `vla` dependency
group，原因包括：

- Hey Robot 当前是 Python 3.12、`lerobot>=0.6.0`；
- RoboCasa 自身 `setup.py` 固定依赖 `lerobot==0.3.3`；
- RoboCasa 和 robosuite 未发布为可直接依赖的稳定 PyPI 组合；
- MuJoCo、NumPy、Numba、图形后端和资产版本需要整体锁定；
- checkpoint、模型 cache 和厨房资产体积大，不应进入主 runtime 镜像。

安装 RoboCasa 时必须使用 `--no-deps`，再显式安装其运行依赖。不要让其声明的
`lerobot==0.3.3` 覆盖 LeRobot 0.6 环境。

### 4.2 已实现文件

首版已新增以下文件，且不修改现有 `docker/Dockerfile.vla`：

```text
docker/Dockerfile.robocasa365
deploy/robocasa365/
  benchmark.py
  server.py
  runtime_server.py
  runtime_smoke.py
  rollout.py
  requirements.txt
configs/robocasa365.worker.yaml
configs/robocasa365.remote.yaml
proto/hey_robot/robocasa_runtime/v1/robocasa_runtime.proto
src/hey_robot/robot_runtime/robocasa_remote/
tests/integration/test_robocasa365_contract.py
tests/robot_runtime/test_robocasa_remote_driver.py
```

`server.py` 同时注册现有 `ModelService v1` 和 `RoboCasaRuntime v1`；`rollout.py` 负责以参数
数组调用 `lerobot-eval`、解析 `eval_info.json`、管理 timeout/cancel 和输出目录。
`runtime_server.py` 在容器内独占一个 environment episode，负责 JPEG 三相机观测与严格的
`CreateEpisode / Observe / Step / Reset / CloseEpisode` 因果 RPC。容器无需启动 NATS，也不应
导入 Hey Robot Agent、Skill OS 或 Robot Runtime。

### 4.3 Dockerfile 与存储基线

实际镜像定义在 `docker/Dockerfile.robocasa365`：以 Python 3.12 为基础，固定
`torch==2.7.1` / `torchvision==0.22.1` 的 CUDA 12.6 wheel，安装锁定的 LeRobot、RoboCasa
和 robosuite 源码，再生成 ModelService v1 与 RoboCasaRuntime v1 的 gRPC stub。这样既满足 LeRobot 0.6 的
Python 版本要求，也避免主 Hey Robot 环境引入 MuJoCo 依赖。

资产不复制进镜像：将已解压的
`artifacts/robocasa365/merged-assets` 以可写 volume 挂到
`/opt/robocasa/robocasa/models/assets`。该目录由镜像内随源码提供的 XML/YAML 基础资产和
`extracted/{textures,generative_textures,fixtures,objects/lightwheel}` 合并生成；不能直接将
下载包覆盖整个 `assets` 目录，否则会丢失 `arenas/*.xml`、
`fixtures/fixture_registry/*.yaml` 等定义。Hugging Face cache 默认挂载宿主
`/home/liber/.cache/huggingface`，输出挂载为 `runtime/robocasa365`。这使重建镜像不重复
复制约 3 GB 的资产或 checkpoint。

Docker 的 image/containerd 存储也应放在有足够容量的数据盘。本机已将 Docker data root
设为 `/home/liber/docker-data`，并将 `/var/lib/containerd` 指向
`/home/liber/containerd-root`；因此构建期间的大型 CUDA 解压层不会耗尽 `/var`。

首次准备或上游镜像版本变更后，按以下命令重建合并资产目录。原始下载包保持不变，运行时
生成的派生 XML 只会写入 `merged-assets`：

```bash
docker create --name robocasa-assets-seed hey-robot-robocasa365:latest
mkdir -p artifacts/robocasa365/merged-assets
docker cp robocasa-assets-seed:/opt/robocasa/robocasa/models/assets/. \
  artifacts/robocasa365/merged-assets/
docker rm robocasa-assets-seed
cp -a artifacts/robocasa365/extracted/. artifacts/robocasa365/merged-assets/
```

上游版本、Python package pin 和镜像 digest 会写入构建产物版本信息。不要长期依赖漂移的
`main/master/latest`；gRPC wire compatibility 由 proto package 和 service path 决定，容器
不需要安装完整 Hey Robot。
不允许把资产放入镜像，可改成只读 asset volume；无论选择哪种方式，不能在每次容器启动时
重新拉取。

### 4.4 Compose 服务

建议在 `docker-compose.yml` 中新增单独 profile：

```yaml
services:
  robocasa365:
    build:
      context: .
      dockerfile: docker/Dockerfile.robocasa365
    image: ${HEY_ROBOT_ROBOCASA_IMAGE:-hey-robot-robocasa365:latest}
    environment:
      MUJOCO_GL: egl
      HF_HOME: /cache/huggingface
      HF_TOKEN: ${HF_TOKEN:-}
      ROBOCASA_POLICY: ${ROBOCASA_POLICY:-lerobot/smolvla_robocasa}
      ROBOCASA_DEFAULT_TASK: ${ROBOCASA_DEFAULT_TASK:-CloseFridge}
      ROBOCASA_OUTPUT_ROOT: /outputs
    volumes:
      - ${HF_CACHE_DIR:-/home/liber/.cache/huggingface}:/cache/huggingface
      - ./artifacts/robocasa365/merged-assets:/opt/robocasa/robocasa/models/assets
      - ./runtime/robocasa365:/outputs
    ports:
      - "${HEY_ROBOT_ROBOCASA_PORT:-9092}:9092"
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${HEY_ROBOT_ROBOCASA_GPU:-0}"]
              capabilities: [gpu]
    restart: unless-stopped
    profiles:
      - robocasa
```

如果 Hey Robot 在宿主机运行，gRPC target 使用 `127.0.0.1:9092`；如果 Hey Robot
runtime 也在同一个 Compose network 中运行，则使用 `robocasa365:9092`。不要从容器内
使用 `127.0.0.1` 访问另一个容器。

## 5. 分阶段验证

### 5.0 本工作区验收记录（2026-07-19）

- 独立镜像 `hey-robot-robocasa365:latest` 已构建；Docker image/containerd 数据位于
  `/home/liber` 数据盘，而不是系统 `/var` 分区。重启后发现 Snap Docker 与系统 Docker
  会争用 socket，已停用 Snap 服务并使用系统 Docker。
- `textures`、`generative_textures`、`fixtures`、`objects/lightwheel` 和
  `.robocasa-assets-ready` 均已在容器中确认；`lerobot`、`robocasa`、`robosuite`、`mujoco`
  import 成功。RoboSuite 的 private macro / 可选 robot-model 警告不影响本次 lightwheel
  benchmark 资产检查。
- RoboCasa 契约与 Remote Driver 的 15 项测试和 Ruff 均通过。
- 重启后，容器内 `torch 2.7.1+cu126` 能识别 GTX 1650。Gate 1 使用固定 seed 42 跑完
  `CloseFridge` 单 episode（1000 steps，约 336 秒），已写出 `eval_info.json` 和视频；该
  episode 为 0/1 success，属于策略质量结果。
- Gate 2 的 `GetHealth` 返回 `online=true`、`loaded=true`；Gate 3 通过 ModelService v1
  执行 `robocasa_rollout`，同样完整写出 request、versions、stdout/stderr、评测 JSON 和视频，
  并返回结构化 `task_unsuccessful`（0/1）。
- Gate 4 已完成 20 episodes 和版本审计；评测进程、版本 manifest、20 个 episode、视频与
  `eval_info.json` 均完整，策略结果为 0/20。该数值作为 checkpoint 已知限制保留，不作为
  Hey Robot 接入失败处理。
- Gate 5 已真实执行 `CreateEpisode -> Observe -> Step -> Reset -> CloseEpisode`：返回
  16D state、三路 JPEG 和 12D action step，frame id 按 `0 -> 1 -> 2` 推进，关闭后
  `busy=false`。任务级 rollout 与帧级 episode 的共享互斥也已验证。
- 真实 rollout cancel 返回 `status=cancelled`、`failure_mode=rollout_cancelled`，随后
  `busy=false`，可再次创建 frame-level episode，证明子进程与共享资源租约均已释放。
- 下载包不完整地覆盖源码 assets 会导致 arena XML、fixture registry 或 fixture XML 缺失；
  `merged-assets` 是镜像基础 assets 与下载包合并后的运行目录，并保持可写以容纳 RoboCasa
  生成的派生 MJCF XML。

### 5.1 Gate 0：仅验证镜像与环境

在接入 Hey Robot 前，先确认镜像内能够 import、创建环境和无头渲染：

```bash
docker compose --profile robocasa build robocasa365

docker compose --profile robocasa run --rm \
  --entrypoint python robocasa365 \
  -c "import robocasa, robosuite, mujoco; print('robocasa imports ok')"
```

通过条件：进程退出码为 0，且没有缺失纹理、fixture、EGL device 或 MuJoCo license
错误。

### 5.2 Gate 1：官方预训练 VLA 独立 smoke test

先绕过 Hey Robot，直接运行官方 LeRobot 评测：

```bash
docker compose --profile robocasa run --rm robocasa365 \
  lerobot-eval \
  --policy.path=lerobot/smolvla_robocasa \
  --env.type=robocasa \
  --env.task=CloseFridge \
  --env.split=target \
  --env.obj_registries='[lightwheel]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --output_dir=/outputs/smoke-close-fridge \
  '--rename_map={"observation.images.robot0_agentview_left":"observation.images.camera1","observation.images.robot0_agentview_right":"observation.images.camera2","observation.images.robot0_eye_in_hand":"observation.images.camera3"}'
```

LeRobot 会在以下位置生成结果：

```text
runtime/robocasa365/smoke-close-fridge/eval_info.json
runtime/robocasa365/smoke-close-fridge/videos/
```

`eval_info.json` 包含逐 episode 的 `success` 和聚合的 `pc_success`。一个 episode 失败
不等于接入失败；Gate 1 的系统通过条件是：checkpoint 成功加载、episode 完整结束、
结果 JSON 和视频成功写出。策略质量判断需要至少 20 episodes。

### 5.3 Gate 2：独立 worker 健康检查

启动服务：

```bash
docker compose --profile robocasa up -d robocasa365
docker compose ps robocasa365
docker compose logs --tail=200 robocasa365
```

用现有 `ModelService.GetHealth` 调用验证：

```text
online=true
loaded=true
busy=false
robot_id=robocasa365
version 中包含 bridge、LeRobot、RoboCasa、robosuite 和 checkpoint revision
```

`loaded=true` 必须表示模型和最小资产均可用，不能只表示 gRPC 端口已经监听。

### 5.4 Gate 3：Hey Robot 到 worker 的显式 Skill 调用

首版新增 `robocasa_rollout` Skill。它只调用 ModelService，不读取本地
`RobotObservation`，也不执行本地 primitive。

建议输入契约：

```json
{
  "type": "object",
  "properties": {
    "task": {"type": "string"},
    "n_episodes": {"type": "integer", "const": 1},
    "seed": {"type": "integer"},
    "policy_path": {"type": "string"},
    "obj_registries": {
      "type": "array",
      "items": {"type": "string"}
    },
    "record_video": {"type": "boolean"}
  },
  "required": ["task"]
}
```

Skill 规范建议：

```text
name: robocasa_rollout
required_model_service: robocasa_rollout
required_resources: ()
driver_primitives: ()
supported_robots: ()
feedback_mode: status
refresh_observation: false
timeout_sec: 1800
```

这里 `supported_robots: ()` 表示不限定当前 Hey Robot 本体。RoboCasa PandaOmron 是外部
benchmark embodiment，不能冒充 XLeRobot 的本地 RobotRuntime。

建议配置：

```yaml
model_services:
  robocasa365:
    type: vla_policy
    robot_id: ""
    enabled: true
    target: grpc://127.0.0.1:9092
    provides:
      - robocasa_rollout
    timeout_sec: 1800
    settings:
      health_timeout_sec: 10
      policy_path: lerobot/smolvla_robocasa
      default_task: CloseFridge
```

首版 Agent-facing Skill 把 `n_episodes` 固定为 1，确保 `success` 能无歧义地映射为这次
具身任务是否完成。20-episode 评测走同一镜像的独立 benchmark/admin 命令，不作为一次
Agent Skill。

`robot_id: ""` 在当前 `ModelServiceRegistry` 中表示不按本地 robot id 限定；如果以后
为 RoboCasa 实现正式 Remote Driver，应改为明确的 `robocasa365`。

首次从 Hey Robot 只调用一个 episode：

```json
{
  "task": "CloseFridge",
  "n_episodes": 1,
  "seed": 1000,
  "policy_path": "lerobot/smolvla_robocasa",
  "obj_registries": ["lightwheel"],
  "record_video": true
}
```

通过条件：

- Hey Robot 生成 Skill lifecycle 和 trace；
- gRPC health、execute、timeout、cancel 均可观测；
- worker 返回真实 `success`，而不是把进程退出码当成任务成功；
- Task 页面能够看到 task、seed、checkpoint、耗时、episode success 和输出目录；
- 同一个 `skill_id` 不会并发启动两个 rollout；
- 超时或取消会终止 rollout，并释放 MuJoCo 环境和 GPU 内存。

### 5.5 Gate 4：重复性评测

系统接通后通过独立 benchmark/admin 命令运行 20 episodes，不要从 Agent-facing Skill
发起：

```text
task=CloseFridge
split=target
seeds=1000..1019
n_episodes=20
batch_size=1
use_async_envs=false
obj_registries=[lightwheel]
```

记录：

```text
checkpoint id + resolved revision
LeRobot commit
RoboCasa commit
robosuite commit
container image digest
GPU/driver/CUDA
task/split/seed
success rate
mean episode time
failure mode distribution
videos and eval_info.json
```

不要只报告一次成功视频。官方建议 20 episodes/task 才用于可复现比较。

仓库提供独立入口 `deploy/robocasa365/benchmark.py`，容器中执行：

```bash
docker-compose --profile robocasa run --rm robocasa365 \
  python benchmark.py --task CloseFridge --episodes 20 --seed 1000 \
  --output-dir /outputs/benchmark-close-fridge-1000-1019
```

它将 `benchmark_manifest.json`、完整命令、20 个连续 seed、GPU/版本信息、标准输出和
`eval_info.json` 一并写到输出目录。

帧级 Runtime 的可重复 smoke test 位于 `deploy/robocasa365/runtime_smoke.py`。更新镜像并启动
服务后，在容器内执行：

```bash
docker exec hey-robot-robocasa365-1 \
  python /opt/bridge/runtime_smoke.py --target 127.0.0.1:9092 \
  --task CloseFridge --seed 4242
```

它同时验证三路图像、16D state、12D step、frame id、Reset、Close，以及 frame-level
episode 存在时任务级 rollout 返回 `model_service_busy`。

## 6. Worker RPC 语义

### 6.1 ExecuteSkill

`ExecuteSkillRequest` 到 rollout 的映射：

| RPC 字段 | RoboCasa 行为 |
|---|---|
| `skill_id` | 幂等键和输出目录名的一部分 |
| `objective` | 日志中的用户目标，不直接替代 task class |
| `arguments.task` | `--env.task`，首版只接受 allowlist |
| `arguments.policy_path` | `--policy.path`，默认固定为官方 checkpoint |
| `arguments.seed` | LeRobot eval seed |
| `arguments.n_episodes` | 首版 Agent Skill 只接受 1；benchmark/admin 接口可接受更多 |
| `timeout_sec` | worker 侧硬超时，必须早于 gRPC deadline 清理资源 |

首版建议 allowlist：

```text
CloseFridge
OpenDrawer
OpenCabinet
TurnOnMicrowave
TurnOffStove
```

先确认 checkpoint 对这些任务的训练覆盖和所需资产，再逐个开放。不能把任意用户自然语言
直接拼接进 shell；worker 必须使用参数数组调用 CLI，或直接调用 Python API。

### 6.2 ExecuteSkillResponse

成功完成 rollout 后建议返回：

```json
{
  "success": true,
  "status": "completed",
  "summary": "RoboCasa CloseFridge: 1/1 successful",
  "metrics": {
    "benchmark": "robocasa365",
    "task": "CloseFridge",
    "policy_path": "lerobot/smolvla_robocasa",
    "policy_revision": "<resolved revision>",
    "n_episodes": 1,
    "success_count": 1,
    "success_rate": 1.0,
    "seeds": [1000],
    "duration_sec": 42.5,
    "output_dir": "/outputs/<skill-id>",
    "video_paths": ["/outputs/<skill-id>/videos/episode-0.mp4"]
  }
}
```

这里 `success` 表示任务 success，不是“评测进程正常退出”。如果进程正常但 episode 失败：

```text
success=false
status=failed
failure_mode=task_unsuccessful
```

环境启动、模型下载、CUDA OOM、资产缺失和超时应使用不同 failure mode：

```text
invalid_task
checkpoint_unavailable
asset_unavailable
environment_reset_failed
policy_load_failed
cuda_out_of_memory
rollout_timeout
rollout_cancelled
task_unsuccessful
result_parse_failed
```

### 6.3 Health 与 Cancel

`GetHealth` 至少报告：

```text
online, loaded, busy, current_skill_id
policy_path, policy_revision
lerobot_version, robocasa_commit, robosuite_commit
mujoco_version, cuda_available, gpu_name
asset_profile=lightwheel
```

`CancelSkill` 必须：

1. 标记当前 skill 已取消；
2. 终止 rollout task 或子进程；
3. 调用 env close；
4. 等待 GPU/视频写线程退出；
5. 返回 `accepted=true`；
6. 使 ExecuteSkill 最终返回 `rollout_cancelled`。

## 7. 可观察性与数据边界

容器输出统一挂载到：

```text
runtime/robocasa365/<skill-id>/
  request.json
  versions.json
  eval_info.json
  stdout.log
  stderr.log
  videos/
```

`request.json` 不保存 Hugging Face token。`HF_TOKEN` 只通过环境变量或 Docker secret
注入，不写入配置、日志或镜像层。

Hey Robot 事件和 Task metrics 只保存结构化摘要及相对 artifact 路径，视频保留在共享
volume。后续如果需要在 Web UI 播放视频，再通过 `LocalMediaStore` 导入，不让 gRPC
response 携带视频二进制。

## 8. 测试矩阵

### 8.1 不需要 GPU 的测试

- proto request/response 兼容性；
- task allowlist 和参数校验；
- `eval_info.json` 解析；
- success 与 process exit code 的区分；
- busy、timeout、cancel 状态机；
- ModelServiceRegistry 的 `provides` 路由；
- RoboCasa Skill 的 Task feedback 映射。

这些测试使用假的 eval runner，不下载模型或资产。

### 8.2 需要 GPU 的 smoke test

- 镜像 build；
- EGL headless render；
- 官方 checkpoint 下载和 load；
- `CloseFridge` 单 episode；
- 结果和视频 volume；
- Hey Robot 到 worker 的真实 gRPC 调用；
- rollout 中途取消后再次执行。

GPU smoke test 使用独立 pytest marker 或 CI job，不能进入默认 `uv run poe test`。

## 9. 可选帧级 Runtime 接入

为支持 Hey Robot 自己编排 VLA action loop，第二套远程 Runtime 协议已经实现：

```text
CreateEpisode(task, seed) -> episode_id
Observe(episode_id) -> 3 RGB + 16D state + frame_id
Step(episode_id, action[12]) -> observation + reward + success + done
Reset(episode_id)
CloseEpisode(episode_id)
```

实现边界：

```text
RoboCasaRemoteDriver implements RobotDriver                 已实现
RobotManager 支持 robocasa + remote + grpc                  已实现
PandaOmron embodiment profile（3 camera / 16D / 12D）       已实现
RoboCasaRuntime v1 proto 与容器内 episode owner             已实现
远程 JPEG 图像 materialization                              已实现
12D 维度、有限数值、expected_frame_id 因果校验             已实现
按 env.action_space 的物理限幅                              环境端实现
```

它不能复用 SO101 `manipulation_adapter`；12D action 的原生布局为 `base_motion(4) +
control_mode(1) + end_effector_position(3) + end_effector_rotation(3) + gripper(1)`，由
LeRobot RoboCasa wrapper 转换为环境 dict action。主 runtime 不做臆测的关节转换。

启动容器后，可用主 runtime 配置创建 driver：

```bash
hey-robot run --config configs/robocasa365.remote.yaml
```

`RoboCasaRuntime` 与 task-level `ModelService` 共用容器的 gRPC 端口 `9092`。二者共享同一个
资源租约：服务端一次只允许一个 frame-level episode 或一个任务级 rollout；并发请求返回
`model_service_busy`，不会同时抢占同一张 GPU。

首版任务级 worker 和后续远程 Driver 可以共存：前者用于官方 benchmark 回归，后者用于
Hey Robot 自己的闭环控制研究。

## 10. 实施顺序与验收

按以下顺序实施，任一 gate 未通过都不要继续扩大任务范围：

1. 新建并锁定 `Dockerfile.robocasa365`；
2. Gate 0：import、资产、EGL；
3. Gate 1：官方 SmolVLA + CloseFridge 单 episode；
4. 实现最小 ModelService v1 worker；
5. 添加 `robocasa_rollout` Skill 和配置；
6. Gate 2/3：health、单次 RPC、真实任务结果、cancel（已通过）；
7. Gate 4：20 episodes 与版本审计（系统验收已通过，策略为 0/20）；
8. Gate 5：真实 `CreateEpisode -> Observe -> Step -> Reset -> CloseEpisode`（已通过）；
9. 再开放更多 atomic task。

首版完成定义：

- Hey Robot 主环境没有 RoboCasa/robosuite 依赖；
- RoboCasa 使用独立 GPU 镜像和独立 cache/outputs volume；
- 使用公开 checkpoint，不需要训练；
- 用户从 Hey Robot 发起任务后，Task lifecycle 能追踪外部 rollout；
- success、失败、超时和取消都返回可解释的结构化反馈；
- 20-episode 结果可通过固定版本和 seed 复现；
- 任何结果都不被描述为 XLeRobot 真机能力。

## 11. 常见问题

### 为什么不直接修改 `docker/Dockerfile.vla`？

现有 VLA 镜像服务 XLeRobot/SO101 policy inference。RoboCasa 还拥有 MuJoCo 环境、
robosuite、厨房资产和不同的 LeRobot 安装约束。合并会扩大镜像、制造依赖冲突，并模糊
谁拥有控制循环。

### 为什么 ModelService 返回的是整个任务结果，而不是 action chunk？

首版的目标是验证 Agent 集成。官方 checkpoint 和官方 benchmark 已经能在同一进程完成
观测—动作循环，让这个高频闭环跨容器不会增加验证价值，只会增加时序与动作映射风险。

### 可以换成别人训练的其他 VLA 吗？

可以，但必须同时满足：checkpoint 能被当前 LeRobot 版本加载、输入相机和 state schema
匹配、输出是 RoboCasa 12D action schema、训练任务覆盖目标 task。不能只凭模型名是 VLA
就假设可以控制 PandaOmron。

### 首轮成功后可以直接部署到 XLeRobot 吗？

不可以。RoboCasa checkpoint 面向 PandaOmron，和 XLeRobot 的相机、状态、机械结构及动作
空间不同。这个接入验证 Hey Robot 的任务调度和外部具身反馈，不证明真机策略可迁移。
