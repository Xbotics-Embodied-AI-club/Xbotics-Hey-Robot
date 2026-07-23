# Habitat 3 接入 Hey Robot：基于源码审计的实施方案

本文基于 2026-07-19 拉取的 Habitat-Lab、Habitat-Sim 与当前 Hey Robot 代码，给出
不接入 VLA 的 Habitat 3 集成方案。目标不是只运行导航或操作 benchmark，而是让 Hey
Robot 作为上层 embodied agent，持续感知环境、与人类角色共处、选择技能、执行任务并根据
结果重规划。

## 1. 结论

Habitat 3 已经提供本方案需要的仿真基础能力，不需要在 Hey Robot 中重新实现机械臂仿真、
人类角色或多智能体物理；但具体 profile 中是否已经配置可直接调用的导航/操作执行器，必须按
action space 和 checkpoint 逐项确认：

- Habitat-Lab 的 `Env` 统一管理 dataset、simulator、task、observation、action 和 metrics；
- `RearrangeSim` 支持移动操作、刚体/关节物体、抓取、多个 articulated agent；
- 官方动作包括 base velocity、arm joint、grip、oracle navigation、humanoid joint/pick；
- 官方任务包括 PointNav、ObjectNav、VLN、Pick、Place、开关柜门、PDDL Rearrange、
  SocialNav 和多智能体协作；
- Habitat Baselines 已有 PDDL hierarchical policy、planner、oracle navigation、learned
  navigation、pick/place 等低层技能实现；
- Habitat 3 官方配置直接提供 Spot + KinematicHumanoid、Spot + Spot 等组合。

Hey Robot 应负责语言交互、目标管理、记忆、技能选择、失败恢复和任务完成判定；Habitat 容器
负责环境状态、物理、NPC 行为、profile 内明确配置的低层技能闭环和 benchmark metrics。
首版不接 VLA，也不把 Habitat 依赖装入 Hey Robot 主 Python 环境。

推荐架构是“独立 Habitat 容器 + gRPC episode runtime + Hey Robot Remote Driver +
Habitat 语义 Skill”，而不是让 Agent 直接构造 Habitat 的 Gym action dict。

按当前源码，本方案的可行性边界是：

- 独立容器、episode runtime、Remote Driver 和 SocialNav 仿真链路可行；
- 首版 oracle SocialNav 需要派生官方 config，为受控 Spot 增加 oracle coordinate executor，
  不能原样使用官方 SocialNav config 就声称 agent 0 已有 OracleNav；
- 不加载 RL checkpoint 时，pick/place 只能作为 privileged PDDL/symbolic executor 验证 Agent
  编排，不能算机械臂物理执行；
- 真正的物理 pick/place 需要 Habitat Baselines 的 pick/place checkpoint，或另行实现确定性的
  机械臂闭环控制器；
- 当前结论是源码级可行性审计。完成 Phase 1 的资产、EGL、Bullet 和 episode smoke 后，才能
  称为运行级验证。

## 2. 本地源码与审计基线

源码已放在 Git 忽略的 `artifacts/habitat3/`，不会把两个上游仓库嵌入本项目提交：

```text
artifacts/habitat3/habitat-lab
  remote: https://github.com/facebookresearch/habitat-lab.git
  commit: 0fb6f43ffe806a8088a171b036336c093bcf604e
  package VERSION: 0.3.3

artifacts/habitat3/habitat-sim
  remote: https://github.com/facebookresearch/habitat-sim.git
  commit: 57ee4941dc4765240f0f91f70b2c97a919bf9038
  package version: 0.3.3
```

两个 commit 都来自 2026-05-07 的官方 `main`。Habitat-Lab README 已明确提示：v0.3.4
之后不再由 Meta 内部团队主动维护。因此运行镜像必须锁 commit/version，不能持续跟踪
`main` 或 nightly。

本次审计的关键上游代码入口：

| 能力 | 实际代码入口 | 对接意义 |
|---|---|---|
| Episode 生命周期 | `habitat-lab/habitat/core/env.py::Env` | `reset/step/get_metrics/close` 是服务端唯一环境入口 |
| Habitat-Sim | `src_python/habitat_sim/simulator.py::Simulator` | 渲染、物理 step、多 agent observation |
| 移动操作 | `habitat/tasks/rearrange/rearrange_sim.py::RearrangeSim` | 管理 articulated agents、物体、抓取和 physics |
| 多智能体 | `rearrange_sim.py`、`articulated_agent_manager.py` | action/sensor 使用 `agent_0_`、`agent_1_` 前缀 |
| 底盘/手臂动作 | `tasks/rearrange/actions/actions.py` | 连续 base velocity 和 arm action |
| 抓取 | `tasks/rearrange/actions/grip_actions.py` | Magic、Suction、Gaze grasp |
| 人类动作 | `tasks/rearrange/actions/humanoid_actions.py` | humanoid joint/pick 行为 |
| 导航 | `actions/oracle_nav_action.py`、Habitat-Sim `PathFinder` | 首版可用确定性的 oracle executor |
| SocialNav | `tasks/rearrange/social_nav/` | 跟随、避碰、朝向、人类检测和成功指标 |
| 组合任务 | `tasks/rearrange/multi_task/` | PDDL entity、predicate、action、goal |
| 分层技能 | `habitat_baselines/rl/hrl/` | planner + nav/pick/place/wait 等技能闭环 |
| Habitat 3 配置 | `config/benchmark/multi_agent/` 和 `rearrange/hab3_bench/` | Spot/Human 与多 agent 官方参考配置 |

## 3. Habitat 3 能力边界

### 3.1 已有能力

导航算法基础不需要从零实现。Habitat-Sim `PathFinder` 提供 navmesh、最短路径、可导航点采样
和碰撞约束；Habitat-Lab 还提供离散导航、base velocity、oracle navigation 及 learned policy
接口。但 action 是否存在于当前受控 agent，取决于 profile：官方 SocialNav config 只给 Spot
配置 base velocity，不能把仓库中存在 `OracleNavAction` 等同于该 Spot profile 已经可调用它。

机械臂仿真和动作原语也不需要从零实现。`RearrangeSim` 和 task actions 已支持相对/绝对关节
位置、末端执行器控制、底盘与手臂组合动作以及多种抓取方式。官方 task config 已覆盖 pick、
place、open/close fridge/cabinet 和多阶段 rearrangement。但 Habitat Baselines 的物理
`PickSkillPolicy`、`PlaceSkillPolicy` 分别依赖 `data/models/pick.pth` 和
`data/models/place.pth`。oracle skill 配置中的 pick/place 是 `NoopSkillPolicy +
apply_postconds`，属于 privileged symbolic state transition，不是机械臂轨迹执行。

Habitat 3 的独特价值是人类角色和多智能体。官方
`hssd_spot_human_social_nav.yaml` 配置了 `agent_0=Spot`、`agent_1=Human`，包含机器人深度
相机、人类检测、双方定位、碰撞、跟随和 SocialNav 成功指标。PDDL multi-agent task 还能表达
两个 agent 分工搬运和共享 stage goals。

这里的官方 `HumanoidDetectorSensor` 通过 panoptic semantic id 判断人类是否在画面中，是
模拟器语义传感器，不是 RGB learned detector。报告必须区分 `oracle_detection`、
`sim_semantic_detection` 和 `rgb_learned_detection`，不能把前两者记为 Hey Robot 的视觉检测
能力。

### 3.2 不直接提供的能力

Habitat 3 不提供 Hey Robot 所需的完整语言 Agent 产品层：

- 不管理用户对话、长期记忆和跨任务目标；
- 不理解 Hey Robot 的 `SkillIntent`、证据链和通知系统；
- 不提供稳定的跨进程服务契约；
- 不会自动把任意自然语言转成可执行的 PDDL entity/action；
- 官方 oracle 和 benchmark policy 是仿真执行器，不是通用 VLA。

因此不能把 Habitat policy 当成 Hey Robot Agent。正确分工是：Hey Robot 选择语义技能，
Habitat runtime 将技能解析为当前 episode 中的 entity 和 action，并在容器内运行有限 horizon
闭环。

## 4. 推荐总体架构

```text
User / Voice / Web
        |
Hey Robot RobotAgent
  conversation + memory + goal/evidence + replanning
        |
Skill OS
  habitat_navigate_to / habitat_follow_human
  habitat_symbolic_pick / habitat_symbolic_place
  habitat_pick / habitat_place / habitat_wait  (checkpoint/controller profile only)
        |
RobotRuntime + HabitatRemoteDriver
        |  gRPC (typed lifecycle + Struct action/metrics)
        v
独立 habitat3 容器
  HabitatRuntimeServer
    -> one owner thread / one active Env
    -> skill executor (oracle/PDDL/optional trained policy)
    -> Habitat-Lab Env / RearrangeSim
    -> Habitat-Sim + Bullet + EGL
        |
HSSD / Habitat 3 assets / episodes / videos
```

Habitat runtime 是 robot runtime，不是 `ModelService`。本方案没有模型推理服务；即使以后加载
RL checkpoint，也应视作容器内部的技能执行器，除非它真正形成独立、可复用的模型服务。

## 5. 角色与首版场景

首个 profile 派生官方 SocialNav 结构：

```text
profile: habitat3_social_spot_human_oracle
controlled_agent: agent_0
embodiment: SpotRobot
npc_agent: agent_1
npc_embodiment: KinematicHumanoid
task: RearrangePddlSocialNavTask-v0
base_config: benchmark/multi_agent/hssd_spot_human_social_nav.yaml
config_patch:
  - add agent_0 oracle coordinate navigation action
  - preserve agent_0 base velocity action
  - preserve agent_1 oracle random-coordinate humanoid action
executor_mode: privileged_oracle
```

官方原始 config 中，`agent_0` 只有 `agent_0_base_velocity`；
`agent_1_oracle_nav_action` 和 `agent_1_oracle_nav_randcoord_action` 都属于 Human。因此首版必须
派生受控 profile：由服务端读取当前 Human 位置，使用 agent 0 的 oracle coordinate executor
生成 Spot 底盘动作；Human 继续使用官方 oracle/humanoid controller。受控/NPC action 必须在
同一个 `env.step()` 中提交。

这条链路能验证 Hey Robot 的“获取模拟器语义发现结果、接近人、保持社交距离、跟随、避碰、
等待、重新规划”，但不能验证 RGB 人体检测或 learned SocialNav。第二个 SocialNav profile
`habitat3_social_spot_human_learned` 再加载 Spot policy checkpoint，并与 oracle 结果分开报告。

完成两个 SocialNav profile 后，再加入 `habitat3_spot_human_rearrange_symbolic`，使用官方
Spot + Human PDDL rearrangement 验证导航和 privileged symbolic pick/place 编排。该 profile
必须使用 `benchmark/multi_agent/hssd_spot_human.yaml` 和 `RearrangePddlTask-v0`；不得在
SocialNav config 上伪造 `agent_0_pddl_apply_action`。加载低层 checkpoint 后再提供
`habitat3_spot_human_rearrange_physical`，验证真实 arm/grip action 闭环。
Humanoid 作为 Hey Robot 本体可以作为后续 profile，不应阻塞第一条链路。

每个 profile 必须固定以下内容，禁止客户端自由传任意 Python config 路径：

- 允许使用的 Habitat config；
- controlled agent 和 NPC agent；
- observation key allowlist；
- semantic skill allowlist；
- 最大 episode/skill steps；
- dataset split 和资产根目录；
- 是否允许 privileged state。

profile 还必须声明 `executor_mode`，至少支持：

- `privileged_oracle`：允许服务端用 simulator/PDDL truth 执行；
- `trained_policy`：加载并记录具体 checkpoint SHA；
- `physical_controller`：使用实际 arm/base action 闭环，不允许 apply-postcondition 冒充执行。

## 6. gRPC 契约

新增独立 `HabitatRuntime v1`，不要复用 RoboCasa 的固定 16D state、3 camera、12D action
契约。Habitat action 是具名、嵌套且随 agent/profile 变化的 Gym `Dict`。

建议 RPC：

```protobuf
service HabitatRuntime {
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc CreateEpisode(CreateEpisodeRequest) returns (EpisodeResponse);
  rpc Observe(EpisodeRequest) returns (ObservationResponse);
  rpc Step(StepRequest) returns (StepResponse);             // 调试/策略接口
  rpc ExecuteSkill(ExecuteSkillRequest) returns (SkillResponse);
  rpc CancelSkill(CancelSkillRequest) returns (CancelSkillResponse);
  rpc Reset(MutateEpisodeRequest) returns (ObservationResponse);
  rpc CloseEpisode(MutateEpisodeRequest) returns (CloseEpisodeResponse);
}
```

关键字段：

- `CreateEpisodeRequest`: `profile`, `task`, `split`, `requested_dataset_episode_id`,
  `seed`, `controlled_agent`；其中 profile/task/split 必须服务端 allowlist，runtime episode id 由
  服务端生成并返回，不能与 dataset episode id 混为一谈；
- `ObservationResponse`: `episode_id`, `frame_id`, repeated image/depth artifacts,
  `proprioception`, `task`, `done`, `success`, `metrics`, `entities`；
- `StepRequest.action`: `google.protobuf.Struct`，保留 Habitat 的嵌套 action dict；
- `ExecuteSkillRequest`: `skill_name`, `arguments`, `expected_frame_id`,
  `max_steps`；
- `SkillResponse`: 最终 observation、steps、success、failure mode、metrics、trace；
- `Step`、`ExecuteSkill`、`Reset` 和 `CloseEpisode` 等变更状态的请求都带
  `expected_frame_id`，拒绝 stale action；`MutateEpisodeRequest` 至少包含 `episode_id` 和
  `expected_frame_id`；
- 一个 episode 同时只允许一个 step/skill，服务端用资源锁保证因果顺序。

`Step` 只用于 contract test、debug 或以后接 policy；正常 Agent 路径走 `ExecuteSkill`。不能把
每个物理 timestep 通过 Agent/消息总线往返，否则延迟、取消和因果一致性都会变差。

dataset episode id 不能只是响应标签。若允许客户端指定官方 dataset episode id，服务端必须在
`Env.reset()` 前过滤/重排 dataset episode iterator，并校验 scene 和 split；否则只能把服务端
实际选中的 dataset episode id 返回给客户端。`seed` 也不能替代 episode id 的确定性选择。

## 7. 观测映射

Habitat observation 是按 config 动态生成的 dict。服务端先按 profile 归一化，再交给 Hey
Robot：

| Habitat 数据 | Hey Robot 映射 |
|---|---|
| RGB sensor | `ObservationAsset(kind="image")` |
| Depth sensor | 原始 16-bit PNG 或 float NPZ artifact；可另生成 uint8 可视化 image |
| joint/base/localization | `proprioception` 和 `raw.habitat.sensors` |
| PDDL entities/predicates | `metadata.entities` -> `SceneEntity` |
| task measurements | `raw.habitat.metrics` 和 `RobotStatus.metrics` |
| episode/scene/agent roles | `raw.habitat.episode` |

Hey Robot 的 `ObservationPipeline` 已能把 `DriverObservation.metadata.entities` 转成
`SceneEntity`，无需另建场景协议。首版允许用 PDDL/模拟器真值生成 entity 和成功证据，但必须
标记 `privileged: true`；Agent 默认只消费 RGB/depth 与允许的任务传感器，oracle state 只用于
验收和失败诊断，避免把 oracle benchmark 误报为感知能力。

现有 `LocalMediaStore.put_image()` 会把输入转换为 uint8 RGB，普通 artifact 路径只支持 JSON，
所以不能按现状无损保存 16-bit depth。实现时必须新增二进制 artifact/depth 写入路径（例如
`put_artifact_bytes()`），或把 float depth 存为已有 NPZ artifact；不能把完整 depth array 展开成
JSON。协议层 `ArtifactRef` 可以复用，但 `ObservationPipeline`/媒体存储实现需要小幅扩展。

若 runtime 使用 Habitat Gym wrapper，`habitat.gym.obs_keys` 会过滤返回观测；若直接使用 core
`Env`，则应由 profile observation allowlist 过滤。两条路径不能混用，否则即使 simulator 配置
了 RGB sensor，RPC 也可能只得到 policy 所需的 depth keys。

图像 key 必须保留 agent 前缀，例如 `agent_0_articulated_agent_arm_rgb`，否则多智能体下会把
NPC 视角误当成受控角色视角。

## 8. Hey Robot 代码接入点

建议新增：

```text
docker/Dockerfile.habitat3
evaluation/habitat3/worker/
  requirements.txt
  runtime_server.py
  environment.py
  observation.py
  skill_executor.py
  smoke.py
proto/hey_robot/habitat_runtime/v1/habitat_runtime.proto
src/hey_robot/habitat_runtime/v1/*_pb2*.py
src/hey_robot/robot_runtime/habitat_remote/
  protocol.py
  client.py
  driver.py
src/hey_robot/skill_os/builtins/habitat.py
configs/habitat3.social.yaml
tests/integration/test_habitat3_contract.py
tests/robot_runtime/test_habitat_remote_driver.py
```

已有代码的最小改动：

1. 在 embodiment registry 注册 `habitat3_social_spot_human`；
2. `RobotManager` 支持 `family=habitat3, environment=remote, driver=grpc`；
3. `HabitatRemoteDriver` 实现现有 `RobotDriver`，把语义 `RobotSkillAction` 转给
   `ExecuteSkill`；
4. `ObservationPipeline` 不改 `RobotObservation`/`ArtifactRef` 协议，但扩展二进制 depth
   artifact 存储；
5. 注册 Habitat 专用 semantic skills；
6. 在 `supported_driver_primitives()` 中增加 Habitat family，或在 robot settings 显式声明
   `supported_driver_primitives`，否则 Habitat 默认暴露空原语集合；
7. Compose 增加独立 profile `habitat3`，端口建议 `9093`，避免与 RoboCasa `9092`
   冲突。

首版 Agent-visible skills：

| Skill | 参数 | Habitat 执行器 |
|---|---|---|
| `habitat_navigate_to` | `entity_id` 或受限 `position` | profile-specific OracleNav/OracleNavCoordinate |
| `habitat_follow_human` | `human_id`, `distance_m`, `max_steps` | oracle coordinate loop 或 trained Spot policy + SocialNav measure |
| `habitat_symbolic_pick` | `object_id` | PDDL apply/postcondition，必须标记 privileged |
| `habitat_symbolic_place` | `object_id`, `receptacle_id` | PDDL apply/postcondition，必须标记 privileged |
| `habitat_pick` | `object_id` | PickSkillPolicy checkpoint 或 physical controller |
| `habitat_place` | `object_id`, `receptacle_id` | PlaceSkillPolicy checkpoint 或 physical controller |
| `habitat_wait` | `steps` | zero action loop；使用 Baselines 时可接 WaitSkillPolicy |
| `habitat_stop` | 无 | stop action + cooperative cancel |

不要首版覆盖现有 `navigate_to`：当前 Hey Robot 的该 skill 明确依赖 VLN ModelService。
Habitat 专用名字能避免同名 skill 注册冲突；等多 backend dispatch 设计成熟后再统一表面 API。

## 9. 容器与依赖隔离

Habitat 必须使用独立镜像，原因是它包含 Habitat-Sim 原生 C++/Magnum、Bullet、EGL、
旧 Gym/OmegaConf 依赖和可选 PyTorch baselines。它不能并入 Hey Robot 主环境，也不能并入
RoboCasa 镜像。

首版镜像建议：

- Linux + micromamba/conda；
- Python 3.9（与 Habitat-Lab 0.3.3 官方配置最保守兼容）；
- `habitat-sim=0.3.3 withbullet headless`；
- Habitat-Lab 固定 commit `0fb6f43...`；
- 仅 SocialNav/Env 所需依赖；oracle profile 不安装 VLA，也不要求低层 RL checkpoint；
- `MAGNUM_LOG=quiet`, EGL headless，GPU 通过 Compose reservation 注入；
- 镜像写入 Habitat-Lab SHA、Habitat-Sim version 和 image revision labels。

数据不进入镜像，统一挂载到项目所在大容量磁盘：

```text
artifacts/habitat3/data        -> /opt/habitat/data:ro
runtime/habitat3               -> /outputs
```

首条 SocialNav 链路至少需要官方 downloader 中实际声明的：

```text
hssd-hab
hab3-episodes
habitat_humanoids
hab_spot_arm
```

`hab3_bench_assets` 仅在使用 `benchmark/rearrange/hab3_bench/*` profile 时需要，不是上述
HSSD SocialNav profile 的必需资产。若 episode 引用了额外对象集，再根据 reset 的缺失 handle
补充 `ycb`/`ovmm_objects` 等资源，不能仅凭 config 中存在 `additional_object_paths` 就假设资产已
齐全。

官方 downloader 对多个 Hugging Face 资源声明的 version 仍是 `main`。下载完成后必须记录每个
Git/LFS 数据仓库的实际 commit SHA 或不可变 snapshot revision；只记录 downloader UID 不足以
复现实验。

下载前先做磁盘预算。HSSD 和 episode assets 远大于代码仓库，不能落到 `/var`。Docker
data-root/BuildKit cache 也应继续使用项目所在大盘或已迁移的数据目录。

实现中应提供可在宿主机和容器内运行的资产预检：

```bash
python -m evaluation.habitat3.worker.preflight \
  --data-root artifacts/habitat3/data \
  --profile habitat3_social_spot_human_oracle
```

预检至少检查 HSSD scene dataset config、实际 stage/uncluttered scene instance 目录、对应 split 的
rearrange episode、Spot URDF 和 humanoid data；
任一缺失时以非零状态退出，避免把“镜像能启动”误报为环境可运行。

## 10. 服务端并发和生命周期

Habitat Env、OpenGL context 和 simulator state 都有线程归属，不应从多个 asyncio handler
并行访问。服务端采用：

- 一个 runtime process；
- 一个 owner thread；
- 一个串行 command queue；
- 首版一个 active episode；
- gRPC handler 只做验证、入队和等待结果；
- skill 每个 env step 检查 cancel token 和 deadline；
- `CloseEpisode`/server shutdown 必须在 owner thread 调用 `Env.close()`。

`Observe` 不能隐式推进物理；`frame_id` 只在 reset/step/skill step 后递增。NPC 的 action 与
受控 agent action 必须在同一个 Habitat `env.step()` dict 中提交，确保多智能体同步。

当前 Hey Robot `RobotDriver` 没有显式 `cancel_skill()` 接口，Skill OS 超时主要取消本地 task，
不会自动调用 Habitat `CancelSkill`。首版至少要求 gRPC client 在等待 `ExecuteSkill` 被取消时捕获
`CancelledError`，用 episode/operation id 调用 `CancelSkill`；稳妥实现应为 RobotRuntime/Driver
增加显式 cancel hook。`habitat_stop` 还必须能绕过正在等待的长 RPC，不能排在同一串行 action
队列尾部后失去停止意义。

## 11. 分阶段实施与验收

### Phase 1：环境和官方能力 smoke

- 构建独立 Habitat 3 镜像；
- 加载官方 Habitat 3 benchmark asset；
- 用原生 config 完成 `reset -> observe -> step -> metrics -> close`；
- 确认 Bullet、EGL、Spot、Humanoid、多 agent sensor key；
- 输出 RGB/depth、episode metadata 和视频。

验收：容器重启后可重复运行；主 Hey Robot 环境没有 `habitat`/`habitat_sim` 依赖。

### Phase 2：gRPC 与 Remote Driver

- 实现 `HabitatRuntime v1`；
- 完成 health/create/observe/step/reset/close/cancel；
- 加入 profile allowlist、frame CAS、资源锁和 owner thread；
- 实现 `HabitatRemoteDriver`、Manager 和 embodiment 注册；
- 映射 RGB/depth/state/entities/metrics。

验收：Hey Robot `observe/status/reset` 可用；stale frame、错误 agent key、重复 episode 和
并发 step 都会被明确拒绝。

### Phase 3A：Oracle/Symbolic Embodied Agent 语义技能

- 使用派生 oracle profile 实现 navigate/follow/wait/stop；
- 再实现 privileged symbolic pick/place，并在名称、trace 和报告中明确标记；
- 每个 skill 返回 steps、PDDL/measure evidence、碰撞和 failure mode；
- 支持中途取消和 timeout；
- Agent 根据一次失败结果继续观察并重规划。

验收：自然语言请求通过 Hey Robot Agent 形成 SkillIntent，完成“根据模拟器语义结果找到人并
保持距离跟随”和“导航到目标后完成 symbolic rearrangement”等组合过程，而不是直接运行整集
脚本。此阶段不声称 RGB 感知成功或物理搬运成功。

### Phase 3B：Learned SocialNav 与物理操作

- 加载并固定 Spot SocialNav checkpoint，替换 agent 0 privileged oracle executor；
- 加载并固定 pick/place checkpoint，或实现并测试 deterministic physical controller；
- 使用 arm/base/grip observation 和 action 验证物体确实被抓取、移动和放置；
- 与 Phase 3 的 oracle/symbolic 指标、视频和 trace 分开报告。

验收：learned SocialNav 不读取 simulator human pose；物理 pick/place 不调用 apply-postcondition
代替轨迹执行，且 success 由 grasp/object pose/task measure 共同证明。

### Phase 4：基准与回归

- 固定 source SHA、image digest、profile、dataset revision、episode id 和 seed；
- 分开报告 oracle executor、trained policy 和 Agent orchestration；
- 记录 task success、stage goals、steps、collision、distance、timeout、视频；
- 加入 contract/unit/container smoke/agent end-to-end 四层测试。

验收时不能把 oracle success 称为 learned policy 或感知成功率。首版的价值是验证 Hey Robot
编排、记忆、交互、技能切换和恢复链路。

## 12. 主要风险与处理

| 风险 | 处理 |
|---|---|
| 上游停止主动维护 | 锁 commit、镜像 digest、dataset revision；保留本地源码镜像 |
| Habitat action space 随 config 变化 | profile allowlist + 服务端启动时导出 schema |
| 多 agent sensor/action 串线 | 所有 key 保留 agent 前缀并验证 controlled agent |
| Oracle 泄漏到 Agent | privileged 标记、默认不进 Agent prompt、单独报告指标 |
| Env/OpenGL 并发崩溃 | owner thread + serial queue + one active episode |
| 大资产占满系统盘 | artifacts/data 外挂大盘，Docker data-root 不放 `/var` |
| 语义 entity 不存在或过期 | entity id + expected frame + PDDL current state 校验 |
| 长技能无法取消 | 每个 env step 检查 cancel/deadline，返回 typed cancellation |
| SocialNav config 没有 agent 0 OracleNav | 派生受控 profile，启动时断言 action schema |
| symbolic pick/place 被误报为物理执行 | 技能分名、executor_mode 标记、分开报告 |
| depth 被降为 uint8 或展开成 JSON | 二进制 16-bit PNG 或 float NPZ artifact |
| dataset UID 指向可变 main | 记录下载后的实际 Git/LFS revision |

## 13. 最终建议

Habitat 3 很适合 Hey Robot 的 embodied agent 方向，而且比 RoboCasa 更贴近“人物/角色”：它
原生支持人类 avatar、社交导航、多智能体协作和移动操作。接入难度主要不在算法能力，而在
依赖隔离、动态 action/observation schema、多智能体角色映射和长技能生命周期。

首条实施路线应是：独立容器运行派生自官方 Spot + Human SocialNav 的 oracle profile，为
agent 0 显式增加 coordinate executor，让 Hey Robot 负责对话、任务分解、技能编排和失败
恢复。随后接 trained SocialNav checkpoint。操作侧先用明确标记的 symbolic PDDL 技能验证
编排，再接 pick/place checkpoint 或 physical controller；不能把 apply-postcondition 成功称为
机械臂搬运成功。当前没有必要引入 VLA。
