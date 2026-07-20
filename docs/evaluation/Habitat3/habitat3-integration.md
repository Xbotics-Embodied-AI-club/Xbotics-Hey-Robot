# Habitat 3 接入 Hey Robot：基于源码审计的实施方案

本文基于 2026-07-19 拉取的 Habitat-Lab、Habitat-Sim 与当前 Hey Robot 代码，给出
不接入 VLA 的 Habitat 3 集成方案。目标不是只运行导航或操作 benchmark，而是让 Hey
Robot 作为上层 embodied agent，持续感知环境、与人类角色共处、选择技能、执行任务并根据
结果重规划。

## 1. 结论

Habitat 3 已经提供本方案需要的仿真基础能力，不需要在 Hey Robot 中重新实现导航、机械臂、
人类角色或多智能体物理：

- Habitat-Lab 的 `Env` 统一管理 dataset、simulator、task、observation、action 和 metrics；
- `RearrangeSim` 支持移动操作、刚体/关节物体、抓取、多个 articulated agent；
- 官方动作包括 base velocity、arm joint、grip、oracle navigation、humanoid joint/pick；
- 官方任务包括 PointNav、ObjectNav、VLN、Pick、Place、开关柜门、PDDL Rearrange、
  SocialNav 和多智能体协作；
- Habitat Baselines 已有 PDDL hierarchical policy、planner、oracle navigation、learned
  navigation、pick/place 等低层技能实现；
- Habitat 3 官方配置直接提供 Spot + KinematicHumanoid、Spot + Spot 等组合。

Hey Robot 应负责语言交互、目标管理、记忆、技能选择、失败恢复和任务完成判定；Habitat 容器
负责环境状态、物理、NPC 行为、低层技能闭环和 benchmark metrics。首版不接 VLA，也不把
Habitat 依赖装入 Hey Robot 主 Python 环境。

推荐架构是“独立 Habitat 容器 + gRPC episode runtime + Hey Robot Remote Driver +
Habitat 语义 Skill”，而不是让 Agent 直接构造 Habitat 的 Gym action dict。

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

导航不需要单独实现。Habitat-Sim `PathFinder` 提供 navmesh、最短路径、可导航点采样和碰撞
约束；Habitat-Lab 还提供离散导航、base velocity、oracle navigation 及 learned policy 接口。

机械臂控制也不需要从零实现。`RearrangeSim` 和 task actions 已支持相对/绝对关节位置、
末端执行器控制、底盘与手臂组合动作以及多种抓取方式。官方 task config 已覆盖 pick、place、
open/close fridge/cabinet 和多阶段 rearrangement。

Habitat 3 的独特价值是人类角色和多智能体。官方
`hssd_spot_human_social_nav.yaml` 配置了 `agent_0=Spot`、`agent_1=Human`，包含机器人深度
相机、人类检测、双方定位、碰撞、跟随和 SocialNav 成功指标。PDDL multi-agent task 还能表达
两个 agent 分工搬运和共享 stage goals。

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
  habitat_pick / habitat_place / habitat_wait
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

首个 profile 使用官方 SocialNav 结构：

```text
profile: habitat3_social_spot_human
controlled_agent: agent_0
embodiment: SpotRobot
npc_agent: agent_1
npc_embodiment: KinematicHumanoid
task: RearrangePddlSocialNavTask-v0
base_config: benchmark/multi_agent/hssd_spot_human_social_nav.yaml
```

Hey Robot 控制 `agent_0`，`agent_1` 由容器内 oracle/humanoid controller 驱动。这样第一阶段
就能验证“发现人、接近人、保持社交距离、跟随、避碰、等待、重新规划”，而不是只验证
PointNav。

第二个 profile 再加入 `habitat3_spot_human_rearrange`，使用官方 Spot + Human PDDL
rearrangement，验证导航和操作组合。Humanoid 作为 Hey Robot 本体可以作为后续 profile，
不应阻塞第一条链路。

每个 profile 必须固定以下内容，禁止客户端自由传任意 Python config 路径：

- 允许使用的 Habitat config；
- controlled agent 和 NPC agent；
- observation key allowlist；
- semantic skill allowlist；
- 最大 episode/skill steps；
- dataset split 和资产根目录；
- 是否允许 privileged state。

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
  rpc Reset(EpisodeRequest) returns (ObservationResponse);
  rpc CloseEpisode(EpisodeRequest) returns (CloseEpisodeResponse);
}
```

关键字段：

- `CreateEpisodeRequest`: `profile`, `task`, `split`, `episode_id`, `seed`,
  `controlled_agent`；其中 profile/task/split 必须服务端 allowlist；
- `ObservationResponse`: `episode_id`, `frame_id`, repeated image/depth artifacts,
  `proprioception`, `task`, `done`, `success`, `metrics`, `entities`；
- `StepRequest.action`: `google.protobuf.Struct`，保留 Habitat 的嵌套 action dict；
- `ExecuteSkillRequest`: `skill_name`, `arguments`, `expected_frame_id`,
  `max_steps`；
- `SkillResponse`: 最终 observation、steps、success、failure mode、metrics、trace；
- 所有变更状态的请求都带 `expected_frame_id`，拒绝 stale action；
- 一个 episode 同时只允许一个 step/skill，服务端用资源锁保证因果顺序。

`Step` 只用于 contract test、debug 或以后接 policy；正常 Agent 路径走 `ExecuteSkill`。不能把
每个物理 timestep 通过 Agent/消息总线往返，否则延迟、取消和因果一致性都会变差。

## 7. 观测映射

Habitat observation 是按 config 动态生成的 dict。服务端先按 profile 归一化，再交给 Hey
Robot：

| Habitat 数据 | Hey Robot 映射 |
|---|---|
| RGB sensor | `ObservationAsset(kind="image")` |
| Depth sensor | 16-bit PNG artifact；可另生成可视化 image |
| joint/base/localization | `proprioception` 和 `raw.habitat.sensors` |
| PDDL entities/predicates | `metadata.entities` -> `SceneEntity` |
| task measurements | `raw.habitat.metrics` 和 `RobotStatus.metrics` |
| episode/scene/agent roles | `raw.habitat.episode` |

Hey Robot 的 `ObservationPipeline` 已能把 `DriverObservation.metadata.entities` 转成
`SceneEntity`，无需另建场景协议。首版允许用 PDDL/模拟器真值生成 entity 和成功证据，但必须
标记 `privileged: true`；Agent 默认只消费 RGB/depth 与允许的任务传感器，oracle state 只用于
验收和失败诊断，避免把 oracle benchmark 误报为感知能力。

图像 key 必须保留 agent 前缀，例如 `agent_0_articulated_agent_arm_rgb`，否则多智能体下会把
NPC 视角误当成受控角色视角。

## 8. Hey Robot 代码接入点

建议新增：

```text
docker/Dockerfile.habitat3
deploy/habitat3/
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
4. `ObservationPipeline` 不改协议，只使用已有 image/artifact/entity 能力；
5. 注册 Habitat 专用 semantic skills；
6. Compose 增加独立 profile `habitat3`，端口建议 `9093`，避免与 RoboCasa `9092`
   冲突。

首版 Agent-visible skills：

| Skill | 参数 | Habitat 执行器 |
|---|---|---|
| `habitat_navigate_to` | `entity_id` 或受限 `position` | OracleNav/PDDL nav |
| `habitat_follow_human` | `human_id`, `distance_m`, `max_steps` | SocialNav action + success measure |
| `habitat_pick` | `object_id` | PDDL entity resolve + pick skill |
| `habitat_place` | `object_id`, `receptacle_id` | PDDL place skill |
| `habitat_wait` | `steps` | WaitSkillPolicy/zero action |
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
- 仅 SocialNav/Env 所需依赖；不安装 VLA；
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
hab3_bench_assets
```

下载前先做磁盘预算。HSSD 和 episode assets 远大于代码仓库，不能落到 `/var`。Docker
data-root/BuildKit cache 也应继续使用项目所在大盘或已迁移的数据目录。

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

### Phase 3：Embodied Agent 语义技能

- 实现 navigate/follow/wait/stop；
- 再实现 pick/place；
- 每个 skill 返回 steps、PDDL/measure evidence、碰撞和 failure mode；
- 支持中途取消和 timeout；
- Agent 根据一次失败结果继续观察并重规划。

验收：自然语言请求通过 Hey Robot Agent 形成 SkillIntent，完成“找到人并保持距离跟随”、
“走到目标物体并搬到 receptacle”等组合过程，而不是直接运行整集脚本。

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

## 13. 最终建议

Habitat 3 很适合 Hey Robot 的 embodied agent 方向，而且比 RoboCasa 更贴近“人物/角色”：它
原生支持人类 avatar、社交导航、多智能体协作和移动操作。接入难度主要不在算法能力，而在
依赖隔离、动态 action/observation schema、多智能体角色映射和长技能生命周期。

首条实施路线应是：独立容器跑官方 Spot + Human SocialNav，以 oracle/PDDL 作为可控低层
执行器，让 Hey Robot 真正负责对话、任务分解、技能编排和失败恢复。等这条链路稳定后，再接
trained SocialNav/HRL checkpoint；当前没有必要引入 VLA。
