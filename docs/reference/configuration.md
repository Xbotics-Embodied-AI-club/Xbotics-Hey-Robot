# 部署配置参考

本文描述当前 `DeploymentConfig` 接受的核心 YAML 结构。字段事实源是
`src/hey_robot/config/model.py`，跨字段约束事实源是
`src/hey_robot/config/validation.py`。具体硬件、通道和模型后端的 `settings` 仍应参考对应
运行手册和仓库中的 profile。

修改配置后必须先运行：

```bash
uv run hey-robot inspect --config <path-to-config.yaml>
uv run hey-robot inspect skill-surface --config <path-to-config.yaml>
```

第一条检查引用、目录、Skill、driver primitive 和 ModelService 约束；第二条输出实际暴露给
Agent 的 Tool schema。`inspect` 成功只表示静态配置有效，不代表硬件、网络、凭据、模型权重
或任务效果已通过运行验证。

## 顶层结构

```yaml
deployment: {}
logging: {}
resources: {}
identity: {}
channels: {}
robots: {}
policies: {}
model_services: {}
agents: {}
skills: {}
agent_runtime: {}
```

| 区域 | 作用 | 关键约束 |
|---|---|---|
| `deployment` | deployment ID 和消息总线 | bus 仅支持实际实现的 `in_memory` 或 `nats` |
| `logging` | 文本/JSON 日志 | level 会转为大写 |
| `resources` | runtime、media、episode、event 路径 | `inspect` 会尝试创建目录 |
| `identity` | 跨通道用户绑定 | binding key/value 不能为空 |
| `channels` | CLI、Web、Voice、Feishu | 各通道专有字段进入 settings |
| `robots` | robot family、environment、driver | 未显式填写时会从 `type` 推断 |
| `policies` | Agent 到 robot 的关联 | `robot_id` 必须存在 |
| `model_services` | VLA/VLN gRPC 服务 | capability 由 `provides` 声明 |
| `agents` | Agent、robot、policy 和模型配置 | 当前一个 deployment 最多一个 enabled Agent |
| `skills` | registry 和 Agent Tool allowlist | 仅支持 local execution |
| `agent_runtime` | 持续任务上限 | 不接受已删除的旧 autonomy 字段 |

## deployment 与消息总线

```yaml
deployment:
  id: xlerobot.sim.ubuntu
  bus:
    type: in_memory
    url: memory://xlerobot.sim.ubuntu
    options:
      robot_publish_hz: 2.0
```

- `in_memory`：同一进程内通信，不需要 broker，不能连接外部 consumer。
- `nats`：需要可访问的 NATS server；默认 URL 是 `nats://127.0.0.1:4222`。
- NATS core publish/subscribe 默认不提供持久化；只有显式启用并使用 JetStream 的路径才有
  broker 侧持久化语义。

当前 Agent、Skill Worker 和 Robot Runtime 仍在同一 `hey-robot run` 进程内组合。选择
NATS 不会把 Skill 或 Robot 主执行链自动变成远程服务。

## resources

```yaml
resources:
  runtime_dir: runtime/example
  media:
    root: runtime/example/media
    max_items: 5000
    image_save_every_n: 1
  episodes:
    root: runtime/example/episodes
  events:
    retain: 1000
```

`runtime_dir` 是该 deployment 的最终运行根目录，运行时不会再次追加
`deployment.id`；省略时默认为 `runtime/<deployment.id>`。路径相对于进程工作目录解析。
生产部署应使用持久卷或明确的绝对路径，并为运行用户设置最小读写权限。不要把包含用户对话、
图像或机器人现场信息的 `runtime/` 产物提交到仓库。

## channels

```yaml
channels:
  web:
    type: web
    enabled: true
    account_id: sim-web
    host: 127.0.0.1
    port: 8080
```

所有 Channel 共享 `type`、`enabled` 和可选 `account_id`；其余字段原样进入该 Channel 的
settings。Web、Voice 和 Feishu 的字段分别由对应实现解释。公网暴露 Web 前必须另行配置
认证、TLS、反向代理和网络访问控制；当前示例 profile 不应直接作为公网生产配置。

## robots、policies 与 agents

```yaml
robots:
  sim_robot:
    type: xlerobot_sim
    family: xlerobot
    environment: sim
    driver: mujoco
    embodiment_profile: xlerobot_sim
    enabled: true
    settings: {}

policies:
  embodied_skills:
    robot_id: sim_robot
    enabled: true
    freq_hz: 1.0

agents:
  main:
    robot_id: sim_robot
    policy_id: embodied_skills
    enabled: true
    settings:
      models: {}
```

`RobotSpec` 的默认推断规则：

- `type: mock` -> family 来自 `settings.body`/`settings.embodiment_type`，driver 为 `mock`；
- 以 `_sim` 结尾 -> 去掉后缀作为 family，environment 为 `sim`，driver 为 `mujoco`；
- 其他 type -> environment 为 `real`，driver 为 `native`。

生产 profile 建议显式填写 `family`、`environment` 和 `driver`，避免重命名 `type` 时改变
推断结果。Agent 引用的 robot/policy 必须存在；当前校验拒绝多个 enabled Agent。

## skills：真实的 Agent 能力面

```yaml
skills:
  modules:
    - hey_robot.skills.builtins
  mode: bringup
  tools:
    - inspect_scene
    - move_base
    - turn_base
  implementations: {}
  execution_mode: local
```

- `modules` 只接受 `hey_robot.skills.*` 下的 native module；
- `tools` 是必须显式填写的 Agent Tool allowlist；
- `implementations` 的 key 必须也出现在 `tools`；
- `mode` 只接受 `production` 或 `bringup`，目前不会自动过滤能力；
- `execution_mode` 当前只接受 `local`。

因此 registry 中存在 Skill、ModelService 声明 capability、或 `mode: production`，都不等于
Agent 可以调用该能力。发布前应审查 `inspect skill-surface` 的实际输出。

## model_services

```yaml
model_services:
  vln_nav:
    type: vln_planner
    robot_id: sim_robot
    enabled: true
    target: grpc://127.0.0.1:9091
    provides:
      - navigate_to
      - approach_object
    timeout_sec: 60
    settings: {}
```

通用字段为 `type`、`robot_id`、`enabled`、`target`、`provides`、`timeout_sec` 和
`settings`。VLA/VLN 的必填 settings 由配置校验器按 type 检查：

- `vln_planner` 当前使用 `backend: internvla_n1_dualvln`。执行模式为
  `control_mode: base_action_chunk`，并要求 `base_linear_speed`、
  `base_angular_speed`、`max_action_chunk_steps`、
  `system1_replans_per_waypoint`、`discrete_forward_cm` 和
  `discrete_turn_deg`。离散动作必须能在 Robot Runtime 的 1000ms 速度安全窗口内
  完成；非 mock 模式还需要 `model_path`、`internnav_repo`、`media_root`，且
  `model_path` 必须指向包含 System 1 权重的 DualVLN checkpoint。
- `robot_policy` 当前只支持 `runtime: lerobot`，并要求 `policy_path`、
  `policy_device`、`action_space`、正数 `action_dimensions`。

`provides` 必须与 Skill 的 `required_models` 匹配。模型服务通常由
`hey-robot model-service` 独立启动；`hey-robot run` 不会因配置中存在普通
ModelService entry 就自动启动它。

## agent_runtime

```yaml
agent_runtime:
  enabled: true
  robot_id: sim_robot
  hard_max_wall_time_sec: 3600
  hard_max_skills: 24
```

这是任务持续时间与 Skill 数量的硬上限。旧字段 `task_runtime`、`autonomy` 以及旧
`agent_runtime` 的 goal/retry/resume 类字段已经删除，解析时会直接报错。

## 环境变量与秘密

YAML 保存环境变量名，`.env` 保存本地值。模板见 `.env.example`。默认示例使用：

```text
DEEPSEEK_MODEL
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DASHSCOPE_MODEL
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
ARK_API_KEY
FEISHU_APP_ID
FEISHU_APP_SECRET
FEISHU_ENCRYPT_KEY
FEISHU_VERIFICATION_TOKEN
```

不要把真实 secret 写入 YAML、文档、Issue、日志或评测 artifact。生产环境应由 secret
manager 或编排平台注入；`.env` 只适合本地开发。

## 兼容性和变更

配置当前没有独立的 schema version。重命名或删除字段会由解析器直接拒绝，但新旧进程之间
没有自动迁移协议。变更公开配置时应：

1. 同时修改 dataclass、解析、校验、示例 profile 和测试；
2. 更新本文与相关运行手册；
3. 在 release notes 中列出破坏性变化和迁移示例；
4. 对生产 profile 运行 `inspect`、对应诊断和最小端到端 smoke test。
