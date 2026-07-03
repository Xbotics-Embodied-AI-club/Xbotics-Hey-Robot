<div align="center">
  <img src="docs/images/xbotics-logo.jpg" alt="Xbotics 社区 Logo" width="140" />
</div>

# Hey Robot

<div align="center">
  <sub>简体中文 | <a href="docs/README_EN.md">English</a></sub>
</div>

Hey Robot 是一个不依赖通用 LLM Agent 框架、面向真实机器人原生构建的
Embodied Agent Harness。它采用异步快慢双系统架构：上层 LLM Agent 负责长程认知与
任务规划，下层 VLA/VLN、Skill OS 和 Robot Runtime 负责短时域具身决策与执行。

项目目前以 XLeRobot 为主要载体，支持 MuJoCo 仿真和真机部署。这里的“原生”是指
核心 Agent Runtime、tool protocol、任务状态与执行闭环均围绕物理世界自主实现，而不是
将通用聊天 Agent 框架改造成机器人控制器。

> 项目状态：当前处于 active development。XLeRobot 仿真和真机链路均已跑通，仍建议任何硬件改动先在仿真中验证。

<p align="center">
  <img src="docs/images/xlerobot.png" alt="XLeRobot" width="360" />
</p>

## 特性

- 自主实现的 Embodied Agent Harness，不依赖通用 LLM Agent 编排框架。
- 异步快慢双系统：上层长程认知与下层短时域具身执行解耦。
- Skill 层抽象：Agent 只请求机器人能力，不直接控制硬件。
- 支持 MuJoCo 仿真和 XLeRobot 真机部署。
- 支持 Web、CLI、语音、飞书等用户入口。
- 内置 tasks UI：展示任务状态、timeline、scene evidence 和 recovery。
- VLA/VLN 能力通过独立 ModelService 接入，不塞进 robot driver。
- 支持 execution feedback、resource gate、readiness gate、timeout 和恢复流程。

## 架构

快慢双系统描述的是决策层级：

- **慢系统**：上层 Agent/Cognition，负责语言理解、目标分解、记忆、长程规划和恢复。
- **快系统**：下层 Foundation Model、Skill OS 和 Robot Runtime，负责 VLA/VLN
  推理、短时域技能闭环、安全门控和机器人执行。

这里的“快”表示更靠近具身控制、决策周期更短，不承诺模型推理或 Python 执行路径具备
硬实时性能。四层架构则描述这套双系统在代码和部署中的具体边界：

```mermaid
flowchart TD
    U[用户入口<br/>Web / 语音 / 飞书 / CLI]
    G[Gateway + NATS<br/>身份 / Episode / 消息路由]
    A[Agent 层<br/>任务 / 记忆 / 主动感知 / Tool loop]
    S[Skill OS<br/>合约 / 调度 / 资源门禁]
    F[Foundation Model 层<br/>VLA / VLN / gRPC]
    R[Robot Runtime<br/>MuJoCo / 真机]

    U -->|用户指令| G
    G -->|UserTurn| A
    A -->|能力请求| S
    S -->|机器人动作| R
    S -->|模型请求| F
    F -->|规划或动作结果| S
    R -->|状态 / 观察| A
    A -->|回复| G
    G --> U
```

核心边界：

- Gateway 负责用户入口、身份归一化、Episode 分配和消息路由，不参与机器人决策。
- `Robot` 只表示身体和硬件执行边界。
- `Skill` 是 Agent 调用机器人能力的统一入口。
- Agent 通过 `request_skill` 调用机器人 skill，不直接提交 `RobotAction`。
- `RobotService / RobotRuntime / PerceptionService` 负责 observation 与相机帧发布。
- VLA/VLN 通过独立 ModelService 接入；模型服务只返回规划或动作结果，闭环和硬件执行仍由 Skill OS 与 Robot Runtime 负责。

`hey-robot run` 默认把 Gateway、Agent、Task Supervisor、Skill Controller 和 Robot Service
作为同一 asyncio 进程中的独立服务启动；这些服务仍通过 NATS 协议通信。ModelService 和
NATS broker 是独立进程，也可以使用各自 CLI 将主服务进一步拆分部署。

## 快速开始

### 1. 环境要求

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- NATS server
- MuJoCo，用于 XLeRobot 仿真
- XLeRobot 真机硬件，用于真实机器人部署

### 2. 安装依赖

```bash
uv sync --dev
```

如果需要仿真：

```bash
uv sync --dev --group sim
```

如需启动 VLA/VLN 模型服务，使用独立环境：

```bash
uv sync --group vla    # VLA 模型服务（LeRobot ACT）
uv sync --group vln    # VLN 模型服务（InternVLA-N1）
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

根据使用的 provider 填写 `.env`。常用变量：

```text
DEEPSEEK_API_KEY
DEEPSEEK_MODEL
DASHSCOPE_API_KEY
DASHSCOPE_MODEL
ARK_API_KEY
```

### 4. 启动 NATS

```bash
nats-server
```

也可以只用 Docker 启动 NATS：

```bash
docker compose up nats
```

### 5. 运行 XLeRobot 仿真

```bash
uv run hey-robot run --config configs/xlerobot.sim.windows.yaml
```

Ubuntu 仿真配置：

```bash
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

默认 Web 入口：

```text
http://127.0.0.1:8080
```

仿真环境适合验证 Agent、Skill、Web tasks UI 和机器人执行链路，不需要真实机器人硬件。

## XLeRobot 真机

真机部署前先检查平台和硬件映射：

```powershell
uv run python scripts\ops\check_platform.py --config configs\xlerobot.real.windows.yaml
uv run hey-robot inspect --config configs\xlerobot.real.windows.yaml
uv run python scripts\robots\xlerobot\diagnose.py --config configs\xlerobot.real.windows.yaml
```

启动完整系统：

```powershell
uv run hey-robot run --config configs\xlerobot.real.windows.yaml
```

当前 XLeRobot real/sim 配置默认启用 11 个非 VLA skill：

```text
inspect_scene
look_around
detect_marker
move_base
turn_base
human_follow
stop_motion
reset_posture
set_arm_pose
move_arm_joints
set_gripper
```

VLA 能力作为可选扩展能力接入，建议在 ModelService 稳定后再开放给 Agent 使用。

实验配置 `configs/xlerobot.sim.vla_vln.yaml` 额外开放 VLN 与 VLA skill。它不代表默认
真机能力面：其中 VLN 需要单独初始化 InternNav 和模型，VLA 尚未配置真实 `model_path`，
当前只能用于接口联调。使用前请阅读仿真部署文档中的限制说明。

## 安全提示

这个项目可以向真实机器人硬件下发运动命令。请遵守以下原则：

- 先在仿真环境验证，再接入真实机器人硬件。
- 真机运行时保持急停或断电手段可用。
- 不要在人员、宠物、易碎物或不稳定环境附近直接测试 motion skill。
- 修改硬件、串口、舵机 ID、相机编号后，先运行诊断脚本。
- VLA / foundation model 能力必须经过单独验证后再暴露给 Agent。

## 常用命令

```bash
# 格式化和自动修复
poe style

# 配置校验、ruff、mypy
poe lint

# 测试
poe test
```

也可以直接运行：

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q --no-cov
```

## 常用配置

- `configs/xlerobot.real.windows.yaml`：XLeRobot Windows 真机
- `configs/xlerobot.sim.windows.yaml`：XLeRobot Windows 仿真
- `configs/xlerobot.sim.ubuntu.yaml`：XLeRobot Ubuntu 仿真

## 后续计划

项目接下来主要聚焦两件事：更好的机器人交互体验，以及半开放环境中的长程任务能力。

- Agent：增强 memory、plan 和多轮纠偏能力。
- Skill：接入 VLA、VLN、WAM 等基础模型能力。
- 系统可靠性：完善执行反馈、失败恢复和任务状态追踪。

## 目录结构

```text
configs/                    部署配置文件
docs/                       架构、部署、开发文档
frontend/views/             Web 前端页面
frontend/shared/            Web 前端公共样式和脚本
proto/                      ModelService protobuf 协议源文件
src/hey_robot/cognition/    Agentic cognition、主循环、核心决策、任务状态
src/hey_robot/skill_os/     Skill 注册、合约、调度控制器和内置技能
src/hey_robot/foundation/   VLA/VLN ModelService、catalog 与 gRPC transport
src/hey_robot/robot_runtime/ Robot runtime 和机器人驱动
src/hey_robot/robot_runtime/observations/  观察流水线和运行时感知快照
src/hey_robot/cognition/perception/        场景理解
src/hey_robot/channels/     CLI / Web / 语音 / 飞书通道
tests/                      单元测试和集成测试
```

## 文档

- [部署与运行形态](docs/overview/runtime-shape.md)
- [系统架构](docs/architecture/system-architecture.md)
- [Agent 与 Skill 边界](docs/architecture/agent-skill-boundaries.md)
- [ModelService RPC 协议](docs/architecture/capability-rpc-proto.md)
- [部署矩阵](docs/operations/deployment-matrix.md)
- [XLeRobot 真机部署](docs/operations/xlerobot-real.md)
- [XLeRobot 仿真部署](docs/operations/xlerobot-sim.md)
- [运行脚本索引](docs/operations/runtime-scripts.md)
- [Skill 扩展指南](docs/development/skill-extension.md)
- [贡献指南](docs/development/contributing.md)

## 活动与参考

本项目来自开源机器人 XLeRobot 动手实战工作坊相关实践。

- 工作坊记录：[开源机器人 XLeRobot 动手实战工作坊](https://mp.weixin.qq.com/s/TahLTjvvP9MoisCOCVkEBA)
- 架构设计参考：[逐际动力相关架构文章](https://www.limxdynamics.com/zh/news/BK000054)
- Agent 设计参考：[HKUDS/nanobot](https://github.com/HKUDS/nanobot)
- XLeRobot 官方仓库：[Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
- 功能参考：[choco-robot/HomeBot](https://github.com/choco-robot/HomeBot)
- 更多链接：[项目活动与参考资料](docs/references/project-references.md)

## 社区

关注公众号或联系开发者：

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="docs/images/xbotics-wechat-official-account.png" alt="Xbotics 微信公众号" width="150" />
        <br />
        <sub>Xbotics 公众号</sub>
      </td>
      <td align="center">
        <img src="docs/images/developer-wechat.jpg" alt="开发者微信" width="110" />
        <br />
        <sub>开发者微信</sub>
      </td>
    </tr>
  </table>
</div>

## 贡献

欢迎提交 issue 和 pull request。新增 skill 或硬件能力前，建议先阅读：

- [贡献指南](docs/development/contributing.md)
- [Skill 扩展指南](docs/development/skill-extension.md)
- [Agent 与 Skill 边界](docs/architecture/agent-skill-boundaries.md)

提交前请运行：

```bash
poe style
poe lint
poe test
```

## License

MIT License. See [LICENSE](LICENSE).
