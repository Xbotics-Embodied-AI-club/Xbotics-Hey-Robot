<div align="center">

<pre>
  ██╗  ██╗██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
  ╚██╗██╔╝██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
   ╚███╔╝ ██████╔╝██║   ██║   ██║   ██║██║     ███████╗
   ██╔██╗ ██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
  ██╔╝ ██╗██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
  ╚═╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝╚═════╝╚══════╝
</pre>

<img src="images/hey-robot-icon.png" alt="Hey Robot project icon" width="300" />

<h1>Hey Robot</h1>

<p><em>Embodied Agent Harness · Fast–Slow Dual System · Distributed Model Services</em></p>

<p>
  <a href="https://github.com/Xbotics-Embodied-AI-club/Xbotics-Hey-Robot">GitHub</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#architecture">系统架构</a> ·
  <a href="#community">社区</a> ·
  <a href="../README.md">English</a>
</p>

<p>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-0b7285?style=flat-square" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Harness-Embodied_Agent-6f42c1?style=flat-square" alt="Embodied Agent Harness" />
  <img src="https://img.shields.io/badge/Embodiment-XLeRobot-f59f00?style=flat-square" alt="XLeRobot" />
  <img src="https://img.shields.io/badge/Status-Active_Development-e8590c?style=flat-square" alt="Active Development" />
</p>

</div>

<br />

面向交互式长程机器人任务的开源 Embodied Agent Harness。

Hey Robot 把一个机器人 Agent 拆成两个有清晰边界的系统：

```text
用户 / 环境事件
        ↓
慢系统：Agent、任务连续性、用户纠偏、Skill 选择
        ↓ 结构化 proposal
快系统：Skill、观测、Robot Runtime、安全和驱动
        ↓ 结构化 outcome + 新观测
慢系统继续决策
```

它关注的不是让单个模型一次生成更长的动作序列，而是让机器人在真实约束下能够持续交互、
跨多个 Skill 推进任务、处理失败和用户纠正，并在进程重启后安全地恢复任务事实。

<h2 id="status">当前状态</h2>

Hey Robot 已经从 Harness 基础骨架进入模型和环境集成阶段：

- Agent、Skill、Task、Robot Runtime 和 ModelService 之间的主链已经建立；
- MuJoCo 用于 XLeRobot 仿真闭环；
- InternNav 已通过独立 VLN ModelService 接入 XLeRobot MuJoCo 仿真；
- LeRobot policy 已通过统一 ModelService 接入 RoboCasa365，并完成完整系统链路验证；
- XLeRobot native driver、底盘、机械臂、相机和 Robot Runtime 已具备真机部署基础；
- 下一阶段是 InternNav 与 LeRobot policy 的 XLeRobot 真机闭环、标定和安全验证。

下一阶段将把已经接入的导航和操作 policy 迁移到 XLeRobot 真机，验证真实观测、动作和安全闭环。

<p align="center">
  <img src="images/architecture.png"
       alt="Hey Robot Embodied Agent Harness architecture"
       width="100%" />
</p>

<p align="center"><sub>分布式模型服务、快慢双系统与单一物理执行主链。</sub></p>

<h2 id="why">为什么需要 Harness</h2>

普通的 LLM tool loop 可以选择下一次调用，但机器人还需要解决：

- 观测会过期，动作会改变世界状态；
- 物理动作共享底盘、机械臂、相机和安全资源；
- 用户可能在动作执行期间纠正目标；
- Skill 可能超时、失败、取消或失去执行归属；
- 进程可能在任务完成前重启；
- “调用返回”不等于“物理目标完成”。

Hey Robot 将这些问题放在模型之外的 Harness 层处理。模型只提出一个受 schema 约束的
proposal；Skill 和 Robot Runtime 负责有界执行、资源互斥、取消、安全检查、结果和观测。

<h2>核心设计</h2>

### 一个 Agent，一套任务事实，一条物理动作路径

当前系统坚持以下不变量：

1. 一个 deployment 只启用一个 autonomous Agent；
2. 一个 session 最多拥有一个未终止任务；
3. Agent 一次最多提出一个物理 proposal；
4. Skill submission 先持久化，再提交执行；
5. terminal Skill event 以幂等方式更新 task step，并唤醒 Agent；
6. 未知的物理执行结果不会被自动重放；
7. emergency stop 绕过 Agent 推理，走确定性控制路径。

### 配置驱动，但配置不承载业务状态

部署配置选择 Channel、Robot、ModelService、Skill surface、总线和资源路径。配置不保存
任务进度、机器人状态，也不实现隐藏的工作流语言。运行事实由 ConversationStore、
AgentTaskStore、RunStore 和 Robot Runtime 分别维护。

### 可替换的模型和机器人边界

同一个 Skill surface 可以包装经典控制、InternNav、LeRobot policy 或其他独立模型服务。
模型服务不拥有任务生命周期，Agent 不直接访问驱动，Robot Runtime 不依赖上层 Agent。
Agent 主系统与模型服务通过 typed ModelService/gRPC contract 分割，可跨进程、跨环境和跨
GPU 部署。

<h2 id="architecture">分布式 Embodied Agent Harness · 快慢双系统</h2>

Hey Robot 的 Agent 主系统与 InternNav、VLA、VLN 和 LeRobot policy 等模型服务保持服务
分割；模型可独立进程、独立依赖环境和独立 GPU 运行。“快慢”描述决策时间尺度，不表示
Python、NATS 或 gRPC 提供硬实时保证。慢系统维持目标、交互和任务连续性；快系统把一个
有界能力安全地落实到模型、仿真或硬件。

<table>
  <thead><tr><th></th><th>慢系统 · Deliberative</th><th>快系统 · Embodied Execution</th></tr></thead>
  <tbody>
    <tr><td>时间范围</td><td>跨轮次、跨 Skill、跨服务重启</td><td>一次有界 Skill 与局部控制过程</td></tr>
    <tr><td>负责</td><td>理解目标、选择 Tool、任务推进、暂停与恢复</td><td>感知、资源门控、模型推理、安全检查和机器人执行</td></tr>
    <tr><td>当前实现</td><td><code>Agent</code>、<code>AgentRunner</code>、<code>AgentTaskStore</code>、<code>TaskCoordinator</code></td><td><code>SkillWorker</code>、VLA/VLN option、<code>RobotRuntime</code>、Robot Driver</td></tr>
  </tbody>
</table>

<h2 id="capability-status">已验证和进行中的能力</h2>

<table>
  <thead><tr><th>能力</th><th>状态</th><th>边界</th></tr></thead>
  <tbody>
    <tr><td>Agent Tool loop</td><td>已实现</td><td>一次最多一个 proposal，错误结构化拒绝</td></tr>
    <tr><td>持久任务</td><td>已实现</td><td>SQLite task/step、续跑、暂停、取消、启动恢复</td></tr>
    <tr><td>Skill Harness</td><td>已实现</td><td>schema、资源、timeout、cancel、事件、RunStore</td></tr>
    <tr><td>XLeRobot MuJoCo</td><td>已接入</td><td>用于仿真驱动、观测和机器人能力闭环</td></tr>
    <tr><td>InternNav XLeRobot 仿真</td><td>已接入并验证链路</td><td>独立 VLN ModelService、observe-plan-act、移动动作映射；真机待验证</td></tr>
    <tr><td>LeRobot policy ModelService</td><td>已接入</td><td>独立 policy 进程、observation/action mapping、统一 gRPC contract</td></tr>
    <tr><td>RoboCasa365</td><td>完整系统链路已验证</td><td>LeRobot policy、ModelService、Robot Runtime 与环境端到端贯通</td></tr>
    <tr><td>XLeRobot native driver</td><td>已接入</td><td>仍需逐机标定、诊断、动作范围和物理安全验证</td></tr>
    <tr><td>XLeRobot 真机 InternNav / LeRobot</td><td>下一阶段</td><td>需要真实观测、动作空间、取消、超时和安全闭环</td></tr>
  </tbody>
</table>

<h2 id="quick-start">快速开始</h2>

推荐环境：Ubuntu/Linux、Python 3.12 和 [uv](https://docs.astral.sh/uv/)。裸
<code>uv sync</code> 不包含完整的 Gateway、Agent、Robot 和 MuJoCo 依赖，请使用下面的
profile 安装命令。

```bash
git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-Hey-Robot.git
cd Xbotics-Hey-Robot

uv sync --extra gateway --extra agent --extra robot --group sim --group dev
cp .env.example .env

uv run hey-robot inspect --config configs/xlerobot.sim.ubuntu.yaml
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

InternNav 仿真需要独立模型环境和 InternNav submodule，具体步骤见
[`operations/xlerobot-sim.md`](operations/xlerobot-sim.md)。

RoboCasa365 完整系统评测见
[`evaluation/robocasa365/runbook.zh-CN.md`](evaluation/robocasa365/runbook.zh-CN.md)。

<h2 id="real-robot">XLeRobot 真机</h2>

默认真机 profile 只开放安全的场景检查和底盘基础动作。连接硬件前先完成平台、串口、舵机、
相机、电池和急停检查：

```bash
uv run python scripts/ops/check_platform.py \
  --config configs/xlerobot.real.ubuntu.yaml
uv run hey-robot inspect --config configs/xlerobot.real.ubuntu.yaml
uv run python scripts/robots/xlerobot/diagnose.py \
  --config configs/xlerobot.real.ubuntu.yaml
```

InternNav 和 LeRobot policy 已有统一接入路径，但不要直接把仿真或 RoboCasa365 配置用于真机。
真机 profile 必须单独验证：

- camera 和 observation mapping；
- action dimensions、范围和频率；
- 标定、home/rest position 和资源互斥；
- timeout、cancel、emergency stop；
- 空载、低速和受控场景下的动作闭环。

完整流程见 [`operations/xlerobot-real.md`](operations/xlerobot-real.md)。

<h2 id="safety">安全边界</h2>

- 所有运动先在 MuJoCo 中验证，再连接真实机器人；
- 真机测试必须保持物理急停或断电手段可用；
- 面向目标机器人验证模型观测、动作和安全设置；
- 机械臂、底盘和 VLA/VLN 的权限必须通过 `skills.tools` 显式开放。

<h2 id="code-structure">代码结构</h2>

<table>
  <thead><tr><th>路径</th><th>职责</th></tr></thead>
  <tbody>
    <tr><td><code>src/hey_robot/cognition</code></td><td>Agent、任务状态、对话上下文和 Tool loop</td></tr>
    <tr><td><code>src/hey_robot/skills</code></td><td>Skill schema、worker、option runner 和结果 contract</td></tr>
    <tr><td><code>src/hey_robot/robot_runtime</code></td><td>资源、安全、观测和机器人执行边界</td></tr>
    <tr><td><code>src/hey_robot/robot_backends</code></td><td>MuJoCo、XLeRobot 和 RoboCasa 环境适配</td></tr>
    <tr><td><code>src/hey_robot/foundation</code></td><td>VLA/VLN/LeRobot ModelService contract 和 backend</td></tr>
    <tr><td><code>src/hey_robot/config</code></td><td>typed deployment config 和启动校验</td></tr>
    <tr><td><code>configs/</code></td><td>deployment、仿真、真机和评测 profile</td></tr>
    <tr><td><code>tests/</code></td><td>contract、架构边界、组件和集成测试</td></tr>
  </tbody>
</table>

<h2 id="documentation">文档入口</h2>

- [文档索引](index.md)
- [系统架构](architecture/system-architecture.md)
- [配置参考](reference/configuration.md)
- [XLeRobot 仿真](operations/xlerobot-sim.md)
- [XLeRobot 真机](operations/xlerobot-real.md)
- [RoboCasa365 评测](evaluation/robocasa365/runbook.zh-CN.md)
- [最小 Embodied Agent Harness 开发指南](development/minimal-embodied-agent-harness-guide.zh-CN.md)
- [论文初稿](references/paper-draft.md)

<h2 id="development">开发检查</h2>

```bash
uv run poe style
uv run poe lint
uv run poe test
```

<h2 id="community">社区与贡献</h2>

Hey Robot 来自 XLeRobot 开源机器人实践。欢迎通过 Issue、Pull Request 或社区渠道参与
Harness、机器人驱动、交互体验与具身模型集成。

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="images/xbotics-wechat-official-account.png" alt="Xbotics 微信公众号" width="150" />
        <br /><sub>Xbotics 公众号</sub>
      </td>
      <td align="center">
        <img src="images/developer-wechat.jpg" alt="开发者微信" width="110" />
        <br /><sub>开发者微信</sub>
      </td>
    </tr>
  </table>
</div>

贡献前请阅读 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 和
[`development/skill-extension.md`](development/skill-extension.md)。

<p align="center">
  <a href="https://github.com/Vector-Wangel/XLeRobot">XLeRobot</a> ·
  <a href="references/project-references.md">项目活动与参考</a> ·
  <a href="../LICENSE">MIT License</a>
</p>

<h2 id="license">许可证</h2>

本项目采用 [MIT License](../LICENSE)。论文、第三方模型和参考材料的许可范围以各自文件和
上游项目为准。
