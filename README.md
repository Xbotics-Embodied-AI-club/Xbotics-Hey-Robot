<div align="center">

  <pre>
  ██╗  ██╗██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
  ╚██╗██╔╝██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
   ╚███╔╝ ██████╔╝██║   ██║   ██║   ██║██║     ███████╗
   ██╔██╗ ██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
  ██╔╝ ██╗██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
  ╚═╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
  </pre>

<img src="docs/images/hey-robot-icon.png" alt="Hey Robot project icon" width="300" />

<h1>Hey Robot</h1>

<p>
  <em>面向真实机器人的 Embodied Agent Harness · 快慢双系统 · 分布式协同架构</em>
</p>

<p><strong>让机器人在持续交互中完成长程任务。</strong></p>

<p>
  一个面向真实机器人的开源 <strong>Embodied Agent Harness</strong>：<br />
  用慢系统理解目标、维持任务和处理修正，用快系统执行受控 Skill、感知环境并驱动机器人。
</p>

<p>
  <a href="#why-hey-robot">为什么是 Hey Robot</a> ·
  <a href="#architecture">快慢双系统</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#capability-status">能力状态</a> ·
  <a href="docs/index.md">文档</a> ·
  <a href="docs/references/paper-draft.md">论文草稿</a> ·
  <a href="docs/README_EN.md">English</a>
</p>

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0b7285?style=flat-square" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Harness-Embodied_Agent-6f42c1?style=flat-square" alt="Embodied Agent Harness" />
  <img src="https://img.shields.io/badge/Embodiment-XLeRobot-f59f00?style=flat-square" alt="XLeRobot" />
  <img src="https://img.shields.io/badge/Status-Active_Development-e8590c?style=flat-square" alt="Active Development" />
</p>

</div>

<br />

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>💬 持续交互</h3>
      <p>在机器人执行过程中继续追问、纠正、暂停、恢复或急停，而不是等待一次黑盒调用结束。</p>
    </td>
    <td width="33%" valign="top">
      <h3>🧭 长程任务</h3>
      <p>将目标、步骤、运行结果和恢复点持久化；Skill 完成后由事件驱动 Agent 继续推进下一步。</p>
    </td>
    <td width="33%" valign="top">
      <h3>🧩 Agent Harness</h3>
      <p>把模型推理、能力边界、执行安全、机器人驱动、观测和运维界面组织成可替换、可审计的系统。</p>
    </td>
  </tr>
</table>

<blockquote>
  <strong>项目状态：</strong>Harness 主链、MuJoCo、XLeRobot 驱动、多通道交互和持久任务机制
  已进入持续开发；VLA/VLN 与复杂真机长程任务仍属于实验能力。任何运动都应先在仿真中验证。
</blockquote>

<h2 id="why-hey-robot">为什么是 Hey Robot</h2>

<p>
普通的「LLM + Tool Calling」可以决定下一次调用，却不会自然解决真实机器人的状态问题：
观测会过期，硬件会忙碌，动作不可随意回滚，用户会在执行中改变要求，进程也可能在任务完成前重启。
</p>

<p>Hey Robot 将这些问题放进 Harness，而不是全部交给 prompt：</p>

<table>
  <thead>
    <tr>
      <th>真实机器人问题</th>
      <th>Hey Robot 的系统机制</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>任务跨越多次模型调用和物理动作</td>
      <td>持久化 Task、step/run 关联、自动续跑和启动恢复</td>
    </tr>
    <tr>
      <td>用户在执行中追问或改变要求</td>
      <td>Web / CLI / Voice / Feishu 统一会话、暂停、恢复、取消和急停</td>
    </tr>
    <tr>
      <td>模型可能提出不可执行的动作</td>
      <td>显式 Skill surface、参数校验、资源互斥、超时和 readiness gate</td>
    </tr>
    <tr>
      <td>模型、仿真与硬件迭代速度不同</td>
      <td>Agent、Skill、ModelService、Robot Runtime 和 Driver 分层</td>
    </tr>
    <tr>
      <td>“调用成功”不等于“任务完成”</td>
      <td>Skill lifecycle、Robot observation/status、任务时间线和恢复上下文</td>
    </tr>
  </tbody>
</table>

<h2 id="interaction">交互式长程任务</h2>

<p>
Hey Robot 把长程任务定义为一个可持续推进的状态对象，而不是一段不断增长的聊天历史。
当前实现维持每个会话最多一个未终止任务，并记录目标、步骤、Skill run、结果、暂停状态和恢复位置。
</p>

```text
用户目标
  ↓
Agent 推理并选择一个可见 Tool / Skill
  ↓
Skill 异步执行，结果持久化
  ↓
terminal event 唤醒 Agent
  ↓
结合任务历史与最新结果继续下一步
  ↓
完成 / 暂停等待用户 / 取消 / 失败
```

<p>同一个任务执行期间，用户仍然可以：</p>

<ul>
  <li>询问“现在进行到哪一步了？”</li>
  <li>发送“先停一下”，随后继续任务；</li>
  <li>纠正目标或提供缺失信息；</li>
  <li>通过确定性控制路径触发急停；</li>
  <li>在服务重启后恢复尚未完成的任务。</li>
</ul>

<h2 id="architecture">Embodied Agent Harness：快慢双系统</h2>

<p>
“快慢”描述决策时间尺度，不表示 Python、NATS 或 gRPC 提供硬实时保证。
慢系统维持目标与语义决策；快系统把单个有界能力安全地落实到模型、仿真或硬件。
</p>

<table>
  <thead>
    <tr>
      <th></th>
      <th>慢系统 · Deliberative</th>
      <th>快系统 · Embodied Execution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>时间范围</strong></td>
      <td>跨轮次、跨 Skill、跨服务重启</td>
      <td>一次有界 Skill 与局部控制过程</td>
    </tr>
    <tr>
      <td><strong>负责</strong></td>
      <td>理解目标、选择 Tool、任务推进、暂停与恢复</td>
      <td>感知、资源门控、局部模型推理、安全检查和机器人执行</td>
    </tr>
    <tr>
      <td><strong>当前实现</strong></td>
      <td><code>Agent</code>、<code>AgentRunner</code>、<code>AgentTaskStore</code>、<code>TaskCoordinator</code></td>
      <td><code>SkillWorker</code>、VLA/VLN option、<code>RobotRuntime</code>、Robot Driver</td>
    </tr>
  </tbody>
</table>

<p align="center">
  <img src="docs/images/architecture.png"
       alt="Hey Robot Embodied Agent Harness architecture"
       width="100%" />
</p>

<p align="center">
  <sub>用户可以在 Skill 运行期间持续交互；terminal event 在安全边界恢复慢系统，物理动作始终经过单一执行主链。</sub>
</p>

<p>
默认部署将 Agent、Skill Worker 与 Robot Runtime 组合在同一个 asyncio 进程；
VLA/VLN 可以作为独立 gRPC ModelService 部署。物理能力只通过
<code>SkillClient → LocalRobotClient → RobotRuntime</code> 主链提交。
</p>

<p>
深入设计见
<a href="docs/architecture/system-architecture.md">系统架构</a>，
研究动机、主张边界和实验计划见
<a href="docs/references/paper-draft.md">论文草稿</a>。
</p>

<h2 id="capability-status">能力状态</h2>

<table>
  <thead>
    <tr>
      <th>能力</th>
      <th>状态</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>交互通道</td>
      <td>✅ 已实现</td>
      <td>Web、CLI、Voice、Feishu，共享 Gateway / identity / episode 边界</td>
    </tr>
    <tr>
      <td>持久长程任务</td>
      <td>✅ 已实现</td>
      <td>SQLite task/step 状态、异步续跑、暂停/继续、取消与启动恢复</td>
    </tr>
    <tr>
      <td>Skill Harness</td>
      <td>✅ 已实现</td>
      <td>显式 Tool surface、校验、资源、超时、取消、事件和 FileRunStore</td>
    </tr>
    <tr>
      <td>MuJoCo / Mock / 真机 Driver</td>
      <td>✅ 已接入</td>
      <td>共享 Robot Runtime 边界；真机使用前仍需逐机标定与诊断</td>
    </tr>
    <tr>
      <td>VLA / VLN ModelService</td>
      <td>🧪 实验中</td>
      <td>具备 gRPC contract、路由和实验 profile；权重与闭环需单独验证</td>
    </tr>
    <tr>
      <td>开放世界复杂真机长程任务</td>
      <td>🗺️ 研究目标</td>
      <td>尚不能从 Harness 测试外推任务成功率，不作为当前能力承诺</td>
    </tr>
  </tbody>
</table>

<h2 id="quick-start">快速开始</h2>

<h3>1 · 安装</h3>

<p>
推荐 Ubuntu / Linux、Python 3.12 和
<a href="https://docs.astral.sh/uv/">uv</a>。默认仿真使用进程内总线，不需要 NATS。
</p>

```bash
git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-Hey-Robot.git
cd Xbotics-Hey-Robot

uv sync --group dev --group sim
cp .env.example .env
```

<p>
根据 <a href="docs/reference/configuration.md">配置参考</a> 填写模型环境变量。
默认 Ubuntu 仿真 profile 使用 DeepSeek 作为 Agent model，并可选用 DashScope 视觉模型。
</p>

<h3>2 · 检查配置</h3>

```bash
uv run hey-robot inspect --config configs/xlerobot.sim.ubuntu.yaml
```

<h3>3 · 启动 MuJoCo Harness</h3>

```bash
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

<table>
  <tr>
    <td><strong>Chat</strong></td>
    <td><a href="http://127.0.0.1:8080/chat"><code>http://127.0.0.1:8080/chat</code></a></td>
  </tr>
  <tr>
    <td><strong>Tasks</strong></td>
    <td><a href="http://127.0.0.1:8080/tasks"><code>http://127.0.0.1:8080/tasks</code></a></td>
  </tr>
</table>

<details>
<summary><strong>使用 NATS 的 profile</strong></summary>
<br />
<p>只有配置显式设置 <code>deployment.bus.type: nats</code> 时才需要：</p>

```bash
nats-server
# 或
docker compose up -d nats
```

</details>

<h2 id="real-robot">XLeRobot 真机</h2>

<p>
XLeRobot 是 Hey Robot 当前主要支持的真机 embodiment。默认真机 profile 公开
<code>inspect_scene</code>、<code>move_base</code> 和
<code>turn_base</code>；机械臂、操作策略与 VLA 闭环需要显式启用并逐机验证。
</p>

<p>连接硬件前，依次检查平台、配置、串口、舵机、相机和电池：</p>

```bash
uv run python scripts/ops/check_platform.py \
  --config configs/xlerobot.real.ubuntu.yaml

uv run hey-robot inspect \
  --config configs/xlerobot.real.ubuntu.yaml

uv run python scripts/robots/xlerobot/diagnose.py \
  --config configs/xlerobot.real.ubuntu.yaml
```

<p>
确认诊断通过后再运行 Harness。完整流程见
<a href="docs/operations/xlerobot-real.md">XLeRobot 真机部署</a>。
</p>

<h2 id="safety">安全边界</h2>

<ul>
  <li>所有运动先在 Mock / MuJoCo 中验证，再连接真实机器人。</li>
  <li>真机运行时保持物理断电或急停手段可用。</li>
  <li>不要在人员、宠物、易碎物和不稳定机械结构附近直接测试。</li>
  <li>修改舵机 ID、串口、相机、标定或机械结构后重新运行诊断。</li>
  <li>VLA/VLN 必须用目标 checkpoint、目标 embodiment 和真实观测单独闭环验证。</li>
</ul>

<h2 id="development">开发与扩展</h2>

```bash
uv run poe style
uv run poe lint
uv run poe test
```

<table>
  <tr><td><code>src/hey_robot/cognition</code></td><td>Agent、长程任务状态与 Tool loop</td></tr>
  <tr><td><code>src/hey_robot/skills</code></td><td>Skill 定义、执行、生命周期与持久结果</td></tr>
  <tr><td><code>src/hey_robot/foundation</code></td><td>VLA/VLN ModelService contract 与 backend</td></tr>
  <tr><td><code>src/hey_robot/robot_runtime</code></td><td>观测、安全、control plane 与本地执行</td></tr>
  <tr><td><code>src/hey_robot/robot_backends</code></td><td>Mock、MuJoCo、XLeRobot 和 RoboCasa driver</td></tr>
</table>

<p>
贡献前请阅读 <a href="CONTRIBUTING.md">贡献指南</a> 和
<a href="docs/development/skill-extension.md">Skill 扩展指南</a>。
</p>

<h2 id="documentation">文档</h2>

<table>
  <thead>
    <tr><th>主题</th><th>入口</th></tr>
  </thead>
  <tbody>
    <tr><td>文档事实源</td><td><a href="docs/index.md">文档索引</a></td></tr>
    <tr><td>系统设计</td><td><a href="docs/architecture/system-architecture.md">系统架构</a></td></tr>
    <tr><td>研究草稿</td><td><a href="docs/references/paper-draft.md">论文草稿</a></td></tr>
    <tr><td>配置字段</td><td><a href="docs/reference/configuration.md">配置参考</a></td></tr>
    <tr><td>MuJoCo</td><td><a href="docs/operations/xlerobot-sim.md">仿真部署</a></td></tr>
    <tr><td>真实机器人</td><td><a href="docs/operations/xlerobot-real.md">真机部署</a></td></tr>
    <tr><td>RoboCasa365</td><td><a href="docs/evaluation/robocasa365/runbook.zh-CN.md">评测运行手册</a></td></tr>
  </tbody>
</table>

<h2 id="community">社区</h2>

<p>
本项目来自开源机器人 XLeRobot 动手实战工作坊相关实践。欢迎通过 Issue、Pull Request
或社区渠道参与 Harness、机器人驱动、交互体验与具身模型集成。
</p>

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

<p align="center">
  <a href="https://github.com/Vector-Wangel/XLeRobot">XLeRobot</a> ·
  <a href="docs/references/project-references.md">项目活动与参考</a> ·
  <a href="LICENSE">MIT License</a>
</p>
