<div align="center">

  <pre>
  ██╗  ██╗██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
  ╚██╗██╔╝██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
   ╚███╔╝ ██████╔╝██║   ██║   ██║   ██║██║     ███████╗
   ██╔██╗ ██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
  ██╔╝ ██╗██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
  ╚═╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
  </pre>

<img src="images/hey-robot-icon.png" alt="Hey Robot project icon" width="300" />

<h1>Hey Robot</h1>

<p>
  <em>An Embodied Agent Harness for real robots · Fast–slow dual-system architecture · Distributed coordination</em>
</p>

<p><strong>Build interactive robots that can stay on task.</strong></p>

<p>
  An open-source <strong>Embodied Agent Harness</strong> for real robots:<br />
  a slow system maintains goals and interaction, while a fast system executes
  bounded skills, observes the world, and controls the robot.
</p>

<p>
  <a href="#why">Why Hey Robot</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#status">Capability Status</a> ·
  <a href="index.md">Docs</a> ·
  <a href="references/paper-draft.md">Paper Draft</a> ·
  <a href="../README.md">简体中文</a>
</p>

<p>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-0b7285?style=flat-square" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Harness-Embodied_Agent-6f42c1?style=flat-square" alt="Embodied Agent Harness" />
  <img src="https://img.shields.io/badge/Embodiment-XLeRobot-f59f00?style=flat-square" alt="XLeRobot" />
</p>

</div>

<br />

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>💬 Continuous interaction</h3>
      <p>Ask, correct, pause, resume, cancel, or emergency-stop while physical work is in progress.</p>
    </td>
    <td width="33%" valign="top">
      <h3>🧭 Long-horizon tasks</h3>
      <p>Persist goals, steps, runs, and recovery points; resume reasoning from terminal skill events.</p>
    </td>
    <td width="33%" valign="top">
      <h3>🧩 Agent harness</h3>
      <p>Keep cognition, capabilities, models, safety, robot drivers, observations, and operations replaceable and auditable.</p>
    </td>
  </tr>
</table>

<blockquote>
  <strong>Status:</strong> the harness, MuJoCo path, XLeRobot drivers, interaction
  channels, and durable task machinery are under active development. VLA/VLN and
  complex real-robot long-horizon tasks remain experimental.
</blockquote>

<h2 id="why">Why Hey Robot</h2>

<p>
A conventional LLM tool loop can choose the next call, but real robots add state
that prompts do not solve: observations become stale, hardware stays busy, actions
are costly to undo, users revise requests mid-execution, and processes may restart
before a physical task is complete.
</p>

<table>
  <thead>
    <tr><th>Robotics problem</th><th>Harness mechanism</th></tr>
  </thead>
  <tbody>
    <tr><td>Work spans multiple model calls and actions</td><td>Durable tasks, step/run correlation, event-driven continuation, startup recovery</td></tr>
    <tr><td>Users interact during execution</td><td>Unified sessions plus pause, resume, cancel, and emergency-stop controls</td></tr>
    <tr><td>Models can propose infeasible actions</td><td>Explicit skill surface, validation, resource exclusion, timeouts, and readiness gates</td></tr>
    <tr><td>Models, simulation, and hardware evolve separately</td><td>Agent, Skill, ModelService, Robot Runtime, and Driver boundaries</td></tr>
    <tr><td>A successful call is not task completion</td><td>Skill lifecycle, robot status/observation, task timeline, and recovery context</td></tr>
  </tbody>
</table>

<h2 id="long-horizon">Interactive long-horizon tasks</h2>

<p>
Hey Robot represents a long-horizon objective as durable state rather than an
ever-growing chat transcript. Each session has at most one non-terminal task,
with persisted steps, skill runs, results, pause state, and resume position.
</p>

```text
user objective
  → reason and select a visible tool / skill
  → execute the skill asynchronously
  → persist its terminal result
  → wake the agent from the terminal event
  → continue, pause for the user, cancel, fail, or complete
```

<h2 id="architecture">Embodied Agent Harness: fast and slow</h2>

<p>
Fast and slow describe decision horizons, not hard real-time guarantees.
The slow system owns semantic continuity; the fast system turns one bounded
capability into guarded model, simulation, or hardware execution.
</p>

<table>
  <thead>
    <tr><th></th><th>Slow · deliberative</th><th>Fast · embodied execution</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Horizon</strong></td><td>Across turns, skills, and service restarts</td><td>One bounded skill and its local control process</td></tr>
    <tr><td><strong>Responsibilities</strong></td><td>Goals, tool choice, task progress, pause, recovery</td><td>Perception, admission, local model inference, safety, robot execution</td></tr>
    <tr><td><strong>Implementation</strong></td><td><code>Agent</code>, <code>AgentRunner</code>, <code>AgentTaskStore</code>, <code>TaskCoordinator</code></td><td><code>SkillWorker</code>, VLA/VLN options, <code>RobotRuntime</code>, drivers</td></tr>
  </tbody>
</table>

<p align="center">
  <img src="images/architecture.png"
       alt="Hey Robot Embodied Agent Harness architecture"
       width="100%" />
</p>

<p align="center">
  <sub>Interaction continues while a Skill runs; terminal events resume the slow system at a safe boundary, while physical actions follow one execution path.</sub>
</p>

<p>
The default deployment composes the Agent, Skill Worker, and Robot Runtime in one
asyncio process. VLA/VLN ModelServices can run independently over gRPC. See the
<a href="architecture/system-architecture.md">system architecture</a> and
<a href="references/paper-draft.md">paper draft</a>.
</p>

<h2 id="status">Capability status</h2>

<table>
  <thead>
    <tr><th>Capability</th><th>Status</th><th>Boundary</th></tr>
  </thead>
  <tbody>
    <tr><td>Web / CLI / Voice / Feishu</td><td>Implemented</td><td>Shared Gateway, identity, and episode model</td></tr>
    <tr><td>Durable long-horizon tasks</td><td>Implemented</td><td>SQLite task/step state, continuation, pause/resume, startup recovery</td></tr>
    <tr><td>Skill harness</td><td>Implemented</td><td>Explicit tools, validation, resources, timeout, cancellation, events, run store</td></tr>
    <tr><td>Mock / MuJoCo / real drivers</td><td>Integrated</td><td>Real robots still require per-machine calibration and diagnosis</td></tr>
    <tr><td>VLA / VLN ModelService</td><td>Experimental</td><td>Contract and profiles exist; checkpoints and closed loops need validation</td></tr>
    <tr><td>Open-world real-robot tasks</td><td>Research target</td><td>Not a capability claim of the current release</td></tr>
  </tbody>
</table>

<h2 id="quick-start">Quick start</h2>

<p>
Recommended: Ubuntu/Linux, Python 3.12, and
<a href="https://docs.astral.sh/uv/">uv</a>. The default simulation profile uses
an in-process bus and does not require NATS.
</p>

```bash
git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-Hey-Robot.git
cd Xbotics-Hey-Robot

uv sync --group dev --group sim
cp .env.example .env

uv run hey-robot inspect --config configs/xlerobot.sim.ubuntu.yaml
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

<table>
  <tr><td><strong>Chat</strong></td><td><a href="http://127.0.0.1:8080/chat"><code>http://127.0.0.1:8080/chat</code></a></td></tr>
  <tr><td><strong>Tasks</strong></td><td><a href="http://127.0.0.1:8080/tasks"><code>http://127.0.0.1:8080/tasks</code></a></td></tr>
</table>

<p>
See the <a href="reference/configuration.md">configuration reference</a> for model
credentials and profile fields.
</p>

<h2 id="real-robot">XLeRobot hardware</h2>

<p>
XLeRobot is the primary real-robot embodiment currently supported by Hey Robot.
The default hardware profile exposes <code>inspect_scene</code>,
<code>move_base</code>, and <code>turn_base</code>. Arm control, manipulation
policies, and VLA closed loops require explicit enablement and per-robot
validation.
</p>

```bash
uv run python scripts/ops/check_platform.py \
  --config configs/xlerobot.real.ubuntu.yaml

uv run hey-robot inspect \
  --config configs/xlerobot.real.ubuntu.yaml

uv run python scripts/robots/xlerobot/diagnose.py \
  --config configs/xlerobot.real.ubuntu.yaml
```

<p>
Read the <a href="operations/xlerobot-real.md">real-hardware guide</a> before
starting the harness on a physical robot.
</p>

<h2 id="safety">Safety</h2>

<ul>
  <li>Validate motion in Mock or MuJoCo before using real hardware.</li>
  <li>Keep a physical emergency stop or power cutoff available.</li>
  <li>Do not test motion near people, pets, fragile objects, or unstable mechanics.</li>
  <li>Re-run diagnostics after changing ports, servo IDs, cameras, calibration, or structure.</li>
  <li>Validate VLA/VLN with the target checkpoint, embodiment, and real observation path.</li>
</ul>

<h2 id="development">Development</h2>

```bash
uv run poe style
uv run poe lint
uv run poe test
```

<p>
Start with the <a href="index.md">documentation index</a>,
<a href="../CONTRIBUTING.md">contribution guide</a>, and
<a href="development/skill-extension.md">skill extension guide</a>.
</p>

<p align="center">
  <a href="https://github.com/Vector-Wangel/XLeRobot">XLeRobot</a> ·
  <a href="references/project-references.md">Project references</a> ·
  <a href="../LICENSE">MIT License</a>
</p>
