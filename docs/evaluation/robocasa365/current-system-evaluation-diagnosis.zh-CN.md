# Hey Robot × RoboCasa365 当前系统评估诊断

更新时间：2026-07-22

## 1. 结论

当前系统的 Runtime、Skill OS、ModelService、PI0.5、MuJoCo/EGL 和官方 evaluator 链路是可用的，
但还不能据此宣称具备稳定长程能力。

- `CloseFridge` 完整回归成功，证明集成链路和基础操作能力正常；
- `KettleBoiling` 在官方完整指令的 flat B0 条件下仍失败，说明该任务的首要瓶颈是当前
  `pi052_robocasa` checkpoint 的任务执行能力，而不是高层规划器没有分解；
- 原系统同时存在评测目标、视觉表示、重复观察、任务 horizon 和失败退出等系统问题，现已修复；
- 当前 PI0.5 使用 `subtask_mem` recipe，自带高层 subtask 生成与记忆，不能未经验证就把外部
  Agent 子目标替换到 checkpoint 的根 `task` 通道。

因此，下一阶段应先完成 target50 的分层能力画像，再决定哪些失败适合用更强编排解决，哪些失败
必须更换或后训练 VLA。不能用高层 harness 掩盖底层根本不会执行的技能。

## 2. 本轮真实运行

| 任务 | 条件 | 根任务来源 | 动作数 | option 数 | 官方成功 | 结论 |
|---|---:|---|---:|---:|---:|---|
| KettleBoiling | B2，错误人工目标 | CLI：electric kettle | 200 | 4 | 否 | 目标语义与官方任务不一致，trial 无效 |
| KettleBoiling | B1，错误人工目标 | CLI：electric kettle | 50 | 1 | 否 | 视觉漏检且 Agent 提前阻塞，trial 无效 |
| KettleBoiling | B0 | 环境官方指令 | 300 | 1 | 否 | Flat PI0.5 始终围绕锅具运动，未接近水壶 |
| CloseFridge | B2 | 环境官方指令 | 281 | 5 | **是** | 完整 Hey Robot 链路回归通过 |

关键 artifact：

- `runtime/robocasa365/optimized4-kettle-boiling-b0-seed1000/`
- `runtime/robocasa365/optimized-close-fridge-b2-seed1000/`

全量软件验证：814 tests passed，0 failure，0 error。

## 3. 发现并修复的系统问题

### 3.1 人工 objective 可以与官方任务语义冲突

`KettleBoiling` 的官方定义是：把非电水壶从台面放到 stove burner，再打开 burner。人工输入
“Boil water using the electric kettle”导致 Agent 虚构底座、电源开关、壶盖和加水步骤。

修复后，评测默认从 live RoboCasa trial 的 `policy_task` 读取官方语言指令。只有语言改写实验
才显式传 `--objective`；artifact 同时记录 `official_objective` 和 `objective_source`。

### 3.2 场景理解丢弃了结构化物体信息

scene captioner 已生成 `objects`、位置和置信度，但 Runtime 过去只把一句 `summary` 交给
Agent，而且观察问题没有传给视觉模型。水壶明明位于画面左侧，caption 却连续十次只报告锅具。

修复后：

- observation question 会进入视觉模型；
- 物体、位置、置信度、task relevance 和补充观察提示会进入规划上下文；
- 同一 frame 连续分析两次后，第三次观察由硬门禁拒绝，防止无信息循环。

### 3.3 Agent 返回但没有创建任务时，benchmark 会空等

DeepSeek 曾返回 `APITimeoutError`，Agent turn 已结束，但 benchmark 因找不到 AgentTask 而继续
等待到 wall-clock timeout。现增加 `agent_no_task` 终止原因，使 provider 失败可以快速、明确地
归因到 planner，而不是表现成仿真卡死。

### 3.4 所有任务被错误限制为 1000 步

RoboCasa365 target50 的官方 horizon 随任务变化，最长达到 2900。原环境没有传
`episode_length`/`horizon`，Agent 预算也只覆盖约 1000 步。

修复后 backend 从 RoboCasa dataset registry 读取任务 horizon，同时设置 LeRobot wrapper 和
底层 robosuite horizon；Agent skill 预算提升到 220，默认 wall-clock timeout 提升到 7200 秒。

### 3.5 PI0.5 的输入契约不能按普通 steerable VLA 假设处理

checkpoint 配置表明它使用 `recipes/subtask_mem.yaml`，会从根任务自行生成低层 subtask，并在
action chunks 间维护状态。当前生产配置继续使用 `prompt_mode: environment_root`，与独立 evaluator
契约一致。代码保留 prompt 改变时清 action queue 的通用能力，但该配置下 effective policy prompt
在一个 trial 内保持官方根任务，不会在 option 边界破坏 PI0.5 内部状态。

## 4. 与 Hi-VLA、Harness VLA 的对应关系

### Hi-VLA

本轮证据直接支持论文的三个结论：

1. **观测表示重要**：针对问题的结构化物体描述明显优于一句朴素 caption；
2. **终止机制重要**：只按固定步数返回会让 Agent 把 option 结束误认为物理子目标成功；
3. **VLA steerability 是层级系统前提**：不能仅因为模型接受字符串，就假定它能可靠执行任意
   外部短子目标。

当前最缺的是可靠的 option success detector。下一版应判断“门是否闭合、物体是否随夹爪移动、
物体是否到达目标区域”等可观察状态变化，而不是让 Agent仅凭单帧 caption 猜测进度。

### Harness VLA

Harness VLA 的 Task Specific Memory、Global Memory、执行后诊断和官方最终谓词值得采用，但不能
直接照搬其“VLA 只做局部联系动作”的前提：论文在 RoboCasa365 使用 RLDX-1，而当前系统使用
`pi052_robocasa`，两者接口和可转向性不同。

适合当前系统的增量路线是：

1. 保持唯一 `manipulate` 接口和一套 Hey Robot Runtime；
2. 先保存成功 trial 的 option/观察骨架，形成只描述顺序与失败规则的跨 episode memory；
3. 所有空间关系在新 seed 中重新感知，不重放旧坐标；
4. 只有在同 backbone 实验证明外部短子目标有效后，才切换到外部 subgoal steering；
5. 正式对比 Harness VLA 时增加其 RLDX-1 同 backbone 轨道，避免把 VLA 差异误判为 harness 差异。

## 5. 下一轮评估顺序

1. 对 18 个 Atomic-Seen 任务先各跑 B0 seed 1000，得到底层能力矩阵；
2. 对 B0 成功的原子任务跑 B1/B2，测量编排是否保持或降低成功率；
3. Composite-Seen 先选其所需原子技能均已通过的任务；
4. 每个失败都按 `planner / observation / VLA grounding / grasp / transport / fixture actuation /
   termination` 分类；
5. 完成小规模机制验证后，再按 comparison plan 扩展到 target50 held-out seeds。

在当前证据下，不应优先继续调 KettleBoiling 的提示词。Flat 官方指令已经失败，应该先确认
checkpoint 在 target50 上的真实能力覆盖，或选择可与 Harness VLA 公平对比的 RLDX-1 backend。
