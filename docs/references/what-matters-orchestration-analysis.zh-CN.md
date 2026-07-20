# 《What Matters in Orchestrating Robot Policies》与 Hey Robot 实现分析

> 文档状态：基于 2026-07-19 仓库实现的代码审计与设计评述  
> 论文：`What Matters in Orchestrating Robot Policies.md`，arXiv:2606.10267v1  

## 1. 结论

《What Matters in Orchestrating Robot Policies》很适合作为 Hey Robot 下一阶段的设计检查表。
它最重要的结论不是提出一种全新的机器人 Agent 算法，而是用系统性消融实验说明：

> 高层规划器、低层 VLA、终止判定、观察表示和经验记忆之间的编排逻辑，本身就是机器人
> 策略的重要组成部分，不只是工程胶水。

Hey Robot 的总体方向与论文高度一致，而且在任务状态、安全门控、服务隔离、取消、审计和
执行证据方面，比论文中的实验原型更工程化。但按照论文定义的 Hi-VLA 闭环衡量，Hey Robot
目前仍处于“架构基本成形、真实 VLA 能力和长程收益尚未验证”的阶段。

一个便于沟通的主观评分如下。分数表示当前仓库可见证据，不表示项目的最终能力上限：

| 维度 | 评价 |
| --- | --- |
| 论文作为系统设计参考 | 8/10 |
| Hey Robot 架构完整度 | 8/10 |
| Hey Robot 当前 Hi-VLA 闭环完成度 | 5/10 |
| 真实模型与具身任务验证程度 | 2–3/10 |

因此，Hey Robot 当前最准确的定位是：

> 一个设计扎实、适合承载具身 Agent 和分层机器人策略实验的 Embodied Agent Harness，
> 但还不是经过系统实验验证的 Hi-VLA 系统。

## 2. 论文的核心贡献

### 2.1 论文研究的系统结构

论文把层级机器人策略描述为一个 options 风格的系统：

```text
环境观测
   |
   v
观察表示 phi
   |
   v
高层 VLM/VLM policy ---- memory
   |
   | 自然语言子目标
   v
低层 VLA policy
   |
   | 动作
   v
机器人/环境
   |
   +---- termination beta ----> 切换或继续当前子目标
```

论文没有只比较“层级模型”和“单模型”，而是进一步拆解了决定层级系统效果的五类因素：

1. 高层模型是否具备推理能力；
2. 低层 VLA 是否能被自然语言稳定操控；
3. 当前子目标何时结束；
4. 高层模型接收什么形式的环境表示；
5. 系统如何利用当前 episode 和跨 episode 的历史经验。

### 2.2 最有价值的实验结论

#### 推理能力比单纯扩大参数量更重要

论文发现，高层模型的 thinking/reasoning 能力对复杂任务收益明显，而模型规模本身不是主要
决定因素。这对 Hey Robot 的启示是：高层 Planner 的评估应关注分解、前置条件、恢复和
多步一致性，而不能只看模型参数或普通聊天基准。

#### VLA 的语言可控性是独立能力

低层 VLA 在单一操作上的成功率高，不代表它适合被高层 Agent 调度。层级系统还要求 VLA：

- 能区分近义但目标不同的子指令；
- 能理解属性、空间关系、否定和组合条件；
- 面对不同措辞时保持稳定；
- 失败后能接受修正后的语言子目标；
- 不会因为特定任务微调而丢失通用语言可控性。

论文观察到，某些 in-domain 微调虽然提高了局部任务能力，却显著降低了语言 steerability。
因此，选择第三方 VLA 时不能只看官方 success rate，还要单独测试其作为层级系统低层策略的
可操控性。

#### 子目标终止判定是层级系统的关键组件

论文比较了 VLM 预测执行时长、固定 horizon 和独立 success detector。独立 success detector
效果最好，VLM 预估时长最差；固定约 4–8 秒的执行窗口可作为中等质量基线。

这说明高层模型生成正确子目标还不够。如果系统不知道子目标何时完成，就会出现：

- 尚未完成就切换，破坏后续任务的前置条件；
- 已经完成仍继续动作，引入碰撞或把物体推离目标；
- 在不可恢复状态里反复执行同一策略；
- 局部状态看似成功，但并不满足整条任务链的要求。

论文还指出，success detector 的 false positive 尤其危险，因为错误切换会将不成立的状态
传播给后续所有子目标。

#### 结构化观察优于普通场景摘要

论文结果并不支持“把图像总结成一段自然语言就一定更好”。普通文字摘要有时甚至弱于直接
使用图像；带 bbox 的对象表示或包含接触状态的结构化表示明显更有效。

这说明观察表示需要保留与动作相关的关系，例如：

- 目标物体位置和 bbox/segmentation；
- 物体与容器、台面、机器人的空间关系；
- 夹爪是否真正持有物体；
- 门、抽屉、开关等 fixture state；
- 接触、碰撞、可达性和遮挡；
- 跨帧稳定的实体 identity。

#### 跨 episode 经验比堆叠当前历史更有效

增加当前 episode 的原始历史长度没有带来稳定收益，简单的 episode 内摘要甚至可能产生
干扰。相反，把过去 episode 提炼成 affordance summary，可以显著提高后续任务表现。

真正有价值的记忆不是完整对话或动作流水账，而是：

```text
什么子目标措辞对当前 VLA 有效
什么物体或区域可达
什么观察角度能验证成功
某类失败对应什么恢复动作
某个机器人和场景组合有哪些稳定约束
```

### 2.3 整体结果

论文报告的聚合结果表明，优化后的层级系统明显优于朴素层级系统和平坦 VLA：

| 系统 | 短程任务 | 长程任务 | 推理任务 |
| --- | ---: | ---: | ---: |
| 优化后的层级系统 | 78.22% | 67.08% | 80.89% |
| 朴素层级系统 | 69.57% | 40.56% | 66.49% |
| 平坦 VLA | 69.63% | 25.30% | 50.90% |

这组结果最值得关注的是长程任务：层级结构本身会带来收益，但只有把终止判定、观察表示、
低层可控性和经验记忆做好，层级系统的优势才会充分显现。

## 3. 对论文的评价和适用边界

### 3.1 优点

这是一篇优秀的系统消融论文，主要优点是：

- 把“编排”从模糊工程经验拆成了可独立实验的组件；
- 不把所有性能变化笼统归因于更大模型；
- 同时研究短程、长程和需要推理的任务；
- 结论可直接转化为系统设计和评测项目；
- 对 Hey Robot 这类上层 Agent + Skill/VLA + Runtime 架构高度相关。

### 3.2 局限

论文结论不能不加条件地外推到 Hey Robot 的完整 embodied agent 场景：

- 实验主要是静态桌面操作，不是导航、操作和人物交互的联合任务；
- 最强 success detector 使用了仿真器特权状态，现实机器人无法原样获得；
- contact 和部分结构化信息在仿真器中更容易可靠取得；
- 实验假设模型和服务延迟不是主要瓶颈；
- 覆盖的 VLA 和高层模型家族仍然有限；
- 实机 ALOHA 评估规模较小；
- 没有充分研究移动人物、动态障碍和执行期间的人类修正。

因此，这篇论文更适合被看作“Hi-VLA 系统的强设计证据”，而不是已经证明一种通用具身
Agent 架构可以解决开放世界移动操作。

## 4. Hey Robot 与论文框架的对应关系

### 4.1 总体映射

Hey Robot 并不是论文结构的简单复刻。论文高层通常直接生成语言子目标交给 VLA，而
Hey Robot 高层生成的是具名 Skill 和结构化参数：

```text
论文：高层 VLM -> language subgoal -> VLA -> robot

Hey Robot：Agent -> Skill(name, slots)
                   |-- 经典控制技能
                   |-- 导航/跟随技能
                   |-- VLA/VLN ModelService
                   `-- Robot Runtime
```

这种额外的 Skill 层对通用 embodied agent 是合理的。它允许导航、人物跟随、语音交互、
确定性操作和 VLA 共存，也便于加入资源、权限和安全门控。代价是系统需要明确回答：哪些
子目标由 Agent 分解、哪些由 Skill 内部处理、哪些又交给 VLA 自己完成。

### 4.2 逐项状态

| 论文关键因素 | Hey Robot 当前实现 | 判断 |
| --- | --- | --- |
| 推理型高层策略 | 持续 Agent 循环、单步工具选择、执行结果回灌 | 机制已具备，能力未基准验证 |
| 语言可控的 VLA | 独立 ModelService、manipulation skill、实验配置 | 接口存在，checkpoint 能力未验证 |
| 子目标终止检测 | VLA 自报 `task_done` 或 `max_steps` | 当前最大缺口 |
| 结构化观察 | 独立 scene captioner，输出对象、风险和任务相关信息 | 中等，缺 bbox/contact/fixture state |
| episode 内任务状态 | SQLite 任务、步骤和 evidence 账本 | 做得较好 |
| 跨 episode 经验 | 已完成任务可持久化，但无经验提炼和检索 | 基本缺失 |
| 安全与可审计性 | Skill 契约、证据审计、取消、超时、服务隔离 | 强于一般论文原型 |
| 系统性评测 | 无真实 checkpoint 的分层对照 benchmark | 严重不足 |

## 5. Hey Robot 已经做得好的部分

### 5.1 高层任务循环不是一次性工具调用

`src/hey_robot/cognition/autonomous_agent_service.py` 中的 `AutonomousAgentService` 支持持续
任务循环。Agent 可以请求观察、选择 Skill、等待实际结果、记录步骤和证据，然后继续规划
或提出任务完成。

这已经具备论文高层 option policy 的基本结构，并且比“LLM 生成一次动作序列后直接执行”
更适合真实机器人。执行结果会进入下一次决策，而不是被当作日志旁路掉。

### 5.2 整任务完成证据审计较扎实

`src/hey_robot/cognition/runtime/agent_task_store.py` 中的完成检查要求完成声明引用当前任务的
真实 evidence。发生物理动作或世界状态变化后，系统还要求引用动作之后的新观察，避免用
动作前的旧画面证明任务已经完成。

`src/hey_robot/cognition/runtime/completion_verifier.py` 又提供了独立的结构化审核 pass，对任务
目标、执行步骤和 evidence 进行复核，并采用 fail-closed 的结果协议。

这些机制比单纯让 Planner 自己输出“任务完成”可靠。不过必须明确：它是整任务完成
auditor，不是论文中每个 VLA 子目标的 termination beta。

### 5.3 模型环境和主 Runtime 解耦

Foundation ModelService 通过 gRPC 提供健康检查、执行和取消接口，并能按 capability 与
`robot_id` 路由服务。这适合将第三方 VLA、RoboCasa365、LeRobot 和 GPU 依赖放在独立
环境或 Docker 中，避免污染 Hey Robot 主运行环境。

这一服务边界也让系统可以替换 checkpoint、单独重启模型容器、隔离显存，并在主 Agent
侧保留统一的任务和审计逻辑。

### 5.4 VLA 执行采用重观察的闭环方式

`src/hey_robot/skill_os/builtins/manipulation_adapter.py` 当前只执行 action chunk 的第一个动作，
然后重新观察和重新推理。这能减少长 action chunk 的开环漂移，符合真实机器人谨慎执行的
需求。

但这只是一个保守基线，不一定是最终最优控制频率。它可能造成推理频率过高、延迟增大，
也没有充分利用 action chunk 的时序结构。后续应评测可配置的 receding horizon，而不是
固定执行一个动作或无条件执行完整 action chunk。

### 5.5 系统工程边界比实验原型更完整

Hey Robot 已经拥有论文不重点讨论、但真实系统必需的能力：

- Skill 契约和 backend readiness 检查；
- 安全状态和资源门控；
- 执行超时、取消和结构化失败；
- 任务、步骤和 evidence 持久化；
- Agent、Skill、模型服务与 Robot Runtime 的清晰边界；
- 导航、操作、人物跟随和交互技能共存的扩展面。

这些能力不会直接产生更高 benchmark 分数，但决定了模型能力能否被可靠部署和诊断。

## 6. 当前主要缺口

### 6.1 缺少独立的 VLA 子目标终止检测

`src/hey_robot/skill_os/builtins/manipulation.py` 的操作循环主要依赖模型结果中的
`task_done`，否则执行到 `max_steps`。这等于把“动作怎么做”和“是否已经成功”同时交给同一个
低层模型，而且缺少独立证据校验。

建议新增 `SubgoalTerminationService`，至少接收：

```text
当前语言子目标
动作前观察
最新观察
机器人状态
执行步数和异常
可选的仿真特权状态
```

输出应是结构化状态，而不仅是布尔值：

```text
SUCCESS   当前子目标已经完成，可以切换
CONTINUE  尚未完成，但策略仍在正常推进
FAILED    当前策略无法继续，应重观察、恢复或重规划
UNKNOWN   证据不足，不能宣布成功
```

在 RoboCasa365 中，可以先使用 fixture/contact/object state 构造近似 oracle detector；迁移到
真实机器人时，再用视觉、夹爪反馈、力觉和 VLM 审核替代特权状态。

### 6.2 场景表示仍接近普通结构化摘要

`src/hey_robot/cognition/perception/scene/captioner.py` 会从多张图片生成场景摘要、对象、实体、
风险、任务相关信息和下一步提示。这提供了清晰的感知边界，但目前缺少论文中收益更明确的
bbox、接触和物体状态信息。

此外，实体解析主要依赖可信 ID 和配置别名，更偏向安全绑定机制，不是完整的视觉语义
grounding。后续应将以下信息纳入统一 scene representation：

- bbox、mask、深度和坐标系；
- 可见性、遮挡、可达性；
- gripper holding/contact；
- fixture 开合、开关和锁定状态；
- 对象关系和跨帧 track ID；
- 信息来源、时间戳和置信度。

### 6.3 持久任务记录还没有变成跨 episode 经验

当前 SQLite 任务存储对恢复和审计有价值，但高层上下文主要使用当前活动任务和最近步骤。
系统没有从已完成任务中提炼 affordance，也没有根据 embodiment、场景、对象和 VLA 型号检索
过去经验。

建议在 episode 结束后生成受约束的 `ExperienceRecord`：

```yaml
embodiment: panda_omron
policy: lerobot/smolvla_robocasa
scene_family: kitchen
task_family: close_fixture
effective_subgoals:
  - "move in front of the refrigerator"
  - "push the refrigerator door until fully closed"
failed_patterns:
  - "close the fridge"  # 范围过大，完成判定不稳定
verification:
  - fixture_joint_state
  - post_action_wide_view
recovery:
  - re-center_mobile_base
```

检索时只注入与当前机器人、策略和任务相关的少量经验，不应直接把完整历史日志塞进 Planner。

### 6.4 第三方 VLA 的层级可控性没有评测

实验配置可以把 manipulation 指向 VLA ModelService，但仓库中还没有证据证明某个真实
checkpoint 已在目标 observation/action schema 上稳定跑通。更缺少以下 steerability 测试：

- 同义改写是否产生一致行为；
- 多对象场景能否遵守颜色、位置和类别限定；
- 能否理解否定和排除条件；
- 组合指令拆成子目标后是否优于整句输入；
- 失败后修改措辞是否真正改变策略；
- 微调是否破坏原有语言泛化。

因此，“ModelService 接口存在”和“Hey Robot 已支持可用 VLA”应继续被明确区分。

### 6.5 缺少与论文对应的对照实验

目前没有可以回答以下问题的数据：

- Hey Robot 层级编排是否优于 flat VLA；
- 独立终止检测提高了多少长程成功率；
- scene caption、bbox 和 privileged state 分别带来多少收益；
- reasoning Planner 是否优于普通 Planner；
- 跨 episode 经验是否提高成功率或只增加提示噪声；
- 服务延迟是否抵消了更复杂编排的收益。

单元测试能验证协议和控制路径没有明显断裂，但不能替代真实 checkpoint 的任务评测。

## 7. RoboCasa365 接入对这篇论文的意义

### 7.1 容器内完整 rollout 只能验证基础设施

RoboCasa365 canonical 文档将容器内完整 rollout 标记为 B0 flat VLA baseline：

```text
Hey Robot 下发完整任务
    -> RoboCasa365 容器内部 reset/observe/policy/step/success
    -> 返回 episode 结果
```

这是合理的第一阶段，因为它可以低风险验证：

- 独立 Docker 和 GPU 环境；
- RoboCasa、LeRobot 和 checkpoint 安装；
- observation/action schema；
- Agent -> Skill -> ModelService -> 结果回灌；
- 日志、视频、超时和取消。

但这种模式不能验证论文的核心命题。整个 episode 都在容器里闭环时，Hey Robot 无法在中途
生成子目标、判断子目标完成、切换策略或根据失败重规划。此时 RoboCasa365 只是一个黑盒
操作 Skill。

### 7.2 真正的 Hi-VLA 评测需要 step/subgoal 级边界

基础 smoke test 稳定后，应增加至少以下接口：

```text
reset_episode
get_observation
set_subgoal
step_policy
get_robot_state
check_subgoal
close_episode
```

高频 MuJoCo step 可以继续留在容器内，但 Hey Robot 必须能在 option 边界获得控制权：

```text
Hey Robot 产生子目标
    -> RoboCasa VLA 执行有限 horizon
    -> 返回最新观察、状态和轨迹摘要
    -> termination detector 判断
       |-- SUCCESS: 生成下一个子目标
       |-- CONTINUE: 延长当前子目标
       `-- FAILED: 恢复或重规划
```

这才是在 RoboCasa365 中测试论文所说 policy orchestration，而不仅是测试第三方 VLA 能否运行。

## 8. 建议的验证矩阵

### 8.1 三组系统基线

| 组别 | 高层编排 | 终止方式 | 观察表示 | 目的 |
| --- | --- | --- | --- | --- |
| Flat VLA | 无 | 模型 done/episode limit | VLA 原始输入 | 测量低层模型本身能力 |
| Naive Hey Robot | Agent 子目标 | 固定 horizon 或 VLA done | 普通 scene caption | 测量基础层级收益 |
| Improved Hey Robot | reasoning Agent | 独立 detector | bbox/state/contact | 测量优化编排收益 |

### 8.2 任务类别

建议至少分为：

1. 原子操作：开关门、移动单个物体、放入指定容器；
2. 复合操作：连续操作多个 fixture 或对象；
3. 移动操作：底盘定位后再操作；
4. 推理任务：根据属性、容器状态或空间条件选择目标；
5. 恢复任务：人为注入抓取失败、遮挡、错误接近方向；
6. 扰动任务：执行过程中改变对象位置或引入动态干扰。

每个条件应固定任务集合和 seed，并记录足够多 episode，避免用单次成功视频代替统计结果。

### 8.3 指标

除整任务成功率外，至少记录：

- 子目标成功率；
- termination false positive/false negative；
- 每个任务的子目标数量和切换次数；
- VLA policy 调用次数；
- 每次调用执行的 action horizon；
- 重规划、恢复和人工介入次数；
- 碰撞、越界和无效动作；
- 首 token、Planner、VLA 和端到端延迟；
- GPU/CPU/显存使用；
- 不同语言改写下的成功率方差。

## 9. 实施优先级

### P0：建立可复现基线

1. 在独立 RoboCasa365 Docker 中运行预训练 VLA；
2. 固定环境、资产、checkpoint、任务、seed 和输出格式；
3. 先取得 flat VLA 原子任务结果；
4. 保存 episode 视频、状态轨迹和 success 原因；
5. 将基础设施成功与任务成功分开报告。

### P0：补齐子目标终止服务

1. 定义 `SUCCESS/CONTINUE/FAILED/UNKNOWN` 协议；
2. 在 RoboCasa 中先实现 privileged detector；
3. 将 VLA 自报 done 作为证据之一，而不是唯一事实来源；
4. 对 detector 单独统计 false positive 和 false negative；
5. 在证据冲突时默认不宣布成功。

### P1：增强观察表示

1. 输出 bbox、mask、fixture state、gripper/contact；
2. 把所有状态绑定时间戳、来源和置信度；
3. 对 image、caption、bbox 和 privileged state 做消融；
4. 保持仿真特权字段与真机可观测字段的清晰边界。

### P1：评测 VLA steerability

1. 为每个任务生成一组语义等价改写；
2. 加入属性、关系、否定和组合测试；
3. 比较整句任务与分解子目标；
4. 用层级可控性选择 checkpoint，而不只看单任务成功率。

### P1：实现跨 episode affordance memory

1. episode 结束后提炼结构化经验；
2. 按机器人、策略、场景和任务检索；
3. 限制注入数量，避免把日志噪声重新带回 Planner；
4. 对比无记忆、原始历史和 affordance summary。

### P2：优化 action horizon 和动态场景

1. 比较单动作、固定短 chunk、4–8 秒 option horizon；
2. 支持风险或显著场景变化触发提前中断；
3. 记录模型推理和跨服务延迟；
4. 增加移动人物、动态障碍和用户中途修正测试。

## 10. 验证证据与边界

本次代码审计对以下相关测试进行了小范围运行：

```bash
.venv/bin/pytest -q --no-cov \
  tests/cognition/runtime/test_completion_verifier.py \
  tests/cognition/perception/test_scene_captioner.py \
  tests/cognition/test_conversation_routing.py \
  tests/cognition/test_robot_agent_loop.py \
  tests/skill_os/test_registry.py \
  tests/foundation/test_transport_grpc_server.py
```

结果为 79 项通过。这说明相关结构化完成审核、场景描述、Agent 循环、Skill registry 和 gRPC
模型服务路径具有单元测试覆盖。

该结果不能证明：

- 真实 VLA checkpoint 已正确加载；
- observation/action schema 与目标机器人完全一致；
- VLA 在 RoboCasa365 或真机上具有可用成功率；
- 分层编排优于 flat VLA；
- 导航、操作和人物交互已经形成统一长程闭环。

这些结论必须由独立容器中的真实 checkpoint 评测、仿真 episode 统计和最终真机实验给出。

## 11. 最终判断

论文证明了一个对 Hey Robot 很重要的方向：具身 Agent 的主要价值不一定来自让一个模型承担
全部导航和操作，而是让不同时间尺度、不同能力边界的策略被可靠地组织起来。

Hey Robot 已经具备成为这类实验平台的关键系统骨架：持续任务循环、Skill 抽象、模型隔离、
任务状态、证据审核、安全和执行反馈。当前最需要补的不是另一个大模型入口，而是三条可度量的
闭环：

1. 独立的 VLA 子目标终止检测；
2. 可验证的空间、接触和机器人状态表示；
3. flat VLA、朴素层级和增强层级之间的可复现实验。

完成这三项后，Hey Robot 才能从“架构上可以接 VLA”推进到“有数据证明编排提高了长程具身
任务能力”。RoboCasa365 是适合完成这一验证的第一站，但基础设施 smoke test 和真正的
Hi-VLA orchestration benchmark 必须分阶段、分结论报告。
