# Hey Robot 与 Harness VLA 的 RoboCasa365 对比实验方案

状态：实验协议草案。本文描述计划实施的实验，不代表相关条件均已实现或已经取得结果。

## 1. 目标与范围

本文定义 Hey Robot 与 Harness VLA 的公平对比方法。评估范围只包含
**RoboCasa365 target50**，不使用 LIBERO、LIBERO-Pro、RoboTwin 或其他机器人基准。

实验需要回答四个问题：

1. 在冻结同一个 VLA 时，Hey Robot 的编排是否优于直接运行 VLA？
2. Hey Robot 与 Harness VLA 的差异来自 planner、解析 primitive、局部重试，还是记忆？
3. Task Specific Memory 和 Global Memory 能否迁移到未参与 bootstrap 的 held-out seed？
4. 成功率提升需要多少环境交互、VLA 调用、planner token、墙钟时间和人工先验？

本项目继续坚持一套 Hey Robot 系统：所有新增能力必须进入现有 Agent、Skill OS、ModelService
和 Robot Runtime，不引入第二个 Harness worker、第二套 episode owner 或任务专用控制器。

参考资料：

- `docs/references/harness_VLA.md`；
- `docs/references/Hi-VLA.md`；
- `docs/evaluation/robocasa365/runbook.zh-CN.md`。

## 2. 对比结论的边界

Harness VLA 在 RoboCasa365 中使用冻结的 RLDX-1，当前 Hey Robot 使用
`lerobot/pi052_robocasa`。不同 VLA 的最终成功率不能直接证明 harness 优劣。因此实验分为两条轨道。

### 2.1 PI0.5 同骨干消融（主开发轨道）

所有条件使用同一个 PI0.5 checkpoint、相同任务、相同 seed 和相同预算，只改变编排机制。
该轨道用于证明 Hey Robot 各组件的增益，是工程迭代的主要依据。

### 2.2 RLDX-1 同骨干复现（严格论文对比轨道）

直接 RLDX-1、Harness VLA 报告结果和 Hey Robot 必须使用同一个 RLDX-1 checkpoint，并对齐
观测、primitive、任务 horizon、bootstrap 和成功判定协议。只有该轨道可以用于“Hey Robot 与
Harness VLA 谁更强”的直接结论。

如果暂时无法获得完全相同的 RLDX-1、prompt、primitive 或 seed 映射，只能将论文结果标为
`reported reference`，不得写成同条件复现结果。

## 3. RoboCasa365 target50

正式评估包含 50 个任务、340 个计分 rollout。

| Split | 任务数 | 每任务计分 rollout | 合计 |
|---|---:|---:|---:|
| Atomic-Seen | 18 | 10 | 180 |
| Composite-Seen | 16 | 5 | 80 |
| Composite-Unseen | 16 | 5 | 80 |
| 总计 | 50 | — | 340 |

### 3.1 Atomic-Seen（18）

`CloseBlenderLid`、`CloseFridge`、`CloseToasterOvenDoor`、`CoffeeSetupMug`、
`NavigateKitchen`、`OpenCabinet`、`OpenDrawer`、`OpenStandMixerHead`、
`PickPlaceCounterToCabinet`、`PickPlaceCounterToStove`、`PickPlaceDrawerToCounter`、
`PickPlaceSinkToCounter`、`PickPlaceToasterToCounter`、`SlideDishwasherRack`、
`TurnOffStove`、`TurnOnElectricKettle`、`TurnOnMicrowave`、`TurnOnSinkFaucet`。

### 3.2 Composite-Seen（16）

`DeliverStraw`、`GetToastedBread`、`KettleBoiling`、`LoadDishwasher`、
`PackIdenticalLunches`、`PreSoakPan`、`PrepareCoffee`、`RinseSinkBasin`、
`ScrubCuttingBoard`、`SearingMeat`、`SetUpCuttingStation`、`StackBowlsCabinet`、
`SteamInMicrowave`、`StirVegetables`、`StoreLeftoversInBowl`、`WashLettuce`。

### 3.3 Composite-Unseen（16）

`ArrangeBreadBasket`、`ArrangeTea`、`BreadSelection`、`CategorizeCondiments`、
`CuttingToolSelection`、`GarnishPancake`、`GatherTableware`、`HeatKebabSandwich`、
`MakeIceLemonade`、`PanTransfer`、`PortionHotDogs`、`RecycleBottlesByType`、
`SeparateFreezerRack`、`WaffleReheat`、`WashFruitColander`、`WeighIngredients`。

### 3.4 Seed 协议

- 每个任务的 reference seed `s0` 只用于 bootstrap，不计入正式成功率；
- Atomic-Seen 使用 10 个 held-out seed；
- Composite-Seen 和 Composite-Unseen 各使用 5 个 held-out seed；
- held-out rollout 禁止 reset、轨迹搜索和在线修改 memory；
- 每个实验条件必须使用同一份 seed 清单，禁止按结果筛选 seed。

Harness VLA 论文正文对 composite 的 seed 描述与表格计数存在歧义。实施前必须从其发布代码或
数据清单解析 `s0` 和 held-out seed 的准确标识，并将解析后的不可变清单提交到仓库。当前项目中
使用过的随机种子 `1000` 不能未经映射就被假定为论文中的 `s0`。

## 4. 实验条件

### H0：Frozen VLA Direct

- 完整环境根任务直接传给冻结 VLA；
- 无高层 Agent 分解；
- 无 analytic primitive；
- 无 Task Specific Memory 或 Global Memory；
- 允许执行到任务官方 horizon 或提前成功。

H0 是每个 VLA checkpoint 自己的直接基线。

### H1-B1：Hey Robot 当前层级规划

- 使用统一 Hey Robot Agent；
- 使用 `inspect_scene` 和 `manipulate`；
- Agent 正常规划并在 bounded option 后重新观察；
- VLA 使用当前经过验证的 prompt 契约；
- 无跨 episode memory；
- 无额外 analytic staging primitive。

### H1-B2：Hey Robot frozen-oracle pattern

- `inspect_scene`；
- 用完整根任务调用 `manipulate`；
- 到 option 边界后重新观察；
- 重复至官方成功、环境终止或预算耗尽。

H1-B2 用于区分“延长并闭环执行 VLA”与“真正的高层任务分解”。

### H2：Hey Robot Zero-shot Harness

- 使用现有统一 Agent、Skill OS 和 Robot Runtime；
- `manipulate` 作为接触密集的冻结 VLA primitive；
- 开放固定、通用的 analytic primitive；
- 支持局部 staging、post-condition、失败诊断和 retry；
- 不读取 Task Specific Memory 或 Global Memory。

### H3：H2 + Task Specific Memory

- 每个任务只从 reference seed `s0` 构建一份成功轨迹；
- memory 包含参数化 JSONL primitive trace 和语义审计摘要；
- held-out seed 只复用程序结构，所有对象和空间参数必须从当前观察重新 grounding；
- 只有官方 predicate 成功的 bootstrap rollout 才能生成正向 memory。

### H4：H3 + Global Memory

- 增加跨任务的 primitive 成功规则；
- 增加空抓、假视觉成功、错误 staging、无进展和不稳定接触等失败模型；
- Global Memory 只能提供决策上下文，不能包含 held-out seed 的状态或动作轨迹。

H4 是与完整 Harness VLA 最接近的 Hey Robot 条件。

## 5. Analytic primitive 的公平接口

Harness VLA 的 RoboCasa primitive 包括 `move_to`、`rotate_pitch`、`set_gripper`、`release`、
`navigate_to`、`move_base` 和 `vla_act`。Hey Robot 对比条件需要提供功能等价接口，但必须遵守：

1. primitive 进入通用 Skill OS，不新增 RoboCasa 专用 Agent 工具；
2. 所有物理执行仍由唯一 Robot Runtime 和唯一 EpisodeManager 拥有；
3. planner 只能调用结构化 schema，不能直接发送 MuJoCo action；
4. 评测条件中双方可访问的 RGB、depth、proprioception 和几何信息必须相同；
5. 如果使用 simulator world coordinate、对象真值或 embedded solver，必须明确标记为
   `sim-privileged`，不能与 camera-only 结果混报；
6. 面向真实机器人可部署的 `camera-only` 条件应作为独立结果列；
7. primitive 集合在正式评测前冻结，held-out rollout 中禁止动态增加 task-specific skill。

建议把 `manipulate` 与论文的 `vla_act` 视为同一能力类别，而不是再增加一个 `vla_act` skill。

## 6. Bootstrap 与 Memory 协议

### 6.1 Bootstrap 阶段

reference seed `s0` 可以允许 reset 和探索，但必须预先冻结以下预算：

- 最大 reset 次数；
- 最大环境 step；
- 最大 VLA invocation；
- 最大 planner 调用和 token；
- 最大墙钟时间；
- memory 生成模型和 prompt。

不能对某个难任务无限探索直到成功。若预算内没有找到成功轨迹，该任务的 Task Specific Memory
状态为 `bootstrap_failed`，held-out 评测仍必须运行并如实报告。

### 6.2 Task Specific Memory

至少包含：

```json
{
  "task": "SteamInMicrowave",
  "source_seed": "s0",
  "verified_success": true,
  "semantic_stages": [],
  "parameterized_trace": [],
  "preconditions": [],
  "postconditions": [],
  "recoverable_failures": [],
  "source_artifact": "...",
  "content_sha256": "..."
}
```

不得保存可直接泄露 held-out 初始化状态的绝对坐标。正式评测前对 memory 计算 hash，并写入每个
trial manifest，保证评测期间未被修改。

### 6.3 Global Memory

Global Memory 只保存跨任务规则，例如：

- 怎样判断空抓；
- 什么情况下需要重新 staging；
- 哪类 fixture 交给 VLA，哪类自由空间运动交给 analytic primitive；
- 连续多少次 observation 无变化算无进展；
- 什么视觉证据不足以宣布阶段完成。

Global Memory 的数据来源、生成时间和包含哪些任务必须可审计，不能从 held-out 失败中在线学习后
继续计入同一轮正式结果。

## 7. 必做主实验

完整主表至少包含 H0、H1-B1、H1-B2、H2、H3、H4。

| 条件 | Agent | Analytic | Staging/Retry | Task Memory | Global Memory |
|---|---:|---:|---:|---:|---:|
| H0 | 否 | 否 | 否 | 否 | 否 |
| H1-B1 | 是 | 否 | option 级 | 否 | 否 |
| H1-B2 | 固定模式 | 否 | option 级 | 否 | 否 |
| H2 | 是 | 是 | 是 | 否 | 否 |
| H3 | 是 | 是 | 是 | 是 | 否 |
| H4 | 是 | 是 | 是 | 是 | 是 |

每个条件包含 340 个计分 rollout。完整六条件共 2040 个计分 rollout，另加 bootstrap 过程。

资源不足时，第一张可对外报告的最小主表为 H0、H2、H4，共 1020 个 rollout；H1-B1 和 H1-B2
仍须在随后补齐，不能仅凭单任务结果推断当前 Hey Robot 的整体表现。

## 8. 必做机制消融

### A1：Analytic primitive

比较 H4 完整系统与关闭 analytic primitive 的版本，回答成功率提升是否主要依赖确定性运动接口。

### A2：Staging 与 retry

比较：

- 第一次 VLA 局部失败即终止；
- 观察失败原因后重新 staging 并 retry。

### A3：VLA invocation cap

推荐设置 `max_vla_calls = 1, 2, 4, 8, unlimited`，分别绘制 Atomic-Seen、
Composite-Seen、Composite-Unseen 的成功率曲线。

### A4：Memory

比较无 memory、仅 Task Specific Memory、仅 Global Memory、两者同时启用。

### A5：VLA prompt

比较：

- `environment_root`；
- `agent_subgoal`；
- `environment_root + structured local context`。

在 PI0.5 对局部子目标的 steerability 没有得到多 seed 证据前，不改变生产默认值。

### A6：Option termination

比较：

- 固定 50 步；
- 视觉阶段成功检测；
- 成功检测加最大 horizon 兜底。

阶段检测器不能读取最终官方 predicate；最终官方 predicate 继续只由独立 evaluator 读取。

### A7：Planner

至少比较当前 DeepSeek planner 与一个 reasoning planner。若要复现 Harness VLA 的 Codex/Claude
Code planner 结果，必须冻结相同系统 prompt、primitive schema、memory 和预算。

## 9. 指标与统计

### 9.1 主指标

- `official_success`；
- Atomic-Seen、Composite-Seen、Composite-Unseen success rate；
- 50 个任务的 macro average；
- 按 340 个 rollout 加权的 overall success rate；
- `false_completion`；
- timeout rate。

### 9.2 执行指标

- environment action 数；
- task horizon 使用比例；
- VLA invocation 数；
- 每类 analytic primitive 调用数和比例；
- inspect/re-observation 数；
- staging、retry 和 no-progress 次数；
- planner step、token、费用；
- VLA 推理时间、planner 时间、总墙钟时间；
- 最终官方 predicate 在哪一类 primitive 后触发；
- failure stage 和标准化 failure mode。

### 9.3 Bootstrap 成本

单独报告每个任务的：

- 成功前 reset 次数；
- bootstrap environment step；
- VLA invocation；
- planner token 和费用；
- 墙钟时间；
- 是否在预算内获得成功 memory。

不能因为 seed 0 不计入成功率而隐藏其成本。

### 9.4 统计方法

- 报告 95% 置信区间；
- 同任务同 seed 条件使用配对检验；
- 同时报告 per-task 结果，避免 overall 掩盖少数完全失败任务；
- 不把不同 VLA checkpoint 的结果用于配对显著性结论；
- 所有失败和超时都进入分母。

## 10. 每个 trial 的可复现产物

每个 trial 至少保存：

- `manifest.json`：代码 commit、依赖锁、checkpoint、planner、prompt、condition、seed；
- `result.json`：官方成功、终止原因、计数与时延；
- `actions.jsonl`；
- `observations.jsonl`；
- `agent_trace.jsonl`；
- `skill_events.jsonl`；
- `primitive_calls.jsonl`；
- `memory_manifest.json`；
- `evaluator_truth.json`；
- `video.mp4`。

正式 batch 根目录还应保存：

- 不可变 target50 与 seed 清单；
- bootstrap 预算；
- primitive schema；
- planner 和 detector prompt；
- checkpoint revision；
- 汇总脚本版本；
- per-task 与 per-split CSV/JSON；
- 被排除 trial 列表及预先声明的排除规则。

## 11. 实施前置条件

当前系统不能直接开始完整对比，必须先完成：

1. 将 allowlist 从当前 17 个任务扩展到完整 target50；
2. 从 RoboCasa task registry 加载每个任务 300–2900 步的官方 horizon；
3. 同步设置 environment horizon、LeRobot episode length、Agent skill budget 和 benchmark timeout；
4. 固化论文 seed 标识到实际环境 seed 的映射；
5. 在现有 Skill OS 中提供通用 analytic primitive；
6. 增加 RGB-D、proprioception 和结构化 observation contract；
7. 实现 staging、post-condition、retry 和 no-progress detector；
8. 实现 bootstrap 与正式评测的隔离生命周期；
9. 实现带 hash 的 Task Specific Memory 和 Global Memory；
10. 接入 RLDX-1 同骨干轨道，或明确声明暂时只做 PI0.5 内部消融。

其中任务 horizon 是硬门槛。当前 LeRobot RoboCasa wrapper 默认 1000 步，无法公平覆盖官方 horizon
达到 2900 步的任务。

## 12. 分阶段执行

### 阶段 P0：协议与基础设施

- target50、seed、horizon 和预算全部冻结；
- 所有 trial 产物可复现；
- 保证 Agent 看不到官方成功 predicate；
- 测试无 memory 泄漏、无 seed 交叉污染。

### 阶段 P1：三任务 pilot

选择：

- `CloseFridge`：Atomic-Seen；
- `SteamInMicrowave`：Composite-Seen；
- `ArrangeBreadBasket` 或 `MakeIceLemonade`：Composite-Unseen。

先运行 H0、H1-B1、H1-B2，验证长 horizon、终止和指标链路。

### 阶段 P2：17 任务工程 pilot

使用当前 17 个任务，每任务至少 3 个 held-out seed，运行 H0、H1、H2、H3。该阶段只用于发现
系统问题，不作为 target50 正式结果。

### 阶段 P3：target50 最小主表

运行 H0、H2、H4，共 1020 个计分 rollout，并完成 50 个任务的 bootstrap 审计。

### 阶段 P4：完整主表与消融

补齐 H1-B1、H1-B2、H3 和 A1–A7，形成完整机制结论。

### 阶段 P5：RLDX-1 严格复现

在相同 checkpoint、planner、primitive、memory 和 seed 协议下与论文数字做 head-to-head 对比。

## 13. 结果表模板

| Method | VLA | Planner | Atomic | Comp-Seen | Comp-Unseen | Overall | Bootstrap Cost |
|---|---|---|---:|---:|---:|---:|---:|
| Direct VLA | RLDX-1 | — | 60.0 | 21.3 | 5.0 | 30.0 | 0 |
| Harness VLA（论文报告） | RLDX-1 | Codex | 91.6 | 56.3 | 13.8 | 待按协议计算 | 未完整披露 |
| Harness VLA（论文报告） | RLDX-1 | Claude Code | 79.4 | 47.5 | 15.0 | 待按协议计算 | 未完整披露 |
| H0 | PI0.5 | — | 待测 | 待测 | 待测 | 待测 | 0 |
| H1-B1 | PI0.5 | DeepSeek | 待测 | 待测 | 待测 | 待测 | 0 |
| H1-B2 | PI0.5 | Frozen pattern | 待测 | 待测 | 待测 | 待测 | 0 |
| H2 | PI0.5 | DeepSeek | 待测 | 待测 | 待测 | 待测 | 0 |
| H3 | PI0.5 | DeepSeek | 待测 | 待测 | 待测 | 待测 | 单独报告 |
| H4 | PI0.5 | DeepSeek | 待测 | 待测 | 待测 | 待测 | 单独报告 |
| Hey Robot 同骨干 | RLDX-1 | 同条件 | 待测 | 待测 | 待测 | 待测 | 单独报告 |

论文报告行必须保留“论文报告”标签；只有本项目按冻结协议实际执行的结果才能标为 reproduced。

## 14. 完成标准

只有满足以下条件，才认为 RoboCasa365 对比实验完成：

1. target50 的 340 个 rollout 在每个主条件下全部产生结果；
2. seed 0 不进入正式成功率，且 bootstrap 成本完整披露；
3. 每个任务使用官方 task-specific horizon；
4. Agent 与 memory 无法访问 held-out evaluator truth；
5. direct 与 harness 条件除待消融组件外保持一致；
6. 不同 checkpoint 的结果不被描述为同骨干对比；
7. 结果包含置信区间、per-task 表、失败类型和资源成本；
8. 所有代码、配置、prompt、memory hash、视频和 trial artifact 可追溯到明确 commit。
