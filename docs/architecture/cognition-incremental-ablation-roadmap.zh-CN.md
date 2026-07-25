# Cognition 渐进增量与消融路线

> 日期：2026-07-25
>
> 本文记录可能重新加入的机制，不是当前实现清单。默认答案是“不加入”。

## 1. 当前最小 baseline

当前 cognition 只承诺两个产品能力：可交互，以及可跨物理等待和进程重启继续的
long-horizon task。主路径保持为：

```text
ConversationTurn / ResumeTrigger
              ↓
       per-session Agent
              ↓
          AgentRunner
              ↓
     one PreparedToolCall
              ↓
       AgentToolExecutor
              ↓
 ToolOutcome / durable run receipt
              ↓
       continue or yield
```

baseline 保留以下不可绕过的不变量：

- 每次模型决策最多一个工具调用；同一 task 最多一个 active physical run；
- 物理请求先持久化 task step/run receipt，再提交给 Skill transport；
- Skill terminal event 按 sequence 幂等归约；
- terminal outcome 形成 durable wakeup，重启后继续但不重放旧动作；
- objective amendment 持久化，physical-running steer 在安全点生效；
- pause、cancel、emergency stop 不依赖 LLM；
- task 受 max steps 和 deadline 限制；
- ToolOutcome 和 evidence 由 Store 自动保存；
- `environment_done` 是权威完成条件；显式 `complete_task` 只要求已有成功步骤。
- system policy、runtime messages 和 function schemas 保持独立；工具能力只在各自
  schema 中描述，不拼入 System prompt。

baseline 明确不包含：独立 planner/task graph、sub-agent、长期记忆、第二次 completion
LLM、模型手工 evidence ID、强制 reobservation gate、独立 entity cache、动态 provider
framework 和 self-improvement。

## 2. 新机制的准入门槛

候选机制只有同时回答下列问题才进入实现：

1. 对应哪个可以稳定复现的失败，而不是哪个理论上可能发生的问题？
2. 使用哪个固定评测集和指标判断改善？
3. 能否只改一个边界，避免同时引入多个机制？
4. 与 baseline 相比增加多少生产代码、模型调用、token 和持久状态？
5. 关闭 feature flag 后能否恢复 baseline，并得到真正的 A/B 结果？
6. 如果没有显著收益，哪些代码和 schema 可以完整删除？

每次只加入一个机制。实验报告至少记录 task success、false completion、用户干预
延迟、crash recovery success、model calls/step、token/task 和 wall time/task。安全相关
失败单独列出，不能被平均成功率掩盖。

## 3. 候选增量

### A1. 动作后确定性重新观察

- 触发失败：Agent 在动作后依据“命令执行成功”误判世界状态，或连续动作累积状态漂移。
- 最小实现：Skill outcome 声明 world-state invalidation；Executor 只阻止下一个会改变世界
  的 Skill，observation Skill 清除 invalidation。
- 对照：当前由模型自行选择 observation 的 baseline。
- 主要指标：false completion、无效连续动作、额外 model/observation calls。
- 删除条件：成功率无改善，或额外观察显著增加延迟而错误率不下降。

不要把具体 Skill 名（例如 `inspect_scene`）写入 Service；策略只能依赖通用 intent 或
capability contract。

### A2. 完成审计 verifier

- 触发失败：开放任务出现足够多、可复现的 false completion，且结构化 predicate 无法覆盖。
- 最小实现：只审计显式 `complete_task`；`environment_done` 永远跳过；使用独立窄 prompt
  返回结构化 accept/reject。
- 对照：成功步骤确定性检查。
- 主要指标：false completion、false rejection、额外模型调用和 token。
- 删除条件：审计与主模型高度一致，或拒绝后不能产生有效恢复动作。

### A3. 模型选择 evidence

- 触发失败：任务有多个互相冲突或过期 outcome，自动使用近期成功步骤导致误完成。
- 最小实现：先尝试 completion tool 的单个 `evidence_id`，不要一开始设计复杂 evidence DSL。
- 对照：Store 自动选取当前任务成功步骤。
- 主要指标：冲突证据任务的完成精度、无效 tool-call 参数率、prompt token。
- 删除条件：模型经常引用无效 ID，或与自动选择相比没有改善。

### A4. resume 时新鲜观察

- 触发失败：pause 时间较长后环境发生变化，Agent 从旧 outcome 继续导致错误动作。
- 最小实现：通用 freshness policy 向 Agent 注入“状态可能过期”，由 Agent 选择 observation；
  只有数据证明软提示不足时才升级为 gate。
- 对照：直接从最近 durable outcome 继续。
- 主要指标：resume 后首步错误率、恢复延迟、额外 observation calls。
- 删除条件：短暂停和长暂停均无可测差异。

### A5. Scene/entity context

- 触发失败：同一会话频繁重复观察，只为重新获得稳定实体引用。
- 最小实现：从最新 observation outcome 投影少量、有 freshness 的实体；不建立独立订阅、
  alias catalog 或自然语言 resolver。
- 对照：显式 observation Skill。
- 主要指标：重复观察数、引用错误率、context token。
- 删除条件：实体引用没有被后续 Skill 消费，或 freshness 错误抵消收益。

### A6. Context compaction

- 触发失败：长任务达到 context window、token 成本过高，或早期无关对话干扰当前决策。
- 最小实现：只压缩 conversation transcript；objective、amendment、step/outcome、pending
  run receipt 始终从结构化 Store 投影，不能被 summary 替代。
- 对照：当前 recent transcript + recent steps。
- 主要指标：token/task、长任务成功率、关键信息丢失率。

### A7. Lifecycle metrics/events

- 触发失败：现有日志无法测量上述实验，或 UI 必须展示 tool progress。
- 最小实现：从已有 Agent/Coordinator 边界发布少数事件；subscriber 负责 trace 和 metrics，
  决策路径不依赖 subscriber。
- 对照：直接 callback 和现有 runtime events。
- 主要指标：能否回答实验问题、事件量和运行开销。

### A8. Affordance/failure memory

- 触发失败：跨 episode 重复发生同类、可归因的失败，且当前 observation 无法避免。
- 最小实现：先离线生成只读、可追溯的少量 failure hints，再评估是否在线写入。
- 对照：无长期记忆。
- 主要指标：重复失败率、错误迁移率、token/task。

这是优先级最低的近期候选。没有稳定 episode 数据集之前，不实现 memory curator、
reflection loop 或自动写 prompt。

## 4. 明确延后

以下方向只有在单 Agent、单物理 run baseline 出现明确能力上限后才重新评估：

- planner、task graph 和 workflow DSL；
- sub-agent、并行 agent 和动态 delegation；
- task-specific/global long-term memory 系统；
- 自动生成 Skill、自动修改 prompt/code 的 self-improvement；
- 多 provider registry、capability matrix、自动 fallback router；
- 默认并行物理工具。

模型 transport 继续使用官方 OpenAI SDK，cognition 只维护薄的 canonical model protocol
和 adapter。是否增加第二个 provider，应由真实部署需求触发，而不是预先抽象。

## 5. 推荐实验顺序

先运行 baseline 并建立固定任务集。若没有明确失败，不增加任何机制。出现失败后按最窄
原因选择候选，而不是按编号全部实现：

```text
动作后状态误判 ──→ A1
开放任务误完成 ──→ A2；有冲突证据时再考虑 A3
暂停后环境漂移 ──→ A4
重复实体观察 ──→ A5
context/token 超限 ──→ A6
无法测量实验 ──→ A7
跨 episode 重复失败 ──→ A8
```

每个实验独立提交，保留 baseline 配置和结果。只有指标改善超过预先设定阈值、额外成本
可接受，并通过 crash recovery、steer、cancel 和 emergency-stop 回归，才成为默认能力。
