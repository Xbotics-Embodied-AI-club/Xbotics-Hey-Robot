# Tool/Skill 与 VLA 重构 RoboCasa365 评估记录

日期：2026-07-24

## 1. 评估目的

验证重构后的真实执行链：

```text
Agent Tool Registry
  -> typed prepared call
  -> ToolDispatcher
  -> SkillCallProposal / TaskCoordinator
  -> SkillRunner
  -> VLAOptionRunner
  -> ModelService
  -> Robot Runtime
  -> RoboCasa365 official evaluator
```

成功只能由 RoboCasa365 官方谓词确认，不能根据 Agent 文本或 `SkillResult.success` 推断。

## 2. 环境与协议

| 项目 | 值 |
|---|---|
| Task | `CloseFridge` |
| Split | RoboCasa365 target |
| Seed | `1000` |
| Condition | `b1` |
| VLA | `lerobot/pi052_robocasa` |
| Prompt mode | `environment_root` |
| Runtime | managed RoboCasa Runtime + gRPC ModelService |
| Evaluator | 独立 RoboCasa official predicate |

启动入口：

```bash
bash scripts/evaluation/run_robocasa365_full_system.sh \
  --task CloseFridge \
  --seed 1000 \
  --condition b1 \
  --output-dir runtime/robocasa365/refactor-close-fridge-b1-seed1000-20260724-v3 \
  --timeout-sec 1800
```

## 3. 最终结果

| 指标 | 结果 |
|---|---:|
| `official_success` | `true` |
| `episode_done` | `true` |
| `termination_reason` | `episode_done` |
| `false_completion` | `false` |
| `failure_stage` | `null` |
| environment action | 520 |
| Agent planner step | 28 |
| VLA option record | 8 |
| observation | 221 |
| duration | 553.894 s |

Agent 在环境成功时仍保持 active，没有抢先调用 `complete_task`；benchmark 由独立环境
`episode_done` 结束，因此不存在 false completion。

最终 artifact 位于：

```text
runtime/robocasa365/refactor-close-fridge-b1-seed1000-20260724-v3/
```

其中包括 520 行 `actions.jsonl`、8 行 `options.jsonl`、22 行 `skill_events.jsonl`、
221 行 `observations.jsonl`、官方 truth、Agent trace 和 `video.mp4`。

## 4. 实际仿真发现的问题

### 4.1 fresh observation 门槛

第一次 trial 暴露了等价抽取偏差：下一次观察错误地使用动作返回的最新 frame 作为
`after_frame_id`，Runtime 因无法立即提供更高 frame 而返回 `observation_stale`。

修复后恢复原有语义：使用本次动作前 observation frame 作为 freshness 下界。新增回归测试
固定 `None -> previous observation frame` 的调用序列。最终 trial 中没有 stale failure。

### 4.2 option 统计投影

第二次 trial 已获得 `official_success=true`，但 `option_count=0`。原因是 Gateway 当前 payload
输出 `skill`，评测器仍按旧 `name/timeline` 格式过滤。

修复后 Gateway 明确输出 `name` 和提交时 `envelope`，评测器通过 `envelope.chat_id` 关联 trial，
同时保留旧 timeline 读取兼容性。最终 trial 正确记录 8 个 option。

## 5. 结论

本次门禁证明：

1. Tool/Skill 分层没有阻断 Agent 参数进入 VLA Skill；
2. 普通 Tool 扩展通道没有改变现有机器人 Skill 执行路径；
3. VLA 在一个语义 Tool 内连续执行 action chunks，高层 Agent 不处理 action tensor；
4. option budget、subgoal success 和 task success 已分离；
5. 现有 Skill Worker、ModelService、Robot Runtime、Gateway 和分布式边界保持有效；
6. option、动作、观察和官方 truth 可以在同一 trial 下审计。

这是一项单任务工程门禁，不代表 RoboCasa365 target50 的总体成功率。对外报告模型或 Harness
性能仍需按既定协议运行多任务、多 seed 和对照条件。
