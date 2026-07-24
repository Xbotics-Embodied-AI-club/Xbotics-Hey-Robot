# Hey Robot Tool 与 Skill 重构收口

## 1. 范围与目标

本文记录本次重构的收口边界。总体设计和固定部署边界见
[Tool 与 Skill 总体重构目标](simple-tool-skill-refactoring.zh-CN.md)。

重构目标是保持一个简单、通用、配置驱动的 Embodied Agent Harness：

- Agent/Cognition 是长程任务慢系统；
- SkillRunner、ModelService 和 Robot Runtime 是有界执行快系统；
- Tool 是 Agent 可见的 Skill 投影，Agent 不直接构造机器人动作；
- classic、VLA、VLN 和 hybrid 共用 SkillCommand/SkillEvent/SkillResult；
- 每个模块只有一种生产部署方式，不维护 local/remote 两套兼容实现；
- ModelService、Embodiment、Environment 和 endpoint 由配置组合；
- Robot Runtime 始终拥有最终动作准入、安全检查和急停权；
- SQLite、JSONL 和 artifact 文件是运行事实，NATS 只投影通知事件。

本次架构重构已经达到目标，不再为了完善运行细节增加核心抽象、状态字段、兼容层或调度机制。
后续验证由真实部署需求触发，不作为本次架构重构的阻塞条件。

## 2. 非阻塞验证

### 2.1 使用 RoboCasa365 前

- 先完成一条固定任务、固定 seed 的 Agent end-to-end smoke；
- 只有进入批量评测时，再增加多任务、多 seed 和常驻 ModelService 验证；
- 只有进入持续运行或生产部署时，再执行 crash recovery 和 30 分钟泄漏测试。

### 2.2 接入 XLeRobot 真机前

- 先验证 stop、单步 classic action 和 emergency stop；
- 再验证一条 VLA/VLN 单步 action 或 action chunk；
- fresh-frame timeout、ModelService disconnect 和长程任务在实际部署前验证；
- 只有实际选择 `fastwam` checkpoint 时，才执行对应启动、单步推理和 action contract smoke。

新增 checkpoint、机器人或 environment 时，优先只修改配置、mapping 或 LeRobot plugin；验证失败本身
不构成增加新 executor、远程 Skill transport 或新状态机的理由。

## 3. 不再继续的工作

- 不把 termination reason、retryable 或 re-observation 拆成更多核心状态类型；
- 不为普通 cancel 增加第二条控制协议；需要立即停止物理运动时使用既有 emergency stop；
- 不为每个 bounded step 增加新的持久化实体，RunStore 继续保存紧凑结果和 execution trace artifact；
- 不为 Gateway 增加通用查询服务或 event replay 系统；真实 UI 出现多用户查询需求时再扩展现有 TaskStore 查询；
- 不维护 local/remote 两套 Skill Runtime，也不按 VLA、WAM、benchmark 或机器人复制 executor。

修改核心边界后的最低收口命令：

```bash
uv run --no-sync ruff check src tests evaluation
uv run --no-sync python -m pytest -q --no-cov
git diff --check
```
