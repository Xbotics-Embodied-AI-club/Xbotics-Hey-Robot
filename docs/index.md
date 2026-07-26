# Hey Robot 文档

本文是仓库内文档的统一索引。运行行为以当前代码、配置和锁文件为准；外部文章、活动材料、
论文草稿和带日期的审计报告不是规范性事实源。

## 新用户

1. 从项目 [README](../README.md) 完成默认 Ubuntu MuJoCo 快速开始；
2. 阅读 [配置参考](reference/configuration.md)，确认所选 YAML 的消息总线、通道、机器人、
   Skill surface 和模型服务；
3. 运行 `uv run hey-robot inspect --config <yaml>`；
4. 根据场景进入对应运行手册。

## 运行手册

| 场景 | 文档 | 配置事实源 |
|---|---|---|
| XLeRobot Ubuntu/Windows 仿真 | [XLeRobot 仿真](operations/xlerobot-sim.md) | `configs/xlerobot.sim.*.yaml` |
| XLeRobot 真机 | [XLeRobot 真机](operations/xlerobot-real.md) | `configs/xlerobot.real.*.yaml` |
| 飞书通道 | [飞书接入](operations/feishu.md) | `channels.<id>` |
| RoboCasa365 评测 | [RoboCasa365 runbook](evaluation/robocasa365/runbook.zh-CN.md) | `configs/evaluation/` |
| 脚本与诊断 | [运行脚本索引](operations/runtime-scripts.md) | `scripts/` |

[部署矩阵](operations/deployment-matrix.md)列出仓库提供的主要 profile。配置文件存在不等于
该硬件、模型 checkpoint 或云服务已经在你的环境中验证。

## 架构与扩展

- [系统架构](architecture/system-architecture.md)：当前运行拓扑和边界，架构事实主文档；
- [论文草稿](references/paper-draft.md)：研究动机、系统主张、局限与待执行实验，不是能力事实源；
- [部署模式边界](architecture/deployment-modes.zh-CN.md)：`in_memory`、NATS 与 sidecar；
- [ModelService 协议](architecture/model-service-rpc-proto.md)：proto、gRPC 和 codegen；
- [Skill 扩展指南](development/skill-extension.md)：新增或修改 Skill；
- [质量门禁](overview/quality-gates.md)：合并前检查；
- [贡献指南](../CONTRIBUTING.md)：开发与 PR 约定。

## 研究与非规范性材料

[最小 Embodied Agent Harness 开发指南](development/minimal-embodied-agent-harness-guide.zh-CN.md)
对照四篇材料、pi-agent-core、RPent 与当前代码，给出配置驱动、模块边界、核心收敛和
分层验证顺序。

带日期的维护审计用于解释某次提交的状态，不应用来生成部署配置或判断当前 API。历史重构
记录应通过 Git 历史追溯。项目自己的 [论文草稿](references/paper-draft.md) 随当前代码维护，
但研究假设和计划实验不构成已交付能力。`docs/references/` 中的第三方论文转录是非规范性
研究材料，必须逐份确认来源与再分发许可；根目录 MIT License 不自动覆盖这些内容。

当前代码事实与历史文档冲突时，优先级为：

1. proto、Python 类型和配置解析/校验代码；
2. `configs/`、`pyproject.toml` 和 `uv.lock`；
3. 当前架构文档与运行手册；
4. README 示例；
5. 带日期的审计、重构记录、论文草稿和外部文章。

## 文档维护规则

- 运行命令统一使用 `uv run ...`；依赖组以 `pyproject.toml` 为准。
- 一个概念只保留一个规范性文档，其余页面链接到它，不复制字段清单。
- 文档提到配置能力时同时写明 profile；不要把“已注册”写成“Agent 可见”。
- Agent 可见能力只以 `skills.tools` 为准；模型服务声明不等于 Skill 已开放。
- 所有本地 Markdown 链接必须存在；删除文档时在同一 PR 修复引用。
- 带测试数量、覆盖率、版本或验证结果的报告必须写提交 SHA 和日期。
- 行为、CLI、配置、协议或安全边界变更必须在同一 PR 更新对应文档。

文档现状和后续治理项见
[2026-07-25 文档审计](maintenance/documentation-audit-2026-07-25.zh-CN.md)。
