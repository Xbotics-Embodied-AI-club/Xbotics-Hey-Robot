# 文档审计：2026-07-25

> 初始基线：提交 `f70e118`，分支 `refactor/simple-tool-skill-harness`。本文随后记录同一轮
> 文档治理的实际处理结果，但仍不是架构规范；当前行为以代码、配置和
> [系统架构](../architecture/system-architecture.md) 为准。

## 结论

文档的主要问题不是文字陈旧，而是事实源和受众混在一起：

- README 曾链接 7 个已经删除的文档，并把外部飞书页面当作“完整配置”；
- 默认仿真说明与 YAML 相反：实际是 `in_memory` + Web-only，不需要 NATS/Voice/Feishu；
- 当前架构已收敛为 local Skill execution；论文草稿已重写为当前 `AgentTaskStore`、
  `TaskCoordinator`、`SkillWorker` 和 `RobotRuntime` 主链；
- 生产运行手册、重构记录、架构审计、论文全文转录并列在 `docs/`，没有稳定/历史/第三方
  材料边界；
- 开源治理文档和自动化检查不足，文档漂移没有 CI 阻止。

本次已经修复 README 的断链与默认启动路径，新增仓库内文档索引和配置参考，并把贡献指南
移动到 GitHub 可自动发现的根目录。

## P0：合并或发布前处理

| 问题 | 代码证据 | 建议 |
|---|---|---|
| Docker runtime 文档声称可启动，但 compose 使用不存在的 `/app/configs/deployment/mock.dev.yaml` | `docker-compose.yml` | 修复 compose 后增加 build/start smoke test；修复前把整套 Docker 部署标为实验性 |
| 第三方论文全文/转录仍存放在 `docs/references/` | `docs/references/*.md` | 逐份补充来源和再分发许可；不能确认权利时删除全文，只保留题录、上游链接和原创摘要 |
| 没有私密漏洞披露入口和 `SECURITY.md` | 仓库根目录、`.github/` | 启用 GitHub Private Vulnerability Reporting，再写明支持版本、响应目标和披露流程 |
| 没有 CI workflow | `.github/workflows/` 不存在 | 至少自动运行链接检查、`poe lint`、`poe test` 和 Docker/config smoke |
| 两套 Poe 任务定义曾经冲突 | `pyproject.toml` 与 `poe_tasks.toml` | 本次已收敛到独立 `poe_tasks.toml`，并增加重复定义检查 |

`SECURITY.md` 不应在没有真实私密联系方式或已启用平台私密报告能力时填一个猜测邮箱。

## P1：应更新或重构

### 运行文档

- `operations/xlerobot-sim.md` 本次已修正 viewer、VLA 进程边界、命令入口和 Docker
  稳定性表述；后续仍应：
  - 把普通 `in_memory` 与实验 NATS 拓扑拆成两张图；
  - 从镜像实际构建结果核对依赖和入口；
  - 为“某 checkpoint 已通过”附日期、commit、artifact manifest，避免永久能力承诺。
- `operations/xlerobot-real.md`
  - 安装和启动命令已统一为 `uv sync --group dev`、`uv run ...`；
  - 真机 profile 默认启用 Voice/Feishu，生产部署需要额外写清凭据缺失时的失败语义、网络
    暴露、急停演练和最小权限。
- `evaluation/robocasa365/runbook.zh-CN.md` 本次已替换旧 Skill OS 术语和失效 lint 路径；
  “已验证成功”的产物目录仍不在版本控制中，公开复现需要发布 manifest、硬件/GPU/driver
  元数据和可校验摘要。

### 架构文档

- `architecture/system-architecture.md` 可继续作为当前唯一架构主文档。
- 本次已删除过时的架构审计/重构快照；当前架构只保留
  `architecture/system-architecture.md` 作为主文档。
- 本次已停用无生产调用方的 `skill.intent`、`robot.action` 和旧 Skill control/result
  总线执行面，避免形成第二条未受支持的机器人执行入口。项目已经是 `1.0.0`，相关公开
  DTO 和 Topic 名称仅为 1.x 下游源码兼容保留，计划删除时必须走弃用和大版本流程。
- 配置没有 schema version；文档可以说明现状，但长期应由代码提供机器可读 schema 和迁移
  策略。

### 开源贡献体验

建议新增：

- `CODE_OF_CONDUCT.md`：采用明确版本的 Contributor Covenant，并填写真实执行联系人；
- `SUPPORT.md`：区分使用问题、bug、安全问题和硬件事故；
- Issue/PR templates：要求复现配置、平台、commit、日志脱敏和真机安全信息；
- release notes / `CHANGELOG.md`：记录配置、协议和部署的破坏性变化；
- maintainer/ownership 文档：至少覆盖 runtime、robot hardware、model service 和 safety。

## P2：删除、归档或降级

| 内容 | 处理建议 | 原因 |
|---|---|---|
| 已删除实现对应的重构计划 | 删除或移到外部设计记录 | 用户无法执行，且容易被搜索结果当成现状 |
| 旧 `paper-draft.md` 内容 | 本次已按当前代码重写，并明确标记为非规范性草稿 | 原稿包含已删除类名、目录和未经验证的能力表述 |
| 第三方论文全文/转录 | 建议完成许可审查；不能确认时删除全文 | MIT 根许可证不应暗示覆盖第三方内容 |
| 精确测试数、覆盖率、GPU 显存和耗时 | 只保留在带 SHA 的报告/CI artifact | 高频变化，不适合常青文档 |
| 外部飞书“完整配置指南” | 降级为补充教程 | fork、离线和长期可维护性差，无法随 PR 原子更新 |

## 推荐的信息架构

```text
README.md                         项目定位 + 最短可运行路径
CONTRIBUTING.md                   贡献入口
SECURITY.md                       私密漏洞披露与支持版本
docs/index.md                     文档总索引和事实优先级
docs/tutorials/                   首次运行、学习路径
docs/how-to/                      仿真、真机、飞书、评测
docs/reference/                   配置、CLI、协议、环境变量
docs/architecture/                当前架构与 ADR
docs/maintenance/                 带日期审计和维护手册
docs/archive/                     明确非规范性的历史材料
research/                         论文草稿与原创研究笔记
```

迁移时不必一次改完目录；先保证每份文档标明受众、稳定性和事实源，再逐步移动。

## 建议自动化门禁

1. 校验所有本地 Markdown 链接存在；
2. 扫描文档中的 `src/`、`scripts/`、`configs/` 路径是否存在；
3. 从 `hey-robot --help` 和 Poe task list 生成或核对 CLI 文档；
4. 对所有公开 profile 运行 `hey-robot inspect`；
5. 检查 README 中不出现已删除的架构类名和配置字段；
6. 对 Docker Compose 执行 config 校验和最小容器 smoke；
7. PR 模板要求行为改动勾选对应文档。

## 完成标准

- README 的默认路径可以从干净 checkout 复现；
- 仓库内不存在失效的相对 Markdown 链接；
- 当前架构、配置、CLI 和 Skill surface 各只有一个规范性事实源；
- 历史/研究材料不会被误认为运行手册；
- 安全披露、支持、贡献、发布和许可证边界对外明确；
- 文档漂移由 CI 自动发现，而不是依赖人工审查记忆。
