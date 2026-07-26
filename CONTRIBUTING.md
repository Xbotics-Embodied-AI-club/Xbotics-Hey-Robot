# 开发流程

感谢参与 Hey Robot。提交代码前，请先用 Issue 或 Pull Request 描述问题、影响范围和验证
方式。涉及真机动作、安全边界、公开协议或配置兼容性的变更，应在 PR 中明确风险与回滚
方式。

## 分支

从 `main` 拉分支，不直接在 `main` 上提交。

```bash
git checkout main
git pull --ff-only
git checkout -b <your-branch>
```

### 分支命名

```
{type}/{short-description}
```

| type | 用途 |
| --- | --- |
| `feature/` | 新功能、新模块 |
| `fix/` | bug 修复 |
| `refactor/` | 重构（不改变外部行为） |
| `chore/` | 构建、依赖、配置维护 |
| `docs/` | 文档 |
| `test/` | 测试补充 |

用 `-` 连接词，英文小写，控制在 3-5 个词以内：

```
feature/embodied-agent-harness
fix/sim-camera-calibration
refactor/skill-backend-decouple
```

## Commit

遵循 conventional commits 格式：

```
{type}({scope}): {简短说明}
```

| type | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `refactor` | 重构 |
| `test` | 测试 |
| `docs` | 文档 |
| `chore` | 杂项（lint、构建、依赖） |

scope 可选，用于标明影响模块。说明用英文，小写开头，不加句号。

示例：

```
feat(skills): add semantic-only agent-visible skill surface
fix(sim): align forward movement with visual chassis direction
refactor(vla): move adapter factory to driver methods
chore: fix lint and style issues to make CI gates green
```

## 提交前检查

提交前必须跑通以下三个命令，缺一不可：

```bash
uv run poe style
uv run poe lint
uv run poe test
```

- `uv run poe style`：ruff 格式化 + 自动修复
- `uv run poe lint`：配置校验 + ruff 检查 + mypy 类型检查
- `uv run poe test`：全量测试和覆盖率收集（`pytest -q`）

`style` 会修改文件，运行后应重新检查 diff，再执行 `lint` 和 `test`。

## 测试要求

### 新代码必须写测试

以下情况必须有测试覆盖：

- 新增的公开函数、类、方法
- 新增的 API 端点
- 修改的行为逻辑（原测试不再覆盖时）
- 修复的 bug（补回归测试）

纯重构（不改变外部行为）可以沿用现有测试。

### 测试落位

```
tests/{module_name}/test_{file_name}.py
```

示例：

```
src/hey_robot/cognition/core.py        -> tests/cognition/test_core.py
src/hey_robot/skills/runner.py         -> tests/skills/test_runner.py
src/hey_robot/robot_backends/xlerobot/... -> tests/robot_backends/xlerobot/...
```

### 最低通过标准

- 不引入新的 ruff 警告
- 不引入新的 mypy 类型错误
- 全量测试通过（`uv run poe test`）
- 新增代码有对应测试

## 开发环境

```bash
# 全量非 GPU 开发与测试环境
uv sync \
  --extra gateway \
  --extra agent \
  --extra robot \
  --extra voice \
  --group sim \
  --group dev

# 确认版本
uv run python -c "import sys; print(sys.version)"  # 必须是 3.12.x
```

`dev` 只包含 lint、类型检查和测试工具，不重复声明各服务的运行依赖。需要执行
Torch/Ultralytics 专项测试时额外同步 `--extra human-follow`；LeRobot Policy、VLN 和
RoboCasa365 使用各自隔离的 dependency group 与环境，不要混装到上述开发环境。

## 代码风格

项目已配置 ruff 和 mypy，不要绕过。风格和类型检查的规则在 `pyproject.toml` 中定义。不要在不理解的情况下使用 `# type: ignore`、`# noqa` 等抑制注释。

## Pull Request

PR 至少应包含：

- 变更动机和用户可见行为；
- 测试命令与结果；
- 配置、协议、数据库或运行产物是否兼容；
- 涉及真机时的仿真验证、安全措施和回滚方式；
- 文档是否同步更新；若不需要，说明原因。

不要提交 API key、访问令牌、真实用户标识、模型权重或包含敏感画面的运行产物。
安全漏洞不要提交公开 Issue；在项目建立正式私密披露渠道前，请先通过仓库维护者的私密
联系方式报告。
