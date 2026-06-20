# 开发流程

## 分支

从 `main` 拉分支，不直接在 `main` 上提交。

```bash
git checkout main
git pull
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
feature/embodied-agent-runtime
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
poe style
poe lint
poe test
```

- `poe style`：ruff 格式化 + 自动修复
- `poe lint`：ruff 检查 + mypy 类型检查
- `poe test`：全量测试（`pytest -q`）

只有三个命令全部通过才能提交。如果 `poe test` 失败，先修测试，不要跳过。

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
src/hey_robot/agents/core.py        -> tests/agents/test_core.py
src/hey_robot/skills/catalog.py     -> tests/skills/test_catalog.py
src/hey_robot/robots/xlerobot/...   -> tests/robots/test_xlerobot.py
```

### 最低通过标准

- 不引入新的 ruff 警告
- 不引入新的 mypy 类型错误
- 全量测试通过（`poe test`）
- 新增代码有对应测试

## 开发环境

```bash
# 安装依赖
uv sync --dev

# 确认版本
uv run python -c "import sys; print(sys.version)"  # 必须是 3.12.x
```

## 代码风格

项目已配置 ruff 和 mypy，不要绕过。风格和类型检查的规则在 `pyproject.toml` 中定义。不要在不理解的情况下使用 `# type: ignore`、`# noqa` 等抑制注释。
