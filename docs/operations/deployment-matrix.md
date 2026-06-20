# 部署文档

按部署场景选择对应文档：

| 场景 | 文档 |
| --- | --- |
| XLeRobot 真机部署（含 VLA） | [xlerobot-real.md](./xlerobot-real.md) |
| XLeRobot MuJoCo 仿真部署 | [xlerobot-sim.md](./xlerobot-sim.md) |
| 飞书通道接入 | [feishu.md](./feishu.md) |
| 诊断和硬件脚本索引 | [runtime-scripts.md](./runtime-scripts.md) |

## 配置文件

| 配置 | 文件 |
| --- | --- |
| XLeRobot 真机（Windows） | `configs/xlerobot.real.windows.yaml` |
| XLeRobot 仿真（Windows） | `configs/xlerobot.sim.windows.yaml` |
| XLeRobot 仿真（Ubuntu） | `configs/xlerobot.sim.ubuntu.yaml` |
| Mock 开发环境 | `configs/mock.dev.yaml` |
| Mock 测试环境 | `configs/mock.test.yaml` |

命名约定：`{robot}.{env}.{os}.yaml`，其中 `env` 为 `real` / `sim` / `mock`。
