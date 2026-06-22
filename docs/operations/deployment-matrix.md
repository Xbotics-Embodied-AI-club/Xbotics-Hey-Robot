# 部署文档索引

按场景选择对应文档：

| 场景 | 文档 |
|---|---|
| XLeRobot 真机部署（含 VLA） | [xlerobot-real.md](./xlerobot-real.md) |
| XLeRobot MuJoCo 仿真部署 | [xlerobot-sim.md](./xlerobot-sim.md) |
| 飞书通道接入 | [feishu.md](./feishu.md) |
| 诊断和硬件脚本索引 | [runtime-scripts.md](./runtime-scripts.md) |

## 配置文件

命名约定：`{robot}.{env}.{os}.yaml`

| 配置 | 文件 | 场景 |
|---|---|---|
| XLeRobot 真机（Windows） | `configs/xlerobot.real.windows.yaml` | 真机 |
| XLeRobot 真机（Ubuntu） | `configs/xlerobot.real.ubuntu.yaml` | 真机 |
| XLeRobot 仿真（Windows） | `configs/xlerobot.sim.windows.yaml` | 仿真 |
| XLeRobot 仿真（Ubuntu） | `configs/xlerobot.sim.ubuntu.yaml` | 仿真 |
| Mock 开发环境 | `configs/mock.dev.yaml` | 开发 |
| Mock 测试环境 | `configs/mock.test.yaml` | 测试 |
