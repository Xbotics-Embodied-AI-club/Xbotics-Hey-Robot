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
| XLeRobot 仿真（VLN 实验） | `configs/xlerobot.sim.vln.yaml` | VLN 模型服务链路验证 |
| 内部开发测试配置 | `configs/mock.dev.yaml` | 非部署环境 |
| 自动化测试配置 | `configs/mock.test.yaml` | 非部署环境 |

`xlerobot.sim.vln.yaml` 是实验 profile，不等同于默认仿真或真机能力面。VLN 需要
InternNav submodule 和独立模型环境。
