# XLeRobot 仿真部署

本文说明如何在本地 MuJoCo 中运行 XLeRobot 仿真，用于验证仿真驱动、场景文件、相机观测和 runtime 配置。

## 依赖

```bash
uv sync --extra sim
```

如果只需要补 MuJoCo：

```bash
uv pip install "mujoco>=3.3.0"
```

## 生成仿真模型

`assets/robots/xlerobot/xlerobot.xml` 是生成文件，不建议手工编辑。

```bash
python scripts/robots/xlerobot/generate_mjcf.py
```

生成器会读取本地官方仓库：

```text
D:\agent_robot\XLeRobot
```

并生成：

```text
assets/robots/xlerobot/xlerobot.xml
assets/robots/xlerobot/xlerobot.official.generated.urdf
assets/robots/xlerobot/meshes/
```

## 快速验证

运行仿真测试：

```bash
pytest tests/robots/test_simulation.py -q --no-cov
```

启动仿真 deployment：

```bash
hey-robot run --config configs/xlerobot.sim.windows.yaml
```

运行 agent 前需要配置 LLM provider 的 API key，例如 `DEEPSEEK_API_KEY`。

## 最小配置

```yaml
robots:
  sim_robot:
    type: xlerobot_sim
    enabled: true
```

常用配置项：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `mjcf_path` | `assets/robots/xlerobot/scene.xml` | MuJoCo 场景文件 |
| `render_width` | `640` | 渲染宽度 |
| `render_height` | `480` | 渲染高度 |
| `control_hz` | `2.0` | 控制频率 |
| `linear_speed` | `0.2` | 默认线速度，单位 m/s |
| `angular_speed` | `0.45` | 默认角速度，单位 rad/s |
| `viewer.enabled` | `false` | 是否打开 MuJoCo 交互窗口 |

## 支持技能

当前 XLeRobot 仿真配置启用 11 个非 VLA skill，和真机配置的默认 Agent 可调用面保持一致：

| Skill | 说明 |
| --- | --- |
| `inspect_scene` | 获取当前场景观察和摘要 |
| `look_around` | 转动/扫描视野并观察 |
| `detect_marker` | 检测可见 marker |
| `move_base` | 底盘前进/后退一小段距离 |
| `turn_base` | 底盘左转/右转 |
| `human_follow` | 基于视觉的人体跟随 |
| `stop_motion` | 停止底盘和机械臂运动 |
| `reset_posture` | 回到安全姿态 |
| `set_arm_pose` | 设置机械臂命名姿态 |
| `move_arm_joints` | 控制机械臂关节 |
| `set_gripper` | 控制夹爪开合 |

`vla_manipulation` 已注册，并且仿真配置可声明 VLA capability service。它默认不在 `skills.enabled` 中，因此 Agent 不能直接调用；需要显式加入后才会进入可调用面。

## 常见问题

### 看不到 LeKiwi 底盘

重新生成模型：

```bash
python scripts/robots/xlerobot/generate_mjcf.py
```

生成后的 MJCF 应包含：

```text
lekiwi_chassis_visual
base_plate_layer1-v5-1_geom
4-Omni-Directional-Wheel_Single_Body-v1_geom
```

### 中间出现黑色实体块

通常是 collision box 被渲染出来。不要手工修改 `xlerobot.xml`，应修改生成器后重新生成。

### 修改 `xlerobot.xml` 后被覆盖

这是预期行为。请修改：

```text
scripts/robots/xlerobot/generate_mjcf.py
```

然后重新运行生成命令。
