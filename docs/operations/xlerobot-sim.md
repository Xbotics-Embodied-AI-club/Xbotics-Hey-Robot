# XLeRobot 仿真

XLeRobot 仿真使用统一的 Agent、Skill 和 Robot Runtime 启动路径。模型后端属于
Embodied Agent 进程的一部分；本仓库不提供单独的 model-service 容器或 CLI。

## 配置

| 平台 | 配置 |
|---|---|
| Ubuntu | `configs/xlerobot.sim.ubuntu.yaml` |
| Windows | `configs/xlerobot.sim.windows.yaml` |

## 启动

```bash
uv sync --extra gateway --extra agent --extra robot --group sim --group dev
export MUJOCO_GL=egl
uv run hey-robot inspect --config configs/xlerobot.sim.ubuntu.yaml
uv run hey-robot run --config configs/xlerobot.sim.ubuntu.yaml
```

`inspect` 用于确认配置、机器人能力和依赖状态；`run` 启动完整系统。Windows 请改用
对应的配置文件。

## Docker

基础 Compose 拓扑只包含 NATS、Gateway、Agent、Robot 和前端：

```bash
docker-compose up -d nats robot agent gateway frontend
docker-compose ps
```

RoboCasa365 评测使用独立的 [评测说明](../../evaluation/robocasa365/README.md)，不通过
XLeRobot 仿真 Compose 启动。

## 常见问题

- headless Linux 没有 X11 时，设置 `MUJOCO_GL=egl`。
- 修改机器人结构请编辑 `scripts/robots/xlerobot/generate_mjcf.py` 后重新生成 MJCF，
  不要手改生成的 XML。
- Ubuntu 使用默认音频设备；Windows 可用 `scripts/audio/list_devices.py` 查设备索引。
