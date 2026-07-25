# Robot Runtime 边界重构

> 状态：2026-07-25 已实施。本文记录本次无旧 API 兼容层的重构结果和后续约束。

## 1. 重构目标

重构前 `src/hey_robot/robot_runtime` 有 71 个 Python 文件、14,440 行，占生产源码
41.27%。其中只有约 15.6% 是运行时编排，大部分是硬件、仿真、Mock、RoboCasa RPC、
媒体和 Bus 适配。

本次不保留旧 import path，也不保留未进入产品 composition root 的 Driver。目标不是把
大文件机械切小，而是让依赖方向成为：

```text
robot_api
   ↑
robot_runtime ← robot_transport
   ↑                 ↑
robot_backends ──────┘
   ↑
robot_hardware

robot_media 为 Robot 数据面共享基础设施
robocasa_backend 为隔离环境服务端
```

## 2. 当前物理边界

| 包 | 所有权 | Python 行数 |
|---|---|---:|
| `hey_robot.robot_api` | Driver、Client、Observation、Embodiment、primitive 契约 | 486 |
| `hey_robot.robot_runtime` | lifecycle、safety、control、admission、observation orchestration | 2,421 |
| `hey_robot.robot_backends` | Mock、XLeRobot native、MuJoCo、RoboCasa remote client | 6,746 |
| `hey_robot.robot_hardware` | Feetech、servo bus、arm、mobile base、camera、battery | 1,637 |
| `hey_robot.robot_transport` | Bus action、status、observation、frame 和 velocity stream | 407 |
| `hey_robot.robot_media` | content-addressed media store、resolver、frame codec | 558 |
| `hey_robot.robocasa_backend` | 环境所有者、trial、EGL、RPC server 和 generated RPC | 1,139 |

`robot_runtime` 已从 14,440 行降为 2,421 行，占当前 33,986 行生产 Python 的 7.12%。
后端代码没有伪装成核心复杂度，也不会因导入 Manager 而全部加载。

## 3. 已删除内容

- 删除 `SO101Driver`、`SO101Client`、`LeKiwiDriver`、`LeKiwiClient`。它们没有
  `RobotManager` 注册或部署配置入口，只有单元测试直接构造，属于不可达产品代码。
- 保留仍被 XLeRobot 使用的 SO101 arm、LeKiwi mobile base 和 VLA codec，并分别迁入
  `robot_hardware` 与 `robot_backends.xlerobot.hardware`。
- 删除引用已经不存在的 `simulation.so101_mobile` 的
  `scripts/record_dock_pick_place.py`。
- 删除 `robot_runtime`、`xlerobot`、`simulation`、`components`、`robocasa_remote` 的
  聚合导出和类型别名兼容层；调用方必须从定义模块导入。
- 删除 Runtime/Mock primitive 列表中没有 Skill 注册入口的旧 `human_follow` action。
  Human Follow 继续由独立服务和 frame/velocity topic 承担。
- 删除 VLA codec 中只被测试调用的 3 个旧转换函数、Runtime 和 Mock 中 3 个零调用
  helper、24 个零引用舵机常量，以及只为测试公开的 `StatefulPacketHandler` 别名。

以上删除均由 Git 管理，但当前源码不提供运行时兼容转发。

## 4. Driver 加载

`RobotManager` 只依赖 Config、`robot_api` 和 embodiment registry。内置 Driver 使用
`module:factory` 描述，匹配配置后才通过 `import_module()` 加载。

当前内置组合为：

| family | environment | driver | factory |
|---|---|---|---|
| 任意 | 任意 | `mock` | `robot_backends.mock:MockRobotDriver` |
| `xlerobot` | `sim` | `mujoco` | `robot_backends.simulation...:XLeRobotSimDriver` |
| `xlerobot` | `real` | `native` | `robot_backends.xlerobot.driver:XLeRobotDriver` |
| `robocasa` | `remote` | `grpc` | `robot_backends.robocasa_remote.driver:create_driver` |

部署可以用 `settings.driver_factory` 指定受信任的扩展 factory。Mock Manager 的边界测试
明确检查不会导入 `grpc` 或 `scservo_sdk`。

## 5. 稳定契约

Skill 和 Runtime 共同依赖 `robot_api`，不再通过 Runtime 实现文件共享类型：

- `RobotClient`、`RobotActionResult`、`RobotClientCapabilities`；
- `RobotDriver`、`RobotDriverContext`、`RobotCapabilities`、`RobotHealth`；
- `DriverObservation`、`ObservationAsset`；
- `EmbodimentProfile`、classic primitive。

`RobotDriverContext` 不再持有 `config.RobotSpec`。Manager 在 composition boundary 将配置
转换为 family、environment、driver kind、settings 和 embodiment，避免后端反向依赖
Config domain model。

持续控制和感知回调使用显式的可选 Protocol：

- `BaseVelocityStreamDriver`；
- `PerceptionRequestDriver`。

Service/Runtime 不再用 `getattr()` 猜测 Driver 能力。

## 6. 观测并发语义

每个 RobotService robot 现在有独立 observation loop。一个机器人采集失败或变慢，不再
阻塞其他机器人的采集周期。

`PerceptionService.refresh()` 使用单一 refresh lock，并合并重叠请求。Bus 发布、Skill
observe 和 action 后观测同时到达时，共享同一个 Driver observation，不会并发读取相机
或推进仿真 frame。

## 7. 平台边界修复

MuJoCo backend 选择 EGL 或 OSMesa 时会同时设置 `MUJOCO_GL` 与
`PYOPENGL_PLATFORM`。这修复了 RoboCasa EGL 探测后再启动本地 MuJoCo 时两个环境变量
不一致、Renderer 静默缺失的问题。

RoboCasa 环境 server、trial owner、EGL 和 generated RPC 已整体移入
`robocasa_backend`；主进程的 remote robot backend 只保留 gRPC client、Driver 和远程
数据契约。

## 8. 强制依赖规则

后续修改必须遵守：

1. `robot_api` 不得导入 Config、Runtime、Backend、Bus 或具体硬件。
2. `robot_runtime` 不得导入具体 Backend、gRPC、MuJoCo、OpenCV 或 servo SDK。
3. `robot_transport` 可以依赖 Runtime，不得包含 Driver 实现。
4. Backend 只能通过 `robot_api` 和明确的 Runtime application boundary 接入。
5. Backend 之间不得导入私有 helper；共享硬件能力必须下沉到 `robot_hardware`。
6. 新 Driver 必须提供 factory、契约测试和导入隔离测试。
7. 不新增旧 import path 的 re-export；迁移必须一次性更新所有调用方和文档。
8. `tests` 必须镜像源码所有权目录；Backend、Hardware、Transport、Media 和 RoboCasa
   测试不得继续堆放在 `tests/robot_runtime`。

## 9. 验证结果

- `uv run ruff check src tests scripts evaluation`：通过；
- `uv run mypy src`：通过（218 个纳入 mypy 的源码文件）；
- `uv run pytest`：715 项全部通过；
- 覆盖率：85.63%，达到并保持 85% 门槛；
- 架构测试确认 Mock Manager 不加载 gRPC/servo SDK，且旧聚合 API、旧顶层
  `hey_robot.media` 和旧 Robot Runtime 子包不会重新出现。
- Robot 子系统的七个源码包均有同名测试根目录，架构测试会阻止测试目录再次退化为
  `tests/robot_runtime` 聚合区。

覆盖率排除项只是把重构前已经排除的硬件、仿真、生成 protobuf 路径一一改为新路径，
没有降低阈值或新增同类排除范围。

## 10. 后续可选优化

当前仍可继续做两项独立优化，但不再是包边界阻塞项：

- 把 `inspect_scene`、`look_around`、`detect_marker` 的复合 Skill 编排从
  `RobotRuntime` 移到 perception Skill application service；Runtime 只保留 observation
  与最终 safety boundary。
- 把 `MockRobotDriver` 内的 world model、failure script 和 rendering 分成 Mock backend
  内部子模块。它已不影响核心 Runtime 体积或导入闭包。
