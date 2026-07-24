# XLeRobot Home SO101 VLA 端到端运行手册

本文对应当前已实现的最小端到端闭环：

```text
数据采集 -> LeRobot policy 训练 -> RobotPolicyService 部署 -> Hey Robot 完整链路评测
```

这份文档的定位是工程运行手册，不是论文式方案。读完以后应该能判断：

```text
1. 当前代码能跑通哪条链路。
2. SmolVLA、ACT、pi0 或自定义 LeRobot policy 应该接在什么位置。
3. home 场景数据采集、训练、部署、评测各自的输入输出是什么。
4. 哪些能力已经实现，哪些仍然只是下一阶段扩展点。
```

## 0. 环境与目录约定

建议把 Hey Robot 运行环境和 LeRobot 训练环境分开理解：

```text
D:/agent_robot/Xbotics-Hey-Robot
  Hey Robot 主系统
  负责 XLeRobot sim/home scene、SkillRunner、ModelService 和 Robot Runtime

D:/agent_robot/lerobot-mujoco-tutorial
  LeRobot 训练侧项目
  负责具体 policy 训练脚本和 LeRobot 依赖
```

第一阶段推荐使用以下目录：

```text
data/lerobot/xlerobot_home_so101_single_arm
  采集得到的 LeRobotDataset

models/xlerobot-home-so101-smolvla
  SmolVLA 示例 checkpoint

runtime/eval/home_so101_policy
  自动评测输出
```

所有 Hey Robot 侧命令都在：

```powershell
cd D:\agent_robot\Xbotics-Hey-Robot
$env:PYTHONPATH="src"
```

## 1. 数据采集

在带 LeRobot 的训练环境中运行：

```powershell
$env:PYTHONPATH="src"
python scripts\lerobot\record_home_so101_lerobot.py `
  --config configs\xlerobot.sim.vla_vln.yaml `
  --repo-id xlerobot_home_so101_single_arm `
  --root data\lerobot\xlerobot_home_so101_single_arm `
  --task "pick up the object" `
  --arm right `
  --episodes 100 `
  --fps 20 `
  --overwrite
```

输出是 LeRobotDataset：

```text
data/lerobot/xlerobot_home_so101_single_arm
```

当前 schema：

```text
observation.images.front
observation.images.handeye
observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
action:            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
```

这一步运行的是完整 XLeRobot home sim，但数据集只抽取单臂 SO101 的 VLA 子空间：

```text
机器人上下文:
- home 场景
- XLeRobot sim driver
- front camera
- right_wrist/handeye camera
- 机器人 reset 和仿真 step

VLA 学习对象:
- 一个 arm
- 5 个关节角
- 1 个 gripper 标量
- 当前图像和当前 state 到下一步 action
```

### 1.1 当前采集方式

当前脚本使用 POC 级 waypoint expert：

```text
scripts/lerobot/record_home_so101_lerobot.py
```

它适合验证链路，不适合直接产出高质量真实训练集。原因是：

```text
1. waypoint expert 轨迹单一，覆盖的状态分布窄。
2. 任务成功条件没有和具体物体 pose 严格绑定。
3. 失败样本、恢复样本、多样化扰动还没有系统采集。
```

### 1.2 接入 teleop 或真实采集时的接口边界

后续接入 teleop/真实机器人时，不建议改 dataset schema。应该替换 action 来源：

```text
当前:
state_from_sim_driver(driver) -> waypoint_expert.act(state) -> dataset.add_frame(...)

teleop:
state_from_robot_status(...) -> teleop_action -> dataset.add_frame(...)

真实机器人:
state_from_arm_status(...) -> operator/action_server action -> dataset.add_frame(...)
```

必须保持不变的是：

```text
observation.images.front
observation.images.handeye
observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
action:            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
state_schema:      so101_single_arm_rad_gripper01
```

这样训练、部署、评测都不用跟着重写。

## 2. 训练 LeRobot Policy

训练脚本不绑定具体 policy。选择 ACT、SmolVLA、pi0 或其他 LeRobot policy 时，核心要求是训练配置产出的 dataset feature schema 与第 1 节一致。

SmolVLA 示例配置：

```text
configs/examples/smolvla_home_so101.yaml
```

启动训练：

```powershell
$env:PYTHONPATH="src"
python scripts\lerobot\train_home_so101_policy.py `
  --config-path configs\examples\smolvla_home_so101.yaml `
  --python python
```

默认会优先调用：

```text
D:/agent_robot/lerobot-mujoco-tutorial/train_model.py
```

训练产物默认写到：

```text
models/xlerobot-home-so101-smolvla
```

如果训练 ACT 或其他 policy，应新建对应配置文件，并保持：

```text
dataset.repo_id: xlerobot_home_so101_single_arm
observation.images.front
observation.images.handeye
observation.state
action
```

### 2.1 多 policy 的接入原则

当前 pipeline 不把 SmolVLA 写死在 Hey Robot 系统里。训练侧只要能产出 LeRobot 标准 checkpoint，部署侧就通过：

```text
--policy-type <LeRobot factory registered name>
--checkpoint <checkpoint path>
```

加载。

推荐把不同 policy 的配置拆成不同 YAML，而不是在代码里做条件分支：

```text
configs/examples/smolvla_home_so101.yaml
configs/examples/act_home_so101.yaml
configs/examples/pi0_home_so101.yaml
```

ACT 配置应重点确认：

```text
policy.type: act
dataset.repo_id: xlerobot_home_so101_single_arm
input image keys: observation.images.front, observation.images.handeye
input state key: observation.state
output action key: action
action dimension: 6
```

pi0/其他 VLA policy 也遵循同样边界。如果某个 policy 需要语言字段、额外相机、不同 action horizon
或不同归一化统计，应该扩展 checkpoint processor 和部署配置，而不是在 Hey Robot 主系统里写
policy 专用分支。

训练与部署环境必须使用同一个 LeRobot 版本和 checkpoint 规范。Hey Robot 不维护新旧 LeRobot API
兼容表，也不要求配置重复声明 policy type；部署时由 checkpoint 的 `config.json.type` 和 LeRobot
factory 共同决定具体 policy class。

### 2.2 训练产物检查

训练完成后至少检查：

```text
checkpoint 目录存在
dataset stats 可读取
policy 能 from_pretrained(checkpoint)
policy.select_action(batch) 或 policy.predict_action_chunk(batch) 可运行
输出 action 最后一维能解释为 gripper01
```

最小 smoke test 是先检查 checkpoint 是否能被当前 LeRobot factory 识别：

```powershell
python -m hey_robot.foundation.backends.lerobot.checkpoint `
  --policy-path models\xlerobot-home-so101-smolvla\checkpoints\last\pretrained_model
```

## 3. 启动 Robot Policy ModelService

把下面配置片段合并到 `configs/xlerobot.sim.vla_vln.yaml` 的：

```text
model_services.manipulate.settings
```

片段位置：

```text
configs/examples/xlerobot.lerobot_manipulate_service.yaml
```

关键配置：

```yaml
runtime: lerobot
policy_path: models/xlerobot-home-so101-smolvla/checkpoints/last/pretrained_model
policy_device: cuda
action_space: xlerobot_single_arm_joint
action_dimensions: 6
state_dimensions: 6
camera_names: [front, right_wrist]
```

启动 Hey Robot gRPC VLA service：

```powershell
$env:PYTHONPATH="src"
python -m hey_robot.cli.model_service `
  --config configs\xlerobot.sim.vla_vln.yaml `
  --service-id manipulate
```

此时唯一生产链路是：

```text
SkillRunner manipulate
  -> gRPC ExecuteSkill
  -> RobotPolicyService
  -> LeRobotPolicyExecutor
  -> LeRobot factory + checkpoint processors
  -> embodiment_native_action
  -> Robot Runtime safety/action gate
```

LeRobot 仍运行在独立 ModelService 进程中，因此 CUDA 和 checkpoint 依赖与 Core Harness 隔离；不再
增加一层 HTTP policy server。VLA、WAM、ACT、Diffusion 等 LeRobot policy 使用同一服务和 executor。

## 4. 自动评测

不再维护直接请求 policy runtime、再直接写仿真 driver 的评测脚本。XLeRobot 评测必须经过与生产
相同的完整链路：

```text
Agent -> SkillRunner -> gRPC RobotPolicyService -> Robot Runtime -> simulator
```

评测器只负责初始化场景、提交任务、读取独立 success predicate 和汇总 artifact，不得成为第二个
动作所有者。完整 XLeRobot simulation benchmark 仍是重构计划中的 P3 工作。

### 4.1 推荐的评测分层

建议按四层推进：

```text
L0 ModelService smoke:
  gRPC health loaded=true
  ExecuteSkill 返回合法 embodiment_native_action

L1 Robot Runtime smoke:
  1-3 个 episode 能完成通过 safety gate 的 action rollout
  action 数值不爆炸，gripper 范围正确

L2 sim task eval:
  50+ episodes
  物体 pose 随机化
  success predicate 绑定物体状态，例如 object lifted / object in target zone

L3 real/home constrained eval:
  限制工作空间
  低速执行
  人工 emergency stop
  记录成功、失败、人工接管原因
```

RoboCasa365 已覆盖该完整链路；XLeRobot L0-L3 仍需按相同边界实现和验证。

## 5. 当前边界

当前已经具备闭环验证能力，但任务专家和 success predicate 还是 POC 级别：

```text
已完成:
- LeRobotDataset 采集脚本
- LeRobot policy 训练启动脚本
- 通用 LeRobot gRPC RobotPolicyService
- 单臂 SO101 state/action schema

还需要针对具体 home task 强化:
- 更真实的 scripted expert 或 teleop 采集
- 按任务定义 success predicate
- 物体随机化和失败标注
- UI 回放和 episode store 对齐
```

## 6. 常见问题与排障

### 6.1 `ModuleNotFoundError: No module named 'hey_robot'`

在 Hey Robot 仓库根目录设置：

```powershell
$env:PYTHONPATH="src"
```

或者通过项目的 `uv run` 入口运行。

### 6.2 checkpoint policy type 加载失败

通用 executor 从 checkpoint 读取 `config.json.type`，再调用：

```python
lerobot.policies.factory.get_policy_class(policy_type)
```

因此需要确认：

```text
1. 当前 Python 环境安装了 LeRobot。
2. 该 policy 已经在 LeRobot factory 注册。
3. checkpoint 的 `config.json.type` 与训练产物一致。
```

Hey Robot 不会通过部署配置覆盖 checkpoint 的 policy type。

### 6.3 image key 不匹配

现象通常是模型报缺少 image feature，或 batch key 不存在。

检查训练 dataset feature：

```text
observation.images.front
observation.images.handeye
```

如果训练时用了其他 key，通过 ModelService 配置：

```yaml
observation_features:
  observation.images.training_front: observation.images.front
```

显式映射。

### 6.4 state/action 维度不匹配

当前 SO101 单臂约定是 6 维：

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
```

如果 policy 输出不是 6 维，需要先统一训练配置和 action schema。不要在 executor 里靠截断或填充修补。

### 6.5 角度单位错误

`xlerobot_single_arm_joint` action space 使用 rad。训练数据、checkpoint 输出、ModelService
`action_space` 和 Robot Runtime mapping 必须一致，不在 executor 内做隐式 degree/rad 转换。

### 6.6 policy 能启动但动作很差

优先排查：

```text
1. 采集数据是否覆盖目标任务。
2. 图像 camera 是否与训练一致。
3. state 关节顺序是否一致。
4. gripper 开合方向是否一致。
5. 训练 normalization stats 是否来自同一个 dataset。
6. 评测场景物体 pose 是否超出数据分布。
```

## 7. 最小验收清单

代码级验收：

```powershell
uv run --no-sync poe style
uv run --no-sync poe lint
uv run --no-sync poe test
```

链路级验收：

```text
1. record 脚本能生成 LeRobotDataset。
2. train 脚本能启动指定 policy 训练。
3. checkpoint inspector 确认 policy type 已由 LeRobot factory 注册。
4. RobotPolicyService health 返回 loaded=true。
5. gRPC ModelService 能为 manipulate 返回合法 `embodiment_native_action`。
6. Robot Runtime 能校验并执行该 action。
7. full-system benchmark 能输出 summary.json、episodes.jsonl、trace.jsonl。
```

工程交付级验收：

```text
1. 至少一个 policy checkpoint 完成 50 episode sim eval。
2. failure_modes 有明确分类，不只是 timeout。
3. 成功标准绑定任务状态，而不是只看 gripper 是否闭合。
4. 数据集、checkpoint、eval 输出都能追溯到同一个 run id。
5. 真实机器人 dry-run 有低速限制和 emergency stop。
```
