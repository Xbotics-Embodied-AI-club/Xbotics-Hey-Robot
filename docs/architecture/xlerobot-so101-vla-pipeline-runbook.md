# XLeRobot Home SO101 VLA 端到端运行手册

本文对应当前已实现的最小端到端闭环：

```text
数据采集 -> LeRobot policy 训练 -> 通用 policy endpoint 部署 -> Hey Robot RPC 接入 -> 仿真评测
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
  负责 XLeRobot sim/home scene、Skill OS、ModelService、VLA endpoint glue code

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
python scripts\vla\record_home_so101_lerobot.py `
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
scripts/vla/record_home_so101_lerobot.py
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
python scripts\vla\train_home_so101_policy.py `
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

pi0/其他 VLA policy 也遵循同样边界。如果某个 policy 需要语言字段、额外相机、不同 action horizon 或不同归一化统计，应该扩展训练配置和 endpoint 参数，而不是在 Hey Robot 主系统里写 policy 专用分支。

注意：当前 `D:/agent_robot/lerobot-mujoco-tutorial/train_model.py` 使用的是它自身 LeRobot 环境里的训练 API。部署侧 `serve_lerobot_policy.py` 使用当前运行环境里的 LeRobot factory。为了保持系统简单，不在 Hey Robot 里做多版本 import 兼容；训练环境和部署环境必须选定同一个 LeRobot 版本/导入规范。

如果你的 LeRobot fork 仍然只暴露 `lerobot.common.*` 路径，应优先升级或统一部署环境，而不是在 server 里同时维护新旧两套路由。

### 2.2 训练产物检查

训练完成后至少检查：

```text
checkpoint 目录存在
dataset stats 可读取
policy 能 from_pretrained(checkpoint)
policy.select_action(batch) 或 policy.predict_action_chunk(batch) 可运行
输出 action 最后一维能解释为 gripper01
```

最小 smoke test 是启动 server 后访问：

```powershell
curl http://127.0.0.1:18080/health
```

也可以直接跑 endpoint smoke CLI。它会检查 `/health`、`/predict`、`action_chunk` schema、关节字段和 gripper 范围：

```powershell
$env:PYTHONPATH="src"
python scripts\vla\smoke_lerobot_policy_endpoint.py `
  --endpoint http://127.0.0.1:18080/predict `
  --task "pick up the object"
```

## 3. 启动 LeRobot Policy Endpoint

```powershell
$env:PYTHONPATH="src"
python scripts\vla\serve_lerobot_policy.py `
  --policy-type smolvla `
  --checkpoint models\xlerobot-home-so101-smolvla\checkpoints\last\pretrained_model `
  --dataset-repo-id xlerobot_home_so101_single_arm `
  --dataset-root data\lerobot `
  --host 127.0.0.1 `
  --port 18080
```

`--policy-type` 可以换成当前 LeRobot 环境中已经通过 factory 注册的 policy，例如 `act`、`smolvla`、`pi0` 或自定义 policy。Server 不猜测类型，也不维护 policy 专用 import 兼容表。

健康检查：

```powershell
curl http://127.0.0.1:18080/health
```

### 3.1 Endpoint 输入契约

Hey Robot 发送给 policy endpoint 的核心 payload 是：

```json
{
  "task": "pick up the object",
  "observation": {
    "images": [
      {"camera": "front", "format": "jpeg", "data": "..."},
      {"camera": "right_wrist", "format": "jpeg", "data": "..."}
    ],
    "state": [0.0, 0.8, 0.7, -0.6, 0.0, 1.0],
    "state_schema": "so101_single_arm_rad_gripper01",
    "active_arm": "right"
  }
}
```

Server 默认映射：

```text
front       -> observation.images.front
right_wrist -> observation.images.handeye
state       -> observation.state
task        -> task: list[str]
```

如果训练集用了不同 camera key，用 `--camera-features` 显式指定：

```powershell
python scripts\vla\serve_lerobot_policy.py `
  --policy-type act `
  --checkpoint models\xlerobot-home-so101-act\checkpoints\last\pretrained_model `
  --camera-features '{"front":"observation.images.front","right_wrist":"observation.images.handeye"}'
```

### 3.2 Endpoint 输出契约

Server 输出统一 Hey Robot `action_chunk`：

```text
kind: action_chunk
action_space: xlerobot_single_arm_joint
actions[0].joints:
  shoulder_pan
  shoulder_lift
  elbow_flex
  wrist_flex
  wrist_roll
actions[0].gripper: 0.0 - 1.0
```

这也是 Skill OS manipulate 最终能执行的格式。

## 4. 接入 Hey Robot RPC ModelService

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
backend: action_chunk_policy
backend_mode: action_chunk_policy
action_chunk_endpoint: http://127.0.0.1:18080/predict
model_path: ""
```

启动 Hey Robot gRPC VLA service：

```powershell
$env:PYTHONPATH="src"
python -m hey_robot.cli.model_service `
  --config configs\xlerobot.sim.vla_vln.yaml `
  --service-id manipulate
```

此时系统链路是：

```text
Skill OS manipulate
  -> gRPC ExecuteSkill
  -> LeRobotVLAPolicyExecutor
  -> HTTP /predict
  -> LeRobot policy endpoint
  -> action_chunk
  -> move_arm_joints / set_gripper
```

### 4.1 为什么用 HTTP policy endpoint，而不是把 LeRobot 直接塞进 Hey Robot

第一阶段推荐外部 endpoint：

```text
Hey Robot 主进程:
- 保持轻量
- 不强绑定 CUDA/LeRobot 大模型依赖
- 只负责 RPC、Skill OS、机器人运行时和动作执行

LeRobot policy endpoint:
- 单独选择 Python/CUDA/LeRobot 环境
- 单独加载大 checkpoint
- 单独替换 ACT/SmolVLA/pi0
```

这样部署更清晰，排障也更直接。

## 5. 自动评测

在 LeRobot policy endpoint 启动后，运行：

```powershell
$env:PYTHONPATH="src"
python scripts\vla\evaluate_home_so101_policy.py `
  --config configs\xlerobot.sim.vla_vln.yaml `
  --policy-endpoint http://127.0.0.1:18080/predict `
  --task "pick up the object" `
  --episodes 50 `
  --out runtime\eval\home_so101_smolvla
```

输出：

```text
runtime/eval/home_so101_smolvla/summary.json
runtime/eval/home_so101_smolvla/episodes.jsonl
runtime/eval/home_so101_smolvla/trace.jsonl
```

`summary.json` 包含：

```text
episodes
success_count
success_rate
mean_steps
failure_modes
```

### 5.1 当前评测能力

当前评测脚本做的是 endpoint rollout：

```text
1. reset XLeRobot sim
2. 渲染 front/right_wrist 图像
3. 读取 SO101 单臂 state
4. POST /predict
5. 把 action_chunk 写回 sim driver
6. 统计 episode 结果和 trace
```

当前 success predicate 支持：

```text
--success-mode gripper_closed
--success-mode object_lifted --object-body <body_name> --min-lift-m 0.03
--success-mode object_near_target --object-body <body_name> --target-body <body_name> --max-distance-m 0.08
--success-mode none
```

`gripper_closed` 只能证明 policy 调用了夹爪闭合动作，不能证明真实 pick 成功。因此它适合 smoke test，不适合最终论文或产品级评测。

`object_lifted` 和 `object_near_target` 会显式读取 MuJoCo body 在机器人 base frame 下的位置。它们比 `gripper_closed` 更接近任务指标，但要求场景里有稳定的 body name。不要让脚本自动猜对象；在命令中显式传：

```powershell
python scripts\vla\evaluate_home_so101_policy.py `
  --config configs\xlerobot.sim.vla_vln.yaml `
  --policy-endpoint http://127.0.0.1:18080/predict `
  --success-mode object_lifted `
  --object-body cube `
  --min-lift-m 0.03
```

### 5.2 推荐的评测分层

建议按四层推进：

```text
L0 endpoint smoke:
  /health 正常
  /predict 返回合法 action_chunk

L1 sim control smoke:
  1-3 个 episode 能完成非空 action rollout
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

当前代码覆盖 L0-L1，并提供 L2 的文件输出骨架。

## 6. 当前边界

当前已经具备闭环验证能力，但任务专家和 success predicate 还是 POC 级别：

```text
已完成:
- LeRobotDataset 采集脚本
- LeRobot policy 训练启动脚本
- 通用 LeRobot HTTP policy endpoint
- Hey Robot gRPC ModelService endpoint 接入
- 自动 rollout 评测脚本
- 单臂 SO101 state/action schema

还需要针对具体 home task 强化:
- 更真实的 scripted expert 或 teleop 采集
- 按任务定义 success predicate
- 物体随机化和失败标注
- UI 回放和 episode store 对齐
```

## 7. 常见问题与排障

### 7.1 `ModuleNotFoundError: No module named 'hey_robot'`

在 Hey Robot 仓库根目录设置：

```powershell
$env:PYTHONPATH="src"
```

或者通过项目的 `uv run` 入口运行。

### 7.2 `policy_type` 加载失败

通用 server 只调用：

```python
lerobot.policies.factory.get_policy_class(policy_type)
```

因此需要确认：

```text
1. 当前 Python 环境安装了 LeRobot。
2. 该 policy 已经在 LeRobot factory 注册。
3. `--policy-type` 与训练配置里的 policy type 一致。
```

Server 不会帮你猜 `act`、`smolvla`、`pi0` 的模块路径。

### 7.3 image key 不匹配

现象通常是模型报缺少 image feature，或 batch key 不存在。

检查训练 dataset feature：

```text
observation.images.front
observation.images.handeye
```

如果训练时用了其他 key，通过：

```text
--camera-features
```

显式映射。

### 7.4 state/action 维度不匹配

当前 SO101 单臂约定是 6 维：

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper01]
```

如果 policy 输出不是 6 维，需要先统一训练配置和 action schema。不要在 server 里靠截断或填充修补。

### 7.5 角度单位错误

默认 action 单位是 rad：

```text
--action-units rad
```

如果训练出的 action 是 degree，可显式使用：

```text
--action-units deg
```

但推荐从数据集开始就统一 rad，减少部署时的隐式转换。

### 7.6 policy 能启动但动作很差

优先排查：

```text
1. 采集数据是否覆盖目标任务。
2. 图像 camera 是否与训练一致。
3. state 关节顺序是否一致。
4. gripper 开合方向是否一致。
5. 训练 normalization stats 是否来自同一个 dataset。
6. 评测场景物体 pose 是否超出数据分布。
```

## 8. 最小验收清单

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
3. serve_lerobot_policy.py --help 正常。
4. /health 返回 loaded=true。
5. smoke_lerobot_policy_endpoint.py 能验证 /predict 返回合法 action_chunk。
6. model_service 能把 manipulate 请求转发到 endpoint。
7. evaluate 脚本能输出 summary.json、episodes.jsonl、trace.jsonl。
```

工程交付级验收：

```text
1. 至少一个 policy checkpoint 完成 50 episode sim eval。
2. failure_modes 有明确分类，不只是 timeout。
3. 成功标准绑定任务状态，而不是只看 gripper 是否闭合。
4. 数据集、checkpoint、eval 输出都能追溯到同一个 run id。
5. 真实机器人 dry-run 有低速限制和 emergency stop。
```
