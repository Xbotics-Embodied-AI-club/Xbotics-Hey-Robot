# XLeRobot Home SO101 VLA 端到端运行手册

本文对应当前已实现的最小端到端闭环：

```text
数据采集 -> LeRobot policy 训练 -> 通用 policy endpoint 部署 -> Hey Robot RPC 接入 -> 仿真评测
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
