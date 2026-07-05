# SO101 tabletop 仿真

该环境复现 Vector OS Nano 的 SO101 MuJoCo tabletop manipulation baseline。
它独立于 XLeRobot 双臂/LeKiwi 家庭场景。

## 环境内容

- 单 SO101：5 个 arm joint 和 1 个 jaw visual joint。
- overhead、front、side 三路固定相机。
- banana、mug、bottle、screwdriver、duck、lego。
- MuJoCo ground-truth oracle perception。
- 5cm 条件 weld gripper。
- 位置型 Jacobian DLS IK。
- `pick` 和 `place` Skill。

Oracle 和 weld 只用于验证仿真运动与 Skill 生命周期，不代表真实 RGB 感知或接触抓取。

## 安装

```bash
uv sync --group sim --group dev
```

Linux headless 环境：

```bash
export MUJOCO_GL=egl
```

## 启动

```bash
uv run hey-robot run --config configs/so101.tabletop.sim.yaml
```

需要 viewer 时，将配置中的 `robots.tabletop_arm.settings.viewer.enabled`
改为 `true`，并在有图形显示的终端启动。

## 测试

```bash
MUJOCO_GL=egl uv run --no-sync pytest \
  tests/assets/test_so101_tabletop_scene.py \
  tests/motion/test_so101_tabletop_kinematics.py \
  tests/robot_runtime/test_so101_tabletop_driver.py \
  tests/robot_runtime/test_so101_tabletop_gripper.py \
  tests/robot_runtime/test_so101_tabletop_oracle.py \
  tests/skill_os/test_tabletop_pick_place.py \
  tests/integration/test_so101_tabletop_pick_e2e.py \
  tests/integration/test_so101_tabletop_place_e2e.py -q
```

运行 L5 六对象重复性验收：

```bash
MUJOCO_GL=egl PYTHONPATH=src python \
  scripts/robots/so101_tabletop/acceptance.py --repeats 10
```

脚本要求 banana `10/10`，并要求六对象总成功率不低于 90%。

## 行为边界

- `pick(mode=hold)` 必须激活目标对象 weld，并把目标抬升超过 5cm。
- `place` 必须释放 weld，并验证最终 XY 位置。
- `get_depth_frame` 是全零图，不提供 RGB-D。
- 所有 oracle 数据在 observation 中标记为 `mujoco_ground_truth`。
- 当前机械臂和 jaw collision 均按上游基线关闭。
