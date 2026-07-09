# VLN 模型诊断：InternVLA-N1 System2 左转偏置问题

**日期**：2026-07-09
**状态**：已确认 — 模型在当前家庭仿真场景中存在根本性偏置，无法用于真实导航。

## 概述

InternVLA-N1 System2 模型（基于 Qwen2.5-VL 7B，经 InternNav 微调）**无论看到什么画面、使用什么语言指令、采用什么 prompt 格式，始终输出 `←`（左箭头，action code 2）**。这使得该模型在当前家庭仿真场景（`assets/scenes/home_scene.xml`）中完全不可用。

## 证据

### 推理输出规律

多次独立测试中，每一次推理都只输出左转动作：

```
============ output 1  ←←←←
[VLN-Executor] instruction='厨房' output_action=[2, 2, 2, 2]

============ output 2  ←←←←
[VLN-Executor] instruction='厨房' output_action=[2, 2, 2, 2]

...（多次运行中累计 21+ 次推理，结果完全一致）...
```

### 已排除的假设

| 假设 | 验证方式 | 结论 |
|---|---|---|
| 摄像头帧冻结 | 帧哈希唯一性检测 | 帧正常更新（5秒内10个唯一哈希） |
| 编码管线损坏 | 保存 diag_frames 检查尺寸/数值 | 640×480×3，均值163，正常图像 |
| 中文指令导致模型困惑 | 改用英文 "Go to the kitchen" | 同样输出 ← |
| Prompt 格式不对 | 对比坐标格式 vs 方向格式 prompt | 方向格式下重复次数减少，但方向仍为 ← |
| 转弯角度太小 | max_turn_deg 从 15° 提高到 30° | 转弯加快，但模型仍输出 ← |
| 模型状态异常 | 重启模型服务（重新加载权重） | 同样输出 ← |
| 视角不够多样化 | 机器人累计旋转 630°+ | 模型从未改变输出 |

### 多步导航管线（已验证通过）

在 mock 模式（`mock_mode: true`）下，机器人正确执行：
- 30 步 move_base 前进（每步 15cm，共 4.5m）
- 每步之间穿插 inspect_scene
- 技能结果正确上报

这证明导航循环代码正确，问题纯粹出在模型本身。

## 根因分析

模型对左转动作有强烈的学习偏置（learned bias）。这很可能源于训练数据分布不均 — 室内导航基准数据集（如 Matterport3D、Habitat）中，受典型走廊布局影响，左转动作往往比右转出现更频繁。当模型在不熟悉的环境中无法自信导航时，会退回到训练数据中最常见的动作。

当前家庭仿真场景（`home_scene.xml`）与模型训练数据的视觉分布差异较大，模型无法有效泛化。

## 架构与调用链

```
Agent (DeepSeek) → request_skill("navigate_to", {target: "厨房"})
  → navigate_to 技能 (_VLNNavigationSkill)
    → camera_aware_invoke() 将帧编码为 JPEG q85 base64
      → gRPC → VLN 模型服务 (端口 9091)
        → InternVLAN1System2Executor._internvla_plan()
          → model.s2_step(rgb, depth, pose, instruction, intrinsic)
            → Qwen2.5-VL 7B 推理 (cuda:1)
          ← action=[2] → heading=-90° → "left"
        ← VLNPlannerResult(mode="heading", heading_deg=-90)
      ← navigation_adapter.planner_output_to_primitive()
        → PrimitiveCommand(turn_base, {direction:"left", angle_deg:30})
    → ctx.robot.turn_base(direction="left", angle_deg=30)
      → skill_adapter.on_turn_base() → 正角速度（逆时针旋转）
```

### Hey Robot 可控制的参数（不修改 InternNav 第三方代码）

以下参数均在 `executor.py` 中通过覆写方式控制，不直接修改 InternNav：

- **`vln_prompt_template`**：通过 `_apply_prompt_override()` 覆写 prompt 模板。在配置文件的 `model_services.vln_nav.settings` 中设置。模板必须包含 `<instruction>.` 占位符，`s2_step()` 会在运行时替换。
- **`navigation_adapter.planner_output_to_primitive()`**：控制 VLN 输出（heading / pixel_goal / stop）如何映射为机器人原语（move_base / turn_base）。
- **`max_steps`**（navigation.py）：每次 navigate_to 调用中的最大 VLN 规划步数。
- **`timeout_sec`**（navigation.py）：技能的总体超时时间。

## 可行方案

### 1. 替换 VLN 模型

将 InternVLA-N1 System2 替换为泛化能力更强的模型：
- 其他 VLN 检查点（如 Navid、NoMaD、GNM）
- 基础 VLM（GPT-4V、Gemini）+ prompt 驱动的导航

### 2. 收集场景专用训练数据

- 在家庭场景中录制导航轨迹（摄像头 + 动作序列）
- 用此数据微调模型
- 长期来看是最稳健的方案

### 3. 启发式覆盖层

增加一个启发式检测层，当模型连续 N 次输出相同动作时（表明模型"卡住"了），切换到探索策略（随机方向、沿墙走等）。

### 4. 演示用 Mock 模式

配置文件中设置 `mock_mode: true`。Mock planner 始终返回画面中心的 pixel goal → 机器人直线前进。适用于管线验证，但不适用于真实导航。

## 相关文件

| 文件 | 作用 |
|---|---|
| `src/hey_robot/foundation/backends/vln/internvla_n1_system2/executor.py` | VLN 执行器 + prompt 覆写 |
| `src/hey_robot/skill_os/builtins/navigation.py` | NavigateTo / ApproachObject 技能 |
| `src/hey_robot/skill_os/builtins/navigation_adapter.py` | VLN 输出 → 机器人原语适配 |
| `configs/xlerobot.sim.vla_vln.yaml` | 模型服务 + 技能配置 |
| `third_party/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` | 模型推理（InternNav 第三方包，不直接修改） |
| `scripts/dev/diag_vln_navigation.py` | VLN 诊断工具 |
| `scripts/dev/debug_vln.py` | 多阶段调试工具 |
