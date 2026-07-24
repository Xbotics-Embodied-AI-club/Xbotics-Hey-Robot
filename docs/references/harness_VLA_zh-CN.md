# Harness VLA：通过记忆引导的智能体将冻结的VLA转化为可靠的操控原语

**作者：** 张义贤<sup>1,\*</sup>, 张焕明<sup>1,\*</sup>, 高峰<sup>2</sup>, 李潇<sup>3</sup>, 刘志浩<sup>4</sup>, 朱春阳<sup>5</sup>, 邱嘉兴<sup>5</sup>, 闫宇辰<sup>5</sup>, 刘纪远<sup>7</sup>, 唐文浩<sup>1</sup>, 方正儒<sup>6</sup>, 聂毅<sup>1,2</sup>, 魏昌旭<sup>1</sup>, 王宇<sup>1</sup>, 丁文博<sup>1,†</sup>, 余超<sup>1,†</sup>

**所属机构：** <sup>1</sup>清华大学, <sup>2</sup>星步科技(Striding AI), <sup>3</sup>普渡大学, <sup>4</sup>中国科学院自动化研究所, <sup>5</sup>无界AI(Infinigence AI), <sup>6</sup>香港科技大学, <sup>7</sup>中关村学院

<sup>\*</sup>同等贡献。 <sup>†</sup>通讯作者。 网站: [harnessvla.github.io](https://harnessvla.github.io/)

> arXiv:2607.08448v3 [cs.RO] 2026年7月15日

## 摘要

语言条件操控既需要精确的接触丰富控制，也需要对语言、场景和长时序任务的鲁棒推理。端到端的视觉-语言-动作（VLA）模型提供了强大的局部视觉运动技能，但它们是在分布内的任务轨迹上训练的，在部署扰动下往往会退化，例如语义重定向、目标重绑定、空间布局变化以及不稳定的局部接触。LLM编码智能体提供了互补的语义和组合推理能力，但纯解析原语难以处理不规则抓取、约束放置和关节物体交互。我们提出了Harness VLA，一个记忆增强的智能体框架，将冻结的VLA暴露为可重试的接触丰富原语，并将其与一组小型固定的解析原语库组合，用于接地、阶段化、传输、导航和释放。该方法不是扩展技能库，而是从任务特定的执行轨迹、全局成功规则和失败模型中学习这些固定原语的操作范围。通过将语义重接地、非接触执行和VLA重阶段化提升到规划器层面，同时将冻结的VLA保留用于局部接触丰富阶段，Harness VLA无需微调即可将预训练的VLA扩展到其原始轨迹分布之外。在受扰动的桌面操控、家庭厨房操控以及清洁到随机的双臂操控任务中，Harness VLA在LIBERO-Pro和RoboCasa365上分别比最强的相关基线高出38.6和25.4个百分点，并在RoboTwin C2R上达到58.4%的成功率。

> **图1：** Harness VLA系统概览。给定任务描述、RGB-D观测和机器人状态，智能体规划器从固定的原语库中选择结构化调用，而不是直接发出低级动作。该原语库将冻结的VLA暴露为vla_act用于接触丰富行为，并使用解析原语（如move_to、rotate和set_gripper）进行感知条件的阶段化、传输、姿态调整和释放。任务特定记忆存储来自参考种子探索的成功命令轨迹，用于少样本重接地；而全局记忆存储可复用的成功规则和失败模型。右侧面板总结了相对于最强基线的提升，底部条展示了交替稀疏VLA调用与解析控制的展开过程。

## 1. 引言

机器人操控的一个长期目标是构建一个系统，能够在变化的物体、布局和形态下可靠地执行自由形式的自然语言指令。两种主导范式从相反的方向接近这一目标。端到端的视觉-语言-动作（VLA）模型直接从机器人轨迹中学习接触丰富的视觉运动控制，而LLM编码智能体则使用语言模型推理来组合显式的感知与控制API。每种范式都很强大，但每种都将过多的责任分配给错误的组件：单体VLA必须在单一策略中吸收语言接地、长时序组合和低级控制，而编码智能体必须通过手工设计或智能体生成的API来实现物理上精妙的交互。图2展示了我们的应对方案：使用解析原语来穿越部署扰动，并仅在局部接触丰富区域内调用VLA，在这些区域内其训练分布是有信息量的。

端到端VLA模型已经快速发展，从通用机器人策略[6, 31]到流匹配和动作推理架构[5, 53, 27, 32]。它们的优势在于局部的、基于图像的接触：抓取不规则物体、以紧公差放置、或驱动对解析控制器来说脆弱的夹具。它们的弱点是部署在其训练的轨迹分布之外。在分布内任务轨迹上训练的模型可能知道如何抓取牛奶盒或转动水龙头，但当语义目标被重定向、目标谓词被重绑定、物体布局发生变化或短技能必须组合成更长的例程时就会失败。在这种部署扰动下，策略可能重复熟悉的训练时行为，即使指令或场景绑定已发生变化[31, 53, 65]；单次不稳定的接触失败也可能导致整个单体展开过程脱轨。

LLM编码智能体和外挂框架提供了互补的语义和组合推理能力。诸如Code as Policies和ProgPrompt等系统在精心策划的感知和控制API上合成可执行程序[36, 59]，最近的视觉多模态或智能体变体通过更丰富的感知、工具使用、反馈和持久执行状态来扩展这一思想[21, 56, 15, 75]。更广泛地说，编码智能体外挂框架将模型输出包装在带有工具接口、记忆、验证器、执行循环和反馈通道的结构化运行时中，使智能体能够修正决策、将成功轨迹或失败诊断写回记忆，并在共同的控制表面下编排异构工具[67, 71, 68, 47]。然而在机器人操控中，扩展此类系统通常仍意味着扩展原语或技能库，而纯解析原语——确定性运动学或基于模型的控制（如IK传输、手腕旋转、基座运动、夹爪开合和释放）——仍然不适合处理不规则抓取、约束放置和关节物体操控。

Harness VLA为机器人操控实例化了这种编码智能体外挂框架视图：保持原语库固定且小型，让智能体学习如何编排它。规划器组合解析原语用于非接触结构，如目标接地、自由空间传输、姿态调整、移动阶段化、失败后的重阶段化以及释放。对于接触丰富阶段，它通过一个单一的学习原语vla_act调用冻结的VLA。这将VLA从单体轨迹策略转变为可复用的接触专家，将其扩展到其原始轨迹分布之外的任务，而无需微调或部署时扩展原语。

> **图2：** 原语组合将冻结的VLA扩展到其轨迹分布之外。部署扰动将可能的任务配置扩展到冻结VLA所覆盖的分布内轨迹之外。直接的VLA展开可能试图跨越扰动空间并在到达目标之前失败。Harness VLA则将任务分解为局部接触丰富的VLA调用和解析原语控制：解析原语感知当前目标、重接地任务绑定，并将机器人在VLA兼容的局部区域之间移动，而vla_act仅在这些区域内的接触丰富阶段被调用。

关键在于不仅仅是暴露vla_act，还要学习何时以及如何使用它。Harness VLA将VLA执行视为一个可重试的局部尝试：规划器可以将机器人阶段化到有利的局部观测中，调用VLA，检查接触结果，并在需要时重新阶段化。两个记忆模块在智能体外挂框架内支持这一过程[67, 71, 64]：任务特定轨迹存储成功的原语组合，用于少样本重接地，而全局记忆存储可复用的成功规则和失败模型。Harness VLA不是添加更多技能，而是教导规划器每种固定原语的操作范围：哪些子问题应该由解析方式处理，何时适合使用vla_act，以及失败的接触尝试应如何重新阶段化。我们的核心贡献如下：

- **一个记忆增强的智能体框架**，将冻结的VLA作为原语使用。Harness VLA将vla_act与固定的解析原语组合，将预训练的VLA从局部接触丰富控制扩展到长时序、受扰动的操控，而无需微调VLA或在部署时扩展原语词汇表。
- **一项实证分析**，展示了为什么当规划器学会如何使用它时，一个小型的固定原语库就足够。重复的规划器阶段化调用可以重构脆弱的VLA尝试，而解析原语解决了每个接触丰富阶段周围的大部分非接触结构。
- **在标准和受扰动的桌面操控、家庭厨房操控以及清洁到随机迁移中的强大基准测试结果**。Harness VLA保持了标准LIBERO的竞争性能，在LIBERO-Pro和RoboCasa365上分别比最强的相关基线高出38.6和25.4个百分点，并在RoboTwin C2R上达到58.4%。

## 2. Harness VLA框架

我们的语言条件操控智能体框架遵循图1中的系统视图。任务描述、RGB-D观测和机器人状态传递给智能体规划器，规划器在固定原语库上推理，并从任务特定记忆和全局记忆中检索上下文。智能体外挂框架（第2.2节）将此规划器通过JSON序列化的原语接口与模拟器耦合，驱动基于回合的执行循环，并将成功的探索轨迹写入任务特定记忆，同时将通用的启发式规则提交到全局记忆。原语库（第2.3节）定义了规划器被允许调用的唯一操作：一小组解析原语以及一个结构上特殊的VLA原语，该原语封装了用于接触丰富交互的预训练视觉运动策略。第2.1节首先形式化了任务和这些组件所构建的迭代执行循环。

### 2.1 问题形式化与智能体执行循环

**任务设置。** 我们考虑在由刚体物理引擎（例如通过Robosuite的MuJoCo）驱动的环境ℰ中的语言条件机器人操控。在每个时间步t，环境暴露一个多模态观测元组o_t = (I_t^rgb, I_t^d, q_t)，包括智能体视角的RGB图像I_t^rgb、对齐的度量深度图I_t^d以及机器人本体感受状态q_t（连接末端执行器位姿和夹爪状态）。任务由自然语言描述ℓ以及二元完成谓词𝒢定义，仅作为回合终止时的稀疏成功信号暴露。

**智能体执行循环。** 如图1中的展开条所示，任务展开被形式化为高级智能体规划器Π与底层物理引擎之间的自回归、基于回合的交互。我们不将视觉运动策略视为单独的分层层级，而是将所有低级控制机制——包括冻结的预训练VLA f_θ和所有确定性操作空间控制器——统一到一个单一的预定义原语库𝒫中。

在每个执行回合t，规划器Π处理当前的多模态观测o_t、任务描述ℓ以及从任务特定记忆和全局记忆中检索的上下文。作为唯一的认知编排者，Π发出一个选定原语c_t ∈ 𝒫的结构化JSON调用。物理引擎直接接收此调用，并在模拟器中执行相应的物理运动，直到原语的内部后置条件满足。原语终止后，引擎产生后续观测o_{t+1}和更新后的机器人状态q_{t+1}。此环境-规划器循环持续迭代，直到目标谓词𝒢满足或最大步数预算耗尽。

### 2.2 Harness VLA架构

受最近的编码智能体（通过外挂执行-反馈循环使模型决策可执行[67, 71, 68]）的启发，Harness VLA以相同的REPL风格形式封装机器人操控。外挂框架是规划器与环境之间的运行时契约：它暴露原语模式，将决策序列化为JSON命令，执行原语，刷新RGB-D和本体感受观测，记录轨迹，检索任务特定记忆和全局记忆，执行重置和预算策略，并通过基准谓词检查进度[47]。

由于此外挂框架将所有细粒度执行委托给原语库，智能体规划器Π可以完全专注于组合推理。为此，它在很大程度上依赖多模态观测通道：RGB图像支持定性场景推理（例如，杂乱程度、语义身份），而对齐的深度图和本体感受提供用于精确定位的度量空间数据。

我们将智能体在此外挂框架内的生命周期结构化为两个不同的阶段：探索引导阶段和严格部署评估阶段。

**探索引导阶段。** 在任务的单个参考实例化上操作，智能体自主地与环境交互以发现一个有效的解决方案。在此阶段，规划器Π独特地被授予访问重置原语的权限，并在宽松的挂钟时间预算下运行。由于原语词汇表是固定的，探索完全专注于迭代组合：发现学习到的VLA原语与确定性解析原语的优化编排。规划器Π反复尝试不同的阶段化顺序、接触前位姿、vla_act的调用时机以及早返回终止阈值。它观察每个原语调用的物理效果，并在失败时修正路线。

成功完成任务后，智能体系统地将其经验抽象为图1所示的两个记忆模块。首先，经过验证的原语调用序列被序列化为JSONL格式。此文件显式记录成功的逐步原语调用，通过将具体空间坐标替换为符号感知查询来参数化它们，使序列在不同空间布局中可复用。此参数化的JSONL轨迹存储在任务特定记忆中，作为后续泛化测试的结构先验。其次，智能体从探索过程中提取通用启发式规则，并将其提交到持久的全局记忆。此共享仓库显式聚合了成功规则，例如利用完整任务指令的最优提示策略。它同时记录了关键失败模型，包括空抓取执行和虚假成功检测的识别。此聚合确保规划器避免在不同任务中重复历史上的陷阱。

**部署评估阶段。** 在未见环境变体（包括位置交换、指令重定向以及跨多个初始状态种子的测试）的正式评估期间，外挂框架施加严格的执行制度。重置原语完全禁用，操作步数预算显著缩短。为解决受扰动任务，规划器Π从任务特定记忆中检索预计算的JSONL轨迹，并使用实时RGB-D观测动态地将其接地。通过参考全局记忆中积累的成功规则和失败模型，智能体确定性地执行轨迹。在此严格阶段下取得的性能直接构成我们报告的基准测试结果，验证了Harness VLA框架的整体有效性。

### 2.3 统一原语接口

**表1：原语词汇表。** 本文通篇使用相同的原语名称；RoboCasa365额外使用移动基座原语用于厨房规模的阶段化。

| 原语 | 类型 | 角色 |
|---|---|---|
| move_to | 组合 | 使用环境嵌入求解器将末端执行器移动到世界坐标系的笛卡尔目标。 |
| move_pose | 组合 | 移动末端执行器同时协变姿态变量（如俯仰角），用于受限可达配置。 |
| rotate_wrist | 原子 | 在保持当前空间位置的同时施加手腕偏航设定点。 |
| rotate_pitch | 原子 | 在保持当前空间位置的同时施加手腕俯仰设定点。 |
| set_gripper | 原子 | 将夹爪驱动到打开或关闭的设定点，执行固定步数。 |
| release | 原子 | 在释放后置条件下打开夹爪。 |
| vla_act | VLA | 以短脉冲方式执行冻结的VLA用于局部接触丰富交互。 |
| navigate_to | 组合 | (RoboCasa365) 将移动基座驱动到世界坐标系位置用于厨房规模阶段化。 |
| move_base | 原子 | (RoboCasa365) 施加开环局部基座速度设定点用于精细重定位。 |

原语库𝒫是暴露给规划器的唯一动作接口。每个原语由单个JSON对象调用，在环境中执行直到内部后置条件达成，然后返回控制权以及刷新的观测。规划器因此从不直接发出低级力矩、关节目标或动作块；它选择一个原语，并从语言、RGB-D观测、本体感受和记忆中绑定其参数。

我们将𝒫组织为两个操控家族。**解析原语**是确定性的、基于模型的控制，由机器人运动学指定，不需要训练数据。它们分为组合原语（接收世界坐标系空间目标并运行嵌入求解器以协调多个自由度）和原子原语（驱动一个固有通道，如手腕方向、夹爪状态或基座速度，到参数化设定点）。**VLA原语** vla_act是一个学习到的策略调用，将提示和实时摄像头映射到用于局部接触丰富行为的动作块。探索性重置工具仅在引导期间使用，不计为操控原语。

表1给出了本文通篇使用的原语词汇表。共享的操控接口包含六个解析原语和一个VLA原语；RoboCasa365额外使用两个移动基座原语navigate_to和move_base进行厨房规模阶段化。RoboTwin双臂执行的细节见附录B。关键是，原语词汇表在评估前固定；规划器在部署时不能发明新的原语。

下面的紧凑JSON契约说明了共享接口；附录B使用相同的原语名称给出基准特定的可用性和实现说明。

```json
{"action": "move_to", "xyz": [<x>,<y>,<z>], ...}
{"action": "move_pose", "xyz": [<x>,<y>,<z>], "pose": <orientation>, ...}
{"action": "rotate_wrist","target_yaw": <float>, ...}
{"action": "rotate_pitch","target_pitch": <float>, ...}
{"action": "set_gripper", "gripper": <open|close>, ...}
{"action": "release", ...}
{"action": "navigate_to", "xy": [<x>,<y>], ...}
{"action": "move_base", "forward": <float>, "lateral": <float>, "turn": <float>, ...}
{"action": "vla_act", "prompt": <str>, "max_chunks": <int>, "stop": <predicate>}
```

**VLA支持的接触原语。** vla_act是用于接触丰富交互的学习原语。跨基准测试，vla_act涵盖抓取、约束放置、夹具驱动、按钮按压、抽屉或门操控、插入以及形态特定的接触行为。规划器提供任务条件的提示和早返回谓词τ。冻结的VLA f_θ然后发出动作块，直到τ被满足或块预算耗尽。这使VLA作为局部接触专家，而语义接地、空间重绑定、导航、重阶段化和长时序组合则保持在规划器控制之下。

## 3. 实验

我们将实证研究组织为两个部署制度和三个机制分析。在**少样本制度**下，Harness VLA遵循第2.2节中的记忆支持工作流：智能体在一个参考种子上执行任务级引导，将成功的原语轨迹存储在任务特定记忆中，并在新种子或扰动下重接地该轨迹。在**零样本制度**下，智能体必须在没有检索目标设置的任务特定记忆或全局记忆的情况下解决问题，测试在线规划器推理和冻结原语接口在没有任务特定外挂记忆支持的情况下的迁移能力。第3.1节详细说明配置，第3.2节展示少样本和零样本基准性能，第3.3节分析性能提升背后的机制。

### 3.1 实验设置

我们在四个基准家族上评估Harness VLA。桌面套件为LIBERO [38]和LIBERO-Pro [79]；家庭和双臂套件分别为RoboCasa365 [46]和RoboTwin C2R [44]。我们将基准特定的任务划分和种子协议推迟到附录C，这里聚焦于主要实证结果。在所有评估中，规划器在相同的冻结原语词汇表𝒫（第2.3节）上操作，并且不允许在部署时引入新原语。VLA原语使用基准特定的冻结策略实例化，同时保持统一的vla_act接口：对于LIBERO和LIBERO-Pro使用RLinf发布的pi05_libero130_fullshot π₀.₅-SFT检查点，记为π_RLinf [73]；对于RoboCasa365使用冻结的RLDX-1 RoboCasa检查点[30]；对于RoboTwin C2R使用我们后训练的LingBot-VLA检查点[70]。在下表中，Harness VLA (Codex)和Harness VLA (CC)分别表示使用Codex和Claude Code规划器实例化的相同外挂框架；CC是Claude Code的缩写。π_RLinf、RLDX-1和LingBot-VLA行作为其对应基准的直接冻结VLA基线。

### 3.2 整体基准性能

**使用任务特定记忆的少样本评估。** 我们首先在任务级引导已填充任务特定记忆后评估Harness VLA。此设置测试外挂框架是否可以复用在参考种子上发现的原语组织，同时从当前观测中接地所有空间参数。我们评估这种记忆支持的执行是否在标准LIBERO上保持强大的分布内操控性能，在LIBERO-Pro的指令重定向（T）和位置交换（S）扰动下保持鲁棒性，并将相同的原语接口扩展到RoboCasa365中的家庭厨房操控。

**标准LIBERO。** 表2报告了四个标准LIBERO套件的结果。Harness VLA (CC) 达到96.0%（384/400）的总体成功率，包括Object上的100.0%和LIBERO-10上的93.0%。与vla_act内部使用的冻结π_RLinf检查点（总体95.3%）相比，Harness VLA保持了有竞争力的标准套件性能，同时通过可控的原语接口暴露相同的策略，用于下面的扰动评估。

**表2：标准LIBERO上的成功率（%）。** π_RLinf [73]和Harness VLA (CC)由我们在每个套件100次试验（10个任务×10个种子）上评估。Harness VLA在vla_act内部使用相同的RLinf发布的pi05_libero130_fullshot π₀.₅-SFT检查点。加粗标记每个套件或总体列中的最佳方法。

| 方法 | Spatial | Object | Goal | LIBERO-10 | 总体 |
|---|---|---|---|---|---|
| OpenVLA [31] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| NORA [27] | 85.6 | 89.4 | 80.0 | 63.0 | 79.5 |
| π₀ [5] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π_RLinf | 99.0 | 96.0 | 97.0 | 89.0 | 95.3 |
| AtomVLA [60] | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |
| **Harness VLA (CC)** | **97.0** | **100.0** | **94.0** | **93.0** | **96.0** |

**表3：LIBERO-Pro在指令重定向（T）和位置交换（S）扰动下跨Spatial、Object、Goal和LIBERO-10的总体成功率（%）。** 每个单元格聚合100次试验（10个任务×10个种子）；"/"表示不可用或未报告的单元格。Cap-X和RATS仅报告六个非LIBERO-10单元格，因此其总体值仅对报告的单元格取平均。π_RLinf和Harness VLA由我们使用RLinf发布的pi05_libero130_fullshot π₀.₅-SFT检查点[73]评估；Harness VLA通过vla_act原语暴露此冻结检查点。加粗标记每个评估单元格或总体列中报告的最佳方法。

| 方法 | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | L10-T | L10-S | 总体 |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA [31] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| π₀ [5] | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| π₀.₅ [53] | 1.0 | 20.0 | 1.0 | 17.0 | 2.0 | 38.0 | 1.0 | 8.0 | 11.0 |
| MolmoAct [32] | 0.0 | 0.0 | 0.0 | 6.0 | 0.0 | 0.0 | 6.0 | 0.0 | 1.5 |
| NORA [27] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| X-VLA [78] | 0.0 | 0.0 | 8.0 | 2.0 | 9.0 | 1.0 | 10.0 | 0.0 | 3.8 |
| AtomVLA [60] | 1.0 | 16.0 | 0.0 | 10.0 | 11.0 | 2.0 | 9.0 | 1.0 | 6.3 |
| Cap-X [15] | 14.0 | 12.0 | 18.0 | 22.0 | 17.0 | 26.0 | / | / | 18.2 |
| RATS [75] | 31.0 | 29.0 | 63.0 | 61.0 | 36.0 | 43.0 | / | / | 43.8 |
| π_RLinf | 42.0 | 59.0 | 71.0 | 78.0 | 45.0 | 42.0 | 49.0 | 14.0 | 50.0 |
| Harness VLA (Codex) | 81.0 | 69.0 | 94.0 | 91.0 | 75.0 | 66.0 | 52.0 | 49.0 | 72.1 |
| **Harness VLA (CC)** | **94.0** | **80.0** | **88.0** | **90.0** | **87.0** | **87.0** | **71.0** | **62.0** | **82.4** |

**LIBERO-Pro。** 表3报告了LIBERO-Pro的总体结果，涵盖Spatial、Object、Goal和LIBERO-10在指令重定向（T）和位置交换（S）扰动下的表现。现有端到端VLA模型在这些分布偏移下急剧退化，RATS是其报告单元格上最强的先前基线，总体为43.8%。Harness VLA使用Codex达到72.1%，使用CC达到82.4%，在标题比较中比RATS高出38.6个百分点。直接的π_RLinf基线在我们的协议下达到50.0%总体，表明提升并不简单地来自冻结的VLA骨干。指令重定向和位置交换设置中的全面提升表明，少样本外挂框架已经学到了对固定原语库的可复用分工：规划器可以重绑定目标，使用解析原语重阶段化场景并处理非接触执行，并仅在局部接触丰富操控中调用VLA。

**RoboCasa365。** RoboCasa365将评估从桌面操控扩展到家庭厨房任务，涉及移动阶段化、关节夹具和更长的组合例程。表4比较了Harness VLA与先前RoboCasa365论文中报告的结果以及用于标题比较的RLDX-1基线。RLDX-1达到30.0%的任务加权总体成功率，而Harness VLA使用Codex达到55.4%，使用CC达到48.6%；因此Codex实例化比RLDX-1高出25.4个百分点。这些提升与预期分解一致：规划器处理导航、阶段化和局部失败后的重阶段化，而冻结的VLA保持为局部接触丰富原语。

**表4：RoboCasa365成功率（%）。** 分隔线上方的基线行是相应先前论文中报告的结果，RLDX-1在我们的协议下作为直接冻结VLA基线评估。Harness VLA仅使用一个参考种子进行引导；报告的评估使用Atomic-Seen的10个保留种子以及Composite-Seen和Composite-Unseen的5个保留种子。加粗标记每个划分中的最佳方法。

| 方法 | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---|---|---|
| RLDX-1 [30] | 60.0 | 21.3 | 5.0 |
| WorldDreamer [66] | 66.3 | 26.7 | 9.0 |
| π₀.₅ [53] | 39.6 | 7.1 | 1.2 |
| π₀ [5] | 34.6 | 6.1 | 1.1 |
| **Harness VLA (Codex)** | **91.6** | **56.3** | **13.8** |
| Harness VLA (CC) | 79.4 | 47.5 | 15.0 |

**无引导外挂记忆的零样本评估。**
**LIBERO-Pro Goal。** 为将在线规划器推理与引导的外挂记忆分离，我们在严格的零样本设置下评估LIBERO-Pro Goal，其中智能体不检索目标设置的任务特定记忆或相应的全局记忆。表5显示，零样本Harness VLA (CC)在两种扰动制度下均优于Cap-X，在位置交换（Pos-S）上达到31.0%，在指令重定向（Task-T）上达到79.0%，而Cap-X分别为25.6%和16.8%。与表3中少样本Goal单元格的比较阐明了引导外挂记忆的贡献。没有此记忆，规划器在指令重定向下保留了大部分语义重绑定能力（Goal-T上零样本79.0%对比少样本87.0%），但在位置交换下大幅下降（Goal-S上零样本31.0%对比少样本87.0%）。这一差距表明，空间扰动操控从探索期间发现的任务特定原语组织中获益良多：解析原语在接触丰富阶段周围提供定位、阶段化、传输和释放，而vla_act在学习到的交互点被调用，并可在失败后重阶段化。

**表5：LIBERO-Pro Goal上每任务成功率（%）：零样本Harness VLA (CC)（无任务特定记忆检索，每任务10种子）vs Cap-X [15]。** Pos = 交换（S），Task = 指令重定向（T）。加粗标记每个设置和任务或平均列中的获胜者。

| 设置 | 方法 | 任务0 | 任务1 | 任务2 | 任务3 | 任务4 | 任务5 | 任务6 | 任务7 | 任务8 | 任务9 | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pos (S) | Cap-X | 0.0 | 4.0 | 0.0 | 36.0 | 22.0 | 60.0 | 4.0 | 2.0 | 62.0 | 66.0 | 25.6 |
| Pos (S) | **Harness VLA (CC)** | 0.0 | 10.0 | 0.0 | **20.0** | **90.0** | 0.0 | **10.0** | **80.0** | **100.0** | 0.0 | **31.0** |
| Task (T) | Cap-X | 0.0 | 0.0 | 10.0 | 38.0 | 12.0 | 4.0 | 34.0 | 12.0 | 40.0 | 18.0 | 16.8 |
| Task (T) | **Harness VLA (CC)** | **10.0** | **100.0** | **90.0** | **100.0** | **20.0** | **80.0** | **90.0** | **100.0** | **100.0** | **100.0** | **79.0** |

**RoboTwin C2R。** RoboTwin C2R评估零样本清洁到随机迁移：智能体从清洁设置获取任务特定记忆轨迹，并将其直接迁移到随机化的任务实例，无需随机化设置引导、额外的任务级探索或VLA微调。这里的vla_act后端是LingBot-VLA，我们针对RoboTwin专门训练的VLA检查点；在后训练后，它对直接VLA基线和Harness VLA评估均为冻结状态。表6将C2R成功率与代表性VLA基线进行了比较。直接LingBot-VLA达到50.4%，而Harness VLA使用Codex将相同冻结后端提升到58.0%，使用CC提升到58.4%。表中还报告了外部VLA基线供参考。

**表6：RoboTwin C2R成功率（%）。** LingBot-VLA是我们后训练的RoboTwin VLA检查点，在没有智能体级分解的情况下直接评估，也是Harness VLA使用的冻结vla_act后端。其他VLA行是代表性外部基线。Harness VLA在50个任务上评估，每个任务5个随机化种子。加粗标记最佳方法。

| 基准 | GR00T-N1.7 [4] | π₀.₅ [53] | StarVLA [12] | LingBot-VLA [70] | Harness VLA (Codex) | **Harness VLA (CC)** |
|---|---|---|---|---|---|---|
| RoboTwin C2R | 20.7 | 47.9 | 10.6 | 50.4 | 58.0 | **58.4** |

### 3.3 实验分析

> **图3：** 两个LIBERO-Pro单元格的终止状态帧。第一个三元组比较了π_RLinf在标准Object任务上、π_RLinf在任务扰动Object-Pro变体上、以及Harness VLA在相同扰动任务上的表现；当任务描述重定向目标而视觉场景保持相似时，π_RLinf重复标准行为而非遵循新指令。第二个三元组展示了交换扰动Goal-Pro任务的类似比较；π_RLinf在布局变化后盲目地将物体移向训练时区域，而Harness VLA通过智能体规划器Π重接地目标，使用解析原语进行阶段化，并调用vla_act执行局部接触丰富操作。

我们通过三个不同的发现来分析这些结果背后的机制。**关键发现1** 聚焦于规划器的语义和场景重接地；**关键发现2** 研究规划器阶段化的vla_act调用和重试；**关键发现3** 研究解析原语如何将非接触执行与接触丰富控制隔离。除非另有说明，以下分析以Harness VLA (CC)为代表实例化，不失一般性：Codex和CC共享相同的外挂框架、记忆接口、原语库、冻结VLA接口和评估协议，仅在规划器骨干上有所不同。LIBERO家族分析使用与主评估相同的π_RLinf检查点，而非LIBERO分析使用其基准特定的冻结VLA原语。

**关键发现1：规划器级语义重接地恢复任务条件行为。**
表3中Harness VLA与端到端VLA之间的巨大差距是在不改变视觉运动骨干的情况下实现的。π_RLinf已经解决了这些任务的标准变体，但图3显示其行为对任务描述和当前场景绑定的条件较弱。在任务扰动的Object-Pro案例中，视觉场景保持相似而指令重定向了目标，但π_RLinf重复标准行为而非遵循新任务描述。在交换扰动的Goal-Pro案例中，物体布局发生变化，但π_RLinf仍然将物体移向训练时区域。Harness VLA使语义接地在规划器层面显式化：规划器Π解析任务描述，从实时RGB-D观测中解析当前接触目标，使用解析原语进行阶段化和重定位，并仅在局部接触丰富阶段调用或重新调用vla_act（关键发现2）。因此，语义和场景级推理由规划器处理，而冻结的VLA仅负责在规划器提供的绑定下执行接触丰富操作。

**关键发现2：规划器阶段化的VLA调用提高冻结策略的可靠性。**
在Harness VLA中，规划器不将VLA作为一次性黑盒调用。虽然VLA作为稀疏调用而非连续控制器使用，但每次调用都是规划器选择的局部接触丰富尝试，其阶段化可以决定冻结策略是否成功。因此，规划器Π将vla_act视为一个局部接触丰富原语，其调用可以被重阶段化和重试。给定期望的接触目标（待操作的物体、夹具或局部交互区域），规划器使用解析原语将机器人置于可行的接触前配置，调用VLA，观察产生的接触状态，并决定下一步是继续任务还是重构局部尝试。

> **图4：** 自适应VLA调用跨基准提升成功率。每个面板绘制了累积任务成功率作为每个回合允许的最大VLA原语调用次数的函数。蓝色虚线标记对应的冻结策略基线，灰色虚线标记具有所有规划器选定调用的完整Harness VLA性能。跨LIBERO-Pro、RoboCasa365和RoboTwin C2R，成功率在前几次VLA调用后快速上升，然后向完整外挂结果饱和，表明重复的规划器阶段化调用是有用的但保持稀疏。

**首先，阶段化恢复VLA兼容的局部状态。** 在语义重定向和空间布局变化等部署扰动下，原始VLA视角或接触前位姿可能不再以熟悉的配置暴露正确的接触目标。通过在当前场景周围重阶段化机器人，智能体规划器Π将目标带回VLA兼容的局部观测中，同时保留正确的语义绑定。这一机制解释了外挂框架为何能在不改变参数的情况下改进冻结的VLA：它学习了VLA应该从何处开始行动，而不是要求策略自行吸收完整的分布偏移。

**其次，重试将接触失败局部化。** 由于VLA执行是随机的且短时序接触在物理上是脆弱的，单次失败尝试不必终止整个展开过程。Harness VLA将此类错误局部化到当前的接触丰富子任务：规划器可以观察到不完整或不稳定的结果，重阶段化机器人，并重新调用vla_act，而不是让瞬时失败通过单体长时序策略传播。因此，重复的VLA调用不是连续的低级控制；它们是稀疏的、规划器选择的尝试，使接触丰富执行可恢复。

> **图5：** 自适应VLA调用的代表性展开帧。上行：LIBERO-Pro Object任务4上的Harness VLA展开。规划器在牛奶盒周围反复调用vla_act，因为中间的抓取或放置尝试使物体处于篮子外或仅部分在篮子内；重阶段化末端执行器并重试局部接触丰富操作后，牛奶盒最终稳定放入篮子内。下行：RoboCasa365 PreSoakPan展开。规划器调整移动基座和手臂姿态围绕锅，重试vla_act直到获得稳定抓取，将锅放入水槽，随后再次调用vla_act驱动水龙头。这些示例表明重复的VLA调用不是连续控制，而是嵌入解析导航、阶段化和验证中的规划器选择的接触尝试。

图4通过限制每个回合允许的最大VLA原语调用次数提供了此效应的总体证据。少量规划器选择的调用已经超过相应的冻结策略基线，而额外的调用进一步提升了更长或更接触密集型任务的成功率。图5给出了此曲线背后的代表性案例研究：规划器观察到不完整或不稳定的接触结果，重阶段化机器人或基座，并再次调用vla_act进行下一次局部接触尝试。综合来看，这些结果表明VLA被稀疏地使用，但重阶段化和再次调用的能力对外挂框架的鲁棒性至关重要。

**关键发现3：解析原语将非接触执行与接触丰富控制隔离。**
解析原语不替代VLA在接触丰富操作上的作用。相反，它们处理任务周围的非接触结构：自由空间传输、接触前阶段化、手腕或基座重定向、回退和接触后重定位。这让规划器将vla_act保留用于需要学习到的视觉运动控制的局部接触丰富阶段，包括抓取、约束放置、按钮按压、水龙头转动、抽屉操控和咖啡机操作。

一旦机器人与目标建立稳定接触，规划器Π可以使用解析原语移动、旋转或导航机器人到下一个相关区域，同时在下一个接触丰富阶段开始时再次调用VLA。因此，解析词汇表不是自行解决接触丰富操控；它扩展了相同冻结VLA可复用的条件。通过处理每个局部交互周围的非接触上下文，规划器防止VLA对长时序组合、场景级接地以及展开中的每个中间动作负责。

> **图6：** 跨基准的任务完成归因。条形图显示成功展开的最终基准完成谓词在解析原语（蓝色）或VLA原语（橙色）之后触发的比例。LIBERO Pro家族任务大多在VLA建立稳定接触后由解析原语完成，而RoboCasa365和RoboTwin C2R包含更多终端接触丰富操作，如夹具驱动、约束放置或双臂物体交互。

图6通过按触发最终基准完成谓词的原语类别分离成功展开，提供了此分工的总体归因。LIBERO Pro家族任务通常在接触建立后通过解析传输、释放或重定位完成。在RoboCasa365和RoboTwin C2R中，最终谓词通常直接依赖于接触丰富操作，因此成功展开更频繁地在VLA原语内部完成。图7给出了代表性示例：解析原语定位执行、暴露失败或不完整的接触，并将机器人移回可以再次调用vla_act的配置。综合证据支持相同的分工：解析原语在接触丰富阶段周围组织任务，而VLA保持负责需要学习到的视觉运动控制的阶段。

> **图7：** 围绕接触丰富阶段的解析分解的代表性展开帧。上行：在LIBERO-10-Pro交换任务上，智能体首先调用vla_act并开始向篮子移动，然后在move_to期间检测到VLA实际上没有抓取奶油奶酪盒。规划器移回，重试vla_act，并在成功抓取后使用move_to和release完成子任务。下行：在RoboCasa SteamInMicrowave composite-seen任务上，智能体使用vla_act成功抓取碗，搜索并重定位直到微波炉被定位，调用vla_act将碗放入内部，使用move_to将其推入，关上门，最后使用move_to和navigate_to按下开关。

## 4. 相关工作

Harness VLA处于三条工作线的交汇处：端到端机器人基础策略、多模态LLM智能体和程序化机器人控制系统。我们通过每种方法分配给学习策略和显式控制的角色来审视这些领域。这一视角阐明了我们的定位：不是微调更强的VLA或扩展原语库，而是研究记忆引导的智能体如何将冻结的VLA转化为可控的接触丰富原语，并将其与固定的解析控制器组合。

**VLA模型。** 端到端的视觉-语言-动作（VLA）模型通过用动作头扩展预训练的视觉-语言骨干，将自然语言指令和视觉观测直接映射到低级机器人动作。RT-1 [7]和Octo [48]开辟了通用策略路线，由RT-2 [6]和Open X-Embodiment发布[49]扩展，建立了将大型VLM与跨形态机器人演示联合训练的训练模式。OpenVLA [31]通过将Prismatic风格VLM [29]与Llama-2动作分词器结合，将这一范式带入开放领域，而流匹配π₀ [5]和π₀.₅ [53]模型报告了通过异构数据和分布外语言联合训练获得的显著提升。近期大规模系统如GR00T [4]和Gemini Robotics [16]继续将这一模式扩展到人形和通用形态，同时还有3D感知VLA [77, 14]、基于VLM的模仿[35, 22]以及CLIP条件控制器[58, 11, 76]等相关工作。然而，经验上这些模型表现出明显的不对称性：它们在接触丰富的视觉运动阶段最强——特别是对不规则抓取和击败解析控制器的夹具驱动——但在指令跟随、长时序组合和分布外场景中急剧退化[31, 53, 65]。这种不对称性激励了一种分解方式：将VLA委托为规划器选择的接触丰富操作，而高级控制器负责语言解释、目标接地、传输、姿态调整、导航和释放。

**LLM驱动的多模态智能体。** 前沿的多模态大型语言模型已经快速缩小了密集感知、空间推理和长时序工具使用方面的差距。近期发布的GPT-5.2 [52]、Gemini 3 [54]、Qwen3-VL [3]、Claude 4家族（Sonnet 4.5和Opus 4.7 [2]）以及Llama 4 [42]展示了比GPT-4o [50] / Gemini 1.5 [17]一代在物理场景接地上质的飞跃，而Molmo [13]和Qwen3-VL [3]等开源权重模型使这些能力广泛可及。针对性的空间基准[8]和工具增强的浏览智能体[51, 20, 69, 18, 26, 28, 33]进一步表明，在闭环接口下，这些骨干可以在丰富的部分观测环境中维持多跳感知和决策。这些进步使得将语义接地和确定性操控阶段——语言解析、目标定位、传输规划、姿态调整、导航和释放时机——委托给智能体栈顶层的前沿VLM变得越来越可行[16, 62, 64]。我们建立在这一前提上，但不是将VLM端到端地驱动机器人，而是将其置于一个智能体外挂框架内，该框架发出结构化原语调用，观察执行反馈，并迭代——将直接动作预测保留给规划器选择的接触丰富阶段。

**程序化和工具使用的机器人智能体。** Code-as-policies系统将机器人控制重塑为程序合成：模型编写一个可执行程序，协调感知和运动API，利用LLM的组合泛化能力同时保持低级控制的确定性。从Code-as-Policies [36]、ProgPrompt [59]、Instruct2Act [23]和ChatGPT-for-Robotics [63]开始，这一范式已被扩展为多模态程序合成（RoboCodeX [43]、ViperGPT [61]、VisProg [21]）、3D价值图生成[24]、VLM监督装配[19]以及长时序智能体框架[34, 56]。Harness VLA共享这一文献的目标——通过显式感知和控制接口使语言模型推理可执行——但在动作表示上有所不同：我们的智能体规划器不合成可执行代码或新控制程序。它在闭环外挂框架内发出结构化JSON原语调用，在每个原语之后观察执行结果，并从当前RGB-D证据和记忆中重新绑定下一个原语参数。近期工作还研究了智能体如何增长自己的可复用技能库：ASPIRE [39]使用细粒度执行轨迹来诊断失败，合成经过验证的修复，并将产生的定位、导航、运动、抓取和调试模式纳入一个持续扩展的技能库。这一方向与我们的工作互补：ASPIRE扩展智能体的可复用技能，而Harness VLA刻意保持原语词汇表固定，并研究记忆引导的组合如何在不进行部署时原语扩展的情况下扩展冻结的VLA。另一条线索借鉴了软件工程智能体：可执行代码在经验上是LLM智能体的强动作表示[67]，SWE-agent [71]和OpenHands [68]等系统为迭代编辑、执行和反馈形式化了外挂框架。我们借用了外挂框架原则——结构化接口、持久状态、执行反馈和记忆——而非要求动作以代码表示。通过自我修正[57, 40, 9]和跨步骤持久符号状态[72]进一步提高可靠性，而LLM驱动的规划器[1, 25, 45, 74, 37, 10, 55, 41]展示了LLM可以序列化预训练原语或模块、接地3D场景并从失败中恢复。然而，两个局限性在文献中反复出现。首先，任务特定的执行轨迹很少被表示为可复用的参数化记忆，可以在新的空间布局下重新接地。其次，失败知识很少被提炼为全局记忆，以防止规划器重复已知的空抓取、虚假成功或不稳定阶段化选择。Voyager [64]展示了持久记忆可以改善数字沙盒中的具身智能体，但这种以记忆为中心的设计尚未与VLA支持的接触丰富原语结合用于物理操控。我们的框架将两者耦合：冻结的VLA作为通过单一原语接口调用的接触丰富专家，成功的原语序列存储在任务特定记忆中，可复用的成功规则和失败模型被提炼到全局记忆中。这两个设计选择——接触丰富操作的VLA委托，以及所有其他部分的记忆增强原语组合——共同使单一记忆引导的智能体规划器覆盖给定环境暴露的完整任务分布，包括击败其语言通道基本退化[31, 53]的单体VLA的改写和重定向自然语言指令。

## 5. 结论与局限性

我们引入了Harness VLA，一个非对称分层框架，它将冻结的VLA作为LLM驱动智能体内的单一接触丰富原语接口，将传输、姿态、导航和释放阶段委托给规划器。从智能体外挂工程的角度来看，Harness VLA表明可靠的操控不仅可以来自训练更强的策略，还可以来自用可审计的执行循环、固定原语契约、记忆、反馈和任务级验证围绕冻结策略。跨标准和严重扰动基准的评估证实，Harness VLA实现了最先进的鲁棒性。这些结果表明，预训练的VLA在被隔离到接触丰富视觉运动控制时最有效；将语义和空间绑定从VLA中抽象出来，防止了在单体部署中频繁观察到的灾难性失败。

**局限性与未来工作。** 我们当前的框架受到高级规划器与低级VLA之间开环反馈回路的限制。此外，系统缺乏通过环境奖励和人类偏好的联合微调——这个问题需要未来样本高效的强化学习（如GRPO）来解决。最后，缺乏细粒度图像标注限制了在高度杂乱的长时序任务中的结构推理。一个互补的未来方向是将我们的固定词汇表组合策略与ASPIRE [39]等自动技能发现系统结合：当重复的原语组合揭示缺失的抽象时，智能体可以提出、验证并接纳一个新的可复用技能，同时保留本文研究的可审计原语接口和VLA支持的接触专业化。

## 参考文献

[1] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. Jauregui Ruano, K. Jeffrey, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao, K. Rao, J. Reymann, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, C. Watkins, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, A. Zeng, and Y. Zheng (2022) Do as I can, not as I say: grounding language in robotic affordances. 发表于 Conference on Robot Learning (CoRL). 引用：§4.

[2] Anthropic (2025) Introducing Claude Sonnet 4.5. 在线. 引用：§4.

[3] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, et al. (2025) Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. 引用：§4.

[4] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025) Gr00t n1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734. 引用：Table 6, §4.

[5] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2025) π₀: A vision-language-action flow model for general robot control. 发表于 Robotics: Science and Systems (RSS). 引用：§1, Table 2, Table 3, Table 4, §4.

[6] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. (2023) RT-2: vision-language-action models transfer web knowledge to robotic control. 发表于 Conference on Robot Learning (CoRL). 引用：§1, §4.

[7] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2022) Rt-1: robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817. 引用：§4.

[8] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia (2024) SpatialVLM: endowing vision-language models with spatial reasoning capabilities. 发表于 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 引用：§4.

[9] X. Chen, M. Lin, N. Schärli, and D. Zhou (2024) Teaching large language models to self-debug. 发表于 International Conference on Learning Representations (ICLR). 引用：§4.

[10] Y. Chen, J. Arkin, C. Dawson, Y. Zhang, N. Roy, and C. Fan (2024) AutoTAMP: autoregressive task and motion planning with LLMs as translators and checkers. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§4.

[11] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song (2023) Diffusion policy: visuomotor policy learning via action diffusion. 发表于 Robotics: Science and Systems (RSS). 引用：§4.

[12] S. Community (2026) StarVLA: a lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014. 引用：Table 6.

[13] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, et al. (2025) Molmo and pixmo: open weights and open data for state-of-the-art vision-language models. 发表于 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 引用：§4.

[14] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Toussaint, K. Greff, A. Zeng, I. Mordatch, and P. Florence (2023) PaLM-E: an embodied multimodal language model. 发表于 International Conference on Machine Learning (ICML). 引用：§4.

[15] M. Fu, J. Yu, K. El-Refai, E. Kou, H. Xue, H. Huang, W. Xiao, G. Wang, F. Li, G. Shi, et al. (2026) CaP-X: a framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435. 引用：§1, Table 3, Table 5, Table 5.

[16] Gemini Robotics Team, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. (2025) Gemini robotics: bringing ai into the physical world. arXiv preprint arXiv:2503.20020. 引用：§4, §4.

[17] Gemini Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023) Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 引用：§4.

[18] X. Geng, P. Xia, Z. Zhang, X. Wang, Q. Wang, R. Ding, C. Wang, J. Wu, Y. Zhao, K. Li, et al. (2025) Webwatcher: breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748. 引用：§4.

[19] A. Goldberg, K. Kondap, T. Qiu, Z. Ma, L. Fu, J. Kerr, H. Huang, K. Chen, K. Fang, and K. Goldberg (2025) Blox-net: generative design-for-robot-assembly using VLM supervision, physics simulation, and a robot with reset. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§4.

[20] Google (2024) Gemini Deep Research. Google Blog. 引用：§4.

[21] T. Gupta and A. Kembhavi (2023) Visual programming: compositional visual reasoning without training. 发表于 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 引用：§1, §4.

[22] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S. Zhu, B. Jia, and S. Huang (2024) An embodied generalist agent in 3d world. 发表于 International Conference on Machine Learning (ICML). 引用：§4.

[23] S. Huang, Z. Jiang, H. Dong, Y. Qiao, P. Gao, and H. Li (2023) Instruct2Act: mapping multi-modality instructions to robotic actions with large language model. arXiv preprint arXiv:2305.11176. 引用：§4.

[24] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei (2023) Voxposer: composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973. 引用：§4.

[25] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al. (2022) Inner monologue: embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608. 引用：§4.

[26] W. Huang, Y. Zeng, Q. Wang, Z. Fang, S. Cao, Z. Chu, Q. Yin, S. Chen, Z. Yin, L. Chen, et al. (2026) Vision-deepresearch: incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060. 引用：§4.

[27] C. Hung, Q. Sun, P. Hong, A. Zadeh, C. Li, U. Tan, N. Majumder, S. Poria, et al. (2025) NORA: a small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854. 引用：§1, Table 2, Table 3.

[28] B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, and J. Han (2025) Search-r1: training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516. 引用：§4.

[29] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh (2024) Prismatic vlms: investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865. 引用：§4.

[30] D. Kim, H. Jang, M. Koo, S. Jang, T. Kim, B. Kim, B. Yoon, C. Jang, D. Choi, D. Han, et al. (2026) Rldx-1 technical report. arXiv preprint arXiv:2605.03269. 引用：§3.1, Table 4.

[31] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024) OpenVLA: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. 引用：§1, Table 2, Table 3, §4, §4.

[32] J. Lee, J. Duan, H. Fang, Y. Deng, S. Liu, B. Li, B. Fang, J. Zhang, Y. R. Wang, S. Lee, W. Han, W. Pumacay, A. Wu, R. Hendrix, K. Farley, E. VanderBilt, A. Farhadi, D. Fox, and R. Krishna (2025) MolmoAct: action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917. 引用：§1, Table 3.

[33] K. Li, Z. Zhang, H. Yin, L. Zhang, L. Ou, J. Wu, W. Yin, B. Li, Z. Tao, X. Wang, et al. (2025) WebSailor: navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592. 引用：§4.

[34] R. Li, Y. Zhou, Y. Zhu, K. Chen, J. Wang, S. Wang, K. Hu, M. Yu, B. Jiang, Z. Su, J. Ma, X. He, Y. Shen, Y. Yang, G. Ren, M. Yao, W. Wang, and Y. Mu (2026) RoboClaw: an agentic framework for scalable long-horizon robotic tasks. arXiv preprint arXiv:2603.11558. 引用：§4.

[35] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, H. Dong, H. Hu, W. Zhan, H. Wu, Y. Han, and T. Kong (2024) Vision-language foundation models as effective robot imitators. 发表于 International Conference on Learning Representations (ICLR). 引用：§4.

[36] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng (2023) Code as policies: language model programs for embodied control. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§1, §4.

[37] B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone (2023) LLM+P: empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477. 引用：§4.

[38] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone (2023) LIBERO: benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems (NeurIPS) 36. 引用：§C.1, §3.1.

[39] R. Lu, Y. Wu, E. Kou, L. Fu, W. Xiao, A. Mandlekar, Y. Xu, G. Shi, K. Goldberg, A. Chen, et al. (2026) ASPIRE: agentic/skills discovery for robotics. arXiv preprint arXiv:2607.00272. 引用：§4, §5.

[40] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. (2023) Self-refine: iterative refinement with self-feedback. 发表于 Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. 引用：§4.

[41] Z. Mandi, S. Jain, and S. Song (2024) RoCo: dialectic multi-robot collaboration with large language models. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§4.

[42] Meta (2025) Llama 4 Herd. Meta Blog. 引用：§4.

[43] Y. Mu, J. Chen, Q. Zhang, S. Chen, Q. Yu, C. Ge, R. Chen, Z. Liang, M. Hu, C. Tao, P. Sun, H. Yu, C. Yang, W. Shao, W. Wang, J. Dai, Y. Qiao, M. Ding, and P. Luo (2024) RoboCodeX: multimodal code generation for robotic behavior synthesis. arXiv preprint arXiv:2402.16117. 引用：§4.

[44] Y. Mu, T. Chen, Z. Chen, S. Peng, Z. Lan, Z. Gao, Z. Liang, Q. Yu, Y. Zou, M. Xu, et al. (2025) RoboTwin: dual-arm robot benchmark with generative digital twins. 发表于 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 引用：§3.1.

[45] Y. Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai, Y. Qiao, and P. Luo (2023) EmbodiedGPT: vision-language pre-training via embodied chain of thought. 发表于 Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. 引用：§4.

[46] S. Nasiriany, S. Nasiriany, A. Maddukuri, and Y. Zhu (2026) RoboCasa365: a large-scale simulation framework for training and benchmarking generalist robots. arXiv preprint arXiv:2603.04356. 引用：§3.1.

[47] X. Ning, K. Tieu, D. Fu, T. Wei, Z. Li, Y. Bei, J. Zou, M. Ai, Z. Liu, T. Li, et al. (2026) Code as agent harness. arXiv preprint arXiv:2605.18747. 引用：§1, §2.2.

[48] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. L. Tan, D. Sadigh, C. Finn, and S. Levine (2023) Octo: an open-source generalist robot policy. 在线. 引用：§4.

[49] Open X-Embodiment Collaboration (2023) Open X-Embodiment: robotic learning datasets and RT-X models. 在线. 引用：§4.

[50] OpenAI (2024) GPT-4o system card. arXiv preprint arXiv:2410.21276. 引用：§4.

[51] OpenAI (2025) Introducing deep research. OpenAI Blog. 引用：§4.

[52] OpenAI (2025) Introducing GPT-5.2. 在线. 引用：§4.

[53] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky (2025) π₀.₅: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. 引用：§1, Table 3, Table 4, Table 6, §4, §4.

[54] S. Pichai, D. Hassabis, and K. Kavukcuoglu (2025) A new era of intelligence with Gemini 3. Google Blog. 引用：§4.

[55] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. Reid, and N. Suenderhauf (2023) SayPlan: grounding large language models using 3D scene graphs for scalable task planning. 发表于 Conference on Robot Learning (CoRL). 引用：§4.

[56] J. Shi, R. Yang, K. Chao, B. S. Wan, Y. S. Shao, J. Lei, J. Qian, L. Le, P. Chaudhari, K. Daniilidis, et al. (2025) Maestro: orchestrating robotics modules with vision-language models for zero-shot generalist robots. 发表于 NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI. 引用：§1, §4.

[57] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao (2023) Reflexion: language agents with verbal reinforcement learning. 发表于 Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. 引用：§4.

[58] M. Shridhar, L. Manuelli, and D. Fox (2022) Cliport: what and where pathways for robotic manipulation. 发表于 Conference on Robot Learning (CoRL). 引用：§4.

[59] I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg (2023) ProgPrompt: generating situated robot task plans using large language models. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§1, §4.

[60] X. Sun, Z. Xu, C. Cao, Z. Liu, Y. Sun, J. Pang, R. Zhang, Z. Yang, K. Pang, D. He, et al. (2026) AtomVLA: scalable post-training for robotic manipulation via predictive latent world models. arXiv preprint arXiv:2603.08519. 引用：Table 2, Table 3.

[61] D. Surís, S. Menon, and C. Vondrick (2023) ViperGPT: visual inference via Python execution for reasoning. 发表于 IEEE/CVF International Conference on Computer Vision (ICCV). 引用：§4.

[62] G. R. Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham, et al. (2025) Gemini robotics 1.5: pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342. 引用：§4.

[63] S. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor (2023) ChatGPT for robotics: design principles and model abilities. arXiv preprint arXiv:2306.17582. 引用：§4.

[64] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar (2023) Voyager: an open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291. 引用：§1, §4, §4.

[65] J. Wang, M. Leonard, K. Daniilidis, D. Jayaraman, and E. S. Hu (2025) Evaluating π₀ in the wild: strengths, problems, and the future of generalist robot policies. 在线. 引用：§1, §4.

[66] X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu (2024) Worlddreamer: towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985. 引用：Table 4.

[67] X. Wang, Y. Chen, L. Yuan, Y. Zhang, Y. Li, H. Peng, and H. Ji (2024) Executable code actions elicit better LLM agents. 发表于 International Conference on Machine Learning (ICML). 引用：§1, §1, §2.2, §4.

[68] X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F. Li, R. Ma, M. Zheng, B. Qian, Y. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig (2025) OpenHands: an open platform for AI software developers as generalist agents. 发表于 International Conference on Learning Representations (ICLR). 引用：§1, §2.2, §4.

[69] J. Wu, Z. Deng, W. Li, Y. Liu, B. You, B. Li, Z. Ma, and Z. Liu (2025) MMSearch-r1: incentivizing lmms to search. arXiv preprint arXiv:2506.20670. 引用：§4.

[70] W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma, et al. (2026) A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692. 引用：§D.3, §3.1, Table 6.

[71] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press (2024) SWE-agent: agent-computer interfaces enable automated software engineering. 发表于 Advances in Neural Information Processing Systems (NeurIPS). 引用：§1, §1, §2.2, §4.

[72] T. Yoneda, J. Fang, P. Li, H. Zhang, T. Jiang, S. Lin, B. Picker, D. Yunis, H. Mei, and M. R. Walter (2024) Statler: state-maintaining language models for embodied reasoning and planning. 发表于 IEEE International Conference on Robotics and Automation (ICRA). 引用：§4.

[73] C. Yu, Y. Wang, Z. Guo, H. Lin, S. Xu, H. Zang, Q. Zhang, Y. Wu, C. Zhu, J. Hu, et al. (2025) RLinf: flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965. 引用：§3.1, Table 2, Table 2, Table 3, Table 3.

[74] A. Zeng, M. Attarian, B. Ichter, K. Choromanski, A. Wong, S. Welker, F. Tombari, A. Purohit, M. Ryoo, V. Sindhwani, J. Lee, V. Vanhoucke, and P. Florence (2023) Socratic models: composing zero-shot multimodal reasoning with language. 发表于 International Conference on Learning Representations (ICLR). 引用：§4.

[75] J. Zhang, J. Ge, H. Yoo, L. Fu, Z. Yang, Y. Liu, R. Saravanan, S. Yin, J. Yu, D. Niu, et al. (2026) Playful agentic robot learning. arXiv preprint arXiv:2606.19419. 引用：§1, Table 3.

[76] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023) Learning fine-grained bimanual manipulation with low-cost hardware. 发表于 Robotics: Science and Systems (RSS). 引用：§4.

[77] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan (2024) 3D-vla: 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631. 引用：§4.

[78] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, et al. (2025) X-VLA: soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274. 引用：Table 3.

[79] Y. Zhou et al. (2025) LIBERO-Pro: towards realistic robotic manipulation benchmarks via systematic perturbations. arXiv preprint arXiv:2510.03827. 引用：§C.2, §3.1.

## 附录A. 文件介导的REPL协议

第2.2节的外挂框架将第2.1节的执行循环实现为同步的文件介导Read-Eval-Print Loop（REPL）。一个长时间运行的环境工作器拥有实时模拟器状态，而规划器Π仅通过序列化的原语调用和持久化观测与其交互。规划器不访问特权模拟器状态、物体位姿或控制器内部。

在回合t，规划器读取当前观测o_t、任务语言ℓ以及从任务特定记忆和全局记忆中检索的上下文。然后它通过将JSON对象写入command.json来发出一个原语调用c_t ∈ 𝒫。该对象在其action字段中包含原语名称和相应的关键字参数。工作器消耗此文件，在实时环境中执行选定的原语，并将下一个索引观测o_{t+1}连同轻量级执行记录写入。规划器等待这些文件后再选择下一个原语。因此，每个物理动作后都有观测和诊断，然后展开继续。

**表7：文件介导REPL使用的文件。** 正文将观测抽象为RGB-D和机器人状态；附录还列出了用于同步和可审计性的诊断记录。

| 文件或产物 | 角色 |
|---|---|
| command.json | 规划器发出的原语调用c_t。 |
| state_NN.json | 步索引的任务语言、机器人本体感受和基准成功信号。 |
| RGB-D / world-map 文件 | 基准特定的感知证据，用于语义识别和度量重接地。 |
| log_NN.json | 诊断记录，包含接受的命令、原语状态、步数计数和可用的失败信息。 |
| done_NN.flag 或 terminal 文件 | 指示工作器已完成当前原语的同步信号。 |
| 任务特定记忆轨迹 | 一个任务的可追加JSONL程序记忆。每行是一个原语命令。 |
| 任务特定记忆摘要 | JSON语义记忆，总结结果、策略、恢复决策和失败模式。 |
| 全局记忆 | 使用原语库的跨任务成功规则和失败模型。 |

索引NN单调递增。初始观测在NN=00写入；每个执行的原语产生下一个索引状态、感知文件和诊断日志。这些记录使展开可审计，而不向规划器暴露预言物体坐标。

**任务特定记忆。** 任务特定记忆存储已解决任务实例的可复用结构。它包含一个程序JSONL轨迹和一个语义JSON摘要。轨迹记录发出了哪些原语调用；摘要记录策略为何有效以及应避免什么。一个简化的摘要是：

```json
{"task":"put the black bowl on the wooden tray",
"success":true,
"trace_file":"task_specific_memory_put_black_bowl_on_tray_s0.jsonl",
"strategy":"use VLA for grasping, then analytic transport and release",
"avoid":["do not reuse reference xyz values",
"verify placement with the benchmark success signal"]}
```

配对的程序轨迹存储原语顺序：

```json
{"action":"vla_act","prompt":"grasp the black bowl","max_chunks":2}
{"action":"move_to","xyz":[0.12,-0.08,0.92],"gripper":null}
{"action":"release"}
```

轨迹是一个任务级解框架，而非开环轨迹。它记录解析和VLA支持原语的顺序、VLA调用的位置以及接触丰富执行、传输、释放和验证之间的过渡点。存储轨迹中的空间参数被视为参考场景绑定。在部署时，规划器复用记忆结构但从当前观测重接地物体、夹具、支撑面和目标位姿。

**全局记忆。** 全局记忆存储原语库的任务无关操作知识。一个紧凑的示例是：

成功规则：
对接触丰富阶段（如不规则抓取或夹具交互）使用VLA原语。稳定抓取后，优先使用解析运动进行长距离传输和精确定位。

失败模型：
如果夹爪关闭但物体未随末端执行器移动，将此次尝试视为空抓取。重定位物体并在重试前重阶段化。

失败模型：
不要仅凭视觉近距离就终止。检查基准成功信号和最新执行记录。

**迭代记忆构建。** 记忆在交互过程中构建，而非仅在展开后写入。每次原语之后，规划器读取新观测和诊断记录，然后将结果分类为进展、可恢复失败或不可恢复失败。成功的展开存储为任务特定记忆。可恢复的失败保留在轨迹中并在语义摘要中解释，以便后续步骤记录修正。失败的尝试也作为负面证据保留，并可能向全局记忆贡献失败模型。

跨尝试，记忆被精炼而非简单累积。后续尝试如果产生更短或更可靠的方案，可以替换程序轨迹，而早期的失败观测作为未来规划的约束仍然有用。这种分离让Harness VLA迁移任务应该如何解决，而不是重放物体恰好在参考场景中的位置。

## 附录B. 原语词汇表与环境特定扩展

本附录扩展第2.3节的原语词汇表。我们在所有基准中使用相同的原语名称。跨环境的差异仅通过可用性、手臂绑定和实现后端来表达；它们不被视为新的原语名称，除非形态暴露了新的自由度。

**表8：三个基准形态中的原语可用性。** 探索性重置工具支持引导，不计为操控原语。

| 原语 | LIBERO | RoboCasa365 | RoboTwin C2R |
|---|---|---|---|
| move_to | 是 | 是 | 是 |
| move_pose | 是 | 通过组合 | – |
| rotate_wrist | 是 | – | 是 |
| rotate_pitch | 是 | 是 | – |
| set_gripper | 是 | 是 | 是 |
| release | 是 | 是 | 是 |
| vla_act | 是 | 是 | 是 |
| navigate_to | – | 是 | – |
| move_base | – | 是 | – |
| arm binding | – | – | 是 |

**通用解析原语。** move_to是共享的末端执行器传输原语：它接收世界坐标系笛卡尔目标，并委托给当前环境可用的求解器。内部后端可以是操作空间伺服、基于雅可比的控制或IK规划器，但暴露的原语语义相同。move_pose通过将位置与姿态分量（如俯仰角）共同变化来扩展move_to；当环境不直接暴露它时，相同的行为表示为rotate_pitch和move_to的简短组合。rotate_wrist和rotate_pitch在保持当前空间位置的同时施加偏航和俯仰设定点。set_gripper将夹爪驱动到打开或关闭状态，release是带有释放后置条件的相应打开夹爪原语。环境特定的夹爪约定隐藏在原语接口之后。

**移动基座和双臂细节。** RoboCasa365添加两个移动基座原语，因为厨房规模任务需要在固定手臂工作空间外进行阶段化。navigate_to是一个组合原语，将基座驱动到世界坐标系平面目标，而move_base是一个原子原语，施加局部基座速度设定点用于精细重定位。RoboTwin C2R不添加新的操控原语名称；而是每个原语可以通过arm参数绑定到左臂、右臂或双臂任务模式。因此交接风格的任务表示为在此双臂绑定下的vla_act、解析传输和释放的组合，而非单独的原语。

**vla_act。** vla_act是词汇表中唯一的学习原语。它绑定到当前基准使用的冻结VLA，并根据提示和实时观测执行以动作块为条件的控制。规划器配置一个停止谓词τ，可以对应提升抓取条件、接触状态条件、基准谓词或块预算。因此同一原语涵盖抓取、放置、夹具驱动、插入和双臂接触，同时保留规划器对语义接地、空间重绑定、导航和重阶段化的责任。

**JSON调用示例。** 所有原语共享紧凑的JSON命令格式。代表性调用为：

```json
{"action": "move_to", "xyz": [-0.101, 0.202, 1.05],
"arm": "auto", "gripper": "open", "tol": 0.012, "max_steps": 80}
{"action": "navigate_to", "xy": [1.20, -0.35], "tol": 0.05}
{"action": "move_base", "forward": 0.10, "lateral": 0.00,
"turn": -0.15, "steps": 12}
{"action": "vla_act", "prompt": "grasp the black bowl",
"arm": "auto", "max_chunks": 30, "stop": "object_lifted"}
```

确切的数值容差和停止谓词是基准特定的，但规划器始终通过这些原语名称进行交互。

## 附录C. 评估基准详情

本附录记录了我们评估中使用的基准组成、任务划分、展开协议和成功标准。

> **图8：** 评估中使用的四个基准家族的代表性环境概览。每个基准捕捉一个不同的操控设置：结构化桌面操控（LIBERO）、分布偏移下的鲁棒性（LIBERO-Pro）、长时序厨房操控（RoboCasa365）以及清洁到随机设置下的双臂操控（RoboTwin C2R）。

我们在四个基准家族上评估：LIBERO、LIBERO-Pro、RoboCasa365和RoboTwin C2R。LIBERO、LIBERO-Pro和RoboCasa365使用少样本协议，其中每个任务的种子s₀（种子0）仅作为任务特定记忆构建的探索参考种子。在此种子上，智能体搜索成功的原语序列，并将产生的审计摘要和JSONL命令轨迹存储为任务特定记忆。种子s₀不计入报告的评估。报告的评估展开在保留种子上运行，这些种子在新的初始状态下检索并重接地相应的任务特定记忆。RoboTwin C2R使用单独的零样本清洁到随机化协议。

跨所有基准，任务成功由基准提供的二元完成谓词确定。如果在回合视野或最大步数预算耗尽前满足任务完成谓词，则展开计为成功。原语级后置条件（如vla_act或release的返回条件）仅决定单个原语何时将控制权返回给规划器；它们不被用作最终任务成功谓词的替代。

### C.1 LIBERO评估基准

LIBERO [38]是一个语言条件操控基准，组织为多个任务套件。我们在四个标准套件上评估：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal和LIBERO-10。LIBERO-Spatial包含变化空间关系的任务，LIBERO-Object变化目标物体身份，LIBERO-Goal在相关场景下变化目标谓词，LIBERO-10包含更长时序的组合操控任务。

每个套件包含10个语言条件任务。对于每个任务，种子s₀（种子0）仅用于探索任务并构建任务特定记忆。报告的评估使用10个保留种子，记为s₁–s₁₀，这些种子检索此任务特定记忆并在新初始状态下将其接地。因此，每个LIBERO套件包含10个任务×10个评估种子=100次报告展开，四个套件总共包含400次报告展开。

**表9：LIBERO评估协议。**

| 套件 | 任务数 | 每任务评估种子数 | 报告展开数 |
|---|---|---|---|
| LIBERO-Spatial | 10 | 10 | 100 |
| LIBERO-Object | 10 | 10 | 100 |
| LIBERO-Goal | 10 | 10 | 100 |
| LIBERO-10 | 10 | 10 | 100 |
| 总计 | 40 | – | 400 |

成功率使用本附录开头定义的基于谓词的规则计算。

### C.2 LIBERO-Pro评估基准

LIBERO-Pro [79]通过受控扰动扩展LIBERO任务家族。我们评估四个任务家族：Spatial、Object、Goal和LIBERO-10。每个任务家族在两种扰动设置下评估，记为T和S。T指任务或指令重定向设置，其中指令被重定向到另一个有效目标物体或目标条件。S指交换或位置交换设置，其中物体初始位置被交换或重新排列，而指令保持不变。

我们评估八个LIBERO-Pro单元格：Spatial-T、Spatial-S、Object-T、Object-S、Goal-T、Goal-S、LIBERO-10-T和LIBERO-10-S。每个单元格包含10个任务。如同LIBERO，种子s₀（种子0）仅用于探索任务并构建任务特定记忆；报告的评估使用种子s₁–s₁₀，这些种子检索存储的任务特定记忆并在新初始状态下将其接地。因此每个单元格包含10个任务×10个评估种子=100次报告展开，总共800次报告展开。

**表10：LIBERO-Pro评估协议。**

| 评估单元格 | 任务数 | 每任务评估种子数 | 报告展开数 |
|---|---|---|---|
| Spatial-T | 10 | 10 | 100 |
| Spatial-S | 10 | 10 | 100 |
| Object-T | 10 | 10 | 100 |
| Object-S | 10 | 10 | 100 |
| Goal-T | 10 | 10 | 100 |
| Goal-S | 10 | 10 | 100 |
| LIBERO-10-T | 10 | 10 | 100 |
| LIBERO-10-S | 10 | 10 | 100 |
| 总计 | 80 | – | 800 |

成功率使用本附录开头定义的基于谓词的规则计算。

### C.3 RoboCasa评估基准

RoboCasa365将评估扩展到厨房家庭操控。我们使用RoboCasa365 target50划分，由三个任务组组成：Atomic-Seen、Composite-Seen和Composite-Unseen。Atomic-Seen包含18个原子任务，对应短时序厨房操作。Composite-Seen包含16个组合任务，其模板也存在于预训练集中。Composite-Unseen包含16个组合任务，其模板从预训练中保留，仅出现在目标评估中。这里，"seen"和"unseen"指的是任务模板是否出现在预训练集中，而非具体回合、轨迹或场景是否被观察到。

RoboCasa365使用划分特定的少样本种子协议。在每个划分中，种子s₀（种子0）仅用于探索任务并构建任务特定记忆。报告的评估使用保留种子：Atomic-Seen使用种子s₁–s₁₀，而Composite-Seen和Composite-Unseen使用种子s₁–s₅。这些保留种子检索并重接地相应的任务特定记忆。因此，Atomic-Seen包含18×10=180次报告展开，Composite-Seen包含16×5=80次报告展开，Composite-Unseen包含16×5=80次报告展开。RoboCasa365 target50评估在此划分特定少样本协议下包含340次报告展开。

**表11：RoboCasa365评估协议。**

| 划分 | 任务数 | 每任务评估种子数 | 报告展开数 |
|---|---|---|---|
| Atomic-Seen | 18 | 10 | 180 |
| Composite-Seen | 16 | 5 | 80 |
| Composite-Unseen | 16 | 5 | 80 |
| 总计 | 50 | – | 340 |

成功率使用本附录开头定义的基于谓词的规则计算。

### C.4 RoboTwin清洁到随机化评估基准

RoboTwin C2R是一个双臂操控基准，包含50个任务。任务集涵盖拾取放置、堆叠、排序、交接、双臂传输、关节物体交互、按压和点击、旋转、扫描和容器放置行为。

RoboTwin C2R使用独立的清洁到随机化协议。对于每个任务，任务特定记忆轨迹从demo_clean设置中的一个官方脚本专家验证种子获得。评估然后直接在官方demo_randomized设置中在五个脚本专家验证的随机化种子上进行。专家验证步骤仅用于确保采样的任务实例在官方任务定义下可行；它独立于我们的方法，不使用Harness VLA展开进行种子选择。在随机化设置中不执行额外的轨迹搜索、微调或任务级适配。此协议评估从清洁设置轨迹到随机化任务实例的零样本迁移。

我们评估所有50个RoboTwin C2R任务。每个任务在demo_randomized设置中的5个种子上评估，产生50×5=250次报告展开。

**表12：RoboTwin清洁到随机化评估协议。**

| 协议 | 任务数 | 每任务评估种子数 | 报告展开数 |
|---|---|---|---|
| RoboTwin C2R | 50 | 5 | 250 |

成功率使用本附录开头定义的基于谓词的规则计算，完成谓词由官方RoboTwin C2R任务特定评估器提供。

### C.5 基准摘要

表13总结了每个评估基准的规模。LIBERO和LIBERO-Pro使用种子s₀（种子0）仅用于构建任务特定记忆，并在每个任务10个保留种子s₁–s₁₀上报告评估。RoboCasa365使用相同的参考种子约定，采用划分特定的评估种子：Atomic-Seen使用s₁–s₁₀，两个组合划分使用s₁–s₅。RoboTwin C2R使用清洁到随机化评估，其中任务特定记忆从一个专家验证的demo_clean种子获得，并在专家验证的demo_randomized种子上评估。

**表13：评估基准摘要。**

| 基准 | 任务数 | 每任务试验数 | 报告展开数 |
|---|---|---|---|
| LIBERO | 40 | 10 | 400 |
| LIBERO-Pro | 80 | 10 | 800 |
| RoboCasa365 | 50 | 10/5/5 | 340 |
| RoboTwin C2R | 50 | 5 | 250 |

## 附录D. VLA模型实例化

我们跨基准实例化不同的视觉-语言-动作模型，在Harness VLA框架内统一抽象为单一的接触丰富原语vla_act。

### D.1 π_RLinf：RLinf发布的LIBERO检查点

对于LIBERO和LIBERO-Pro，我们使用RLinf发布的pi05_libero130_fullshot检查点，记为π_RLinf，作为冻结的视觉-语言-动作策略。它基于π₀.₅架构，我们直接采用此官方π₀.₅-SFT检查点作为Harness VLA框架内的冻结vla_act接触丰富执行原语。

**架构。** π_RLinf遵循π₀.₅视觉-语言-动作架构，将多模态输入（包括视觉观测I_t、语言指令ℓ和机器人状态q_t）编码为统一的transformer表示。模型从预训练的视觉-语言骨干初始化，并通过监督学习对齐到机器人动作空间。

与π₀.₅一致，π_RLinf支持分层推理，其中高级语义子任务预测和低级动作生成在单一策略内联合建模。

**动作建模。** π_RLinf采用π₀.₅引入的两阶段推理范式。给定观测和语言指令，模型首先预测高级子任务ℓ̂（例如，"pick up the plate"），然后用于条件低级动作生成。

低级策略产生连续动作块a_{t:t+H}，通过FAST分词或基于流的连续建模表示，实现稳定的接触丰富操控。

**训练。** 模型遵循π₀.₅训练协议在LIBERO-130数据集上进行监督微调，产生官方π₀.₅-SFT检查点。在本文中，不进行额外训练或适配，模型在评估期间以完全冻结的方式使用。

**性能。** 在LIBERO基准上，π_RLinf达到95.3%的成功率，展示了强大的分布内操控能力。在引入指令扰动和组合变体的LIBERO-Pro上，性能下降到50.0%，表明对分布偏移的敏感性。

**本文中的角色。** 在本文中，π_RLinf用作Harness VLA框架内的冻结低级执行模块，作为LIBERO和LIBERO-Pro任务的接触丰富操控原语。

### D.2 RLDX-1

RLDX-1用于RoboCasa365的厨房操控任务。它是一个大规模视觉-语言-动作（VLA）基础模型，设计用于跨多种机器人形态的通用灵巧操控。在本文中，我们直接使用官方RLDX-1检查点，并在评估期间保持模型完全冻结，将其视为Harness VLA框架内的vla_act接触丰富执行原语。

**架构。** RLDX-1采用多流动作Transformer（MSAT）作为其核心动作建模架构。系统首先使用基于Qwen3-VL 8B的视觉-语言模型（VLM）编码多帧视频观测和语言指令，并通过认知token提取动作相关表示。

进一步引入记忆模块以聚合历史认知特征，产生历史感知表示。动作模型建立在MSAT之上，解耦认知流和动作流，并在物理信号可用时可选引入物理流。这些流通过跨流自注意力联合建模，实现视觉、语言、状态和物理信号的统一处理。

**动作建模。** RLDX-1使用流匹配扩散transformer进行连续动作预测训练。模型学习一个速度场，将含噪动作轨迹映射为干净动作序列，并通过迭代去噪生成未来动作。

推理期间，模型以块方式产生动作块，并顺序执行部分块以实现稳定的闭环控制。模型在物理信号可用时还联合建模物理信号，提高接触丰富操控能力。

**训练。** 本文直接使用官方RLDX-1检查点，并在评估期间保持所有参数冻结，不进行任何额外训练或微调。模型在其原始流程中已完成多阶段训练，在此用作统一执行策略。

**性能。** 在RoboCasa365基准上，RLDX-1在Atomic-Seen任务上达到60.0%，在Composite-Seen任务上达到21.3%，在Composite-Unseen任务上达到5.0%，总体加权成功率为30.0%。这些结果显示在原子接触丰富操控任务上表现强劲，而在组合和分布外设置上性能显著下降。

**本文中的角色。** 在本文中，RLDX-1用作Harness VLA框架内的冻结低级执行模块，作为RoboCasa365厨房操控的接触丰富操控原语。

### D.3 LingBot-VLA

LingBot-VLA [70]是我们用于双臂操控的RoboTwin后端的视觉-语言-动作模型。它是一个大规模VLA基础模型，设计用于跨多种真实世界形态的连续机器人控制。在本文中，我们使用RoboTwin后训练的LingBot-VLA检查点作为Harness VLA框架内的冻结低级执行模块。

**架构。** 模型建立在预训练的Qwen2.5-VL视觉-语言骨干之上，并扩展了混合Transformer（MoT）架构，将视觉-语言推理和动作生成分离到专用transformer通路中。这些通路通过共享自注意力耦合，实现统一的多模态序列建模同时减轻跨模态干扰。引入动作专家模块以根据多模态嵌入预测连续控制信号。

**动作建模。** LingBot-VLA采用流匹配公式进行连续动作预测。为提高长时序操控中的时序一致性，采用分块动作解码，其中固定长度的动作序列在单次前向传播中自回归预测。块大小设置为T=50，实现稳定且时序连贯的控制。

**训练。** 模型首先在跨9种机器人形态收集的大规模真实世界双臂遥操作数据上预训练，提供广泛的跨形态泛化。然后通过RoboTwin操控轨迹上的监督微调（SFT）进一步适配，专门针对双臂操控任务。此后训练阶段后，检查点在所有直接VLA和Harness VLA评估中保持冻结。

**训练配置。** 我们在表14中总结了后训练的关键超参数。这些参数对应于本文报告的所有LingBot-VLA后训练实验所使用的配置。

**表14：LingBot-VLA在RoboTwin上的后训练配置。**

| 类别 | 配置 |
|---|---|
| **优化** | |
| 优化器 | AdamW |
| 学习率 | 1×10⁻⁴ |
| 视觉编码器学习率 | 1×10⁻⁶ |
| 权重衰减 | 0 |
| 损失函数 | L1流匹配 (L1_FM) |
| **序列建模** | |
| 块大小 | 50 |
| 最大序列长度 | 2048 |
| 流步数 | 10 |
| 最大动作维度 | 75 |
| 最大状态维度 | 75 |
| **训练设置** | |
| 全局批量大小 | 256 |
| 图像分辨率 | 224×224 |
| 相机视图 | top + wrist left + wrist right |
| **系统** | |
| 精度 | 混合精度 (bf16/fp32) |
| 分布式训练 | FSDP2 |

**性能。** 在RoboTwin随机化评估设置下，LingBot-VLA在直接冻结智能体配置（即作为独立策略，无智能体级分解或外部规划）中达到50.4%的成功率。此直接基线与主表中的外部π₀.₅比较不同：LingBot-VLA是Harness VLA使用的RoboTwin专用冻结VLA后端，而π₀.₅是代表性外部VLA基线。结果表明，LingBot-VLA在智能体级分解之前已经提供了强大且稳定的接触丰富操控能力。

**本文中的角色。** 在本文中，LingBot-VLA用作Harness VLA内的冻结执行模块，作为RoboTwin双臂控制的低级接触丰富操控原语。

### D.4 统一抽象

跨所有基准，异构的视觉-语言-动作模型被统一抽象为可互换的接触丰富执行原语。LLM规划器负责语义接地、空间分解和长时序任务规划，而每个VLA仅在当前观测条件下为局部交互执行而被调用。

## 附录E. 智能体提示规范

本附录指定Harness VLA中LLM规划器使用的任务提示。提示不仅是自然语言任务指令。它是每次展开前给予智能体的操作手册：它定义文件介导的交互协议、可用于感知的观测文件、允许的原语词汇表、冻结VLA原语的接口、任务特定记忆的使用以及为可重复性必须写入的输出产物。

所有基准提示遵循共享核心设计。一个单一的基准无关的提示模板定义智能体的职责，每个基准实例化对应于成功谓词、机器人形态、相机文件、原语模式、VLA后端、任务特定记忆路径和已知恢复规则的槽位。这种共享结构很重要，因为论文中的实证比较评估了相同的智能体外挂框架跨LIBERO/LIBERO-Pro、RoboCasa365和RoboTwin C2R，而非为每个环境手工制作无关的控制器。

### E.1 共享提示核心

共享提示以第二人称书写，因为它是直接针对智能体的。其第一段定义智能体角色：

```
你是{BENCHMARK}的LLM-in-the-loop混合操控智能体。
基准驱动程序已在运行并等待你的命令。
你的工作是通过读取任务状态、从感知中定位物体、
选择并执行可用原语、在需要接触丰富行为时调用VLA、
并写入可重复的审计来完成此任务。
```

共享提示的其余部分组织为表15总结的模块。每个模块存在于所有基准提示中，而基准特定提示填充具体字段，如state.libero_terminated、state.success、eval_success、相机名称和原语模式。

**表15：智能体任务提示中的共享模块。** 每个基准特定提示保持此结构并填充环境特定细节。

| 提示模块 | 给予智能体的信息 |
|---|---|
| 角色和成功信号 | 闭环控制；优化基准谓词，而非视觉猜测。 |
| 感知隔离 | 无真值位姿或模拟器内部；从RGB-D和世界地图定位。 |
| 基于文件的REPL | 写入一个JSON命令，等待执行，读取刷新的产物，然后迭代。 |
| 原语词汇表 | 允许的原语模式和控制语义，包括夹爪、手臂和步数约定。 |
| VLA分工 | VLA用于接触丰富阶段；解析原语用于接地、阶段化、传输、释放和恢复。 |
| 任务语言 | 状态文件的任务语言是权威的；不要从文件名或索引推断任务。 |
| 种子0任务特定记忆 | JSON审计用于策略和失败模式；JSONL轨迹用于原语执行顺序。 |
| 全局记忆 | 可复用的成功规则和失败观测，提供超越种子0任务特定记忆的额外上下文。 |
| 闭环恢复 | 每次原语后验证状态、日志、RGB和几何；诊断后再重试。 |
| 预算和重置策略 | 跟踪预算和重置策略；严格评估中重置被禁用。 |
| 输出规范 | 为成功和失败的展开写入审计和命令轨迹。 |

### E.2 感知和文件介导控制

共享提示使感知隔离显式化，明确禁止访问特权信息（例如真值物体位姿或模拟器内部状态），以强制执行现实的局部观测设置并防止在决策期间依赖预言级环境访问。智能体从状态文件接收物体名称和本体感受，但不接收物体坐标。它必须通过在RGB图像中选择像素并索引相应的预计算世界地图来定位实体，从而将所有空间推理建立在感知输入而非隐藏状态变量上。通用定位指令是：

```
1. 从RGB中识别相关物体、夹具、目标表面或关系地标。
2. 选择该实体可见表面上的像素。
3. 在这些像素处索引匹配的预计算世界地图。
4. 采样多个稳定像素并使用鲁棒统计量，通常是中位数。
5. 避免边缘、物体边界、桌面间隙、孔洞、反射和背景像素。
6. 每当机器人、相机、物体、基座、夹具或抓取状态改变时重新定位。
```

此感知规则与论文通篇使用的相同REPL风格执行契约配对：

```
1. 将一个JSON命令写入{WORKDIR}/command.json。
2. 等待驱动程序完成该原语，通常通过done_NN.flag、
   log_NN.json或基准特定的终端文件。
3. 读取新的state_NN.json、log_NN.json、图像、深度图和世界地图。
4. 根据新证据决定下一个命令。
```

因此，提示强制执行与框架描述中使用的相同的闭环行为：每个原语调用都被视为一个实验，在发出下一个命令之前必须观察其结果。

### E.3 种子0任务特定记忆

提示中最重要的记忆相关部分是使用种子0任务特定记忆的指令。任务特定记忆不是要重放的普通演示。它是一个结构化记忆对象，将语义策略与具体原语执行分离。跨基准，提示告诉智能体查找两个互补文件：

```
{TASK}_s0.json（任务特定记忆审计JSON）
{TASK}_s0.jsonl（任务特定记忆命令JSONL）
```

JSON文件是种子0展开的审计和策略摘要。它记录参考运行的结果，并提供关于解策略、有用的原语选择、恢复决策和探索期间观察到的失败模式的高级注释。它不作为动作轨迹重放；而是智能体在查阅JSONL命令轨迹以获取具体原语顺序之前，用它来解释参考方案。根据基准，此JSON包括展开结果、成功状态、命令或步数计数、策略注释、失败观察以及最终状态摘要。

JSONL文件是可执行的命令轨迹。每行存储智能体在参考展开期间发出的一个JSON原语。智能体读取此轨迹以恢复方案的程序结构：原语调用的顺序、解析与VLA支持动作的选择、VLA调用的数量和位置、以及感知、阶段化、接触丰富执行、传输、释放和验证之间的过渡点。轨迹被用作结构先验而非要重放的轨迹；所有空间参数在执行前从当前观测重接地。

提示给予智能体以下规则：

```
使用JSON审计理解策略为何有效以及应避免什么。
使用JSONL轨迹理解执行了什么以及按什么顺序。
复用任务特定记忆程序结构，但绝不重放字面坐标。
先前的xyz、xy、四元数、像素位置、基座位姿和夹具坐标属于种子0场景。
从当前图像和世界地图重新定位每个当前物体、目的地、
支撑面、关系地标和夹具。
```

此规则是第2.2节中任务特定记忆的提示级实现。它让规划器迁移成功方案的结构，同时在当前展开中接地所有几何。

### E.4 全局记忆

全局记忆补充种子0任务特定记忆。任务特定记忆是任务特定的程序上下文：它记录一个参考展开的JSON审计和JSONL命令轨迹。全局记忆是任务无关的。它存储固定原语库的可复用成功规则和失败模型，包括已知的VLA操作条件、空抓取失败、虚假视觉成功、不稳定阶段化和恢复模式。它在闭环执行期间用作上下文指导，而非作为要重放的动作轨迹。

```
使用全局记忆检查：
1. VLA和解析原语的已知成功规则；
2. 重复或修复动作前的已知失败模型；
3. 空抓取、错误物体尝试、虚假视觉成功和不稳定阶段化；
```

### E.5 基准特定提示实例化

表16和17总结了共享提示如何为每个基准实例化。我们将实例化分为接口级字段和环境上下文字段。前者指定成功谓词、冻结VLA入口点和所需的审计产物；后者记录专门针对每个基准共享提示的形态和控制假设。

**表16：跨基准的接口级提示实例化。** 在正文中，异构的VLA原语名称被抽象为统一的vla_act接口。

| 基准 | 成功信号 | VLA接口 | 输出产物 |
|---|---|---|---|
| LIBERO / LIBERO-Pro | libero_terminated | vla_act | 命令JSONL；审计JSON |
| RoboCasa365 | success | vla_act | 命令JSONL；审计JSON |
| RoboTwin C2R | eval_success | vla_act | 命令JSONL；审计JSON |

**表17：基准特定提示提供的环境上下文。**

| 基准 | 环境特定的提示专业化 |
|---|---|
| LIBERO / LIBERO-Pro | 单臂桌面操控；固定agentview和移动手腕RGB-D/世界地图观测；接触丰富步骤后的智能体端传输、释放和视觉验证。 |
| RoboCasa365 | 移动厨房操控；基座运动可用于接触远处夹具；智能体在基座移动后重定位；夹具面向的阶段化以及对已封顶但有进展的接触尝试的延续被视为操作策略的一部分。 |
| RoboTwin C2R | 双臂操控；手动运动命令绑定到指定手臂；头部和左/右手腕观测提供感知；手动原语支持观测刷新、非抓取运动、释放、终止和接触丰富尝试周围的恢复。 |

**LIBERO / LIBERO-Pro。** LIBERO家族提示由标准LIBERO和LIBERO-Pro共享：

```
你是LIBERO PRO/LIBERO基准的LLM-in-the-loop混合驱动程序。
```

此实例化将共享提示专门用于单臂桌面操控。它定义了用于基于感知的接地的LIBERO观测文件，包括固定agentview RGB-D/世界地图文件和用于近距离重定位的移动手腕相机文件。统一的vla_act接口用于接触丰富步骤，包括抓取和闭环关节物体、按钮或旋钮操控。接触建立后，智能体仍负责目标识别、场景重定位、自由空间传输、释放和进展验证。

**RoboCasa365。** RoboCasa提示为移动厨房操控实例化共享结构：

```
你是RoboCasa365厨房基准的LLM-in-the-loop混合驱动程序。
```

此实例化向共享操控循环添加移动基座阶段化。智能体从RGB-D/世界地图观测中接地物体和夹具，使用navigate_to进行粗略基座放置，并使用move_base进行小的局部修正。由于基座运动会改变机器人视点和相对手臂工作空间，提示强调在导航后重定位，然后再继续操控。

统一的vla_act接口提供接触丰富原语。VLA指令是完整任务语言，规划器决定是在局部阶段化后还是在更广泛的全身交互期间调用它。被封顶但仍取得进展的VLA调用被视为延续情况而非立即失败，因此相同的VLA调用可以被继续而非被手动命令中断。

**RoboTwin C2R。** RoboTwin提示为双臂操控实例化共享核心：

```
你是RoboTwin基准的LLM-in-the-loop混合操控智能体。
```

此实例化将提示专门用于双臂设置。手动运动命令包括显式左/右臂绑定，观测流包括头部和左/右手腕视图以及相应的深度和世界地图文件。驱动程序通过eval_success报告官方RoboTwin成功信号，并在展开退出时写入终端final.json。

接触丰富接口暴露为vla_act。此原语用于抓取形成、重新抓取、交接抓取、双臂抓取形成和其他接触丰富阶段。手动原语在这些VLA尝试周围使用，用于观测刷新、非抓取运动、释放、终止和恢复。RoboTwin额外要求一个诊断Markdown文件用于事后分析。

### E.6 紧凑提示骨架

为完整起见，以下列表显示了所有四个提示文件底层的紧凑共享骨架。基准特定提示用上述具体值填充括号槽位。

```
你是{BENCHMARK}的LLM-in-the-loop混合操控智能体。
基准驱动程序已在{WORKDIR}中运行。通过读取状态和感知文件、
从RGB和世界地图定位任务实体、仅调用允许的原语、
使用冻结VLA进行接触丰富行为并写入可重复审计来完成任务。

1. 角色和成功信号
你是闭环控制器。优化{SUCCESS_SIGNAL}，而非视觉猜测。
持续直到成功、预算耗尽或不可恢复。

2. 感知隔离
不要查询模拟器物体位姿或隐藏任务初始化。使用RGB进行语义
识别，使用深度/世界地图进行度量定位。
在每次物体、相机、机器人、基座或抓取改变后重新定位。

3. 基于文件的REPL
将一个JSON命令写入{WORKDIR}/command.json。等待驱动程序结果。
读取state_NN.json、log_NN.json、图像、深度图和世界地图。
然后决定下一个命令。

4. 原语词汇表
仅使用{PRIMITIVE_SCHEMAS}。保持确切的语法和控制语义，
包括夹爪符号、手臂绑定、块预算和步数成本。

5. 你与VLA的分工
使用{VLA_PRIMITIVE}进行抓取、重新抓取、关节接触、
插入、按压、就位和其他接触丰富阶段。使用解析原语进行接地、
阶段化、自由空间传输、释放、验证和恢复。

6. 任务语言
从状态文件读取task_language。它是权威的。不要从文件名、
物体列表、任务索引或相邻任务特定记忆文件推断任务。

7. 种子0任务特定记忆和全局记忆
读取任务匹配的种子0任务特定记忆审计JSON以理解为何任务特定
策略有效以及什么失败了。读取种子0任务特定记忆JSONL以恢复
执行了什么以及按什么顺序。读取全局记忆以获取跨任务成功规则
和失败模型。使用任务特定记忆作为任务特定程序骨架，但使用
全局记忆和当前感知来决定何时重接地、验证、恢复或停止。
绝不重放字面坐标。

8. 闭环验证和恢复
在每次命令后，检查状态、日志、RGB和世界地图。在再次行动前
诊断错误物体选择、不良站位、VLA失误、短放置、隐藏谓词失败
或不可恢复的位移。

9. 预算、重置和终止
跟踪基准预算和重置策略。在严格评估中不要重置。
仅在成功、预算耗尽或不可恢复时停止。

10. 输出规范
写入所需的审计JSON和命令轨迹JSONL。审计JSON记录基准成功状态
和用于评估和成功率计算的最终结果字段。JSONL轨迹按顺序记录
智能体执行的原语命令，以便检查和分智能体的决策过程。

11. 操作循环
读取提示、状态、任务语言、感知、任务特定记忆和全局记忆。
定位实体。执行一个原语。观察。恢复。重复。写入输出。
```

这种结构化的提示设计将通用的Harness VLA操作协议与基准特定的假设分开，为在额外的操控基准中构建智能体提示提供了可复用的模板。

## 附录F. 原语使用统计

表18聚合了Harness VLA (CC)在LIBERO Pro家族、RoboTwin C2R和RoboCasa365运行中发出的原语调用。我们遵循表8中的分类法和可用性摘要报告规范原语名称：所有后端特定的VLA调用合并到统一的vla_act原语，实现级运动宏折叠到其暴露的解析原语中，渲染、重置、注释和no-ops等非操控辅助工具被排除。百分比在每个环境的操控原语调用总数内计算。

使用模式支持预期的非对称分解。在LIBERO中，解析原语占主导：move_to单独占调用的61.8%，而vla_act占15.8%。这匹配任务的桌面结构：VLA主要用于建立接触丰富的抓取或夹具交互，之后解析传输、夹爪控制和释放完成大部分展开。RoboCasa365将混合转向移动阶段化和更长时序交互：navigate_to和move_base合计占调用的19.4%，而vla_act升至35.3%，因为厨房任务需要在更大场景中进行学习到的抓取、夹具驱动和约束放置。RoboTwin C2R具有最高的VLA份额（47.4%），反映了双臂抓取和交接式接触，但解析原语仍然提供略过半数的调用，用于规划的手臂运动、释放和最终排列。

表19在类别级别总结了相同的证据。跨所有三种形态，冻结的VLA不被用作单体端到端控制器；它被作为接触丰富原语在更大的解析支架内调用。确切的比率随形态和任务家族变化，但定性分工保持稳定：解析原语处理可重复的几何和阶段化，而vla_act提供难以脚本化的学习到的局部交互。

**表18：跨基准环境的规范原语使用。** 每个单元格报告该环境内操控原语调用的计数和百分比。破折号表示该原语未在相应环境中暴露。

| 规范原语 | 类型 | LIBERO | RoboTwin C2R | RoboCasa365 |
|---|---|---|---|---|
| move_to | 解析组合 | 6263 (61.8%) | 685 (40.9%) | 3004 (38.7%) |
| move_pose | 解析组合 | 203 (2.0%) | – | – |
| navigate_to | 解析组合 | – | – | 701 (9.0%) |
| rotate_wrist | 解析原子 | 44 (0.4%) | 1 (0.1%) | – |
| rotate_pitch | 解析原子 | 58 (0.6%) | – | 66 (0.8%) |
| set_gripper | 解析原子 | 1137 (11.2%) | 71 (4.2%) | 371 (4.8%) |
| release | 解析原子 | 831 (8.2%) | 124 (7.4%) | 76 (1.0%) |
| move_base | 解析原子 | – | – | 808 (10.4%) |
| vla_act | VLA | 1598 (15.8%) | 794 (47.4%) | 2746 (35.3%) |
| **总计** | | **10134 (100.0%)** | **1675 (100.0%)** | **7772 (100.0%)** |

**表19：按类别分组的原语使用。** 解析原语包括组合目标到达控制器和原子设定点命令。

| 原语类别 | LIBERO | RoboTwin C2R | RoboCasa365 |
|---|---|---|---|
| 解析原语 | 8536 (84.2%) | 881 (52.6%) | 5026 (64.7%) |
| VLA原语, vla_act | 1598 (15.8%) | 794 (47.4%) | 2746 (35.3%) |
