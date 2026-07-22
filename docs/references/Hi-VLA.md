# What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents

**Authors:** Jiaheng Hu, Mohit Shridhar, Caden Lu, Dhruv Shah, Hao-Tien Lewis Chiang, Jie Tan, Annie Xie

**Affiliation:** Google DeepMind

*Work done while interning at Google DeepMind*

> arXiv:2606.10267v1 [cs.RO] 09 Jun 2026 &nbsp;|&nbsp; License: CC BY 4.0

## Abstract

Hierarchical vision-language-action (Hi-VLA) systems have emerged as a promising paradigm for complex robot manipulation, by using high-level VLM planners to decompose tasks into language subgoals executed by low-level VLA controllers. Despite recent empirical progress, there is a lack of unified design principles for these systems: existing Hi-VLA systems differ in how they choose and connect planners, controllers, mechanisms to switch between the two, and how observations and memory are represented in the planner. In this paper, we present a systematic study of Hi-VLA design for robot manipulation. We unify representative Hi-VLA agents under an options-style control framework and benchmark core design choices across short-horizon, long-horizon, and reasoning-intensive tasks. Our analysis distills practical principles for building Hi-VLA systems, showing how model choices and interface mechanisms jointly shape performance. Applying these principles yields a substantially stronger system than either flat VLA control or a naively designed hierarchy, across experiments both in simulation and on a real ALOHA robot. Overall, our results provide a foundation for building more capable, robust, and principled hierarchical VLA agents. More information and video at [jiahenghu.github.io/hi-vla](https://jiahenghu.github.io/hi-vla).

## 1. Introduction

Recent advances in vision-language-action (VLA) models [46, 7, 27, 50, 47, 1, 35, 34] have demonstrated impressive generalization capabilities in solving robotics tasks by directly mapping natural-language commands to robot actions. These models offer a high degree of steerability, enabling robots to execute open-ended and potentially nuanced language prompts, such as "put the red mug on top of the blue plate on the left." However, monolithic VLAs still remain limited in their ability to perform long-horizon, compositional, or abstract reasoning tasks [7, 46], due to two main reasons. First, since these models are primarily trained with short, easy-to-collect trajectory segments, they struggle to generalize to long-horizon tasks and commands. Second, fine-tuning VLMs on action data often catastrophically compromises the VLM's original reasoning and compositional capabilities, preventing them from repurposing learned robot skills towards novel or complex tasks.

Hierarchical VLA systems (Fig. 1) [5, 24, 1, 40, 30, 14] naturally address this challenge by introducing a high-level VLM planner that reasons over the task and scene, proposes language subgoals, and delegates execution to a low-level VLA policy. This biologically inspired division of labor [26] allows the system to combine the semantic and compositional strengths of VLMs with the physical grounding of VLAs. Recent Hi-VLA systems can perform multi-step household tasks [46, 24], adapt across embodiments [44], and perform semantic reasoning [5], suggesting hierarchy can be a powerful paradigm for more capable embodied agents.

Despite these successes, the design principles behind Hi-VLA systems remain under-explored. A Hi-VLA agent is a complex system shaped by many design choices: the choice of VLM planner and the low-level VLA, the observation representation given to the VLM, the criteria for switching control back from the VLA to the VLM, and the memory mechanism. Existing systems instantiate these choices in different ways, making it difficult to determine which components are important, how they interact, and which design choices are most relevant for different task regimes.

> **Figure 1:** Hierarchical VLA systems have the potential to compensate for the deficiencies of the low-level VLA by generating suitable commands, thereby achieving compositional generalization, especially for long-horizon and reasoning tasks. In this paper, we study the key design choices of Hi-VLA systems, towards a better understanding of how and why they impact the overall performance.

In this work, we take a step toward a principled understanding of hierarchical VLA design. We first unify hierarchical VLA agents under a shared control loop inspired by the options framework [43], which allows us to isolate and evaluate major design choices in a controlled manner. Based on this framework, we conduct a systematic empirical study across diverse manipulation tasks both in simulation and in the real world, spanning three task categories: short-horizon tasks that resemble the length of typical VLA training trajectories, long-horizon tasks that require composing multiple short-horizon skills, and reasoning tasks that require interpreting indirect or semantic instructions. This structured evaluation allows us to ask not only whether a design choice improves average success, but also where it matters most: atomic execution, skill composition, or semantic reasoning.

Our results reveal that while a naive Hi-VLA can improve over flat VLA, carefully chosen hierarchical designs yield substantially larger gains, especially on long-horizon and reasoning-intensive tasks. Specifically, strong Hi-VLA performance depends jointly on the model backbones and the interfaces between them: reasoning-enabled VLMs improve high-level decision-making, steerable VLA controllers are essential for reliable subgoal execution, and termination rules, suitable memory mechanisms, and observation representations provide high-leverage mechanisms for connecting planning with control. Together, these findings point to practical design principles for building more capable and robust hierarchical VLA agents, opening new possibilities for modular embodied systems that combine high-level reasoning with reliable low-level control.

## 2. Related Work

While flat VLAs can execute dexterous motion and handle a broad range of tasks, they often struggle with long-horizon and reasoning tasks, due to a lack of coverage of the training data [7] and the catastrophic forgetting during finetuning [15]. The idea of hierarchy [8, 43, 2, 21, 42, 26, 16, 39] offers a natural way to decompose such tasks and enable VLAs to achieve compositional generalization. Hierarchical VLA systems combine the reasoning capabilities of large VLMs with the control fidelity of low-level VLA or skill-conditioned policies. This paradigm has been adopted in many state-of-the-art systems, including G0 [25], Humanoid-VLA [11], RoboOS [44], Hi-Robot [40], Pi-0.5 [24], Gemini Robotics 1.5 [46], HAMSTER [30], and Helix [14]. Despite this rapid empirical progress, however, there remains limited understanding of what components matter most for the effectiveness of hierarchical VLA systems. Questions such as how model capabilities, memory handling, or policy switching mechanisms influence task success remain largely unexplored. This paper takes the first step toward a systematic analysis of hierarchical VLA architectures, with a focus on the system-design layer on top of existing VLM and VLA backbones. We defer detailed discussion of flat VLAs and their evaluations to Appendix D.

## 3. A Unified View of Hierarchical VLAs

The first step in our analysis seeks to encapsulate different hierarchical VLA design choices under a shared framework. In particular, we notice that seemingly different hierarchical systems can all be subsumed by a unified control loop that resembles the options framework [43]. In a hierarchical VLA system formed by a high-level VLM and a low-level VLA, the VLA can be viewed as an intra-option policy $\pi_{\text{VLA}}(a|o,l)$ that maps language instruction $l \in \mathcal{L}$ and observation $o \in \mathcal{O}$ to low-level robot actions $a \in \mathcal{A}$; whereas the high-level VLM can be viewed as an option selection policy $\pi_{\text{VLM}}(l|\boldsymbol{o},I)$ that maps (a set of) observations $\boldsymbol{o}$ and a task instruction $I$ to some language instruction $l$. Additionally, the observation input to the VLM is managed by a memory module $\boldsymbol{o} \leftarrow \text{mem}([o_i]_{i \leq t})$ that processes historical interactions; as well as an observation representation module $o' \leftarrow \phi(o)$ that optionally post-processes the raw image observations. Together, the VLM, VLA, memory module and the observation representation module define a hierarchical visuomotor policy over low-level robot actions:

$$\pi_{\text{HiVLA}}(a \mid [o_i]_{i \leq t}, I) = \int_l \pi_{\text{VLA}}(a \mid o_t, l) \; \pi_{\text{VLM}}(l \mid \boldsymbol{o}, I) \; dl, \quad (1)$$

$$\text{where} \quad \boldsymbol{o} = \text{mem}([\phi(o_i)]_{i \leq t}).$$

In practical systems, $\pi_{\text{VLA}}$ and $\pi_{\text{VLM}}$ typically do not operate at the same frequency. The VLM operates at a much lower frequency due to the high inference cost, where the same language instruction $l$ is kept fixed for multiple steps until a termination condition $\beta(o, t)$ is met. Once the termination condition is met, the VLM generates a new temporally extended language command $l'$.

We present this unified control loop in Alg. 1, which allows us to discuss each component of the hierarchical VLA as a function that can be implemented in different ways. For example, in previous works, the high-level VLM has been implemented with PaliGemma [24] or finetuned Gemini [46], while the termination condition has been implemented as a success detector [46] or as a fixed timer [14]. We base our subsequent experiments on this unified control loop, where we examine the effect of different implementations of each component.

```
Algorithm 1: Unified Hi-VLA Control Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Task instruction I

Initialize: t ← 0, env ε, observation history 𝐨 ← ∅
Functions: Termination β(·), Memory mem(·),
           Obs. Rep. φ(·), Policies π_vlm and π_vla

while not ε.done do
    // High-level VLM Execution Loop
    o_t ← ε.observe()
    o_t' ← φ(o_t)                    // Process image
    𝐨 ← mem([o_i']_{i≤t})            // Update memory
    l ∼ π_vlm(l | 𝐨, I)              // VLM inference

    while not β(o_t, t) do
        // Low-level VLA Execution Loop
        a ∼ π_vla(a | o_t, l)        // VLA inference
        o_t ← ε.step(a)              // Env Interaction
        t ← t + 1
```

## 4. A Systematic Study of Hierarchical VLA Agents

### 4.1 Evaluation Setup

We conduct our main experiments in the MuJoCo ALOHA suite: a simulated table-top manipulation benchmark with demonstrated real-to-sim transferability [46]. Additionally, we run experiments on a real ALOHA robot. We categorize our evaluation tasks into three categories, namely short-horizon, long-horizon, and reasoning tasks. We discuss our detailed experimental setup in Sec. C.

> **Figure 2:** An overview of our experimental results. In these plots, we visualize how different design choices increase (red) or decrease (blue) the overall performance of hierarchical VLA systems on different types of tasks. We show the detailed results in Appendix J.

In the following sections, we present the results and analysis for a comprehensive set of controlled experiments designed to systematically evaluate the effect of different hierarchical VLA components presented in Alg. 1, including the high-level VLM policy (Sec. 4.2), the low-level VLA policy (Sec. 4.3), the termination condition (Sec. 4.4), the observation representation (Sec. 4.5), and the memory system (Sec. 4.6). Then, we aggregated the best design choices from each component, and compared the result with flat VLA as well as a naive hierarchical VLA (Sec. 4.7).

### 4.2 Effect of High-level VLM Policy

The high-level VLM policy is responsible for aggregating information and generating language commands for the low-level VLA to execute (prompt in Sec. E). Our first set of experiments evaluates the impact of the VLM on the eventual performance[^1].

[^1]: Note that our goal is not to find the single optimal VLM for building hierarchical systems. Rather, we want to understand how features of the VLM impact the overall system.

To be able to change the model size and reasoning ability while keeping everything else fixed, we stick with a family of Gemini models which gives us a good basis for comparison. Specifically, we test with the Gemini 2.5 series, spanning the scale from the resource-efficient Lite model, through Flash, to the highly capable Pro model [10]. For Lite and Flash, we evaluate two different inference modes: with and without the thinking capability toggled on. When the thinking is toggled on, the model runs multiple internal passes that generate, criticize, and refine the output, which enhances its reasoning capabilities at the cost of slightly lower inference speed. We evaluate 2.5 Pro only with thinking on, since the Pro does not allow disabling the thinking capabilities. We visualize the results in Fig. 2 top-left and Fig. 3, and report the detailed results in Table 2.

> **Figure 3:** Change in success rates after adding VLM thinking.

Across all task categories and models, VLM inference with thinking enabled consistently outperforms the counterpart (Fig. 3), indicating that the reasoning capability of the VLM is critical towards better performance of hierarchical VLA. This is likely due to the fact that effectively making use of all the information available to the high-level policy is non-trivial, and thinking allows the model to better utilize this information (see Fig. 1 for an example). Additionally, notice that long-horizon tasks benefit more from thinking compared to the short-horizon tasks, suggesting that the reasoning capability becomes more important as the task gets more complex.

Surprisingly, the model size of the VLM does not have a significant impact on performance, where Lite, Flash, and Pro have similar performance across the board when thinking is on (Fig. 2 top-left). This is consistent with how existing hierarchical systems such as Pi-0.5 and Pi-0.7 [24, 23] can be performant despite using smaller high-level VLMs. One possible interpretation is that most existing robotics benchmark tasks do not require knowledge and instruction following ability beyond what smaller VLMs such as Lite can already provide, although larger models may become more important for tasks involving unfamiliar interfaces (e.g. operating a new coffee machine). Thus, despite Gemini-Pro outperforming Flash and Lite on existing VLM benchmarks [10], our results suggest that such benchmarks may not directly predict performance in hierarchical VLA systems.

**Key Takeaways.** Improved reasoning capability (via thinking) of the VLM has a big impact on the overall performance of the system. By comparison, the model size of the VLM seems to matter much less, where smaller reasoning-based models can work as well as larger models.

### 4.3 Effect of Low-Level VLA Policy

Next, we examine the impact of different low-level VLAs on the overall performance of the hierarchical system, where we focus on evaluating the impact of the size of the VLA as well as the effect of fine-tuning. We use a family of Gemini Robotics On-Device (GROD) Model [46] for our experiments. Specifically, we test three types of VLAs: GROD model trained with only real robot data; GROD (small), a smaller model trained with the same real dataset; and GROD (small) model, which is post-trained with in-domain simulation demonstrations from the same Mujoco environment that we evaluated in. We visualize the results in Fig. 2 top-mid, and report the detailed results in Table 3.

We can see that changing the VLA has a big effect on the performance of the system. This is unsurprising considering that the low-level VLA is ultimately the module that controls the robot. More specifically, we noticed that the GROD model with larger number of parameters consistently shows better performance than the smaller model, due to its better instruction following (as shown by the strong performance on short-horizon tasks) and generalization capability. Interestingly, the smaller GROD model finetuned with in-domain simulation data gives the worst performance, especially for long-horizon tasks. This is likely due to the fact the fine-tuning often results in worse instruction following capability of the VLA [22, 9] (i.e., slight rephrasings of the same instruction), which turns out to be very critical for maintaining good hierarchical performance.

**Key Takeaways.** Unlike high-level VLMs, low-level VLAs benefit significantly from increased size, likely due to the improved instruction following capabilities, making it more suitable for VLM orchestration. Simultaneously, loss of VLA steerability can lead to significant drop in performance, as shown by the poor performance of the simulation fine-tuned GROD model.

### 4.4 Termination Conditions

Similar to the options framework, a critical design choice in Hi-VLAs is the choice of when to hand the control back to the high-level VLM, also known as the "termination condition." We evaluate three types of termination conditions in our work:

- **Fixed Frequency:** The VLA hands control back to the VLM at a fixed, preset frequency.
- **Success Detection:** A success detector decides whether the instruction generated by the VLM is successfully completed given the current state.
- **VLM Termination:** The VLM generates an "expected execution time" along with the language command whenever it is queried.

In this work, we implement the success detection by querying a VLM with privileged state of the simulator to make it as accurate as possible, where the prompt is shown in Appendix F. We visualize the results in Fig. 2 top-right, and report the detailed results in Table 4. Among the three tested methods, the success detector consistently achieves good performance, showing that having a good termination condition can indeed positively impact the performance. VLM termination performs the worst, likely due to the stochastic nature of the low-level VLA which makes it hard to accurately predict execution length in advance. Finally, the short-horizon tasks are relatively agnostic to the termination conditions, likely because they do not require command sequencing and is thus less affected by when to switch.

> **Figure 4:** VLA Exec. Horizon
> **Figure 5:** Success Detection Error

**What is the effect of the VLA execution horizon?** An important hyperparameter for Fixed Frequency Termination is the execution horizon, which controls the number of low-level VLA steps before handing control back to the VLM. We conduct additional experiments evaluating this important hyper-parameter, and present the results in Fig. 5. As shown by the results, a too-long horizon can lead to timeouts for multi-step tasks, causing a significant drop in performance. While shorter execution horizon generally improves performance, the computational cost of higher-frequency VLM query makes it less ideal. Overall, we recommend selecting a moderate horizon, e.g., 4-8 seconds, that reduces the computational costs of VLM queries while maintaining a frequent-enough VLM control shown by very little performance drop.

**What if the success detector is not accurate?** While having a success detector as a termination condition can improve performance, obtaining a good success detector is not always easy [13]. Therefore, a natural question is how the system will behave as the accuracy of the success detector deteriorates. Our results (Fig. 5) show that success detector is a robust termination condition under moderate error. We elaborate on this experiment in Appendix B.

**Key Takeaways.** Success detection, even with moderate detection error, can be a powerful termination condition. However, too high of a detection error will significantly impact its performance. For fixed-horizon termination, a good VLA execution horizon (e.g. 4-8 seconds) allows us to reduce the cost of frequent VLM calls without significantly sacrificing the performance.

### 4.5 Observation Representations

Whenever the high-level VLM makes decisions, it needs to know the current state of the robot. While the naive approach would be to rely entirely on the image observation, we found that we often see better performance by carefully processing these image observations into text descriptions. Here, we examine four types of observation representation methods:

- **Raw Image:** Where we do not pass in any additional text description.
- **Naive Summarization:** Where we ask the VLM to first describe the image.
- **Summarize with Privileged Info:** Where we additionally pass in privileged information (more specifically, contact between objects) from the simulator when generating the text description.
- **Summarize with Bounding Box:** Where we first ask the VLM to generate bounding boxes of the objects, and use that information to generate the text description.

We visualize these representations in Fig. 6, and show the prompt for text summarization and bounding box generation in Appendix G and Appendix H. We visualize the results in Fig. 2 bottom-left, and present the detailed results in Table 5. Results show that both adding bounding box information and adding privileged information from simulator when generating text description significantly boost performance. This result is quite interesting since ideally the raw image already contains all the information, and we shouldn't have to pass in additional text description. One explanation for this could be due to the phenomenon that VLMs tend to ignore image inputs as task becomes harder [33], which is why passing in additional text information will boost performance.

Given that privileged information gives us the best performance across the board, we anticipate future enhancements in the image processing and spatial understanding capabilities of the VLM and/or incorporation of extra sensor modality to be important to the performance of hierarchical VLAs.

**Key Takeaways.** Good observation representation is critical to the performance of Hi-VLA. For example, bounding box description notably boost performance without requiring any extra information. Adding privileged information can further boost performance, calling for more focus on improving spatial understanding capabilities of the VLM.

> **Figure 6:** Observation representation pipeline. We study three ways of converting the raw image observation to text: (1) querying a VLM naively, (2) incorporating (VLM-generated) bounding box information to the query, and (3) incorporating privileged contact information to the query.

### 4.6 Memory

In this section, we examine the effect of appending previous interactions to the VLM context. Specifically, we accumulate the VLM history for 1 step, for 3 steps, for 5 steps, and from the entire episode so far, and feed it to the VLM planner. We visualize the results in Fig. 2 bottom-mid, and present the detailed results in Table 6. We find that the length of history context doesn't affect the performance much. This result suggests that the VLM cannot extract useful information from raw history of current episode, possibly because there is too little information to learn from.

**Experience summarization and reflection.** Memory does not necessarily have to appear in raw form. Instead, we can distill useful information from raw memory via summarization and reflection [36, 41]. We test summarizing experiences across 1-step, the current episode, and from previous episodes. Specifically for the previous episode setup, we first roll out the system for 10 episodes and then let a VLM summarize the experiences into affordances. We show the prompt for memory summarization in Sec. I. We visualize the results in Fig. 2 bottom-right, and present the detailed results in Table 7.

We find that in-episode summarization generally has a neutral to negative effect on performance, as seen when comparing "Memory Window - 1" with "Summary of last step" and "Full Memory" with "Summary of current episode." However, summarizing experiences across episodes can positively impact performance. This suggests that for hierarchical systems, extracting affordances from cross-episode information (especially previous successful episodes) is more beneficial than relying on in-episode failure signals for on-the-fly VLM corrections. Future work could explore more powerful techniques, such as reinforcement learning or supervised finetuning of the VLM, to better leverage cross-episodic interactions with the VLA.

**Key Takeaways.** The current system does not benefit much from in-episode memory even when summarization is available, and lacks the ability to do in-context learning, calling for more sophisticated memory processing mechanisms. However, cross-episodic knowledge helps a lot with task completion. A fruitful future direction is to explore how to perform VLM post-training with these cross-episodic experiences.

### 4.7 Aggregation of Discoveries

To put all our discoveries together, we evaluate:

- **Best hierarchical system:** the system that takes the combination of best-performing parameters from each of the earlier experiments, i.e. cross-episodic memory, thinking VLM, contact-based observation description, and success-based termination condition.
- **Naive hierarchical system:** the system that takes the combination of naively-chosen parameters from earlier experiments, i.e. no memory, VLM without thinking, raw observation, and fixed-horizon termination.
- **Flat VLA:** No VLM planner. The task prompt is directly passed to the low-level VLA.

Importantly, for each evaluation task, all three setups have the same VLA and input task prompt. We present the results in Table 1. We can see that even a naive implementation of a hierarchical system often outperforms the flat architecture, demonstrating the importance of introducing hierarchy. However, as the task becomes more challenging, the naive implementation quickly fall short, where more sophisticated hierarchical systems lead to a greater performance uplift. We further test these conclusions on a real robot, where we command an ALOHA robot to place fruits onto plates of matching color (Fig. 7). We report the number of correctly placed fruits across 5 trials in Table 1 (right), where the results indicate that our conclusions in simulation transfer to the real robot.

Lastly, we experiment on how potential improvements in VLA capabilities may affect our conclusions, and discuss the results in Appendix A.

| Configuration | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) | Real ALOHA |
|---|---|---|---|---|
| Best Hierarchy | 78.22 ± 0.91 | 67.08 ± 1.38 | 80.89 ± 1.17 | 12 / 15 |
| Naive Hierarchy | 69.57 ± 1.15 | 40.56 ± 1.37 | 66.49 ± 1.31 | 9 / 15 |
| Flat VLA | 69.63 ± 1.07 | 25.30 ± 1.22 | 50.90 ± 1.20 | 3 / 15 |

**Table 1:** Performance of the best hierarchy, naive hierarchy, and flat VLA.

**Key Takeaways.** A system with a clear, hierarchical control structure (orchestration) significantly boosts performance compared to a flat structure. However, simply introducing hierarchy is not enough; a good implementation of orchestration can make a big difference, especially for long-horizon and reasoning tasks, and will remain as VLA capabilities improve in the future.

## 5. Conclusion

In this work, we presented a systematic and comprehensive evaluation of design choices within Hierarchical Vision-Language-Action (Hi-VLA) systems, addressing the critical question of "what matters" for performance in complex robotic manipulation tasks. By constructing a flexible framework, we rigorously benchmarked the impact of various high-level VLMs, low-level VLAs, termination conditions, observation modalities, and memory mechanisms across diverse task genres including short-horizon, long-horizon, and reasoning challenges. Our key findings provide concrete guidance for researchers and practitioners to design future Hi-VLA systems.

**Limitations and Future Work.** A limitation of our current study is its focus on static environments and the assumption that latency is not a critical performance factor. Future work should explicitly investigate the impact of latency-sensitive scenarios and dynamic task environments on Hi-VLA performance. Furthermore, promising avenues include exploring Reinforcement Learning or supervised finetuning of the high-level VLM to better integrate cross-episodic knowledge and thereby more effectively guide the VLA's understanding of its low-level capabilities. Last but not least, the hierarchical architecture itself could be leveraged to inform and guide low-level continual policy improvement [20], leading to more robust and adaptable agents.

## Acknowledgments

We thank Travers Rhodes and Kevin Sayed for help on real robot experiments. We thank Laura Graesser for feedback on the paper, and Yilun Du, Wentao Yuan, Fei Xia, Wenhao Yu, Ted Xiao, Sandy Huang, Martin Riedmiller, and Jinyu Xie for the valuable discussions.

## References

[1] A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham, M. Bloesch, et al. (2025) Gemini robotics 1.5: pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342. Cited by: §1, §1.

[2] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al. (2022) Do as i can, not as i say: grounding language in robotic affordances. arXiv preprint arXiv:2204.01691. Cited by: §2.

[3] J. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. (2022) Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, pp. 23716–23736. Cited by: Appendix D.

[4] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. (2023) Qwen technical report. arXiv preprint arXiv:2309.16609. Cited by: Appendix D.

[5] S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson, Y. Chebotar, D. Dwibedi, and D. Sadigh (2024) Rt-h: action hierarchies using language. arXiv preprint arXiv:2403.01823. Cited by: §1.

[6] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. (2024) Paligemma: a versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726. Cited by: Appendix D.

[7] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024) π₀: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: Appendix D, §1, §2.

[8] R. Brooks (2003) A robust layered control system for a mobile robot. IEEE journal on robotics and automation 2 (1), pp. 14–23. Cited by: §2.

[9] W. Chen, J. S. Bhatia, C. Glossop, N. Mathihalli, R. Doshi, A. Tang, D. Driess, K. Pertsch, and S. Levine (2026) Steerable vision-language-action policies for embodied reasoning and hierarchical control. ArXiv abs/2602.13193. External Links: Link Cited by: §4.3.

[10] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. (2025) Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261. Cited by: §4.2, §4.2.

[11] P. Ding, J. Ma, X. Tong, B. Zou, X. Luo, Y. Fan, T. Wang, H. Lu, P. Mo, J. Liu, et al. (2025) Humanoid-vla: towards universal humanoid control with visual integration. arXiv preprint arXiv:2502.14795. Cited by: §2.

[12] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, et al. (2023) Palm-e: an embodied multimodal language model. arXiv preprint. Cited by: Appendix D.

[13] Y. Du, K. Konyushkova, M. Denil, A. Raju, J. Landon, F. Hill, N. De Freitas, and S. Cabi (2023) Vision-language models as success detectors. arXiv preprint arXiv:2303.07280. Cited by: §4.4.

[14] A. Figure (2024) Helix: a vision-language-action model for generalist humanoid control. Figure AI News. Cited by: §1, §2, §3.

[15] R. M. French (1999) Catastrophic forgetting in connectionist networks. Trends in cognitive sciences 3 (4), pp. 128–135. Cited by: §2.

[16] M. Fu, J. Yu, K. El-Refai, E. Kou, H. Xue, H. Huang, W. Xiao, G. Wang, F. Li, G. Shi, J. Wu, S. Sastry, Y. Zhu, K. Goldberg, and L. J. Fan (2026) CaP-x: a framework for benchmarking and improving coding agents for robot manipulation. External Links: Link Cited by: §2.

[17] C. Gao, Z. Liu, Z. Chi, J. Huang, X. Fei, Y. Hou, Y. Zhang, Y. Lin, Z. Fang, Z. Jiang, and L. Shao (2025) VLA-os: structuring and dissecting planning representations and paradigms in vision-language-action models. External Links: 2506.17561, Link Cited by: Appendix D.

[18] J. Gao, S. Belkhale, S. Dasari, A. Balakrishna, D. Shah, and D. Sadigh (2025) A taxonomy for evaluating generalist robot policies. arXiv preprint arXiv:2503.01238. Cited by: Appendix D.

[19] P. Guruprasad, H. Sikka, J. Song, Y. Wang, and P. P. Liang (2024) Benchmarking vision, language, and action models on robotic learning tasks. External Links: 2411.05821, Link Cited by: Appendix D.

[20] J. Hu, J. Shim, C. Tang, Y. Sung, B. Liu, P. Stone, and R. Martin-Martin (2026) Simple recipe works: vision-language-action models are natural continual learners with reinforcement learning. External Links: 2603.11653, Link Cited by: §5.

[21] J. Hu, P. Stone, and R. Martín-Martín (2025) SLAC: simulation-pretrained latent action space for whole-body real-world rl. arXiv preprint arXiv:2506.04147. Cited by: §2.

[22] S. Huang, J. Shao, K. Wang, Q. Chen, J. Sun, Y. Guo, M. Schwager, and J. Bohg (2026) Breaking lock-in: preserving steerability under low-data vla post-training. External Links: Link Cited by: §4.3.

[23] P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, V. Choudhary, F. Collins, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, M. Dhaka, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Habeeb, H. Hancock, K. Hausman, G. Hussein, V. Hwang, B. Ichter, C. Jacobsen, S. Jakubczak, R. Jen, T. Jones, G. Kammerer, B. Katz, L. Ke, M. Khadikov, C. Kuchi, M. Lamb, D. LeBlanc, B. LeCount, S. Levine, X. Li, A. Li-Bell, V. Lialin, Z. Liang, W. Lim, Y. Lu, E. Luo, V. Mano, N. Marwaha, A. Mongush, L. Murphy, S. Nair, T. Patterson, K. Pertsch, A. Z. Ren, G. Schelske, C. Sharma, B. Shi, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, J. Tang, J. Tanner, S. Tekeste, M. Torne, K. Vedder, Q. Vuong, A. Walling, H. Wang, J. Wang, X. Wang, C. Whalen, S. Whitmore, B. Williams, C. Xu, S. Yoo, L. Yu, W. Zhang, Z. Zhang, and U. Zhilinsky (2026) π₀.₇: A steerable generalist robotic foundation model with emergent capabilities. External Links: 2604.15483, Link Cited by: §4.2.

[24] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025) π₀.₅: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: §1, §2, §3, §4.2.

[25] T. Jiang, T. Yuan, Y. Liu, C. Lu, J. Cui, X. Liu, S. Cheng, J. Gao, H. Xu, and H. Zhao (2025) Galaxea open-world dataset and g0 dual-system vla model. External Links: 2509.00576, Link Cited by: §2.

[26] D. Kahneman (2011) Thinking, fast and slow. Farrar, Straus and Giroux. Cited by: §1, §2.

[27] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024) Openvla: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: Appendix D, §1.

[28] J. Li, D. Li, S. Savarese, and S. Hoi (2023) Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. Cited by: Appendix D.

[29] X. Li, P. Li, M. Liu, D. Wang, J. Liu, B. Kang, X. Ma, T. Kong, H. Zhang, and H. Liu (2024) Towards generalist robot policies: what matters in building vision-language-action models. External Links: 2412.14058, Link Cited by: Appendix D.

[30] Y. Li, Y. Deng, J. Zhang, J. Jang, M. Memmel, R. Yu, C. R. Garrett, F. Ramos, D. Fox, A. Li, et al. (2025) Hamster: hierarchical action models for open-world robot manipulation. arXiv preprint arXiv:2502.05485. Cited by: §1, §2.

[31] H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023) Visual instruction tuning. Advances in neural information processing systems 36, pp. 34892–34916. Cited by: Appendix D.

[32] Y. Ma, Z. Song, Y. Zhuang, J. Hao, and I. King (2025) A survey on vision-language-action models for embodied ai. External Links: 2405.14093, Link Cited by: Appendix D.

[33] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud, et al. (2024) Openeqa: embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16488–16498. Cited by: §4.5.

[34] NVIDIA, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. ". Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu (2025) GR00T n1: an open foundation model for generalist humanoid robots. External Links: 2503.14734, Link Cited by: §1.

[35] A. O'Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. (2024) Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. Cited by: §1.

[36] J. S. Park, J. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein (2023) Generative agents: interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pp. 1–22. Cited by: §4.6.

[37] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. Cited by: Appendix D.

[38] R. Sapkota, Y. Cao, K. I. Roumeliotis, and M. Karkee (2025) Vision-language-action models: concepts, progress, applications and challenges. External Links: 2505.04769, Link Cited by: Appendix D.

[39] J. Shi, R. Yang, K. Chao, S. Wan, Y. S. Shao, J. Lei, J. Qian, L. Le, P. Chaudhari, K. Daniilidis, C. Wen, and D. Jayaraman (2025) Maestro: orchestrating robotics modules with vision-language models for zero-shot generalist robots. ArXiv abs/2511.00917. External Links: Link Cited by: §2.

[40] L. X. Shi, B. Ichter, M. Equi, L. Ke, K. Pertsch, Q. Vuong, J. Tanner, A. Walling, H. Wang, N. Fusai, et al. (2025) Hi robot: open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417. Cited by: §1, §2.

[41] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023) Reflexion: language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems 36, pp. 8634–8652. Cited by: §4.6.

[42] Z. Su, B. Zhang, N. Rahmanian, Y. Gao, Q. Liao, C. Regan, K. Sreenath, and S. S. Sastry (2025) Hitter: a humanoid table tennis robot via hierarchical planning and learning. arXiv preprint arXiv:2508.21043. Cited by: §2.

[43] R. S. Sutton, D. Precup, and S. Singh (1999) Between mdps and semi-mdps: a framework for temporal abstraction in reinforcement learning. Artificial intelligence 112 (1-2), pp. 181–211. Cited by: §1, §2, §3.

[44] H. Tan, X. Hao, C. Chi, M. Lin, Y. Lyu, M. Cao, D. Liang, Z. Chen, M. Lyu, C. Peng, et al. (2025) Roboos: a hierarchical embodied framework for cross-embodiment and multi-agent collaboration. arXiv preprint arXiv:2505.03673. Cited by: §1, §2.

[45] G. Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023) Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. Cited by: Appendix D.

[46] G. R. Team, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. (2025) Gemini robotics: bringing ai into the physical world. arXiv preprint arXiv:2503.20020. Cited by: Appendix D, §1, §1, §2, §3, §4.1, §4.3.

[47] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al. (2025) Tinyvla: towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters. Cited by: §1.

[48] Y. Zhong, F. Bai, S. Cai, X. Huang, Z. Chen, X. Zhang, Y. Wang, S. Guo, T. Guan, K. N. Lui, Z. Qi, Y. Liang, Y. Chen, and Y. Yang (2025) A survey on vision-language-action models: an action tokenization perspective. External Links: 2507.01925, Link Cited by: Appendix D.

[49] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny (2023) Minigpt-4: enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592. Cited by: Appendix D.

[50] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023) Rt-2: vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. Cited by: Appendix D, §1.

## Supplementary Materials

> **Figure 7:** Motion sequence of the real ALOHA robot. Orchestration allows the robot to recover from the fruit misplacement (at step 6) and eventually solve the task.
> - (a) Step 1 &nbsp; (b) Step 2 &nbsp; (c) Step 3 &nbsp; (d) Step 4 &nbsp; (e) Step 5 &nbsp; (f) Step 6 &nbsp; (g) Step 7 &nbsp; (h) Step 8

---

## Appendix A: Effect of Improving VLA Action Quality

> **Figure 8:** Performance of different hierarchies with a scripted low-level policy.

In this section, we experiment on how potential improvements in the VLA's action predictions may affect our conclusions.

First of all, note that for a "perfect" VLA, hierarchical systems are almost meaningless, since it should be able to directly complete any given task without orchestration. However, we believe that a more realistic future VLA would be one that can complete a range of short-horizon commands with high accuracy, but not necessarily longer or reasoning-based ones.

As a proxy for such VLAs, we created a low-level controller in the form of a scripted policy that utilizes privileged information in the simulator to take actions, such that it can nearly perfectly complete tasks when conditioned on the right language command, but would do nothing when it cannot parse the command. We test four different settings: the full hierarchical system, removing the observation description, removing the memory, and finally the naive hierarchical system, and show the average performances on a set of challenging long-horizon tasks in Fig. 8.

We can see that the full hierarchical system achieves a very high average success rate of around 95%. However, ablations of hierarchical components (e.g., observation representation, memory, or naive orchestration) can degrade performance from ~95% success to nearly 0%. This result suggests that as VLA capabilities improve, hierarchical design and orchestration will remain an important factor, rather than being obviated by better low-level policies.

## Appendix B: Robustness to Imperfect Success Detectors

In this section, we detail the experiments we conducted for testing the robustness of success detection as a termination criteria under accuracy deterioration.

We consider this problem from two settings: increasing false positive rate and increasing false negative rate. For each setting, we consider a corruption probability of 10%, 30%, and 50% respectively, and report the results in Fig. 5.

Intriguingly, a small amount of success detection error (10%) does not hurt performance at all, and in fact slightly boost it, suggesting that success detection can be a robust termination condition for hierarchical VLAs. As the detection error rate goes up, however, false positive errors start to drastically impact the system performance, likely because of the false "command completed" feedback that causes the VLM to move on to a new command.

While the impact of false negatives (FN) error remains moderate, it is important to note that this result may stem from the fact that our corruption is independent across different states. In other words, even if the current success check failed due to FN, subsequent success checks at later timesteps will still have a good chance of correctly terminating the command. By contrast, a success detector in the real world may show high correlation across detection error of consecutive states, meaning that once a FN occurs, the command may fail to terminate for a very long time. As we have shown in the "execution horizon" experiment from the previous paragraphs, such a behavior can actually hurt the performance quite significantly.

## Appendix C: Evaluation Setup

We carry out our main experiments in the MuJoCo Aloha suite, and show some of these environments in Fig. 9. Operating in simulation brings us two main advantages: first, simulation parallelization allows us to run large-scale evaluations and obtain results with statistical significance. Second, we can leverage privileged information available in simulation to examine counterfactual hypotheses that can suggest directions for future improvement.

Ideally, a good hierarchical VLA system should have the following desired abilities:

- The high-level VLM has the ability to learn about the language affordance of the VLA from experiences, and eventually, only generate commands that respect the affordance.
- The system can tackle long-horizon tasks by breaking it down into executable sub-tasks.
- The system is able to interpret and reason about indirect instructions by leveraging the strong prior from the VLM.

In an effort to disentangle these different capabilities, we categorize our evaluation tasks into three different categories, such that each category of tasks roughly corresponds to one desired capability, allowing us to examine these capabilities separately. We describe the categorization below and discuss the tasks in detail in Section K.

- **Short-horizon:** the task has length similar to the VLA training trajectories, such as pick and place of a single object.
- **Long-horizon:** the task length is significantly longer than the VLA training trajectories, requiring non-trivial compositions of short-horizon skills.
- **Reasoning:** instructions need interpretation, e.g. "put the object you pour coffee in on plate".

In our experiments, the result for each task category (i.e. short-horizon, long-horizon, reasoning) is averaged over 5 tasks and 200 independent trials per task. The error bar represents the standard error arising from finite-sample binomial uncertainty. Whenever we are examining one component, we fix the rest of the components to a standard setup for result consistency, as described below:

- The VLM is set to be Gemini 2.5 Flash with thinking on.
- The VLA is set to be the GROD 3B model trained only on real robot data.
- The termination condition is set to be fixed frequency switch with a duration of 8 seconds.
- The observation representation is set to be scene description with contact information.
- The memory window is set to be 3, without using any summarization.

## Appendix D: Discussion of Previous Work on Flat VLAs

**Vision-Language-Action Models.** Internet-scale multi-modal data and transformer-based architectures have given rise to models capable of jointly processing visual and linguistic information, commonly known as vision-language models [37, 3, 28, 45, 6, 4, 49, 12, 31]. Building upon these foundation models, the robotics community recently introduced the notion of vision-language-action models, including RT-2 [50], Pi [7], OpenVLA [27], Gemini Robotics [46], and more. These models are fine-tuned from VLMs to map natural-language instructions and perceptual inputs directly to robot actions, allowing VLMs to "speak the language of robotics" and ground their extensive knowledge in the physical, embodied world. This adaptation unlocks a paradigm shift in robot learning, enabling unprecedented levels of zero-shot generalization across varied tasks and environments. These VLA models serve as the foundation for the study of this work.

**Benchmarks and Evaluations of VLA Systems.** VLAs have quickly gained traction ever since their introduction, due to the promise of generalization across tasks and embodiments. This increased attention led to surveys [48, 38, 32], benchmarks [19, 18], and studies [29, 17] that seeks to assess and systematize these advances. However, existing works primarily focus on flat VLAs, where a single, monolithic model directly maps instructions to low-level robot actions. By contrast, our work focuses on systematic dissection and evaluation of hierarchical VLA systems, where a high-level VLM planner orchestrates low-level VLA modules.

## Appendix E: High-Level VLM Prompt

```
High-level VLM Decision Making
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image Input: [Current Camera Observations]

Text Input: You are a decision-making agent tasked with generating
natural language command that is sent to a Robotics Vision-Language-Action
(VLA) model for a two-armed robot towards completing the given task
[Task Instruction]. The command must be in the active, second-person voice
(addressing the robot), based on the current observation of the robot shown
in the image above. [Observation Representation].

Key Directives:
1. Output a single command that should be executed immediately. The command
   should facilitate completion of the given task.
2. The command should be doable within 10 seconds. Consider the affordance
   of the VLA based on the history steps as well as the current state of
   the robot.
3. Think step by step internally to arrive at the command. Do not output
   your thought process.

Current Memory: [Memory].

VLM Policy Output: [Language Command]
```

## Appendix F: Success Detection Prompt

```
Success Detection Prompt
━━━━━━━━━━━━━━━━━━━━━━━━
Image Input: [A sequence of Observations]

Text Input: You are a success detector for a robot. Your job is to check
whether the robot has successfully completed a command based on the current
observation of the robot cameras and the scene information below.
[Privileged State Information].

Think about what criteria are required for success. Only output yes if all
of the criteria are met.

Examples:
1) pick up requires that the object is NOT making contact with the table
   AND is in contact with the gripper
2) put in requires that the object is in contact with the container AND is
   NOT in contact with the gripper

Has the robot completed the following command "[Language Command]"?
Answer with only "yes" or "no" or "uncertain" (in lowercase).

VLM Output: [yes / no / uncertain]
```

## Appendix G: Observation Description Prompt

```
Observation Description Prompt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image Input: [Current Observation]

Text Input: The user wants a robot to perform the following task:
[Instruction].

Scene information: [BBox info / Privileged info / None]

Based on the current observation of the robot shown in the image above,
and the scene information, please provide a concise description of the
scene, focus only on task relevant objects and what the robot is doing
(e.g. is it trying to pick something?).

VLM Output: [Scene Descriptions]
```

## Appendix H: Bounding Box Prompt

```
Bounding Box Prompt
━━━━━━━━━━━━━━━━━━━
Image Input: [Current Observation]

Text Input: Detect the locations of the objects: [Object List],
'Left Gripper', 'Right Gripper'. Output a json list where each entry
contains the 2D bounding box in "box_2d" and a text label in "label".
Only one entry per object.

VLM Output: Below are the bounding boxes for the objects in the image,
which are represented as [y_min, x_min, y_max, x_max] and normalized to
be between 0-1000. [Bounding Box JSON Output]
```

## Appendix I: Memory Summarization Prompt

```
Memory Summarization Prompt
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: You are a decision-making agent tasked with generating a sequence
of natural language commands for a two-armed robot to complete the given
task [Instruction].

Step history [× n]:
  Step [i]
    Instruction: [language command].
    Instruction successfully completed? [yes / no]
    Instruction result: [observation representation]

Based on the step history above, summarize the affordance of this
language-conditioned policy into two or three short bullet points that
can help the agent make better decisions. Be as concise as possible.
```

## Appendix J: Results Tables

### J.1 VLM

**Table 2:** Evaluating different choices of VLM

| VLM Configuration | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | 70.48 ± 1.05 | 48.73 ± 1.52 | 58.51 ± 1.19 |
| Gemini 2.5 Flash-Lite (thinking) | 74.44 ± 0.89 | 58.21 ± 1.52 | 75.20 ± 1.25 |
| Gemini 2.5 Flash | 72.63 ± 0.94 | 47.02 ± 1.45 | 71.79 ± 1.22 |
| Gemini 2.5 Flash (thinking) | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |
| Gemini 2.5 Pro (thinking) | 70.10 ± 1.01 | 53.06 ± 1.42 | 74.39 ± 1.21 |

### J.2 VLA

**Table 3:** Evaluating different low-level VLA models.

| VLA Configuration | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| GROD-1B | 63.40 ± 1.16 | 41.30 ± 1.49 | 66.90 ± 1.25 |
| GROD-1B (FT with sim) | 54.60 ± 1.09 | 7.50 ± 0.80 | 43.00 ± 1.05 |
| GROD-3B | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |

### J.3 Termination Condition

**Table 4:** Termination Condition

| Termination Condition | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| VLM-based Horizon | 72.16 ± 0.95 | 43.50 ± 1.14 | 72.27 ± 1.14 |
| Success Detector | 74.65 ± 0.92 | 57.39 ± 1.52 | 80.89 ± 1.17 |
| Fixed Horizon (T=400) | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |

### J.4 Observation Description

**Table 5:** System performance with different observation representations

| Observation Representation | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| Image | 67.56 ± 0.98 | 38.84 ± 1.50 | 69.21 ± 1.23 |
| Image + description | 67.93 ± 1.01 | 35.70 ± 1.43 | 62.77 ± 1.31 |
| Image + description + bboxes | 73.94 ± 0.95 | 47.90 ± 1.67 | 68.51 ± 1.26 |
| Image + description + contact info | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |

### J.5 Memory Length

**Table 6:** Memory Length

| Memory Length | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| Full Memory | 76.53 ± 0.99 | 58.98 ± 1.61 | 72.77 ± 1.22 |
| Memory Window - 5 | 76.09 ± 0.87 | 57.76 ± 1.66 | 72.20 ± 1.26 |
| Memory Window - 3 | 75.81 ± 0.93 | 58.21 ± 1.52 | 72.62 ± 1.17 |
| Memory Window - 1 | 76.76 ± 0.96 | 59.89 ± 1.60 | 74.27 ± 1.20 |

### J.6 Memory Summary

**Table 7:** Memory Summarization

| Memory Summarization | Short-Horizon (%) | Long-Horizon (%) | Reasoning (%) |
|---|---|---|---|
| No summary | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |
| Summary of last step | 74.61 ± 1.00 | 52.57 ± 1.52 | 72.82 ± 1.03 |
| Summary of current episode | 71.66 ± 1.04 | 50.12 ± 1.52 | 75.72 ± 1.17 |
| Summary of previous episodes | 79.45 ± 0.81 | 60.00 ± 1.50 | 80.30 ± 1.23 |

## Appendix K: Tasks Description

In this section, we provide detailed descriptions of the tasks that we evaluated upon. These tasks are implemented based on the open-sourced MuJoCo ALOHA suite. We visualize some of these scenes as well as the real robot scene in Fig. 9.

> **Figure 9:** Example scenes from our study. Each scene is intentionally designed to support multiple different tasks, as specified in detail in App. K.
> - (a) Dining Scene &nbsp; (b) Cup Scene &nbsp; (c) Fruit Scene &nbsp; (d) Real ALOHA

### K.1 Reasoning Tasks

**Reasoning Task 1:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the banana in the bowl.
- Instruction: "Put the item that monkey can eat into the bowl".

**Reasoning Task 2:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the mug on the plate.
- Instruction: "Put the object you pour coffee in on the plate".

**Reasoning Task 3:**
- Scene: The table contains three cups and three cubes, each set corresponding to the colors red, green, and blue.
- Goal: The robot needs to put each cube into the cup with the same color.
- Instruction: "Put all the cubes into their matching colored cups.".

**Reasoning Task 4:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put both the banana and the pen in the bowl.
- Instruction: "Put all things that are long in the bowl".

**Reasoning Task 5:**
- Scene: The table contains a plate, a bowl, a banana, a lime, an orange, an apple, and a bottle.
- Goal: The robot needs to put the lime in the bowl.
- Instruction: "Put the sourest fruit in the bowl.".

### K.2 Long-horizon Tasks

**Long-horizon Task 1:**
- Scene: The table contains three cups and three cubes, each set corresponding to the colors red, green, and blue.
- Goal: The robot needs to put all the cubes into the green cup.
- Instruction: "Put all the cubes into the green cup".

**Long-horizon Task 2:**
- Scene: The table contains three cups and three cubes, each set corresponding to the colors red, green, and blue.
- Goal: The robot needs to first stack the red cup into the green cup, and then put the red cube into the red cup.
- Instruction: "Put the red cup into the green cup. Then put the red cube into the red cup.".

**Long-horizon Task 3:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put both the banana and the pen in the bowl.
- Instruction: "Put the banana and the pen in the bowl.".

**Long-horizon Task 4:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put both the mug and the banana in the plate.
- Instruction: "Put the banana and the mug on the plate.".

**Long-horizon Task 5:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the banana in the bowl; and also put the mug on the plate.
- Instruction: "Put the banana in the bowl and the mug on the plate".

### K.3 Short-horizon Tasks

**Short-horizon Task 1:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the banana in the bowl.
- Instruction: "Put the banana in the bowl".

**Short-horizon Task 2:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the mug on the plate.
- Instruction: "Put the mug on the plate".

**Short-horizon Task 3:**
- Scene: The table contains a plate, a bowl, a white container, a red mug, a banana, and a pen.
- Goal: The robot needs to put the pen in the container.
- Instruction: "Put the pen in the white container".

**Short-horizon Task 4:**
- Scene: The table contains a plate, a bowl, a banana, a lime, an orange, an apple, and a bottle.
- Goal: The robot needs to put the bottle in the bowl.
- Instruction: "Place the bottle in the left bowl".

**Short-horizon Task 5:**
- Scene: The table contains a plate, a bowl, a banana, a lime, an orange, an apple, and a bottle.
- Goal: The robot needs to put the lime in the bowl.
- Instruction: "Place the lime in the left bowl".
