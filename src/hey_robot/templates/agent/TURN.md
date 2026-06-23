当前任务：{{ task }}

机器人状态：{{ robot_state }}
{{ skill_status_context }}
{{ last_feedback }}
{{ recovery_context }}
{{ task_contract_context }}
{{ next_hint }}
{{ loop_warning }}
{{ memory_context }}
{{ autonomy_context }}

请判断下一步。
- 普通回复直接用简体中文纯文本回答，不使用 Markdown 标题、列表、代码块或表格。
- 复杂任务可以在这一轮连续调用多个工具，但总数通常控制在 3 到 5 次。
- 每次只执行一个最小、安全、可验证的动作。
- 每做完一步，都要利用 execution feedback、任务上下文、感知证据或记忆再决定下一步。
- 如果缺少关键信息，优先使用 `get_task_context`、`get_robot_status`、`request_perception` 或 `search_memory`。
- 如果已经进入恢复状态，先用 `get_task_context` 理解失败原因和恢复建议，再决定是否重试、重看、澄清或终止。
- 如果连续没有进展，不要重复同一个动作，应解释阻塞点并给出更稳妥的下一步。
