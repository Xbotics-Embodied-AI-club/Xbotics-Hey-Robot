"""单一 Robot Agent 使用的仅提案式工具。"""

from hey_robot.cognition.tools.robot import (
    ToolDependencies,
    ToolRegistry,
)
from hey_robot.cognition.tools.skill_tools import SkillCallProposal, SkillTool
from hey_robot.cognition.tools.task_tools import (
    CompleteTaskTool,
    ControlTaskTool,
)

__all__ = [
    "CompleteTaskTool",
    "ControlTaskTool",
    "SkillCallProposal",
    "SkillTool",
    "ToolDependencies",
    "ToolRegistry",
]
