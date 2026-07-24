"""Compatibility imports for the former robot-only Tool Registry module."""

from hey_robot.cognition.tools.models import (
    CompleteTaskProposal,
    ControlTaskProposal,
)
from hey_robot.cognition.tools.registry import (
    SkillCatalogView,
    ToolDependencies,
    ToolRegistry,
)

__all__ = [
    "CompleteTaskProposal",
    "ControlTaskProposal",
    "SkillCatalogView",
    "ToolDependencies",
    "ToolRegistry",
]
