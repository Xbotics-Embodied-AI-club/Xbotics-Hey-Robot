"""Compatibility name for the long-horizon harness state store."""

from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStep,
    AgentTaskStore,
    CompletionCheck,
    StepStatus,
    TaskStatus,
)

HarnessStore = AgentTaskStore

__all__ = [
    "AgentTask",
    "AgentTaskStep",
    "CompletionCheck",
    "HarnessStore",
    "StepStatus",
    "TaskStatus",
]
