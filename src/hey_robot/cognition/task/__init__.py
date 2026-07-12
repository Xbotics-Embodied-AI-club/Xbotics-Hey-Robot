"""Immutable goal contracts and typed evidence for autonomous control."""

from hey_robot.cognition.task.contract import TaskContract, create_task_contract
from hey_robot.cognition.task.evidence import project_robot_status

__all__ = ["TaskContract", "create_task_contract", "project_robot_status"]
