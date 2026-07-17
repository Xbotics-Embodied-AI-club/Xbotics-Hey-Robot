"""供自主控制使用的不可变 Goal 契约和带类型证据。"""

from hey_robot.cognition.task.contract import TaskContract, create_task_contract
from hey_robot.cognition.task.evidence import project_robot_status

__all__ = ["TaskContract", "create_task_contract", "project_robot_status"]
