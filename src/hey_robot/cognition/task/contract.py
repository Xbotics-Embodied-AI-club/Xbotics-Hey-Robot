"""Goal contracts are immutable and deliberately contain no execution plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from hey_robot.protocol import SuccessCriterion


@dataclass(frozen=True)
class TaskContract:
    contract_id: str
    task_id: str
    goal_id: str
    objective: str
    success_criteria: tuple[SuccessCriterion, ...]
    schema_version: int
    contract_hash: str


def create_task_contract(
    *,
    goal_id: str,
    task_id: str,
    contract_id: str,
    objective: str,
    success_criteria: tuple[SuccessCriterion, ...],
) -> TaskContract:
    if not success_criteria:
        raise ValueError("GOAL_CONTRACT_REQUIRED")
    payload = {
        "schema_version": 1,
        "goal_id": goal_id,
        "task_id": task_id,
        "contract_id": contract_id,
        "objective": objective,
        "success_criteria": [asdict(item) for item in success_criteria],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TaskContract(
        contract_id=contract_id,
        task_id=task_id,
        goal_id=goal_id,
        objective=objective,
        success_criteria=success_criteria,
        schema_version=1,
        contract_hash=digest,
    )
