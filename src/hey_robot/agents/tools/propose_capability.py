import time
import uuid

from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import (
    BooleanSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)


@tool_parameters(
    tool_parameters_schema(
        capability=StringSchema("Robot capability to propose for confirmation."),
        objective=StringSchema("What would be accomplished if confirmed."),
        slots=ObjectSchema(
            description="Capability slots to use after confirmation", nullable=True
        ),
        interrupt=BooleanSchema(
            description="Whether the confirmed action would interrupt current execution"
        ),
        confirmation_prompt=StringSchema(
            "User-facing question asking for confirmation."
        ),
        required=["capability", "objective", "confirmation_prompt"],
    )
)
class ProposeCapabilityTool(Tool):
    name = "propose_capability"
    description = "Create a task-level pending confirmation and ask the user to confirm before execution."
    safety_level = "communicate"
    read_only = True
    result_policy = "return_direct"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._current_envelope = ctx._current_envelope
        self._get_task = ctx._get_task

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(
        self,
        capability: str,
        objective: str,
        confirmation_prompt: str,
        slots: dict | None = None,
        interrupt: bool = False,
    ) -> str:
        capability = (capability or "").strip()
        objective = (objective or "").replace("__TASK__", self._get_task()).strip()
        prompt = (confirmation_prompt or "").strip()
        if not capability:
            raise ValueError("capability must not be empty")
        if not objective:
            raise ValueError("objective must not be empty")
        if not prompt:
            raise ValueError("confirmation_prompt must not be empty")
        envelope = self._current_envelope()
        proposal = {
            "proposal_id": f"proposal_{uuid.uuid4().hex[:16]}",
            "capability": capability,
            "objective": objective,
            "slots": dict(slots or {}),
            "interrupt": bool(interrupt),
            "prompt": prompt,
            "episode_id": envelope.episode_id,
            "robot_id": envelope.robot_id,
            "agent_id": envelope.agent_id,
            "created_at": time.time(),
        }
        self._ctx.io.task_runtime.store_pending_confirmation(
            envelope.episode_id, proposal
        )
        return prompt
