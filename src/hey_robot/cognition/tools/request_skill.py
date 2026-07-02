from typing import cast

from hey_robot.cognition.skill_gateway import (
    SkillGateway,
    SkillGatewayRequest,
    WaitPolicy,
)
from hey_robot.cognition.tools.base import Tool, tool_parameters
from hey_robot.cognition.tools.context import ToolContext
from hey_robot.cognition.tools.schema import (
    BooleanSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

_VALID_WAIT_POLICIES = {"wait_result", "wait_acceptance", "return_handle"}


@tool_parameters(
    tool_parameters_schema(
        skill=StringSchema("Robot skill to request."),
        objective=StringSchema("What to accomplish with this skill"),
        slots=ObjectSchema(
            description="Skill slots passed to the resolver/executor",
            nullable=True,
        ),
        interrupt=BooleanSchema(description="Whether this is an interrupt signal"),
        wait_policy=StringSchema(
            "wait_result, wait_acceptance, or return_handle",
            enum=["wait_result", "wait_acceptance", "return_handle"],
        ),
        required=["skill", "objective"],
    )
)
class RequestSkillTool(Tool):
    """Single Agent-facing gateway for robot skill requests."""

    name = "request_skill"
    description = "Request one robot skill without exposing low-level tool scheduling to the Agent."
    safety_level = "actuate"
    exclusive = True
    resources = ("robot.actuation",)

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._gateway = cast(SkillGateway | None, getattr(ctx, "skill_gateway", None))

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(
        self,
        skill: str,
        objective: str,
        slots: dict | None = None,
        interrupt: bool = False,
        wait_policy: str = "wait_result",
    ) -> str:
        if self._gateway is None:
            raise RuntimeError("skill gateway is not configured")
        normalized_wait_policy = wait_policy or "wait_result"
        if normalized_wait_policy not in _VALID_WAIT_POLICIES:
            raise ValueError(f"unknown wait_policy: {wait_policy}")
        return await self._gateway.submit(
            SkillGatewayRequest(
                skill=skill,
                objective=objective,
                slots=dict(slots or {}),
                interrupt=interrupt,
                wait_policy=cast(WaitPolicy, normalized_wait_policy),
            )
        )
