from typing import cast

from hey_robot.agents.skill_gateway import SkillGateway, SkillGatewayRequest, WaitPolicy
from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import (
    BooleanSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

_VALID_WAIT_POLICIES = {"wait_result", "wait_acceptance", "return_handle"}


@tool_parameters(
    tool_parameters_schema(
        capability=StringSchema("Robot capability to request."),
        objective=StringSchema("What to accomplish with this skill"),
        slots=ObjectSchema(
            description="Capability slots passed to the resolver/executor",
            nullable=True,
        ),
        interrupt=BooleanSchema(description="Whether this is an interrupt signal"),
        wait_policy=StringSchema(
            "wait_result, wait_acceptance, or return_handle",
            enum=["wait_result", "wait_acceptance", "return_handle"],
        ),
        required=["capability", "objective"],
    )
)
class RequestCapabilityTool(Tool):
    """Single Agent-facing gateway for robot capability requests."""

    name = "request_capability"
    description = "Request one robot capability without exposing low-level tool scheduling to the Agent."
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
        capability: str,
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
                capability=capability,
                objective=objective,
                slots=dict(slots or {}),
                interrupt=interrupt,
                wait_policy=cast(WaitPolicy, normalized_wait_policy),
            )
        )
