from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.schema import StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        reason=StringSchema("Why no action is being taken"),
    )
)
class WaitTool(Tool):
    name = "wait"
    description = "Do nothing and wait for more information or execution progress."
    read_only = True
    safety_level = "normal"

    def __init__(self) -> None:
        pass

    @classmethod
    def create(cls, _ctx: object) -> "WaitTool":
        return cls()

    async def execute(self, reason: str = "") -> str:
        return f"waiting: {reason or 'no action'}"
