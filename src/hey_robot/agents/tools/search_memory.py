from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import (
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("What to search for in long-term memory", nullable=True),
        kind=StringSchema(
            "Memory kind/filter",
            enum=[
                "event",
                "entity",
                "place",
                "skill_experience",
                "task_result",
                "user_preference",
                "scene_anchor",
                "task_lesson",
            ],
            nullable=True,
        ),
        mode=StringSchema(
            "'structured' for exact search; 'semantic' for relevance search",
            enum=["structured", "semantic"],
            nullable=True,
        ),
        limit=IntegerSchema(
            8, description="Max records to return", minimum=1, maximum=20
        ),
    )
)
class SearchMemoryTool(Tool):
    name = "search_memory"
    description = "Search long-term memory."
    read_only = True
    safety_level = "observe"

    def __init__(self, ctx: ToolContext) -> None:
        self._memory = ctx.memory
        self._get_task = ctx._get_task

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(
        self,
        query: str | None = None,
        kind: str | None = None,
        mode: str | None = None,
        limit: int = 8,
    ) -> str:
        return self._memory.search(
            query=query,
            kind=kind,
            mode=mode,
            limit=limit,
            fallback_query=self._get_task(),
        )
