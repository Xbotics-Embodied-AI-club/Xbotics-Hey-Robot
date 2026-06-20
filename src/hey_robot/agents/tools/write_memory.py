from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import (
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)


@tool_parameters(
    tool_parameters_schema(
        kind=StringSchema(
            "Memory kind to write",
            enum=[
                "event",
                "entity",
                "place",
                "location",
                "skill_experience",
                "task_result",
                "user_preference",
                "scene_anchor",
                "task_lesson",
            ],
        ),
        summary=StringSchema("What to remember", nullable=True),
        name=StringSchema("Entity, place, event, or task-result name", nullable=True),
        value=StringSchema("Preference value for user_preference kind", nullable=True),
        location=StringSchema("Where the entity is", nullable=True),
        entity_type=StringSchema("Object, tool, person, etc.", nullable=True),
        confidence=NumberSchema(
            1.0, description="Confidence 0-1", minimum=0, maximum=1, nullable=True
        ),
        skill_name=StringSchema("Skill name for skill_experience kind", nullable=True),
        success=BooleanSchema(
            description="Whether the skill or task succeeded", nullable=True
        ),
        object_name=StringSchema("Object involved in skill", nullable=True),
        failure_mode=StringSchema("Failure mode if any", nullable=True),
        recovery_hint=StringSchema("Recovery hint", nullable=True),
        verification_summary=StringSchema("What verification found", nullable=True),
        duration_sec=NumberSchema(
            0.0, description="How long the skill took", nullable=True
        ),
        attributes=StringSchema("JSON string of extra attributes", nullable=True),
        required=["kind"],
    )
)
class WriteMemoryTool(Tool):
    name = "write_memory"
    description = "Write long-term memory."
    safety_level = "memory_write"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._memory = ctx.memory
        self._get_task = ctx._get_task

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(
        self,
        kind: str,
        summary: str | None = None,
        name: str | None = None,
        value: str | None = None,
        location: str | None = None,
        entity_type: str | None = None,
        confidence: float | None = None,
        skill_name: str | None = None,
        success: bool | None = None,
        object_name: str | None = None,
        failure_mode: str | None = None,
        recovery_hint: str | None = None,
        verification_summary: str | None = None,
        duration_sec: float | None = None,
        attributes: str | None = None,
    ) -> str:
        return self._memory.write(
            kind=kind,
            summary=summary,
            name=name,
            value=value,
            location=location,
            entity_type=entity_type,
            confidence=confidence,
            skill_name=skill_name,
            success=success,
            object_name=object_name,
            failure_mode=failure_mode,
            recovery_hint=recovery_hint,
            verification_summary=verification_summary,
            duration_sec=duration_sec,
            attributes=attributes,
            turn_context=self._ctx.turn_context,
            task=self._get_task(),
        )
