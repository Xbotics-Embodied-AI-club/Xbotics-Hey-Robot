from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from hey_robot.agents.context import RobotContextBuilder
from hey_robot.agents.core import RobotAgentCore
from hey_robot.agents.injection import RobotTurnInjector
from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.agents.task_runtime import TaskRunManager
from hey_robot.agents.turn_policy import RobotTurnPolicy
from hey_robot.agents.types import AgentCoreResult, AgentTurnInput, RobotSnapshot
from hey_robot.episode import EpisodeRecord
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import UserTurn

logger = HeyRobotLogger(name="agent.loop")


class RobotTurnState(StrEnum):
    RESTORE = "restore"
    BUILD = "build"
    RUN = "run"
    SAVE = "save"
    DONE = "done"


@dataclass
class RobotTurnTraceEntry:
    state: str
    event: str


@dataclass
class ToolTurnContext:
    turn: UserTurn
    snapshot: RobotSnapshot
    history: list[EpisodeRecord] = field(default_factory=list)
    recovery_context: str | None = None
    block_actuation: bool = False
    perception_context: str | None = None
    allowed_tools: set[str] | None = None
    state: RobotTurnState = RobotTurnState.RESTORE
    result: AgentCoreResult | None = None
    trace: list[RobotTurnTraceEntry] = field(default_factory=list)


class RobotAgentLoop:
    """Robot-specific turn loop around RobotAgentCore.

    The loop mirrors nanobot's separation between product turn lifecycle and
    generic tool execution, while keeping robot concerns explicit: skill
    checkpoints, pending corrections, recovery context, and observation state.
    """

    _TRANSITIONS: ClassVar[dict[tuple[RobotTurnState, str], RobotTurnState]] = {
        (RobotTurnState.RESTORE, "ok"): RobotTurnState.BUILD,
        (RobotTurnState.BUILD, "ok"): RobotTurnState.RUN,
        (RobotTurnState.RUN, "ok"): RobotTurnState.SAVE,
        (RobotTurnState.SAVE, "ok"): RobotTurnState.DONE,
    }

    def __init__(
        self,
        core: RobotAgentCore,
        *,
        task_runtime: TaskRunManager,
        context_builder: RobotContextBuilder | None = None,
        injector: RobotTurnInjector | None = None,
        turn_policy: RobotTurnPolicy | None = None,
    ) -> None:
        self.core = core
        self.task_runtime = task_runtime
        self.context_builder = context_builder or RobotContextBuilder()
        self.injector = injector or RobotTurnInjector()
        self.turn_policy = turn_policy or RobotTurnPolicy(core.spec)

    async def run_turn(
        self,
        *,
        turn: UserTurn,
        snapshot: RobotSnapshot,
        history: list[EpisodeRecord],
        recovery_context: str | None = None,
        progress_callback=None,
    ) -> tuple[AgentCoreResult, list[RobotTurnTraceEntry]]:
        ctx = ToolTurnContext(
            turn=turn,
            snapshot=snapshot,
            history=history,
            recovery_context=recovery_context,
        )
        while ctx.state is not RobotTurnState.DONE:
            handler = getattr(self, f"_state_{ctx.state.value}")
            event = await handler(ctx)
            if progress_callback is not None:
                await progress_callback(
                    RobotAgentProgress(
                        phase=ctx.state.value,
                        summary=f"agent turn state {ctx.state.value}: {event}",
                        episode_id=ctx.turn.envelope.episode_id,
                        agent_id=ctx.turn.envelope.agent_id,
                        robot_id=ctx.turn.envelope.robot_id,
                        trace_id=ctx.turn.envelope.trace_id,
                    )
                )
            ctx.trace.append(RobotTurnTraceEntry(state=ctx.state.value, event=event))
            next_state = self._TRANSITIONS.get((ctx.state, event))
            if next_state is None:
                raise RuntimeError(f"no transition from {ctx.state} on {event!r}")
            ctx.state = next_state
        assert ctx.result is not None
        return ctx.result, ctx.trace

    async def _state_restore(self, ctx: ToolTurnContext) -> str:
        self.task_runtime.mark_restore(ctx.turn)
        return "ok"

    async def _state_build(self, ctx: ToolTurnContext) -> str:
        built = self.task_runtime.build_turn(
            turn=ctx.turn,
            snapshot=ctx.snapshot,
            history=ctx.history,
            recovery_context=ctx.recovery_context,
            context_builder=self.context_builder,
            injector=self.injector,
        )
        ctx.turn = built.turn
        ctx.recovery_context = built.recovery_context
        ctx.block_actuation = built.block_actuation
        ctx.turn.metadata.update({"_agent_context": built.metadata})
        ctx.result = AgentCoreResult(
            metadata={
                "_memory_context": built.memory_context,
                "_recovery_context": built.recovery_context,
            }
        )
        policy = self.turn_policy.build(
            AgentTurnInput(
                turn=ctx.turn,
                snapshot=ctx.snapshot,
                memory_context=built.memory_context,
                recovery_context=built.recovery_context,
                block_actuation=built.block_actuation,
            )
        )
        ctx.allowed_tools = policy.allowed_tools
        ctx.perception_context = await self.turn_policy.collect_perception_context(
            core=self.core,
            payload=AgentTurnInput(
                turn=ctx.turn,
                snapshot=ctx.snapshot,
                memory_context=built.memory_context,
                recovery_context=built.recovery_context,
                block_actuation=built.block_actuation,
            ),
            policy=policy,
        )
        return "ok"

    async def _state_run(self, ctx: ToolTurnContext) -> str:
        assert ctx.result is not None
        memory_context = ctx.result.metadata.get("_memory_context")
        recovery_context = ctx.result.metadata.get("_recovery_context")
        result = await self.core.handle_turn(
            AgentTurnInput(
                turn=ctx.turn,
                snapshot=ctx.snapshot,
                memory_context=memory_context,
                recovery_context=recovery_context,
                block_actuation=ctx.block_actuation,
                perception_context=ctx.perception_context,
                allowed_tools=ctx.allowed_tools,
            )
        )
        ctx.result = result
        skill_id = result.metadata.get("skill_id")
        logger.info(
            f"observe_turn_result 调用：tool={result.tool} "
            f"task_finished={result.task_finished} "
            f"reply_len={len(result.reply_text or '')} "
            f"skill_id={skill_id}"
        )
        self.task_runtime.observe_turn_result(
            turn=ctx.turn,
            result_tool=result.tool,
            reply_text=result.reply_text,
            task_finished=result.task_finished,
            skill_id=str(skill_id) if skill_id else None,
            last_observation_frame=ctx.snapshot.observation.frame_id
            if ctx.snapshot.observation
            else None,
        )
        return "ok"

    async def _state_save(self, ctx: ToolTurnContext) -> str:
        self.task_runtime.save_turn(
            episode_id=ctx.turn.envelope.episode_id,
            task_finished=bool(ctx.result and ctx.result.task_finished),
        )
        self.task_runtime.flush_episode_state(
            episode_id=ctx.turn.envelope.episode_id,
            status=ctx.snapshot.status,
            observation=ctx.snapshot.observation,
        )
        return "ok"
