from __future__ import annotations

import asyncio
import sqlite3
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest

from hey_robot.cognition.autonomous_agent_service import AutonomousAgentService
from hey_robot.cognition.runtime.agent import Agent, AgentCommand, ResumeTrigger
from hey_robot.cognition.runtime.agent_context import AgentContextBuilder
from hey_robot.cognition.runtime.agent_runner import (
    AgentToolCallRecord,
    AgentTurnResult,
)
from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.runtime.task_coordinator import TaskCoordinator
from hey_robot.cognition.tools.executor import AgentToolExecutor, ToolExecution
from hey_robot.cognition.tools.models import (
    AgentResponseCall,
    HarnessToolCall,
    PhysicalToolCall,
)
from hey_robot.protocol import AgentControl, Envelope, ToolOutcome
from hey_robot.skills.models import SkillEvent, SkillResult


class _TaskRuntime:
    hard_max_skills = 24
    hard_max_wall_time_sec = 3600.0


class _Config:
    agent_runtime = _TaskRuntime()

    @staticmethod
    def default_robot_id(_agent_id: str | None) -> str:
        return "sim_robot"


class _Runner:
    def __init__(self, results: list[AgentTurnResult]) -> None:
        self.results = iter(results)
        self.requests = []

    async def run(self, request, *, on_text_delta=None):
        del on_text_delta
        self.requests.append(request)
        return next(self.results)


class _SteerRunner:
    def __init__(self) -> None:
        self.requests = []
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, request, *, on_text_delta=None):
        del on_text_delta
        self.requests.append(request)
        if len(self.requests) == 1:
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        proposal = AgentResponseCall("none", "adjusted")
        return AgentTurnResult(
            "action_proposed",
            None,
            "stop_slice",
            (
                AgentToolCallRecord(
                    "respond-steer",
                    "respond",
                    {"task_state": "none", "message": "adjusted"},
                ),
            ),
            proposal,
        )


class _Tools:
    names = frozenset({"inspect_scene", "move_base", "respond"})


class _Templates:
    def render(self, _name, **kwargs):
        return "\n".join(str(value) for value in kwargs.values())


class _SkillClient:
    def __init__(self):
        self.cancelled = []
        self.emergency_stops = []

    async def cancel(self, run_id, *, reason):
        self.cancelled.append((run_id, reason))

    async def emergency_stop(self, robot_id, *, reason):
        self.emergency_stops.append((robot_id, reason))


class _SubmittingCoordinator:
    def __init__(self, tasks: AgentTaskStore) -> None:
        self.tasks = tasks
        self.proposals = []

    async def submit(
        self,
        *,
        task_id,
        proposal,
        envelope,
        tool_call_id,
        deadline_at,
    ):
        del envelope, deadline_at
        self.proposals.append(proposal)
        return self.tasks.add_pending_step(
            task_id,
            proposal,
            run_id="run-submitted",
            tool_call_id=tool_call_id,
        )


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload):
        self.published.append((topic, payload))


class _InlineExecutor:
    def __init__(self, tasks: AgentTaskStore, executions: list[ToolExecution]):
        self.tasks = tasks
        self.executions = iter(executions)

    async def execute(self, **kwargs):
        execution = next(self.executions)
        if isinstance(execution.proposal, PhysicalToolCall):
            task = self.tasks.active_task("session-1")
            if task is None:
                task = self.tasks.create_task(
                    session_key="session-1",
                    envelope=Envelope(robot_id="sim_robot"),
                    objective="test objective",
                )
            if execution.directive != "wait":
                step = self.tasks.add_step(
                    task.task_id, execution.proposal, execution.outcome
                )
                return dataclass_replace(execution, step=step, task=task)
            return dataclass_replace(execution, task=task)
        if isinstance(execution.proposal, AgentResponseCall):
            task = self.tasks.active_task(kwargs["session_key"])
            if execution.proposal.task_state == "wait":
                if task is None:
                    task = self.tasks.create_task(
                        session_key=kwargs["session_key"],
                        envelope=kwargs["envelope"],
                        objective=kwargs["objective"],
                    )
                return dataclass_replace(execution, task=task)
            if task is not None and execution.proposal.task_state in {
                "complete",
                "cancel",
            }:
                self.tasks.close_task(task.task_id, recap=execution.proposal.message)
                return dataclass_replace(execution, task=self.tasks.task(task.task_id))
        return execution


def dataclass_replace(value, **changes):
    data = value.__dict__ | changes
    return type(value)(**data)


def _agent(
    tmp_path,
    runner: _Runner,
    executions: list[ToolExecution],
) -> tuple[Agent, AgentTaskStore, ConversationStore]:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    context = AgentContextBuilder(_Templates(), conversations, tasks)
    agent = Agent(
        session_key="session-1",
        runner=runner,  # type: ignore[arg-type]
        tools=_Tools(),  # type: ignore[arg-type]
        executor=_InlineExecutor(tasks, executions),  # type: ignore[arg-type]
        context=context,
        tasks=tasks,
        conversations=conversations,
    )
    return agent, tasks, conversations


def _decision(call_id: str, proposal) -> AgentTurnResult:
    if isinstance(proposal, AgentResponseCall):
        name = "respond"
        arguments = {
            "task_state": proposal.task_state,
            "message": proposal.message,
        }
    else:
        name = proposal.name
        arguments = dict(getattr(proposal, "arguments", {}))
    return AgentTurnResult(
        "action_proposed",
        None,
        "stop_slice",
        (AgentToolCallRecord(call_id, name, arguments),),
        proposal,
    )


def test_task_store_persists_one_canonical_envelope(tmp_path) -> None:
    path = tmp_path / "tasks.sqlite3"
    store = AgentTaskStore(path)
    envelope = Envelope(
        trace_id="trace-1",
        account_id="account-1",
        chat_type="group",
        deployment_id="deployment-1",
        robot_id="sim_robot",
    )
    task = store.create_task(
        session_key="session-1",
        envelope=envelope,
        objective="inspect",
    )
    assert store.task_envelope(task.task_id) == envelope
    store.close()

    db = sqlite3.connect(path)
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(sustained_tasks)")}
    tables = {
        str(row[0])
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    db.close()

    assert "envelope_json" in columns
    assert not {
        "robot_id",
        "channel",
        "chat_id",
        "sender_id",
        "user_id",
        "agent_id",
        "episode_id",
    }.intersection(columns)
    assert "task_envelopes" not in tables


def test_task_store_updates_continuation_route_without_changing_task_identity(
    tmp_path,
) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(
            trace_id="trace-1",
            robot_id="sim_robot",
            episode_id="episode-1",
        ),
        interaction_id="turn-1",
        objective="inspect",
    )
    latest = Envelope(
        trace_id="trace-2",
        robot_id="sim_robot",
        episode_id="episode-1",
    )

    store.update_route(task.task_id, envelope=latest, interaction_id="turn-2")

    assert store.task(task.task_id).objective == "inspect"  # type: ignore[union-attr]
    assert store.task_envelope(task.task_id) == latest
    assert store.task_interaction_id(task.task_id) == "turn-2"
    store.close()


def test_task_store_rejects_unversioned_runtime_instead_of_migrating(tmp_path) -> None:
    path = tmp_path / "tasks.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE sustained_tasks (task_id TEXT PRIMARY KEY)")
    db.commit()
    db.close()

    with pytest.raises(RuntimeError, match="archive or reset"):
        AgentTaskStore(path)


@pytest.mark.asyncio
async def test_agent_returns_waiting_after_physical_submit(tmp_path) -> None:
    move = PhysicalToolCall("move_base", {})
    runner = _Runner([_decision("move-1", move)])
    waiting = ToolExecution(
        "wait",
        ToolOutcome("accepted", "submitted", operation_id="run-1"),
        move,
    )
    agent, tasks, conversations = _agent(tmp_path, runner, [waiting])

    result = await agent.prompt(
        AgentCommand("session-1", "turn-1", Envelope(robot_id="sim_robot"), "move")
    )

    assert result.status == "waiting"
    assert result.operation_id == "run-1"
    assert len(runner.requests) == 1
    assert [message.role for message in conversations.recent("session-1")] == ["user"]
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_plain_text_after_tool_outcome_keeps_task_active(tmp_path) -> None:
    observe = PhysicalToolCall("inspect_scene", {})
    response = AgentResponseCall("none", "done")
    runner = _Runner(
        [
            _decision("observe-1", observe),
            AgentTurnResult("returned", "done", "model_returned"),
            _decision("respond-1", response),
        ]
    )
    executions = [
        ToolExecution("continue", ToolOutcome("completed", "seen"), observe),
        ToolExecution(
            "respond",
            ToolOutcome("completed", "done"),
            response,
            final_text="done",
        ),
    ]
    agent, tasks, conversations = _agent(tmp_path, runner, executions)

    result = await agent.prompt(
        AgentCommand("session-1", "turn-1", Envelope(robot_id="sim_robot"), "inspect")
    )

    assert result.status == "responded"
    assert result.text == "done"
    assert len(runner.requests) == 3
    assert runner.requests[1].messages[-1].role == "tool"
    assert (
        runner.requests[2]
        .messages[-1]
        .content.startswith("普通文本不是有效的 Agent 响应")
    )
    assert tasks.active_task("session-1") is not None
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_steer_during_skill_waits_for_safe_point(tmp_path) -> None:
    runner = _Runner([])
    agent, tasks, conversations = _agent(tmp_path, runner, [])
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="move to the desk",
    )
    tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-active",
        tool_call_id="move-1",
    )

    result = await agent.steer(
        AgentCommand(
            "session-1",
            "steer-1",
            Envelope(robot_id="sim_robot"),
            "go to the dining table instead",
        )
    )

    assert result.status == "waiting"
    assert result.operation_id == "run-active"
    assert runner.requests == []
    transcript = conversations.recent("session-1")
    assert [(message.role, message.content) for message in transcript] == [
        ("user", "go to the dining table instead")
    ]
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_steer_during_inference_rebuilds_context(tmp_path) -> None:
    runner = _SteerRunner()
    response = AgentResponseCall("none", "adjusted")
    agent, tasks, conversations = _agent(  # type: ignore[arg-type]
        tmp_path,
        runner,
        [
            ToolExecution(
                "respond",
                ToolOutcome("completed", "adjusted"),
                response,
                final_text="adjusted",
            )
        ],
    )
    first = asyncio.create_task(
        agent.prompt(
            AgentCommand(
                "session-1",
                "turn-1",
                Envelope(robot_id="sim_robot"),
                "inspect the desk",
            )
        )
    )
    await runner.started.wait()

    result = await agent.steer(
        AgentCommand(
            "session-1",
            "steer-1",
            Envelope(robot_id="sim_robot"),
            "inspect the table instead",
        )
    )
    with suppress(asyncio.CancelledError):
        await first

    assert runner.cancelled is True
    assert result.text == "adjusted"
    assert runner.requests[-1].messages[-1].content == "inspect the table instead"
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_plain_text_does_not_implicitly_complete_active_task(tmp_path) -> None:
    returned = AgentTurnResult("returned", "still working", "model_returned")
    response = AgentResponseCall("none", "still working")
    runner = _Runner([returned, _decision("respond-1", response)])
    agent, tasks, conversations = _agent(
        tmp_path,
        runner,
        [
            ToolExecution(
                "respond",
                ToolOutcome("completed", "still working"),
                response,
                final_text="still working",
            )
        ],
    )
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="inspect the whole room",
    )

    result = await agent.prompt(
        AgentCommand(
            "session-1",
            "turn-1",
            Envelope(robot_id="sim_robot"),
            "continue inspecting",
        )
    )

    assert result.status == "responded"
    assert result.text == "still working"
    assert len(runner.requests) == 2
    assert tasks.task(task.task_id).status == "active"  # type: ignore[union-attr]
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_failed_terminal_response_is_not_reported_as_completed(tmp_path) -> None:
    response = AgentResponseCall("complete", "done")
    runner = _Runner([_decision("respond-1", response)])
    execution = ToolExecution(
        "finish",
        ToolOutcome(
            "failed",
            "无法完成任务：当前没有进行中的持续任务。",
            {"failure_mode": "task_not_active"},
        ),
        response,
        final_text="无法完成任务：当前没有进行中的持续任务。",
    )
    agent, tasks, conversations = _agent(tmp_path, runner, [execution])

    result = await agent.prompt(
        AgentCommand(
            "session-1",
            "turn-1",
            Envelope(robot_id="sim_robot"),
            "done",
        )
    )

    assert result.status == "failed"
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_explicit_task_lifecycle_preserves_confirmed_goal_and_step_facts(
    tmp_path,
) -> None:
    wait_for_confirmation = AgentResponseCall("wait", "要开始吗？")
    move = PhysicalToolCall("move_base", {"direction": "forward", "distance_cm": 50})
    wait_for_more = AgentResponseCall("wait", "已完成两步，要继续吗？")
    complete = AgentResponseCall("complete", "目标已完成。")
    runner = _Runner(
        [
            _decision("task-wait-1", wait_for_confirmation),
            _decision("move-1", move),
            _decision("move-2", move),
            _decision("task-wait-2", wait_for_more),
            _decision("task-complete", complete),
        ]
    )
    executions = [
        ToolExecution(
            "respond",
            ToolOutcome("completed", wait_for_confirmation.message),
            wait_for_confirmation,
            final_text=wait_for_confirmation.message,
        ),
        ToolExecution("continue", ToolOutcome("completed", "moved 50cm"), move),
        ToolExecution("continue", ToolOutcome("completed", "moved 50cm"), move),
        ToolExecution(
            "respond",
            ToolOutcome("completed", wait_for_more.message),
            wait_for_more,
            final_text=wait_for_more.message,
        ),
        ToolExecution(
            "finish",
            ToolOutcome("completed", complete.message),
            complete,
            final_text=complete.message,
        ),
    ]
    agent, tasks, conversations = _agent(tmp_path, runner, executions)

    confirmation = await agent.prompt(
        AgentCommand(
            "session-1",
            "turn-1",
            Envelope(robot_id="sim_robot"),
            "往前走1米",
        )
    )
    progress = await agent.prompt(
        AgentCommand(
            "session-1",
            "turn-2",
            Envelope(robot_id="sim_robot"),
            "好",
        )
    )
    task = tasks.active_task("session-1")

    assert confirmation.status == "responded"
    assert progress.status == "responded"
    assert task is not None
    assert task.objective == "往前走1米"
    assert task.step_count == 2
    projection = tasks.projection("session-1")
    assert "objective=往前走1米" in projection
    assert "move_base×2" in projection

    finished = await agent.prompt(
        AgentCommand(
            "session-1",
            "turn-3",
            Envelope(robot_id="sim_robot"),
            "已经完成",
        )
    )

    assert finished.status == "completed"
    assert tasks.active_task("session-1") is None
    assert tasks.projection("session-1") == "当前会话没有进行中的持续任务。"
    conversations.close()
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_creates_task_and_persists_original_proposal(
    tmp_path,
) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    coordinator = _SubmittingCoordinator(tasks)
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        coordinator,  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    proposal = PhysicalToolCall("move_base", {"meters": 0.2})

    execution = await executor.execute(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="reach the table",
        proposal=proposal,
        tool_call_id="call-1",
    )

    assert execution.directive == "wait"
    assert execution.outcome.operation_id == "run-submitted"
    assert execution.step is not None
    assert execution.step.proposal == proposal
    assert coordinator.proposals == [proposal]
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_applies_explicit_wait_and_complete_transitions(
    tmp_path,
) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        _SubmittingCoordinator(tasks),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    envelope = Envelope(
        trace_id="trace-original",
        channel="web",
        account_id="web-account",
        chat_type="web",
        deployment_id="deployment-1",
        robot_id="sim_robot",
    )

    waiting = await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="move five metres",
        proposal=AgentResponseCall("wait", "confirm?"),
        tool_call_id="wait-1",
    )
    task = tasks.active_task("session-1")

    assert waiting.directive == "respond"
    assert task is not None
    assert task.objective == "move five metres"
    assert tasks.task_envelope(task.task_id) == envelope
    tasks.add_step(
        task.task_id,
        PhysicalToolCall("move_base", {"distance_cm": 500}),
        ToolOutcome("completed", "moved five metres"),
    )

    completed = await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="yes",
        proposal=AgentResponseCall("complete", "done"),
        tool_call_id="complete-1",
    )

    assert completed.directive == "finish"
    assert tasks.task(task.task_id).status == "completed"  # type: ignore[union-attr]
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_rejects_none_while_task_is_active(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        _SubmittingCoordinator(tasks),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    envelope = Envelope(robot_id="sim_robot")
    await executor.execute(
        session_key="session-1",
        envelope=envelope,
        interaction_id="turn-1",
        objective="inspect",
        proposal=AgentResponseCall("wait", "working"),
        tool_call_id="wait-1",
    )

    rejected = await executor.execute(
        session_key="session-1",
        envelope=envelope,
        interaction_id="turn-1",
        objective="inspect",
        proposal=AgentResponseCall("none", "done"),
        tool_call_id="respond-1",
    )

    assert rejected.directive == "continue"
    assert rejected.outcome.data["failure_mode"] == "active_task_state_required"
    assert tasks.active_task("session-1") is not None
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_rejects_completion_without_physical_evidence(
    tmp_path,
) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        _SubmittingCoordinator(tasks),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    envelope = Envelope(robot_id="sim_robot")
    await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="move",
        proposal=AgentResponseCall("wait", "confirm?"),
        tool_call_id="wait-1",
    )

    completion = await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="move",
        proposal=AgentResponseCall("complete", "done"),
        tool_call_id="complete-1",
    )

    assert completion.directive == "respond"
    assert completion.outcome.data["failure_mode"] == "task_evidence_missing"
    assert tasks.active_task("session-1") is not None
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_cancels_a_withdrawn_task(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        _SubmittingCoordinator(tasks),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    envelope = Envelope(robot_id="sim_robot")
    await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="move",
        proposal=AgentResponseCall("wait", "confirm?"),
        tool_call_id="wait-1",
    )
    task = tasks.active_task("session-1")
    assert task is not None

    cancelled = await executor.execute(
        session_key="session-1",
        envelope=envelope,
        objective="never mind",
        proposal=AgentResponseCall("cancel", "cancelled"),
        tool_call_id="cancel-1",
    )

    assert cancelled.directive == "finish"
    assert tasks.task(task.task_id).status == "cancelled"  # type: ignore[union-attr]
    tasks.close()


@pytest.mark.asyncio
async def test_tool_executor_runs_nonphysical_harness_tool(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")

    async def handler(arguments):
        return ToolOutcome("completed", "lookup complete", dict(arguments))

    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        SimpleNamespace(),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )
    proposal = HarnessToolCall("lookup", {"query": "cup"}, handler)

    execution = await executor.execute(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="lookup",
        proposal=proposal,
        tool_call_id="lookup-1",
    )

    assert execution.directive == "continue"
    assert execution.outcome.data == {"query": "cup"}
    tasks.close()


@pytest.mark.asyncio
async def test_control_pause_resume_and_cancel_are_durable(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="move",
    )
    tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-1",
        tool_call_id="call-1",
    )
    client = _SkillClient()
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        SimpleNamespace(),  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
    )
    envelope = Envelope(agent_id="main", robot_id="sim_robot")

    await executor.control(
        AgentControl(envelope, "session-1", "pause-1", "pause", "pause")
    )
    assert tasks.current_task("session-1").status == "paused"  # type: ignore[union-attr]
    assert client.cancelled == [("run-1", "pause")]

    tasks.apply_skill_event(
        "run-1",
        outcome=ToolOutcome("failed", "cancelled"),
        status="cancelled",
        event_sequence=1,
    )
    await executor.control(AgentControl(envelope, "session-1", "resume-1", "resume"))
    assert tasks.active_task("session-1") is not None

    await executor.control(
        AgentControl(envelope, "session-1", "cancel-1", "cancel", "cancel")
    )
    assert tasks.current_task("session-1") is None
    tasks.close()


@pytest.mark.asyncio
async def test_resume_waits_for_cancelled_run_terminal(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="move",
    )
    tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-stopping",
        tool_call_id="call-1",
    )
    tasks.pause_task(task.task_id, "pause")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        SimpleNamespace(),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )

    text = await executor.control(
        AgentControl(Envelope(robot_id="sim_robot"), "session-1", "resume-1", "resume")
    )

    assert "仍在停止中" in text
    assert tasks.current_task("session-1").status == "paused"  # type: ignore[union-attr]
    tasks.close()


@pytest.mark.asyncio
async def test_user_resume_continues_without_hard_coded_observation(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="inspect the room",
    )
    tasks.pause_task(task.task_id, "pause")
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        SimpleNamespace(),  # type: ignore[arg-type]
        _SkillClient(),  # type: ignore[arg-type]
    )

    class ResumeAgent:
        trigger = None

        async def resume(self, trigger):
            self.trigger = trigger
            return SimpleNamespace(text="已从最近结果继续。")

    agent = ResumeAgent()

    service = object.__new__(AutonomousAgentService)
    service.tasks = tasks
    service.tool_executor = executor
    service._agent = lambda _session_key: agent
    command = AgentControl(
        Envelope(robot_id="sim_robot"), "session-1", "resume-1", "resume"
    )

    text = await service._resume_task_from_control(command)

    assert text == "已从最近结果继续。"
    assert tasks.active_task("session-1") is not None
    assert agent.trigger.task_id == task.task_id
    assert agent.trigger.source == "user_resume"
    tasks.close()


@pytest.mark.asyncio
async def test_emergency_stop_bypasses_agent_runner(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    tasks.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="move",
    )
    client = _SkillClient()
    executor = AgentToolExecutor(
        _Config(),  # type: ignore[arg-type]
        tasks,
        SimpleNamespace(),  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
    )

    await executor.control(
        AgentControl(
            Envelope(robot_id="sim_robot"),
            "session-1",
            "stop-1",
            "emergency_stop",
            "operator stop",
        )
    )

    assert client.emergency_stops == [("sim_robot", "operator stop")]
    assert tasks.active_task("session-1") is None
    tasks.close()


class _ResumeAgent:
    def __init__(self):
        self.triggers = []

    async def resume(self, trigger: ResumeTrigger):
        self.triggers.append(trigger)
        return SimpleNamespace(text="resumed")


class _WaitingResumeAgent(_ResumeAgent):
    async def resume(self, trigger: ResumeTrigger):
        self.triggers.append(trigger)
        return SimpleNamespace(text="still running", status="waiting")


class _NoReconciliation:
    @staticmethod
    async def reconcile_active_run_results() -> tuple[Any, ...]:
        return ()


@pytest.mark.asyncio
async def test_startup_pauses_terminal_undeliberated_step(tmp_path) -> None:
    path = tmp_path / "tasks.sqlite3"
    original = AgentTaskStore(path)
    task = original.create_task(
        session_key="session-1",
        envelope=Envelope(channel="web", robot_id="sim_robot"),
        objective="inspect",
    )
    pending = original.add_pending_step(
        task.task_id,
        PhysicalToolCall("inspect_scene", {}),
        run_id="run-1",
        tool_call_id="call-1",
    )
    original.apply_skill_event(
        pending.run_id or "",
        outcome=ToolOutcome("completed", "seen"),
        status="completed",
        event_sequence=2,
    )
    original.close()

    restarted = AgentTaskStore(path)
    resume_agent = _ResumeAgent()
    service = object.__new__(AutonomousAgentService)
    service.tasks = restarted
    service.task_coordinator = _NoReconciliation()
    service._agents = {"session-1": resume_agent}
    service.bus = _Bus()
    service.topics = SimpleNamespace(conversation_result="conversation.result")

    await service._recover_tasks()

    assert resume_agent.triggers == []
    recovered = restarted.task(task.task_id)
    assert recovered is not None
    assert recovered.status == "paused"
    assert "避免自动触发新的机器人动作" in (recovered.last_error or "")
    assert service.bus.published
    restarted.close()


@pytest.mark.asyncio
async def test_terminal_event_resumes_once_and_replay_is_ignored(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(channel="web", robot_id="sim_robot"),
        objective="inspect",
    )
    pending = tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("inspect_scene", {}),
        run_id="run-1",
        tool_call_id="call-1",
    )
    coordinator = TaskCoordinator(tasks, SimpleNamespace())  # type: ignore[arg-type]
    service = object.__new__(AutonomousAgentService)
    service.tasks = tasks
    service.task_coordinator = coordinator
    service._agents = {"session-1": _ResumeAgent()}
    service.bus = _Bus()
    service.topics = SimpleNamespace(conversation_result="conversation.result")
    event = SkillEvent(
        Envelope(robot_id="sim_robot"),
        pending.run_id or "",
        2,
        "inspect_scene",
        "completed",
        0.0,
        result=SkillResult(True, "seen", "completed"),
    )

    await service._handle_skill_event(event)
    await service._handle_skill_event(event)

    resume_agent = service._agents["session-1"]
    assert len(resume_agent.triggers) == 1
    assert service.bus.published[-1][1]["text"] == "resumed"
    tasks.close()


@pytest.mark.asyncio
async def test_chained_physical_wait_is_not_published_as_final(tmp_path) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(channel="web", robot_id="sim_robot"),
        objective="move twice",
    )
    pending = tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-1",
        tool_call_id="call-1",
    )
    service = object.__new__(AutonomousAgentService)
    service.tasks = tasks
    service.task_coordinator = TaskCoordinator(tasks, SimpleNamespace())  # type: ignore[arg-type]
    service._agents = {"session-1": _WaitingResumeAgent()}
    service.bus = _Bus()
    service.topics = SimpleNamespace(conversation_result="conversation.result")
    event = SkillEvent(
        Envelope(robot_id="sim_robot"),
        pending.run_id or "",
        2,
        "move_base",
        "completed",
        0.0,
        result=SkillResult(True, "moved", "completed"),
    )

    await service._handle_skill_event(event)

    assert service.bus.published[-1][1]["text"] == "still running"
    assert service.bus.published[-1][1]["final"] is False
    tasks.close()


@pytest.mark.asyncio
async def test_environment_done_publishes_without_agent_resume(
    tmp_path,
) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session-1",
        envelope=Envelope(channel="web", robot_id="sim_robot"),
        objective="finish",
    )
    pending = tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-1",
        tool_call_id="call-1",
    )
    service = object.__new__(AutonomousAgentService)
    service.tasks = tasks
    service.task_coordinator = TaskCoordinator(tasks, SimpleNamespace())  # type: ignore[arg-type]
    service._agents = {"session-1": _ResumeAgent()}
    service.bus = _Bus()
    service.topics = SimpleNamespace(conversation_result="conversation.result")
    event = SkillEvent(
        Envelope(robot_id="sim_robot"),
        pending.run_id or "",
        2,
        "move_base",
        "completed",
        0.0,
        result=SkillResult(
            True,
            "environment complete",
            "completed",
            data={"termination_reason": "environment_done"},
        ),
    )

    await service._handle_skill_event(event)

    assert tasks.task(task.task_id).status == "completed"  # type: ignore[union-attr]
    assert service.bus.published[-1][1]["text"] == "environment complete"
    assert not service._agents["session-1"].triggers
    tasks.close()
