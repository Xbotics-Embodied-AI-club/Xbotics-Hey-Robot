from __future__ import annotations

import asyncio
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
from hey_robot.cognition.tools.models import HarnessToolCall, PhysicalToolCall
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
        return AgentTurnResult("returned", "adjusted", "model_returned")


class _Tools:
    names = frozenset({"inspect_scene", "move_base"})


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

    async def execute(self, **_kwargs):
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
    name = proposal.name
    return AgentTurnResult(
        "action_proposed",
        None,
        "stop_slice",
        (AgentToolCallRecord(call_id, name, dict(getattr(proposal, "arguments", {}))),),
        proposal,
    )


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
async def test_agent_continues_from_tool_outcome_until_complete(tmp_path) -> None:
    observe = PhysicalToolCall("inspect_scene", {})
    runner = _Runner(
        [
            _decision("observe-1", observe),
            AgentTurnResult("returned", "done", "model_returned"),
        ]
    )
    executions = [
        ToolExecution("continue", ToolOutcome("completed", "seen"), observe),
    ]
    agent, tasks, conversations = _agent(tmp_path, runner, executions)

    result = await agent.prompt(
        AgentCommand("session-1", "turn-1", Envelope(robot_id="sim_robot"), "inspect")
    )

    assert result.status == "completed"
    assert result.text == "done"
    assert len(runner.requests) == 2
    assert runner.requests[1].messages[-1].role == "tool"
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
    agent, tasks, conversations = _agent(tmp_path, runner, [])  # type: ignore[arg-type]
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
async def test_active_task_does_not_prevent_natural_text_stop(tmp_path) -> None:
    returned = AgentTurnResult("returned", "still working", "model_returned")
    runner = _Runner([returned])
    agent, tasks, conversations = _agent(tmp_path, runner, [])
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

    assert result.status == "completed"
    assert result.text == "still working"
    assert len(runner.requests) == 1
    assert tasks.task(task.task_id).status == "completed"  # type: ignore[union-attr]
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
async def test_startup_resumes_terminal_undeliberated_step(tmp_path) -> None:
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

    assert len(resume_agent.triggers) == 1
    assert resume_agent.triggers[0].source == "startup_recovery"
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
