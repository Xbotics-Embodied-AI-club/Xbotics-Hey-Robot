from __future__ import annotations

from types import SimpleNamespace

import pytest

from hey_robot.cognition.autonomous_agent_service import AutonomousAgentService
from hey_robot.cognition.runtime.agent_runner import (
    AgentToolCallRecord,
    AgentTurnResult,
)
from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.robot import CompleteTaskProposal, ControlTaskProposal
from hey_robot.protocol import ActionProposal, Envelope, ToolOutcome
from hey_robot.protocol.messages import to_payload
from hey_robot.providers import ReasoningMessage
from hey_robot.skills.models import SkillEvent, SkillResult


class _TaskRuntime:
    hard_max_skills = 24
    hard_max_continuations = 12
    hard_max_wall_time_sec = 3600.0


class _Config:
    agent_runtime = _TaskRuntime()


_TOOL_NAMES = frozenset({"inspect_scene", "move_base", "complete_task", "control_task"})


class _Tasks:
    def __init__(self) -> None:
        self.current = None
        self.steps = []
        self.completed = []

    def active_task(self, _session_key):
        return self.current

    def create_task(self, **kwargs):
        self.current = SimpleNamespace(
            task_id="task-1",
            objective=kwargs["objective"],
            step_count=0,
            continuation_count=0,
            deadline_at=kwargs.get("deadline_at"),
        )
        return self.current

    def add_step(self, _task_id, proposal, outcome):
        step = SimpleNamespace(
            step_id=f"step-{len(self.steps) + 1}",
            proposal=proposal,
            outcome=outcome,
            evidence_ids=(f"step:step-{len(self.steps) + 1}",),
        )
        self.steps.append(step)
        self.current.step_count = len(self.steps)
        return step

    def recent_steps(self, _task_id, limit=12):
        return tuple(self.steps[-limit:])

    def complete_task(self, task_id, *, recap, evidence_ids):
        self.completed.append((task_id, recap, evidence_ids))
        self.current = None
        return SimpleNamespace(accepted=True)

    def check_completion(self, _task_id, _evidence_ids):
        return SimpleNamespace(accepted=True)

    def control_task(self, *_args: object, **_kwargs: object) -> None:
        self.completed.append(_args)
        self.current = None


def test_duplicate_observation_gate_stops_third_caption_on_same_frame() -> None:
    observe = ActionProposal(
        "observation", "inspect_scene", "find kettle", {"question": "find kettle"}
    )
    steps = [
        SimpleNamespace(
            proposal=observe,
            outcome=ToolOutcome("completed", "seen", data={"frame_id": 7}),
        ),
        SimpleNamespace(
            proposal=observe,
            outcome=ToolOutcome("completed", "seen again", data={"frame_id": 7}),
        ),
    ]
    service = object.__new__(AutonomousAgentService)
    service.tasks = SimpleNamespace(recent_steps=lambda _task_id, limit: steps[-limit:])

    outcome = service._duplicate_observation_gate(
        SimpleNamespace(task_id="task-1"), observe
    )

    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.retryable is True
    assert outcome.data["failure_mode"] == "duplicate_observation"

    def continue_task(self, *_args: object, **_kwargs: object) -> None:
        self.current.continuation_count += 1


class _CompletionVerifier:
    def __init__(self, accepted: bool | list[bool] = True) -> None:
        values = accepted if isinstance(accepted, list) else [accepted]
        self._accepted = iter(values)
        self.calls = []

    async def verify(self, task, recap, steps, evidence_ids):
        self.calls.append((task, recap, steps, evidence_ids))
        accepted = next(self._accepted)
        return SimpleNamespace(
            accepted=accepted,
            reason="证据支持任务完成。" if accepted else "证据不足。",
        )


class _Runner:
    def __init__(self, results: list[AgentTurnResult]) -> None:
        self.results = iter(results)
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return next(self.results)


class _Execution:
    def __init__(self, outcomes: list[ToolOutcome]) -> None:
        self.outcomes = iter(outcomes)
        self.proposals = []

    async def execute(self, proposal, _envelope, _session_key):
        self.proposals.append(proposal)
        return next(self.outcomes)


class _Templates:
    def render(self, _name, **kwargs):
        return "\n".join(str(value) for value in kwargs.values())


class _Entities:
    def context(self, _robot_id):
        return "entities: none"


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload):
        self.published.append((topic, payload))


class _SkillClient:
    def __init__(self):
        self.cancelled = []

    async def cancel(self, run_id, *, reason):
        self.cancelled.append((run_id, reason))


class _Coordinator:
    def __init__(self, store, outcome):
        self.store = store
        self.outcome = outcome

    def apply(self, event):
        return self.store.resolve_pending_step(
            event.run_id,
            outcome=self.outcome,
            status=event.phase,
            event_sequence=event.sequence,
        )


@pytest.mark.asyncio
async def test_conversation_loop_continues_after_observation_failure() -> None:
    observe = ActionProposal(
        "observation", "inspect_scene", "check ahead", {"question": "check ahead"}
    )
    move = ActionProposal(
        "skill", "move_base", "move forward", {"direction": "forward"}
    )
    service = object.__new__(AutonomousAgentService)
    service.tools = SimpleNamespace(names=_TOOL_NAMES)
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-1", "inspect_scene", observe.arguments),),
                observe,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-1", "move_base", move.arguments),),
                move,
            ),
            AgentTurnResult("returned", "已向前移动。", "model_returned"),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "complete-1",
                        "complete_task",
                        {"recap": "已向前移动。", "evidence_ids": ["step:step-2"]},
                    ),
                ),
                CompleteTaskProposal("已向前移动。", ("step:step-2",)),
            ),
        ]
    )
    service.execution = _Execution(
        [
            ToolOutcome("failed", "scene recognition unavailable", retryable=True),
            ToolOutcome("completed", "Base motion completed."),
        ]
    )
    service.config = _Config()
    service.tasks = _Tasks()
    service.completion_verifier = _CompletionVerifier()

    text = await service._run_conversation_loop(
        [ReasoningMessage("user", "往前走走")],
        Envelope(robot_id="sim_robot"),
        "session-1",
        "turn-1",
        "往前走走",
    )

    assert text == "已向前移动。"
    assert [item.skill_name for item in service.execution.proposals] == [
        "inspect_scene",
        "move_base",
    ]
    second_messages = service.runner.requests[1].messages
    assert second_messages[-1].role == "tool"
    assert "这次观察只更新证据" in second_messages[-1].content
    assert "active_task id=task-1; objective=往前走走" in second_messages[-1].content
    completion_messages = service.runner.requests[3].messages
    assert completion_messages[-2].role == "assistant"
    assert completion_messages[-2].content == "已向前移动。"
    assert completion_messages[-1].role == "user"
    assert "继续当前 active task" in completion_messages[-1].content


@pytest.mark.asyncio
async def test_conversation_loop_never_finalizes_pending_robot_outcome() -> None:
    move = ActionProposal(
        "skill", "move_base", "move forward", {"direction": "forward"}
    )
    service = object.__new__(AutonomousAgentService)
    service.tools = SimpleNamespace(names=_TOOL_NAMES)
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-1", "move_base", move.arguments),),
                move,
            ),
        ]
    )
    service.execution = _Execution(
        [
            ToolOutcome(
                "waiting",
                "操作仍在执行，等待机器人返回结果。",
                retryable=True,
            ),
        ]
    )
    service.config = _Config()
    service.tasks = _Tasks()
    service.completion_verifier = _CompletionVerifier()

    text = await service._run_conversation_loop(
        [ReasoningMessage("user", "往前走走")],
        Envelope(robot_id="sim_robot"),
        "session-1",
        "turn-1",
        "往前走走",
    )

    assert text == "这次操作没有完成：机器人还没有返回最终执行结果。"
    assert "等待机器人返回结果" not in text


@pytest.mark.asyncio
async def test_conversation_loop_tracks_every_robot_step_in_one_task() -> None:
    move = ActionProposal(
        "skill", "move_base", "move forward", {"direction": "forward"}
    )
    observe = ActionProposal(
        "observation", "inspect_scene", "check result", {"question": "check result"}
    )
    service = object.__new__(AutonomousAgentService)
    service.tools = SimpleNamespace(names=_TOOL_NAMES)
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-1", "move_base", move.arguments),),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-1", "inspect_scene", observe.arguments),),
                observe,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "complete-1",
                        "complete_task",
                        {
                            "recap": "已进入门内并完成观察。",
                            "evidence_ids": ["step:step-2"],
                        },
                    ),
                ),
                CompleteTaskProposal("已进入门内并完成观察。", ("step:step-2",)),
            ),
        ]
    )
    service.execution = _Execution(
        [
            ToolOutcome("completed", "Base motion completed."),
            ToolOutcome("completed", "已经看到门内环境。"),
        ]
    )
    service.config = _Config()
    tasks = _Tasks()
    service.tasks = tasks
    service.completion_verifier = _CompletionVerifier()

    text = await service._run_conversation_loop(
        [ReasoningMessage("user", "往前走走")],
        Envelope(robot_id="sim_robot"),
        "session-1",
        "turn-1",
        "进入门里并观察里面有什么",
    )

    assert text == "已进入门内并完成观察。"
    assert [item.skill_name for item in service.execution.proposals] == [
        "move_base",
        "inspect_scene",
    ]
    assert len(tasks.steps) == 2
    assert tasks.completed == [("task-1", "已进入门内并完成观察。", ("step:step-2",))]


@pytest.mark.asyncio
async def test_rejected_completion_keeps_driving_the_same_task() -> None:
    move = ActionProposal(
        "skill", "move_base", "继续进入门内", {"direction": "forward"}
    )
    observe = ActionProposal(
        "observation", "inspect_scene", "确认是否进入", {"question": "在哪里"}
    )
    complete = CompleteTaskProposal("已经进入门内。", ("step:step-2",))
    service = object.__new__(AutonomousAgentService)
    service.tools = SimpleNamespace(names=_TOOL_NAMES)
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-1", "move_base", {}),),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-1", "inspect_scene", {}),),
                observe,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("complete-1", "complete_task", {}),),
                complete,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-2", "move_base", {}),),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-2", "inspect_scene", {}),),
                observe,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("complete-2", "complete_task", {}),),
                CompleteTaskProposal("已经进入门内。", ("step:step-4",)),
            ),
        ]
    )
    service.execution = _Execution(
        [
            ToolOutcome("completed", "前进30厘米。"),
            ToolOutcome("completed", "门仍在前方。"),
            ToolOutcome("completed", "再次前进30厘米。"),
            ToolOutcome("completed", "机器人已经位于门内。"),
        ]
    )
    service.config = _Config()
    tasks = _Tasks()
    service.tasks = tasks
    verifier = _CompletionVerifier([False, True])
    service.completion_verifier = verifier

    text = await service._run_conversation_loop(
        [ReasoningMessage("user", "进入前面的门")],
        Envelope(robot_id="sim_robot"),
        "session-1",
        "turn-1",
        "进入前面的门",
    )

    assert text == "已经进入门内。"
    assert len(service.execution.proposals) == 4
    assert len(verifier.calls) == 2
    assert tasks.completed[-1][1] == "已经进入门内。"


@pytest.mark.asyncio
async def test_terminal_skill_event_resumes_active_task_and_publishes_result(
    tmp_path,
) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(
            channel="web",
            chat_id="chat-1",
            sender_id="sender-1",
            user_id="user-1",
            agent_id="agent-1",
            robot_id="sim_robot",
        ),
        objective="检查桌面",
    )
    proposal = ActionProposal(
        "observation", "inspect_scene", "检查桌面", {"question": "桌面上有什么"}
    )
    pending = store.add_pending_step(
        task.task_id,
        proposal,
        run_id="run-1",
        tool_call_id="observe-1",
    )
    outcome = ToolOutcome(
        "completed",
        "桌面上有一个杯子。",
        data={"evidence_ids": ["scene:cup"]},
        operation_id="run-1",
    )
    complete = CompleteTaskProposal("桌面上有一个杯子。", ("scene:cup",))
    service = object.__new__(AutonomousAgentService)
    service.tasks = store
    service.conversations = conversations
    service.templates = _Templates()
    service.entities = _Entities()
    service.tools = SimpleNamespace(names=_TOOL_NAMES, instructions="tools")
    service.config = _Config()
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("complete-1", "complete_task", {}),),
                complete,
            )
        ]
    )
    service.completion_verifier = _CompletionVerifier()
    service.task_coordinator = _Coordinator(store, outcome)
    service._session_locks = {}
    service.bus = _Bus()
    service.topics = SimpleNamespace(conversation_result="conversation.result")

    await service._on_skill_event(
        "skill.run.event",
        to_payload(
            SkillEvent(
                envelope=Envelope(robot_id="sim_robot"),
                run_id=pending.run_id or "",
                sequence=3,
                name="inspect_scene",
                phase="completed",
                timestamp=0.0,
                result=SkillResult(True, "桌面上有一个杯子。", "completed"),
            )
        ),
    )

    assert store.task(task.task_id).status == "completed"  # type: ignore[union-attr]
    assert service.runner.requests[0].run_id == "skill_event_run-1_3"
    assert (
        "机器人 Skill 已返回终态事件" in service.runner.requests[0].messages[-2].content
    )
    assert service.bus.published[0][0] == "conversation.result"
    payload = service.bus.published[0][1]
    assert payload["text"] == "桌面上有一个杯子。"
    assert payload["interaction_id"] == "run-1"
    assert payload["envelope"]["channel"] == "web"
    assert payload["envelope"]["chat_id"] == "chat-1"
    conversations.close()
    store.close()


@pytest.mark.asyncio
async def test_control_task_cancels_active_skill_runs(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="移动到桌边",
    )
    store.add_pending_step(
        task.task_id,
        ActionProposal("skill", "move_base", "移动到桌边", {}),
        run_id="run-active",
        tool_call_id="move-1",
    )
    service = object.__new__(AutonomousAgentService)
    service.tasks = store
    service.skill_client = _SkillClient()

    text = await service._control_task(
        ControlTaskProposal("cancel", "用户取消任务。"), "session-1"
    )

    assert text == "用户取消任务。"
    assert service.skill_client.cancelled == [("run-active", "用户取消任务。")]
    assert store.task(task.task_id).status == "cancelled"  # type: ignore[union-attr]
    store.close()
