from __future__ import annotations

from types import SimpleNamespace

import pytest

from hey_robot.cognition.autonomous_agent_service import AutonomousAgentService
from hey_robot.cognition.runtime.agent_runner import (
    AgentToolCallRecord,
    AgentTurnResult,
)
from hey_robot.cognition.tools.robot import CompleteTaskProposal
from hey_robot.protocol import ActionProposal, Envelope, ToolOutcome
from hey_robot.providers import ReasoningMessage


class _TaskRuntime:
    hard_max_skills = 24
    hard_max_continuations = 12
    hard_max_wall_time_sec = 3600.0


class _Config:
    agent_runtime = _TaskRuntime()


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


@pytest.mark.asyncio
async def test_conversation_loop_continues_after_observation_failure() -> None:
    observe = ActionProposal(
        "observation", "inspect_scene", "check ahead", {"question": "check ahead"}
    )
    move = ActionProposal(
        "skill", "move_base", "move forward", {"direction": "forward"}
    )
    service = object.__new__(AutonomousAgentService)
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "observe-1", "request_observation", observe.arguments
                    ),
                ),
                observe,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "move-1",
                        "request_skill",
                        {"skill": "move_base", "slots": move.arguments},
                    ),
                ),
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
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "move-1",
                        "request_skill",
                        {"skill": "move_base", "slots": move.arguments},
                    ),
                ),
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
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "move-1",
                        "request_skill",
                        {"skill": "move_base", "slots": move.arguments},
                    ),
                ),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (
                    AgentToolCallRecord(
                        "observe-1", "request_observation", observe.arguments
                    ),
                ),
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
    service.runner = _Runner(
        [
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("move-1", "request_skill", {}),),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-1", "request_observation", {}),),
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
                (AgentToolCallRecord("move-2", "request_skill", {}),),
                move,
            ),
            AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (AgentToolCallRecord("observe-2", "request_observation", {}),),
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
