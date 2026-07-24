from __future__ import annotations

import pytest

from hey_robot.cognition.runtime.agent_task_store import AgentTask, AgentTaskStep
from hey_robot.cognition.runtime.completion_verifier import TaskCompletionVerifier
from hey_robot.cognition.tools.skill_tools import SkillCallProposal
from hey_robot.model import ModelResponse, ModelToolCall
from hey_robot.protocol import ToolOutcome


class _Model:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-1",
        session_key="session-1",
        robot_id="sim_robot",
        objective="进入前面的门",
        ui_summary="进入前面的门",
        status="active",
        channel=None,
        chat_id=None,
        sender_id=None,
        user_id=None,
        agent_id=None,
        episode_id=None,
        created_at=1.0,
        updated_at=1.0,
        step_count=1,
        continuation_count=0,
        deadline_at=None,
        last_error=None,
        final_recap=None,
    )


def _step() -> AgentTaskStep:
    return AgentTaskStep(
        step_id="step-1",
        task_id="task-1",
        sequence=1,
        proposal=SkillCallProposal(
            "observation", "inspect_scene", "确认是否进入", {"question": "在哪里"}
        ),
        outcome=ToolOutcome("completed", "机器人前方仍有两个门洞。"),
        started_at=1.0,
        completed_at=2.0,
        evidence_ids=("step:step-1",),
    )


@pytest.mark.asyncio
async def test_completion_verifier_rejects_unsupported_world_state() -> None:
    model = _Model(
        ModelResponse(
            tool_calls=[
                ModelToolCall(
                    "verdict-1",
                    "reject_task_completion",
                    {"reason": "观察只证明门仍在前方，不能证明已经进入。"},
                )
            ]
        )
    )
    verifier = TaskCompletionVerifier(model)

    verdict = await verifier.verify(
        _task(), "已经进入门内。", (_step(),), ("step:step-1",)
    )

    assert not verdict.accepted
    assert "不能证明已经进入" in verdict.reason
    request = model.calls[0]
    assert {tool["function"]["name"] for tool in request["tools"]} == {
        "accept_task_completion",
        "reject_task_completion",
    }
    assert "机器人前方仍有两个门洞" in request["messages"][1].content


@pytest.mark.asyncio
async def test_completion_verifier_fails_closed_without_structured_verdict() -> None:
    verifier = TaskCompletionVerifier(_Model(ModelResponse(content="看起来完成了")))

    verdict = await verifier.verify(
        _task(), "已经进入门内。", (_step(),), ("step:step-1",)
    )

    assert not verdict.accepted
    assert "结构化结论" in verdict.reason
