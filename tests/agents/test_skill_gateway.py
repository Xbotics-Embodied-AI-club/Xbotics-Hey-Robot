from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from hey_robot.agents.skill_gateway import SkillGateway, SkillGatewayRequest
from hey_robot.config import AgentSpec
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.skills.base import SkillCatalog, SkillSpec


@dataclass
class FakeGatewayIO:
    skills: list[SkillIntent] = field(default_factory=list)
    on_submit: Callable[[SkillIntent], None] | None = None

    async def submit_skill(self, skill: SkillIntent) -> None:
        self.skills.append(skill)
        if self.on_submit is not None:
            self.on_submit(skill)


def _catalog() -> SkillCatalog:
    return SkillCatalog(
        (
            SkillSpec(
                name="inspect_scene",
                description="Inspect the current scene",
                safety_level="observe",
                feedback_mode="status",
            ),
            SkillSpec(
                name="set_gripper",
                description="Open or close the gripper",
                safety_level="actuate",
                feedback_mode="status",
            ),
        )
    )


def _gateway(
    *,
    io: FakeGatewayIO | None = None,
    pending_skills: dict[str, asyncio.Future[str]] | None = None,
    on_submit: Callable[[SkillIntent], None] | None = None,
    recovery_required: Callable[[], bool] | None = None,
) -> SkillGateway:
    shared_pending_skills = pending_skills if pending_skills is not None else {}
    return SkillGateway(
        io=io or FakeGatewayIO(),
        spec=AgentSpec(type="robot_agent", settings={}),
        skill_catalog=_catalog(),
        runtime_state=type("RuntimeState", (), {})(),
        pending_skills=shared_pending_skills,
        current_envelope=lambda: Envelope(
            agent_id="main",
            robot_id="mock0",
            episode_id="ep1",
            channel="web",
        ),
        get_task=lambda: "inspect the desk",
        on_submit=on_submit,
        recovery_required=recovery_required,
    )


def test_skill_gateway_submit_direct_notifies_and_submits() -> None:
    io = FakeGatewayIO()
    submitted: list[SkillIntent] = []
    gateway = _gateway(io=io, on_submit=submitted.append)

    intent = asyncio.run(
        gateway.submit_direct(
            objective="pick up the block",
            slots={"objective": "pick up the block", "interrupt": False},
            metadata={"source": "direct"},
        )
    )

    assert io.skills == [intent]
    assert submitted == [intent]
    assert intent.name == ""
    assert intent.objective == "pick up the block"
    assert intent.metadata["source"] == "direct"


def test_skill_gateway_build_interrupt_intent_preserves_active_skill_id() -> None:
    intent = SkillGateway.build_interrupt_intent(
        envelope=Envelope(agent_id="main", robot_id="mock0", episode_id="ep1"),
        active_skill_id="skill_active",
        objective="stop that",
        metadata={"source": "busy_turn"},
    )

    assert intent.skill_id == "skill_active"
    assert intent.name == "interrupt"
    assert intent.interrupt is True
    assert intent.objective == "stop that"
    assert intent.metadata["source"] == "busy_turn"


@pytest.mark.parametrize(
    ("wait_policy", "expected_prefix"),
    [
        ("wait_acceptance", "skill_accepted:"),
        ("return_handle", "skill_submitted:"),
    ],
)
def test_skill_gateway_nonblocking_wait_policies(
    wait_policy: str, expected_prefix: str
) -> None:
    gateway = _gateway()

    result = asyncio.run(
        gateway.submit(
            SkillGatewayRequest(
                capability="inspect_scene",
                objective="inspect current scene",
                wait_policy=wait_policy,  # type: ignore[arg-type]
            )
        )
    )

    assert result.startswith(expected_prefix)
    assert "capability=inspect_scene" in result


def test_skill_gateway_wait_result_resolves_and_clears_pending_future() -> None:
    pending_skills: dict[str, asyncio.Future[str]] = {}

    def resolve_pending(skill: SkillIntent) -> None:
        pending_skills[skill.skill_id].set_result("inspect_scene completed")

    io = FakeGatewayIO(on_submit=resolve_pending)
    gateway = _gateway(io=io, pending_skills=pending_skills)

    result = asyncio.run(
        gateway.submit(
            SkillGatewayRequest(
                capability="inspect_scene",
                objective="inspect current scene",
                wait_policy="wait_result",
            )
        )
    )

    assert result == "inspect_scene completed"
    assert pending_skills == {}


def test_skill_gateway_recovery_guard_blocks_unsafe_skill() -> None:
    gateway = _gateway(recovery_required=lambda: True)

    with pytest.raises(RuntimeError, match="recovery required"):
        asyncio.run(
            gateway.submit(
                SkillGatewayRequest(
                    capability="set_gripper",
                    objective="close the gripper",
                    slots={"action": "close"},
                    enforce_motion_guards=False,
                )
            )
        )
