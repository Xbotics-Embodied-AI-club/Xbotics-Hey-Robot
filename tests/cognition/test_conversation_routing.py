from __future__ import annotations

import asyncio

import pytest

from hey_robot.cognition.conversation_entities import EntityResolver
from hey_robot.cognition.conversation_execution import RobotExecutionAdapter
from hey_robot.cognition.conversation_goal import (
    GoalContractBuilder,
    GoalProposal,
    GoalTemplate,
    GoalTemplateRegistry,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.robot import (
    ToolDependencies,
    ToolRegistry,
)
from hey_robot.protocol import (
    Envelope,
    GoalCommand,
    RobotObservation,
    SceneEntity,
    SceneRelation,
    Topics,
)
from hey_robot.protocol.messages import from_payload
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


def test_conversation_skill_does_not_upgrade_by_category() -> None:
    catalog = SkillCatalog(
        (
            SkillSpec(
                name="navigate_once",
                description="bounded navigation skill",
                category="navigation",
                input_schema={"type": "object", "properties": {}},
            ),
        )
    )
    tools = ToolRegistry(ToolDependencies(catalog))

    proposal = tools.proposal("request_skill", {"skill": "navigate_once"})

    assert proposal.intent_kind == "skill"
    assert proposal.skill_name == "navigate_once"


def test_goal_template_compiles_enter_through_a_generic_relation() -> None:
    builder = GoalContractBuilder(("robot:sim_robot", "room:kitchen"))
    target = SceneEntity(
        "passage:1",
        "passage",
        42,
        relations=[SceneRelation("leads_to", "room:kitchen")],
    )

    criteria, _budgets = builder.build(
        GoalProposal("enter", "enter the passage", "passage:1"),
        robot_id="sim_robot",
        target_entity=target,
    )

    assert criteria[0].predicate == "inside"
    assert criteria[0].object_id == "room:kitchen"


def test_goal_templates_are_open_for_domain_extensions() -> None:
    templates = GoalTemplateRegistry(
        (
            GoalTemplate(
                "inspect_area", "evidence_present", "observed", "robot", "target"
            ),
        )
    )
    builder = GoalContractBuilder(
        ("robot:sim_robot", "area:workbench"), templates=templates
    )

    criteria, _budgets = builder.build(
        GoalProposal("inspect_area", "inspect workbench", "area:workbench"),
        robot_id="sim_robot",
    )

    assert criteria[0].predicate == "observed"
    assert builder.goal_kinds == ("inspect_area",)


def test_exact_entity_id_resolves_only_from_latest_robot_observation() -> None:
    resolver = EntityResolver(("robot:sim_robot", "room:kitchen"))
    resolver.update(
        RobotObservation(
            envelope=Envelope(robot_id="sim_robot"),
            frame_id=42,
            entities=[
                SceneEntity(
                    "passage:1",
                    "passage",
                    42,
                    relations=[SceneRelation("leads_to", "room:kitchen")],
                )
            ],
        )
    )

    resolved = resolver.resolve("passage:1", robot_id="sim_robot")

    assert resolved.entity is not None
    assert resolved.entity.relations[0].object_id == "room:kitchen"


def test_natural_language_is_not_interpreted_by_resolver() -> None:
    resolver = EntityResolver(("robot:sim_robot", "room:kitchen"))

    with pytest.raises(ValueError, match="唯一的受信实体"):
        resolver.resolve("right door", robot_id="sim_robot")


def test_known_entity_alias_resolves_without_visual_entity() -> None:
    resolver = EntityResolver(
        ("robot:sim_robot", "room:kitchen"), aliases={"厨房": "room:kitchen"}
    )

    resolved = resolver.resolve("厨房", robot_id="sim_robot")

    assert resolved.target_id == "room:kitchen"


def test_newer_empty_observation_invalidates_previous_entity() -> None:
    resolver = EntityResolver(("robot:sim_robot", "room:kitchen"))
    resolver.update(
        RobotObservation(
            Envelope(robot_id="sim_robot"),
            1,
            entities=[SceneEntity("passage:1", "passage", 1)],
        )
    )
    resolver.update(RobotObservation(Envelope(robot_id="sim_robot"), 2))

    with pytest.raises(ValueError, match="唯一的受信实体"):
        resolver.resolve("passage:1", robot_id="sim_robot")


def test_goal_proposal_publishes_goal_command_and_updates_projection(tmp_path) -> None:
    class Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    bus = Bus()
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    adapter = RobotExecutionAdapter(
        bus,
        Topics(),
        SkillCatalog(()),
        store,
        known_entities=("robot:sim_robot", "room:kitchen"),
    )
    outcome = asyncio.run(
        adapter.execute(
            GoalProposal("enter", "enter the kitchen", "room:kitchen"),
            Envelope(robot_id="sim_robot"),
            "session-1",
        )
    )

    assert outcome.status == "accepted"
    command = from_payload(GoalCommand, bus.published[0][1])
    assert command.action == "create"
    assert command.success_criteria[0].predicate == "inside"
    assert store.active_goal("session-1") == {
        "goal_id": outcome.goal_id,
        "objective": "enter the kitchen",
        "status": "pending",
    }
    store.close()
