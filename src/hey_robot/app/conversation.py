"""Application wiring for the conversation Agent and the Skill OS gateway."""

from __future__ import annotations

from typing import Any

from hey_robot.cognition.conversation_execution import RobotExecutionAdapter
from hey_robot.cognition.conversation_service import ConversationAgentService
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.autonomous import (
    AgentToolDependencies,
    build_agent_tools,
)
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Topics
from hey_robot.skill_os.registry import registry_from_config


def build_conversation_agent(
    config: DeploymentConfig, *, agent_id: str
) -> ConversationAgentService:
    """Compose cognition ports with the Skill OS implementation at the app edge."""
    catalog = registry_from_config(config).catalog()
    tools = build_agent_tools(AgentToolDependencies(catalog))

    def execution_factory(bus: Any, topics: Topics, store: ConversationStore):
        return RobotExecutionAdapter(
            bus,
            topics,
            catalog,
            store,
            known_entities=config.autonomy.entity_catalog,
        )

    return ConversationAgentService(
        config,
        agent_id=agent_id,
        tools=tools,
        execution_factory=execution_factory,
    )
