"""Skill command/event transport implementations."""

from hey_robot.skills.transport.legacy import (
    LegacySkillWorkerBridge,
    LegacySkillWorkerBridgeService,
    adapt_legacy_skill,
    adapt_legacy_skills,
    to_legacy_result,
)
from hey_robot.skills.transport.local import LocalSkillClient
from hey_robot.skills.transport.nats import NatsSkillClient, NatsSkillWorker

__all__ = [
    "LegacySkillWorkerBridge",
    "LegacySkillWorkerBridgeService",
    "LocalSkillClient",
    "NatsSkillClient",
    "NatsSkillWorker",
    "adapt_legacy_skill",
    "adapt_legacy_skills",
    "to_legacy_result",
]
