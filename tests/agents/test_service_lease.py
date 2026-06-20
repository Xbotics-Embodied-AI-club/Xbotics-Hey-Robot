from __future__ import annotations

import time
from dataclasses import dataclass, field

from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import AgentReply, SkillIntent


@dataclass
class FakeAgentIO:
    skills: list[SkillIntent] = field(default_factory=list)
    replies: list[AgentReply] = field(default_factory=list)
    task_results: list[tuple[bool, str]] = field(default_factory=list)

    async def submit_skill(self, skill: SkillIntent) -> None:
        self.skills.append(skill)

    async def publish_reply(self, reply: AgentReply) -> None:
        self.replies.append(reply)

    async def publish_task_result(self, *, success: bool, summary: str) -> None:
        self.task_results.append((success, summary))


def test_robot_agent_service_lease_blocks_busy_robot(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.turn_sessions.robot_leases["mock0"] = ("cmd1", time.time())
    service.skill_lease_timeout_sec = 999999.0

    lease = service.turn_sessions.active_robot_lease(
        "mock0", timeout_sec=service.skill_lease_timeout_sec
    )
    assert lease is not None
    assert lease[0] == "cmd1"
