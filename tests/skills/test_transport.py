from __future__ import annotations

import asyncio
import uuid

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.skills import (
    BusSkillClient,
    BusSkillServer,
    Skill,
    SkillCommand,
    SkillRegistry,
    SkillResult,
    SkillWorker,
)

TRANSPORT_TIMEOUT_SEC = 5


@pytest.mark.asyncio
async def test_bus_skill_transport_runs_command_and_relays_events(tmp_path) -> None:
    async def inspect(_context, arguments):
        return SkillResult(
            True,
            f"observed {arguments['target']}",
            "completed",
        )

    registry = SkillRegistry()
    registry.register(
        Skill(
            "inspect",
            "Inspect a target.",
            {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            inspect,
        )
    )
    config = DeploymentConfig.from_dict(
        {
            "deployment": {
                "id": "transport-test",
                "bus": {
                    "type": "in_memory",
                    "url": f"memory://skill-transport-{uuid.uuid4().hex}",
                },
            }
        }
    )
    worker = SkillWorker(registry, run_store=FileRunStore(tmp_path / "runs"))
    server = BusSkillServer(config, worker)
    client = BusSkillClient(config)
    server_task = asyncio.create_task(server.start())
    await asyncio.wait_for(server.ready.wait(), timeout=TRANSPORT_TIMEOUT_SEC)

    command = SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-transport",
        task_id="task-transport",
        robot_id="mock0",
        name="inspect",
        arguments={"target": "table"},
    )
    try:
        await client.submit(command)
        stream = client.events()
        terminal = None
        while terminal is None:
            event = await asyncio.wait_for(anext(stream), timeout=TRANSPORT_TIMEOUT_SEC)
            if event.phase in {"completed", "failed", "cancelled"}:
                terminal = event

        assert terminal.phase == "completed"
        assert terminal.result is not None
        assert terminal.result.summary == "observed table"
        assert await client.status(command.run_id) == terminal
        await stream.aclose()
    finally:
        await client.close()
        await server.close()
        await asyncio.gather(server_task, return_exceptions=True)
