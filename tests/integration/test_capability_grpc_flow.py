from __future__ import annotations

import asyncio

import grpc

from hey_robot.capability.contract.v1 import capability_pb2_grpc
from hey_robot.capability.runtime import CapabilityExecutionRequest, CapabilityRuntime
from hey_robot.capability.transport.grpc.client import GrpcCapabilityClient
from hey_robot.capability.transport.grpc.server import (
    VLACapabilityServicer,
    VLAServiceState,
)
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.skills import load_skill_registry


def test_deployment_style_capability_grpc_flow(tmp_path) -> None:
    class FakeExecutor:
        def health(self) -> dict[str, object]:
            return {
                "name": "arm_vla",
                "online": True,
                "loaded": True,
                "robot_id": "xlerobot",
                "metrics": {"runtime": "integration"},
            }

        def execute(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "success": True,
                "status": "completed",
                "summary": f"set {payload['arguments']['action']}",
                "metrics": {
                    "source": "integration",
                    "action": payload["arguments"]["action"],
                },
            }

        def cancel(self) -> None:
            return None

    async def run_once() -> None:
        config = DeploymentConfig.from_dict(
            {
                "resources": {
                    "runtime_dir": str(tmp_path / "runtime"),
                    "media": {"root": str(tmp_path / "media")},
                    "episodes": {"root": str(tmp_path / "episodes")},
                },
                "robots": {"xlerobot": {"type": "xlerobot"}},
                "policies": {
                    "embodied_skills": {
                        "type": "skill",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "settings": {"codec": "skill"},
                    }
                },
                "capability_services": {
                    "arm_vla": {
                        "type": "vla_service",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "skill_names": ["set_gripper"],
                        "resources": ["gripper"],
                        "timeout_sec": 20,
                        "target": "127.0.0.1:0",
                    }
                },
            }
        )
        spec = config.capability_services["arm_vla"]
        server = grpc.aio.server()
        capability_pb2_grpc.add_CapabilityServiceServicer_to_server(
            VLACapabilityServicer(VLAServiceState("arm_vla", spec), FakeExecutor()),  # type: ignore[arg-type]
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        object.__setattr__(spec, "target", f"127.0.0.1:{port}")
        await server.start()
        try:
            intent = SkillIntent(
                envelope=Envelope(
                    trace_id="tr-integration",
                    episode_id="ep-integration",
                    robot_id="xlerobot",
                ),
                skill_id="skill-integration",
                name="set_gripper",
                arguments={"action": "close"},
                objective="close the gripper",
            )
            client = GrpcCapabilityClient("arm_vla", spec)
            result = await client.execute(
                CapabilityExecutionRequest(
                    service_id="arm_vla",
                    intent=intent,
                    contract=load_skill_registry()
                    .robot_skill_catalog()
                    .get("set_gripper"),
                    timeout_sec=20.0,
                )
            )
        finally:
            await server.stop(grace=0.1)

        assert result.success is True
        assert result.status == "completed"
        assert result.summary == "set close"
        assert result.metrics == {"source": "integration", "action": "close"}

    asyncio.run(run_once())


def test_foundation_composite_capability_flow_keeps_skill_surface() -> None:
    async def run_once() -> None:
        config = DeploymentConfig.from_dict(
            {
                "capability_services": {
                    "arm_vla": {
                        "type": "vla_service",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "skill_names": ["set_gripper"],
                        "resources": ["gripper"],
                        "timeout_sec": 20,
                        "target": "127.0.0.1:9191",
                    }
                }
            }
        )

        match = CapabilityRuntime(config).service_for("set_gripper", "xlerobot")

        assert match is not None
        service_id, spec, _client = match
        assert service_id == "arm_vla"
        assert spec.skill_names == ("set_gripper",)

    asyncio.run(run_once())
