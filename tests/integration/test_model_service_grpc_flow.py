from __future__ import annotations

import asyncio

import grpc
import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import ModelServiceRegistry, ServiceInvocationRequest
from hey_robot.foundation.contract.v1 import model_service_pb2_grpc
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
from hey_robot.foundation.transport.grpc.server import (
    ModelServiceServicer,
    ModelServiceState,
)
from hey_robot.protocol import Envelope, SkillIntent


def test_deployment_style_model_service_grpc_flow(tmp_path) -> None:
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
                "model_services": {
                    "arm_vla": {
                        "type": "robot_policy",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "provides": ["set_gripper"],
                        "resources": ["gripper"],
                        "timeout_sec": 20,
                        "target": "127.0.0.1:0",
                    }
                },
            }
        )
        spec = config.model_services["arm_vla"]
        server = grpc.aio.server()
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            ModelServiceServicer(ModelServiceState("arm_vla", spec), FakeExecutor()),  # type: ignore[arg-type]
            server,
        )
        try:
            port = server.add_insecure_port("127.0.0.1:0")
        except RuntimeError as exc:
            pytest.skip(f"gRPC loopback binding unavailable in this environment: {exc}")
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
                task_id="task-integration",
                intent_kind="skill",
                name="set_gripper",
                arguments={"action": "close"},
                objective="close the gripper",
            )
            client = GrpcModelServiceClient("arm_vla", spec)
            result = await client.execute(
                ServiceInvocationRequest(
                    service_id="arm_vla",
                    intent=intent,
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


def test_foundation_model_service_flow_keeps_skill_surface() -> None:
    async def run_once() -> None:
        config = DeploymentConfig.from_dict(
            {
                "model_services": {
                    "arm_vla": {
                        "type": "robot_policy",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "provides": ["set_gripper"],
                        "resources": ["gripper"],
                        "timeout_sec": 20,
                        "target": "127.0.0.1:9191",
                    }
                }
            }
        )

        match = ModelServiceRegistry(config).service_for("set_gripper", "xlerobot")

        assert match is not None
        service_id, spec, _client = match
        assert service_id == "arm_vla"
        assert spec.provides == ("set_gripper",)

    asyncio.run(run_once())
