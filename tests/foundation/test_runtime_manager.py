# ruff: noqa: N802 閳?gRPC stub method names are defined in .proto, not our choice
from __future__ import annotations

import asyncio

import grpc
import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import (
    MockModelServiceClient,
    ModelServiceRegistry,
    ServiceInvocationRequest,
)
from hey_robot.foundation.contract.v1 import model_service_pb2
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
from hey_robot.protocol import Envelope, SkillIntent


def _config() -> DeploymentConfig:
    return DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "mock",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "provides": ["set_gripper"],
                    "success": False,
                    "failure_mode": "policy_timeout",
                    "error": "timed out",
                    "result_metrics": {"duration_sec": 3.0},
                },
                "disabled": {
                    "type": "mock",
                    "enabled": False,
                    "robot_id": "xlerobot",
                    "provides": ["set_gripper"],
                },
                "other_robot": {
                    "type": "mock",
                    "enabled": True,
                    "robot_id": "other",
                    "provides": ["set_gripper"],
                },
            }
        }
    )


def test_capability_runtime_routes_enabled_service_by_skill_and_robot() -> None:
    runtime = ModelServiceRegistry(_config())

    match = runtime.service_for("set_gripper", "xlerobot")

    assert match is not None
    service_id, spec, client = match
    assert service_id == "arm_vla"
    assert spec.robot_id == "xlerobot"
    assert isinstance(client, MockModelServiceClient)
    assert runtime.service_for("set_gripper", "missing") is None
    assert runtime.service_for("unknown_skill", "xlerobot") is None


def test_capability_runtime_prefers_first_matching_enabled_service() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla_primary": {
                    "type": "mock",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "provides": ["set_gripper"],
                },
                "arm_vla_secondary": {
                    "type": "mock",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "provides": ["set_gripper"],
                },
            }
        }
    )

    runtime = ModelServiceRegistry(config)
    match = runtime.service_for("set_gripper", "xlerobot")

    assert match is not None
    service_id, _spec, client = match
    assert service_id == "arm_vla_primary"
    assert isinstance(client, MockModelServiceClient)


def test_capability_runtime_allows_global_service_when_robot_id_not_specified() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "shared_vla": {
                    "type": "mock",
                    "enabled": True,
                    "provides": ["set_gripper"],
                },
            }
        }
    )

    runtime = ModelServiceRegistry(config)
    match = runtime.service_for("set_gripper", "xlerobot")

    assert match is not None
    service_id, spec, _client = match
    assert service_id == "shared_vla"
    assert spec.robot_id == ""


def test_capability_runtime_routes_robot_policy_to_grpc_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _channel: object(),
    )
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "robot_policy",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["manipulate"],
                }
            }
        }
    )

    runtime = ModelServiceRegistry(config)
    match = runtime.service_for("manipulate", "xlerobot")

    assert match is not None
    assert isinstance(match[2], GrpcModelServiceClient)


def test_capability_runtime_routes_vln_planner_to_grpc_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _channel: object(),
    )
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "vln_nav": {
                    "type": "vln_planner",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9091",
                    "provides": ["navigate_to", "approach_object"],
                }
            }
        }
    )

    runtime = ModelServiceRegistry(config)
    match = runtime.service_for("navigate_to", "xlerobot")

    assert match is not None
    service_id, spec, client = match
    assert service_id == "vln_nav"
    assert spec.type == "vln_planner"
    assert isinstance(client, GrpcModelServiceClient)


def test_mock_capability_client_records_execution_and_cancel() -> None:
    runtime = ModelServiceRegistry(_config())
    match = runtime.service_for("set_gripper", "xlerobot")
    assert match is not None
    _, _, client = match
    assert isinstance(client, MockModelServiceClient)
    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="skill1",
        task_id="task1",
        intent_kind="skill",
        name="set_gripper",
        arguments={},
        objective="close the gripper",
    )
    request = ServiceInvocationRequest(
        service_id="arm_vla",
        intent=intent,
        timeout_sec=10.0,
    )

    health = asyncio.run(client.health())
    result = asyncio.run(client.execute(request))
    asyncio.run(client.cancel("skill1"))

    assert health.online is True
    assert result.success is False
    assert result.failure_mode == "policy_timeout"
    assert result.error == "timed out"
    assert result.metrics == {"duration_sec": 3.0}
    assert client.executed == [request]
    assert client.cancelled == ["skill1"]


def test_grpc_capability_client_maps_health_execute_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = {"health_requests": [], "execute_requests": [], "cancel_requests": []}

    class FakeStub:
        async def GetHealth(self, request, **kwargs):
            recorded["health_requests"].append((request, kwargs.get("timeout")))
            return model_service_pb2.GetHealthResponse(
                service_id="arm_vla",
                name="arm_vla",
                robot_id="xlerobot",
                online=True,
                loaded=True,
                busy=False,
                current_skill_id="",
                metrics=_struct(policy_type="pi05"),
                version="grpc-v1",
            )

        async def ExecuteSkill(self, request, **kwargs):
            recorded["execute_requests"].append((request, kwargs.get("timeout")))
            return model_service_pb2.ExecuteSkillResponse(
                success=True,
                status="completed",
                summary="done",
                metrics=_struct(frames=12),
            )

        async def CancelSkill(self, request, **kwargs):
            recorded["cancel_requests"].append((request, kwargs.get("timeout")))
            return model_service_pb2.CancelSkillResponse(
                accepted=True, summary="cancel requested"
            )

    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _: FakeStub(),
    )
    spec = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "robot_policy",
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["set_gripper"],
                    "health_timeout_sec": 1.5,
                }
            }
        }
    ).model_services["arm_vla"]
    client = GrpcModelServiceClient("arm_vla", spec)
    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot", trace_id="trace-1", episode_id="ep-1"),
        skill_id="skill1",
        task_id="task1",
        intent_kind="skill",
        name="set_gripper",
        objective="close the gripper",
        arguments={"action": "close"},
    )

    health = asyncio.run(client.health())
    result = asyncio.run(
        client.execute(
            ServiceInvocationRequest(
                service_id="arm_vla",
                intent=intent,
                timeout_sec=2.0,
                arguments={"action": "open", "camera": "front"},
            )
        )
    )
    asyncio.run(client.cancel("skill1"))

    assert health.metrics == {"policy_type": "pi05"}
    assert result.success is True
    assert result.metrics == {"frames": 12.0}
    execute_request = recorded["execute_requests"][0][0]
    assert execute_request.skill_id == "skill1"
    assert execute_request.skill_name == "set_gripper"
    assert dict(execute_request.arguments) == {"action": "open", "camera": "front"}
    cancel_request = recorded["cancel_requests"][0][0]
    assert cancel_request.skill_id == "skill1"


def test_grpc_capability_client_health_reports_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRpcError(Exception):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "server down"

    class FailingStub:
        async def GetHealth(self, request, **_kwargs):
            del request
            raise FakeRpcError()

    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.AioRpcError", FakeRpcError
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _: FailingStub(),
    )
    spec = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "robot_policy",
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["set_gripper"],
                }
            }
        }
    ).model_services["arm_vla"]

    health = asyncio.run(GrpcModelServiceClient("arm_vla", spec).health())

    assert health.online is False
    assert health.loaded is False
    assert health.error == "UNAVAILABLE: server down"
    assert health.error_code == "UNAVAILABLE"


def test_grpc_capability_client_execute_reports_rpc_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRpcError(Exception):
        def code(self):
            return grpc.StatusCode.DEADLINE_EXCEEDED

        def details(self):
            return "execution timed out"

    class FailingStub:
        async def ExecuteSkill(self, request, **_kwargs):
            del request
            raise FakeRpcError()

    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.AioRpcError", FakeRpcError
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _: FailingStub(),
    )
    spec = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "robot_policy",
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["set_gripper"],
                }
            }
        }
    ).model_services["arm_vla"]
    client = GrpcModelServiceClient("arm_vla", spec)
    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot", trace_id="trace-1"),
        skill_id="skill1",
        task_id="task1",
        intent_kind="skill",
        name="set_gripper",
        arguments={},
        objective="close the gripper",
    )

    result = asyncio.run(
        client.execute(
            ServiceInvocationRequest(
                service_id="arm_vla",
                intent=intent,
                timeout_sec=2.0,
            )
        )
    )

    assert result.success is False
    assert result.status == "failed"
    assert result.failure_mode == "model_service_unavailable"
    assert result.summary == "execution timed out"
    assert result.error == "execution timed out"
    assert result.error_code == "DEADLINE_EXCEEDED"


def test_grpc_capability_client_cancel_propagates_rpc_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRpcError(Exception):
        pass

    class FailingStub:
        async def CancelSkill(self, request, **_kwargs):
            del request
            raise FakeRpcError("cancel failed")

    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.grpc.aio.insecure_channel",
        lambda target: target,
    )
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.client.model_service_pb2_grpc.ModelServiceStub",
        lambda _: FailingStub(),
    )
    spec = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "robot_policy",
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["set_gripper"],
                }
            }
        }
    ).model_services["arm_vla"]

    with pytest.raises(FakeRpcError, match="cancel failed"):
        asyncio.run(GrpcModelServiceClient("arm_vla", spec).cancel("skill1"))


def _struct(**kwargs):
    message = model_service_pb2.ExecuteSkillRequest().arguments
    message.update(kwargs)
    return message
