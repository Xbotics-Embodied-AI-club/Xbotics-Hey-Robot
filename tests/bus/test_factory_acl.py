import asyncio

import pytest

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import BusSpec


def test_bus_factory_uses_service_role_credentials(monkeypatch) -> None:
    monkeypatch.setenv("SUPERVISOR_PASSWORD", "secret")
    client = create_bus_client(
        BusSpec(
            options={
                "credentials": {
                    "autonomy_supervisor": {
                        "username": "supervisor",
                        "password_env": "SUPERVISOR_PASSWORD",
                    }
                }
            }
        ),
        role="autonomy_supervisor",
    )
    assert client.username == "supervisor"
    assert client.password == "secret"  # noqa: S105


def test_bus_factory_rejects_missing_acl_role() -> None:
    with pytest.raises(ValueError, match="missing role"):
        create_bus_client(BusSpec(options={"credentials": {}}), role="robot")


def test_in_memory_bus_delivers_only_within_shared_process_hub() -> None:
    async def run() -> None:
        received: list[dict] = []
        spec = BusSpec(type="in_memory", url="memory://test-factory")
        publisher = create_bus_client(spec, role="gateway")
        subscriber = create_bus_client(spec, role="autonomy_supervisor")
        await publisher.connect()
        await subscriber.connect()
        await subscriber.subscribe(
            ["goal.command"], lambda _topic, payload: _append(received, payload)
        )
        await publisher.publish("goal.command", {"action": "create"})
        assert received == [{"action": "create"}]
        await publisher.close()
        await subscriber.close()

    asyncio.run(run())


async def _append(received: list[dict], payload: dict) -> None:
    received.append(payload)
