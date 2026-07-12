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
    assert client.password == "secret"


def test_bus_factory_rejects_missing_acl_role() -> None:
    with pytest.raises(ValueError, match="missing role"):
        create_bus_client(BusSpec(options={"credentials": {}}), role="robot")
