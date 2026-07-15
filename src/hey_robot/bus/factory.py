from __future__ import annotations

import os
from typing import Any

from hey_robot.bus.client import BusClient
from hey_robot.bus.in_memory import InMemoryBusClient, InMemoryBusHub
from hey_robot.bus.types import MessageBus
from hey_robot.config import BusSpec

_BUS_CLIENT_KEYS = {
    "tls_ca_file",
    "tls_cert_file",
    "tls_key_file",
    "username",
    "password",
    "token",
    "reconnect",
    "max_reconnect_attempts",
    "reconnect_time_wait_ms",
    "use_jetstream",
    "js_stream",
}
_IN_MEMORY_HUBS: dict[str, InMemoryBusHub] = {}


def create_bus_client(spec: BusSpec, *, role: str | None = None) -> MessageBus:
    if spec.type == "in_memory":
        hub = _IN_MEMORY_HUBS.setdefault(spec.url, InMemoryBusHub())
        return InMemoryBusClient(hub)
    if spec.type != "nats":
        raise ValueError(f"unsupported bus type: {spec.type}")
    options: dict[str, Any] = {
        key: value for key, value in spec.options.items() if key in _BUS_CLIENT_KEYS
    }
    credentials = spec.options.get("credentials")
    if role is not None and isinstance(credentials, dict):
        entry = credentials.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"NATS credentials missing role: {role}")
        username = entry.get("username")
        password = entry.get("password")
        password_env = entry.get("password_env")
        if password_env is not None:
            password = os.environ.get(str(password_env))
        if (
            not isinstance(username, str)
            or not isinstance(password, str)
            or not password
        ):
            raise ValueError(f"NATS credentials for role {role} are incomplete")
        options.update(username=username, password=password)
    return BusClient(url=spec.url, **options)
