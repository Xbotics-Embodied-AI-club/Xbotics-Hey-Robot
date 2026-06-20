from __future__ import annotations

from typing import Any

from hey_robot.bus.client import BusClient
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


def create_bus_client(spec: BusSpec) -> BusClient:
    if spec.type != "nats":
        raise ValueError(f"unsupported bus type: {spec.type}")
    options: dict[str, Any] = {
        key: value for key, value in spec.options.items() if key in _BUS_CLIENT_KEYS
    }
    return BusClient(url=spec.url, **options)
