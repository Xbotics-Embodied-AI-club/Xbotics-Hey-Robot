from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoBusConfig:
    port: str = "COM5"
    baudrate: int = 1_000_000
