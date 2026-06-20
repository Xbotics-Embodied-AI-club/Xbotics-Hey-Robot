from __future__ import annotations

from hey_robot.perception.codecs.base import ObservationActionCodec
from hey_robot.perception.codecs.simple import SimpleVectorCodec
from hey_robot.perception.codecs.skill import RobotSkillActionCodec


class CodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[str, ObservationActionCodec] = {
            "mock": SimpleVectorCodec(),
            "simple": SimpleVectorCodec(),
            "skill": RobotSkillActionCodec(),
        }

    def register(self, codec: ObservationActionCodec) -> None:
        self._codecs[codec.name] = codec

    def get(self, name: str) -> ObservationActionCodec:
        codec = self._codecs.get(name)
        if codec is None:
            raise KeyError(f"unknown observation/action codec: {name}")
        return codec
