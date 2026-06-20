from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.gateway import GatewayService


def test_gateway_registers_feishu_channel(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"episodes": {"root": str(tmp_path / "episodes")}},
            "robots": {"xlerobot": {"type": "mock"}},
            "channels": {
                "feishu": {
                    "type": "feishu",
                    "enabled": True,
                    "allow_from": ["*"],
                }
            },
        }
    )
    gateway = GatewayService(config, episode_dir=tmp_path / "episodes")

    assert gateway.channels.get("feishu") is not None
