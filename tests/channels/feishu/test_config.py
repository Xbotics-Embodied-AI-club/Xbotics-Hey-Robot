from __future__ import annotations

from hey_robot.channels.feishu import feishu_config_from_settings


def test_feishu_config_from_settings_resolves_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_FEISHU_APP_ID", "app-id")
    expected_secret = "app" + "-secret"
    monkeypatch.setenv("TEST_FEISHU_APP_SECRET", expected_secret)

    config = feishu_config_from_settings(
        {
            "app_id_env": "TEST_FEISHU_APP_ID",
            "app_secret_env": "TEST_FEISHU_APP_SECRET",
            "group_policy": "open",
            "allow_from": ["*"],
        }
    )

    assert config.resolved_app_id == "app-id"
    assert config.resolved_app_secret == expected_secret
    assert config.group_policy == "open"
    assert config.allow_from == ["*"]
