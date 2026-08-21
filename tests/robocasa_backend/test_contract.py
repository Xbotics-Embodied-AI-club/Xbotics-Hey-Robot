from __future__ import annotations

import pytest

from hey_robot.robocasa_backend.contract import DEFAULT_REGISTRIES, load_manifest


def test_load_manifest_normalizes_the_runtime_contract(tmp_path) -> None:
    manifest = tmp_path / "tasks.yaml"
    manifest.write_text(
        "version: 2\nsuites:\n  simple:\n    - OpenDrawer\n    - WashLettuce\nregistries:\n  - custom\n",
        encoding="utf-8",
    )

    loaded = load_manifest(manifest)

    assert loaded == {
        "version": "2",
        "split": "target",
        "registries": ("custom",),
        "suites": {"simple": ["OpenDrawer", "WashLettuce"]},
        "tasks": ["OpenDrawer", "WashLettuce"],
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("- not-a-mapping", "mapping"),
        ("version: 1", "define suites"),
        ("suites:\n  invalid:\n    - UnknownTask", "outside"),
        ("suites: {}", "outside"),
    ],
)
def test_load_manifest_rejects_invalid_or_unknown_tasks(
    tmp_path, contents: str, message: str
) -> None:
    manifest = tmp_path / "tasks.yaml"
    manifest.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(manifest)


def test_load_manifest_uses_default_registry_when_not_declared(tmp_path) -> None:
    manifest = tmp_path / "tasks.yaml"
    manifest.write_text("suites:\n  suite:\n    - OpenDrawer", encoding="utf-8")

    assert load_manifest(manifest)["registries"] == DEFAULT_REGISTRIES
