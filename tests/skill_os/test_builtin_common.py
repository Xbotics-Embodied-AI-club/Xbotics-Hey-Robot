import asyncio
from types import SimpleNamespace

import pytest

from hey_robot.skill_os.builtins.common import invoke, schema, spec


def test_schema_adds_required_only_when_present() -> None:
    assert schema({"name": {"type": "string"}}) == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    assert schema({"name": {"type": "string"}}, ("name",)) == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }


def test_spec_builds_skill_spec_with_defaults_and_metadata() -> None:
    skill_spec = spec(
        "demo_skill",
        "Demo skill",
        category="demo",
        required_resources=("camera",),
        dependencies=("inspect_scene",),
        driver_primitives=("capture",),
        required_model_service="vision",
        goal_effects=("observed",),
        evidence_outputs=("scene",),
        cannot_satisfy=("no_camera",),
    )

    assert skill_spec.name == "demo_skill"
    assert skill_spec.category == "demo"
    assert skill_spec.required_resources == ("camera",)
    assert skill_spec.dependencies == ("inspect_scene",)
    assert skill_spec.driver_primitives == ("capture",)
    assert skill_spec.required_model_service == "vision"
    assert skill_spec.goal_effects == ("observed",)
    assert skill_spec.evidence_outputs == ("scene",)
    assert skill_spec.cannot_satisfy == ("no_camera",)


def test_invoke_delegates_and_rejects_missing_invoke() -> None:
    calls = []

    async def call(name, arguments):
        calls.append((name, arguments))
        return "ok"

    result = asyncio.run(invoke(SimpleNamespace(invoke=call), "inner", {"x": 1}))

    assert result == "ok"
    assert calls == [("inner", {"x": 1})]

    with pytest.raises(RuntimeError):
        asyncio.run(invoke(SimpleNamespace(invoke=None), "inner"))
