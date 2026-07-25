"""Small provider-neutral validator for model-visible object schemas."""

from __future__ import annotations

from typing import Any


def validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    if schema.get("type") not in {None, "object"}:
        raise ValueError("tool parameters must be an object schema")
    resolved = dict(arguments)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return resolved
    if schema.get("additionalProperties") is False:
        unknown = set(resolved) - set(properties)
        if unknown:
            raise ValueError(f"unexpected arguments: {', '.join(sorted(unknown))}")
    for name, definition in properties.items():
        if (
            name not in resolved
            and isinstance(definition, dict)
            and "default" in definition
        ):
            resolved[name] = definition["default"]
    required = schema.get("required", ())
    for name in required if isinstance(required, list | tuple) else ():
        if name not in resolved:
            raise ValueError(f"missing required argument: {name}")
    for name, value in resolved.items():
        definition = properties.get(name)
        if isinstance(definition, dict):
            _validate_value(name, value, definition)
    return resolved


def _validate_value(name: str, value: Any, definition: dict[str, Any]) -> None:
    expected = definition.get("type")
    valid = {
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }
    if expected in valid and not valid[expected]:
        raise ValueError(f"argument {name} must be a {expected}")
    if "enum" in definition and value not in definition["enum"]:
        raise ValueError(f"argument {name} must be one of {definition['enum']}")
    min_length = definition.get("minLength")
    if (
        isinstance(value, str)
        and isinstance(min_length, int)
        and len(value) < min_length
    ):
        raise ValueError(f"argument {name} must have length >= {min_length}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")
        if isinstance(minimum, int | float) and value < minimum:
            raise ValueError(f"argument {name} must be >= {minimum}")
        if isinstance(maximum, int | float) and value > maximum:
            raise ValueError(f"argument {name} must be <= {maximum}")
