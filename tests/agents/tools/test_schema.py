from __future__ import annotations

from hey_robot.agents.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    Schema,
    StringSchema,
    tool_parameters_schema,
)


class TestStringSchema:
    def test_basic(self):
        s = StringSchema("test param")
        js = s.to_json_schema()
        assert js["type"] == "string"
        assert js["description"] == "test param"
        assert s.validate_value("hello") == []

    def test_nullable(self):
        s = StringSchema("nullable param", nullable=True)
        js = s.to_json_schema()
        assert js["type"] == ["string", "null"]
        assert s.validate_value(None) == []

    def test_min_max_length(self):
        s = StringSchema("bounded", min_length=2, max_length=5)
        assert s.validate_value("ab") == []
        assert s.validate_value("abcde") == []
        assert len(s.validate_value("a")) > 0
        assert len(s.validate_value("abcdef")) > 0

    def test_enum(self):
        s = StringSchema("choice", enum=["red", "green", "blue"])
        assert s.validate_value("red") == []
        assert len(s.validate_value("yellow")) > 0


class TestIntegerSchema:
    def test_basic(self):
        s = IntegerSchema(0, description="count")
        js = s.to_json_schema()
        assert js["type"] == "integer"
        assert s.validate_value(5) == []
        assert len(s.validate_value(3.5)) > 0

    def test_bounds(self):
        s = IntegerSchema(0, minimum=1, maximum=10)
        assert s.validate_value(1) == []
        assert s.validate_value(10) == []
        assert len(s.validate_value(0)) > 0
        assert len(s.validate_value(11)) > 0

    def test_nullable(self):
        s = IntegerSchema(0, nullable=True)
        assert s.validate_value(None) == []

    def test_enum(self):
        s = IntegerSchema(0, enum=[1, 2, 3])
        assert s.validate_value(2) == []
        assert len(s.validate_value(5)) > 0


class TestNumberSchema:
    def test_basic(self):
        s = NumberSchema(0.0, description="ratio")
        js = s.to_json_schema()
        assert js["type"] == "number"
        # integers pass for number
        assert s.validate_value(3) == []
        assert s.validate_value(3.5) == []

    def test_bounds(self):
        s = NumberSchema(0.5, minimum=0.0, maximum=1.0)
        assert s.validate_value(0.0) == []
        assert s.validate_value(1.0) == []
        assert len(s.validate_value(-0.1)) > 0
        assert len(s.validate_value(1.1)) > 0

    def test_nullable(self):
        s = NumberSchema(0.0, nullable=True)
        assert s.validate_value(None) == []


class TestBooleanSchema:
    def test_basic(self):
        s = BooleanSchema(description="flag")
        js = s.to_json_schema()
        assert js["type"] == "boolean"
        assert s.validate_value(True) == []
        assert s.validate_value(False) == []

    def test_default(self):
        s = BooleanSchema(default=False)
        js = s.to_json_schema()
        assert js["default"] is False

    def test_nullable(self):
        s = BooleanSchema(nullable=True)
        assert s.validate_value(None) == []


class TestArraySchema:
    def test_basic(self):
        s = ArraySchema(items=StringSchema("item"))
        js = s.to_json_schema()
        assert js["type"] == "array"
        assert js["items"]["type"] == "string"
        assert s.validate_value(["a", "b"]) == []

    def test_empty_items_validates(self):
        s = ArraySchema(items=StringSchema("item"))
        assert s.validate_value([]) == []

    def test_min_max_items(self):
        s = ArraySchema(items=StringSchema("x"), min_items=1, max_items=2)
        assert len(s.validate_value([])) > 0
        assert s.validate_value(["a"]) == []
        assert s.validate_value(["a", "b"]) == []
        assert len(s.validate_value(["a", "b", "c"])) > 0

    def test_nullable(self):
        s = ArraySchema(nullable=True)
        assert s.validate_value(None) == []


class TestObjectSchema:
    def test_basic(self):
        s = ObjectSchema(properties={"name": StringSchema("n")})
        js = s.to_json_schema()
        assert js["type"] == "object"
        assert "name" in js["properties"]
        assert s.validate_value({"name": "test"}) == []

    def test_required(self):
        s = ObjectSchema(properties={"x": StringSchema("x")}, required=["x"])
        assert s.validate_value({"x": "hello"}) == []
        errors = s.validate_value({})
        assert len(errors) > 0
        assert any("missing" in e.lower() for e in errors)

    def test_nested_object(self):
        s = ObjectSchema(
            properties={
                "inner": ObjectSchema(
                    properties={"val": IntegerSchema(0)}, required=["val"]
                ),
            },
            required=["inner"],
        )
        assert s.validate_value({"inner": {"val": 42}}) == []
        errors = s.validate_value({"inner": {}})
        assert len(errors) > 0

    def test_nullable(self):
        s = ObjectSchema(nullable=True)
        assert s.validate_value(None) == []

    def test_additional_properties(self):
        s = ObjectSchema(additional_properties=False)
        js = s.to_json_schema()
        assert js["additionalProperties"] is False


class TestToolParametersSchema:
    def test_builds_object_schema(self):
        schema = tool_parameters_schema(
            name=StringSchema("tool name"),
            required=["name"],
        )
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["required"] == ["name"]

    def test_empty_schema(self):
        schema = tool_parameters_schema()
        assert schema["type"] == "object"
        assert schema["properties"] == {}


class TestSchemaStaticMethods:
    def test_resolve_type_simple(self):
        assert Schema.resolve_json_schema_type("string") == "string"
        assert Schema.resolve_json_schema_type(["string", "null"]) == "string"
        assert Schema.resolve_json_schema_type(["null", "integer"]) == "integer"

    def test_validate_json_schema_value_type_mismatch(self):
        errors = Schema.validate_json_schema_value(123, {"type": "string"}, "param")
        assert len(errors) > 0

    def test_validate_json_schema_value_ok(self):
        assert Schema.validate_json_schema_value("hi", {"type": "string"}) == []

    def test_fragment_from_dict(self):
        d = {"type": "string"}
        assert Schema.fragment(d) == d

    def test_fragment_from_schema_instance(self):
        s = StringSchema("test")
        assert Schema.fragment(s) == s.to_json_schema()
