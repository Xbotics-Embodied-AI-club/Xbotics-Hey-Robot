"""Class-based robot agent tools: one :class:`Tool` subclass per file.

Foundation
----------
* :class:`Tool`: abstract base class
* :class:`ToolContext`: dependency-injection container
* :class:`ToolRegistry`: class-based registry with cast+validate pipeline
* :class:`ToolLoader`: auto-discovery via ``pkgutil``

Schema types
------------
* :class:`StringSchema`, :class:`IntegerSchema`, :class:`NumberSchema`
* :class:`BooleanSchema`, :class:`ArraySchema`, :class:`ObjectSchema`
* :func:`tool_parameters_schema`: convenience builder
* :func:`tool_parameters`: class decorator
"""

from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext, ToolTurnContext
from hey_robot.agents.tools.loader import ToolLoader
from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.agents.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

__all__ = [
    "ArraySchema",
    "BooleanSchema",
    "IntegerSchema",
    "NumberSchema",
    "ObjectSchema",
    "StringSchema",
    "Tool",
    "ToolContext",
    "ToolLoader",
    "ToolRegistry",
    "ToolTurnContext",
    "tool_parameters",
    "tool_parameters_schema",
]
