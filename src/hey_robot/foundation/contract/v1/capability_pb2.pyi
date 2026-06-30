from collections.abc import Mapping as _Mapping
from typing import (
    ClassVar as _ClassVar,
)

from google.protobuf import (
    descriptor as _descriptor,
    message as _message,
    struct_pb2 as _struct_pb2,
)

DESCRIPTOR: _descriptor.FileDescriptor

class GetHealthRequest(_message.Message):
    __slots__ = ("service_id",)
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    def __init__(self, service_id: str | None = ...) -> None: ...

class GetHealthResponse(_message.Message):
    __slots__ = (
        "busy",
        "current_skill_id",
        "error_code",
        "error_message",
        "loaded",
        "metrics",
        "name",
        "online",
        "robot_id",
        "service_id",
        "version",
    )
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ONLINE_FIELD_NUMBER: _ClassVar[int]
    LOADED_FIELD_NUMBER: _ClassVar[int]
    BUSY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SKILL_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    name: str
    robot_id: str
    online: bool
    loaded: bool
    busy: bool
    current_skill_id: str
    error_code: str
    error_message: str
    metrics: _struct_pb2.Struct
    version: str
    def __init__(
        self,
        service_id: str | None = ...,
        name: str | None = ...,
        robot_id: str | None = ...,
        online: bool = ...,
        loaded: bool = ...,
        busy: bool = ...,
        current_skill_id: str | None = ...,
        error_code: str | None = ...,
        error_message: str | None = ...,
        metrics: _struct_pb2.Struct | _Mapping | None = ...,
        version: str | None = ...,
    ) -> None: ...

class ExecuteCapabilityRequest(_message.Message):
    __slots__ = (
        "arguments",
        "episode_id",
        "metadata",
        "objective",
        "robot_id",
        "service_id",
        "skill_id",
        "skill_name",
        "timeout_sec",
        "trace_id",
    )
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    SKILL_ID_FIELD_NUMBER: _ClassVar[int]
    SKILL_NAME_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    trace_id: str
    episode_id: str
    skill_id: str
    skill_name: str
    robot_id: str
    objective: str
    arguments: _struct_pb2.Struct
    timeout_sec: float
    metadata: _struct_pb2.Struct
    def __init__(
        self,
        service_id: str | None = ...,
        trace_id: str | None = ...,
        episode_id: str | None = ...,
        skill_id: str | None = ...,
        skill_name: str | None = ...,
        robot_id: str | None = ...,
        objective: str | None = ...,
        arguments: _struct_pb2.Struct | _Mapping | None = ...,
        timeout_sec: float | None = ...,
        metadata: _struct_pb2.Struct | _Mapping | None = ...,
    ) -> None: ...

class ExecuteCapabilityResponse(_message.Message):
    __slots__ = (
        "error_code",
        "error_message",
        "failure_mode",
        "metrics",
        "status",
        "success",
        "summary",
    )
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    FAILURE_MODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    status: str
    summary: str
    failure_mode: str
    error_code: str
    error_message: str
    metrics: _struct_pb2.Struct
    def __init__(
        self,
        success: bool = ...,
        status: str | None = ...,
        summary: str | None = ...,
        failure_mode: str | None = ...,
        error_code: str | None = ...,
        error_message: str | None = ...,
        metrics: _struct_pb2.Struct | _Mapping | None = ...,
    ) -> None: ...

class CancelCapabilityRequest(_message.Message):
    __slots__ = ("service_id", "skill_id")
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SKILL_ID_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    skill_id: str
    def __init__(
        self, service_id: str | None = ..., skill_id: str | None = ...
    ) -> None: ...

class CancelCapabilityResponse(_message.Message):
    __slots__ = ("accepted", "error_code", "error_message", "summary")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    summary: str
    error_code: str
    error_message: str
    def __init__(
        self,
        accepted: bool = ...,
        summary: str | None = ...,
        error_code: str | None = ...,
        error_message: str | None = ...,
    ) -> None: ...
