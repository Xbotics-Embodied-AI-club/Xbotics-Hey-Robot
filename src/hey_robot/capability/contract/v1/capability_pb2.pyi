from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetHealthRequest(_message.Message):
    __slots__ = ("service_id",)
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    def __init__(self, service_id: _Optional[str] = ...) -> None: ...

class GetHealthResponse(_message.Message):
    __slots__ = ("service_id", "name", "robot_id", "online", "loaded", "busy", "current_skill_id", "error_code", "error_message", "metrics", "version")
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
    def __init__(self, service_id: _Optional[str] = ..., name: _Optional[str] = ..., robot_id: _Optional[str] = ..., online: bool = ..., loaded: bool = ..., busy: bool = ..., current_skill_id: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., version: _Optional[str] = ...) -> None: ...

class ExecuteCapabilityRequest(_message.Message):
    __slots__ = ("service_id", "trace_id", "episode_id", "skill_id", "skill_name", "robot_id", "objective", "arguments", "timeout_sec", "metadata")
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
    def __init__(self, service_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., episode_id: _Optional[str] = ..., skill_id: _Optional[str] = ..., skill_name: _Optional[str] = ..., robot_id: _Optional[str] = ..., objective: _Optional[str] = ..., arguments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., timeout_sec: _Optional[float] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExecuteCapabilityResponse(_message.Message):
    __slots__ = ("success", "status", "summary", "failure_mode", "error_code", "error_message", "metrics")
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
    def __init__(self, success: bool = ..., status: _Optional[str] = ..., summary: _Optional[str] = ..., failure_mode: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CancelCapabilityRequest(_message.Message):
    __slots__ = ("service_id", "skill_id")
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SKILL_ID_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    skill_id: str
    def __init__(self, service_id: _Optional[str] = ..., skill_id: _Optional[str] = ...) -> None: ...

class CancelCapabilityResponse(_message.Message):
    __slots__ = ("accepted", "summary", "error_code", "error_message")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    summary: str
    error_code: str
    error_message: str
    def __init__(self, accepted: bool = ..., summary: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...
