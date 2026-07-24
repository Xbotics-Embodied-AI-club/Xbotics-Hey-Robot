from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EmptyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("online", "loaded", "busy", "error_message", "metrics")
    ONLINE_FIELD_NUMBER: _ClassVar[int]
    LOADED_FIELD_NUMBER: _ClassVar[int]
    BUSY_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    online: bool
    loaded: bool
    busy: bool
    error_message: str
    metrics: _struct_pb2.Struct
    def __init__(
        self,
        online: bool = ...,
        loaded: bool = ...,
        busy: bool = ...,
        error_message: _Optional[str] = ...,
        metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class BeginTrialRequest(_message.Message):
    __slots__ = ("trial_id", "task", "seed", "split", "registries")
    TRIAL_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    REGISTRIES_FIELD_NUMBER: _ClassVar[int]
    trial_id: str
    task: str
    seed: int
    split: str
    registries: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        trial_id: _Optional[str] = ...,
        task: _Optional[str] = ...,
        seed: _Optional[int] = ...,
        split: _Optional[str] = ...,
        registries: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ImageFrame(_message.Message):
    __slots__ = ("camera", "data", "content_type", "width", "height")
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    camera: str
    data: bytes
    content_type: str
    width: int
    height: int
    def __init__(
        self,
        camera: _Optional[str] = ...,
        data: _Optional[bytes] = ...,
        content_type: _Optional[str] = ...,
        width: _Optional[int] = ...,
        height: _Optional[int] = ...,
    ) -> None: ...

class ObservationResponse(_message.Message):
    __slots__ = ("trial_id", "frame_id", "state", "images", "task", "done", "metadata")
    TRIAL_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    trial_id: str
    frame_id: int
    state: _containers.RepeatedScalarFieldContainer[float]
    images: _containers.RepeatedCompositeFieldContainer[ImageFrame]
    task: str
    done: bool
    metadata: _struct_pb2.Struct
    def __init__(
        self,
        trial_id: _Optional[str] = ...,
        frame_id: _Optional[int] = ...,
        state: _Optional[_Iterable[float]] = ...,
        images: _Optional[_Iterable[_Union[ImageFrame, _Mapping]]] = ...,
        task: _Optional[str] = ...,
        done: bool = ...,
        metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("action", "expected_frame_id", "raw_action", "action_clipped")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    RAW_ACTION_FIELD_NUMBER: _ClassVar[int]
    ACTION_CLIPPED_FIELD_NUMBER: _ClassVar[int]
    action: _containers.RepeatedScalarFieldContainer[float]
    expected_frame_id: int
    raw_action: _containers.RepeatedScalarFieldContainer[float]
    action_clipped: bool
    def __init__(
        self,
        action: _Optional[_Iterable[float]] = ...,
        expected_frame_id: _Optional[int] = ...,
        raw_action: _Optional[_Iterable[float]] = ...,
        action_clipped: bool = ...,
    ) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("observation", "reward", "done", "metrics")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    observation: ObservationResponse
    reward: float
    done: bool
    metrics: _struct_pb2.Struct
    def __init__(
        self,
        observation: _Optional[_Union[ObservationResponse, _Mapping]] = ...,
        reward: _Optional[float] = ...,
        done: bool = ...,
        metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class TruthResponse(_message.Message):
    __slots__ = ("done", "official_success", "frame_id", "metrics")
    DONE_FIELD_NUMBER: _ClassVar[int]
    OFFICIAL_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    done: bool
    official_success: bool
    frame_id: int
    metrics: _struct_pb2.Struct
    def __init__(
        self,
        done: bool = ...,
        official_success: bool = ...,
        frame_id: _Optional[int] = ...,
        metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class EndTrialRequest(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class EndTrialResponse(_message.Message):
    __slots__ = ("ended",)
    ENDED_FIELD_NUMBER: _ClassVar[int]
    ended: bool
    def __init__(self, ended: bool = ...) -> None: ...
