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
    __slots__ = (
        "trial_id",
        "task",
        "seed",
        "split",
        "registries",
        "execution_artifact_dir",
    )
    TRIAL_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    REGISTRIES_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ARTIFACT_DIR_FIELD_NUMBER: _ClassVar[int]
    trial_id: str
    task: str
    seed: int
    split: str
    registries: _containers.RepeatedScalarFieldContainer[str]
    execution_artifact_dir: str
    def __init__(
        self,
        trial_id: _Optional[str] = ...,
        task: _Optional[str] = ...,
        seed: _Optional[int] = ...,
        split: _Optional[str] = ...,
        registries: _Optional[_Iterable[str]] = ...,
        execution_artifact_dir: _Optional[str] = ...,
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
    __slots__ = ("session_id", "instruction", "max_actions", "reset_session")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    RESET_SESSION_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    instruction: str
    max_actions: int
    reset_session: bool
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        instruction: _Optional[str] = ...,
        max_actions: _Optional[int] = ...,
        reset_session: bool = ...,
    ) -> None: ...

class StepResponse(_message.Message):
    __slots__ = (
        "observation",
        "status",
        "done",
        "actions_executed",
        "chunks_executed",
        "progress",
        "diagnostics",
        "error_message",
    )
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_EXECUTED_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_EXECUTED_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTICS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    observation: ObservationResponse
    status: str
    done: bool
    actions_executed: int
    chunks_executed: int
    progress: _struct_pb2.Struct
    diagnostics: _struct_pb2.Struct
    error_message: str
    def __init__(
        self,
        observation: _Optional[_Union[ObservationResponse, _Mapping]] = ...,
        status: _Optional[str] = ...,
        done: bool = ...,
        actions_executed: _Optional[int] = ...,
        chunks_executed: _Optional[int] = ...,
        progress: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        diagnostics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        error_message: _Optional[str] = ...,
    ) -> None: ...

class NativeStepRequest(_message.Message):
    __slots__ = ("action", "expected_frame_id")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    action: _containers.RepeatedScalarFieldContainer[float]
    expected_frame_id: int
    def __init__(
        self,
        action: _Optional[_Iterable[float]] = ...,
        expected_frame_id: _Optional[int] = ...,
    ) -> None: ...

class NativeStepResponse(_message.Message):
    __slots__ = ("observation", "done", "progress")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    observation: ObservationResponse
    done: bool
    progress: _struct_pb2.Struct
    def __init__(
        self,
        observation: _Optional[_Union[ObservationResponse, _Mapping]] = ...,
        done: bool = ...,
        progress: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class ImagePixel(_message.Message):
    __slots__ = ("row", "col")
    ROW_FIELD_NUMBER: _ClassVar[int]
    COL_FIELD_NUMBER: _ClassVar[int]
    row: int
    col: int
    def __init__(
        self, row: _Optional[int] = ..., col: _Optional[int] = ...
    ) -> None: ...

class LocalizePixelsRequest(_message.Message):
    __slots__ = ("camera", "pixels", "expected_frame_id")
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    PIXELS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    camera: str
    pixels: _containers.RepeatedCompositeFieldContainer[ImagePixel]
    expected_frame_id: int
    def __init__(
        self,
        camera: _Optional[str] = ...,
        pixels: _Optional[_Iterable[_Union[ImagePixel, _Mapping]]] = ...,
        expected_frame_id: _Optional[int] = ...,
    ) -> None: ...

class LocalizePixelsResponse(_message.Message):
    __slots__ = ("frame_id", "camera", "localization")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    camera: str
    localization: _struct_pb2.Struct
    def __init__(
        self,
        frame_id: _Optional[int] = ...,
        camera: _Optional[str] = ...,
        localization: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
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
