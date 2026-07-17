"""基础模型服务的 RPC 契约。"""

from hey_robot.foundation.contract.v1 import (
    CancelSkillRequest,
    CancelSkillResponse,
    ExecuteSkillRequest,
    ExecuteSkillResponse,
    GetHealthRequest,
    GetHealthResponse,
    ModelService,
    ModelServiceServicer,
    ModelServiceStub,
    add_ModelServiceServicer_to_server,
)

__all__ = [
    "CancelSkillRequest",
    "CancelSkillResponse",
    "ExecuteSkillRequest",
    "ExecuteSkillResponse",
    "GetHealthRequest",
    "GetHealthResponse",
    "ModelService",
    "ModelServiceServicer",
    "ModelServiceStub",
    "add_ModelServiceServicer_to_server",
]
