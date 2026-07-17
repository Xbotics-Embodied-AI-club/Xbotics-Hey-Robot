"""Canonical message types for the deployable Hey Robot runtime.

These dataclasses are the boundary between independently deployed services.
Channels, agents, policies, and robot drivers exchange these shapes instead of
ad hoc dictionaries.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Envelope:
    trace_id: str = field(default_factory=lambda: _new_id("tr"))
    episode_id: str | None = None
    turn_id: str | None = None
    channel: str | None = None
    account_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    chat_type: str | None = None
    sender_id: str | None = None
    message_id: str | None = None
    reply_to_id: str | None = None
    robot_id: str | None = None
    agent_id: str | None = None
    deployment_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def child(self, **updates: Any) -> Envelope:
        data = asdict(self)
        data.update(updates)
        if not data.get("trace_id"):
            data["trace_id"] = _new_id("tr")
        return Envelope(**data)


@dataclass(frozen=True)
class MediaRef:
    uri: str
    media_type: str
    name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageRef:
    uri: str
    camera: str | None = None
    width: int | None = None
    height: int | None = None
    timestamp: float | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    artifact_type: str
    role: str | None = None
    name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserTurn:
    envelope: Envelope
    text: str
    media: list[MediaRef] = field(default_factory=list)
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTurn:
    envelope: Envelope
    session_key: str
    interaction_id: str
    text: str


@dataclass(frozen=True)
class ConversationResult:
    envelope: Envelope
    interaction_id: str
    text: str


@dataclass(frozen=True)
class ToolOutcome:
    """Trusted structured result returned to a conversation tool loop."""

    status: Literal["completed", "failed", "waiting", "accepted"]
    user_summary: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    operation_id: str | None = None
    goal_id: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class ShortOperationCommand:
    """Conversation request for one bounded operation, admitted by Supervisor."""

    envelope: Envelope
    operation_id: str
    proposal: ActionProposal
    timeout_sec: float = 45.0


@dataclass(frozen=True)
class AgentReply:
    envelope: Envelope
    text: str
    media: list[MediaRef] = field(default_factory=list)
    final: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneRelation:
    """场景实体之间的一条开放式关系。"""

    predicate: str
    object_id: str


@dataclass(frozen=True)
class SceneEntity:
    """仅用于目标解析的帧级视觉实体。"""

    entity_id: str
    entity_type: str
    frame_id: int
    attributes: dict[str, Any] = field(default_factory=dict)
    relations: list[SceneRelation] = field(default_factory=list)


@dataclass(frozen=True)
class RobotObservation:
    envelope: Envelope
    frame_id: int
    images: list[ImageRef] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    proprioception: list[float] = field(default_factory=list)
    task: str | None = None
    entities: list[SceneEntity] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotStatus:
    envelope: Envelope
    frame_id: int | None = None
    state: Literal["idle", "executing", "error", "offline", "unknown"] = "unknown"
    location_id: str | None = None
    motion_state: Literal["idle", "moving", "stopped", "unknown"] = "unknown"
    battery_percentage: float | None = None
    task: str | None = None
    skill_id: str | None = None
    success: bool | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillIntent:
    envelope: Envelope
    skill_id: str
    goal_id: str
    task_id: str
    deliberation_id: str
    intent_kind: Literal["skill", "observation"]
    name: str
    arguments: dict[str, Any]
    objective: str
    priority: int = 0
    timeout_sec: float | None = None
    feedback_mode: str = "status"


@dataclass(frozen=True)
class SkillEvent:
    envelope: Envelope
    skill_id: str
    name: str = ""
    phase: str = "created"
    step: str | None = None
    text: str | None = None
    mode: str | None = None
    policy_id: str | None = None
    steps_executed: int | None = None
    frame_id: int | None = None
    progress: float | None = None
    error: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotAction:
    envelope: Envelope
    values: list[float]
    action_id: str = field(default_factory=lambda: _new_id("act"))
    skill_id: str = ""
    goal_id: str = ""
    task_id: str = ""
    deliberation_id: str = ""
    intent_kind: Literal["skill", "observation"] = "skill"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


CriterionType = Literal["robot_state", "object_relation", "evidence_present"]
CriterionPredicate = Literal["equals", "at", "near", "inside", "held_by", "observed"]


@dataclass(frozen=True)
class GoalBudgets:
    max_wall_time_sec: float = 1800.0
    max_deliberations: int = 20
    max_skills: int = 12
    min_battery_percentage: float = 20.0


@dataclass(frozen=True)
class SuccessCriterion:
    criterion_id: str
    criterion_type: CriterionType
    subject_id: str
    predicate: CriterionPredicate
    object_id: str
    max_age_sec: float


@dataclass(frozen=True)
class EvidenceFact:
    evidence_id: str
    goal_id: str
    source_kind: Literal["robot_status", "skill_result"]
    source_id: str
    observed_at: float
    frame_id: int | None
    subject_id: str
    predicate: CriterionPredicate
    object_id: str
    artifacts: tuple[ArtifactRef | ImageRef, ...] = ()


@dataclass(frozen=True)
class GoalCommand:
    envelope: Envelope
    command_id: str
    action: Literal["create", "cancel", "emergency_stop", "confirm", "reconcile"]
    goal_id: str | None = None
    condition_id: str | None = None
    skill_id: str | None = None
    objective: str = ""
    success_criteria: tuple[SuccessCriterion, ...] = ()
    budgets: GoalBudgets = field(default_factory=GoalBudgets)


@dataclass(frozen=True)
class GoalSnapshot:
    goal_id: str
    version: int
    task_id: str
    contract_id: str
    contract_hash: str
    objective: str
    success_criteria: tuple[SuccessCriterion, ...]
    status: Literal[
        "pending",
        "active",
        "waiting",
        "waiting_condition",
        "needs_review",
        "blocked",
        "completed",
        "failed",
        "cancelled",
    ]
    termination_reason: Literal["cancel", "budget", "emergency"] | None = None


@dataclass(frozen=True)
class ActionSnapshot:
    skill_id: str
    deliberation_id: str
    intent_kind: Literal["skill", "observation"]
    name: str
    objective: str
    arguments: dict[str, Any]
    status: Literal[
        "persisted",
        "publishing",
        "published",
        "accepted",
        "running",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
        "unknown",
        "reconciled_idle",
    ]


@dataclass(frozen=True)
class BudgetState:
    elapsed_wall_time_sec: float
    deliberations_used: int
    skills_used: int
    battery_percentage: float | None


@dataclass(frozen=True)
class ActionProposal:
    intent_kind: Literal["skill", "observation"]
    skill_name: str
    objective: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FailurePayload:
    stage: str
    code: str
    component: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskEvaluationPayload:
    outcome: Literal["satisfied", "inconclusive"]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    missing_criteria_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillResult:
    envelope: Envelope
    skill_id: str
    name: str = ""
    status: Literal["completed", "failed", "interrupted", "unknown"] = "unknown"
    success: bool | None = None
    steps_executed: int = 0
    progress: float = 0.0
    summary: str | None = None
    failure_mode: str | None = None
    frame_id: int | None = None
    error: str | None = None
    observations: list[ImageRef] = field(default_factory=list)
    evidence: tuple[EvidenceFact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliberationRequest:
    envelope: Envelope
    deliberation_id: str
    trigger_event_id: str
    goal: GoalSnapshot
    robot_status: RobotStatus | None
    robot_observation: RobotObservation | None
    actions: tuple[ActionSnapshot, ...]
    evidence: tuple[EvidenceFact, ...]
    latest_skill_result: SkillResult | None
    budget_state: BudgetState


@dataclass(frozen=True)
class DeliberationResult:
    envelope: Envelope
    deliberation_id: str
    request_hash: str
    goal_id: str
    task_id: str
    status: Literal["completed", "action_proposed", "failed"]
    proposal: ActionProposal | None = None
    failure: FailurePayload | None = None
    evaluation: TaskEvaluationPayload | None = None


@dataclass(frozen=True)
class GoalEvent:
    envelope: Envelope
    event_id: str
    goal_id: str
    task_id: str
    status: Literal[
        "pending",
        "active",
        "waiting",
        "waiting_condition",
        "needs_review",
        "blocked",
        "completed",
        "failed",
        "cancelled",
    ]
    active_skill_id: str | None = None
    active_control_id: str | None = None
    termination_reason: Literal["cancel", "budget", "emergency"] | None = None
    evaluation: TaskEvaluationPayload | None = None
    failure: FailurePayload | None = None


@dataclass(frozen=True)
class SkillControl:
    envelope: Envelope
    control_id: str
    action: Literal["interrupt", "emergency_stop"]
    target_skill_id: str | None
    goal_id: str | None
    reason: str


@dataclass(frozen=True)
class SkillControlResult:
    envelope: Envelope
    control_id: str
    action: Literal["interrupt", "emergency_stop"]
    target_skill_id: str | None
    status: Literal["completed", "failed", "unknown"]
    robot_idle_confirmed: bool
    error: str | None = None


@dataclass(frozen=True)
class RobotExecutionGate:
    robot_id: str
    version: int
    state: Literal["ready", "stop_pending", "uncertain"]
    control_id: str | None = None
    reason: str | None = None
    updated_at: float = 0.0


def to_payload(message: DataclassInstance) -> dict[str, Any]:
    return asdict(message)


def from_payload[T](cls: type[T], payload: dict[str, Any]) -> T:
    """Decode a protocol message without accepting unknown or malformed fields."""
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} payload must be an object")
    known = {item.name for item in fields(cls)}  # type: ignore[arg-type]
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):  # type: ignore[arg-type]
        if item.name not in payload:
            if item.default is MISSING and item.default_factory is MISSING:
                raise ValueError(f"{cls.__name__} missing required field: {item.name}")
            continue
        kwargs[item.name] = _decode_value(
            payload[item.name], hints.get(item.name, item.type)
        )
    result = cls(**kwargs)
    _validate_message(result)
    return result


def _decode_value(value: Any, target: Any) -> Any:
    if value is None:
        if type(None) in get_args(target):
            return None
        raise TypeError(f"expected {target!r}, got null")
    origin = get_origin(target)
    args = get_args(target)
    if origin in (Union, UnionType):
        errors: list[str] = []
        for subtype in args:
            if subtype is type(None):
                continue
            try:
                return _decode_value(value, subtype)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(" | ".join(errors) or f"invalid union value {value!r}")
    if origin is Literal:
        if value not in args:
            raise ValueError(f"invalid enum {value!r}; expected one of {args!r}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected array for {target!r}")
        subtype = args[0] if args else Any
        decoded = [_decode_value(entry, subtype) for entry in value]
        return tuple(decoded) if origin is tuple else decoded
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"expected object for {target!r}")
        return dict(value)
    if target is Any:
        return value
    if isinstance(target, type) and is_dataclass(target):
        if isinstance(value, target):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"expected object for {target.__name__}")
        return from_payload(target, value)
    if (
        target is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    if target is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if target is bool and isinstance(value, bool):
        return value
    if target is str and isinstance(value, str):
        return value
    if isinstance(target, type) and isinstance(value, target):
        return value
    raise TypeError(f"expected {target!r}, got {type(value).__name__}")


def _validate_message(message: Any) -> None:
    if isinstance(message, SuccessCriterion):
        allowed = {
            "robot_state": {"equals"},
            "object_relation": {"at", "near", "inside", "held_by"},
            "evidence_present": {"observed"},
        }
        if message.predicate not in allowed[message.criterion_type]:
            raise ValueError(
                f"criterion predicate {message.predicate!r} is invalid for {message.criterion_type!r}"
            )
        if not message.criterion_id or not message.subject_id or not message.object_id:
            raise ValueError("SuccessCriterion ids must be non-empty")
        if message.max_age_sec <= 0:
            raise ValueError("SuccessCriterion max_age_sec must be positive")
    if isinstance(message, EvidenceFact):
        if not message.evidence_id or not message.goal_id or not message.source_id:
            raise ValueError("EvidenceFact identity fields must be non-empty")
        if message.source_kind == "robot_status":
            if message.frame_id is None:
                raise ValueError("robot_status evidence requires frame_id")
            expected = f"status:{message.source_id.split(':')[1] if message.source_id.startswith('status:') else ''}:{message.frame_id}"
            if message.source_id != expected:
                raise ValueError(
                    "robot_status evidence source_id must match status:<robot_id>:<frame_id>"
                )
    if isinstance(message, GoalCommand):
        if not message.command_id:
            raise ValueError("GoalCommand command_id must be non-empty")
        if message.action == "create":
            if not message.envelope.robot_id:
                raise ValueError("create GoalCommand requires robot_id")
            if not message.objective or not message.success_criteria:
                raise ValueError(
                    "create GoalCommand requires objective and success_criteria"
                )
        elif message.action == "cancel" and not message.goal_id:
            raise ValueError("cancel GoalCommand requires goal_id")
        elif message.action == "emergency_stop" and not message.envelope.robot_id:
            raise ValueError("emergency_stop GoalCommand requires robot_id")
        elif message.action == "reconcile" and (
            not message.envelope.robot_id or not message.skill_id
        ):
            raise ValueError("reconcile GoalCommand requires robot_id and skill_id")
    if isinstance(message, SkillResult):
        if message.status == "completed" and message.success is not True:
            raise ValueError("completed SkillResult requires success=True")
        if message.status in {"failed", "interrupted"} and message.success is not False:
            raise ValueError(f"{message.status} SkillResult requires success=False")
        if message.status == "unknown" and message.success is not None:
            raise ValueError("unknown SkillResult requires success=None")
        for fact in message.evidence:
            if fact.source_kind != "skill_result" or fact.source_id != message.skill_id:
                raise ValueError(
                    "SkillResult evidence must be sourced by the result skill_id"
                )
    if isinstance(message, DeliberationResult):
        if message.status == "action_proposed" and message.proposal is None:
            raise ValueError("action_proposed DeliberationResult requires proposal")
        if message.status != "action_proposed" and message.proposal is not None:
            raise ValueError(
                "only action_proposed DeliberationResult may contain proposal"
            )
        if message.status == "completed" and (
            message.evaluation is None or message.evaluation.outcome != "satisfied"
        ):
            raise ValueError(
                "completed DeliberationResult requires satisfied evaluation"
            )
        if message.status == "failed" and message.failure is None:
            raise ValueError("failed DeliberationResult requires failure")
    if isinstance(message, SkillControlResult):
        if message.status == "completed" and not message.robot_idle_confirmed:
            raise ValueError("completed SkillControlResult requires idle confirmation")
        if message.status != "completed" and message.robot_idle_confirmed:
            raise ValueError("non-completed SkillControlResult cannot confirm idle")
