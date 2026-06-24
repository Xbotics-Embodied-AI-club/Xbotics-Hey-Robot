from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.skills.registry import load_skill_registry

EvidenceStrength = Literal["weak", "strong", "status", "operator"]


@dataclass(frozen=True)
class CapabilityRequirement:
    type: str
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "constraints": dict(self.constraints)}


@dataclass(frozen=True)
class TaskContract:
    task_type: str
    user_goal: str
    required_capability: CapabilityRequirement | None = None
    completion_evidence_required: tuple[str, ...] = ()
    allowed_supporting_capabilities: tuple[str, ...] = ()

    @property
    def requires_capability(self) -> bool:
        return self.required_capability is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "user_goal": self.user_goal,
            "required_capability": self.required_capability.to_dict()
            if self.required_capability is not None
            else None,
            "completion_evidence_required": list(self.completion_evidence_required),
            "allowed_supporting_capabilities": list(
                self.allowed_supporting_capabilities
            ),
        }


@dataclass(frozen=True)
class EvidenceRecord:
    source_tool: str
    capability: str
    capability_type: str
    evidence_type: str
    strength: EvidenceStrength
    success: bool
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "capability": self.capability,
            "capability_type": self.capability_type,
            "evidence_type": self.evidence_type,
            "strength": self.strength,
            "success": self.success,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class EvaluationResult:
    can_finalize: bool
    goal_satisfied: bool
    missing_evidence: tuple[str, ...] = ()
    next_capability_type: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_finalize": self.can_finalize,
            "goal_satisfied": self.goal_satisfied,
            "missing_evidence": list(self.missing_evidence),
            "next_capability_type": self.next_capability_type,
            "reason": self.reason,
        }

    def feedback_for_agent(self, contract: TaskContract) -> str:
        lines = [
            "Task evidence evaluator:",
            f"- user_goal: {contract.user_goal}",
            f"- task_type: {contract.task_type}",
            f"- goal_satisfied: {self.goal_satisfied}",
            f"- can_finalize: {self.can_finalize}",
        ]
        if self.missing_evidence:
            lines.append(f"- missing_evidence: {', '.join(self.missing_evidence)}")
        if self.next_capability_type:
            lines.append(f"- next_capability_type: {self.next_capability_type}")
        if self.reason:
            lines.append(f"- reason: {self.reason}")
        lines.append(
            "- instruction: Continue with the next useful capability call, or explain a concrete safety/capability refusal. Do not final-answer as if the task is complete."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class SkillEvidenceSemantics:
    name: str
    capability_type: str
    evidence_outputs: tuple[str, ...] = ()
    cannot_satisfy: tuple[str, ...] = ()


class EvidenceLedger:
    def __init__(
        self, semantics: dict[str, SkillEvidenceSemantics] | None = None
    ) -> None:
        self.records: list[EvidenceRecord] = []
        self.semantics = semantics or default_skill_semantics()

    def add_tool_result(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: str,
        success: bool,
    ) -> None:
        capability = _tool_capability(tool, args)
        capability_type = capability_type_for_name(capability, self.semantics)
        evidence_type = evidence_type_for_capability_type(
            capability_type, self.semantics
        )
        self.records.append(
            EvidenceRecord(
                source_tool=tool,
                capability=capability,
                capability_type=capability_type,
                evidence_type=evidence_type,
                strength=evidence_strength_for_capability_type(capability_type),
                success=success,
                summary=str(result or "")[:500],
            )
        )

    def has_successful_evidence(self, evidence_type: str) -> bool:
        return any(
            record.success and record.evidence_type == evidence_type
            for record in self.records
        )

    def has_successful_capability_type(self, capability_type: str) -> bool:
        return any(
            record.success and record.capability_type == capability_type
            for record in self.records
        )

    def has_attempted_capability_type(self, capability_type: str) -> bool:
        return any(record.capability_type == capability_type for record in self.records)

    def has_failed_capability_type(self, capability_type: str) -> bool:
        return any(
            not record.success and record.capability_type == capability_type
            for record in self.records
        )

    def context_for_agent(self, limit: int = 8) -> str:
        if not self.records:
            return ""
        lines = ["Task evidence ledger:"]
        for record in self.records[-limit:]:
            status = "ok" if record.success else "failed"
            lines.append(
                f"- {record.capability} [{record.capability_type}/{record.evidence_type}/{record.strength}] -> {status}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


class TaskEvidenceEvaluator:
    def evaluate(
        self, contract: TaskContract, ledger: EvidenceLedger
    ) -> EvaluationResult:
        if not contract.requires_capability:
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=True,
                reason="no required capability for this task contract",
            )

        assert contract.required_capability is not None
        required_type = contract.required_capability.type
        required_evidence = contract.completion_evidence_required

        if ledger.has_successful_capability_type(required_type):
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=True,
                reason=f"required capability {required_type} completed",
            )

        missing = tuple(
            evidence
            for evidence in required_evidence
            if not ledger.has_successful_evidence(evidence)
        ) or (evidence_type_for_capability_type(required_type),)

        if ledger.has_failed_capability_type(required_type):
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=False,
                missing_evidence=missing,
                reason=f"required capability {required_type} was attempted and failed",
            )

        return EvaluationResult(
            can_finalize=False,
            goal_satisfied=False,
            missing_evidence=missing,
            next_capability_type=required_type,
            reason=(
                f"task requires {required_type}, but current evidence only covers "
                "supporting context or unrelated capabilities"
            ),
        )


def build_task_contract(task: str) -> TaskContract:
    text = _normalize(task)
    capability_type = infer_required_capability_type(text)
    if capability_type is None:
        return TaskContract(
            task_type="general",
            user_goal=task,
            allowed_supporting_capabilities=("scene_observation",),
        )
    return TaskContract(
        task_type=task_type_for_capability_type(capability_type),
        user_goal=task,
        required_capability=CapabilityRequirement(
            type=capability_type, constraints=infer_constraints(text, capability_type)
        ),
        completion_evidence_required=(
            evidence_type_for_capability_type(capability_type),
        ),
        allowed_supporting_capabilities=("scene_observation", "motion_safety_check"),
    )


def infer_required_capability_type(text: str) -> str | None:
    if _contains_any(text, ("apriltag", "aruco")) or (
        _contains_any(text, ("marker", "工作区标记", "标记"))
        and _contains_any(
            text,
            (
                "detect",
                "check",
                "find",
                "look for",
                "有没有",
                "是否有",
                "检测",
                "识别",
                "查找",
            ),
        )
    ):
        return "marker_detection"
    if _contains_any(text, ("stop", "halt", "停下", "停止")):
        return "stop_motion"
    if _contains_any(text, ("安全姿态", "恢复姿态", "reset posture")):
        return "reset_posture"
    if _contains_any(text, ("follow", "跟随")):
        return "human_follow"
    if _contains_any(text, ("gripper", "夹爪", "爪")) and _contains_any(
        text, ("open", "close", "张开", "闭合", "打开", "合上", "%")
    ):
        return "gripper_control"
    if _contains_any(text, ("home", "位姿")) and _contains_any(
        text, ("arm", "机械臂", "回到", "恢复")
    ):
        return "arm_pose"
    if _contains_any(
        text, ("joint", "关节", "shoulder", "elbow", "wrist")
    ) and _contains_any(text, ("转", "move", "rotate", "度")):
        return "arm_joint_delta"
    if _contains_any(text, ("arm", "机械臂", "末端", "夹爪")) and _contains_any(
        text,
        (
            "raise",
            "lower",
            "lift",
            "up",
            "down",
            "抬高",
            "降低",
            "上抬",
            "下压",
            "升高",
            "微调",
        ),
    ):
        return "arm_joint_delta"
    if _contains_any(
        text, ("left", "right", "turn", "左转", "右转", "转向", "往左", "往右")
    ):
        return "base_turn"
    if _contains_any(
        text,
        (
            "forward",
            "backward",
            "move",
            "walk",
            "前进",
            "后退",
            "往前",
            "向前",
            "走走",
            "走一些",
        ),
    ):
        return "base_move"
    if _contains_any(
        text,
        (
            "what do you see",
            "look",
            "看看",
            "看一下",
            "看到",
            "观察",
            "画面",
            "前面有什么",
        ),
    ):
        return "scene_observation"
    return None


def task_type_for_capability_type(capability_type: str) -> str:
    if capability_type in {"scene_observation", "marker_detection"}:
        return "observation"
    if capability_type in {"base_move", "base_turn", "human_follow"}:
        return "motion"
    if capability_type in {"gripper_control", "arm_pose", "arm_joint_delta"}:
        return "actuation"
    if capability_type in {"stop_motion", "reset_posture"}:
        return "safety"
    return "general"


@lru_cache(maxsize=1)
def default_skill_semantics() -> dict[str, SkillEvidenceSemantics]:
    semantics: dict[str, SkillEvidenceSemantics] = {}
    for spec in load_skill_registry().robot_skill_catalog().list():
        capability_type = spec.capability_type
        if not capability_type:
            continue
        semantics[spec.name] = SkillEvidenceSemantics(
            name=spec.name,
            capability_type=capability_type,
            evidence_outputs=tuple(spec.evidence_outputs),
            cannot_satisfy=tuple(spec.cannot_satisfy),
        )
    return semantics


def capability_type_for_name(
    name: str, semantics: dict[str, SkillEvidenceSemantics] | None = None
) -> str:
    semantic = (semantics or default_skill_semantics()).get(name)
    if semantic is not None:
        return semantic.capability_type
    mapping = {
        "request_perception": "scene_observation",
        "inspect_scene": "scene_observation",
        "look_around": "scene_observation",
        "detect_marker": "marker_detection",
        "move_base": "base_move",
        "turn_base": "base_turn",
        "human_follow": "human_follow",
        "set_gripper": "gripper_control",
        "set_arm_pose": "arm_pose",
        "move_arm_joints": "arm_joint_delta",
        "stop_motion": "stop_motion",
        "reset_posture": "reset_posture",
    }
    if name in mapping:
        return mapping[name]
    if is_perception_skill_name(name):
        return "scene_observation"
    return name


def evidence_type_for_capability_type(
    capability_type: str, semantics: dict[str, SkillEvidenceSemantics] | None = None
) -> str:
    for semantic in (semantics or default_skill_semantics()).values():
        if semantic.capability_type == capability_type and semantic.evidence_outputs:
            return semantic.evidence_outputs[0]
    mapping = {
        "scene_observation": "weak_scene_observation",
        "marker_detection": "marker_detection_result",
        "base_move": "base_move_action_result",
        "base_turn": "base_turn_action_result",
        "human_follow": "human_follow_action_result",
        "gripper_control": "gripper_action_result",
        "arm_pose": "arm_pose_action_result",
        "arm_joint_delta": "arm_joint_action_result",
        "stop_motion": "stop_motion_result",
        "reset_posture": "reset_posture_result",
    }
    return mapping.get(capability_type, f"{capability_type}_result")


def evidence_strength_for_capability_type(capability_type: str) -> EvidenceStrength:
    if capability_type == "scene_observation":
        return "weak"
    if capability_type == "marker_detection":
        return "strong"
    return "status"


def infer_constraints(text: str, capability_type: str) -> dict[str, Any]:
    if capability_type == "base_turn":
        if _contains_any(text, ("left", "左")):
            return {"direction": "left"}
        if _contains_any(text, ("right", "右")):
            return {"direction": "right"}
    return {}


def _tool_capability(tool: str, args: dict[str, Any]) -> str:
    if tool == "request_capability":
        return str(args.get("capability") or "").strip() or tool
    return tool


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)
