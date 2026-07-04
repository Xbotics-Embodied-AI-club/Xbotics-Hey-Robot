from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from hey_robot.cognition.runtime.grounding import is_perception_skill_name
from hey_robot.skill_os.registry import load_skill_registry

EvidenceStrength = Literal["weak", "strong", "status", "operator"]


@dataclass(frozen=True)
class SkillRequirement:
    name: str
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "constraints": dict(self.constraints)}


@dataclass(frozen=True)
class TaskContract:
    task_type: str
    user_goal: str
    required_skill: SkillRequirement | None = None
    completion_evidence_required: tuple[str, ...] = ()
    allowed_supporting_skills: tuple[str, ...] = ()

    @property
    def requires_skill(self) -> bool:
        return self.required_skill is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "user_goal": self.user_goal,
            "required_skill": self.required_skill.to_dict()
            if self.required_skill is not None
            else None,
            "completion_evidence_required": list(self.completion_evidence_required),
            "allowed_supporting_skills": list(self.allowed_supporting_skills),
        }


@dataclass(frozen=True)
class EvidenceRecord:
    source_tool: str
    skill: str
    evidence_type: str
    strength: EvidenceStrength
    success: bool
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "skill": self.skill,
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
    next_skill: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_finalize": self.can_finalize,
            "goal_satisfied": self.goal_satisfied,
            "missing_evidence": list(self.missing_evidence),
            "next_skill": self.next_skill,
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
        if self.next_skill:
            lines.append(f"- next_skill: {self.next_skill}")
            lines.append(
                f'- recommended_skill_call: request_skill(skill="{self.next_skill}", ...)'
            )
        if self.reason:
            lines.append(f"- reason: {self.reason}")
        lines.append(
            "- instruction: Continue with the next useful request_skill call, "
            "or explain a concrete safety/skill refusal. Do not final-answer "
            "as if the task is complete."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class SkillEvidenceSemantics:
    name: str
    evidence_outputs: tuple[str, ...] = ()
    cannot_satisfy: tuple[str, ...] = ()
    category: str = "general"
    safety_level: str = "normal"


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
        skill = _tool_skill(tool, args)
        evidence_type = evidence_type_for_skill(skill, self.semantics)
        self.records.append(
            EvidenceRecord(
                source_tool=tool,
                skill=skill,
                evidence_type=evidence_type,
                strength=evidence_strength_for_skill(skill, self.semantics),
                success=success,
                summary=str(result or "")[:500],
            )
        )

    def has_successful_evidence(self, evidence_type: str) -> bool:
        return any(
            record.success and record.evidence_type == evidence_type
            for record in self.records
        )

    def has_successful_skill(self, skill: str) -> bool:
        return any(record.success and record.skill == skill for record in self.records)

    def has_attempted_skill(self, skill: str) -> bool:
        return any(record.skill == skill for record in self.records)

    def has_failed_skill(self, skill: str) -> bool:
        return any(
            not record.success and record.skill == skill for record in self.records
        )

    def context_for_agent(self, limit: int = 8) -> str:
        if not self.records:
            return ""
        lines = ["Task evidence ledger:"]
        for record in self.records[-limit:]:
            status = "ok" if record.success else "failed"
            lines.append(
                f"- {record.skill} [{record.evidence_type}/{record.strength}] -> {status}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


class TaskEvidenceEvaluator:
    def evaluate(
        self, contract: TaskContract, ledger: EvidenceLedger
    ) -> EvaluationResult:
        if not contract.requires_skill:
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=True,
                reason="no required skill for this task contract",
            )

        assert contract.required_skill is not None
        required_skill = contract.required_skill.name
        required_evidence = contract.completion_evidence_required

        if ledger.has_successful_skill(required_skill):
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=True,
                reason=f"required skill {required_skill} completed",
            )

        if required_evidence and all(
            ledger.has_successful_evidence(evidence) for evidence in required_evidence
        ):
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=True,
                reason=f"required evidence for skill {required_skill} completed",
            )

        missing = tuple(
            evidence
            for evidence in required_evidence
            if not ledger.has_successful_evidence(evidence)
        ) or (evidence_type_for_skill(required_skill, ledger.semantics),)

        if ledger.has_failed_skill(required_skill):
            return EvaluationResult(
                can_finalize=True,
                goal_satisfied=False,
                missing_evidence=missing,
                reason=f"required skill {required_skill} was attempted and failed",
            )

        return EvaluationResult(
            can_finalize=False,
            goal_satisfied=False,
            missing_evidence=missing,
            next_skill=required_skill,
            reason=(
                f"task requires skill {required_skill}, but current evidence only "
                "covers supporting context or unrelated skills"
            ),
        )


def build_task_contract(task: str) -> TaskContract:
    text = _normalize(task)
    skill_name = infer_required_skill_name(text)
    if skill_name is None:
        return TaskContract(
            task_type="general",
            user_goal=task,
            allowed_supporting_skills=("inspect_scene", "request_perception"),
        )
    return TaskContract(
        task_type=task_type_for_skill(skill_name),
        user_goal=task,
        required_skill=SkillRequirement(
            name=skill_name, constraints=infer_constraints(text, skill_name)
        ),
        completion_evidence_required=(evidence_type_for_skill(skill_name),),
        allowed_supporting_skills=("inspect_scene", "request_perception"),
    )


def infer_required_skill_name(text: str) -> str | None:
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
        return "detect_marker"
    if _contains_any(text, ("stop", "halt", "停下", "停止")):
        return "stop_motion"
    if _contains_any(text, ("安全姿态", "恢复姿态", "reset posture")):
        return "reset_posture"
    if _contains_any(text, ("follow", "跟随")):
        return "human_follow"
    if _contains_any(
        text,
        (
            "approach",
            "move closer to",
            "get closer to",
            "靠近",
            "接近",
            "走近",
        ),
    ):
        return "approach_object"
    if _looks_like_semantic_navigation(text):
        return "navigate_to"
    if _contains_any(text, ("gripper", "夹爪", "爪")) and _contains_any(
        text, ("open", "close", "张开", "闭合", "打开", "合上", "%")
    ):
        return "set_gripper"
    if _contains_any(text, ("home", "位姿")) and _contains_any(
        text, ("arm", "机械臂", "回到", "恢复")
    ):
        return "set_arm_pose"
    if _contains_any(
        text, ("joint", "关节", "shoulder", "elbow", "wrist")
    ) and _contains_any(text, ("转", "move", "rotate", "度")):
        return "move_arm_joints"
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
        return "move_arm_joints"
    if _contains_any(
        text, ("left", "right", "turn", "左转", "右转", "转向", "往左", "往右")
    ):
        return "turn_base"
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
            "走一点",
        ),
    ):
        return "move_base"
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
        return "inspect_scene"
    return None


def task_type_for_skill(skill_name: str) -> str:
    semantic = default_skill_semantics().get(skill_name)
    category = semantic.category if semantic is not None else ""
    if skill_name in {
        "inspect_scene",
        "look_around",
        "request_perception",
        "detect_marker",
    }:
        return "observation"
    if category in {"base", "navigation"} or skill_name in {
        "move_base",
        "turn_base",
        "human_follow",
        "navigate_to",
        "approach_object",
    }:
        return "motion"
    if category in {"arm", "gripper", "manipulation"} or skill_name in {
        "set_gripper",
        "set_arm_pose",
        "move_arm_joints",
    }:
        return "actuation"
    if skill_name in {"stop_motion", "reset_posture"}:
        return "safety"
    return "general"


@lru_cache(maxsize=1)
def default_skill_semantics() -> dict[str, SkillEvidenceSemantics]:
    semantics: dict[str, SkillEvidenceSemantics] = {
        "request_perception": SkillEvidenceSemantics(
            name="request_perception",
            evidence_outputs=("weak_scene_observation",),
            category="perception",
            safety_level="observe",
        )
    }
    for spec in load_skill_registry().robot_skill_catalog().list():
        semantics[spec.name] = SkillEvidenceSemantics(
            name=spec.name,
            evidence_outputs=tuple(spec.evidence_outputs),
            cannot_satisfy=tuple(spec.cannot_satisfy),
            category=spec.category,
            safety_level=spec.safety_level,
        )
    return semantics


def evidence_type_for_skill(
    skill_name: str, semantics: dict[str, SkillEvidenceSemantics] | None = None
) -> str:
    semantic = (semantics or default_skill_semantics()).get(skill_name)
    if semantic is not None and semantic.evidence_outputs:
        return semantic.evidence_outputs[0]
    if is_perception_skill_name(skill_name):
        return "weak_scene_observation"
    return f"{skill_name}_result"


def evidence_strength_for_skill(
    skill_name: str, semantics: dict[str, SkillEvidenceSemantics] | None = None
) -> EvidenceStrength:
    semantic = (semantics or default_skill_semantics()).get(skill_name)
    if semantic is not None and semantic.safety_level == "observe":
        return "weak"
    if skill_name == "detect_marker":
        return "strong"
    return "status"


def infer_constraints(text: str, skill_name: str) -> dict[str, Any]:
    if skill_name == "turn_base":
        if _contains_any(text, ("left", "左")):
            return {"direction": "left", "angle_deg": 30}
        if _contains_any(text, ("right", "右")):
            return {"direction": "right", "angle_deg": 30}
    if skill_name == "move_base":
        if _contains_any(text, ("forward", "前", "往前", "向前")):
            return {"direction": "forward", "distance_cm": 20}
        if _contains_any(text, ("backward", "后", "后退", "往后", "向后")):
            return {"direction": "backward", "distance_cm": 20}
    if skill_name == "set_gripper":
        if _contains_any(text, ("open", "打开", "张开")):
            return {"action": "open"}
        if _contains_any(text, ("close", "关闭", "闭合", "合上")):
            return {"action": "close"}
    if skill_name == "set_arm_pose" and "home" in text:
        return {"pose_name": "home"}
    return {}


def _tool_skill(tool: str, args: dict[str, Any]) -> str:
    if tool == "request_skill":
        return str(args.get("skill") or "").strip() or tool
    return tool


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _looks_like_semantic_navigation(text: str) -> bool:
    if _contains_any(
        text,
        (
            "navigate to",
            "go to",
            "move to",
            "walk to",
            "head to",
            "drive to",
            "find your way to",
            "导航到",
            "前往",
            "走到",
            "移动到",
            "去到",
        ),
    ):
        return True
    if not _contains_any(text, ("去", "到")):
        return False
    return _contains_any(
        text,
        (
            "桌",
            "门",
            "房间",
            "厨房",
            "客厅",
            "卧室",
            "走廊",
            "充电",
            "椅",
            "沙发",
            "柜",
            "架",
            "旁边",
            "附近",
        ),
    )
