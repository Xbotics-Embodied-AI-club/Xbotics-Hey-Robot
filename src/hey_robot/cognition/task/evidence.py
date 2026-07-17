"""带类型的证据投影；原始观测绝不能证明物体关系。"""

from __future__ import annotations

from hey_robot.protocol import EvidenceFact, RobotStatus


def project_robot_status(
    *, goal_id: str, status: RobotStatus
) -> tuple[EvidenceFact, ...]:
    if status.frame_id is None or not status.location_id:
        return ()
    return (
        EvidenceFact(
            evidence_id=f"status:{status.envelope.robot_id}:{status.frame_id}:location",
            goal_id=goal_id,
            source_kind="robot_status",
            source_id=f"status:{status.envelope.robot_id}:{status.frame_id}",
            observed_at=status.envelope.timestamp,
            frame_id=status.frame_id,
            subject_id=f"robot:{status.envelope.robot_id}",
            predicate="equals",
            object_id=status.location_id,
        ),
    )
