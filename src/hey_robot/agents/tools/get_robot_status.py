from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import BooleanSchema, tool_parameters_schema
from hey_robot.protocol import RobotObservation, RobotStatus


@tool_parameters(
    tool_parameters_schema(
        include_observation=BooleanSchema(
            description="Include latest observation metadata", default=True
        ),
    )
)
class GetRobotStatusTool(Tool):
    name = "get_robot_status"
    description = (
        "Read current robot state and optionally the latest observation metadata."
    )
    read_only = True
    safety_level = "observe"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(self, include_observation: bool = True) -> str:
        tc = self._ctx.turn_context
        if tc is None:
            return "no robot snapshot"
        snapshot = tc.snapshot
        status_text = _format_status(snapshot.status)
        if not include_observation:
            return status_text
        observation_text = _format_observation(snapshot.observation)
        if not observation_text:
            return status_text
        return f"{status_text}\n{observation_text}"


def _format_status(status: RobotStatus | None) -> str:
    if status is None:
        return "当前没有机器人状态。"
    lines = [f"机器人当前{_state_text(status.state)}。"]
    battery = status.metrics.get("battery")
    if isinstance(battery, dict):
        battery_parts: list[str] = []
        percentage = battery.get("percentage")
        voltage = battery.get("voltage")
        battery_status = battery.get("status")
        if percentage is not None:
            battery_parts.append(f"电池约 {percentage}%")
        if voltage is not None:
            battery_parts.append(f"电压 {voltage}V")
        if battery_status:
            battery_parts.append(f"状态 {battery_status}")
        if battery_parts:
            lines.append("，".join(battery_parts) + "。")
    readiness = status.metrics.get("readiness")
    if isinstance(readiness, dict):
        ready_parts: list[str] = []
        for key, label in (
            ("base", "底盘"),
            ("arm", "机械臂"),
            ("gripper", "夹爪"),
            ("camera", "相机"),
        ):
            item = readiness.get(key)
            if isinstance(item, dict) and item.get("ok") is not None:
                ready_parts.append(f"{label}{'正常' if item.get('ok') else '异常'}")
        if ready_parts:
            lines.append("，".join(ready_parts) + "。")
    if status.error:
        lines.append(f"当前错误：{status.error}。")
    return "\n".join(lines)


def _format_observation(observation: RobotObservation | None) -> str:
    if observation is None:
        return ""
    image_count = len(observation.images)
    if image_count <= 0:
        return "当前没有可用视觉画面。"
    frame_part = f"最近一帧画面 frame={observation.frame_id}"
    task_part = f"，task={observation.task}" if observation.task else ""
    return f"{frame_part}，包含 {image_count} 张图像{task_part}。"


def _state_text(state: str | None) -> str:
    mapping = {
        "idle": "空闲",
        "acting": "执行中",
        "observed": "已获取观察",
        "failed": "异常",
        "degraded": "降级运行",
        "skill_completed": "刚完成动作",
        "terminated": "已结束",
        "unknown": "状态未知",
    }
    normalized = str(state or "unknown").strip().lower()
    return mapping.get(normalized, normalized)
