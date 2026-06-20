from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: str
    success: bool = True


@dataclass
class AgentState:
    task: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    last_observation_summary: str | None = None
    last_error: str | None = None
    last_capability_safety_level: str | None = None
    last_capability_name: str | None = None

    def reset(self) -> None:
        self.task = ""
        self.messages.clear()
        self.tool_calls.clear()
        self.last_observation_summary = None
        self.last_error = None
        self.last_capability_safety_level = None
        self.last_capability_name = None

    def add_message(self, role: str, content: str | None) -> None:
        text = (content or "").strip()
        if text:
            self.messages.append({"role": role, "content": text})

    def add_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        success: bool = True,
    ) -> None:
        self.tool_calls.append(
            ToolCallRecord(
                name=name,
                arguments=dict(arguments),
                result=result,
                success=success,
            )
        )

    def recent_tool_context(self, limit: int = 8) -> str:
        if not self.tool_calls:
            return ""
        lines = ["Recent tool calls:"]
        for record in self.tool_calls[-limit:]:
            status = "ok" if record.success else "error"
            lines.append(
                f"- {record.name}({record.arguments}) -> {status}: {record.result}"
            )
        return "\n".join(lines)

    def loop_warning_context(self, limit: int = 6) -> str:
        recent = self.tool_calls[-max(2, limit) :]
        if len(recent) < 2:
            return ""
        last = recent[-1]
        repeated_tail = self._contiguous_tail_count(recent)
        repeated_failures = self._contiguous_tail_failures(recent)
        no_progress = self._tail_has_no_progress(recent)
        lines: list[str] = []
        if repeated_tail >= 2:
            lines.append(
                f"- repeated_attempt: {last.name}({last.arguments}) was attempted {repeated_tail} times in a row"
            )
        if repeated_failures >= 2:
            lines.append(
                f"- repeated_failure: the last {repeated_failures} matching attempts all failed"
            )
        if no_progress:
            lines.append(
                "- no_progress_signal: recent tool results look repetitive and did not create a new outcome"
            )
        if not lines:
            return ""
        lines.append(
            "- guidance: do not repeat the same action without new evidence; "
            "inspect status, perception, feedback, or choose a different next step"
        )
        return "Loop warning:\n" + "\n".join(lines)

    @staticmethod
    def _signature(record: ToolCallRecord) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            record.name,
            tuple(
                sorted(
                    (str(key), repr(value)) for key, value in record.arguments.items()
                )
            ),
        )

    def _contiguous_tail_count(self, records: list[ToolCallRecord]) -> int:
        target = self._signature(records[-1])
        count = 0
        for record in reversed(records):
            if self._signature(record) != target:
                break
            count += 1
        return count

    def _contiguous_tail_failures(self, records: list[ToolCallRecord]) -> int:
        target = self._signature(records[-1])
        count = 0
        for record in reversed(records):
            if self._signature(record) != target or record.success:
                break
            count += 1
        return count

    def _tail_has_no_progress(self, records: list[ToolCallRecord]) -> bool:
        if len(records) < 3:
            return False
        tail = records[-3:]
        first = tail[0]
        if any(
            self._signature(record) != self._signature(first) for record in tail[1:]
        ):
            return False
        normalized_results = {record.result.strip().lower() for record in tail}
        if len(normalized_results) == 1:
            return True
        return all(not record.success for record in tail)
