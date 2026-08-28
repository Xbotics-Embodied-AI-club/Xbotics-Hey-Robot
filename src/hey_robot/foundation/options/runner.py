"""High-frequency local option loop owned by the foundation-model layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .protocol import OptionRequest, OptionResult, OptionStatus


class EmbodimentRuntime(Protocol):
    def observe(self) -> Any: ...

    def step_block(self, actions: list[Any]) -> tuple[Any, bool]: ...

    def progress(self) -> dict[str, Any]: ...

    def diagnostics(self) -> dict[str, Any]: ...


class ChunkPolicy(Protocol):
    def reset(self, session_id: str) -> None: ...

    def predict(self, observation: Any, instruction: str) -> list[Any]: ...


class LocalPolicyOptionRunner:
    """Run ``predict -> local runtime.step_block`` until done or budget."""

    def __init__(
        self,
        runtime: EmbodimentRuntime,
        policy: ChunkPolicy,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._runtime = runtime
        self._policy = policy
        self._cancelled = cancelled or (lambda: False)
        self._session_id: str | None = None
        self._instruction: str | None = None

    def run(self, request: OptionRequest) -> OptionResult:
        if request.max_actions < 1:
            raise ValueError("max_actions must be positive")
        # Embodiment adapters may aggregate per-option physical diagnostics.
        # RoboCasa uses this to mirror RPent's grasp/lift/base-drift signals.
        begin_option = getattr(self._runtime, "begin_option", None)
        if callable(begin_option):
            begin_option(request)
        changed = (
            request.reset_session
            or request.session_id != self._session_id
            or request.instruction != self._instruction
        )
        if changed:
            self._policy.reset(request.session_id)
            self._session_id = request.session_id
            self._instruction = request.instruction

        actions_executed = 0
        chunks_executed = 0
        while actions_executed < request.max_actions:
            if self._cancelled():
                return self._result(
                    OptionStatus.CANCELLED, actions_executed, chunks_executed, False
                )
            actions = self._policy.predict(
                self._runtime.observe(), request.instruction
            )[: request.max_actions - actions_executed]
            if not actions:
                return self._result(
                    OptionStatus.FAILED,
                    actions_executed,
                    chunks_executed,
                    False,
                    error="policy returned an empty action chunk",
                )
            _, done = self._runtime.step_block(actions)
            actions_executed += len(actions)
            chunks_executed += 1
            if done:
                return self._result(
                    OptionStatus.SUCCESS, actions_executed, chunks_executed, True
                )
        return self._result(
            OptionStatus.BUDGET, actions_executed, chunks_executed, False
        )

    def _result(
        self,
        status: OptionStatus,
        actions_executed: int,
        chunks_executed: int,
        environment_done: bool,
        *,
        error: str | None = None,
    ) -> OptionResult:
        return OptionResult(
            status=status,
            actions_executed=actions_executed,
            chunks_executed=chunks_executed,
            environment_done=environment_done,
            progress=self._runtime.progress(),
            diagnostics=self._runtime.diagnostics(),
            error=error,
        )
