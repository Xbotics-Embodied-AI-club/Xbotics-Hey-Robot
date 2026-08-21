from __future__ import annotations

from hey_robot.foundation.options import (
    LocalPolicyOptionRunner,
    OptionRequest,
    OptionStatus,
)


class _Environment:
    def __init__(self, *, complete_after: int) -> None:
        self.complete_after = complete_after
        self.applied: list[int] = []

    def observe(self):
        return {"frame": len(self.applied)}

    def step_block(self, actions):
        self.applied.extend(actions)
        return self.observe(), len(self.applied) >= self.complete_after

    def progress(self):
        return {"applied": len(self.applied)}

    def diagnostics(self):
        return {"local": True}


class _Policy:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.prompts: list[str] = []

    def reset(self, session_id):
        self.resets.append(session_id)

    def predict(self, _observation, instruction):
        self.prompts.append(instruction)
        return [1, 2, 3, 4]


def test_runner_keeps_a_same_instruction_session_continuous() -> None:
    environment = _Environment(complete_after=6)
    policy = _Policy()
    runner = LocalPolicyOptionRunner(environment, policy)

    result = runner.run(OptionRequest("episode-1", "rinse sink", max_actions=8))

    assert result.status is OptionStatus.SUCCESS
    assert result.actions_executed == 8
    assert result.chunks_executed == 2
    assert policy.resets == ["episode-1"]
    assert policy.prompts == ["rinse sink", "rinse sink"]


def test_runner_resets_only_when_session_or_instruction_changes() -> None:
    environment = _Environment(complete_after=100)
    policy = _Policy()
    runner = LocalPolicyOptionRunner(environment, policy)

    runner.run(OptionRequest("episode-1", "rinse sink", max_actions=1))
    runner.run(OptionRequest("episode-1", "rinse sink", max_actions=1))
    runner.run(OptionRequest("episode-1", "close fridge", max_actions=1))

    assert policy.resets == ["episode-1", "episode-1"]
