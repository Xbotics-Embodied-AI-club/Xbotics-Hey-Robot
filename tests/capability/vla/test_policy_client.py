"""Unit tests for LerobotPolicyClient._extract_task."""

from __future__ import annotations


class TestExtractTask:
    """Cover the nested task-extraction guards added for smolvla / GR00T datasets."""

    @staticmethod
    def _extract(obs: dict[str, object]) -> str:
        from hey_robot.capability.vla.policy_client import LerobotPolicyClient

        return LerobotPolicyClient._extract_task(obs)

    # -- flat ----------------------------------------------------------------
    def test_language_direct_string(self) -> None:
        assert self._extract({"language": "pick up"}) == "pick up"

    def test_task_direct_string(self) -> None:
        assert self._extract({"task": "push button"}) == "push button"

    def test_prompt_direct_string(self) -> None:
        assert self._extract({"prompt": "navigate home"}) == "navigate home"

    # -- nested dict ---------------------------------------------------------
    def test_language_dict_with_task_key(self) -> None:
        assert self._extract({"language": {"task": "grasp"}}) == "grasp"

    def test_language_dict_with_other_key(self) -> None:
        assert (
            self._extract({"language": {"instruction": "open drawer"}}) == "open drawer"
        )

    # -- list wrapping (single) ----------------------------------------------
    def test_language_string_in_list(self) -> None:
        assert self._extract({"language": ["pick up"]}) == "pick up"

    def test_language_dict_task_in_list(self) -> None:
        assert self._extract({"language": {"task": ["pick up"]}}) == "pick up"

    # -- nested lists (batch / time dims) ------------------------------------
    def test_nested_list_unwrapping(self) -> None:
        assert self._extract({"language": {"task": [["pick up"]]}}) == "pick up"

    def test_deeply_nested_list(self) -> None:
        assert self._extract({"language": {"task": [[[["grasp"]]]]}}) == "grasp"

    # -- annotation.human key ------------------------------------------------
    def test_annotation_human_key(self) -> None:
        assert (
            self._extract({"annotation.human.task_description": "wipe table"})
            == "wipe table"
        )

    # -- fallback & empty ----------------------------------------------------
    def test_empty_observation_returns_empty_string(self) -> None:
        assert self._extract({}) == ""

    def test_key_priority(self) -> None:
        """language > task > prompt in key enumeration order."""
        obs = {
            "prompt": "last resort",
            "task": "middle",
            "language": "first",
        }
        assert self._extract(obs) == "first"

    # -- non-string values should not crash ----------------------------------
    def test_int_value_skipped(self) -> None:
        assert self._extract({"language": 123}) == ""

    def test_float_value_skipped(self) -> None:
        assert self._extract({"task": 3.14}) == ""

    def test_none_value_skipped(self) -> None:
        assert self._extract({"language": None}) == ""

    def test_empty_list_skipped(self) -> None:
        assert self._extract({"language": []}) == ""
