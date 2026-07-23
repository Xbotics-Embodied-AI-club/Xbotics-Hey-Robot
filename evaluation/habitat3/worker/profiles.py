"""Allowlisted Habitat 3 runtime profiles.

The runtime deliberately exposes a small, named set of profiles instead of
accepting arbitrary Habitat config paths from an RPC client.  This is both a
safety boundary and an important part of reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HabitatProfile:
    name: str
    base_config: str
    task: str
    executor_mode: str
    skills: frozenset[str]
    dataset_filename: str
    requires_physics: bool = False


SOCIAL_ORACLE_PROFILE = HabitatProfile(
    name="habitat3_social_spot_human_oracle",
    base_config="benchmark/multi_agent/hssd_spot_human_social_nav.yaml",
    task="RearrangePddlSocialNavTask-v0",
    executor_mode="privileged_oracle",
    skills=frozenset(
        {
            "habitat_navigate_to",
            "habitat_follow_human",
            "habitat_wait",
            "habitat_stop",
        }
    ),
    dataset_filename="social_rearrange.json.gz",
)

SYMBOLIC_REARRANGE_PROFILE = HabitatProfile(
    name="habitat3_spot_human_rearrange_symbolic",
    base_config="benchmark/multi_agent/hssd_spot_human.yaml",
    task="RearrangePddlTask-v0",
    executor_mode="privileged_symbolic",
    skills=frozenset(
        {
            "habitat_navigate_to",
            "habitat_symbolic_pick",
            "habitat_symbolic_place",
            "habitat_wait",
            "habitat_stop",
        }
    ),
    dataset_filename="social_rearrange.json.gz",
    requires_physics=True,
)

PROFILES = {
    profile.name: profile
    for profile in (SOCIAL_ORACLE_PROFILE, SYMBOLIC_REARRANGE_PROFILE)
}


def get_profile(name: str) -> HabitatProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"profile is not allowlisted: {name}") from exc
