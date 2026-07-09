from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Any, cast

from hey_robot.robot_runtime.simulation.so101_tabletop import (
    So101TabletopSimDriver,
)
from hey_robot.robot_runtime.simulation.so101_tabletop.scenario import (
    TABLETOP_OBJECTS,
)

from hey_robot.config import RobotSpec
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.skill_os.apis import RobotSkillAPI
from hey_robot.skill_os.builtins.tabletop_manipulation import PickSkill
from hey_robot.skill_os.context import SkillContext


class DirectRobotPort:
    def __init__(self, driver: So101TabletopSimDriver) -> None:
        self.driver = driver

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        action = RobotSkillAction(name, arguments or {}).to_robot_action(
            SkillIntent(envelope=Envelope(robot_id="acceptance"), name=name)
        )
        await self.driver.apply_action(action)
        if self.driver.last_skill_result is None:
            raise RuntimeError(f"{name} produced no driver result")
        return self.driver.last_skill_result.to_dict()


def build_driver() -> So101TabletopSimDriver:
    spec = RobotSpec(
        type="so101_sim",
        family="so101",
        environment="sim",
        driver="mujoco",
        embodiment_profile="so101_tabletop_sim",
        settings={"mjcf_path": "assets/robots/so101_tabletop/scene.xml"},
    )
    return So101TabletopSimDriver(
        RobotDriverContext(
            robot_id="acceptance",
            spec=spec,
            deployment_id="so101-tabletop-acceptance",
            embodiment=get_embodiment_profile(spec),
        )
    )


async def run(repeats: int) -> int:
    driver = build_driver()
    await driver.start()
    context = SkillContext(robot=cast(RobotSkillAPI, DirectRobotPort(driver)))
    outcomes: dict[str, Counter[str]] = {name: Counter() for name in TABLETOP_OBJECTS}
    lifts: dict[str, list[float]] = {name: [] for name in TABLETOP_OBJECTS}
    try:
        for object_name in TABLETOP_OBJECTS:
            for _ in range(repeats):
                await driver.reset()
                result = await PickSkill().execute(
                    context,
                    {
                        "object_label": object_name,
                        "mode": "hold",
                        "max_retries": 1,
                    },
                )
                if result.success and driver.gripper.held_object == object_name:
                    outcomes[object_name]["success"] += 1
                    lifts[object_name].append(float(result.data.get("lift_m", 0.0)))
                else:
                    key = result.failure_mode or "wrong_object"
                    outcomes[object_name][key] += 1
    finally:
        await driver.close()

    successes = 0
    total = repeats * len(TABLETOP_OBJECTS)
    for object_name in TABLETOP_OBJECTS:
        count = outcomes[object_name]["success"]
        successes += count
        minimum_lift = min(lifts[object_name], default=0.0)
        print(
            f"{object_name}: {count}/{repeats} success, "
            f"min_lift={minimum_lift:.3f}m, outcomes={dict(outcomes[object_name])}"
        )
    rate = successes / total if total else 0.0
    print(f"total: {successes}/{total} success ({rate:.1%})")
    banana_ok = outcomes["banana"]["success"] == repeats
    return 0 if banana_ok and rate >= 0.90 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(max(1, args.repeats))))


if __name__ == "__main__":
    main()
