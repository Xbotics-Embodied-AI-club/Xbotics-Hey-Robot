"""Run reset -> observe -> step -> metrics -> close in the Habitat container."""

import argparse

from evaluation.habitat3.worker.environment import HabitatEnvironment
from evaluation.habitat3.worker.preflight import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/opt/habitat/data")
    parser.add_argument("--profile", default="habitat3_social_spot_human_oracle")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    preflight = report(args.data_root, args.profile)
    if not preflight["ok"]:
        raise RuntimeError(f"Habitat asset preflight failed: {preflight['missing']}")
    runtime = HabitatEnvironment(data_root=args.data_root, profile=args.profile)
    runtime.load(
        task=runtime.profile_definition.task,
        split=args.split,
        seed=100,
        controlled_agent="agent_0",
    )
    observed = runtime.observe()
    stepped = runtime.step(runtime._zero_action(), expected_frame_id=observed.frame_id)
    print(
        {
            "episode_id": stepped.episode_id,
            "frame_id": stepped.frame_id,
            "assets": len(stepped.assets),
            "metrics": stepped.metrics,
        }
    )
    runtime.close(expected_frame_id=stepped.frame_id)


if __name__ == "__main__":
    main()
