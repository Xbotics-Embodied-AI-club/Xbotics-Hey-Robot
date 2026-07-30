"""Bootstrap the official RLDX server for its released RoboCasa checkpoint."""

from __future__ import annotations

import argparse
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hey Robot RLDX policy server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="GENERAL_EMBODIMENT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image-max-area", type=int, default=65536)
    return parser


def main() -> None:
    args = _parser().parse_args()
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper
    from rldx.policy.server_client import PolicyServer
    from transformers import AutoProcessor

    descriptor: Any = AutoProcessor.__dict__["from_pretrained"]
    original = AutoProcessor.from_pretrained

    def from_pretrained(path: Any, *values: Any, **kwargs: Any) -> Any:
        # RLDX-1-FT-RC365 currently publishes image_max_area=null even though
        # its training/evaluation contract uses the 256x256 default (65536).
        kwargs.setdefault("image_max_area", args.image_max_area)
        return original(path, *values, **kwargs)

    AutoProcessor.from_pretrained = staticmethod(from_pretrained)
    try:
        tag = EmbodimentTag[args.embodiment_tag]
        policy = RLDXPolicy(
            embodiment_tag=tag,
            model_path=args.model_path,
            device=args.device,
            strict=True,
        )
    finally:
        AutoProcessor.from_pretrained = descriptor
    server = PolicyServer(
        policy=RLDXSimPolicyWrapper(policy, strict=True),
        host=args.host,
        port=args.port,
    )
    server.run()


if __name__ == "__main__":
    main()
