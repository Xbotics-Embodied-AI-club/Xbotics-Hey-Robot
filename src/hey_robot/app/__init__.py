from typing import Any

__all__ = ["DeploymentRunner"]


def __getattr__(name: str) -> Any:
    """Keep lightweight app entrypoints from importing the full deployment."""
    if name == "DeploymentRunner":
        from hey_robot.app.runner import DeploymentRunner

        return DeploymentRunner
    raise AttributeError(name)
