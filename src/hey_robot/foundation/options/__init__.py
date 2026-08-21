"""Environment-neutral local execution for embodied foundation models."""

from .protocol import OptionRequest, OptionResult, OptionStatus
from .runner import LocalPolicyOptionRunner

__all__ = ["LocalPolicyOptionRunner", "OptionRequest", "OptionResult", "OptionStatus"]
