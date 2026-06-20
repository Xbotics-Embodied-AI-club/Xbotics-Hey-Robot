from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    keywords: tuple[str, ...] = ()
    backend: str = "openai_compat"
    default_api_base: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    strip_model_prefix: bool = False
    supports_responses_api: bool = False
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt", "o1", "o3", "o4"),
        default_api_base=None,
        supports_responses_api=True,
        model_overrides=(
            ("gpt-5", {"max_completion_tokens": True}),
            ("o1", {"max_completion_tokens": True}),
            ("o3", {"max_completion_tokens": True}),
            ("o4", {"max_completion_tokens": True}),
        ),
    ),
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        default_api_base="https://openrouter.ai/api/v1",
    ),
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        default_api_base="https://api.deepseek.com",
    ),
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderSpec(
        name="ark",
        keywords=("ark", "doubao", "volces", "volcano"),
        default_api_base="https://ark.cn-beijing.volces.com/api/v3",
    ),
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        default_api_base="https://api.minimax.io/v1",
    ),
    ProviderSpec(
        name="custom",
        keywords=("custom", "openai_compat", "openai-compatible"),
    ),
)


def find_provider(
    name: str | None = None, *, model: str | None = None, api_base: str | None = None
) -> ProviderSpec:
    lowered_name = (name or "").lower().strip()
    if lowered_name:
        for provider in PROVIDERS:
            if lowered_name == provider.name or lowered_name in provider.keywords:
                return provider
    lowered_base = (api_base or "").lower()
    if lowered_base:
        for provider in PROVIDERS:
            if (
                provider.default_api_base
                and provider.default_api_base.lower()
                .split("//", 1)[-1]
                .split("/", 1)[0]
                in lowered_base
            ):
                return provider
            if any(keyword in lowered_base for keyword in provider.keywords):
                return provider
    lowered_model = (model or "").lower()
    if lowered_model:
        for provider in PROVIDERS:
            if any(keyword in lowered_model for keyword in provider.keywords):
                return provider
    return PROVIDERS[0]
