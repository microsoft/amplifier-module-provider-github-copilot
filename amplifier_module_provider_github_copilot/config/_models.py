"""Provider identity and model catalog.

Data only — no logic. All validation is in config_loader.py.
Contract: contracts/provider-protocol.md
"""

VERSION = "1.0"

# Provider identity and credential env vars in priority order (first non-empty wins)
# Priority: Copilot agent mode > recommended > CLI compat > Actions compat
PROVIDER: dict = {
    "id": "github-copilot",
    "display_name": "GitHub Copilot SDK",
    "credential_env_vars": [
        "COPILOT_AGENT_TOKEN",  # Copilot agent mode
        "COPILOT_GITHUB_TOKEN",  # Official recommended
        "GH_TOKEN",  # GitHub CLI compatible
        "GITHUB_TOKEN",  # GitHub Actions compatible
    ],
    # Provider-level capabilities: minimum ALL models support (intersection)
    # Per PROVIDER_CONTRACT.md:97, use kernel constants: TOOLS="tools", STREAMING="streaming"
    # NOTE: Per-model capabilities (vision, thinking) are set dynamically in models.py
    # based on SDK's supports_vision/supports_reasoning_effort flags
    "capabilities": ["streaming", "tools"],
    # Updated defaults to use claude-opus-4.5 with SDK-verified limits
    # SDK limits: max_context_window=200000, max_prompt_tokens=168000
    # max_output_tokens = context_window - max_prompt_tokens = 32000
    "defaults": {
        "model": "claude-opus-4.5",
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": 3600,
        "context_window": 200000,
        "max_output_tokens": 32000,
    },
}

# Full model catalog
# claude-opus-4.5 is the default model with SDK-verified limits
# SDK: max_context_window=200000, max_prompt_tokens=168000 → max_output=32000
MODELS: list[dict] = [
    {
        "id": "claude-opus-4.5",
        "display_name": "Claude Opus 4.5",
        "context_window": 200000,
        "max_output_tokens": 32000,
        # Per kernel capabilities.py: TOOLS="tools", STREAMING="streaming", VISION="vision"
        "capabilities": ["streaming", "tools", "vision"],
        "defaults": {},
    },
    {
        "id": "gpt-4",
        "display_name": "GPT-4",
        "context_window": 128000,
        "max_output_tokens": 4096,
        "capabilities": ["streaming", "tools"],
        "defaults": {},
    },
    {
        "id": "gpt-4o",
        "display_name": "GPT-4o",
        "context_window": 128000,
        "max_output_tokens": 4096,
        "capabilities": ["streaming", "tools"],
        "defaults": {},
    },
]

# Three-Medium Architecture: Fallback values for when SDK returns None
# These are policy values, NOT hardcoded in Python
FALLBACKS: dict[str, int] = {
    "context_window": 128000,
    "max_output_tokens": 16384,
}
