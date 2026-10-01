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
    # Default model "auto" is Copilot's server-side router (the Copilot CLI's own
    # default): each turn is dispatched to a concrete model chosen by the
    # service. list_models() reports "auto" with empty limits and no reasoning
    # efforts, so its window is unknown until routing happens. context_window /
    # max_output_tokens therefore mirror FALLBACKS below, which is also exactly
    # what model_translation derives for "auto" at runtime; get_info() reports
    # the same values on a cold and a warm model cache.
    "defaults": {
        "model": "auto",
        "max_tokens": 4096,
        # Healthy generation has no implicit elapsed-time deadline.
        "timeout": None,
        "context_window": 128000,
        "max_output_tokens": 16384,
    },
}

# Three-Medium Architecture: Fallback values for when SDK returns None
# These are policy values, NOT hardcoded in Python
FALLBACKS: dict[str, int] = {
    "context_window": 128000,
    "max_output_tokens": 16384,
}
