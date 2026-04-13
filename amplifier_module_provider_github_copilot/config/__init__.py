"""Config package for amplifier_module_provider_github_copilot.

Two-Medium Architecture:
  config/data/   — YAML files (SDK-correlated tabular data)
                   errors.yaml, events.yaml
  config/*.py    — Python policy modules (private implementation, _ prefix)
                   _models.py, _policy.py, _sdk_protection.py

The YAML files are accessed via importlib.resources using the
"amplifier_module_provider_github_copilot.config.data" package path.
External callers import policy through config_loader.py, not these modules directly.
"""

# Expose private submodules as package attributes so Pylance can resolve:
#   from .config import _models
#   from .config._sdk_protection import ...
# without needing __init__.py to be empty.
from . import _models, _policy, _sdk_protection  # noqa: F401
