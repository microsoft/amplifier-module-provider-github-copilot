"""Data package for YAML config files.

Contains SDK-correlated tabular data that changes when the SDK event schema changes:
  - errors.yaml  — error pattern mappings (kept as YAML per council verdict)
  - events.yaml  — event classification (kept as YAML per council verdict)

Access via importlib.resources:
    resources.files("amplifier_module_provider_github_copilot.config.data")
"""
