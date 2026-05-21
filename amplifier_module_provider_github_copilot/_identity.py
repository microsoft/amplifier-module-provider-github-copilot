"""Provider identity — single import-time constant.

Contract: contracts/filesystem-layout.md:Identity:MUST:1

`PROVIDER_ID` is the registry-facing string. The literal lives exactly
once at `config/_models.py:12`; every other call site MUST import
`PROVIDER_ID` from here so the source of truth stays singular.
"""

from __future__ import annotations

from .config._models import PROVIDER as _PROVIDER

PROVIDER_ID: str = _PROVIDER["id"]
