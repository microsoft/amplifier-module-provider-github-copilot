"""Provider-local selectors: policy only, never an inventory or inference gate."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase

# Only recognize unambiguous version syntax. Everything else remains visible.
# Keep the whole non-version name: astra/luna/sol/terra, opus/sonnet/haiku,
# pro/flash/flash-lite, mini/codex, preview and picker routes are NOT equivalent.
_VERSION = re.compile(
    r"^(gpt|claude(?:-[a-z]+)?|gemini|grok|mai-[a-z]+)-"
    r"(\d+(?:[.-]\d+)*)(?=-|$)(.*)$"
)
_O_VERSION = re.compile(r"^(o)(\d+)(?=-|$)(.*)$")
_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


def _undated(model_id: str) -> tuple[str, str]:
    match = _DATE.search(model_id)
    return (model_id[: match.start()], match[1].replace("-", "")) if match else (model_id, "")


def _family_version(model_id: str) -> tuple[str, tuple[int, ...], str] | None:
    base, date = _undated(model_id.casefold())
    match = _VERSION.fullmatch(base) or _O_VERSION.fullmatch(base)
    if match is None:
        return None
    prefix, version, suffix = match.groups()
    # Both claude-3.5-sonnet and claude-sonnet-4.5 identify the Sonnet family.
    # Preserve all suffixes, including unknown future tiers, rather than guessing.
    family = prefix + suffix
    numbers = tuple(int(part) for part in re.split(r"[.-]", version))
    return family, numbers, date


@dataclass(frozen=True)
class ModelFilters:
    """Immutable copy of common selectors plus explicit recommendation policy."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False
    latest_per_family: bool = False

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> ModelFilters:
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated", "latest_per_family"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern.strip() for pattern in raw
            ):
                raise ValueError(f"model filters {key} must be a list of nonblank strings")
            return tuple(raw)

        def flag(key: str) -> bool:
            raw = value.get(key, False)
            if not isinstance(raw, bool):
                raise TypeError(f"model filters {key} must be a boolean")
            return raw

        return cls(
            patterns("include", ("*",)),
            patterns("exclude", ()),
            flag("hide_dated"),
            flag("latest_per_family"),
        )

    def select(self, ids: list[str]) -> list[str]:
        selected = {
            model_id
            for model_id in ids
            if any(fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.include)
            and not any(
                fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.exclude
            )
        }
        if self.hide_dated:
            folded = {model_id.casefold() for model_id in selected}
            selected = {
                model_id
                for model_id in selected
                if not ((parts := _undated(model_id.casefold()))[1] and parts[0] in folded)
            }
        if self.latest_per_family:
            # Version first, alias before its SAME-version snapshot, newest date
            # for snapshot-only ties. No cross-family quality/cost ranking.
            best: dict[str, tuple[tuple[int, ...], bool, str]] = {}
            parsed = {model_id: _family_version(model_id) for model_id in selected}
            for parts in parsed.values():
                if parts is not None:
                    family, version, date = parts
                    rank = (version, not date, date)
                    best[family] = max(best.get(family, rank), rank)
            selected = {
                model_id
                for model_id, parts in parsed.items()
                if parts is None or (parts[1], not parts[2], parts[2]) == best[parts[0]]
            }
        return sorted(selected)

    def as_dict(self) -> dict[str, object]:
        return {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "hide_dated": self.hide_dated,
            "latest_per_family": self.latest_per_family,
        }


def selector_schema() -> dict[str, object]:
    """Fresh metadata, not references to live policy."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive OS-independent globs; [] includes none.",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": [],
                "description": "Case-insensitive OS-independent globs; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean",
                "default": False,
                "description": "Hide YYYY-MM-DD/YYYYMMDD snapshots if their alias is selected.",
            },
            "latest_per_family": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Latest numeric version per recognized vendor/family/tier/route; alias before "
                    "matching snapshot, newest snapshot otherwise. Unknown names remain visible."
                ),
            },
        },
    }
