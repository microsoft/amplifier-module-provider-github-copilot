"""Four-Way Gate: Event vocabulary consistency test.

Ensures contracts, config, code, and tests agree on domain events.
This prevents the class of bug where sources drift apart silently.
"""

import re
from pathlib import Path

from amplifier_module_provider_github_copilot.streaming import DomainEventType, load_event_config


class TestFourWayGate:
    """Verify event vocabulary consistency across all sources."""

    def _get_contract_events(self) -> set[str]:
        """Parse domain events from contracts/event-vocabulary.md."""
        contract_path = Path(__file__).parent.parent / "contracts" / "event-vocabulary.md"
        content = contract_path.read_text(encoding="utf-8")

        # Only scan the "Domain Events" table section — stop before finish reason mapping
        # to avoid capturing finish reason values (STOP, TOOL_USE, etc.)
        # Flexible regex: "Five" or "Six" to accommodate THINKING_DELTA lifecycle
        domain_section_match = re.search(
            r"## The (?:Five|Six) Domain Events.*?(?=^##|\Z)", content, re.DOTALL | re.MULTILINE
        )
        if not domain_section_match:
            return set()
        domain_section = domain_section_match.group(0)

        # Pattern: | `EVENT_NAME` | description |
        events: set[str] = set()
        for match in re.finditer(r"\| `(\w+)` \|", domain_section):
            event: str = match.group(1)
            # Filter out SDK events (have dots) and table headers
            if "." not in event and event.isupper():
                events.add(event)
        return events

    def _get_config_events(self) -> set[str]:
        """Extract domain_type values from event config."""
        config = load_event_config()
        return {dt.name for dt, _ in config.bridge_mappings.values()}

    def _get_code_events(self) -> set[str]:
        """Get DomainEventType enum values."""
        return {e.name for e in DomainEventType}

    def test_code_events_match_config(self) -> None:
        """Config domain_type values must be valid DomainEventType members."""
        code_events = self._get_code_events()
        config_events = self._get_config_events()
        invalid = config_events - code_events
        assert not invalid, (
            f"Config references unknown domain types: {invalid}. Valid types: {code_events}"
        )

    def test_code_events_documented_in_contract(self) -> None:
        """All DomainEventType members must appear in contract."""
        code_events = self._get_code_events()
        contract_events = self._get_contract_events()
        undocumented = code_events - contract_events
        assert not undocumented, (
            f"Code has undocumented domain events: {undocumented}. "
            "Update contracts/event-vocabulary.md"
        )

    def test_contract_events_exist_in_code(self) -> None:
        """Contract must not document non-existent events."""
        code_events = self._get_code_events()
        contract_events = self._get_contract_events()
        phantom = contract_events - code_events
        assert not phantom, (
            f"Contract documents non-existent events: {phantom}. Valid code events: {code_events}"
        )
