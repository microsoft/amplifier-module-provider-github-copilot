"""SDK Response Fixtures for Testing.

Contract: contracts/sdk-response.md

These fixtures provide realistic SDK response shapes matching
`github-copilot-sdk` version 0.2.0+.

IMPORTANT: These fixtures MUST match the actual SDK response structure.
When the SDK changes, update these fixtures and run the canary tests.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockData:
    """Realistic SDK Data dataclass matching copilot.generated.session_events.Data.

    This is the shape returned by sdk_session.send_and_wait().

    SDK Version: 0.2.0+
    """

    content: str
    role: str = "assistant"
    model: str = "claude-opus-4.5"


@dataclass
class MockSDKResponse:
    """Realistic SDK response wrapper.

    SDK returns response.data where data is a Data dataclass.
    """

    data: MockData | dict[str, Any] | None = None


@dataclass
class MockToolCall:
    """Realistic SDK tool call object.

    SDK Version: 0.1.32+
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class MockUsage:
    """Realistic SDK usage object.

    SDK Version: 0.1.32+
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# Pre-built fixtures for common test scenarios and SDK_RESPONSE_FIXTURES removed —
# zero consumers found (orphaned from deleted test files).
