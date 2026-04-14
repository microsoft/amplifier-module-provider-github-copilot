"""
Tests for deny hook (in sdk_adapter/client.py).

Contract: contracts/deny-destroy.md

Acceptance Criteria:
- Deny hook is created with correct response
- Deny hook returns denial for all tool calls
- DENY_ALL constant has required keys

Note: create_ephemeral_session() and destroy_session() removed in Change 3.
Session lifecycle now handled by CopilotClientWrapper.session() context manager.
"""


class TestDenyAllConstant:
    """Test DENY_ALL constant (now in sdk_adapter/client.py)."""

    def test_deny_all_has_required_keys(self) -> None:
        """DENY_ALL constant has permissionDecision, reason, and suppressOutput.

        Contract: deny-destroy:DenyHook:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import DENY_ALL

        assert DENY_ALL["permissionDecision"] == "deny"
        # Minimal reason strategy - don't teach model tools are blocked
        assert DENY_ALL["permissionDecisionReason"] == "Processing"
        assert DENY_ALL["suppressOutput"] is True
