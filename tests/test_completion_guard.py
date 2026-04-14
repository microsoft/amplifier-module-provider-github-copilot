"""Tests for StreamingAccumulator completion guard.

Contract: streaming-contract.md — accumulator must respect completion semantics
"""

from amplifier_module_provider_github_copilot.streaming import (
    DomainEvent,
    DomainEventType,
    StreamingAccumulator,
)


class TestCompletionGuard:
    """Events after completion must be ignored."""

    def test_content_delta_after_turn_complete_is_ignored(self):
        """CONTENT_DELTA after TURN_COMPLETE MUST be ignored.

        Contract: streaming-contract:completion:MUST:1
        """
        accumulator = StreamingAccumulator()

        # Add initial content
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello"},
            )
        )

        # Complete the turn
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        assert accumulator.is_complete is True

        # Late CONTENT_DELTA should be ignored
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": " World"},
            )
        )

        # Text content should NOT include the late delta
        assert accumulator.text_content == "Hello"

    def test_content_delta_after_error_is_ignored(self):
        """CONTENT_DELTA after ERROR MUST be ignored.

        Contract: streaming-contract:completion:MUST:2
        """
        accumulator = StreamingAccumulator()

        # Add initial content
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello"},
            )
        )

        # Error marks completion
        accumulator.add(
            DomainEvent(
                type=DomainEventType.ERROR,
                data={"message": "Something failed"},
            )
        )

        assert accumulator.is_complete is True

        # Late CONTENT_DELTA should be ignored
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": " World"},
            )
        )

        # Text content should NOT include the late delta
        assert accumulator.text_content == "Hello"

    def test_tool_call_after_turn_complete_is_ignored(self):
        """TOOL_CALL after TURN_COMPLETE MUST be ignored.

        Contract: streaming-contract:completion:MUST:1
        """
        accumulator = StreamingAccumulator()

        # Add a tool call
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_1", "name": "read_file"},
            )
        )

        # Complete the turn
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "tool_use"},
            )
        )

        assert accumulator.is_complete is True
        assert len(accumulator.tool_calls) == 1

        # Late TOOL_CALL should be ignored
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_2", "name": "spurious_tool"},
            )
        )

        # Tool calls should NOT include the late call
        assert len(accumulator.tool_calls) == 1
        assert accumulator.tool_calls[0]["id"] == "call_1"

    def test_normal_accumulation_still_works(self):
        """Normal event accumulation is unaffected (regression test).

        Contract: streaming-contract:accumulation:MUST:1
        """
        accumulator = StreamingAccumulator()

        # Add content in sequence
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello"},
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": " World"},
            )
        )

        # Add tool call
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_1", "name": "test_tool", "arguments": {}},
            )
        )

        # Add usage update
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 10, "output_tokens": 20},
            )
        )

        # Complete
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        # Verify all content accumulated correctly
        assert accumulator.text_content == "Hello World"
        assert len(accumulator.tool_calls) == 1
        assert accumulator.usage == {"input_tokens": 10, "output_tokens": 20}
        assert accumulator.finish_reason == "stop"
        assert accumulator.is_complete is True

    def test_thinking_content_after_completion_is_ignored(self):
        """Thinking CONTENT_DELTA after TURN_COMPLETE MUST be ignored.

        Contract: streaming-contract:completion:MUST:1
        """
        accumulator = StreamingAccumulator()

        # Add thinking content
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Thinking..."},
                block_type="THINKING",
            )
        )

        # Complete the turn
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        # Late thinking delta should be ignored
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": " More thinking"},
                block_type="THINKING",
            )
        )

        # Thinking content should NOT include the late delta
        assert accumulator.thinking_content == "Thinking..."

    def test_usage_after_turn_complete_is_accepted(self):
        """USAGE_UPDATE after TURN_COMPLETE MUST be accepted.

        SDK sends assistant.usage AFTER assistant.turn_end.
        This is a regression test for the zero-usage bug where
        usage events were dropped by the is_complete guard.

        Contract: streaming-contract:usage:MUST:1
        Bug: Session 65131f78 showed usage={input:0, output:0} when
             SDK sent ASSISTANT_USAGE after TURN_COMPLETE.
        """
        accumulator = StreamingAccumulator()

        # Add content
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Response text"},
            )
        )

        # Complete the turn FIRST (this is the SDK's actual order)
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        assert accumulator.is_complete is True

        # Usage arrives AFTER completion (SDK actual behavior)
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 100, "output_tokens": 50},
            )
        )

        # Usage MUST be captured even after completion
        assert isinstance(accumulator.usage, dict)
        assert accumulator.usage["input_tokens"] == 100
        assert accumulator.usage["output_tokens"] == 50

    def test_usage_after_error_is_accepted(self):
        """USAGE_UPDATE after ERROR MUST be accepted.

        Even on errors, usage data should be captured for billing/tracking.

        Contract: streaming-contract:usage:MUST:1
        """
        accumulator = StreamingAccumulator()

        # Error occurs
        accumulator.add(
            DomainEvent(
                type=DomainEventType.ERROR,
                data={"message": "Rate limit exceeded"},
            )
        )

        assert accumulator.is_complete is True

        # Usage arrives AFTER error
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 50, "output_tokens": 0},
            )
        )

        # Usage MUST be captured even after error
        assert isinstance(accumulator.usage, dict)
        assert accumulator.usage["input_tokens"] == 50
        assert accumulator.usage["output_tokens"] == 0
