"""Tests for response extraction recursion guard.

Contract: sdk-response.md — extraction must terminate safely
"""

from amplifier_module_provider_github_copilot.provider import extract_response_content


class TestRecursionGuard:
    """Recursion must be bounded to prevent stack overflow."""

    def test_deeply_nested_data_chain_terminates(self):
        """Object with .data.data.data... chain MUST terminate safely.

        Contract: sdk-response:extraction:MUST:1
        """

        # Create a deeply nested chain of .data attributes
        class ChainedData:
            def __init__(self, depth: int, max_depth: int = 20):
                if depth < max_depth:
                    self.data = ChainedData(depth + 1, max_depth)
                else:
                    self.data = self  # Circular reference at the end

        nested = ChainedData(0)

        # Should NOT stack overflow - must return safely
        result = extract_response_content(nested)

        # Should return empty string (no content found) after hitting depth limit
        assert result == ""

    def test_circular_data_reference_terminates(self):
        """Circular .data reference MUST terminate safely.

        Contract: sdk-response:extraction:MUST:2
        """

        class CircularData:
            data: "CircularData"

        obj = CircularData()
        obj.data = obj  # Self-referencing

        # Should NOT stack overflow
        result = extract_response_content(obj)

        # Should return empty string after hitting depth limit
        assert result == ""

    def test_normal_data_extraction_still_works(self):
        """Normal response extraction is unaffected (regression test).

        Contract: sdk-response:extraction:MUST:3
        """

        class ResponseWrapper:
            def __init__(self):
                self.data = DataObject()

        class DataObject:
            def __init__(self):
                self.content = "Hello, World!"

        wrapper = ResponseWrapper()

        result = extract_response_content(wrapper)

        assert result == "Hello, World!"

    def test_direct_content_extraction(self):
        """Direct .content attribute extraction works.

        Contract: sdk-response:extraction:MUST:4
        """

        class DirectContent:
            content = "Direct content"

        result = extract_response_content(DirectContent())

        assert result == "Direct content"

    def test_dict_content_extraction(self):
        """Dict with 'content' key extraction works.

        Contract: sdk-response:extraction:MUST:5
        """
        response = {"content": "Dict content"}

        result = extract_response_content(response)

        assert result == "Dict content"

    def test_none_returns_empty_string(self):
        """None response returns empty string.

        Contract: sdk-response:extraction:MUST:6
        """
        result = extract_response_content(None)

        assert result == ""

    def test_data_wrapper_with_content(self):
        """Wrapper with .data that has .content extracts correctly.

        Contract: sdk-response:extraction:MUST:7
        """

        class Wrapper:
            def __init__(self):
                self.data = ContentHolder()

        class ContentHolder:
            content = "Wrapped content"

        result = extract_response_content(Wrapper())

        assert result == "Wrapped content"

    def test_depth_limit_with_magicmock_style_object(self):
        """MagicMock-style object with infinite .data MUST terminate.

        Contract: sdk-response:extraction:SHOULD:1

        This simulates MagicMock behavior where accessing any attribute
        returns another MagicMock with the same behavior.
        """

        class MagicMockLike:
            """Simulates MagicMock's auto-attribute creation."""

            def __getattr__(self, name: str) -> "MagicMockLike":
                # Any attribute access returns a new MagicMockLike
                return MagicMockLike()

        mock_obj = MagicMockLike()

        # Should NOT stack overflow even though .data returns MagicMockLike forever
        result = extract_response_content(mock_obj)

        # Should return empty string after hitting depth limit
        assert result == ""
