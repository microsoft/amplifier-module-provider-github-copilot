"""Secret Redaction Utility for Log Hygiene.

Provides mechanism (not YAML policy) for redacting secrets from log output.

Contract: behaviors:Logging:MUST:4 - logs MUST NOT contain sensitive data

This module handles redaction of:
- Authorization/Bearer headers
- Token fields (token, github_token, access_token, refresh_token, id_token)
- API key fields (api_key, apikey, client_secret, secret)
- Credential fields (password, passwd, pwd, credential)
- GitHub token formats (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_)
"""

from __future__ import annotations

import re
from typing import Any, cast

__all__ = [
    "REDACTED",
    "redact_sensitive_text",
    "redact_dict",
    "redact_exception_message",
    "safe_log_message",
]

# Redaction placeholder
REDACTED = "[REDACTED]"

# Patterns for key-value pairs that should have values redacted
# These match: key=value, key:value, "key":"value", 'key':'value'
_SECRET_KEY_PATTERNS = [
    # Token-related keys (NOT bearer - handled by AUTH pattern)
    r"(token|github_token|access_token|refresh_token|id_token)",
    # API key fields
    r"(api_key|apikey|client_secret|secret)",
    # Credential fields
    r"(password|passwd|pwd|credential|credentials)",
    # NOTE: authorization/auth/bearer handled separately by _AUTH_HEADER_PATTERN
    # to avoid double-matching that produces orphan ] brackets
]

# Combined pattern for key-value matching
# Matches: key=value, key: value, "key": "value", 'key': 'value'
_KEY_VALUE_PATTERN = re.compile(
    r"['\"]?(?P<key>"
    + "|".join(_SECRET_KEY_PATTERNS)
    + r")['\"]?[\s]*[=:][\s]*['\"]?(?P<value>[^'\"\s,}&\[\]]+)['\"]?",
    re.IGNORECASE,
)

# Authorization header pattern (captures full header value)
# Handles: Authorization: Bearer xyz, "Authorization": "Bearer xyz"
_AUTH_HEADER_PATTERN = re.compile(
    # FIX: Added \s to separator class to match "Bearer <token>" (space separator)
    # Contract: behaviors:Logging:MUST:9
    r"['\"]?(Authorization|Bearer)['\"]?[\s]*[\s=:][\s]*['\"]?(?:Bearer[\s]+)?([^\s'\",}\]]+)['\"]?",
    re.IGNORECASE,
)

# GitHub token patterns (standalone tokens without key context)
# FIX: Changed {36} to {20,} for variable-length tokens
# Contract: behaviors:Logging:MUST:7
_GITHUB_TOKEN_PATTERNS = [
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),  # Personal access token
    re.compile(r"\bgho_[A-Za-z0-9]{20,}\b"),  # OAuth token
    re.compile(r"\bghu_[A-Za-z0-9]{20,}\b"),  # User-to-server token
    re.compile(r"\bghs_[A-Za-z0-9]{20,}\b"),  # Server-to-server token
    re.compile(r"\bghr_[A-Za-z0-9]{20,}\b"),  # Refresh token
    re.compile(r"\bgithub_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}\b"),  # Fine-grained PAT
]

# API key patterns (sk- prefix for OpenAI, Anthropic, etc.)
# Contract: behaviors:Logging:MUST:8
_API_KEY_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI API key
    re.compile(r"\bsk-ant-[A-Za-z0-9-]{20,}\b"),  # Anthropic API key
]

# JWT-like patterns (three base64 segments separated by dots)
# Covers: access tokens, ID tokens, Copilot agent tokens
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")

# Opaque bearer tokens (long alphanumeric strings that look like tokens)
# Catches tokens that don't match specific patterns but look suspicious
# FIX: Changed {40,} to {32,} for shorter tokens
# Contract: behaviors:Logging:MUST:10
_OPAQUE_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z0-9_-]{32,}\b"  # 32+ chars, likely a token
)

# PEM block pattern (private keys, certificates, etc.)
# S2 Fix: PEM blocks MUST be redacted BEFORE _OPAQUE_TOKEN_PATTERN, which would
# match individual 64-char base64 lines inside the block leaving the block header/footer.
# Contract: behaviors:Logging:MUST:4
_PEM_BLOCK_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----",
    re.DOTALL,
)

# Database connection URI pattern.
# Matches: postgresql://user:pass@host, mysql://user:pass@host, redis://user:pass@host, etc.
# Captures and redacts ONLY the password segment; preserves scheme://user@host for context.
# Contract: behaviors:Logging:MUST:4
_DB_URI_PATTERN = re.compile(
    r"((?:postgresql|mysql|mssql|mongodb|redis|amqp)://[^:@\s]+):([^@\s]+)(@)",
    re.IGNORECASE,
)


def redact_sensitive_text(value: object) -> str:
    """Return a log-safe string with secret values redacted.

    Preserves keys/structure for debugging while redacting secret values.

    Args:
        value: Any object to redact (will be converted to str).

    Returns:
        Log-safe string with [REDACTED] placeholders.

    Contract: behaviors:Logging:MUST:4

    """
    text = str(value)

    # Already redacted - return as-is (idempotent)
    if REDACTED in text and _count_secrets(text) == 0:
        return text

    # Redact Authorization/Bearer headers
    text = _AUTH_HEADER_PATTERN.sub(r"\1: " + REDACTED, text)

    # Redact key-value pairs with secret keys
    def replace_key_value(match: re.Match[str]) -> str:
        key = match.group("key")
        # Preserve key, redact value
        return f"{key}={REDACTED}"

    text = _KEY_VALUE_PATTERN.sub(replace_key_value, text)

    # Redact standalone GitHub tokens
    for pattern in _GITHUB_TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)

    # Redact API keys (sk- prefix for OpenAI, Anthropic, etc.)
    # Contract: behaviors:Logging:MUST:8
    for pattern in _API_KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)

    # P2-9: Redact JWT-like tokens (base64.base64.base64)
    text = _JWT_PATTERN.sub(REDACTED, text)

    # S2: Redact PEM blocks BEFORE _OPAQUE_TOKEN_PATTERN, which would otherwise
    # match individual 64-char base64 lines, leaving block headers/footers intact.
    text = _PEM_BLOCK_PATTERN.sub(REDACTED, text)

    # S2: Redact database connection URI passwords (preserves scheme://user@host).
    text = _DB_URI_PATTERN.sub(r"\1:" + REDACTED + r"\3", text)

    # P2-9: Redact opaque tokens (32+ char alphanumeric strings)
    # Only if they look like tokens (not already redacted, not common words)
    text = _OPAQUE_TOKEN_PATTERN.sub(REDACTED, text)

    return text


def _count_secrets(text: str) -> int:
    """Count potential secrets in text (for idempotency check)."""
    count = 0
    count += len(_AUTH_HEADER_PATTERN.findall(text))
    count += len(_KEY_VALUE_PATTERN.findall(text))
    for pattern in _GITHUB_TOKEN_PATTERNS:
        count += len(pattern.findall(text))
    # P2-9: Include JWT and opaque tokens in count
    count += len(_JWT_PATTERN.findall(text))
    count += len(_OPAQUE_TOKEN_PATTERN.findall(text))
    # S2: Include PEM blocks and DB URI passwords in count
    count += len(_PEM_BLOCK_PATTERN.findall(text))
    count += len(_DB_URI_PATTERN.findall(text))
    return count


def redact_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values in a dict while preserving structure.

    L2 Fix: Preserves dict structure for queryability instead of str().
    Recursively walks dict/list structures and redacts string values.

    Args:
        value: Dictionary to redact.

    Returns:
        New dict with redacted string values, structure preserved.

    Contract: behaviors:Logging:MUST:4

    """
    result: dict[str, Any] = {}
    for key, val in value.items():
        if isinstance(val, str):
            result[key] = redact_sensitive_text(val)
        elif isinstance(val, dict):
            # Cast to satisfy type checker — isinstance confirms dict
            result[key] = redact_dict(cast(dict[str, Any], val))
        elif isinstance(val, list):
            redacted_list: list[Any] = []
            for item in cast(list[Any], val):
                if isinstance(item, dict):
                    redacted_list.append(redact_dict(cast(dict[str, Any], item)))
                elif isinstance(item, str):
                    redacted_list.append(redact_sensitive_text(item))
                else:
                    redacted_list.append(item)
            result[key] = redacted_list
        else:
            # Primitives (int, float, bool, None) are safe
            result[key] = val
    return result


def redact_exception_message(exc: BaseException) -> str:
    """Redact sensitive text from exception message.

    Wrapper for redact_sensitive_text(str(exc)).

    Args:
        exc: Exception to redact.

    Returns:
        Log-safe exception message.

    Contract: behaviors:Logging:MUST:4

    """
    return redact_sensitive_text(str(exc))


def safe_log_message(message: str, *args: Any) -> tuple[str, ...]:
    """Prepare a log message with redacted arguments.

    Use with logging: logger.info(*safe_log_message("Error: %s", exc))

    The function redacts each argument and returns a tuple that can be
    directly unpacked into logger methods.

    Args:
        message: Log message format string.
        *args: Arguments to redact before formatting.

    Returns:
        Tuple of (message, *redacted_args) for unpacking with * operator.

    Contract: behaviors:Logging:MUST:4

    """
    # S1 Fix: Redact the message itself to catch accidental f-string misuse.
    # WRONG: safe_log_message(f"Token: {token}")  ← token embedded before redaction
    # RIGHT: safe_log_message("Token: %s", token) ← token passed as arg, redacted
    redacted_message = redact_sensitive_text(message)
    redacted_args = tuple(redact_sensitive_text(arg) for arg in args)
    return (redacted_message, *redacted_args)
