"""Deterministic identifiers for versioned CosmosAlpha content."""

from __future__ import annotations

import hashlib

from cosmosalpha.core.serialization import canonical_json_bytes


def stable_sha256(value: object) -> str:
    """Create a stable SHA-256 identifier from canonical JSON content.

    Args:
        value: Supported protocol or JSON value.

    Returns:
        A lowercase 64-character hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
