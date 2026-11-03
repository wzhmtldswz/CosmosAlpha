"""Validate industry provider metadata at mining startup.

Offline industry construction and audits are maintained separately.
"""

from __future__ import annotations

import json
from pathlib import Path

INDUSTRY_MANIFEST_NAME = "cosmosalpha_industry_manifest.json"
INCOMPLETE_MARKER = "industry_build_incomplete.json"


def inspect_industry_provider(provider_uri: str) -> dict[str, object] | None:
    """Validate a local industry manifest before Qlib starts loading its values.

    Local providers require complete source metadata and an explicit PIT status.
    Uncertified reconstruction is allowed, but callers must preserve that fact.
    """
    if "://" in provider_uri:
        return None
    root = Path(provider_uri).expanduser()
    if (root / INCOMPLETE_MARKER).exists():
        raise ValueError("INDUSTRY_BUILD_INCOMPLETE: refusing an unfinished provider")
    path = root / INDUSTRY_MANIFEST_NAME
    if not path.is_file():
        raise ValueError("INDUSTRY_MANIFEST_MISSING: source metadata is required")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            "INDUSTRY_MANIFEST_INVALID: cannot read source manifest"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("build_status") != "complete":
        raise ValueError("INDUSTRY_BUILD_INCOMPLETE: source is not marked complete")
    if type(manifest.get("point_in_time_certified")) is not bool:
        raise ValueError("INDUSTRY_MANIFEST_INVALID: PIT status must be explicit")
    if manifest.get("usable_for_industry_calculations") is False:
        raise ValueError(
            "INDUSTRY_NO_KNOWN_VALUES: source is audit-only, not usable for mining"
        )
    return manifest
