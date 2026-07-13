"""Helpers for validating document metadata used by ranking components."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DEFAULT_TIMESTAMP_KEY = "timestamp"
DEFAULT_IMPORTANCE_KEY = "importance_score"


def parse_utc_timestamp(value: Any) -> datetime | None:
    """Parse an aware ISO-8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value.strip():
        return None

    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_importance(value: Any) -> float | None:
    """Return an importance value in ``[0, 1]``, or ``None`` if invalid."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return None


def normalize_scoring_metadata(
    metadata: dict[str, Any] | None,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    importance_key: str = DEFAULT_IMPORTANCE_KEY,
) -> tuple[dict[str, Any], set[str]]:
    """Copy metadata, canonicalize scoring fields, and report field issues.

    Missing and invalid fields are deliberately left absent. The online scorer
    applies its neutral fallback, while callers can aggregate the returned issue
    names into a single warning.
    """
    normalized = dict(metadata or {})
    issues: set[str] = set()

    timestamp = parse_utc_timestamp(normalized.get(timestamp_key))
    if timestamp is None:
        issues.add(timestamp_key)
        normalized.pop(timestamp_key, None)
    else:
        normalized[timestamp_key] = timestamp.isoformat().replace("+00:00", "Z")

    importance = parse_importance(normalized.get(importance_key))
    if importance is None:
        issues.add(importance_key)
        normalized.pop(importance_key, None)
    else:
        normalized[importance_key] = importance

    return normalized, issues
