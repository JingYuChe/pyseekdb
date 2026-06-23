"""Input validators for namespace names and record identifiers."""

import re

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_NAME_LENGTH = 512
_MAX_NAMESPACE_NAME_LENGTH = 256
_MAX_NAMESPACE_BATCH_SIZE = 100
_MAX_N_RESULTS = 16384  # OceanBase vector-search k upper bound


def _validate_namespace_name(name: str) -> None:
    """Validate a namespace name for type, length, and allowed characters."""
    if not isinstance(name, str):
        raise TypeError(
            f"Invalid namespace name: '{name}'. Namespace name must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError(f"Invalid namespace name: '{name}'. Namespace name must not be empty")
    if len(name) > _MAX_NAMESPACE_NAME_LENGTH:
        raise ValueError(
            f"Invalid namespace name: '{name}'. Namespace name too long: {len(name)} characters; maximum allowed is {_MAX_NAMESPACE_NAME_LENGTH}."
        )
    if _NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Invalid namespace name: '{name}'. Namespace name contains invalid characters. "
            "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
        )


def _validate_record_ids(ids: list[str]) -> None:
    """Validate namespace/collection record id list shape and per-id constraints."""
    if not isinstance(ids, list):
        raise TypeError(
            f"Invalid record ids: expected list[str], got {type(ids).__name__}"
        )
    for rid in ids:
        if not isinstance(rid, str):
            raise TypeError(
                f"Invalid record id: '{rid}'. Record id must be a string, got {type(rid).__name__}"
            )
        if not rid:
            raise ValueError("Invalid record id: Record id must not be empty")
        if len(rid) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Invalid record id: '{rid[:50]}...'. Record id too long: {len(rid)} characters; maximum allowed is {_MAX_NAME_LENGTH}."
            )
        if _NAME_PATTERN.match(rid) is None:
            raise ValueError(
                f"Invalid record id: '{rid}'. Record id contains invalid characters. "
                "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
            )
