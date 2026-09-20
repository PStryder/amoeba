"""Typed error taxonomy shared by the RPC control plane and the MCP facade.

Every error carries a stable machine-readable ``code`` so that clients can
distinguish invalid input from resource exhaustion from a stale state version.
"""

from __future__ import annotations

from typing import Any


class MindError(Exception):
    code = "internal_error"
    http_like = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidInput(MindError):
    code = "invalid_input"


class NotFound(MindError):
    code = "not_found"


class StaleVersion(MindError):
    """Optimistic-concurrency failure: expected state version did not match."""

    code = "stale_version"


class BackendUnavailable(MindError):
    code = "backend_unavailable"


class ResourceExhausted(MindError):
    code = "resource_exhausted"


class DeadlineExceeded(MindError):
    code = "deadline_exceeded"


class IntegrityError(MindError):
    """Committed content is missing or a hash chain does not verify."""

    code = "integrity_error"


class Fenced(MindError):
    """A result arrived from a worker that has been superseded."""

    code = "fenced"


class CapabilityUnsupported(MindError):
    """The active backend does not advertise the requested capability."""

    code = "capability_unsupported"


ERROR_CODES = [
    cls.code
    for cls in (
        InvalidInput, NotFound, StaleVersion, BackendUnavailable,
        ResourceExhausted, DeadlineExceeded, IntegrityError, Fenced,
        CapabilityUnsupported, MindError,
    )
]
