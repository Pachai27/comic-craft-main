"""Bounded retry for transient provider failures (Phase 11).

Only the raw SDK transport call is ever retried, and only when the
failure is clearly transient: HTTP 429 or a 5xx status surfaced by the
SDK. Everything else — validation errors, malformed model output,
authentication/configuration errors, or errors without a status —
propagates immediately with no retry.

Policy: at most 2 total attempts (1 initial + 1 immediate retry, no
sleep/backoff so normal web requests never wait long). Status codes are
read defensively (`code`, `status_code`, or `response.status_code`) so
this works for both `google-genai` (`APIError.code`) and
`huggingface_hub` (`HfHubHTTPError.response.status_code`) without hard
SDK imports or message-text matching.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS_CAP = 2


def status_code_of(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an SDK error, if present."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def is_transient(exc: BaseException) -> bool:
    """Return True only for clearly transient provider failures."""
    code = status_code_of(exc)
    if code is None:
        return False
    return code == 429 or 500 <= code <= 599


def call_with_retry(func: Callable[[], T], max_attempts: int = 2) -> T:
    """Call `func`, retrying once on transient failures.

    Total attempts are capped at 2. The final failure (transient or
    not) is re-raised unchanged for the caller to wrap.
    """
    try:
        attempts = int(max_attempts)
    except (TypeError, ValueError):
        attempts = 2
    attempts = max(1, min(attempts, MAX_ATTEMPTS_CAP))
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
            if not is_transient(exc):
                raise
            last_exc = exc
    assert last_exc is not None  # attempts >= 1 guarantees a failure here
    raise last_exc
