from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable


# Global, process-wide input mode.
# - announce: poll/push only the explicitly allowed input specs.
# - scan: poll/push everything (used by binding + signal workbench).
_MODE_STACK: list[str] = ["announce"]

# Announce-mode allowlist.
# - None means "ALL" (listen).
# - set() means "NONE" (quiet runtime default).
_ANNOUNCE_ALLOW: set[tuple[int, int, int]] | None = set()

# Global denylist applied in both modes.
# (Used to force-disable noisy specs even during listen.)
_DENY: set[tuple[int, int, int]] = set()


# Focus/origin tracking
#
# Motivation: once we have concurrent overlays (menu vs workbench) and background
# systems still running, we need a small, global way to declare which UI
# "origin" currently owns the user's attention/input focus, so other listener
# groups can report themselves as idle (and optionally avoid acting).
#
# This is intentionally orthogonal to scan/announce and allowlist logic.
_FOCUS_STACK: list[str] = []

# Exclusive input capture
#
# When an origin is captured, it is allowed to consume *all* control input and
# other origins should treat inputs as blocked regardless of focus.
#
# This is stronger than focus: focus decides who is foreground; capture decides
# who is allowed to act at all.
_CAPTURE_STACK: list[str] = []

# Optional origin enable/disable flags.
# If an origin is disabled, it is considered OFF regardless of focus.
_ORIGIN_ENABLED: dict[str, bool] = {}


def get_mode() -> str:
    return str(_MODE_STACK[-1]) if _MODE_STACK else "announce"


def get_focus_origin() -> str | None:
    """Return the current focus origin, if any."""

    if not _FOCUS_STACK:
        return None
    o = str(_FOCUS_STACK[-1] or "").strip()
    return o or None


def push_focus(origin: str) -> None:
    """Push a focus origin.

    Later pushes take priority (top-most overlay wins). This is safe to call
    repeatedly; it will not push duplicates if already on top.
    """

    o = str(origin or "").strip()
    if not o:
        return
    if _FOCUS_STACK and str(_FOCUS_STACK[-1]) == o:
        return
    _FOCUS_STACK.append(o)


def pop_focus(origin: str | None = None) -> bool:
    """Pop focus.

    - If origin is None: pop the top.
    - If origin is provided: remove the most-recent matching origin.
    Returns True if something was removed.
    """

    if not _FOCUS_STACK:
        return False
    if origin is None:
        _FOCUS_STACK.pop()
        return True
    o = str(origin or "").strip()
    if not o:
        return False
    for i in range(len(_FOCUS_STACK) - 1, -1, -1):
        if str(_FOCUS_STACK[i]) == o:
            del _FOCUS_STACK[i]
            return True
    return False


def get_capture_origin() -> str | None:
    """Return the current exclusive capture origin, if any."""

    if not _CAPTURE_STACK:
        return None
    o = str(_CAPTURE_STACK[-1] or "").strip()
    return o or None


def push_capture(origin: str) -> None:
    """Push an exclusive capture origin.

    Safe to call repeatedly; it will not push duplicates if already on top.
    """

    o = str(origin or "").strip()
    if not o:
        return
    if _CAPTURE_STACK and str(_CAPTURE_STACK[-1]) == o:
        return
    _CAPTURE_STACK.append(o)


def pop_capture(origin: str | None = None) -> bool:
    """Pop exclusive capture.

    - If origin is None: pop the top capture.
    - If origin is provided: remove the most-recent matching capture.
    Returns True if something was removed.
    """

    if not _CAPTURE_STACK:
        return False
    if origin is None:
        _CAPTURE_STACK.pop()
        return True
    o = str(origin or "").strip()
    if not o:
        return False
    for i in range(len(_CAPTURE_STACK) - 1, -1, -1):
        if str(_CAPTURE_STACK[i]) == o:
            del _CAPTURE_STACK[i]
            return True
    return False


def capture_blocks(origin: str) -> bool:
    """Return True if current capture blocks this origin from acting."""

    o = str(origin or "").strip()
    if not o:
        return False
    cap = get_capture_origin()
    if cap is None:
        return False
    return str(cap) != o


def is_idle(origin: str) -> bool:
    """Return True if origin is not the current focus origin."""

    o = str(origin or "").strip()
    if not o:
        return False
    cur = get_focus_origin()
    if cur is None:
        return False
    return str(cur) != o


def set_origin_enabled(origin: str, enabled: bool) -> None:
    o = str(origin or "").strip()
    if not o:
        return
    _ORIGIN_ENABLED[o] = bool(enabled)


def is_origin_enabled(origin: str) -> bool:
    o = str(origin or "").strip()
    if not o:
        return True
    return bool(_ORIGIN_ENABLED.get(o, True))


def get_origin_state(origin: str) -> str:
    """Return 'off' | 'foreground' | 'background' for an origin.

    - off: explicitly disabled via set_origin_enabled.
    - foreground: enabled and is current focus (or no focus is set).
    - background: enabled but not the current focus.
    """

    o = str(origin or "").strip()
    if not o:
        return "foreground"
    if not is_origin_enabled(o):
        return "off"
    cur = get_focus_origin()
    if cur is None:
        # No focus established => be permissive.
        return "foreground"
    return "foreground" if str(cur) == o else "background"


def set_mode(mode: str) -> None:
    """Set the current global mode (top of stack).

    This is intended for live tools (e.g. Signal Workbench) that want a
    real-time toggle between scan/announce.
    """

    m = str(mode).strip().lower() or "announce"
    if m not in ("announce", "scan"):
        m = "announce"
    if _MODE_STACK:
        _MODE_STACK[-1] = m
    else:
        _MODE_STACK.append(m)


def push_mode(mode: str) -> None:
    """Push a mode onto the global mode stack.

    This is the preferred way for overlays/tools to temporarily force scan/announce
    while they're active, without clobbering whatever was underneath.
    """

    m = str(mode).strip().lower() or "announce"
    if m not in ("announce", "scan"):
        m = "announce"
    _MODE_STACK.append(m)


def pop_mode(expected: str | None = None) -> bool:
    """Pop the top mode.

    If `expected` is provided, only pops when it matches the current top.
    Returns True if a pop occurred.
    """

    if not _MODE_STACK:
        return False
    if expected is not None:
        e = str(expected).strip().lower() or "announce"
        if str(_MODE_STACK[-1]) != e:
            return False
    # Keep at least one element (default announce) to simplify callers.
    if len(_MODE_STACK) <= 1:
        _MODE_STACK[-1] = "announce"
        return True
    _MODE_STACK.pop()
    return True

def set_announce_allowlist(specs: Iterable[tuple[int, int, int]] | None) -> None:
    """Set announce-mode allowlist.

    Each entry is (device, kind, item_id) using signal_kernel_api numeric codes.

    - None => allow ALL (listen)
    - set() => allow NONE (quiet)
    """

    global _ANNOUNCE_ALLOW
    if specs is None:
        _ANNOUNCE_ALLOW = None
        return

    out: set[tuple[int, int, int]] = set()
    for dev, kind, item_id in specs:
        out.add((int(dev), int(kind), int(item_id)))
    _ANNOUNCE_ALLOW = out


def set_announce_interest(specs: Iterable[tuple[int, int, int]] | None) -> None:
    """Backward-compat alias for set_announce_allowlist."""

    set_announce_allowlist(specs)


def set_denylist(specs: Iterable[tuple[int, int, int]] | None) -> None:
    """Set a global denylist (blacklist) of input specs.

    Applied in both scan/listen and announce modes.
    - None clears the denylist.
    """

    global _DENY
    if specs is None:
        _DENY = set()
        return

    out: set[tuple[int, int, int]] = set()
    for dev, kind, item_id in specs:
        out.add((int(dev), int(kind), int(item_id)))
    _DENY = out


def get_effective_set() -> tuple[set[tuple[int, int, int]] | None, set[tuple[int, int, int]]]:
    """Return (allow, deny) for the current mode.

    - allow is None => ALL
    - allow is set() => NONE
    """

    mode = get_mode()
    if mode == "scan":
        return None, set(_DENY)
    return _ANNOUNCE_ALLOW, set(_DENY)


@contextmanager
def temporary_mode(mode: str):
    m = str(mode).strip().lower() or "announce"
    if m not in ("announce", "scan"):
        m = "announce"
    _MODE_STACK.append(m)
    try:
        yield
    finally:
        if _MODE_STACK:
            _MODE_STACK.pop()


@contextmanager
def scan_mode():
    with temporary_mode("scan"):
        yield


@contextmanager
def announce_mode():
    with temporary_mode("announce"):
        yield
