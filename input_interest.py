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


def get_mode() -> str:
    return str(_MODE_STACK[-1]) if _MODE_STACK else "announce"


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
