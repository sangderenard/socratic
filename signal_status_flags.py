from __future__ import annotations

# IMPORTANT:
# - These flag bits are intended to be mirrored exactly in C.
# - Keep in sync with c_physics/signal_status_flags.h

SIGF_DOWN = 1 << 0
SIGF_HOLD = 1 << 1
SIGF_DOWN_EDGE = 1 << 2
SIGF_UP_EDGE = 1 << 3
SIGF_DOUBLE_EDGE = 1 << 4
SIGF_DOUBLE_HOLD = 1 << 5

# Toggle semantics:
# - SIGF_TOGGLED indicates the current latched state is ON.
# - SIGF_TOGGLE_EDGE indicates the latched state changed this frame.
# - SIGF_DOUBLE_TOGGLE_EDGE indicates a toggle-edge caused by a double-press.
SIGF_TOGGLED = 1 << 6
SIGF_TOGGLE_EDGE = 1 << 7
SIGF_DOUBLE_TOGGLE_EDGE = 1 << 8


def format_flags(flags: int) -> str:
    """Format a compact state string.

    Mirrors the Signal Workbench badges:
    - U/D: up/down
    - H: hold
    - 2: double-press edge
    - 2H: double-hold
    """
    flags = int(flags) & 0xFFFFFFFF
    parts: list[str] = []

    if (flags & SIGF_DOWN) != 0:
        parts.append("D")
    else:
        parts.append("U")

    if (flags & SIGF_DOWN_EDGE) != 0:
        parts.append("+")
    if (flags & SIGF_UP_EDGE) != 0:
        parts.append("-")

    if (flags & SIGF_HOLD) != 0:
        parts.append("H")
    if (flags & SIGF_DOUBLE_EDGE) != 0:
        parts.append("2")
    if (flags & SIGF_DOUBLE_HOLD) != 0:
        parts.append("2H")

    if (flags & SIGF_TOGGLED) != 0:
        parts.append("T")
    if (flags & SIGF_TOGGLE_EDGE) != 0:
        parts.append("t")
    if (flags & SIGF_DOUBLE_TOGGLE_EDGE) != 0:
        parts.append("2t")

    return " ".join(parts)
