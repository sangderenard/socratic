from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AppEventName(str, Enum):
    """High-level events for game/menu logic.

    These are intentionally *not* hardware events (no JOYBUTTONDOWN, etc).
    They represent intent-level signals that can be bound to hardware inputs
    via the controller graph / joystick config.
    """

    MENU_TOGGLE = "MENU_TOGGLE"
    MENU_CANCEL = "MENU_CANCEL"
    MENU_CONFIRM = "MENU_CONFIRM"

    MENU_POINTER_AFFIRM = "MENU_POINTER_AFFIRM"  # e.g. click/tap
    MENU_POINTER_MOVE = "MENU_POINTER_MOVE"


@dataclass(frozen=True, slots=True)
class AppEventMeta:
    """Optional metadata describing where an AppEvent came from."""

    # Free-form binding identity (channel name, enum name, etc.)
    binding: str | None = None

    # If the event was derived from a controller-channel.
    channel_1d: int | None = None
    channel_2d: int | None = None

    # Raw device hint (e.g. "keyboard", "mouse", "controller")
    source: str | None = None

    # Optional backend-provided flags/state, if supplied by the dispatcher.
    flags: int | None = None


@dataclass(frozen=True, slots=True)
class AppEvent:
    name: AppEventName

    # Monotonic-ish timestamp in seconds (caller-defined; commonly pygame ticks or perf_counter).
    t_s: float

    # Payload for event-specific data; keep it JSON-like.
    payload: Mapping[str, Any]

    meta: AppEventMeta = AppEventMeta()
