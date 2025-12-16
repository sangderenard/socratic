from __future__ import annotations

"""DEPRECATED: pygame -> AppEvent translation.

This file existed as a short-lived bridge while introducing the event registry.

Correct architecture:
- Main thread pushes low-level input into the C signal kernel.
- Controller graph (channels) is evaluated from kernel peek state.
- AppEvents (MENU_*, GAME_*) are emitted from *channels*, not from hardware.

As a result, this dispatcher intentionally emits no AppEvents.
"""

from typing import Any


class AppEventDispatcher:
    def __init__(self, *, joystick_cfg_path: str = "joystick.json") -> None:
        self._cfg_path = str(joystick_cfg_path)

    def dispatch_pygame(self, *, event: Any, t_s: float):
        return []
