from __future__ import annotations

"""Deterministic numeric channel IDs for *temporary* direct input channels.

These channel numbers are *not* compiled controller-graph channels (0..N).
They are reserved IDs intended for an *uncompiled* / *temporary* mapping layer
that can route raw kernel inputs directly into controller channels.

Important policy:
- Controller-engine passthrough outputs (direct signal->channel) are a temporary
    convenience for live tooling. They must NOT be auto-created for ordinary play.
- Once a compiled controller graph is active, these temporary passthrough outputs
    should be cleared so only compiled channels participate in wheels/hooks.

The mapping is reversible so live tooling can identify which kernel input a
reserved channel refers to.
"""

from dataclasses import dataclass

from c_physics import signal_kernel_api


# Reserved blocks
CH_KEY_BASE = 30_000  # pygame key codes
CH_JOYBTN_BASE = 40_000  # joystick button ids (including virtual hat buttons)
CH_JOYAXIS_BASE = 45_000  # joystick axis ids
CH_MOUSEBTN_BASE = 50_000  # mouse buttons
CH_MOUSEMOTION_BASE = 51_000  # mouse motion ids (0=dx,1=dy)
CH_KBDAXIS_BASE = 52_000  # keyboard virtual axes (WASD, arrows, ...)

# Capacity hints (for decode range checks; these are soft limits)
_MAX_KEYCODE = 8192
_MAX_SMALL_ID = 8192


@dataclass(frozen=True)
class PassthruSpec:
    device: int
    kind: int
    item_id: int
    sigsel: int

    def signal_id(self) -> int:
        return int(signal_kernel_api.compose_signal_id(int(self.device), int(self.kind), int(self.item_id)))


def ch_key(keycode: int) -> int:
    return int(CH_KEY_BASE + int(keycode))


def ch_joy_button(button_id: int) -> int:
    return int(CH_JOYBTN_BASE + int(button_id))


def ch_joy_axis(axis_id: int) -> int:
    return int(CH_JOYAXIS_BASE + int(axis_id))


def ch_mouse_button(button_id: int) -> int:
    return int(CH_MOUSEBTN_BASE + int(button_id))


def ch_mouse_motion(motion_id: int) -> int:
    return int(CH_MOUSEMOTION_BASE + int(motion_id))


def ch_kbd_axis(axis_id: int) -> int:
    return int(CH_KBDAXIS_BASE + int(axis_id))


def try_decode_passthru_channel(channel: int) -> PassthruSpec | None:
    """If channel is in a reserved *temporary direct-mapping* block, return its kernel source."""

    ch = int(channel)

    if CH_KEY_BASE <= ch < CH_KEY_BASE + _MAX_KEYCODE:
        keycode = int(ch - CH_KEY_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_KEYBOARD),
            kind=int(signal_kernel_api.GP_EV_KEY),
            item_id=int(keycode),
            sigsel=2,  # GP_SIGSEL_DOWN
        )

    if CH_JOYBTN_BASE <= ch < CH_JOYBTN_BASE + _MAX_SMALL_ID:
        bid = int(ch - CH_JOYBTN_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
            kind=int(signal_kernel_api.GP_EV_BUTTON),
            item_id=int(bid),
            sigsel=2,  # GP_SIGSEL_DOWN
        )

    if CH_JOYAXIS_BASE <= ch < CH_JOYAXIS_BASE + _MAX_SMALL_ID:
        aid = int(ch - CH_JOYAXIS_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
            kind=int(signal_kernel_api.GP_EV_AXIS),
            item_id=int(aid),
            sigsel=0,  # axis
        )

    if CH_MOUSEBTN_BASE <= ch < CH_MOUSEBTN_BASE + _MAX_SMALL_ID:
        bid = int(ch - CH_MOUSEBTN_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_MOUSE),
            kind=int(signal_kernel_api.GP_EV_MOUSE_BUTTON),
            item_id=int(bid),
            sigsel=2,  # down
        )

    if CH_MOUSEMOTION_BASE <= ch < CH_MOUSEMOTION_BASE + 16:
        mid = int(ch - CH_MOUSEMOTION_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_MOUSE),
            kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
            item_id=int(mid),
            sigsel=0,  # axis
        )

    if CH_KBDAXIS_BASE <= ch < CH_KBDAXIS_BASE + 256:
        aid = int(ch - CH_KBDAXIS_BASE)
        return PassthruSpec(
            device=int(signal_kernel_api.GP_DEV_KEYBOARD),
            kind=int(signal_kernel_api.GP_EV_AXIS),
            item_id=int(aid),
            sigsel=0,  # axis
        )

    return None
