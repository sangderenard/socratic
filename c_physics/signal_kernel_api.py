from __future__ import annotations

import ctypes
from ctypes import c_int, c_uint32, c_uint64

from ctypes import c_float

from ctypes import POINTER

from . import geodesic_ctypes
from .signal_kernel_ctypes import GP_InputEvent, GP_SignalFrame

# Keep in sync with c_physics/signal_kernel_abi.h
GP_DEV_KEYBOARD = 1
GP_DEV_MOUSE = 2
GP_DEV_JOYSTICK = 3

GP_EV_AXIS = 1
GP_EV_BUTTON = 2
GP_EV_HAT = 3
GP_EV_KEY = 4
GP_EV_MOUSE_MOTION = 5
GP_EV_MOUSE_BUTTON = 6
GP_EV_MOUSE_WHEEL = 7


def compose_signal_id(device: int, kind: int, item_id: int) -> int:
    """Compose a stable 32-bit signal_id matching the C kernel."""
    return ((int(device) & 0xFF) << 24) | ((int(kind) & 0xFF) << 16) | (int(item_id) & 0xFFFF)


def bind_signal_kernel(lib: ctypes.CDLL) -> None:
    """Bind kernel symbols if present; raises AttributeError if missing."""
    lib.gp_sigk_reset.argtypes = []
    lib.gp_sigk_reset.restype = None

    lib.gp_sigk_push_events.argtypes = [ctypes.POINTER(GP_InputEvent), c_uint32]
    lib.gp_sigk_push_events.restype = None

    lib.gp_sigk_peek.argtypes = [c_uint64, c_uint32, ctypes.POINTER(GP_SignalFrame)]
    lib.gp_sigk_peek.restype = c_int

    # Optional newer selector-based peek.
    if hasattr(lib, "gp_sigk_peek_sel"):
        lib.gp_sigk_peek_sel.argtypes = [c_uint64, c_uint32, c_uint32, ctypes.POINTER(GP_SignalFrame)]
        lib.gp_sigk_peek_sel.restype = c_int

    # Optional: explicit pulse clearing (recommended when making multiple peeks per frame).
    if hasattr(lib, "gp_sigk_clear_pulses"):
        lib.gp_sigk_clear_pulses.argtypes = []
        lib.gp_sigk_clear_pulses.restype = None

    if hasattr(lib, "gp_sigk_sigtobutton"):
        lib.gp_sigk_sigtobutton.argtypes = [c_uint64, c_uint32, c_float, c_float]
        lib.gp_sigk_sigtobutton.restype = None

    # Optional: signal op helpers
    if hasattr(lib, "gp_sigop_2dseek"):
        lib.gp_sigop_2dseek.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float)]
        lib.gp_sigop_2dseek.restype = None
    if hasattr(lib, "gp_sigop_2dflightstick"):
        lib.gp_sigop_2dflightstick.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float)]
        lib.gp_sigop_2dflightstick.restype = None
    if hasattr(lib, "gp_sigop_2dstereocontrolsurface"):
        lib.gp_sigop_2dstereocontrolsurface.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        lib.gp_sigop_2dstereocontrolsurface.restype = None


def try_load_signal_kernel(*, search_dir: str | None = None):
    """Return (lib, api) or (None, None) if unavailable."""
    try:
        lib = geodesic_ctypes.load_lib(search_dir=search_dir)
        bind_signal_kernel(lib)
        return lib, lib
    except Exception:
        return None, None
