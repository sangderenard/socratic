from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_float, c_int, c_uint32, c_uint64

from . import geodesic_ctypes
from .controller_engine_ctypes import (
    GP_CtlHookEvent,
    GP_CtlHookQMeta,
    GP_CtlHookWatch,
    GP_CtlMeta,
    GP_CtlOutputDesc,
    GP_CtlPassthruDesc,
    GP_WheelMeta,
    GP_WheelSample,
)


def bind_controller_engine(lib: ctypes.CDLL) -> None:
    """Bind controller-engine symbols if present; raises AttributeError if missing."""

    lib.gp_ctl_reset.argtypes = []
    lib.gp_ctl_reset.restype = None

    lib.gp_ctl_load_graph_file.argtypes = [c_char_p]
    lib.gp_ctl_load_graph_file.restype = c_int

    lib.gp_ctl_start.argtypes = [c_uint32, c_uint32]
    lib.gp_ctl_start.restype = c_int

    lib.gp_ctl_stop.argtypes = []
    lib.gp_ctl_stop.restype = None

    lib.gp_ctl_is_running.argtypes = []
    lib.gp_ctl_is_running.restype = c_int

    lib.gp_ctl_step_once.argtypes = [c_uint64]
    lib.gp_ctl_step_once.restype = c_int

    lib.gp_ctl_get_meta.argtypes = [POINTER(GP_CtlMeta)]
    lib.gp_ctl_get_meta.restype = c_int

    lib.gp_ctl_get_outputs.argtypes = [POINTER(GP_CtlOutputDesc), c_uint32]
    lib.gp_ctl_get_outputs.restype = c_uint32

    lib.gp_ctl_get_write_seq.argtypes = []
    lib.gp_ctl_get_write_seq.restype = c_uint64

    lib.gp_ctl_peek_seq.argtypes = [c_uint64, POINTER(c_float), c_uint32, POINTER(c_uint64)]
    lib.gp_ctl_peek_seq.restype = c_int

    lib.gp_ctl_peek_latest.argtypes = [POINTER(c_float), c_uint32, POINTER(c_uint64), POINTER(c_uint64)]
    lib.gp_ctl_peek_latest.restype = c_int

    # Hook watches
    lib.gp_ctl_hooks_clear.argtypes = []
    lib.gp_ctl_hooks_clear.restype = None

    lib.gp_ctl_hooks_add.argtypes = [POINTER(GP_CtlHookWatch)]
    lib.gp_ctl_hooks_add.restype = c_int

    # Hook queue
    lib.gp_ctl_hookq_get_meta.argtypes = [POINTER(GP_CtlHookQMeta)]
    lib.gp_ctl_hookq_get_meta.restype = c_int

    lib.gp_ctl_hookq_get_write_seq.argtypes = []
    lib.gp_ctl_hookq_get_write_seq.restype = c_uint64

    lib.gp_ctl_hookq_peek_seq.argtypes = [c_uint64, POINTER(GP_CtlHookEvent)]
    lib.gp_ctl_hookq_peek_seq.restype = c_int

    lib.gp_ctl_hookq_peek_latest.argtypes = [POINTER(GP_CtlHookEvent), POINTER(c_uint64)]
    lib.gp_ctl_hookq_peek_latest.restype = c_int

    lib.gp_ctl_hookq_wait.argtypes = [c_uint64, c_uint32, POINTER(c_uint64)]
    lib.gp_ctl_hookq_wait.restype = c_int

    # Passthrough outputs
    lib.gp_ctl_passthru_clear.argtypes = []
    lib.gp_ctl_passthru_clear.restype = None

    lib.gp_ctl_passthru_add.argtypes = [POINTER(GP_CtlPassthruDesc)]
    lib.gp_ctl_passthru_add.restype = c_int

    # Signal wheel
    lib.gp_ctl_wheel_get_meta.argtypes = [POINTER(GP_WheelMeta)]
    lib.gp_ctl_wheel_get_meta.restype = c_int

    lib.gp_ctl_wheel_get_tick_seq.argtypes = []
    lib.gp_ctl_wheel_get_tick_seq.restype = c_uint64

    lib.gp_ctl_wheel_wait.argtypes = [c_uint64, c_uint32, POINTER(c_uint64)]
    lib.gp_ctl_wheel_wait.restype = c_int

    lib.gp_ctl_wheel_peek_latest.argtypes = [c_uint32, POINTER(GP_WheelSample), POINTER(c_uint64)]
    lib.gp_ctl_wheel_peek_latest.restype = c_int

    lib.gp_ctl_wheel_peek_seq.argtypes = [c_uint32, c_uint64, POINTER(GP_WheelSample)]
    lib.gp_ctl_wheel_peek_seq.restype = c_int

    lib.gp_ctl_wheel_drain_hot.argtypes = [POINTER(c_uint32), c_uint32, POINTER(c_uint32)]
    lib.gp_ctl_wheel_drain_hot.restype = c_int


def try_load_controller_engine(*, search_dir: str | None = None):
    """Return (lib, api) or (None, None) if unavailable."""
    try:
        lib = geodesic_ctypes.load_lib(search_dir=search_dir)
        bind_controller_engine(lib)
        return lib, lib
    except Exception:
        return None, None
