from __future__ import annotations

import ctypes

from ctypes import c_float, c_int


class GP_FlightIn(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dt", c_float),
        ("pos", c_float * 3),
        ("vel", c_float * 3),
        ("right_b", c_float * 3),
        ("up_b", c_float * 3),
        ("fwd_b", c_float * 3),
        ("up_rad", c_float * 3),
        ("throttle", c_float),
        ("strafe", c_float),
        ("thrust_accel", c_float),
        ("strafe_accel", c_float),
        ("lift_k", c_float),
        ("drag_k", c_float),
        ("gravity_g", c_float),
        ("max_speed", c_float),
        ("rho", c_float),
    ]


class GP_FlightOut(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pos", c_float * 3),
        ("vel", c_float * 3),
    ]


def bind_gp_flight_step(lib: ctypes.CDLL) -> None:
    """Bind gp_flight_step symbol if present."""
    fn = getattr(lib, "gp_flight_step")
    fn.argtypes = [ctypes.POINTER(GP_FlightIn), ctypes.POINTER(GP_FlightOut)]
    fn.restype = c_int


def try_load_flight_step(*, search_dir: str | None = None):
    """Return (lib, gp_flight_step) or (None, None) if unavailable."""
    try:
        from . import geodesic_ctypes

        lib = geodesic_ctypes.load_lib(search_dir=search_dir)
        bind_gp_flight_step(lib)
        return lib, lib.gp_flight_step
    except Exception:
        return None, None
