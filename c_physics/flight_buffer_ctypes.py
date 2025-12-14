from __future__ import annotations

import ctypes
from ctypes import c_float, c_uint32, c_size_t, c_uint64


class GP_FlightBufHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("front_idx", c_uint32),
        ("sim_idx", c_uint32),
        ("swap_requested", c_uint32),
        ("swap_seq", c_uint32),
        ("controls_seq", c_uint32),
        ("_pad0", c_uint32),
    ]


class GP_FlightConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("planet_surface_r", c_float),
        ("flight_r_min", c_float),
        ("flight_r_max", c_float),
        ("atmosphere_alt_max", c_float),
        ("mass", c_float),
        ("inertia_diag", c_float * 3),
        ("terrain_hm_ptr", c_uint64),
        ("terrain_hm_w", c_uint32),
        ("terrain_hm_h", c_uint32),
        ("terrain_hm_stride", c_uint32),
        ("terrain_height_scale", c_float),
        ("terrain_height_bias", c_float),
    ]


class GP_FlightArm(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", c_uint32),
        ("input_idx", c_uint32),
        ("flags", c_uint32),
        ("_pad0", c_uint32),
        ("pos_b", c_float * 3),
        ("dir_b", c_float * 3),
        ("axis_b", c_float * 3),
        ("max_force", c_float),
        ("k_lift", c_float),
        ("k_drag", c_float),
        ("eff_rho_pow", c_float),
        ("eff_speed_ref", c_float),
        ("eff_speed_pow", c_float),
        ("temp_heat_rate", c_float),
        ("temp_cool_rate", c_float),
        ("temp_overheat_start", c_float),
        ("temp_overheat_end", c_float),
    ]


class GP_FlightControls(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("throttle", c_float),
        ("strafe", c_float),
        ("desired_q", c_float * 4),
        ("max_ang_rate", c_float),
        ("mode", c_uint32),
        ("_pad_mode", c_uint32),
        ("auto_kp", c_float),
        ("auto_kd", c_float),
        ("channels", c_float * 8),
        ("thrust_accel", c_float),
        ("strafe_accel", c_float),
        ("lift_k", c_float),
        ("drag_k", c_float),
        ("gravity_g", c_float),
        ("max_speed", c_float),
        ("flags", c_uint32),
        ("_pad0", c_uint32),
    ]


class GP_FlightControlSystem(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("mode", c_uint32),
        ("flags", c_uint32),
        ("_pad0", c_uint32),
        ("_pad1", c_uint32),
        ("desired_q", c_float * 4),
        ("err_axis_b", c_float * 3),
        ("err_angle", c_float),
        ("tau_des_b", c_float * 3),
        ("tau_est_b", c_float * 3),
        ("_pad2", c_float * 2),
        ("channel_cmd", c_float * 8),
    ]


class GP_FlightState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pos", c_float * 3),
        ("vel", c_float * 3),
        ("q", c_float * 4),
        ("omega_b", c_float * 3),
        ("_pad_omega", c_float),
        ("rho", c_float),
        ("altitude", c_float),
        ("r_surface", c_float),
        ("_pad0", c_float),
        ("last_requested_ang", c_float),
        ("last_applied_ang", c_float),
        ("last_ang_clamp_ratio", c_float),
        ("_pad1", c_float),
        ("arm_temp", c_float * 32),
        ("arm_eff", c_float * 32),
        ("ctrl", GP_FlightControlSystem),
    ]


class GP_FlightBuffer(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("hdr", GP_FlightBufHeader),
        ("cfg", GP_FlightConfig),
        ("ctl", GP_FlightControls),
        ("arm_count", c_uint32),
        ("_pad_arm0", c_uint32),
        ("_pad_arm1", c_uint32),
        ("_pad_arm2", c_uint32),
        ("arms", GP_FlightArm * 32),
        ("state", GP_FlightState * 2),
    ]


def try_load_flight_buffer_lib(dll_path: str | None = None):
    """Load geodesic_physics.dll and bind the flight buffer API.

    Returns (lib, api) where api is a small object with bound callables.
    On failure returns (None, None).
    """

    try:
        if dll_path is None:
            # Co-located with this package.
            import os

            here = os.path.dirname(__file__)
            dll_path = os.path.join(here, "geodesic_physics.dll")

        lib = ctypes.CDLL(dll_path)

        lib.gp_flight_required_bytes.restype = c_size_t
        lib.gp_flight_required_bytes.argtypes = []

        lib.gp_flight_init.restype = ctypes.c_int
        lib.gp_flight_init.argtypes = [ctypes.c_void_p]

        lib.gp_flight_request_swap.restype = None
        lib.gp_flight_request_swap.argtypes = [ctypes.c_void_p]

        lib.gp_flight_get_swap_seq.restype = c_uint32
        lib.gp_flight_get_swap_seq.argtypes = [ctypes.c_void_p]

        lib.gp_flight_buf_step.restype = None
        lib.gp_flight_buf_step.argtypes = [ctypes.c_void_p, c_float, c_uint32]

        class _Api:
            required_bytes = lib.gp_flight_required_bytes
            init = lib.gp_flight_init
            request_swap = lib.gp_flight_request_swap
            get_swap_seq = lib.gp_flight_get_swap_seq
            step = lib.gp_flight_buf_step

        return lib, _Api()
    except Exception:
        return None, None
