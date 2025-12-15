from __future__ import annotations

import ctypes

from ctypes import c_float, c_uint16, c_uint32, c_uint64


# IMPORTANT:
# - Keep in sync with c_physics/signal_kernel_abi.h
# - All structs are packed (_pack_=1)


class GP_InputEvent(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("t_mono_ns", c_uint64),
        ("device", c_uint32),
        ("kind", c_uint16),
        ("id", c_uint16),
        ("v0", c_float),
        ("v1", c_float),
        ("flags", c_uint32),
    ]


class GP_SignalFrame(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("t_emit_ns", c_uint64),
        ("signal_id", c_uint32),
        ("flags", c_uint32),
        ("value", c_float),
        ("hold_s", c_float),
        ("aux", c_uint32),
    ]
