from __future__ import annotations

import ctypes
from ctypes import c_float, c_int32, c_uint8, c_uint16, c_uint32, c_uint64


# IMPORTANT:
# - Keep in sync with c_physics/controller_engine_abi.h
# - All structs are packed (_pack_=1)


# Hook watch flags (match c_physics/controller_engine_abi.h)
GP_CTL_HOOK_ON_RISE = 1
GP_CTL_HOOK_ON_FALL = 2
GP_CTL_HOOK_LEVEL = 4

# Hook source
GP_CTL_HOOK_SRC_SIGNAL = 1 << 8

# Wheel sticky flags
GP_WHEEL_F_HOT = 1 << 0
GP_WHEEL_F_CHANGED = 1 << 1


class GP_CtlMeta(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("output_count", c_uint32),
        ("total_floats", c_uint32),
        ("ring_capacity", c_uint32),
        ("tick_hz", c_uint32),
        ("write_seq", c_uint64),
    ]


class GP_CtlOutputDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("channel", c_uint32),
        ("dim", c_uint16),
        ("stride", c_uint16),
        ("offset", c_uint32),
        ("nid", c_uint32),
    ]


class GP_CtlArg(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ref", c_uint8),
        ("comp", c_uint8),
        ("_pad0", c_uint16),
        ("index", c_int32),
        ("imm", c_float),
    ]


class GP_CtlNodeDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nid", c_uint32),
        ("op", c_uint16),
        ("dim", c_uint16),
        ("argc", c_uint32),
        ("args_offset", c_uint32),
        ("value", c_float),
        ("signal_id", c_uint32),
        ("sigsel", c_uint32),
    ]


class GP_CtlInputDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("iid", c_uint32),
        ("signal_id", c_uint32),
        ("sigsel", c_uint32),
        ("flags", c_uint32),
        ("cap_min", c_float),
        ("cap_max", c_float),
        ("trim", c_float),
        ("deadzone", c_float),
    ]


class GP_CtlHookWatch(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("hook_id", c_uint32),
        ("channel", c_uint32),
        ("comp", c_uint16),
        ("flags", c_uint16),
        ("threshold", c_float),
        ("hysteresis", c_float),
        ("signal_id", c_uint32),
        ("sigsel", c_uint32),
    ]


class GP_CtlHookEvent(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("t_ns", c_uint64),
        ("hook_id", c_uint32),
        ("channel", c_uint32),
        ("comp", c_uint16),
        ("flags", c_uint16),
        ("value", c_float),
        ("_pad0", c_float),
    ]


class GP_CtlHookQMeta(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("capacity", c_uint32),
        ("_pad0", c_uint32),
        ("write_seq", c_uint64),
    ]


class GP_CtlPassthruDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("channel", c_uint32),
        ("signal_id", c_uint32),
        ("sigsel", c_uint32),
        ("flags", c_uint32),
    ]


class GP_CtlTimerDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timer_id", c_uint32),
        ("period_ticks", c_uint32),
        ("duty_ticks", c_uint32),
        ("phase_ticks", c_uint32),
        ("flags", c_uint32),
        ("_reserved0", c_uint32),
    ]


class GP_WheelSample(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("t_ns", c_uint64),
        ("value", c_float),
        ("_pad0", c_uint32),
    ]


class GP_WheelMeta(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("signal_count", c_uint32),
        ("history_len", c_uint32),
        ("tick_seq", c_uint64),
    ]
