from __future__ import annotations

import ctypes
import os
from typing import Iterable, Optional, Sequence, Tuple

# Lightweight ctypes wrapper for workbench_rows_abi.

class _Lib:
    handle: Optional[ctypes.CDLL] = None


def _candidate_paths() -> Iterable[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    names = ["c_menu.dll", "libc_menu.so", "c_menu.dylib"]
    for n in names:
        yield os.path.join(here, n)
        yield os.path.join(here, "build", n)
        yield os.path.join(here, "build", "Debug", n)
        yield os.path.join(here, "build", "Release", n)


def _load_lib() -> Optional[ctypes.CDLL]:
    if _Lib.handle is not None:
        return _Lib.handle
    for p in _candidate_paths():
        if os.path.exists(p):
            try:
                _Lib.handle = ctypes.CDLL(p)
                return _Lib.handle
            except Exception:
                continue
    return None


class GP_WbRowKind:
    DEVICE_HEADER = 0
    DEVICE_NOTE = 1
    AXIS = 2
    BUTTON = 3


class GP_WbRow(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("label", ctypes.c_char * 64),
        ("value", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("hold_s", ctypes.c_float),
        ("last_s", ctypes.c_float),
        ("selected", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("wave", ctypes.c_void_p),
        ("reserved1", ctypes.c_int32),
    ]


class GP_WbStyle(ctypes.Structure):
    _fields_ = [
        ("width_px", ctypes.c_int32),
        ("row_h_px", ctypes.c_int32),
        ("name_w_px", ctypes.c_int32),
        ("led_spacing_px", ctypes.c_int32),
        ("led_radius_px", ctypes.c_int32),
        ("axis_w_px", ctypes.c_int32),
        ("wave_w_px", ctypes.c_int32),
        ("wave_h_px", ctypes.c_int32),
        ("bg_rgba", ctypes.c_uint8 * 4),
        ("bg_sel_rgba", ctypes.c_uint8 * 4),
        ("hdr_rgba", ctypes.c_uint8 * 4),
        ("led_on_rgba", ctypes.c_uint8 * 4),
        ("led_off_rgba", ctypes.c_uint8 * 4),
        ("led_edge_rgba", ctypes.c_uint8 * 4),
        ("axis_bg_rgba", ctypes.c_uint8 * 4),
        ("axis_tick_rgba", ctypes.c_uint8 * 4),
        ("axis_val_rgba", ctypes.c_uint8 * 4),
        ("timer_rgba", ctypes.c_uint8 * 4),
        ("wave_bg_rgba", ctypes.c_uint8 * 4),
        ("wave_fg_rgba", ctypes.c_uint8 * 4),
        ("led_mask", ctypes.c_uint32 * 9),
        ("led_edge_mask", ctypes.c_uint32),
    ]


def _rgba(t: Sequence[int]) -> Tuple[int, int, int, int]:
    r, g, b, a = (list(t) + [255, 255, 255, 255])[:4]
    return (int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF, int(a) & 0xFF)


def default_style(*, width_px: int = 640) -> GP_WbStyle:
    s = GP_WbStyle()
    s.width_px = int(width_px)
    s.row_h_px = 20
    s.name_w_px = 160
    s.led_spacing_px = 14
    s.led_radius_px = 4
    s.axis_w_px = 180
    s.wave_w_px = 70
    s.wave_h_px = 14

    def set_color(field: str, val: Tuple[int, int, int, int]):
        arr = getattr(s, field)
        r, g, b, a = val
        arr[0], arr[1], arr[2], arr[3] = r, g, b, a

    set_color("bg_rgba", _rgba((8, 8, 12, 255)))
    set_color("bg_sel_rgba", _rgba((18, 18, 26, 255)))
    set_color("hdr_rgba", _rgba((20, 20, 28, 255)))
    set_color("led_on_rgba", _rgba((255, 210, 90, 255)))
    set_color("led_off_rgba", _rgba((70, 70, 80, 255)))
    set_color("led_edge_rgba", _rgba((255, 255, 255, 255)))
    set_color("axis_bg_rgba", _rgba((18, 18, 22, 255)))
    set_color("axis_tick_rgba", _rgba((200, 200, 200, 255)))
    set_color("axis_val_rgba", _rgba((255, 255, 140, 255)))
    set_color("timer_rgba", _rgba((110, 180, 255, 255)))
    set_color("wave_bg_rgba", _rgba((6, 6, 8, 255)))
    set_color("wave_fg_rgba", _rgba((255, 255, 255, 255)))

    # Default LED masks mirror legacy order: D + - H 2 2H T t 2t
    default_bits = [
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
    ]
    for i, b in enumerate(default_bits):
        s.led_mask[i] = ctypes.c_uint32(b)
    s.led_edge_mask = ctypes.c_uint32((1 << 1) | (1 << 2) | (1 << 7) | (1 << 8) | (1 << 4))
    return s


def make_row(
    *,
    kind: int,
    label: str = "",
    value: float = 0.0,
    flags: int = 0,
    hold_s: float = 0.0,
    last_s: float = 0.0,
    selected: bool = False,
    wave: int | None = None,
) -> GP_WbRow:
    r = GP_WbRow()
    r.kind = int(kind)
    r.value = float(value)
    r.flags = ctypes.c_uint32(int(flags))
    r.hold_s = float(hold_s)
    r.last_s = float(last_s)
    r.selected = 1 if selected else 0
    r.wave = ctypes.c_void_p(int(wave)) if wave else ctypes.c_void_p()
    label_bytes = (label or "").encode("ascii", "ignore")[:63]
    padded = label_bytes + b"\0" * (64 - len(label_bytes))
    r.label = padded
    return r


def render_rows_rgba(rows: Sequence[GP_WbRow], style: Optional[GP_WbStyle] = None) -> Optional[Tuple[int, int, bytes]]:
    lib = _load_lib()
    if lib is None:
        return None

    lib.gp_wb_rows_calc_size.restype = ctypes.c_int32
    lib.gp_wb_rows_calc_size.argtypes = [ctypes.POINTER(GP_WbStyle), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    lib.gp_wb_rows_raster_rgba.restype = ctypes.c_int32
    lib.gp_wb_rows_raster_rgba.argtypes = [ctypes.POINTER(GP_WbRow), ctypes.c_int32, ctypes.POINTER(GP_WbStyle), ctypes.c_void_p, ctypes.c_int32]

    if style is None:
        style = default_style()

    w_out = ctypes.c_int32(0)
    h_out = ctypes.c_int32(0)
    if not lib.gp_wb_rows_calc_size(ctypes.byref(style), ctypes.c_int32(len(rows)), ctypes.byref(w_out), ctypes.byref(h_out)):
        return None
    w = int(w_out.value)
    h = int(h_out.value)
    buf_len = w * h * 4
    buf = (ctypes.c_uint8 * buf_len)()
    arr = (GP_WbRow * len(rows))(*rows)
    ok = lib.gp_wb_rows_raster_rgba(arr, ctypes.c_int32(len(rows)), ctypes.byref(style), buf, ctypes.c_int32(buf_len))
    if not ok:
        return None
    return w, h, bytes(buf)
