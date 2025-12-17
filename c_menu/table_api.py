from __future__ import annotations

import ctypes
import os
from typing import Iterable, Optional, Sequence, Tuple

# Generic ctypes wrapper for table_abi (hierarchical rows + LEDs/axis/timers/wave cells).

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


class GP_TableCellKind:
    TEXT = 0
    LEDS = 1
    AXIS = 2
    TIMERS = 3
    WAVE = 4
    CALIB = 5
    LEDS_ARG = 6
    LEDS_TABLE = 7
    SCROLL = 8


class GP_TableRowKind:
    HEADER = 0
    DEVICE = 1
    AXIS = 2
    BUTTON = 3
    NOTE = 4


class GP_TableHitPart:
    CELL = 0
    EXPAND = 1
    LED = 2
    AXIS = 3
    CALIB_SLOT = 4
    SCROLL_UP = 5
    SCROLL_DOWN = 6
    SCROLL_THUMB = 7
    LED_TABLE = 8
    LED_ARG = 9


class GP_TableColumn(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("width_px", ctypes.c_int32),
        ("align", ctypes.c_int32),
    ]


class GP_TableCell(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("text", ctypes.c_char * 96),
        ("flags", ctypes.c_uint32),
        ("value", ctypes.c_float),
        ("hold_s", ctypes.c_float),
        ("last_s", ctypes.c_float),
        ("wave", ctypes.c_void_p),
        ("reserved0", ctypes.c_int32),
    ]


class GP_TableRow(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("depth", ctypes.c_int32),
        ("expanded", ctypes.c_int32),
        ("selected", ctypes.c_int32),
        ("label", ctypes.c_char * 96),
        ("cells", GP_TableCell * 8),
        ("cell_count", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
    ]


class GP_TableStyle(ctypes.Structure):
    _fields_ = [
        ("width_px", ctypes.c_int32),
        ("row_h_px", ctypes.c_int32),
        ("indent_px", ctypes.c_int32),
        ("expand_w_px", ctypes.c_int32),
        ("name_w_px", ctypes.c_int32),
        ("bg_rgba", ctypes.c_uint8 * 4),
        ("bg_sel_rgba", ctypes.c_uint8 * 4),
        ("hdr_rgba", ctypes.c_uint8 * 4),
        ("text_rgba", ctypes.c_uint8 * 4),
        ("text_hdr_rgba", ctypes.c_uint8 * 4),
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


class GP_TableGeom(ctypes.Structure):
    _fields_ = [
        ("width_px", ctypes.c_int32),
        ("height_px", ctypes.c_int32),
        ("col_x0", ctypes.c_int32 * 8),
        ("col_w", ctypes.c_int32 * 8),
    ]


class GP_TableHitBox(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_int32),
        ("y0", ctypes.c_int32),
        ("x1", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("row_idx", ctypes.c_int32),
        ("col_idx", ctypes.c_int32),
        ("cell_kind", ctypes.c_int32),
        ("part", ctypes.c_int32),
        ("aux0", ctypes.c_int32),
        ("aux1", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class GP_TableRenderState(ctypes.Structure):
    _fields_ = [
        ("mouse_x", ctypes.c_int32),
        ("mouse_y", ctypes.c_int32),
        ("highlight_row", ctypes.c_int32),
        ("highlight_col", ctypes.c_int32),
        ("highlight_part", ctypes.c_int32),
        ("highlight_aux0", ctypes.c_int32),
        ("highlight_color", ctypes.c_uint8 * 4),
    ]


def _rgba(t):
    r, g, b, a = (list(t) + [255, 255, 255, 255])[:4]
    return (int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF, int(a) & 0xFF)


def default_style(*, width_px: int = 800) -> GP_TableStyle:
    s = GP_TableStyle()
    s.width_px = int(width_px)
    s.row_h_px = 20
    s.indent_px = 14
    s.expand_w_px = 12
    s.name_w_px = 160

    def setc(field, val):
        arr = getattr(s, field)
        r, g, b, a = val
        arr[0], arr[1], arr[2], arr[3] = r, g, b, a

    setc("bg_rgba", _rgba((8, 8, 12, 255)))
    setc("bg_sel_rgba", _rgba((18, 18, 26, 255)))
    setc("hdr_rgba", _rgba((20, 20, 28, 255)))
    setc("text_rgba", _rgba((240, 240, 245, 255)))
    setc("text_hdr_rgba", _rgba((255, 255, 255, 255)))
    setc("led_on_rgba", _rgba((255, 210, 90, 255)))
    setc("led_off_rgba", _rgba((70, 70, 80, 255)))
    setc("led_edge_rgba", _rgba((255, 255, 255, 255)))
    setc("axis_bg_rgba", _rgba((18, 18, 22, 255)))
    setc("axis_tick_rgba", _rgba((200, 200, 200, 255)))
    setc("axis_val_rgba", _rgba((255, 255, 140, 255)))
    setc("timer_rgba", _rgba((110, 180, 255, 255)))
    setc("wave_bg_rgba", _rgba((6, 6, 8, 255)))
    setc("wave_fg_rgba", _rgba((255, 255, 255, 255)))

    default_bits = [1 << i for i in range(9)]
    for i, b in enumerate(default_bits):
        s.led_mask[i] = ctypes.c_uint32(b)
    s.led_edge_mask = ctypes.c_uint32((1 << 1) | (1 << 2) | (1 << 4) | (1 << 7) | (1 << 8))
    return s


def make_cell_text(text: str) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.TEXT
    txt = (text or "").encode("ascii", "ignore")[:95]
    padded = txt + b"\0" * (96 - len(txt))
    c.text = padded
    return c


def make_cell_leds(flags: int) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.LEDS
    c.flags = ctypes.c_uint32(int(flags))
    return c


def make_cell_axis(value: float) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.AXIS
    c.value = float(value)
    return c


def make_cell_axis_calib(
    *,
    value: float,
    v_min: float = 0.0,
    v_max: float = 0.0,
    cap_min: float = -1.0,
    cap_max: float = 1.0,
    trim: float = 0.0,
    deadzone: float = 0.10,
    seen_min: bool = True,
    seen_max: bool = True,
) -> GP_TableCell:
    """Axis cell with legacy workbench calibration ticks encoded.

    - value: current axis value (normalized)
    - v_min/v_max: observed mins/maxs
    - cap_min/cap_max/trim/deadzone: calibration markers
    - seen_min/seen_max: if False the end ticks render red
    """

    c = GP_TableCell()
    c.kind = GP_TableCellKind.AXIS
    c.value = float(value)
    c.hold_s = float(v_min)
    c.last_s = float(v_max)
    flags = 0
    if seen_min:
        flags |= 1 << 0
    if seen_max:
        flags |= 1 << 1
    c.flags = ctypes.c_uint32(flags)
    txt = f"{float(cap_min):0.4f} {float(cap_max):0.4f} {float(trim):0.4f} {float(deadzone):0.4f}".encode("ascii", "ignore")[:95]
    c.text = txt + b"\0" * (96 - len(txt))
    return c


def make_cell_timers(hold_s: float, last_s: float) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.TIMERS
    c.hold_s = float(hold_s)
    c.last_s = float(last_s)
    return c


def make_cell_wave(wave_handle: int | None) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.WAVE
    c.wave = ctypes.c_void_p(int(wave_handle)) if wave_handle else ctypes.c_void_p()
    return c


def make_cell_calib(*, invert_on: bool, mode: str = "") -> GP_TableCell:
    """Compact calibration strip rendered in the C raster (no text overlay required).

    mode: "" | "trim" | "cap" | "ded"
    """

    c = GP_TableCell()
    c.kind = GP_TableCellKind.CALIB
    flags = 0
    if invert_on:
        flags |= 1 << 0
    mode_map = {"trim": 1, "cap": 2, "ded": 3}
    flags |= (mode_map.get(str(mode).lower(), 0) & 0x3) << 1
    c.flags = ctypes.c_uint32(flags)
    return c


def make_cell_leds_arg(*, count: int = 12, linked_mask: int = 0, required_mask: int | None = None) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.LEDS_ARG
    c.value = float(int(count))  # clamp in native (currently 32)
    c.flags = ctypes.c_uint32(int(linked_mask))
    req = (1 << min(max(0, int(count)), 31)) - 1 if required_mask is None else int(required_mask)
    c.reserved0 = ctypes.c_int32(req)
    return c


def make_cell_leds_table(strips: Sequence[dict]) -> GP_TableCell:
    """Pack up to 3 LED strips into one cell (stacked vertically).

    Each strip dict: {count:int, on:int, edge:int, active:int}
    """

    class _PackedStrip(ctypes.Structure):
        _fields_ = [
            ("count", ctypes.c_uint8),
            ("reserved0", ctypes.c_uint8),
            ("reserved1", ctypes.c_uint8),
            ("reserved2", ctypes.c_uint8),
            ("on_mask", ctypes.c_uint32),
            ("edge_mask", ctypes.c_uint32),
            ("active_mask", ctypes.c_uint32),
        ]

    class _PackedTable(ctypes.Structure):
        _fields_ = [
            ("n", ctypes.c_uint8),
            ("pad0", ctypes.c_uint8),
            ("pad1", ctypes.c_uint8),
            ("pad2", ctypes.c_uint8),
            ("strips", _PackedStrip * 3),
        ]

    pt = _PackedTable()
    n = min(3, len(strips))
    pt.n = n
    for i in range(n):
        s = strips[i] or {}
        ps = _PackedStrip()
        ps.count = max(0, min(32, int(s.get("count", 0))))
        ps.on_mask = ctypes.c_uint32(int(s.get("on", 0)))
        ps.edge_mask = ctypes.c_uint32(int(s.get("edge", 0)))
        ps.active_mask = ctypes.c_uint32(int(s.get("active", 0)))
        pt.strips[i] = ps

    c = GP_TableCell()
    c.kind = GP_TableCellKind.LEDS_TABLE
    raw = bytes(pt)
    c.text = raw[:96] + b"\0" * (96 - len(raw))
    return c


def make_cell_scroll(*, value01: float, total_rows: int, visible_rows: int, arrow_up_pressed: bool = False, arrow_dn_pressed: bool = False) -> GP_TableCell:
    c = GP_TableCell()
    c.kind = GP_TableCellKind.SCROLL
    c.value = float(value01)
    c.hold_s = float(total_rows)
    c.last_s = float(visible_rows)
    flags = 0
    if arrow_up_pressed:
        flags |= 1 << 0
    if arrow_dn_pressed:
        flags |= 1 << 1
    c.flags = ctypes.c_uint32(flags)
    return c


def make_row(
    *,
    kind: int,
    label: str = "",
    depth: int = 0,
    expanded: bool = True,
    selected: bool = False,
    cells: Sequence[GP_TableCell] = (),
) -> GP_TableRow:
    r = GP_TableRow()
    r.kind = int(kind)
    r.depth = int(depth)
    r.expanded = 1 if expanded else 0
    r.selected = 1 if selected else 0
    lbl = (label or "").encode("ascii", "ignore")[:95]
    padded = lbl + b"\0" * (96 - len(lbl))
    r.label = padded
    n = min(len(cells), 8)
    r.cell_count = n
    for i in range(n):
        r.cells[i] = cells[i]
    return r


def render_table_rgba(
    rows: Sequence[GP_TableRow],
    cols: Sequence[GP_TableColumn],
    style: Optional[GP_TableStyle] = None,
) -> Optional[Tuple[int, int, bytes, GP_TableGeom]]:
    lib = _load_lib()
    if lib is None:
        return None

    lib.gp_table_calc_size.restype = ctypes.c_int32
    lib.gp_table_calc_size.argtypes = [ctypes.POINTER(GP_TableStyle), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(GP_TableGeom)]
    lib.gp_table_raster_rgba.restype = ctypes.c_int32
    lib.gp_table_raster_rgba.argtypes = [
        ctypes.POINTER(GP_TableRow), ctypes.c_int32,
        ctypes.POINTER(GP_TableColumn), ctypes.c_int32,
        ctypes.POINTER(GP_TableStyle),
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(GP_TableGeom),
    ]

    if style is None:
        style = default_style()

    geom = GP_TableGeom()
    if not lib.gp_table_calc_size(ctypes.byref(style), ctypes.c_int32(len(rows)), ctypes.c_int32(len(cols)), ctypes.byref(geom)):
        return None

    w = int(geom.width_px)
    h = int(geom.height_px)
    buf_len = w * h * 4
    buf = (ctypes.c_uint8 * buf_len)()
    arr_rows = (GP_TableRow * len(rows))(*rows)
    arr_cols = (GP_TableColumn * len(cols))(*cols)
    ok = lib.gp_table_raster_rgba(arr_rows, ctypes.c_int32(len(rows)), arr_cols, ctypes.c_int32(len(cols)), ctypes.byref(style), buf, ctypes.c_int32(buf_len), ctypes.byref(geom))
    if not ok:
        return None
    return w, h, bytes(buf), geom


def render_table_rgba_with_hits(
    rows: Sequence[GP_TableRow],
    cols: Sequence[GP_TableColumn],
    style: Optional[GP_TableStyle] = None,
    hitbox_cap: int = 4096,
) -> Optional[Tuple[int, int, bytes, GP_TableGeom, Sequence[GP_TableHitBox]]]:
    """Rasterize and return hitboxes.

    hitbox_cap: max hitboxes to collect; clipped if exceeded.
    """

    lib = _load_lib()
    if lib is None:
        return None

    lib.gp_table_calc_size.restype = ctypes.c_int32
    lib.gp_table_calc_size.argtypes = [ctypes.POINTER(GP_TableStyle), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(GP_TableGeom)]
    lib.gp_table_raster_rgba_with_hits.restype = ctypes.c_int32
    lib.gp_table_raster_rgba_with_hits.argtypes = [
        ctypes.POINTER(GP_TableRow), ctypes.c_int32,
        ctypes.POINTER(GP_TableColumn), ctypes.c_int32,
        ctypes.POINTER(GP_TableStyle),
        ctypes.POINTER(GP_TableRenderState),
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(GP_TableGeom),
        ctypes.POINTER(GP_TableHitBox), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]

    # stateful render entry (context -> rgba with optional transient state)
    lib.gp_table_render_rgba_with_state = getattr(lib, "gp_table_render_rgba_with_state")
    lib.gp_table_render_rgba_with_state.restype = ctypes.c_int32
    lib.gp_table_render_rgba_with_state.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(GP_TableRenderState),
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(GP_TableGeom),
        ctypes.POINTER(GP_TableHitBox), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]

    # click & scroll helpers
    lib.gp_table_on_click = getattr(lib, "gp_table_on_click")
    lib.gp_table_on_click.restype = ctypes.c_int32
    lib.gp_table_on_click.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(GP_TableHitBox)]
    lib.gp_table_set_scroll_fraction = getattr(lib, "gp_table_set_scroll_fraction")
    lib.gp_table_set_scroll_fraction.restype = ctypes.c_int32
    lib.gp_table_set_scroll_fraction.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.gp_table_get_scroll_fraction = getattr(lib, "gp_table_get_scroll_fraction")
    lib.gp_table_get_scroll_fraction.restype = ctypes.c_int32
    lib.gp_table_get_scroll_fraction.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.gp_table_set_scroll_fraction_xy = getattr(lib, "gp_table_set_scroll_fraction_xy")
    lib.gp_table_set_scroll_fraction_xy.restype = ctypes.c_int32
    lib.gp_table_set_scroll_fraction_xy.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
    lib.gp_table_get_scroll_fraction_xy = getattr(lib, "gp_table_get_scroll_fraction_xy")
    lib.gp_table_get_scroll_fraction_xy.restype = ctypes.c_int32
    lib.gp_table_get_scroll_fraction_xy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gp_table_get_row_count = getattr(lib, "gp_table_get_row_count")
    lib.gp_table_get_row_count.restype = ctypes.c_int32
    lib.gp_table_get_row_count.argtypes = [ctypes.c_void_p]
    lib.gp_table_get_row = getattr(lib, "gp_table_get_row")
    lib.gp_table_get_row.restype = ctypes.c_int32
    lib.gp_table_get_row.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(GP_TableRow)]
    lib.gp_table_set_led_selected = getattr(lib, "gp_table_set_led_selected")
    lib.gp_table_set_led_selected.restype = ctypes.c_int32
    lib.gp_table_set_led_selected.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.gp_table_get_led_selected = getattr(lib, "gp_table_get_led_selected")
    lib.gp_table_get_led_selected.restype = ctypes.c_int32
    lib.gp_table_get_led_selected.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.gp_table_add_edge = getattr(lib, "gp_table_add_edge")
    lib.gp_table_add_edge.restype = ctypes.c_int32
    lib.gp_table_add_edge.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.gp_table_clear_edges = getattr(lib, "gp_table_clear_edges")
    lib.gp_table_clear_edges.restype = ctypes.c_int32
    lib.gp_table_clear_edges.argtypes = [ctypes.c_void_p]
    lib.gp_table_get_edge_count = getattr(lib, "gp_table_get_edge_count")
    lib.gp_table_get_edge_count.restype = ctypes.c_int32
    lib.gp_table_get_edge_count.argtypes = [ctypes.c_void_p]
    lib.gp_table_get_edge = getattr(lib, "gp_table_get_edge")
    lib.gp_table_get_edge.restype = ctypes.c_int32
    lib.gp_table_get_edge.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]

    # node-group / edge-rule helpers
    lib.gp_table_node_group_set = getattr(lib, "gp_table_node_group_set")
    lib.gp_table_node_group_set.restype = ctypes.c_int32
    lib.gp_table_node_group_set.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32]
    lib.gp_table_node_group_get = getattr(lib, "gp_table_node_group_get")
    lib.gp_table_node_group_get.restype = ctypes.c_int32
    lib.gp_table_node_group_get.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32)]
    lib.gp_table_node_group_add_allowed = getattr(lib, "gp_table_node_group_add_allowed")
    lib.gp_table_node_group_add_allowed.restype = ctypes.c_int32
    lib.gp_table_node_group_add_allowed.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.gp_table_node_group_clear_allowed = getattr(lib, "gp_table_node_group_clear_allowed")
    lib.gp_table_node_group_clear_allowed.restype = ctypes.c_int32
    lib.gp_table_node_group_clear_allowed.argtypes = [ctypes.c_void_p]
    lib.gp_table_node_group_add_disallowed = getattr(lib, "gp_table_node_group_add_disallowed")
    lib.gp_table_node_group_add_disallowed.restype = ctypes.c_int32
    lib.gp_table_node_group_add_disallowed.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.gp_table_node_group_clear_disallowed = getattr(lib, "gp_table_node_group_clear_disallowed")
    lib.gp_table_node_group_clear_disallowed.restype = ctypes.c_int32
    lib.gp_table_node_group_clear_disallowed.argtypes = [ctypes.c_void_p]
    lib.gp_table_node_group_is_edge_allowed = getattr(lib, "gp_table_node_group_is_edge_allowed")
    lib.gp_table_node_group_is_edge_allowed.restype = ctypes.c_int32
    lib.gp_table_node_group_is_edge_allowed.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

    # relaxer APIs
    lib.gp_table_relax_set_mode = getattr(lib, "gp_table_relax_set_mode")
    lib.gp_table_relax_set_mode.restype = ctypes.c_int32
    lib.gp_table_relax_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.gp_table_relax_get_mode = getattr(lib, "gp_table_relax_get_mode")
    lib.gp_table_relax_get_mode.restype = ctypes.c_int32
    lib.gp_table_relax_get_mode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    lib.gp_table_relax_set_params = getattr(lib, "gp_table_relax_set_params")
    lib.gp_table_relax_set_params.restype = ctypes.c_int32
    lib.gp_table_relax_set_params.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32]
    lib.gp_table_relax_step = getattr(lib, "gp_table_relax_step")
    lib.gp_table_relax_step.restype = ctypes.c_int32
    lib.gp_table_relax_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.gp_table_relax_update = getattr(lib, "gp_table_relax_update")
    lib.gp_table_relax_update.restype = ctypes.c_int32
    lib.gp_table_relax_update.argtypes = [ctypes.c_void_p]
    lib.gp_table_relax_run_until_stable = getattr(lib, "gp_table_relax_run_until_stable")
    lib.gp_table_relax_run_until_stable.restype = ctypes.c_int32
    lib.gp_table_relax_run_until_stable.argtypes = [ctypes.c_void_p]

    # selected query
    lib.gp_table_get_selected_count = getattr(lib, "gp_table_get_selected_count")
    lib.gp_table_get_selected_count.restype = ctypes.c_int32
    lib.gp_table_get_selected_count.argtypes = [ctypes.c_void_p]
    lib.gp_table_get_selected_key = getattr(lib, "gp_table_get_selected_key")
    lib.gp_table_get_selected_key.restype = ctypes.c_int32
    lib.gp_table_get_selected_key.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64)]

    # prospective mode
    lib.gp_table_set_prospective_mode = getattr(lib, "gp_table_set_prospective_mode")
    lib.gp_table_set_prospective_mode.restype = ctypes.c_int32
    lib.gp_table_set_prospective_mode.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.gp_table_get_prospective_mode = getattr(lib, "gp_table_get_prospective_mode")
    lib.gp_table_get_prospective_mode.restype = ctypes.c_int32
    lib.gp_table_get_prospective_mode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    lib.gp_table_prospective_set_params = getattr(lib, "gp_table_prospective_set_params")
    lib.gp_table_prospective_set_params.restype = ctypes.c_int32
    lib.gp_table_prospective_set_params.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_float, ctypes.c_float]
    lib.gp_table_prospective_get_params = getattr(lib, "gp_table_prospective_get_params")
    lib.gp_table_prospective_get_params.restype = ctypes.c_int32
    lib.gp_table_prospective_get_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

    if style is None:
        style = default_style()

    geom = GP_TableGeom()
    if not lib.gp_table_calc_size(ctypes.byref(style), ctypes.c_int32(len(rows)), ctypes.c_int32(len(cols)), ctypes.byref(geom)):
        return None

    w = int(geom.width_px)
    h = int(geom.height_px)
    buf_len = w * h * 4
    buf = (ctypes.c_uint8 * buf_len)()
    arr_rows = (GP_TableRow * len(rows))(*rows)
    arr_cols = (GP_TableColumn * len(cols))(*cols)
    hit_arr = (GP_TableHitBox * max(0, int(hitbox_cap)))()
    written = ctypes.c_int32(0)
    ok = lib.gp_table_raster_rgba_with_hits(
        arr_rows, ctypes.c_int32(len(rows)),
        arr_cols, ctypes.c_int32(len(cols)),
        ctypes.byref(style),
        buf, ctypes.c_int32(buf_len),
        ctypes.byref(geom),
        hit_arr, ctypes.c_int32(len(hit_arr)), ctypes.byref(written),
    )
    if not ok:
        return None
    hits = tuple(hit_arr[: int(written.value)])
    return w, h, bytes(buf), geom, hits


# Stateful context wrapper ----------------------------------------------------

def _bind_stateful(lib):
    # Creation / destruction
    lib.gp_table_create.restype = ctypes.c_void_p
    lib.gp_table_create.argtypes = [ctypes.POINTER(GP_TableStyle)]
    lib.gp_table_destroy.restype = None
    lib.gp_table_destroy.argtypes = [ctypes.c_void_p]

    lib.gp_table_set_style.restype = ctypes.c_int32
    lib.gp_table_set_style.argtypes = [ctypes.c_void_p, ctypes.POINTER(GP_TableStyle)]
    lib.gp_table_set_columns.restype = ctypes.c_int32
    lib.gp_table_set_columns.argtypes = [ctypes.c_void_p, ctypes.POINTER(GP_TableColumn), ctypes.c_int32]
    lib.gp_table_set_rows.restype = ctypes.c_int32
    lib.gp_table_set_rows.argtypes = [ctypes.c_void_p, ctypes.POINTER(GP_TableRow), ctypes.c_int32]

    lib.gp_table_get_geom.restype = ctypes.c_int32
    lib.gp_table_get_geom.argtypes = [ctypes.c_void_p, ctypes.POINTER(GP_TableGeom)]

    lib.gp_table_render_rgba_with_hits.restype = ctypes.c_int32
    lib.gp_table_render_rgba_with_hits.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(GP_TableGeom),
        ctypes.POINTER(GP_TableHitBox), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]

    # Note: template/autosave/file IO APIs intentionally not bound here.
    # Persistence and template library functions live in the native C++ layer
    # and are not exposed to Python by design per user request.


class TableContext:
    """Stateful table context backed by the native library.

    Holds style/columns/rows in the native side so you can update incrementally and render repeatedly.
    """

    def __init__(self, style: Optional[GP_TableStyle] = None):
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("c_menu library not found")
        _bind_stateful(lib)
        self._lib = lib
        self._ctx = lib.gp_table_create(ctypes.byref(style) if style is not None else None)
        if not self._ctx:
            raise RuntimeError("gp_table_create failed")
        self._rows_arr = None  # keep alive
        self._cols_arr = None
        self._style = style

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.gp_table_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def set_style(self, style: Optional[GP_TableStyle]):
        if not self._ctx:
            return
        ok = self._lib.gp_table_set_style(self._ctx, ctypes.byref(style) if style is not None else None)
        if not ok:
            raise RuntimeError("gp_table_set_style failed")
        self._style = style

    def set_columns(self, cols: Sequence[GP_TableColumn]):
        if not self._ctx:
            return
        arr = (GP_TableColumn * len(cols))(*cols)
        ok = self._lib.gp_table_set_columns(self._ctx, arr, ctypes.c_int32(len(cols)))
        if not ok:
            raise RuntimeError("gp_table_set_columns failed")
        self._cols_arr = arr

    def set_rows(self, rows: Sequence[GP_TableRow]):
        if not self._ctx:
            return
        arr = (GP_TableRow * len(rows))(*rows)
        ok = self._lib.gp_table_set_rows(self._ctx, arr, ctypes.c_int32(len(rows)))
        if not ok:
            raise RuntimeError("gp_table_set_rows failed")
        self._rows_arr = arr

    def geom(self) -> GP_TableGeom:
        if not self._ctx:
            raise RuntimeError("context closed")
        g = GP_TableGeom()
        ok = self._lib.gp_table_get_geom(ctypes.c_void_p(int(self._ctx)), ctypes.byref(g))
        if not ok:
            raise RuntimeError("gp_table_get_geom failed")
        return g

    def render(self, hitbox_cap: int = 4096) -> Tuple[int, int, bytes, GP_TableGeom, Sequence[GP_TableHitBox]]:
        if not self._ctx:
            raise RuntimeError("context closed")
        g = GP_TableGeom()
        ok = self._lib.gp_table_get_geom(ctypes.c_void_p(int(self._ctx)), ctypes.byref(g))
        if not ok:
            raise RuntimeError("gp_table_get_geom failed")
        w = int(g.width_px)
        h = int(g.height_px)
        buf_len = w * h * 4
        buf = (ctypes.c_uint8 * buf_len)()
        hits_arr = (GP_TableHitBox * max(0, int(hitbox_cap)))()
        written = ctypes.c_int32(0)
        # Use stateful render entry with no transient state by default
        ok = self._lib.gp_table_render_rgba_with_state(
            ctypes.c_void_p(int(self._ctx)),
            ctypes.POINTER(GP_TableRenderState)(),
            buf, ctypes.c_int32(buf_len),
            ctypes.byref(g),
            hits_arr, ctypes.c_int32(len(hits_arr)), ctypes.byref(written),
        )
        if not ok:
            raise RuntimeError("gp_table_render_rgba_with_hits failed")
        return w, h, bytes(buf), g, tuple(hits_arr[: int(written.value)])

    def render_with_state(self, render_state: Optional[GP_TableRenderState], hitbox_cap: int = 4096) -> Tuple[int, int, bytes, GP_TableGeom, Sequence[GP_TableHitBox]]:
        if not self._ctx:
            raise RuntimeError("context closed")
        g = GP_TableGeom()
        ok = self._lib.gp_table_get_geom(self._ctx, ctypes.byref(g))
        if not ok:
            raise RuntimeError("gp_table_get_geom failed")
        w = int(g.width_px)
        h = int(g.height_px)
        buf_len = w * h * 4
        buf = (ctypes.c_uint8 * buf_len)()
        hits_arr = (GP_TableHitBox * max(0, int(hitbox_cap)))()
        written = ctypes.c_int32(0)
        ok = self._lib.gp_table_render_rgba_with_state(
            ctypes.c_void_p(int(self._ctx)),
            ctypes.byref(render_state) if render_state is not None else ctypes.POINTER(GP_TableRenderState)(),
            buf, ctypes.c_int32(buf_len),
            ctypes.byref(g),
            hits_arr, ctypes.c_int32(len(hits_arr)), ctypes.byref(written),
        )
        if not ok:
            raise RuntimeError("gp_table_render_rgba_with_state failed")
        return w, h, bytes(buf), g, tuple(hits_arr[: int(written.value)])

    def on_click(self, x: int, y: int) -> Optional[GP_TableHitBox]:
        if not self._ctx:
            raise RuntimeError("context closed")
        hb = GP_TableHitBox()
        ok = self._lib.gp_table_on_click(ctypes.c_void_p(int(self._ctx)), int(x), int(y), ctypes.byref(hb))
        if not ok:
            return None
        return hb

    def set_scroll_fraction(self, frac: float) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_set_scroll_fraction(ctypes.c_void_p(int(self._ctx)), ctypes.c_float(frac))
        if not ok:
            raise RuntimeError("gp_table_set_scroll_fraction failed")

    def get_scroll_fraction(self) -> float:
        if not self._ctx:
            raise RuntimeError("context closed")
        outf = ctypes.c_float(0.0)
        ok = self._lib.gp_table_get_scroll_fraction(ctypes.c_void_p(int(self._ctx)), ctypes.byref(outf))
        if not ok:
            raise RuntimeError("gp_table_get_scroll_fraction failed")
        return float(outf.value)

    def set_scroll_fraction_xy(self, frac_x: float, frac_y: float) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_set_scroll_fraction_xy(
            ctypes.c_void_p(int(self._ctx)),
            ctypes.c_float(frac_x),
            ctypes.c_float(frac_y),
        )
        if not ok:
            raise RuntimeError("gp_table_set_scroll_fraction_xy failed")

    def get_scroll_fraction_xy(self) -> Tuple[float, float]:
        if not self._ctx:
            raise RuntimeError("context closed")
        outx = ctypes.c_float(0.0)
        outy = ctypes.c_float(0.0)
        ok = self._lib.gp_table_get_scroll_fraction_xy(
            ctypes.c_void_p(int(self._ctx)),
            ctypes.byref(outx),
            ctypes.byref(outy),
        )
        if not ok:
            raise RuntimeError("gp_table_get_scroll_fraction_xy failed")
        return float(outx.value), float(outy.value)

    def get_row_count(self) -> int:
        if not self._ctx:
            raise RuntimeError("context closed")
        return int(self._lib.gp_table_get_row_count(ctypes.c_void_p(int(self._ctx))))

    def get_row(self, idx: int) -> GP_TableRow:
        if not self._ctx:
            raise RuntimeError("context closed")
        out = GP_TableRow()
        ok = self._lib.gp_table_get_row(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(int(idx)), ctypes.byref(out))
        if not ok:
            raise RuntimeError("gp_table_get_row failed")
        return out

    def get_selected_count(self) -> int:
        if not self._ctx:
            raise RuntimeError("context closed")
        return int(self._lib.gp_table_get_selected_count(ctypes.c_void_p(int(self._ctx))))

    def get_selected_key(self, idx: int) -> Optional[int]:
        if not self._ctx:
            raise RuntimeError("context closed")
        outk = ctypes.c_uint64(0)
        ok = self._lib.gp_table_get_selected_key(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(idx), ctypes.byref(outk))
        if not ok:
            return None
        return int(outk.value)

    def set_prospective_mode(self, enabled: bool) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_set_prospective_mode(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(1 if enabled else 0))
        if not ok:
            raise RuntimeError("gp_table_set_prospective_mode failed")

    def get_prospective_mode(self) -> bool:
        if not self._ctx:
            raise RuntimeError("context closed")
        outm = ctypes.c_int32(0)
        ok = self._lib.gp_table_get_prospective_mode(ctypes.c_void_p(int(self._ctx)), ctypes.byref(outm))
        if not ok:
            raise RuntimeError("gp_table_get_prospective_mode failed")
        return bool(outm.value)

    def set_prospective_params(self, max_history: int = 8, slack: float = 4.0, rope_length: float = 0.0) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_prospective_set_params(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(int(max_history)), ctypes.c_float(float(slack)), ctypes.c_float(float(rope_length)))
        if not ok:
            raise RuntimeError("gp_table_prospective_set_params failed")

    def get_prospective_params(self) -> tuple[int,float,float]:
        if not self._ctx:
            raise RuntimeError("context closed")
        out_max = ctypes.c_int32(0)
        out_slack = ctypes.c_float(0.0)
        out_rope = ctypes.c_float(0.0)
        ok = self._lib.gp_table_prospective_get_params(ctypes.c_void_p(int(self._ctx)), ctypes.byref(out_max), ctypes.byref(out_slack), ctypes.byref(out_rope))
        if not ok:
            raise RuntimeError("gp_table_prospective_get_params failed")
        return int(out_max.value), float(out_slack.value), float(out_rope.value)

    # Edge helpers
    def clear_edges(self) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_clear_edges(ctypes.c_void_p(int(self._ctx)))
        if not ok:
            raise RuntimeError("gp_table_clear_edges failed")

    def get_edges(self) -> list[tuple[int,int]]:
        if not self._ctx:
            raise RuntimeError("context closed")
        n = int(self._lib.gp_table_get_edge_count(ctypes.c_void_p(int(self._ctx))))
        out = []
        a = ctypes.c_uint64(0)
        b = ctypes.c_uint64(0)
        for i in range(n):
            ok = self._lib.gp_table_get_edge(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(i), ctypes.byref(a), ctypes.byref(b))
            if ok:
                out.append((int(a.value), int(b.value)))
        return out

    # Node-group / rule helpers
    def set_node_group(self, node_key: int, group: int) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_set(ctypes.c_void_p(int(self._ctx)), ctypes.c_uint64(node_key), ctypes.c_int32(group))
        if not ok:
            raise RuntimeError("gp_table_node_group_set failed")

    def get_node_group(self, node_key: int) -> Optional[int]:
        if not self._ctx:
            raise RuntimeError("context closed")
        outg = ctypes.c_int32(0)
        ok = self._lib.gp_table_node_group_get(ctypes.c_void_p(int(self._ctx)), ctypes.c_uint64(node_key), ctypes.byref(outg))
        if not ok:
            return None
        return int(outg.value)

    def add_allowed_group_pair(self, from_group: int, to_group: int) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_add_allowed(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(from_group), ctypes.c_int32(to_group))
        if not ok:
            raise RuntimeError("gp_table_node_group_add_allowed failed")

    def clear_allowed_group_pairs(self) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_clear_allowed(ctypes.c_void_p(int(self._ctx)))
        if not ok:
            raise RuntimeError("gp_table_node_group_clear_allowed failed")

    def add_disallowed_group_pair(self, from_group: int, to_group: int) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_add_disallowed(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(from_group), ctypes.c_int32(to_group))
        if not ok:
            raise RuntimeError("gp_table_node_group_add_disallowed failed")

    def clear_disallowed_group_pairs(self) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_clear_disallowed(ctypes.c_void_p(int(self._ctx)))
        if not ok:
            raise RuntimeError("gp_table_node_group_clear_disallowed failed")

    def is_edge_allowed(self, a: int, b: int) -> bool:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_node_group_is_edge_allowed(ctypes.c_void_p(int(self._ctx)), ctypes.c_uint64(a), ctypes.c_uint64(b))
        if ok < 0:
            raise RuntimeError("gp_table_node_group_is_edge_allowed failed")
        return bool(ok)

    # Relaxation control
    def set_relax_mode(self, mode: int) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_relax_set_mode(ctypes.c_void_p(int(self._ctx)), ctypes.c_int32(mode))
        if not ok:
            raise RuntimeError("gp_table_relax_set_mode failed")

    def get_relax_mode(self) -> int:
        if not self._ctx:
            raise RuntimeError("context closed")
        outm = ctypes.c_int32(0)
        ok = self._lib.gp_table_relax_get_mode(ctypes.c_void_p(int(self._ctx)), ctypes.byref(outm))
        if not ok:
            raise RuntimeError("gp_table_relax_get_mode failed")
        return int(outm.value)

    def set_relax_params(self, stiffness: float, damping: float, threshold: float = 1e-3, max_iters: int = 200) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_relax_set_params(ctypes.c_void_p(int(self._ctx)), ctypes.c_float(stiffness), ctypes.c_float(damping), ctypes.c_float(threshold), ctypes.c_int32(max_iters))
        if not ok:
            raise RuntimeError("gp_table_relax_set_params failed")

    def relax_step(self, dt: float) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_relax_step(ctypes.c_void_p(int(self._ctx)), ctypes.c_float(dt))
        if not ok:
            raise RuntimeError("gp_table_relax_step failed")

    def relax_update(self) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_relax_update(ctypes.c_void_p(int(self._ctx)))
        if not ok:
            raise RuntimeError("gp_table_relax_update failed")

    def relax_run_until_stable(self) -> None:
        if not self._ctx:
            raise RuntimeError("context closed")
        ok = self._lib.gp_table_relax_run_until_stable(ctypes.c_void_p(int(self._ctx)))
        if not ok:
            raise RuntimeError("gp_table_relax_run_until_stable failed")
