from __future__ import annotations

"""
Interactive pygame demo for the C++ c_menu table raster.

- Mirrors the signal_workbench layout: left = devices/inputs, right = signals/ops.
- Uses render_table_rgba_with_hits to gather hit metadata and drive interaction:
  click expanders, LEDs, calibration slots, and the synthetic scrollbar to mutate state.
- Intended as a bridge while developing a non-blocking, event-driven replacement for the
  legacy blocking workbench. The demo keeps all layout in the C++ raster while Python
  handles hit interpretation and selection state.
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pygame
except Exception as exc:  # pragma: no cover
    print(f"Pygame is required: pip install pygame ({exc})", file=sys.stderr)
    sys.exit(1)

try:
    from c_menu import table_api as ta
except Exception as exc:  # pragma: no cover
    print(f"Failed to import table ABI helpers: {exc}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------------------
# Small helpers

COL_TEXT = (235, 235, 240)
COL_SHADOW = (18, 18, 22)


@dataclass
class RowMeta:
    label: str
    expand_key: Optional[str] = None
    pane: str = "left"  # "left" | "right"


@dataclass
class DemoState:
    expanded: Dict[str, bool] = field(
        default_factory=lambda: {"inputs": True, "hat.0": True, "hat.1": True, "signals": True, "sig.view2d": False}
    )
    selected_left: int = 1
    selected_right: int = 1
    scroll_value: float = 0.25
    arg_linked_mask: int = 0b100001010101
    led_table: List[Dict[str, int]] = field(
        default_factory=lambda: [
            {"count": 6, "on": 0b101101, "edge": 0b001001, "active": 0b111111},
            {"count": 8, "on": 0b11110000, "edge": 0, "active": 0b01010101},
        ]
    )
    button_leds: int = 0b101011001
    calib_mode: str = "trim"
    invert_on: bool = False

    def toggle_expand(self, key: str, default: bool = True) -> None:
        self.expanded[key] = not self.expanded.get(key, default)

    def set_scroll(self, value: float) -> None:
        self.scroll_value = max(0.0, min(1.0, float(value)))


class SparkStore:
    """Manages per-row sparkline waveforms and sample cursors."""

    def __init__(self, lib, w: int = 120, h: int = 18, cap: int = 256) -> None:
        self._lib = lib
        self._w = int(w)
        self._h = int(h)
        self._cap = int(cap)
        self._handles: Dict[str, int] = {}
        self._t: Dict[str, int] = {}

        lib.gp_menu_waveform_create.restype = ta.ctypes.c_void_p  # type: ignore[attr-defined]
        lib.gp_menu_waveform_create.argtypes = [ta.ctypes.c_int32, ta.ctypes.c_int32, ta.ctypes.c_int32]
        lib.gp_menu_waveform_push_sample.restype = None
        lib.gp_menu_waveform_push_sample.argtypes = [ta.ctypes.c_void_p, ta.ctypes.c_uint64, ta.ctypes.c_float]

    def handle(self, key: str) -> int | None:
        k = str(key)
        if k in self._handles:
            return self._handles[k]
        wf = self._lib.gp_menu_waveform_create(self._w, self._h, self._cap)
        if not wf:
            return None
        self._handles[k] = int(wf)
        self._t[k] = 0
        return int(wf)

    def push(self, key: str, value: float) -> int | None:
        h = self.handle(key)
        if h is None:
            return None
        t = int(self._t.get(str(key), 0))
        v = max(-1.0, min(1.0, float(value)))
        self._lib.gp_menu_waveform_push_sample(ta.ctypes.c_void_p(h), ta.ctypes.c_uint64(t), ta.ctypes.c_float(v))
        self._t[str(key)] = t + 1
        return h


def _pcm_sample(shape: str, t: int) -> float:
    tt = float(t)
    if shape == "saw":
        return ((tt % 64) / 64.0) * 2.0 - 1.0
    if shape == "square":
        return 1.0 if (int(tt) // 16) % 2 == 0 else -1.0
    if shape == "tri":
        phase = (tt % 64) / 64.0
        return 4.0 * abs(phase - 0.5) - 1.0
    return math.sin(tt * 0.12)


def _style_for_workbench(width: int, row_h: int) -> ta.GP_TableStyle:
    s = ta.default_style(width_px=width)
    s.row_h_px = int(row_h)
    s.name_w_px = 260
    s.indent_px = 16
    s.expand_w_px = 12
    # Text is drawn in Python; keep palette aligned with legacy workbench.
    def setc(field, rgba):
        arr = getattr(s, field)
        arr[0], arr[1], arr[2], arr[3] = [int(x) & 0xFF for x in rgba]

    setc("led_on_rgba", (64, 255, 64, 255))
    setc("led_off_rgba", (26, 52, 26, 255))
    setc("led_edge_rgba", (255, 255, 255, 255))
    setc("axis_bg_rgba", (20, 20, 24, 255))
    setc("axis_tick_rgba", (240, 240, 240, 255))
    setc("axis_val_rgba", (255, 255, 90, 255))
    setc("timer_rgba", (110, 180, 255, 255))
    setc("wave_bg_rgba", (12, 16, 22, 255))
    setc("wave_fg_rgba", (90, 180, 255, 255))
    setc("bg_rgba", (6, 6, 8, 255))
    setc("bg_sel_rgba", (18, 18, 26, 255))
    setc("hdr_rgba", (22, 22, 32, 255))
    setc("text_rgba", (240, 240, 245, 255))
    setc("text_hdr_rgba", (255, 255, 255, 255))
    return s


def _decode_text(buf: bytes) -> str:
    return buf.split(b"\0", 1)[0].decode("ascii", "ignore")


def _overlay_text(screen: pygame.Surface, rows: Sequence[ta.GP_TableRow], geom: ta.GP_TableGeom, style: ta.GP_TableStyle, font: pygame.font.Font, *, offset_x: int = 0) -> None:
    row_h = int(style.row_h_px)
    indent = int(style.indent_px)
    expand_w = int(style.expand_w_px)
    for ri, row in enumerate(rows):
        y = ri * row_h + row_h // 2
        x_label = offset_x + 6 + expand_w + indent * int(row.depth)
        label = _decode_text(row.label)
        txt = font.render(label, True, COL_TEXT)
        screen.blit(txt, (x_label, y - txt.get_height() // 2))

        # Draw TEXT cells in their columns so you can see signal args/op labels.
        for ci in range(min(row.cell_count, len(geom.col_x0))):
            cell = row.cells[ci]
            if cell.kind != ta.GP_TableCellKind.TEXT:
                continue
            cx = offset_x + int(geom.col_x0[ci]) + 6
            cw = int(geom.col_w[ci])
            if cw <= 0:
                continue
            ctext = _decode_text(cell.text)
            if not ctext:
                continue
            cimg = font.render(ctext, True, COL_TEXT)
            screen.blit(cimg, (cx, y - cimg.get_height() // 2))


def _make_cols_left() -> List[ta.GP_TableColumn]:
    cols = []
    for kind, w in [
        (ta.GP_TableCellKind.LEDS, 150),
        (ta.GP_TableCellKind.AXIS, 260),
        (ta.GP_TableCellKind.CALIB, 140),
        (ta.GP_TableCellKind.WAVE, 200),
        (ta.GP_TableCellKind.LEDS_ARG, 170),
        (ta.GP_TableCellKind.LEDS_TABLE, 180),
        (ta.GP_TableCellKind.SCROLL, 46),
    ]:
        c = ta.GP_TableColumn()
        c.kind = kind
        c.width_px = w
        c.align = 0
        cols.append(c)
    return cols


def _make_cols_right() -> List[ta.GP_TableColumn]:
    cols = []
    for kind, w in [
        (ta.GP_TableCellKind.TEXT, 220),
        (ta.GP_TableCellKind.TEXT, 140),
        (ta.GP_TableCellKind.WAVE, 160),
        (ta.GP_TableCellKind.TEXT, 60),
        (ta.GP_TableCellKind.TEXT, 80),
    ]:
        c = ta.GP_TableColumn()
        c.kind = kind
        c.width_px = w
        c.align = 0
        cols.append(c)
    return cols


def _build_left_rows(frame: int, sparks: SparkStore, state: DemoState) -> Tuple[List[ta.GP_TableRow], List[RowMeta]]:
    rows: List[ta.GP_TableRow] = []
    meta: List[RowMeta] = []

    header_row_idx = len(rows)
    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.HEADER,
            label="INPUTS",
            depth=0,
            expanded=True,
            selected=state.selected_left == header_row_idx,
            cells=[
                ta.make_cell_text("LEDs"),
                ta.make_cell_text("Axis"),
                ta.make_cell_text("Calib"),
                ta.make_cell_text("Wave"),
                ta.make_cell_text("Args"),
                ta.make_cell_text("LED table"),
                ta.make_cell_text("Scroll"),
            ],
        )
    )
    meta.append(RowMeta(label="INPUTS", expand_key=None, pane="left"))

    device_expanded = state.expanded.get("inputs", True)
    device_row_idx = len(rows)
    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.DEVICE,
            label="joystick.0",
            depth=0,
            expanded=device_expanded,
            selected=state.selected_left == device_row_idx,
            cells=[
                ta.make_cell_leds(flags=state.button_leds),
                ta.make_cell_axis(math.sin(frame * 0.06) * 0.7),
                ta.make_cell_calib(invert_on=state.invert_on, mode=state.calib_mode),
                ta.make_cell_wave(sparks.push("joy.wave", _pcm_sample("sine", frame))),
                ta.make_cell_leds_arg(count=12, linked_mask=state.arg_linked_mask, required_mask=0b111100001111),
                ta.make_cell_leds_table(state.led_table),
                ta.make_cell_scroll(
                    value01=state.scroll_value,
                    total_rows=32,
                    visible_rows=8,
                    arrow_up_pressed=False,
                    arrow_dn_pressed=False,
                ),
            ],
        )
    )
    meta.append(RowMeta(label="joystick.0", expand_key="inputs", pane="left"))

    if device_expanded:
        # Axis row with calibration ticks and waveform.
        axis_row_idx = len(rows)
        rows.append(
            ta.make_row(
                kind=ta.GP_TableRowKind.AXIS,
                label="axis.x",
                depth=1,
                expanded=True,
                selected=state.selected_left == axis_row_idx,
                cells=[
                    ta.make_cell_leds(flags=0),
                    ta.make_cell_axis_calib(
                        value=math.sin(frame * 0.05) * 0.9,
                        v_min=-0.9,
                        v_max=0.8,
                        cap_min=-1.0,
                        cap_max=1.0,
                        trim=0.12,
                        deadzone=0.08,
                        seen_min=True,
                        seen_max=frame > 60,
                    ),
                    ta.make_cell_calib(invert_on=False, mode="cap"),
                    ta.make_cell_wave(sparks.push("axis.x", _pcm_sample("tri", frame))),
                ],
            )
        )
        meta.append(RowMeta(label="axis.x", pane="left"))

        # Button row with manually toggleable LEDs.
        btn_row_idx = len(rows)
        rows.append(
            ta.make_row(
                kind=ta.GP_TableRowKind.BUTTON,
                label="btn.flags",
                depth=1,
                expanded=True,
                selected=state.selected_left == btn_row_idx,
                cells=[
                    ta.make_cell_leds(flags=state.button_leds),
                    ta.make_cell_axis(0.0),
                    ta.make_cell_wave(sparks.push("btn.flags", _pcm_sample("square", frame))),
                ],
            )
        )
        meta.append(RowMeta(label="btn.flags", pane="left"))

        # Two hats that can be collapsed individually.
        for hat_idx in range(2):
            hat_key = f"hat.{hat_idx}"
            expanded = state.expanded.get(hat_key, True)
            dir_cycle = (frame // 14 + hat_idx * 2) % 8
            flags = 1 << dir_cycle if dir_cycle < 8 else 0
            hat_row_idx = len(rows)
            rows.append(
                ta.make_row(
                    kind=ta.GP_TableRowKind.DEVICE,
                    label=hat_key,
                    depth=1,
                    expanded=expanded,
                    selected=state.selected_left == hat_row_idx,
                    cells=[
                        ta.make_cell_leds(flags=flags),
                        ta.make_cell_axis_calib(
                            value=math.cos(dir_cycle * (math.pi / 4.0)),
                            v_min=-1.0,
                            v_max=1.0,
                            cap_min=-1.0,
                            cap_max=1.0,
                            trim=0.0,
                            deadzone=0.0,
                            seen_min=True,
                            seen_max=True,
                        ),
                        ta.make_cell_wave(sparks.push(hat_key, _pcm_sample("saw", frame + hat_idx * 8))),
                    ],
                )
            )
            meta.append(RowMeta(label=hat_key, expand_key=hat_key, pane="left"))

            if expanded:
                hat_btn_idx = len(rows)
                rows.append(
                    ta.make_row(
                        kind=ta.GP_TableRowKind.BUTTON,
                        label=f"{hat_key}.press",
                        depth=2,
                        expanded=True,
                        selected=state.selected_left == hat_btn_idx,
                        cells=[
                            ta.make_cell_leds(flags=(1 << 0) if (frame + hat_idx) % 20 < 10 else 0),
                            ta.make_cell_axis(0.0),
                        ],
                    )
                )
                meta.append(RowMeta(label=f"{hat_key}.press", pane="left"))

    return rows, meta


def _build_right_rows(frame: int, sparks: SparkStore, state: DemoState) -> Tuple[List[ta.GP_TableRow], List[RowMeta]]:
    rows: List[ta.GP_TableRow] = []
    meta: List[RowMeta] = []

    # Signal list mirrors the legacy right pane: label, args, wave, op, channel.
    sig_defs = [
        {"label": "sig.throttle", "req": 2, "linked": 2, "expand": False, "op": "2+", "ch": 0, "dim": 1},
        {"label": "sig.pitch", "req": 1, "linked": 0, "expand": False, "op": "K", "ch": 1, "dim": 1},
        {"label": "sig.view2d", "req": 2, "linked": 1, "expand": True, "op": "2S", "ch": -1, "dim": 2},
        {"label": "sig.raw_kernel", "req": 1, "linked": 1, "expand": False, "op": "K", "ch": 2, "dim": 1},
        {"label": "NEW", "req": 0, "linked": 0, "expand": False, "op": "?", "ch": -1, "dim": 1},
    ]

    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.HEADER,
            label="SIGNALS",
            depth=0,
            expanded=state.expanded.get("signals", True),
            selected=False,
            cells=[
                ta.make_cell_text("Name"),
                ta.make_cell_text("Args"),
                ta.make_cell_text("Wave"),
                ta.make_cell_text("Op"),
                ta.make_cell_text("Ch"),
            ],
        )
    )
    meta.append(RowMeta(label="SIGNALS", expand_key="signals", pane="right"))

    if not state.expanded.get("signals", True):
        return rows, meta

    for i, sig in enumerate(sig_defs):
        label = str(sig["label"])
        req = int(sig["req"])
        linked = int(sig["linked"])
        expandable = bool(sig.get("expand", False))
        op_lbl = str(sig.get("op", "?"))
        ch = int(sig.get("ch", -1))
        dim = int(sig.get("dim", 1))

        v = _pcm_sample("tri", frame + i * 9) * (0.5 if dim == 1 else 1.0)
        wf = sparks.push(f"sig.{label}.x", v)

        flags = (req & 0xFF) | ((linked & 0xFF) << 8) | (12 << 16) | (0x01000000 if expandable else 0)

        cells: List[ta.GP_TableCell] = []
        c_name = ta.make_cell_text(label)
        if expandable:
            c_name.flags = flags
        cells.append(c_name)
        c_args = ta.make_cell_text(f"args {linked}/{req}")
        c_args.flags = flags
        cells.append(c_args)
        cells.append(ta.make_cell_wave(wf))
        cells.append(ta.make_cell_text(op_lbl))
        cells.append(ta.make_cell_text("ch:-" if ch < 0 else f"ch:{ch}"))

        row_idx = len(rows)
        selected = state.selected_right == row_idx
        rows.append(
            ta.make_row(
                kind=ta.GP_TableRowKind.NOTE,
                label=label,
                depth=0,
                expanded=state.expanded.get(label, False),
                selected=selected,
                cells=cells,
            )
        )
        meta.append(RowMeta(label=label, expand_key=(label if expandable else None), pane="right"))

        # Expandable signal shows child axes when opened.
        if expandable and state.expanded.get(label, False):
            for dim_idx in range(dim):
                rows.append(
                    ta.make_row(
                        kind=ta.GP_TableRowKind.NOTE,
                        label=f"{label}[{dim_idx}]",
                        depth=1,
                        expanded=True,
                        selected=False,
                        cells=[
                            ta.make_cell_text(f"{label}[{dim_idx}]"),
                            ta.make_cell_text("arg link"),
                            ta.make_cell_wave(sparks.push(f"sig.{label}.{dim_idx}", _pcm_sample("saw", frame + dim_idx * 7))),
                            ta.make_cell_text("dim"),
                            ta.make_cell_text("ch:-"),
                        ],
                    )
                )
                meta.append(RowMeta(label=f"{label}[{dim_idx}]", pane="right"))

    return rows, meta


# ------------------------------------------------------------------------------
# Interaction helpers


def _find_hover(hits: Sequence[ta.GP_TableHitBox], pos: Tuple[int, int]) -> Optional[ta.GP_TableHitBox]:
    mx, my = pos
    for hb in hits:
        if hb.x0 <= mx < hb.x1 and hb.y0 <= my < hb.y1:
            return hb
    return None


def _handle_hit(state: DemoState, meta: Sequence[RowMeta], hb: ta.GP_TableHitBox, pos_y: int) -> None:
    if hb.row_idx < 0 or hb.row_idx >= len(meta):
        return
    row_meta = meta[hb.row_idx]
    if hb.part == ta.GP_TableHitPart.EXPAND and row_meta.expand_key:
        state.toggle_expand(row_meta.expand_key)
        return
    if hb.part == ta.GP_TableHitPart.CELL:
        if row_meta.pane == "left":
            state.selected_left = hb.row_idx
        else:
            state.selected_right = hb.row_idx
        return
    if hb.part == ta.GP_TableHitPart.LED and row_meta.label == "btn.flags":
        bit = 1 << max(0, int(hb.aux0))
        state.button_leds ^= bit
        return
    if hb.part == ta.GP_TableHitPart.LED_ARG:
        bit = 1 << max(0, int(hb.aux0))
        state.arg_linked_mask ^= bit
        return
    if hb.part == ta.GP_TableHitPart.LED_TABLE:
        strip = int(hb.aux0)
        led = int(hb.aux1)
        if 0 <= strip < len(state.led_table):
            state.led_table[strip]["on"] ^= 1 << max(0, led)
        return
    if hb.part == ta.GP_TableHitPart.CALIB_SLOT:
        slot = int(hb.aux0)
        if slot == 0:
            state.invert_on = not state.invert_on
        elif slot == 4:
            state.calib_mode = ""
            state.invert_on = False
        else:
            mode_map = {1: "trim", 2: "cap", 3: "ded"}
            state.calib_mode = mode_map.get(slot, state.calib_mode)
        return
    if hb.part == ta.GP_TableHitPart.SCROLL_UP:
        state.set_scroll(state.scroll_value - 0.05)
        return
    if hb.part == ta.GP_TableHitPart.SCROLL_DOWN:
        state.set_scroll(state.scroll_value + 0.05)
        return
    if hb.part == ta.GP_TableHitPart.SCROLL_THUMB:
        # Map y within the thumb to 0..1 for demo purposes.
        thumb_h = max(1, hb.y1 - hb.y0)
        thumb_pos = max(0, min(thumb_h, pos_y - hb.y0))
        frac = thumb_pos / float(thumb_h)
        state.set_scroll(frac)
        return


def _render_pane(rows: Sequence[ta.GP_TableRow], cols: Sequence[ta.GP_TableColumn], style: ta.GP_TableStyle, hit_cap: int = 8192) -> Optional[Tuple[int, int, bytes, ta.GP_TableGeom, Sequence[ta.GP_TableHitBox]]]:
    return ta.render_table_rgba_with_hits(rows, cols, style=style, hitbox_cap=hit_cap)


# ------------------------------------------------------------------------------
# Main loop


def main() -> int:
    pygame.init()
    font = pygame.font.SysFont("Consolas", 14)

    width_left = 1100
    width_right = 660
    row_h = 22
    cols_left = _make_cols_left()
    cols_right = _make_cols_right()

    style_left = _style_for_workbench(width=width_left, row_h=row_h)
    style_right = _style_for_workbench(width=width_right, row_h=row_h)
    style_right.name_w_px = 12
    style_right.indent_px = 10
    style_right.expand_w_px = 10

    lib = ta._load_lib()
    if lib is None:
        print("c_menu DLL not found; build c_menu first", file=sys.stderr)
        return 1
    sparks = SparkStore(lib)
    state = DemoState()

    clock = pygame.time.Clock()
    frame = 0
    gap = 10
    info_h = 28

    running = True
    hover: Optional[Tuple[str, ta.GP_TableHitBox, int]] = None  # (pane, hit, offset_x)
    while running:
        rows_left, meta_left = _build_left_rows(frame, sparks, state)
        rows_right, meta_right = _build_right_rows(frame, sparks, state)

        out_left = _render_pane(rows_left, cols_left, style_left)
        out_right = _render_pane(rows_right, cols_right, style_right)
        if out_left is None or out_right is None:
            print("table render failed (DLL missing?)", file=sys.stderr)
            break

        wl, hl, rgba_l, geom_l, hits_l = out_left
        wr, hr, rgba_r, geom_r, hits_r = out_right
        surf_left = pygame.image.frombuffer(rgba_l, (wl, hl), "RGBA")
        surf_right = pygame.image.frombuffer(rgba_r, (wr, hr), "RGBA")

        screen = pygame.display.set_mode((wl + wr + gap, max(hl, hr) + info_h))
        screen.fill(COL_SHADOW)
        screen.blit(surf_left, (0, 0))
        screen.blit(surf_right, (wl + gap, 0))
        _overlay_text(screen, rows_left, geom_l, style_left, font, offset_x=0)
        _overlay_text(screen, rows_right, geom_r, style_right, font, offset_x=wl + gap)

        # Hover overlay
        if hover:
            pane, hb, off_x = hover
            rect = pygame.Rect(off_x + hb.x0, hb.y0, hb.x1 - hb.x0, hb.y1 - hb.y0)
            pygame.draw.rect(screen, (255, 235, 150), rect, 0)
            pygame.draw.rect(screen, (255, 200, 80), rect, 1)

        # Info strip
        info_rect = pygame.Rect(0, max(hl, hr), wl + wr + gap, info_h)
        pygame.draw.rect(screen, (30, 30, 40), info_rect)
        if hover:
            pane, hb, off_x = hover
            label = {
                ta.GP_TableHitPart.CELL: "cell",
                ta.GP_TableHitPart.EXPAND: "expand",
                ta.GP_TableHitPart.LED: "led",
                ta.GP_TableHitPart.AXIS: "axis",
                ta.GP_TableHitPart.CALIB_SLOT: "calib",
                ta.GP_TableHitPart.SCROLL_UP: "scroll up",
                ta.GP_TableHitPart.SCROLL_DOWN: "scroll down",
                ta.GP_TableHitPart.SCROLL_THUMB: "scroll thumb",
                ta.GP_TableHitPart.LED_TABLE: "led table",
                ta.GP_TableHitPart.LED_ARG: "led arg",
            }.get(hb.part, f"part {hb.part}")
            desc = f"{pane} hit row={hb.row_idx} col={hb.col_idx} {label} aux=({hb.aux0},{hb.aux1}) flags={hb.flags}"
        else:
            desc = "click expand boxes, LEDs, calib slots, or scrollbar; ESC to quit"
        txt = font.render(desc, True, (240, 240, 240))
        screen.blit(txt, (8, max(hl, hr) + (info_h - txt.get_height()) // 2))

        pygame.display.set_caption("c_menu interactive workbench demo")
        pygame.display.flip()

        # Input handling AFTER the flip so we process the latest hits.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    state.selected_left = max(1, state.selected_left - 1)
                elif event.key == pygame.K_DOWN:
                    state.selected_left = min(max(1, len(rows_left) - 1), state.selected_left + 1)
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                hover = None
                if 0 <= mx < wl and 0 <= my < hl:
                    hb = _find_hover(hits_l, (mx, my))
                    if hb:
                        hover = ("left", hb, 0)
                elif wl + gap <= mx < wl + gap + wr and 0 <= my < hr:
                    hb = _find_hover(hits_r, (mx - wl - gap, my))
                    if hb:
                        hover = ("right", hb, wl + gap)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if 0 <= mx < wl and 0 <= my < hl:
                    hb = _find_hover(hits_l, (mx, my))
                    if hb:
                        _handle_hit(state, meta_left, hb, my)
                elif wl + gap <= mx < wl + gap + wr and 0 <= my < hr:
                    hb = _find_hover(hits_r, (mx - wl - gap, my))
                    if hb:
                        _handle_hit(state, meta_right, hb, my)
            elif event.type == pygame.MOUSEWHEEL:
                # Wheel adjusts the synthetic scrollbar.
                state.set_scroll(state.scroll_value - event.y * 0.05)

        frame += 1
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
