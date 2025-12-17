from __future__ import annotations

import math
import sys
from typing import Dict, List, Sequence, Tuple

try:
    import pygame
except Exception as exc:  # pragma: no cover
    print(f"Pygame is required: pip install pygame ({exc})", file=sys.stderr)
    sys.exit(1)

try:
    from c_menu import table_api
except Exception as exc:  # pragma: no cover
    print(f"Failed to import table ABI helpers: {exc}", file=sys.stderr)
    sys.exit(1)

# Colors (match legacy workbench semantics)
COL_DIM = (160, 160, 160)
COL_RED = (255, 80, 80)
COL_GREEN = (120, 255, 120)
COL_TEXT = (235, 235, 240)
COL_BLUE = (80, 180, 255)


class RowOverlay:
    def __init__(self, label: str, depth: int, calibr_mode: str = "", phase: float = 0.0, invert_on: bool = False) -> None:
        self.label = label
        self.depth = depth
        self.calibr_mode = calibr_mode  # ""|"trim"|"cap"|"ded"
        self.phase = phase  # 0..1 for red->green animation
        self.invert_on = invert_on


class SparkStore:
    """Manages per-row sparkline waveforms and sample cursors."""

    def __init__(self, lib, w: int = 120, h: int = 18, cap: int = 256) -> None:
        self._lib = lib
        self._w = int(w)
        self._h = int(h)
        self._cap = int(cap)
        self._handles: Dict[str, int] = {}
        self._t: Dict[str, int] = {}

        lib.gp_menu_waveform_create.restype = table_api.ctypes.c_void_p  # type: ignore[attr-defined]
        lib.gp_menu_waveform_create.argtypes = [table_api.ctypes.c_int32, table_api.ctypes.c_int32, table_api.ctypes.c_int32]
        lib.gp_menu_waveform_push_sample.restype = None
        lib.gp_menu_waveform_push_sample.argtypes = [table_api.ctypes.c_void_p, table_api.ctypes.c_uint64, table_api.ctypes.c_float]

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
        self._lib.gp_menu_waveform_push_sample(table_api.ctypes.c_void_p(h), table_api.ctypes.c_uint64(t), table_api.ctypes.c_float(v))
        self._t[str(key)] = t + 1
        return h


def _pcm_sample(shape: str, t: int) -> float:
    # Deterministic small waveforms to mimic PCM streams.
    # shape: "sine" | "saw" | "square" | "tri"
    tt = float(t)
    if shape == "saw":
        return ((tt % 64) / 64.0) * 2.0 - 1.0
    if shape == "square":
        return 1.0 if (int(tt) // 16) % 2 == 0 else -1.0
    if shape == "tri":
        phase = (tt % 64) / 64.0
        return 4.0 * abs(phase - 0.5) - 1.0
    return math.sin(tt * 0.12)


def _style_for_workbench(width: int, row_h: int) -> table_api.GP_TableStyle:
    s = table_api.default_style(width_px=width)
    s.row_h_px = int(row_h)
    s.name_w_px = 260  # wider left gutter for nested labels
    # LED palette to match legacy (level green, edge white)
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
    setc("wave_fg_rgba", (90, 180, 255, 255))  # blue sparkline
    setc("bg_rgba", (6, 6, 8, 255))
    setc("bg_sel_rgba", (18, 18, 26, 255))
    setc("hdr_rgba", (22, 22, 32, 255))
    setc("text_rgba", (240, 240, 245, 255))
    setc("text_hdr_rgba", (255, 255, 255, 255))
    return s


def _calib_color(mode: str, target: str, t_phase: float) -> Tuple[int, int, int]:
    if mode != target:
        return COL_DIM
    # red for first half of window, green for second
    return COL_RED if t_phase < 0.5 else COL_GREEN


def _hat_flags(dir_idx: int | None) -> int:
    if dir_idx is None:
        return 1 << 8
    return 1 << int(max(0, min(7, dir_idx)))


def _build_rows(frame: int, sparks: SparkStore) -> Tuple[List[table_api.GP_TableRow], List[RowOverlay]]:
    rows: List[table_api.GP_TableRow] = []
    overlay: List[RowOverlay] = []

    # LED bit patterns to showcase all status flags
    # Order is D + - H 2 2H T t 2t
    def flags_for_phase(ph: int) -> int:
        bits = 0
        if ph % 2 == 0:
            bits |= 1 << 0  # D
        if ph % 3 == 0:
            bits |= 1 << 1  # +
        if ph % 4 == 0:
            bits |= 1 << 2  # -
        if ph % 5 == 0:
            bits |= 1 << 3  # H
        if ph % 6 == 0:
            bits |= 1 << 4  # 2
        if ph % 7 == 0:
            bits |= 1 << 5  # 2H
        if ph % 8 < 4:
            bits |= 1 << 6  # T
        if ph % 5 == 2:
            bits |= 1 << 7  # t
        if ph % 9 == 1:
            bits |= 1 << 8  # 2t
        return bits

    # Device headers (expanded and collapsed) with +/- handled by row.expanded
    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.HEADER,
            label="DEVICE (open)",
            depth=0,
            expanded=True,
            selected=False,
            cells=[
                table_api.make_cell_text("LEDs"),
                table_api.make_cell_text("Axis"),
                table_api.make_cell_text("Wave"),
            ],
        )
    )
    overlay.append(RowOverlay(label="DEVICE (open)", depth=0))

    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.HEADER,
            label="DEVICE (closed)",
            depth=0,
            expanded=False,
            selected=False,
            cells=[
                table_api.make_cell_text("LEDs"),
                table_api.make_cell_text("Axis"),
                table_api.make_cell_text("Wave"),
            ],
        )
    )
    overlay.append(RowOverlay(label="DEVICE (closed)", depth=0))

    # Button row cycling through all LED states (permutation-ish); no wave column to avoid overlaying controls.
    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.BUTTON,
            label="btn.flags",
            depth=1,
            expanded=True,
            selected=False,
            cells=[
                table_api.make_cell_leds(flags=flags_for_phase(frame)),
                table_api.make_cell_axis(0.0),
            ],
        )
    )
    overlay.append(RowOverlay(label="btn.flags", depth=1))

    # Axis with calibration ticks, initially missing min/max (shows red ends)
    phase = (frame % 180) / 180.0
    val = math.sin(frame * 0.05) * 0.8
    seen_min = frame > 30
    seen_max = frame > 60
    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.AXIS,
            label="axis.calib",
            depth=1,
            expanded=True,
            selected=False,
            cells=[
                table_api.make_cell_leds(flags=0),
                table_api.make_cell_axis_calib(
                    value=val,
                    v_min=-0.6,
                    v_max=0.7,
                    cap_min=-0.85,
                    cap_max=0.95,
                    trim=0.18,
                    deadzone=0.08,
                    seen_min=seen_min,
                    seen_max=seen_max,
                ),
                table_api.make_cell_calib(invert_on=(frame % 40) < 20, mode="trim"),
                table_api.make_cell_wave(sparks.push("axis.calib", _pcm_sample("sine", frame))),
            ],
        )
    )
    overlay.append(RowOverlay(label="axis.calib", depth=1, calibr_mode="trim", phase=phase, invert_on=False))

    # Axis with centered trim + deadzone and full min/max coverage
    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.AXIS,
            label="axis.deadzone",
            depth=1,
            expanded=True,
            selected=False,
            cells=[
                table_api.make_cell_leds(flags=0),
                table_api.make_cell_axis_calib(
                    value=math.sin(frame * 0.08) * 0.6,
                    v_min=-1.0,
                    v_max=1.0,
                    cap_min=-1.0,
                    cap_max=1.0,
                    trim=0.0,
                    deadzone=0.12,
                    seen_min=True,
                    seen_max=True,
                ),
                table_api.make_cell_calib(invert_on=False, mode="cap"),
                table_api.make_cell_wave(sparks.push("axis.deadzone", _pcm_sample("saw", frame))),
            ],
        )
    )
    overlay.append(RowOverlay(label="axis.deadzone", depth=1, calibr_mode="cap", phase=phase, invert_on=False))

    # Hat rows: two hats showing direction LEDs and sparkline of x/y magnitude
    for hat_idx in range(2):
        dir_cycle = (frame // 12 + hat_idx * 2) % 8
        flags = _hat_flags(dir_cycle)
        x = math.cos(dir_cycle * (math.pi / 4.0))
        y = math.sin(dir_cycle * (math.pi / 4.0))
        rows.append(
            table_api.make_row(
                kind=table_api.GP_TableRowKind.DEVICE,
                label=f"hat.{hat_idx}",
                depth=1,
                expanded=True,
                selected=False,
                cells=[
                    table_api.make_cell_leds(flags=flags),
                    table_api.make_cell_axis_calib(
                        value=x,
                        v_min=-1.0,
                        v_max=1.0,
                        cap_min=-1.0,
                        cap_max=1.0,
                        trim=0.0,
                        deadzone=0.0,
                        seen_min=True,
                        seen_max=True,
                    ),
                    table_api.make_cell_wave(sparks.push(f"hat.{hat_idx}", _pcm_sample("square", frame + hat_idx * 8))),
                ],
            )
        )
        overlay.append(RowOverlay(label=f"hat.{hat_idx}", depth=1, calibr_mode="ded", phase=phase, invert_on=False))

    # Button palette: pressed / highlighted permutations
    for i in range(4):
        pressed = (frame + i) % 30 < 12
        highlighted = (frame // 20 + i) % 2 == 0
        rows.append(
            table_api.make_row(
                kind=table_api.GP_TableRowKind.BUTTON,
                label=f"btn.{i}",
                depth=1,
                expanded=True,
                selected=highlighted,
                cells=[
                    table_api.make_cell_leds(flags=(1 << 0) if pressed else 0),
                    table_api.make_cell_axis(0.0),
                ],
            )
        )
        overlay.append(RowOverlay(label=f"btn.{i} ({'P' if pressed else 'U'})", depth=1, calibr_mode="", phase=0.0, invert_on=False))

    return rows, overlay


def _build_signal_rows(frame: int, sparks: SparkStore) -> List[table_api.GP_TableRow]:
    rows: List[table_api.GP_TableRow] = []

    sig_defs = [
        {"label": "sig.throttle", "req": 2, "linked": 2, "expand": False, "op": "2+", "ch": 0, "dim": 1},
        {"label": "sig.pitch", "req": 1, "linked": 0, "expand": False, "op": "K", "ch": 1, "dim": 1},
        {"label": "sig.view2d", "req": 2, "linked": 1, "expand": True, "op": "2S", "ch": -1, "dim": 2},
        {"label": "sig.raw_kernel", "req": 1, "linked": 1, "expand": False, "op": "K", "ch": 2, "dim": 1},
        {"label": "NEW", "req": 0, "linked": 0, "expand": False, "op": "?", "ch": -1, "dim": 1},
    ]

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

        cells: List[table_api.GP_TableCell] = []
        cells.append(table_api.make_cell_text(label))
        c_args = table_api.make_cell_text(f"args {linked}/{req}")
        c_args.flags = flags
        cells.append(c_args)
        cells.append(table_api.make_cell_wave(wf))
        cells.append(table_api.make_cell_text(op_lbl))
        cells.append(table_api.make_cell_text("ch:-" if ch < 0 else f"ch:{ch}"))

        rows.append(
            table_api.make_row(
                kind=table_api.GP_TableRowKind.NOTE,
                label=label,
                depth=0,
                expanded=False,
                selected=(i == (frame // 45) % len(sig_defs)),
                cells=cells,
            )
        )

    return rows


def _overlay_text(screen: pygame.Surface, rows: Sequence[table_api.GP_TableRow], overlay: Sequence[RowOverlay], geom: table_api.GP_TableGeom, style: table_api.GP_TableStyle, font: pygame.font.Font) -> None:
    row_h = int(style.row_h_px)
    expand_w = int(style.expand_w_px)
    indent = int(style.indent_px)

    for i, meta in enumerate(overlay):
        y = i * row_h + row_h // 2
        x_label = 6 + expand_w + indent * int(meta.depth)
        txt = font.render(meta.label, True, COL_TEXT)
        screen.blit(txt, (x_label, y - txt.get_height() // 2))




def main() -> None:
    pygame.init()
    font = pygame.font.SysFont("Consolas", 14)

    width_left = 900
    width_right = 520
    row_h = 22
    cols_left = [
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.LEDS, width_px=150, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.AXIS, width_px=260, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.CALIB, width_px=140, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.WAVE, width_px=200, align=0),
    ]
    cols_right = [
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TEXT, width_px=200, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TEXT, width_px=140, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.WAVE, width_px=140, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TEXT, width_px=60, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TEXT, width_px=70, align=0),
    ]

    style_left = _style_for_workbench(width=width_left, row_h=row_h)
    style_right = _style_for_workbench(width=width_right, row_h=row_h)
    style_right.name_w_px = 10
    style_right.indent_px = 0
    style_right.expand_w_px = 0
    lib = table_api._load_lib()
    if lib is None:
        print("c_menu DLL not found; build c_menu first", file=sys.stderr)
        sys.exit(1)
    sparks = SparkStore(lib)

    clock = pygame.time.Clock()
    frame = 0

    rows_left, overlay = _build_rows(frame, sparks)
    out_left = table_api.render_table_rgba(rows_left, cols_left, style_left)
    rows_right = _build_signal_rows(frame, sparks)
    out_right = table_api.render_table_rgba(rows_right, cols_right, style_right)
    if out_left is None or out_right is None:
        raise RuntimeError("table render failed")
    w = out_left[0] + out_right[0]
    h = max(out_left[1], out_right[1])
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("device + signals table animation")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        rows_left, overlay = _build_rows(frame, sparks)
        rows_right = _build_signal_rows(frame, sparks)
        out_left = table_api.render_table_rgba(rows_left, cols_left, style_left)
        out_right = table_api.render_table_rgba(rows_right, cols_right, style_right)
        if out_left is None or out_right is None:
            break
        w = out_left[0] + out_right[0]
        h = max(out_left[1], out_right[1])
        surf_left = pygame.image.frombuffer(out_left[2], (out_left[0], out_left[1]), "RGBA")
        surf_right = pygame.image.frombuffer(out_right[2], (out_right[0], out_right[1]), "RGBA")
        if screen.get_size() != (w, h):
            screen = pygame.display.set_mode((w, h))
        screen.blit(surf_left, (0, 0))
        screen.blit(surf_right, (out_left[0], 0))
        _overlay_text(screen, rows_left, overlay, out_left[3], style_left, font)
        pygame.display.set_caption(f"device + signals table animation | frame {frame}")
        pygame.display.flip()

        frame += 1
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
