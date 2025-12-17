from __future__ import annotations

import math
import os
import sys
from typing import Optional

try:
    from PIL import Image
except Exception:
    print("Pillow is required: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    from c_menu import table_api
except Exception as exc:  # pragma: no cover
    print(f"Failed to import c_menu.table_api: {exc}", file=sys.stderr)
    sys.exit(1)


def _load_waveform_handle() -> Optional[int]:
    """Create a waveform handle via the c_menu DLL if available."""
    lib = table_api._load_lib()  # type: ignore[attr-defined]
    if lib is None:
        return None

    # Declare the needed C signatures lazily.
    lib.gp_menu_waveform_create.restype = table_api.ctypes.c_void_p  # type: ignore[attr-defined]
    lib.gp_menu_waveform_create.argtypes = [table_api.ctypes.c_int32, table_api.ctypes.c_int32, table_api.ctypes.c_int32]  # type: ignore[attr-defined]
    lib.gp_menu_waveform_push_sample.restype = None
    lib.gp_menu_waveform_push_sample.argtypes = [table_api.ctypes.c_void_p, table_api.ctypes.c_uint64, table_api.ctypes.c_float]

    wf = lib.gp_menu_waveform_create(80, 18, 256)
    if not wf:
        return None

    # Populate with a sine-ish waveform.
    for i in range(160):
        t = i * 0.05
        v = 0.65 * math.sin(t) + 0.25 * math.sin(3.0 * t)
        lib.gp_menu_waveform_push_sample(wf, table_api.ctypes.c_uint64(i), table_api.ctypes.c_float(v))
    return int(wf)


def build_sample_table():
    # Columns: LEDs, axis, timers, wave
    cols = [
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.LEDS, width_px=150, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.AXIS, width_px=200, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TIMERS, width_px=160, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.WAVE, width_px=180, align=0),
    ]

    wf_handle = _load_waveform_handle()

    rows = []
    # Header
    rows.append(table_api.make_row(kind=table_api.GP_TableRowKind.HEADER, label="INPUTS", depth=0, expanded=True, selected=False, cells=[
        table_api.make_cell_text("LEDs"),
        table_api.make_cell_text("Axis"),
        table_api.make_cell_text("Timers"),
        table_api.make_cell_text("Wave"),
    ]))

    # Device row
    rows.append(table_api.make_row(kind=table_api.GP_TableRowKind.DEVICE, label="JOYSTICK", depth=0, expanded=True, selected=False, cells=[
        table_api.make_cell_leds(flags=(1 << 0) | (1 << 1)),
        table_api.make_cell_axis(0.25),
        table_api.make_cell_timers(0.6, 0.3),
        table_api.make_cell_wave(wf_handle),
    ]))

    # Child axis row
    rows.append(table_api.make_row(kind=table_api.GP_TableRowKind.AXIS, label="axis.0", depth=1, expanded=True, selected=True, cells=[
        table_api.make_cell_leds(flags=(1 << 0)),
        table_api.make_cell_axis(-0.65),
        table_api.make_cell_timers(0.2, 0.8),
        table_api.make_cell_wave(wf_handle),
    ]))

    # Child button row
    rows.append(table_api.make_row(kind=table_api.GP_TableRowKind.BUTTON, label="btn.0", depth=1, expanded=True, selected=False, cells=[
        table_api.make_cell_leds(flags=(1 << 0) | (1 << 3) | (1 << 6)),
        table_api.make_cell_axis(0.0),
        table_api.make_cell_timers(1.0, 0.1),
        table_api.make_cell_wave(wf_handle),
    ]))

    style = table_api.default_style(width_px=760)
    style.name_w_px = 120

    out = table_api.render_table_rgba(rows, cols, style)
    if out is None:
        raise RuntimeError("table rendering failed (DLL missing or call failed)")
    return out


def main() -> None:
    w, h, rgba, geom = build_sample_table()
    img = Image.frombytes("RGBA", (w, h), rgba)
    out_path = os.path.join(os.getcwd(), "table_demo.png")
    img.save(out_path)
    print(f"wrote {out_path} ({w}x{h})")
    # show some geometry for debugging
    print("columns:", [(int(geom.col_x0[i]), int(geom.col_w[i])) for i in range(4)])


if __name__ == "__main__":
    main()
