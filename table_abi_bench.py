from __future__ import annotations

import math
import os
import time
from typing import Iterable, List, Sequence

from PIL import Image

from c_menu import table_api


def _wave_handle(lib) -> int | None:
    lib.gp_menu_waveform_create.restype = table_api.ctypes.c_void_p  # type: ignore[attr-defined]
    lib.gp_menu_waveform_create.argtypes = [table_api.ctypes.c_int32, table_api.ctypes.c_int32, table_api.ctypes.c_int32]  # type: ignore[attr-defined]
    lib.gp_menu_waveform_push_sample.restype = None
    lib.gp_menu_waveform_push_sample.argtypes = [table_api.ctypes.c_void_p, table_api.ctypes.c_uint64, table_api.ctypes.c_float]
    wf = lib.gp_menu_waveform_create(96, 18, 256)
    if not wf:
        return None
    for i in range(180):
        t = i * 0.04
        v = 0.65 * math.sin(t) + 0.20 * math.sin(2.7 * t)
        lib.gp_menu_waveform_push_sample(wf, table_api.ctypes.c_uint64(i), table_api.ctypes.c_float(v))
    return int(wf)


def build_rows(n_rows: int, frame: int, wf_handle: int | None) -> List[table_api.GP_TableRow]:
    rows: List[table_api.GP_TableRow] = []

    rows.append(
        table_api.make_row(
            kind=table_api.GP_TableRowKind.HEADER,
            label="INPUTS",
            depth=0,
            expanded=True,
            selected=False,
            cells=[
                table_api.make_cell_text("LEDs"),
                table_api.make_cell_text("Axis"),
                table_api.make_cell_text("Timers"),
                table_api.make_cell_text("Wave"),
            ],
        )
    )

    for i in range(n_rows):
        depth = 0 if i % 7 == 0 else 1
        kind = table_api.GP_TableRowKind.DEVICE if depth == 0 else (table_api.GP_TableRowKind.AXIS if (i % 2 == 0) else table_api.GP_TableRowKind.BUTTON)
        phase = frame * 0.12 + i * 0.3
        axis_v = math.sin(phase) * 0.9
        flags = 0
        if (i + frame) % 3 == 0:
            flags |= 1 << 0
        if (i + frame) % 4 == 0:
            flags |= 1 << 3
        if (i + frame) % 5 == 0:
            flags |= 1 << 6
        hold = (math.sin(phase * 0.5) + 1.0) * 0.5
        last = (math.cos(phase * 0.7) + 1.0) * 0.5
        rows.append(
            table_api.make_row(
                kind=kind,
                label=f"row{i}",
                depth=depth,
                expanded=True,
                selected=(i % 10 == frame % 10),
                cells=[
                    table_api.make_cell_leds(flags=flags),
                    table_api.make_cell_axis(axis_v),
                    table_api.make_cell_timers(hold, last),
                    table_api.make_cell_wave(wf_handle),
                ],
            )
        )
    return rows


def slice_rows_for_scroll(rows: Sequence[table_api.GP_TableRow], visible_rows: int, scroll: int) -> Sequence[table_api.GP_TableRow]:
    hdr = rows[0:1]
    body = list(rows[1:])
    start = max(0, min(len(body) - visible_rows, scroll)) if body else 0
    end = start + visible_rows
    return list(hdr) + body[start:end]


def run_bench(frames: int = 120, n_rows: int = 64, visible_rows: int = 16) -> None:
    lib = table_api._load_lib()
    if lib is None:
        raise RuntimeError("c_menu DLL not found; build c_menu first")
    wf = _wave_handle(lib)

    cols = [
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.LEDS, width_px=150, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.AXIS, width_px=200, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TIMERS, width_px=160, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.WAVE, width_px=180, align=0),
    ]

    style = table_api.default_style(width_px=760)
    style.row_h_px = 20
    style.name_w_px = 120

    total_ms = 0.0
    first_frame_rgba = None
    first_geom = None
    for f in range(frames):
        rows_full = build_rows(n_rows=n_rows, frame=f, wf_handle=wf)
        rows_slice = slice_rows_for_scroll(rows_full, visible_rows=visible_rows, scroll=f % max(1, (n_rows - visible_rows)))

        t0 = time.perf_counter()
        out = table_api.render_table_rgba(rows_slice, cols, style)
        t1 = time.perf_counter()
        if out is None:
            raise RuntimeError("table render failed")
        w, h, rgba, geom = out
        total_ms += (t1 - t0) * 1000.0
        if f == 0:
            first_frame_rgba = (w, h, rgba)
            first_geom = geom

    avg_ms = total_ms / float(frames)
    print(f"frames: {frames}, rows: {n_rows}, visible_rows: {visible_rows}, avg render: {avg_ms:0.3f} ms")

    if first_frame_rgba is not None:
        w, h, rgba = first_frame_rgba
        img = Image.frombytes("RGBA", (w, h), rgba)
        out_path = os.path.join(os.getcwd(), "table_bench_frame0.png")
        img.save(out_path)
        print(f"saved first frame: {out_path} ({w}x{h})")
        if first_geom is not None:
            cols_info = [(int(first_geom.col_x0[i]), int(first_geom.col_w[i])) for i in range(4)]
            print("columns:", cols_info)


if __name__ == "__main__":
    run_bench()
