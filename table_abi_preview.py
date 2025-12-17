from __future__ import annotations

import argparse
import sys
from typing import Tuple

try:
    import pygame
except Exception as exc:  # pragma: no cover
    print(f"Pygame is required: pip install pygame ({exc})", file=sys.stderr)
    sys.exit(1)

try:
    from c_menu import table_api
    import table_abi_bench as bench
except Exception as exc:  # pragma: no cover
    print(f"Failed to import table ABI helpers: {exc}", file=sys.stderr)
    sys.exit(1)


def _clamp_scroll(scroll: int, total_rows: int, visible_rows: int) -> int:
    body = max(0, total_rows - visible_rows)
    return max(0, min(body, scroll))


def _ensure_lib() -> table_api.ctypes.CDLL | None:  # type: ignore[attr-defined]
    lib = table_api._load_lib()  # type: ignore[attr-defined]
    if lib is None:
        print("c_menu DLL not found; build c_menu first", file=sys.stderr)
        return None
    return lib


def _render_surface(
    rows, cols, style
) -> Tuple[int, int, pygame.Surface, table_api.GP_TableGeom]:
    out = table_api.render_table_rgba(rows, cols, style)
    if out is None:
        raise RuntimeError("table render failed (DLL missing?)")
    w, h, rgba, geom = out
    surf = pygame.image.frombuffer(rgba, (w, h), "RGBA")
    return w, h, surf, geom


def main() -> None:
    parser = argparse.ArgumentParser(description="Live pygame preview of the C++ table ABI output")
    parser.add_argument("--rows", type=int, default=64, help="total logical rows (including children)")
    parser.add_argument("--visible", type=int, default=16, help="visible body rows (header is always shown)")
    parser.add_argument("--fps", type=int, default=60, help="target frame rate")
    parser.add_argument("--width", type=int, default=760, help="table width in pixels")
    parser.add_argument("--row-h", type=int, default=20, dest="row_h", help="row height in pixels")
    args = parser.parse_args()

    lib = _ensure_lib()
    if lib is None:
        sys.exit(1)

    wf_handle = bench._wave_handle(lib)  # waveform is optional; None is fine

    cols = [
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.LEDS, width_px=150, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.AXIS, width_px=200, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.TIMERS, width_px=160, align=0),
        table_api.GP_TableColumn(kind=table_api.GP_TableCellKind.WAVE, width_px=180, align=0),
    ]

    style = table_api.default_style(width_px=args.width)
    style.row_h_px = args.row_h
    style.name_w_px = 120

    pygame.init()
    clock = pygame.time.Clock()

    frame = 0
    scroll = 0
    running = True

    rows_all = bench.build_rows(n_rows=args.rows, frame=frame, wf_handle=wf_handle)
    rows_slice = bench.slice_rows_for_scroll(rows_all, visible_rows=args.visible, scroll=scroll)
    w, h, surf, _ = _render_surface(rows_slice, cols, style)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("table preview")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_UP, pygame.K_w):
                    scroll -= 1
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    scroll += 1
                elif event.key == pygame.K_PAGEUP:
                    scroll -= args.visible
                elif event.key == pygame.K_PAGEDOWN:
                    scroll += args.visible
                elif event.key == pygame.K_HOME:
                    scroll = 0
                elif event.key == pygame.K_END:
                    scroll = args.rows
            elif event.type == pygame.MOUSEWHEEL:
                scroll -= event.y

        scroll = _clamp_scroll(scroll, total_rows=max(0, args.rows - 1), visible_rows=args.visible)

        rows_all = bench.build_rows(n_rows=args.rows, frame=frame, wf_handle=wf_handle)
        rows_slice = bench.slice_rows_for_scroll(rows_all, visible_rows=args.visible, scroll=scroll)
        w, h, surf, geom = _render_surface(rows_slice, cols, style)

        if screen.get_width() != w or screen.get_height() != h:
            screen = pygame.display.set_mode((w, h))

        screen.blit(surf, (0, 0))
        pygame.display.set_caption(
            f"table preview | rows={args.rows} visible={args.visible} scroll={scroll} col0=({int(geom.col_x0[0])},{int(geom.col_w[0])})"
        )
        pygame.display.flip()

        frame += 1
        clock.tick(max(1, args.fps))

    pygame.quit()


if __name__ == "__main__":
    main()
