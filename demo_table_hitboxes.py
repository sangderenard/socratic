from __future__ import annotations

"""
Standalone pygame demo for table_abi hitboxes.

- Renders a sample table via the C raster (c_menu DLL/SO).
- Shows interactive hover overlay using emitted hitboxes.
- Keeps dependencies minimal and stays modular (only depends on table_api + pygame).

Run:
    python demo_table_hitboxes.py

Ensure pygame is installed and the c_menu native library is buildable in c_menu/.
"""

import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame
import ctypes

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "c_menu"))
import table_api as ta


@dataclass
class HoverInfo:
    idx: int
    box: ta.GP_TableHitBox


PART_LABEL = {
    ta.GP_TableHitPart.CELL: "cell",
    ta.GP_TableHitPart.EXPAND: "expand",
    ta.GP_TableHitPart.LED: "led",
    ta.GP_TableHitPart.AXIS: "axis",
    ta.GP_TableHitPart.CALIB_SLOT: "calib slot",
    ta.GP_TableHitPart.SCROLL_UP: "scroll up",
    ta.GP_TableHitPart.SCROLL_DOWN: "scroll down",
    ta.GP_TableHitPart.SCROLL_THUMB: "scroll thumb",
    ta.GP_TableHitPart.LED_TABLE: "led table",
    ta.GP_TableHitPart.LED_ARG: "led arg",
}


def _color_for(idx: int) -> Tuple[int, int, int]:
    return ((idx * 73) % 200 + 40, (idx * 41) % 200 + 40, (idx * 19) % 200 + 40)


def _make_columns() -> List[ta.GP_TableColumn]:
    def col(kind: int, w: int, align: int = 0) -> ta.GP_TableColumn:
        c = ta.GP_TableColumn()
        c.kind = kind
        c.width_px = w
        c.align = align
        return c

    return [
        col(ta.GP_TableCellKind.LEDS, 140),
        col(ta.GP_TableCellKind.AXIS, 240),
        col(ta.GP_TableCellKind.CALIB, 160),
        col(ta.GP_TableCellKind.LEDS_ARG, 200),
        col(ta.GP_TableCellKind.LEDS_TABLE, 200),
        col(ta.GP_TableCellKind.SCROLL, 40),
    ]


def _make_rows() -> List[ta.GP_TableRow]:
    rows: List[ta.GP_TableRow] = []

    # Header/device rows to demonstrate expand hitbox.
    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.HEADER,
            label="Controller A",
            depth=0,
            expanded=True,
            selected=False,
            cells=[ta.make_cell_text("header gutter")],
        )
    )

    # LEDs + axis + calib + arg LEDs + stacked LEDs + scrollbar.
    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.DEVICE,
            label="Axes/Inputs",
            depth=0,
            expanded=True,
            selected=False,
            cells=[
                ta.make_cell_leds(flags=0b101011001),
                ta.make_cell_axis_calib(
                    value=0.25,
                    v_min=-0.8,
                    v_max=0.9,
                    cap_min=-1.0,
                    cap_max=1.0,
                    trim=0.05,
                    deadzone=0.08,
                    seen_min=True,
                    seen_max=False,
                ),
                ta.make_cell_calib(invert_on=True, mode="trim"),
                # 12 arg LEDs with mixed required/linked bits to show per-led hits.
                ta.make_cell_leds_arg(count=12, linked_mask=0b100001010101, required_mask=0b111100001111),
                ta.make_cell_leds_table(
                    [
                        {"count": 6, "on": 0b101101, "edge": 0b001001, "active": 0b111111},
                        {"count": 12, "on": 0b111100001111, "edge": 0, "active": 0b010101010101},
                    ]
                ),
                ta.make_cell_scroll(value01=0.3, total_rows=32, visible_rows=8),
            ],
        )
    )

    # Another LED row to show per-led hits.
    rows.append(
        ta.make_row(
            kind=ta.GP_TableRowKind.NOTE,
            label="LEDs and Scroll",
            depth=1,
            expanded=True,
            selected=False,
            cells=[
                ta.make_cell_leds(flags=0b111111111),
                ta.make_cell_axis(value=-0.4),
                ta.make_cell_calib(invert_on=False, mode="cap"),
                ta.make_cell_leds_arg(count=6, linked_mask=0b001011),
                ta.make_cell_leds_table([{"count": 4, "on": 0b0101, "edge": 0b1000, "active": 0b1111}]),
                ta.make_cell_scroll(value01=0.65, total_rows=20, visible_rows=5, arrow_dn_pressed=True),
            ],
        )
    )

    return rows


def render_once() -> Optional[Tuple[int, int, bytes, ta.GP_TableGeom, Sequence[ta.GP_TableHitBox]]]:
    cols = _make_columns()
    rows = _make_rows()
    style = ta.default_style(width_px=1000)
    result = ta.render_table_rgba_with_hits(rows, cols, style=style, hitbox_cap=8192)
    if result is None:
        return None
    w, h, rgba, geom, hits = result
    return w, h, rgba, geom, hits


def _find_hover(hits: Sequence[ta.GP_TableHitBox], pos: Tuple[int, int]) -> Optional[HoverInfo]:
    mx, my = pos
    for idx, hb in enumerate(hits):
        if hb.x0 <= mx < hb.x1 and hb.y0 <= my < hb.y1:
            return HoverInfo(idx=idx, box=hb)
    return None


def main() -> int:
    pygame.init()
    try:
        # Create a stateful TableContext and populate it
        style = ta.default_style(width_px=1000)
        ctx = ta.TableContext(style=style)
        cols = _make_columns()
        rows = _make_rows()
        ctx.set_columns(cols)
        ctx.set_rows(rows)
        # enable prospective live-edge mode for immersive cord jacking
        try:
            ctx.set_prospective_mode(True)
        except Exception:
            pass
        # parse CLI args for prospective params
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--prospective-max-history", type=int, default=8)
        parser.add_argument("--prospective-slack", type=float, default=4.0)
        parser.add_argument("--prospective-stiffness", type=float, default=10.0)
        parser.add_argument("--prospective-damping", type=float, default=2.0)
        args, _ = parser.parse_known_args()
        try:
            ctx.set_prospective_params(args.prospective_max_history, args.prospective_slack, 0.0)
            ctx.set_relax_params(stiffness=args.prospective_stiffness, damping=args.prospective_damping)
        except Exception:
            pass
    except OSError as e:
        print("Failed to load native table library (c_menu). Ensure it is built and on disk.")
        print(e)
        return 1

    g = ctx.geom()
    w = int(g.width_px)
    h = int(g.height_px)
    info_h = 28
    screen = pygame.display.set_mode((w, h + info_h))
    pygame.display.set_caption("Table ABI hitbox demo")
    surf = None
    hits = []

    clock = pygame.time.Clock()
    hover: Optional[HoverInfo] = None
    running = True
    last_hover_idx = -2
    font = pygame.font.SysFont(None, 18)
    mouse_pos = (-1, -1)
    while running:
        need_rerender = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                need_rerender = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # forward click into C; it will update context state (expand/led toggle/scroll)
                hit = ctx.on_click(mx, my)
                # re-render to reflect any state change
                need_rerender = True

        # If prospective mode is enabled and exactly one node is selected, report
        # mouse position every frame so the C renderer can relax the free end.
        try:
            pros = ctx.get_prospective_mode()
            sel_count = ctx.get_selected_count()
        except Exception:
            pros = False
            sel_count = 0

        if pros and sel_count == 1:
            need_rerender = True

        if need_rerender:
            # build transient render state with mouse coords so C++ can highlight under cursor
            rs = ta.GP_TableRenderState()
            rs.mouse_x = int(mouse_pos[0]) if mouse_pos[0] is not None else -1
            rs.mouse_y = int(mouse_pos[1]) if mouse_pos[1] is not None else -1
            rs.highlight_row = -1
            rs.highlight_col = -1
            rs.highlight_part = -1
            rs.highlight_aux0 = -1
            # zero color => C++ default
            rs.highlight_color = (ctypes.c_uint8 * 4)(0, 0, 0, 0)
            w, h, rgba, g, hits = ctx.render_with_state(rs, hitbox_cap=8192)
            surf = pygame.image.frombuffer(bytearray(rgba), (w, h), "RGBA").convert_alpha()

        screen.fill((10, 10, 10))
        if surf:
            screen.blit(surf, (0, 0))

        # compute hover info from latest hits
        hover = _find_hover(hits, mouse_pos) if mouse_pos[0] >= 0 else None
        if hover:
            hb = hover.box
            label = PART_LABEL.get(hb.part, f"part {hb.part}")
            desc = f"hit {hover.idx}: row {hb.row_idx}, col {hb.col_idx}, {label}, aux=({hb.aux0},{hb.aux1}), flags={hb.flags}"
            last_hover_idx = hover.idx
        else:
            desc = "hover to see hit details"
            last_hover_idx = -1

        info_rect = pygame.Rect(0, h, w, info_h)
        pygame.draw.rect(screen, (30, 30, 40), info_rect)
        txt = font.render(desc, True, (240, 240, 240))
        screen.blit(txt, (8, h + (info_h - txt.get_height()) // 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
