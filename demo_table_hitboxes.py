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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

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
        rendered = render_once()
    except OSError as e:
        print("Failed to load native table library (c_menu). Ensure it is built and on disk.")
        print(e)
        return 1

    if rendered is None:
        print("Render failed (library missing or calc_size/raster returned false).")
        return 1

    w, h, rgba, geom, hits = rendered
    info_h = 28
    screen = pygame.display.set_mode((w, h + info_h))
    pygame.display.set_caption("Table ABI hitbox demo")
    surf = pygame.image.frombuffer(bytearray(rgba), (w, h), "RGBA").convert_alpha()

    clock = pygame.time.Clock()
    hover: Optional[HoverInfo] = None
    running = True
    last_hover_idx = -2
    font = pygame.font.SysFont(None, 18)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEMOTION:
                hover = _find_hover(hits, event.pos)

        screen.fill((10, 10, 10))
        screen.blit(surf, (0, 0))
        if hover:
            hb = hover.box
            label = PART_LABEL.get(hb.part, f"part {hb.part}")
            desc = f"hit {hover.idx}: row {hb.row_idx}, col {hb.col_idx}, {label}, aux=({hb.aux0},{hb.aux1}), flags={hb.flags}"
            last_hover_idx = hover.idx
            # Visualize ONLY the hovered region (per-LED/part hitbox), not the whole cell.
            rect = pygame.Rect(hb.x0, hb.y0, hb.x1 - hb.x0, hb.y1 - hb.y0)
            pygame.draw.rect(screen, (255, 235, 150), rect, 0)
            pygame.draw.rect(screen, (255, 200, 80), rect, 1)
            cx = (hb.x0 + hb.x1) // 2
            cy = (hb.y0 + hb.y1) // 2
            pygame.draw.circle(screen, (255, 120, 40), (cx, cy), 3, 0)
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
