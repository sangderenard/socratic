from __future__ import annotations

"""
Interactive pygame demo for the c_menu table ABI using the signal workbench sample rows.

- Renders the C++ raster output for two tables (device view + signals view).
- Uses the native hitboxes to drive selection and expand/collapse state.
- Overlays text labels so you can see the workbench-style content without wiring in the
  full menu system.

Run:
    python table_signal_workbench_interactive.py

Requires:
- pygame
- a built c_menu native library (see c_menu/CMakeLists.txt)
"""

from dataclasses import dataclass
import sys
from typing import Iterable, List, Sequence, Tuple

import pygame

from c_menu import table_api as ta
import table_signal_workbench_anim as anim


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


def _row_label(row: ta.GP_TableRow) -> str:
    raw = bytes(row.label)
    return raw.split(b"\0", 1)[0].decode("ascii", "ignore")


def _cell_text(cell: ta.GP_TableCell) -> str:
    if int(cell.kind) != ta.GP_TableCellKind.TEXT:
        return ""
    raw = bytes(cell.text)
    return raw.split(b"\0", 1)[0].decode("ascii", "ignore")


def _color(rgb_or_rgba: Sequence[int]) -> Tuple[int, int, int]:
    return (int(rgb_or_rgba[0]), int(rgb_or_rgba[1]), int(rgb_or_rgba[2]))


@dataclass
class HitCtx:
    table: str  # "left" or "right"
    hit: ta.GP_TableHitBox
    offset: Tuple[int, int]
    rows: Sequence[ta.GP_TableRow]

    def contains(self, pos: Tuple[int, int]) -> bool:
        dx, dy = self.offset
        hb = self.hit
        x, y = pos
        return (dx + hb.x0) <= x < (dx + hb.x1) and (dy + hb.y0) <= y < (dy + hb.y1)

    def label(self) -> str:
        try:
            return _row_label(self.rows[int(self.hit.row_idx)])
        except Exception:
            return ""


class WorkbenchInteractiveDemo:
    def __init__(self, *, font: pygame.font.Font) -> None:
        lib = ta._load_lib()
        if lib is None:
            raise RuntimeError("c_menu native library not found; build c_menu first")

        self.font = font
        self.lib = lib
        self.frame = 0
        self.selected_left = "axis.calib"
        self.selected_right = "sig.throttle"
        self.collapsed_left: set[str] = set()
        self.collapsed_right: set[str] = set()
        self.last_click_desc = "click any cell to select; click +/- gutter to expand or collapse"
        self.hover: HitCtx | None = None
        self.global_hits: List[HitCtx] = []

        # Geometry mirrors table_signal_workbench_anim.py
        self.width_left = 900
        self.width_right = 520
        self.row_h = 22

        self.cols_left = [
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.LEDS, width_px=150, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.AXIS, width_px=260, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.CALIB, width_px=140, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.WAVE, width_px=200, align=0),
        ]
        self.cols_right = [
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.TEXT, width_px=200, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.TEXT, width_px=140, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.WAVE, width_px=140, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.TEXT, width_px=60, align=0),
            ta.GP_TableColumn(kind=ta.GP_TableCellKind.TEXT, width_px=70, align=0),
        ]

        self.style_left = anim._style_for_workbench(width=self.width_left, row_h=self.row_h)
        self.style_right = anim._style_for_workbench(width=self.width_right, row_h=self.row_h)
        self.style_right.name_w_px = 10
        self.style_right.indent_px = 0
        self.style_right.expand_w_px = 0

        self.ctx_left = ta.TableContext(self.style_left)
        self.ctx_right = ta.TableContext(self.style_right)
        self.ctx_left.set_columns(self.cols_left)
        self.ctx_right.set_columns(self.cols_right)

        self.sparks = anim.SparkStore(lib)
        self.rows_left: Sequence[ta.GP_TableRow] = ()
        self.rows_right: Sequence[ta.GP_TableRow] = ()
        self.overlay_left: Sequence[anim.RowOverlay] = ()
        self.geom_left = ta.GP_TableGeom()
        self.geom_right = ta.GP_TableGeom()
        self.surface_left: pygame.Surface | None = None
        self.surface_right: pygame.Surface | None = None

    def _apply_state(self, rows: Sequence[ta.GP_TableRow], *, selected_label: str, collapsed: set[str]) -> None:
        for r in rows:
            lbl = _row_label(r)
            r.selected = 1 if lbl == selected_label else 0
            if lbl in collapsed:
                r.expanded = 0

    def _build_rows(self) -> None:
        rows_left, overlay_left = anim._build_rows(self.frame, self.sparks)
        rows_right = anim._build_signal_rows(self.frame, self.sparks)
        self._apply_state(rows_left, selected_label=self.selected_left, collapsed=self.collapsed_left)
        self._apply_state(rows_right, selected_label=self.selected_right, collapsed=self.collapsed_right)
        self.rows_left = rows_left
        self.rows_right = rows_right
        self.overlay_left = overlay_left

    def _render_table(
        self, ctx: ta.TableContext, rows: Sequence[ta.GP_TableRow], cols: Sequence[ta.GP_TableColumn], style: ta.GP_TableStyle
    ) -> Tuple[pygame.Surface, ta.GP_TableGeom, Sequence[ta.GP_TableHitBox]]:
        ctx.set_rows(rows)
        w, h, rgba, geom, hits = ctx.render()
        surf = pygame.image.frombuffer(bytearray(rgba), (w, h), "RGBA")
        # Copy to avoid referencing the raw buffer after the next render.
        return surf.copy(), geom, hits

    def render(self) -> Tuple[int, int]:
        self._build_rows()
        self.surface_left, self.geom_left, hits_left = self._render_table(self.ctx_left, self.rows_left, self.cols_left, self.style_left)
        self.surface_right, self.geom_right, hits_right = self._render_table(self.ctx_right, self.rows_right, self.cols_right, self.style_right)

        self.global_hits = []
        for hb in hits_left:
            self.global_hits.append(HitCtx(table="left", hit=hb, offset=(0, 0), rows=self.rows_left))
        for hb in hits_right:
            self.global_hits.append(HitCtx(table="right", hit=hb, offset=(self.surface_left.get_width(), 0), rows=self.rows_right))
        return self.surface_left.get_width() + self.surface_right.get_width(), max(self.surface_left.get_height(), self.surface_right.get_height())

    def step(self) -> None:
        self.frame += 1

    def hit_at(self, pos: Tuple[int, int]) -> HitCtx | None:
        for hc in self.global_hits:
            if hc.contains(pos):
                return hc
        return None

    def handle_click(self, pos: Tuple[int, int]) -> None:
        hc = self.hit_at(pos)
        if hc is None:
            self.last_click_desc = "no hit (try the cells or +/- gutter)"
            return

        hb = hc.hit
        label = hc.label()
        part_name = PART_LABEL.get(int(hb.part), f"part {hb.part}")
        desc = f"{hc.table} row '{label}' col {hb.col_idx} ({part_name})"

        if int(hb.part) == ta.GP_TableHitPart.EXPAND and label:
            collapsed = self.collapsed_left if hc.table == "left" else self.collapsed_right
            if label in collapsed:
                collapsed.remove(label)
                desc += " -> expanded"
            else:
                collapsed.add(label)
                desc += " -> collapsed"
        else:
            if hc.table == "left":
                self.selected_left = label
            else:
                self.selected_right = label
            desc += " -> selected"

        self.last_click_desc = desc

    def draw_text_overlay(
        self, screen: pygame.Surface, *, offset: Tuple[int, int], geom: ta.GP_TableGeom, rows: Sequence[ta.GP_TableRow], style: ta.GP_TableStyle
    ) -> None:
        ox, oy = offset
        row_h = int(style.row_h_px)
        expand_w = int(style.expand_w_px)
        indent = int(style.indent_px)
        col_hdr = _color(style.text_hdr_rgba)
        col_text = _color(style.text_rgba)
        for idx, row in enumerate(rows):
            y_mid = oy + idx * row_h + row_h // 2
            lbl = _row_label(row)
            color = col_hdr if int(row.kind) == ta.GP_TableRowKind.HEADER else col_text
            label_surf = self.font.render(lbl, True, color)
            x_label = ox + 6 + expand_w + indent * int(row.depth)
            screen.blit(label_surf, (x_label, y_mid - label_surf.get_height() // 2))

            for ci in range(int(row.cell_count)):
                txt = _cell_text(row.cells[ci])
                if not txt:
                    continue
                x0 = ox + int(geom.col_x0[ci])
                w = int(geom.col_w[ci])
                cell_surf = self.font.render(txt, True, color)
                screen.blit(cell_surf, (x0 + 6, y_mid - cell_surf.get_height() // 2))


def main() -> int:
    pygame.init()
    font = pygame.font.SysFont("Consolas", 14)

    try:
        demo = WorkbenchInteractiveDemo(font=font)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    info_h = 32
    total_w, total_h = demo.render()
    screen = pygame.display.set_mode((total_w, total_h + info_h))
    pygame.display.set_caption("c_menu signal workbench interactive demo")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEMOTION:
                demo.hover = demo.hit_at(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                demo.handle_click(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                # Optional: feel free to add scroll coupling here; for now it just updates hover.
                demo.hover = demo.hit_at(pygame.mouse.get_pos())

        total_w, total_h = demo.render()
        if screen.get_width() != total_w or screen.get_height() != total_h + info_h:
            screen = pygame.display.set_mode((total_w, total_h + info_h))

        screen.fill((6, 6, 8))
        screen.blit(demo.surface_left, (0, 0))
        screen.blit(demo.surface_right, (demo.surface_left.get_width(), 0))
        demo.draw_text_overlay(screen, offset=(0, 0), geom=demo.geom_left, rows=demo.rows_left, style=demo.style_left)
        demo.draw_text_overlay(
            screen, offset=(demo.surface_left.get_width(), 0), geom=demo.geom_right, rows=demo.rows_right, style=demo.style_right
        )

        if demo.hover is not None:
            hb = demo.hover.hit
            dx, dy = demo.hover.offset
            rect = pygame.Rect(dx + hb.x0, dy + hb.y0, hb.x1 - hb.x0, hb.y1 - hb.y0)
            pygame.draw.rect(screen, (255, 220, 140), rect, 0)
            pygame.draw.rect(screen, (255, 180, 90), rect, 1)

        info_rect = pygame.Rect(0, total_h, total_w, info_h)
        pygame.draw.rect(screen, (18, 18, 24), info_rect)
        hover_desc = (
            f"hover: row '{demo.hover.label()}' col {demo.hover.hit.col_idx} part {PART_LABEL.get(int(demo.hover.hit.part), demo.hover.hit.part)}"
            if demo.hover
            else "hover a cell to see hit metadata"
        )
        msg = f"{demo.last_click_desc} | {hover_desc}"
        txt = font.render(msg, True, (235, 235, 240))
        screen.blit(txt, (8, total_h + (info_h - txt.get_height()) // 2))

        pygame.display.flip()
        demo.step()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
