from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import scroll_model


DrawRectFn = Callable[[int, int, int, int, tuple[float, float, float, float]], None]
DrawTextFn = Callable[[int, int, str, tuple[int, int, int]], None]
RowRenderFn = Callable[[int, Any, int], None]


@dataclass(frozen=True)
class TableGeom:
    x0: int
    y0: int
    w: int
    h: int
    pad: int
    header_h: int
    row_h: int


def max_visible_rows(g: TableGeom) -> int:
    # Keep behavior consistent with existing workbench: a minimum of 6.
    inner_h = int(g.h - (g.pad * 3 + g.header_h))
    return max(6, int(inner_h / max(1, int(g.row_h))))


def visible_window(*, rows_total: int, sel_idx: int, g: TableGeom, scroll: scroll_model.ScrollModel, center: bool) -> tuple[int, int, int]:
    mv = int(max_visible_rows(g))
    sel = int(max(0, min(int(sel_idx), max(0, int(rows_total) - 1)))) if rows_total else 0
    scroll.ensure_visible(sel=int(sel), total=int(rows_total), max_visible=int(mv), center=bool(center))
    first, last, _vis = scroll.window(total=int(rows_total), sel=int(sel), max_visible=int(mv))
    return int(first), int(last), int(mv)


def apply_scroll(
    *,
    scroll: scroll_model.ScrollModel,
    delta_lines: int = 0,
    delta_pages: int = 0,
    rows_total: int,
    max_visible: int,
) -> None:
    if int(delta_lines):
        scroll.scroll_lines(delta=int(delta_lines), total=int(rows_total), max_visible=int(max_visible))
    if int(delta_pages):
        scroll.scroll_pages(delta_pages=int(delta_pages), total=int(rows_total), max_visible=int(max_visible))


def row_baseline_y(*, g: TableGeom, visible_idx: int) -> int:
    # Matches the workbench y layout: pad + header + 8 + (row_index+1)*row_h
    return int(g.y0 + g.pad + g.header_h + 8 + (int(visible_idx) + 1) * int(g.row_h))


def iter_visible_rows(
    *,
    rows: Sequence[Any],
    sel_idx: int,
    g: TableGeom,
    scroll: scroll_model.ScrollModel,
    center: bool,
) -> Iterable[tuple[int, Any, int, int]]:
    first, last, mv = visible_window(rows_total=len(rows), sel_idx=int(sel_idx), g=g, scroll=scroll, center=bool(center))
    for idx in range(int(first), int(last)):
        y = row_baseline_y(g=g, visible_idx=(idx - int(first)))
        yield int(idx), rows[int(idx)], int(y), int(mv)
