from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UiDocBuilder:
    """Small helper for building UiDoc command lists.

    Commands are JSON-serializable dicts.

    Conventions:
    - Existing menu ops use normalized GL-space (x/y in [0,1], origin bottom-left).
    - Dense widgets (tables, LEDs, waveforms) use pixel space with keys `*_px` and
      a top-left origin (pygame-style). Presenter handles both.
    """

    commands: list[dict[str, Any]]

    def clear(self, rgba_u8: tuple[int, int, int, int] = (0, 0, 0, 255)) -> None:
        r, g, b, a = (int(rgba_u8[0]), int(rgba_u8[1]), int(rgba_u8[2]), int(rgba_u8[3]))
        self.commands.append({"op": "clear", "rgba": [r, g, b, a]})

    def text_norm(self, *, x: float, y: float, text: str, kind: str = "item") -> None:
        self.commands.append({"op": "text", "kind": str(kind), "text": str(text), "x": float(x), "y": float(y)})

    def sprite_norm(self, *, x: float, y: float, sprite: str, kind: str = "sprite") -> None:
        self.commands.append({"op": "sprite", "kind": str(kind), "sprite": str(sprite), "x": float(x), "y": float(y)})

    def text_px(self, *, x_px: int, y_px: int, text: str, kind: str = "text") -> None:
        self.commands.append({"op": "text", "kind": str(kind), "text": str(text), "x_px": int(x_px), "y_px": int(y_px)})

    def rect_px(
        self,
        *,
        x_px: int,
        y_px: int,
        w_px: int,
        h_px: int,
        rgba_u8: tuple[int, int, int, int],
        outline_rgba_u8: tuple[int, int, int, int] | None = None,
        outline_w_px: int = 1,
    ) -> None:
        r, g, b, a = (int(rgba_u8[0]), int(rgba_u8[1]), int(rgba_u8[2]), int(rgba_u8[3]))
        cmd: dict[str, Any] = {"op": "rect", "x_px": int(x_px), "y_px": int(y_px), "w_px": int(w_px), "h_px": int(h_px), "rgba": [r, g, b, a]}
        if outline_rgba_u8 is not None:
            or_, og, ob, oa = (int(outline_rgba_u8[0]), int(outline_rgba_u8[1]), int(outline_rgba_u8[2]), int(outline_rgba_u8[3]))
            cmd["outline_rgba"] = [or_, og, ob, oa]
            cmd["outline_w_px"] = int(outline_w_px)
        self.commands.append(cmd)

    def line_px(
        self,
        *,
        x0_px: int,
        y0_px: int,
        x1_px: int,
        y1_px: int,
        rgba_u8: tuple[int, int, int, int],
        w_px: int = 1,
    ) -> None:
        r, g, b, a = (int(rgba_u8[0]), int(rgba_u8[1]), int(rgba_u8[2]), int(rgba_u8[3]))
        self.commands.append(
            {
                "op": "line",
                "x0_px": int(x0_px),
                "y0_px": int(y0_px),
                "x1_px": int(x1_px),
                "y1_px": int(y1_px),
                "rgba": [r, g, b, a],
                "w_px": int(w_px),
            }
        )

    def circle_px(self, *, cx_px: int, cy_px: int, r_px: int, rgba_u8: tuple[int, int, int, int]) -> None:
        r, g, b, a = (int(rgba_u8[0]), int(rgba_u8[1]), int(rgba_u8[2]), int(rgba_u8[3]))
        self.commands.append({"op": "circle", "cx_px": int(cx_px), "cy_px": int(cy_px), "r_px": int(r_px), "rgba": [r, g, b, a]})

    def wheel_waveform_px(
        self,
        *,
        x_px: int,
        y_px: int,
        w_px: int,
        h_px: int,
        signal_idx: int,
        span: int = 1,
    ) -> None:
        self.commands.append(
            {
                "op": "wheel_waveform",
                "x_px": int(x_px),
                "y_px": int(y_px),
                "w_px": int(w_px),
                "h_px": int(h_px),
                "signal_idx": int(signal_idx),
                "span": int(span),
            }
        )


def build_list_menu_uidoc(*, gen: int, title: str, items: list[str], selected_idx: int, include_reticle: bool = True) -> dict[str, Any]:
    """Return a UiDoc-like dict (gen + commands) for the classic list menu."""

    b = UiDocBuilder(commands=[])
    b.clear((0, 0, 0, 255))

    b.text_norm(x=0.06, y=0.78, text=str(title), kind="title")

    y = 0.66
    dy = 0.07
    for it in items:
        b.text_norm(x=0.10, y=float(y), text=str(it), kind="item")
        y -= dy

    if include_reticle and items:
        sel = int(selected_idx) % int(len(items))
        y_sel = float(0.66 - float(sel) * dy)
        b.sprite_norm(x=0.06, y=y_sel, sprite="reticle_hover", kind="reticle")

    return {"gen": int(gen), "commands": b.commands}
