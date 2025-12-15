from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScrollModel:
    """Small helper for list scrolling/windowing.

    - Keeps a `first_idx` window offset.
    - Provides auto-scroll to keep selection visible (optionally centered).
    - Supports manual line/page scrolling.

    This is intentionally UI-framework agnostic; callers decide how to bind controls.
    """

    first_idx: int = 0

    def clamp(self, *, total: int, max_visible: int) -> None:
        total = int(max(0, total))
        max_visible = int(max(1, max_visible))
        max_first = max(0, total - max_visible)
        self.first_idx = int(max(0, min(int(self.first_idx), int(max_first))))

    def ensure_visible(self, *, sel: int, total: int, max_visible: int, center: bool = False) -> None:
        """Adjust first_idx so sel is visible."""
        total = int(max(0, total))
        max_visible = int(max(1, max_visible))
        if total <= max_visible:
            self.first_idx = 0
            return

        sel = int(max(0, min(int(sel), total - 1)))
        self.clamp(total=total, max_visible=max_visible)

        if center:
            half = max_visible // 2
            desired = sel - half
            max_first = max(0, total - max_visible)
            self.first_idx = int(max(0, min(int(desired), int(max_first))))
            return

        if sel < self.first_idx:
            self.first_idx = int(sel)
        elif sel >= (self.first_idx + max_visible):
            self.first_idx = int(sel - max_visible + 1)

        self.clamp(total=total, max_visible=max_visible)

    def scroll_lines(self, *, delta: int, total: int, max_visible: int) -> None:
        self.first_idx = int(self.first_idx) + int(delta)
        self.clamp(total=total, max_visible=max_visible)

    def scroll_pages(self, *, delta_pages: int, total: int, max_visible: int) -> None:
        self.scroll_lines(delta=int(delta_pages) * int(max(1, max_visible)), total=total, max_visible=max_visible)

    def window(self, *, total: int, sel: int, max_visible: int) -> tuple[int, int, int]:
        """Return (first_idx, last_idx_exclusive, visible_sel)."""
        total = int(max(0, total))
        max_visible = int(max(1, max_visible))
        self.clamp(total=total, max_visible=max_visible)
        first = int(self.first_idx)
        last = int(min(total, first + max_visible))
        sel = int(max(0, min(int(sel), max(0, total - 1)))) if total else 0
        visible_sel = int(sel - first)
        return first, last, visible_sel
