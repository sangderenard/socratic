from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


MenuUiDocBuilder = Callable[[Any, int, int], Any]


@dataclass(frozen=True)
class MenuPageContext:
    """Immutable context passed to menu page builders.

    Keep this minimal: page builders should be pure and fast.
    """

    node_id: str


MenuNodeBuilder = Callable[[MenuPageContext], dict[str, Any]]


class MenuPageRegistry:
    """Registry for modular menu node construction.

    Nodes not present in `overrides` fall back to whatever the JSON menu provides.

    This is the bridge for "100 pages forever": pages can be added as independent
    builders without modifying the core menu reducer or presenter.
    """

    def __init__(
        self,
        *,
        overrides: dict[str, MenuNodeBuilder] | None = None,
        uidoc_overrides: dict[str, MenuUiDocBuilder] | None = None,
    ) -> None:
        self._overrides: dict[str, MenuNodeBuilder] = dict(overrides or {})
        self._uidoc_overrides: dict[str, MenuUiDocBuilder] = dict(uidoc_overrides or {})

    def build_node(self, node_id: str) -> dict[str, Any] | None:
        fn = self._overrides.get(str(node_id))
        if fn is None:
            return None
        try:
            return fn(MenuPageContext(node_id=str(node_id)))
        except Exception:
            return None

    def build_uidoc(self, *, snap: Any, width_px: int, height_px: int):
        """Return a UiDoc (or None) for the current node.

        The presenter calls this opportunistically. Builders should be fast and
        should not mutate global state.
        """

        try:
            node_id = str(getattr(snap, "node_id", ""))
        except Exception:
            node_id = ""
        fn = self._uidoc_overrides.get(str(node_id))
        if fn is None:
            return None
        try:
            return fn(snap, int(width_px), int(height_px))
        except Exception:
            return None


def _controls_controller_page(_ctx: MenuPageContext) -> dict[str, Any]:
    # Keep schema identical to menu.json node definition.
    return {
        "title": "CONTROLLER / SIGNALS / CHANNELS",
        "items": [
            {"label": "SIGNAL WORKBENCH", "action": "controller_workbench"},
            {"label": "CHANNEL MIXER", "action": "controller_channel_mixer"},
            {"label": "DISCOVER FEATURE", "action": "controller_discover_feature"},
            {"label": "VIEW CHANNEL", "action": "controller_view_channel"},
            {"label": "MAP CHANNEL", "action": "controller_map_channel"},
            {"label": "BACK", "action": "back"},
        ],
    }


def _controls_controller_uidoc(snap: Any, width_px: int, height_px: int):
    """Dense-ish controls page: classic menu list + a right-side live waveform panel."""

    try:
        from menu_full_overlay import UiDoc
        from menu_archetypes import UiDocBuilder
    except Exception:
        return None

    title = str(getattr(snap, "title", ""))
    items = list(getattr(snap, "item_labels", []) or [])
    sel = int(getattr(snap, "selected_idx", 0))
    gen = int(getattr(snap, "gen", 0))
    sel_action = str(getattr(snap, "selected_action", "") or "")

    # Left: keep the familiar list layout in normalized space.
    b = UiDocBuilder(commands=[])
    b.clear((0, 0, 0, 255))
    b.text_norm(x=0.06, y=0.78, text=title, kind="title")

    y = 0.66
    dy = 0.07
    for it in items:
        b.text_norm(x=0.10, y=float(y), text=str(it), kind="item")
        y -= dy
    if items:
        y_sel = float(0.66 - float(int(sel) % int(len(items))) * dy)
        b.sprite_norm(x=0.06, y=y_sel, sprite="reticle_hover", kind="reticle")

    # Right: preview panel. If SIGNAL WORKBENCH is highlighted, show a compact
    # workbench preview; otherwise keep a small waveform widget.
    pad = 18
    panel_w = max(220, int(width_px * 0.34))
    panel_h = max(140, int(height_px * 0.22))
    x0 = int(width_px - panel_w - pad)
    y0 = int(pad)

    if str(sel_action).strip().lower() == "controller_workbench":
        try:
            import workbench_overlay

            workbench_overlay.append_workbench_preview_panel(
                b=b,
                x_px=x0,
                y_px=y0,
                w_px=panel_w,
                h_px=panel_h,
                signal_idx=0,
            )
        except Exception:
            b.rect_px(x_px=x0, y_px=y0, w_px=panel_w, h_px=panel_h, rgba_u8=(10, 10, 14, 255), outline_rgba_u8=(80, 80, 90, 255))
            b.text_px(x_px=x0 + 10, y_px=y0 + 10, text="WORKBENCH PREVIEW (unavailable)", kind="hud")
    else:
        b.rect_px(x_px=x0, y_px=y0, w_px=panel_w, h_px=panel_h, rgba_u8=(10, 10, 14, 255), outline_rgba_u8=(80, 80, 90, 255))
        b.text_px(x_px=x0 + 10, y_px=y0 + 10, text="WHEEL WAVEFORM (sig 0)", kind="hud")
        b.wheel_waveform_px(x_px=x0 + 10, y_px=y0 + 34, w_px=panel_w - 20, h_px=panel_h - 44, signal_idx=0, span=1)

    return UiDoc(gen=gen, commands=b.commands)


# Default registry used by the runtime.
# Start small (one node) and grow organically.
DEFAULT_REGISTRY = MenuPageRegistry(
    overrides={
        "controls_controller": _controls_controller_page,
    }
    ,
    uidoc_overrides={
        "controls_controller": _controls_controller_uidoc,
        "workbench": (lambda snap, width_px, height_px: __import__("workbench_overlay").build_workbench_uidoc(snap=snap, width_px=width_px, height_px=height_px)),
    },
)
