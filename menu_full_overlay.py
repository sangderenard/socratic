from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MenuSnapshot:
    active: bool
    gen: int
    node_id: str
    title: str
    item_labels: list[str]
    selected_idx: int
    selected_action: str


@dataclass(frozen=True)
class UiDoc:
    """JSON-serializable UI command list.

    This is the boundary between menu state reducers and rendering.
    """

    gen: int
    commands: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"gen": int(self.gen), "commands": list(self.commands)}


class NonBlockingJsonMenu:
    """Thread-safe, event-driven JSON menu state.

    This object is *not* a per-frame tick. It is mutated only by semantic events
    (e.g., menu_open/menu_up/menu_confirm) and produces immutable snapshots.

    Rendering is expected to be "redraw on dirty" using the snapshot generation.
    """

    def __init__(
        self,
        *,
        menu_path: str = "menu.json",
        start_node: str = "main",
        page_registry: Any | None = None,
    ) -> None:
        self._lock = threading.Lock()

        self._menu_path = str(menu_path)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._load_nodes()

        # Optional per-node page builders (for modular menus).
        # If provided, this is consulted first for node definitions.
        self._page_registry = page_registry

        self._active = False
        self._gen = 0

        self._start_node = str(start_node)
        self._node_id = str(start_node)
        self._stack: list[tuple[str, int]] = []  # (node_id, selected_idx)
        self._selected_idx = 0

        self._pending_action: str | None = None

    def _load_nodes(self) -> None:
        try:
            with open(self._menu_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes") if isinstance(data, dict) else None
            self._nodes = nodes if isinstance(nodes, dict) else {}
        except Exception:
            self._nodes = {}

    def _cur_node(self) -> dict[str, Any]:
        # Prefer a registry-provided node (dynamic / modular) if available.
        try:
            reg = getattr(self, "_page_registry", None)
            if reg is not None and hasattr(reg, "build_node"):
                node = reg.build_node(str(self._node_id))
                if isinstance(node, dict):
                    return node
        except Exception:
            pass
        return self._nodes.get(self._node_id, {}) if isinstance(self._nodes, dict) else {}

    def _cur_items(self) -> list[dict[str, Any]]:
        node = self._cur_node()
        items = node.get("items") if isinstance(node, dict) else None
        return items if isinstance(items, list) else []

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._active)

    def pending_action(self) -> str | None:
        with self._lock:
            return self._pending_action

    def consume_pending_action(self) -> str | None:
        with self._lock:
            act = self._pending_action
            self._pending_action = None
            return act

    def toggle(self, *, start_node: str | None = None) -> None:
        with self._lock:
            if self._active:
                self._active = False
                self._gen += 1
                return
            self._active = True
            self._node_id = str(start_node or self._start_node)
            self._stack = []
            self._selected_idx = 0
            self._pending_action = None
            self._gen += 1

    def close(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._gen += 1

    def nav(self, delta: int) -> None:
        with self._lock:
            if not self._active:
                return
            items = self._cur_items()
            if not items:
                return
            n = int(len(items))
            self._selected_idx = int((int(self._selected_idx) + int(delta)) % n)
            self._gen += 1

    def cancel(self) -> None:
        with self._lock:
            if not self._active:
                return
            if self._stack:
                prev_node, prev_sel = self._stack.pop()
                self._node_id = str(prev_node)
                self._selected_idx = int(prev_sel)
                self._gen += 1
                return
            self._active = False
            self._gen += 1

    def confirm(self) -> None:
        with self._lock:
            if not self._active:
                return
            items = self._cur_items()
            if not items:
                return
            idx = int(self._selected_idx) % int(len(items))
            it = items[idx] if isinstance(items[idx], dict) else {}

            submenu = it.get("submenu") if isinstance(it, dict) else None
            if isinstance(submenu, str) and submenu in self._nodes:
                self._stack.append((self._node_id, int(self._selected_idx)))
                self._node_id = str(submenu)
                self._selected_idx = 0
                self._gen += 1
                return

            action = it.get("action") if isinstance(it, dict) else None
            if isinstance(action, str):
                a = action.strip().lower()
                if a == "back":
                    if self._stack:
                        prev_node, prev_sel = self._stack.pop()
                        self._node_id = str(prev_node)
                        self._selected_idx = int(prev_sel)
                        self._gen += 1
                        return
                    self._active = False
                    self._gen += 1
                    return

                # Only "start" is a semantic close. Other actions are surfaced
                # as pending actions but do NOT close the menu; focus decides
                # which overlay receives input.
                if a == "start":
                    self._pending_action = str(action)
                    self._active = False
                    self._gen += 1
                    return

                self._pending_action = str(action)
                self._gen += 1
                return

            # edit_struct and other complex items are not yet implemented.
            # Treat confirm as a no-op for now.

    def snapshot(self) -> MenuSnapshot:
        with self._lock:
            node = self._cur_node()
            title = str(node.get("title", "")) if isinstance(node, dict) else ""
            items = self._cur_items()
            labels: list[str] = []
            for it in items:
                if isinstance(it, dict):
                    labels.append(str(it.get("label", "")))
                else:
                    labels.append(str(it))

            sel = int(self._selected_idx)
            if labels:
                sel = int(sel % len(labels))
            else:
                sel = 0

            sel_action = ""
            try:
                if items and 0 <= int(sel) < int(len(items)):
                    it = items[int(sel)]
                    if isinstance(it, dict):
                        a = it.get("action")
                        if isinstance(a, str):
                            sel_action = str(a)
            except Exception:
                sel_action = ""

            return MenuSnapshot(
                active=bool(self._active),
                gen=int(self._gen),
                node_id=str(self._node_id),
                title=title,
                item_labels=labels,
                selected_idx=sel,
                selected_action=str(sel_action),
            )


def _menu_list_layout_to_uidoc(
    *,
    gen: int,
    title: str,
    items: list[str],
    selected_idx: int,
    include_reticle: bool = True,
    reticle_stage: str = "hover",
) -> UiDoc:
    """Build the old menu's layout as a data-only UI document.

    Coordinates are in normalized [0,1] with origin at bottom-left (OpenGL style),
    matching joystick_menu._draw_menu_list().
    """

    cmds: list[dict[str, Any]] = []
    cmds.append({"op": "clear", "rgba": [0, 0, 0, 255]})

    cmds.append(
        {
            "op": "text",
            "kind": "title",
            "text": str(title),
            "x": 0.06,
            "y": 0.78,
        }
    )

    y = 0.66
    dy = 0.07
    for it in items:
        cmds.append(
            {
                "op": "text",
                "kind": "item",
                "text": str(it),
                "x": 0.10,
                "y": float(y),
            }
        )
        y -= dy

    if include_reticle and items:
        sel = int(selected_idx) % int(len(items))
        y_sel = float(0.66 - float(sel) * dy)
        cmds.append(
            {
                "op": "sprite",
                "kind": "reticle",
                "sprite": f"reticle_{str(reticle_stage).strip().lower()}",
                "x": 0.06,
                "y": y_sel,
            }
        )

    return UiDoc(gen=int(gen), commands=cmds)


class MenuBufferPresenter:
    """Redraw-on-dirty full-screen menu buffer presenter.

    - `update_if_needed()` renders a snapshot into a pixel buffer only when its
      generation changes.
    - `draw()` draws the last rendered buffer via OpenGL each frame while active.

    This intentionally keeps menu logic out of the render loop: the render loop
    only needs the immutable snapshot + generation counter.
    """

    def __init__(self, *, font, width: int, height: int, page_registry: Any | None = None) -> None:
        self._font = font
        self._w = int(width)
        self._h = int(height)

        self._page_registry = page_registry

        self._front_rgba: bytes | None = None
        self._front_gen: int = -1
        self._front_live_seq: int = -1
        self._front_has_live: bool = False

        self._assets = None

        self._ctl_lib = None

    def _ensure_ctl(self) -> None:
        if self._ctl_lib is not None:
            return
        try:
            from c_physics.controller_engine_api import try_load_controller_engine

            lib, _ = try_load_controller_engine(search_dir="c_physics")
            self._ctl_lib = lib
        except Exception:
            self._ctl_lib = None

    def _ensure_assets(self) -> None:
        if self._assets is not None:
            return
        try:
            import reticle_sprite

            self._assets = reticle_sprite.load_reticle_assets(size_px=64)
        except Exception:
            self._assets = None

    def update_if_needed(self, snap: MenuSnapshot) -> None:
        if not bool(getattr(snap, "active", False)):
            return
        gen = int(getattr(snap, "gen", -1))

        # For live widgets (e.g., wheel waveforms), we allow redraw when the
        # controller wheel tick advances even if menu generation is unchanged.
        if gen == int(self._front_gen) and self._front_rgba is not None and not bool(self._front_has_live):
            return

        if gen == int(self._front_gen) and self._front_rgba is not None and bool(self._front_has_live):
            self._ensure_ctl()
            try:
                if self._ctl_lib is not None and hasattr(self._ctl_lib, "gp_ctl_wheel_get_tick_seq"):
                    cur = int(self._ctl_lib.gp_ctl_wheel_get_tick_seq())
                    if cur == int(self._front_live_seq):
                        return
            except Exception:
                # If live-seq check fails, fall through and redraw.
                pass

        title = str(getattr(snap, "title", ""))
        items = list(getattr(snap, "item_labels", []) or [])
        sel = int(getattr(snap, "selected_idx", 0))

        # Optional registry hook: allow a page to supply a custom UiDoc.
        doc = None
        try:
            reg = getattr(self, "_page_registry", None)
            if reg is not None and hasattr(reg, "build_uidoc"):
                maybe = reg.build_uidoc(snap=snap, width_px=int(self._w), height_px=int(self._h))
                if isinstance(maybe, UiDoc):
                    doc = maybe
        except Exception:
            doc = None

        if doc is None:
            doc = _menu_list_layout_to_uidoc(gen=int(gen), title=title, items=items, selected_idx=int(sel))
        rgba = self._render_uidoc_to_rgba(doc)
        self._front_rgba = rgba
        self._front_gen = gen

        # Track whether doc contains any live widgets.
        self._front_has_live = False
        try:
            for cmd in (doc.commands or []):
                if isinstance(cmd, dict) and str(cmd.get("op", "")).strip().lower() == "wheel_waveform":
                    self._front_has_live = True
                    break
        except Exception:
            self._front_has_live = False

        if self._front_has_live:
            self._ensure_ctl()
            try:
                if self._ctl_lib is not None and hasattr(self._ctl_lib, "gp_ctl_wheel_get_tick_seq"):
                    self._front_live_seq = int(self._ctl_lib.gp_ctl_wheel_get_tick_seq())
            except Exception:
                self._front_live_seq = -1

    def _render_uidoc_to_rgba(self, doc: UiDoc) -> bytes:
        import pygame
        import joystick_menu

        import ctypes

        self._ensure_assets()

        surf = pygame.Surface((int(self._w), int(self._h)), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 255))

        def _px(x_norm: float) -> int:
            return int(round(float(x_norm) * float(self._w)))

        def _py_from_gl_y(y_norm_gl: float) -> int:
            # UI docs use bottom-left origin like GL; pygame uses top-left.
            return int(round((1.0 - float(y_norm_gl)) * float(self._h)))

        def _rgba_u8(cmd: dict[str, Any], key: str = "rgba") -> tuple[int, int, int, int]:
            rgba = cmd.get(key)
            try:
                r, g, b, a = (int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
                return (r, g, b, a)
            except Exception:
                return (255, 255, 255, 255)

        def _get_reticle_surface(sprite_name: str):
            if self._assets is None:
                return None
            try:
                import reticle_sprite

                name = str(sprite_name).strip().lower()
                if name.endswith("_idle"):
                    st = reticle_sprite.ReticleStage.IDLE
                elif name.endswith("_locked"):
                    st = reticle_sprite.ReticleStage.LOCKED
                elif name.endswith("_ready"):
                    st = reticle_sprite.ReticleStage.READY
                else:
                    st = reticle_sprite.ReticleStage.HOVER
                return self._assets.surfaces.get(st)
            except Exception:
                return None

        for cmd in (doc.commands or []):
            if not isinstance(cmd, dict):
                continue
            op = str(cmd.get("op", "") or "").strip().lower()
            if op == "clear":
                rgba = cmd.get("rgba")
                try:
                    r, g, b, a = (int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
                except Exception:
                    r, g, b, a = (0, 0, 0, 255)
                surf.fill((r, g, b, a))
                continue
            if op == "text":
                txt = str(cmd.get("text", ""))
                if "x_px" in cmd and "y_px" in cmd:
                    x = int(cmd.get("x_px") or 0)
                    y = int(cmd.get("y_px") or 0)
                else:
                    x = _px(float(cmd.get("x", 0.0) or 0.0))
                    y = _py_from_gl_y(float(cmd.get("y", 0.0) or 0.0))
                s = joystick_menu._render_cell_text(self._font, txt)
                surf.blit(s, (x, y))
                continue
            if op == "sprite":
                sprite = str(cmd.get("sprite", "") or "").strip().lower()
                if sprite.startswith("reticle_"):
                    x = _px(float(cmd.get("x", 0.0) or 0.0))
                    y = _py_from_gl_y(float(cmd.get("y", 0.0) or 0.0))
                    ret = _get_reticle_surface(sprite)
                    if ret is not None:
                        surf.blit(ret, (x, y))
                continue

            if op == "rect":
                x = int(cmd.get("x_px") or 0)
                y = int(cmd.get("y_px") or 0)
                w = int(cmd.get("w_px") or 0)
                h = int(cmd.get("h_px") or 0)
                r, g, b, a = _rgba_u8(cmd, "rgba")
                if w > 0 and h > 0:
                    pygame.draw.rect(surf, (r, g, b, a), pygame.Rect(x, y, w, h), width=0)
                if "outline_rgba" in cmd:
                    or_, og, ob, oa = _rgba_u8(cmd, "outline_rgba")
                    ow = int(cmd.get("outline_w_px") or 1)
                    if w > 0 and h > 0 and ow > 0:
                        pygame.draw.rect(surf, (or_, og, ob, oa), pygame.Rect(x, y, w, h), width=ow)
                continue

            if op == "line":
                x0 = int(cmd.get("x0_px") or 0)
                y0 = int(cmd.get("y0_px") or 0)
                x1 = int(cmd.get("x1_px") or 0)
                y1 = int(cmd.get("y1_px") or 0)
                r, g, b, a = _rgba_u8(cmd, "rgba")
                ww = int(cmd.get("w_px") or 1)
                pygame.draw.line(surf, (r, g, b, a), (x0, y0), (x1, y1), width=max(1, ww))
                continue

            if op == "circle":
                cx = int(cmd.get("cx_px") or 0)
                cy = int(cmd.get("cy_px") or 0)
                rr = int(cmd.get("r_px") or 0)
                r, g, b, a = _rgba_u8(cmd, "rgba")
                if rr > 0:
                    pygame.draw.circle(surf, (r, g, b, a), (cx, cy), rr)
                continue

            if op == "wheel_waveform":
                x = int(cmd.get("x_px") or 0)
                y = int(cmd.get("y_px") or 0)
                w = int(cmd.get("w_px") or 0)
                h = int(cmd.get("h_px") or 0)
                sig = int(cmd.get("signal_idx") or 0)
                span = int(cmd.get("span") or 1)
                if w <= 0 or h <= 0:
                    continue
                self._ensure_ctl()
                if self._ctl_lib is None or not hasattr(self._ctl_lib, "gp_ctl_wheel_raster_rgba"):
                    continue
                try:
                    buf = bytearray(int(w * h * 4))
                    p = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
                    ok = int(self._ctl_lib.gp_ctl_wheel_raster_rgba(int(sig), int(span), int(w), int(h), p, int(len(buf))))
                    if ok:
                        img = pygame.image.frombuffer(bytes(buf), (int(w), int(h)), "RGBA")
                        surf.blit(img, (x, y))
                except Exception:
                    pass
                continue

        return pygame.image.tostring(surf, "RGBA", True)

    def draw(self) -> None:
        if self._front_rgba is None:
            return

        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_RGBA,
            GL_SRC_ALPHA,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_BYTE,
            glBlendFunc,
            glClear,
            glClearColor,
            glDisable,
            glDrawPixels,
            glEnable,
            glIsEnabled,
            glLoadIdentity,
            glMatrixMode,
            glOrtho,
            glPixelStorei,
            glPopMatrix,
            glPushMatrix,
            glRasterPos2f,
        )

        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        if depth_was_enabled:
            glDisable(GL_DEPTH_TEST)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glRasterPos2f(0.0, 0.0)
        glDrawPixels(int(self._w), int(self._h), GL_RGBA, GL_UNSIGNED_BYTE, self._front_rgba)

        glDisable(GL_BLEND)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
