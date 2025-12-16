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

    def __init__(self, *, menu_path: str = "menu.json", start_node: str = "main") -> None:
        self._lock = threading.Lock()

        self._menu_path = str(menu_path)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._load_nodes()

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

                # "start" closes the menu (resume gameplay). Others are surfaced as pending.
                self._pending_action = str(action)
                self._active = False
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

            return MenuSnapshot(
                active=bool(self._active),
                gen=int(self._gen),
                node_id=str(self._node_id),
                title=title,
                item_labels=labels,
                selected_idx=sel,
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

    def __init__(self, *, font, width: int, height: int) -> None:
        self._font = font
        self._w = int(width)
        self._h = int(height)

        self._front_rgba: bytes | None = None
        self._front_gen: int = -1

        self._assets = None

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
        if gen == int(self._front_gen) and self._front_rgba is not None:
            return

        title = str(getattr(snap, "title", ""))
        items = list(getattr(snap, "item_labels", []) or [])
        sel = int(getattr(snap, "selected_idx", 0))

        doc = _menu_list_layout_to_uidoc(gen=int(gen), title=title, items=items, selected_idx=int(sel))
        rgba = self._render_uidoc_to_rgba(doc)
        self._front_rgba = rgba
        self._front_gen = gen

    def _render_uidoc_to_rgba(self, doc: UiDoc) -> bytes:
        import pygame
        import joystick_menu

        self._ensure_assets()

        surf = pygame.Surface((int(self._w), int(self._h)), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 255))

        def _px(x_norm: float) -> int:
            return int(round(float(x_norm) * float(self._w)))

        def _py_from_gl_y(y_norm_gl: float) -> int:
            # UI docs use bottom-left origin like GL; pygame uses top-left.
            return int(round((1.0 - float(y_norm_gl)) * float(self._h)))

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
