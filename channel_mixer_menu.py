from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os

import pygame

import controller_graph_compile


@dataclass
class HitBox:
    x0: int
    y0: int
    x1: int
    y1: int
    payload: dict[str, Any]

    def contains(self, x: int, y: int) -> bool:
        return int(self.x0) <= int(x) < int(self.x1) and int(self.y0) <= int(y) < int(self.y1)


def run_channel_mixer_menu(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    menu_button: int | None,
    get_menu_nav,
    nav_edge,
    poll_joystick_snapshot,
    draw_fullscreen_lines,
    compiled_path: str = "controller_graph_compiled.json",
    mixer_path: str = "channel_mixer.json",
    final_path: str = "controller_graph_final.json",
) -> None:
    """Channel Mixer (placeholder).

    - Top row: input channels (from compiled graph)
    - Bottom row: output channels (same set for now)
    - Click a top channel to select it, then click a bottom channel to route it.
    - Clicking a bottom channel with no selection clears that output route.

    Persists routes into mixer_path, and writes final_path by merging routes onto
    the compiled graph.
    """

    compile_ok = False
    compile_msg = ""

    def _rebuild_graphs() -> dict[str, Any] | None:
        nonlocal compile_ok, compile_msg
        ok, msg, compiled, _final = controller_graph_compile.try_build_final_graph(
            joystick_path="joystick.json",
            mixer_path=str(mixer_path),
            compiled_out_path=str(compiled_path),
            final_out_path=str(final_path),
        )
        compile_ok = bool(ok)
        compile_msg = str(msg)
        return compiled if isinstance(compiled, dict) else None

    def _load_routes() -> dict[str, str]:
        try:
            with open(mixer_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return {}
            rr = d.get("routes") if "routes" in d else d
            if not isinstance(rr, dict):
                return {}
            out: dict[str, str] = {}
            for k, v in rr.items():
                try:
                    out[str(int(k))] = str(int(v))
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    def _save_routes(routes: dict[str, str]) -> None:
        try:
            with open(mixer_path, "w", encoding="utf-8") as f:
                json.dump({"routes": routes}, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass

    def _write_final(compiled: dict[str, Any], routes: dict[str, str]) -> None:
        try:
            final = controller_graph_compile.finalize_compiled_graph(compiled, mixer_routes=routes)
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass

    compiled = _rebuild_graphs()
    if not isinstance(compiled, dict):
        draw_fullscreen_lines(
            font,
            int(width),
            int(height),
            [
                "CHANNEL MIXER",
                "COMPILE ERROR",
                str(compile_msg or "unknown error"),
                "(Esc/menu exits)",
            ],
        )
        pygame.display.flip()
        # wait for exit
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return
            clock.tick(60)

    base_channels = compiled.get("channels")
    if not isinstance(base_channels, dict):
        base_channels = {}
    ch_keys = sorted(list(base_channels.keys()), key=lambda s: int(s) if str(s).isdigit() else 10_000)

    routes = _load_routes()
    # Ensure artifacts match mixer file on entry.
    _save_routes(routes)
    _rebuild_graphs()

    sel_in: str | None = None
    clock = pygame.time.Clock()

    def _draw(routes_now: dict[str, str], sel: str | None) -> list[HitBox]:
        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_RGBA,
            GL_SRC_ALPHA,
            GL_TRIANGLES,
            GL_TRIANGLE_FAN,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_BYTE,
            glBegin,
            glBlendFunc,
            glClear,
            glClearColor,
            glColor4f,
            glDisable,
            glDrawPixels,
            glEnable,
            glEnd,
            glIsEnabled,
            glLoadIdentity,
            glMatrixMode,
            glOrtho,
            glPixelStorei,
            glPopMatrix,
            glPushMatrix,
            glRasterPos2f,
            glVertex2f,
        )

        hit: list[HitBox] = []

        def _px_to_nx(x: int) -> float:
            return float(x) / float(max(1, int(width)))

        def _px_to_ny(y: int) -> float:
            return 1.0 - (float(y) / float(max(1, int(height))))

        def _draw_rect_px(x: int, y: int, w: int, h: int, rgba: tuple[float, float, float, float]) -> None:
            x0 = _px_to_nx(int(x))
            x1 = _px_to_nx(int(x + w))
            y0 = _px_to_ny(int(y + h))
            y1 = _px_to_ny(int(y))
            r, g, b, a = rgba
            glColor4f(float(r), float(g), float(b), float(a))
            glBegin(GL_TRIANGLES)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

        def _draw_circle_px(cx: int, cy: int, r_px: int, rgba: tuple[float, float, float, float], segments: int = 16) -> None:
            cxn = _px_to_nx(int(cx))
            cyn = _px_to_ny(int(cy))
            rn = float(r_px) / float(max(1, int(width)))
            r, g, b, a = rgba
            glColor4f(float(r), float(g), float(b), float(a))
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cxn, cyn)
            for i in range(int(segments) + 1):
                ang = (float(i) / float(max(1, int(segments)))) * 6.283185307179586
                glVertex2f(float(cxn + rn * float(__import__("math").cos(ang))), float(cyn + rn * float(__import__("math").sin(ang))))
            glEnd()

        def _draw_text_px(x: int, y: int, text: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
            s = str(text)
            if not s:
                return
            try:
                surf = font.render(s, True, color)
                surf = surf.convert_alpha()
            except Exception:
                return
            data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(_px_to_nx(int(x)), _px_to_ny(int(y)))
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

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

        pad = 12
        header_h = max(18, int(font.get_linesize()) + 2)

        status = "ACTIVE" if compile_ok else "COMPILE ERROR"
        status_color = (120, 255, 120) if compile_ok else (255, 120, 120)
        _draw_text_px(pad, pad + header_h, f"CHANNEL MIXER (top=input, bottom=output)  [{status}]")
        if not compile_ok and compile_msg:
            _draw_text_px(pad, pad + header_h + 18, str(compile_msg), status_color)
        _draw_text_px(pad, height - pad - 2, "ESC/Q/menu-btn: exit   click top then bottom to route")

        n = max(1, len(ch_keys))
        x0 = pad + 60
        x1 = width - pad - 60
        span = max(1, x1 - x0)

        top_y = pad + header_h + 34
        bot_y = height - pad - 60
        mid_top = top_y + 30
        mid_bot = bot_y - 30

        # Node positions
        top_pos: dict[str, tuple[int, int]] = {}
        bot_pos: dict[str, tuple[int, int]] = {}
        for i, ch in enumerate(ch_keys):
            t = float(i) / float(max(1, n - 1)) if n > 1 else 0.5
            px = int(x0 + t * float(span))
            top_pos[str(ch)] = (px, int(top_y))
            bot_pos[str(ch)] = (px, int(bot_y))

        # Draw identity or mapped lines for every output
        for out_ch in ch_keys:
            src_ch = routes_now.get(str(out_ch), str(out_ch))
            if src_ch not in top_pos or str(out_ch) not in bot_pos:
                continue
            xA, yA = top_pos[str(src_ch)]
            xB, yB = bot_pos[str(out_ch)]
            remap = str(src_ch) != str(out_ch)
            col = (0.25, 1.0, 0.25, 0.9) if remap else (0.25, 0.25, 0.25, 0.75)
            # crude line via thin rect segments
            steps = 24
            for s in range(steps):
                u0 = float(s) / float(steps)
                u1 = float(s + 1) / float(steps)
                xa = int(xA + (xB - xA) * u0)
                ya = int(mid_top + (mid_bot - mid_top) * u0)
                xb = int(xA + (xB - xA) * u1)
                yb = int(mid_top + (mid_bot - mid_top) * u1)
                _draw_rect_px(min(xa, xb), min(ya, yb), max(1, abs(xb - xa) + 1), max(1, abs(yb - ya) + 1), col)

        # Draw top row
        _draw_text_px(pad, top_y + 6, "IN")
        for ch in ch_keys:
            px, py = top_pos[str(ch)]
            is_sel = (sel is not None) and (str(sel) == str(ch))
            _draw_circle_px(px, py, 8 if is_sel else 6, (1.0, 1.0, 0.25, 1.0) if is_sel else (0.35, 0.35, 0.35, 1.0))
            _draw_text_px(px - 10, py - 14, str(ch))
            hit.append(HitBox(px - 10, py - 10, px + 10, py + 10, {"kind": "mixer_in", "ch": str(ch)}))

        # Draw bottom row
        _draw_text_px(pad, bot_y + 6, "OUT")
        for ch in ch_keys:
            px, py = bot_pos[str(ch)]
            mapped = routes_now.get(str(ch), str(ch))
            remap = str(mapped) != str(ch)
            _draw_circle_px(px, py, 6, (0.25, 1.0, 0.25, 1.0) if remap else (0.35, 0.35, 0.35, 1.0))
            _draw_text_px(px - 10, py - 14, str(ch))
            hit.append(HitBox(px - 10, py - 10, px + 10, py + 10, {"kind": "mixer_out", "ch": str(ch)}))

        # Selection status
        if sel is not None:
            _draw_text_px(pad, top_y - 18, f"selected IN: {sel}")

        glDisable(GL_BLEND)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return hit

    while True:
        pending_click: tuple[int, int] | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return
            if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                return
            if event.type == pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) in (1,):
                try:
                    pending_click = (int(event.pos[0]), int(event.pos[1]))
                except Exception:
                    pending_click = None

        # still honor menu-nav cancel edge
        if joystick is not None:
            axes_now, buttons_now, hats_now = poll_joystick_snapshot(joystick)
        else:
            axes_now, buttons_now, hats_now = {}, set(), {}
        nav = get_menu_nav({})
        b_cancel = nav.get("cancel")
        try:
            if nav_edge(b_cancel, axes_now=axes_now, axes_prev={}, buttons_now=buttons_now, buttons_prev=set(), hats_now=hats_now, hats_prev={}):
                return
        except Exception:
            pass

        hitboxes = _draw(routes, sel_in)
        pygame.display.flip()

        if pending_click is not None:
            mx, my = pending_click
            for hb in hitboxes:
                if hb.contains(mx, my):
                    k = hb.payload.get("kind")
                    ch = str(hb.payload.get("ch", ""))
                    if k == "mixer_in":
                        sel_in = ch
                    elif k == "mixer_out":
                        if not ch:
                            break
                        if sel_in is None:
                            routes.pop(ch, None)
                        else:
                            if sel_in == ch:
                                routes.pop(ch, None)
                            else:
                                routes[ch] = sel_in
                        _save_routes(routes)
                        compiled2 = _rebuild_graphs()
                        if isinstance(compiled2, dict):
                            compiled = compiled2
                    break

        clock.tick(60)
