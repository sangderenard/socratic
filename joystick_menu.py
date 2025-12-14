from __future__ import annotations

import json
import os
import shutil
import ctypes
from typing import Any

import pygame

import input_graph
import reticle_sprite


MENU_SPEC_PATH = "menu.json"


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    # Persist changes safely:
    # - If the file already exists, copy it to *.bak before modifying.
    # - Write to a temp file, then atomically replace.
    if os.path.exists(path):
        try:
            shutil.copyfile(path, f"{path}.bak")
        except Exception:
            pass
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_or_create_joystick_config(path: str = "joystick.json") -> dict[str, Any]:
    if not os.path.exists(path):
        # Blank config by default.
        _atomic_write_json(path, {})
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        # If the file exists but is corrupted, overwrite with blank.
        try:
            shutil.copyfile(path, f"{path}.bak")
        except Exception:
            pass
        _atomic_write_json(path, {})
        return {}


def save_joystick_config(cfg: dict[str, Any], path: str = "joystick.json") -> None:
    if not isinstance(cfg, dict):
        cfg = {}
    _atomic_write_json(path, cfg)


def _default_menu_spec() -> dict[str, Any]:
    return {
        "version": 1,
        "nodes": {
            "main": {
                "title": "MAIN MENU",
                "items": [
                    {"label": "CONTROLS", "submenu": "controls"},
                    {"label": "START", "action": "start"},
                ],
            },
            "controls": {
                "title": "CONTROLS",
                "items": [
                    {"label": "BIND RUDDER", "action": "bind_rudder"},
                    {"label": "BIND ELEVATOR", "action": "bind_elevator"},
                    {"label": "BIND VIEW YAW", "action": "bind_view_yaw"},
                    {"label": "BIND VIEW PITCH", "action": "bind_view_pitch"},
                    {"label": "BIND CAMERA ZOOM IN", "action": "bind_camera_zoom_in"},
                    {"label": "BIND CAMERA ZOOM OUT", "action": "bind_camera_zoom_out"},
                    {"label": "BIND DEBUG HUD TOGGLE", "action": "bind_debug_hud_toggle"},
                    {"label": "BIND THROTTLE FWD", "action": "bind_throttle_fwd"},
                    {"label": "BIND THROTTLE REV", "action": "bind_throttle_rev"},
                    {"label": "BACK", "action": "back"},
                ],
            },
        },
    }


def load_or_create_menu_spec(path: str = MENU_SPEC_PATH) -> dict[str, Any]:
    if not os.path.exists(path):
        spec = _default_menu_spec()
        _atomic_write_json(path, spec)
        return spec
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec if isinstance(spec, dict) else _default_menu_spec()
    except Exception:
        try:
            shutil.copyfile(path, f"{path}.bak")
        except Exception:
            pass
        spec = _default_menu_spec()
        _atomic_write_json(path, spec)
        return spec


def _get_menu_node(spec: dict[str, Any], node_id: str) -> dict[str, Any]:
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    node = nodes.get(node_id)
    return node if isinstance(node, dict) else {}


def _node_items(node: dict[str, Any]) -> list[dict[str, Any]]:
    items = node.get("items")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("label"), str):
            out.append(it)
    return out


def _render_cell_text(font: pygame.font.Font, text: str) -> pygame.Surface:
    s = str(text).expandtabs(4)
    try:
        cell_w, cell_h = font.size("M")
    except Exception:
        cell_w, cell_h = (10, font.get_height())

    cell_w = max(1, int(cell_w))
    cell_h = max(1, int(cell_h))

    if len(s) == 0:
        surf = pygame.Surface((cell_w, cell_h), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 255))
        return surf

    surf = pygame.Surface((cell_w * len(s), cell_h), flags=pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    for i, ch in enumerate(s):
        x = int(i * cell_w)
        pygame.draw.rect(surf, (0, 0, 0, 255), pygame.Rect(x, 0, cell_w, cell_h))
        if ch != " ":
            glyph = font.render(ch, True, (255, 255, 255))
            try:
                glyph = glyph.convert_alpha()
            except Exception:
                pass
            gx, gy = glyph.get_size()
            off_x = max(0, (cell_w - gx) // 2)
            off_y = max(0, (cell_h - gy) // 2)
            surf.blit(glyph, (x + off_x, off_y))

    return surf


def _draw_fullscreen_lines(font: pygame.font.Font, width: int, height: int, lines: list[str]) -> None:
    from OpenGL.GL import (
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
        GL_RGBA,
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
        glPushMatrix,
        glPopMatrix,
        glRasterPos2f,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_UNPACK_ALIGNMENT,
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

    # Simple left-side vertical layout.
    y = 0.60
    dy = 0.06
    for line in lines:
        surf = _render_cell_text(font, line)
        text_data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(0.06, float(y))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        y -= dy

    glDisable(GL_BLEND)

    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def _draw_menu_nav_puzzle(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    target_word: str,
    word_y: dict[str, float],
    reticle_y: float,
) -> None:
    # Draw a simple word list with a moving reticle that approaches the target.
    from OpenGL.GL import (
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
        GL_RGBA,
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
        glPushMatrix,
        glPopMatrix,
        glRasterPos2f,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_UNPACK_ALIGNMENT,
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

    words = ["up", "down", "left", "right"]
    x_word = 0.22
    x_ret = x_word
    y0 = 0.68
    dy = 0.10

    # Draw the list.
    for i, w in enumerate(words):
        y = y0 - float(i) * dy
        surf = _render_cell_text(font, w)
        text_data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(float(x_word), float(y))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Draw the reticle (a caret) that moves onto the current target word.
    # The requirement says: reticle below the word, then move onto it.
    # This now uses a transparent PNG sprite (animated by stage).
    if not hasattr(_draw_menu_nav_puzzle, "_reticle_assets"):
        _draw_menu_nav_puzzle._reticle_assets = reticle_sprite.load_reticle_assets(size_px=64)
        _draw_menu_nav_puzzle._reticle_anim = reticle_sprite.ReticleAnimator()

    assets = _draw_menu_nav_puzzle._reticle_assets
    anim = _draw_menu_nav_puzzle._reticle_anim

    # Consider "on target" once the reticle is close to the target word line.
    on_target = abs(float(reticle_y) - float(word_y.get(target_word, y0))) <= 0.015
    now_s = pygame.time.get_ticks() * 0.001
    stage = anim.update(on_target=bool(on_target), now_s=float(now_s))
    surf = assets.surfaces.get(stage)
    if surf is not None:
        text_data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(float(x_ret), float(reticle_y))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glDisable(GL_BLEND)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def _get_menu_nav(cfg: dict[str, Any]) -> dict[str, Any]:
    nav = cfg.get("menu_nav", None)
    return nav if isinstance(nav, dict) else {}


def _used_menu_nav_inputs(cfg: dict[str, Any]) -> tuple[set[tuple[int, int]], set[int], set[tuple[int, int, int]]]:
    # Axes are tracked as (axis, sign) so up/down can share the same axis
    # with opposite extrema.
    used_axes: set[tuple[int, int]] = set()
    used_buttons: set[int] = set()
    # Hats tracked as (hat_index, x, y) so a single hat can bind multiple directions.
    used_hats: set[tuple[int, int, int]] = set()
    nav = _get_menu_nav(cfg)
    for v in nav.values():
        if not isinstance(v, dict):
            continue
        if v.get("type") == "axis":
            a = v.get("axis")
            s = v.get("sign")
            if isinstance(a, int) and isinstance(s, int) and int(s) in (-1, 1):
                used_axes.add((int(a), int(s)))
        elif v.get("type") == "button":
            b = v.get("button")
            if isinstance(b, int):
                used_buttons.add(int(b))
        elif v.get("type") == "hat":
            h = v.get("hat")
            x = v.get("x")
            y = v.get("y")
            if isinstance(h, int) and isinstance(x, int) and isinstance(y, int):
                xi = int(x)
                yi = int(y)
                if (xi, yi) != (0, 0) and xi in (-1, 0, 1) and yi in (-1, 0, 1):
                    used_hats.add((int(h), xi, yi))
    return used_axes, used_buttons, used_hats


def ensure_menu_navigation_bindings(
    *,
    cfg_path: str = "joystick.json",
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    menu_button: int | None,
    axis_threshold: float = 0.85,
    axis_wake_deadzone: float = 0.25,
    keys: list[str] | None = None,
    force: bool = False,
) -> bool:
    """Interactive puzzle to bind menu navigation controls.

        For each missing key in (up, down, left, right, confirm, cancel), waits for the first
        committed input:
        - Axis extreme: abs(value) >= axis_threshold on a woken axis not already used for that sign.
            Stores {"type":"axis", "axis":<idx>, "sign":(+1|-1)}.
        - Hat direction (D-pad): non-neutral (x,y) on a hat direction not already used.
            Stores {"type":"hat", "hat":<idx>, "x":(-1|0|1), "y":(-1|0|1)}.
        - Button hold: any button not already used (and not menu_button).
            Stores {"type":"button", "button":<idx>}.

    Returns True if cfg was modified.
    """
    cfg = load_or_create_joystick_config(cfg_path)
    nav = _get_menu_nav(cfg)
    changed = False

    if joystick is None:
        return False

    directions = ["up", "down", "left", "right", "confirm", "cancel"]
    wanted = [str(d) for d in (keys or directions)]
    targets = [d for d in wanted if d in directions]
    missing = [d for d in targets if bool(force) or (d not in nav)]
    if not missing:
        return False

    clock = pygame.time.Clock()
    pressed_buttons: set[int] = set()
    last_polled_buttons: set[int] = set()
    last_polled_hats: dict[int, tuple[int, int]] = {}

    # Capture initial axis values so we can require a deliberate "wake" move
    # before accepting an extreme as a binding.
    init_axis: dict[int, float] = {}
    axis_woken: set[int] = set()
    try:
        n_axes = int(joystick.get_numaxes())
        for a in range(max(0, n_axes)):
            try:
                init_axis[a] = float(joystick.get_axis(a))
            except Exception:
                init_axis[a] = 0.0
    except Exception:
        pass

    # Reticle assets/animator for the puzzle (fast confirm).
    if not hasattr(ensure_menu_navigation_bindings, "_reticle_assets"):
        ensure_menu_navigation_bindings._reticle_assets = reticle_sprite.load_reticle_assets(size_px=64)
        ensure_menu_navigation_bindings._reticle_anim = reticle_sprite.ReticleAnimator(lock_delay_s=0.20, ready_delay_s=0.45)

    assets = ensure_menu_navigation_bindings._reticle_assets
    anim = ensure_menu_navigation_bindings._reticle_anim

    def _draw_single_task(*, label: str, ret_x: float, ret_y: float, stage: reticle_sprite.ReticleStage) -> None:
        from OpenGL.GL import (
            GL_DEPTH_TEST,
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_SRC_ALPHA,
            GL_UNSIGNED_BYTE,
            GL_RGBA,
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
            glPushMatrix,
            glPopMatrix,
            glRasterPos2f,
            GL_MODELVIEW,
            GL_PROJECTION,
            GL_UNPACK_ALIGNMENT,
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

        # Centered label.
        x_lbl = 0.44
        y_lbl = 0.52
        surf_lbl = _render_cell_text(font, label)
        lbl_data = pygame.image.tostring(surf_lbl, "RGBA", True)
        glRasterPos2f(float(x_lbl), float(y_lbl))
        glDrawPixels(surf_lbl.get_width(), surf_lbl.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, lbl_data)

        # Reticle stage.
        surf_ret = assets.surfaces.get(stage)
        if surf_ret is not None:
            ret_data = pygame.image.tostring(surf_ret, "RGBA", True)
            glRasterPos2f(float(ret_x), float(ret_y))
            glDrawPixels(surf_ret.get_width(), surf_ret.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, ret_data)

        glDisable(GL_BLEND)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    for d in missing:
        used_axes, used_buttons, used_hats = _used_menu_nav_inputs(cfg)
        if menu_button is not None:
            used_buttons.add(int(menu_button))

        label = f"menu {d}"

        # Centered label anchor (match _draw_single_task).
        x_lbl = 0.44
        y_lbl = 0.52

        # Reticle start offsets by direction (as requested):
        # - below up
        # - above down
        # - right of left
        # - left of right
        if d == "up":
            start = (x_lbl, y_lbl - 0.16)
        elif d == "down":
            start = (x_lbl, y_lbl + 0.16)
        elif d == "left":
            start = (x_lbl + 0.22, y_lbl)
        else:  # right
            if d == "right":
                start = (x_lbl - 0.22, y_lbl)
            else:
                # confirm/cancel: start just below
                start = (x_lbl, y_lbl - 0.16)

        target = (x_lbl, y_lbl)
        ret_x, ret_y = float(start[0]), float(start[1])

        candidate: dict[str, Any] | None = None
        candidate_active = False
        candidate_start_s: float | None = None

        latest_axis: dict[int, float] = {}
        for a, v0 in init_axis.items():
            latest_axis[a] = float(v0)

        # Confirm behavior:
        # - Axis: confirm once held at extreme long enough to reach READY.
        # - Button: move while held; confirm on release once at least LOCKED.
        while True:
            now_s = pygame.time.get_ticks() * 0.001

            # Poll buttons/hats so we catch devices that don't emit events reliably
            # (or where the D-pad is exposed as a hat).
            released_button: int | None = None
            try:
                n_buttons = int(joystick.get_numbuttons())
                polled_buttons: set[int] = set()
                for b in range(max(0, n_buttons)):
                    try:
                        if int(joystick.get_button(b)) != 0:
                            polled_buttons.add(int(b))
                    except Exception:
                        pass

                # Derive a release edge even if no JOYBUTTONUP arrives.
                released = list(last_polled_buttons - polled_buttons)
                if released:
                    released_button = int(released[0])

                last_polled_buttons = polled_buttons
                pressed_buttons = polled_buttons
            except Exception:
                pass

            polled_hats: dict[int, tuple[int, int]] = {}
            try:
                n_hats = int(joystick.get_numhats())
                for h in range(max(0, n_hats)):
                    try:
                        v = joystick.get_hat(h)
                        polled_hats[int(h)] = (int(v[0]), int(v[1]))
                    except Exception:
                        polled_hats[int(h)] = (0, 0)
            except Exception:
                pass

            # Poll axes so we work even if the driver drops events.
            try:
                n_axes = int(joystick.get_numaxes())
                for a in range(max(0, n_axes)):
                    try:
                        latest_axis[a] = float(joystick.get_axis(a))
                    except Exception:
                        pass
            except Exception:
                pass

            # Update axis wake status.
            for a, v in list(latest_axis.items()):
                v_init = float(init_axis.get(a, 0.0))
                if abs(float(v) - v_init) >= float(axis_wake_deadzone):
                    axis_woken.add(int(a))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return changed

                if event.type == pygame.JOYAXISMOTION:
                    try:
                        latest_axis[int(event.axis)] = float(event.value)
                    except Exception:
                        pass

                if event.type == pygame.JOYBUTTONDOWN:
                    btn = int(getattr(event, "button", -1))
                    if btn >= 0:
                        pressed_buttons.add(btn)

                if event.type == pygame.JOYBUTTONUP:
                    btn = int(getattr(event, "button", -1))
                    if btn in pressed_buttons:
                        pressed_buttons.discard(btn)
                    released_button = btn

            # Candidate selection if none chosen yet.
            if candidate is None:
                # Prefer axis extreme if any woken axis is at an unused extreme.
                for a, v in latest_axis.items():
                    if int(a) not in axis_woken:
                        continue
                    if abs(float(v)) < float(axis_threshold):
                        continue
                    sign = 1 if float(v) > 0.0 else -1
                    if (int(a), int(sign)) in used_axes:
                        continue
                    candidate = {"type": "axis", "axis": int(a), "sign": int(sign)}
                    candidate_active = True
                    candidate_start_s = float(now_s)
                    break

                if candidate is None:
                    # Next prefer any non-neutral hat direction.
                    for h, (hx, hy) in polled_hats.items():
                        if (int(hx), int(hy)) == (0, 0):
                            continue
                        if (int(h), int(hx), int(hy)) in used_hats:
                            continue
                        candidate = {"type": "hat", "hat": int(h), "x": int(hx), "y": int(hy)}
                        candidate_active = True
                        candidate_start_s = float(now_s)
                        break

                if candidate is None:
                    # Otherwise accept a button press (excluding already used + menu button).
                    for btn in list(pressed_buttons):
                        if btn in used_buttons:
                            continue
                        candidate = {"type": "button", "button": int(btn)}
                        candidate_active = True
                        candidate_start_s = float(now_s)
                        break

            # Candidate active state tracking.
            if candidate is not None:
                if candidate.get("type") == "axis":
                    a = int(candidate.get("axis", -1))
                    s = int(candidate.get("sign", 0))
                    v = float(latest_axis.get(a, 0.0))
                    candidate_active = (abs(v) >= float(axis_threshold)) and ((1 if v > 0.0 else -1) == s)
                elif candidate.get("type") == "hat":
                    h = int(candidate.get("hat", -1))
                    cx = int(candidate.get("x", 0))
                    cy = int(candidate.get("y", 0))
                    hx, hy = polled_hats.get(h, (0, 0))
                    candidate_active = (int(hx), int(hy)) == (int(cx), int(cy))
                else:
                    btn = int(candidate.get("button", -1))
                    candidate_active = btn in pressed_buttons

            # Smooth reticle movement:
            # - While candidate active, move to target.
            # - Otherwise ease back to the hint start position.
            desired = target if candidate_active else start
            ret_x += (float(desired[0]) - float(ret_x)) * 0.20
            ret_y += (float(desired[1]) - float(ret_y)) * 0.20

            # "On target" when close to the word center.
            on_target = (abs(float(ret_x) - float(target[0])) <= 0.015) and (abs(float(ret_y) - float(target[1])) <= 0.020)

            # Drive reticle stage; confirmation depends on stage.
            stage = anim.update(on_target=bool(on_target and candidate_active), now_s=float(now_s))

            _draw_single_task(label=label, ret_x=float(ret_x), ret_y=float(ret_y), stage=stage)
            pygame.display.flip()
            clock.tick(60)

            # Confirm
            if candidate is None:
                continue

            if candidate.get("type") == "axis":
                # Confirm on READY stage while still held.
                if candidate_active and stage == reticle_sprite.ReticleStage.READY:
                    bound = dict(candidate)
                    break
                # If they let go before confirmation, reset candidate.
                if not candidate_active:
                    candidate = None
                    candidate_start_s = None
                    anim._on_since = None  # reset stage progression
            elif candidate.get("type") == "hat":
                # Confirm on READY stage while still held.
                if candidate_active and stage == reticle_sprite.ReticleStage.READY:
                    bound = dict(candidate)
                    break
                if not candidate_active:
                    candidate = None
                    candidate_start_s = None
                    anim._on_since = None
            else:
                # Confirm while the button is held once we reach READY.
                if candidate_active and stage == reticle_sprite.ReticleStage.READY:
                    bound = dict(candidate)
                    break

                # If they released before confirmation, reset candidate.
                btn = int(candidate.get("button", -1))
                if released_button is not None and int(released_button) == btn:
                    candidate = None
                    candidate_start_s = None
                    anim._on_since = None

        # Persist this binding.
        cfg = load_or_create_joystick_config(cfg_path)
        if not isinstance(cfg.get("menu_nav", None), dict):
            cfg["menu_nav"] = {}
        cfg["menu_nav"][d] = bound
        save_joystick_config(cfg, cfg_path)
        changed = True

    return changed


def _poll_joystick_snapshot(joystick: pygame.joystick.Joystick) -> tuple[dict[int, float], set[int], dict[int, tuple[int, int]]]:
    axes: dict[int, float] = {}
    buttons: set[int] = set()
    hats: dict[int, tuple[int, int]] = {}
    try:
        n_axes = int(joystick.get_numaxes())
        for a in range(max(0, n_axes)):
            try:
                axes[int(a)] = float(joystick.get_axis(a))
            except Exception:
                pass
    except Exception:
        pass

    try:
        n_buttons = int(joystick.get_numbuttons())
        for b in range(max(0, n_buttons)):
            try:
                if int(joystick.get_button(b)) != 0:
                    buttons.add(int(b))
            except Exception:
                pass
    except Exception:
        pass

    try:
        n_hats = int(joystick.get_numhats())
        for h in range(max(0, n_hats)):
            try:
                v = joystick.get_hat(h)
                hats[int(h)] = (int(v[0]), int(v[1]))
            except Exception:
                hats[int(h)] = (0, 0)
    except Exception:
        pass

    return axes, buttons, hats


def _nav_edge(
    binding: dict[str, Any] | None,
    *,
    axes_now: dict[int, float],
    axes_prev: dict[int, float],
    buttons_now: set[int],
    buttons_prev: set[int],
    hats_now: dict[int, tuple[int, int]],
    hats_prev: dict[int, tuple[int, int]],
    axis_threshold: float = 0.70,
) -> bool:
    if not isinstance(binding, dict):
        return False
    t = binding.get("type")
    if t == "button":
        b = binding.get("button")
        if not isinstance(b, int):
            return False
        return (int(b) in buttons_now) and (int(b) not in buttons_prev)
    if t == "hat":
        h = binding.get("hat")
        x = binding.get("x")
        y = binding.get("y")
        if not (isinstance(h, int) and isinstance(x, int) and isinstance(y, int)):
            return False
        now = hats_now.get(int(h), (0, 0))
        prev = hats_prev.get(int(h), (0, 0))
        want = (int(x), int(y))
        return (now == want) and (prev != want)
    if t == "axis":
        a = binding.get("axis")
        s = binding.get("sign")
        if not (isinstance(a, int) and isinstance(s, int) and int(s) in (-1, 1)):
            return False
        v_now = float(axes_now.get(int(a), 0.0))
        v_prev = float(axes_prev.get(int(a), 0.0))
        # edge when crossing threshold in the desired sign
        if int(s) > 0:
            return (v_now >= float(axis_threshold)) and (v_prev < float(axis_threshold))
        return (v_now <= -float(axis_threshold)) and (v_prev > -float(axis_threshold))
    return False


def _nav_is_held(
    binding: dict[str, Any] | None,
    *,
    axes_now: dict[int, float],
    buttons_now: set[int],
    hats_now: dict[int, tuple[int, int]],
    axis_threshold: float = 0.70,
) -> bool:
    if not isinstance(binding, dict):
        return False
    t = binding.get("type")
    if t == "button":
        b = binding.get("button")
        return isinstance(b, int) and int(b) in buttons_now
    if t == "hat":
        h = binding.get("hat")
        x = binding.get("x")
        y = binding.get("y")
        if not (isinstance(h, int) and isinstance(x, int) and isinstance(y, int)):
            return False
        return hats_now.get(int(h), (0, 0)) == (int(x), int(y))
    if t == "axis":
        a = binding.get("axis")
        s = binding.get("sign")
        if not (isinstance(a, int) and isinstance(s, int) and int(s) in (-1, 1)):
            return False
        v_now = float(axes_now.get(int(a), 0.0))
        return (v_now >= float(axis_threshold)) if int(s) > 0 else (v_now <= -float(axis_threshold))
    return False


def _draw_menu_list(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    title: str,
    items: list[str],
    selected_idx: int,
    ret_x: float,
    ret_y: float,
    stage: reticle_sprite.ReticleStage,
) -> None:
    from OpenGL.GL import (
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
        GL_RGBA,
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
        glPushMatrix,
        glPopMatrix,
        glRasterPos2f,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_UNPACK_ALIGNMENT,
    )

    if not hasattr(_draw_menu_list, "_reticle_assets"):
        _draw_menu_list._reticle_assets = reticle_sprite.load_reticle_assets(size_px=64)
    assets = _draw_menu_list._reticle_assets

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

    # Title
    surf_t = _render_cell_text(font, title)
    t_data = pygame.image.tostring(surf_t, "RGBA", True)
    glRasterPos2f(0.06, 0.78)
    glDrawPixels(surf_t.get_width(), surf_t.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, t_data)

    # Items
    y = 0.66
    dy = 0.07
    for it in items:
        surf = _render_cell_text(font, it)
        data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(0.10, float(y))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)
        y -= dy

    # Reticle
    surf_ret = assets.surfaces.get(stage)
    if surf_ret is not None:
        r_data = pygame.image.tostring(surf_ret, "RGBA", True)
        glRasterPos2f(float(ret_x), float(ret_y))
        glDrawPixels(surf_ret.get_width(), surf_ret.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, r_data)

    glDisable(GL_BLEND)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def _ctypes_struct_to_dict(s: ctypes.Structure) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = getattr(s.__class__, "_fields_", [])
    for name, _ctype in fields:
        try:
            v = getattr(s, name)
        except Exception:
            continue
        # ctypes scalars convert cleanly via int()/float() in most cases
        if isinstance(v, (int, bool)):
            out[str(name)] = int(v)
        else:
            try:
                out[str(name)] = float(v)
            except Exception:
                try:
                    out[str(name)] = int(v)
                except Exception:
                    pass
    return out


def _apply_dict_to_ctypes_struct(s: ctypes.Structure, data: dict[str, Any]) -> None:
    fields = getattr(s.__class__, "_fields_", [])
    for name, ctype in fields:
        key = str(name)
        if key not in data:
            continue
        try:
            raw = data.get(key)
            # Heuristic: treat float fields as float, otherwise int.
            if ctype in (ctypes.c_float, ctypes.c_double):
                setattr(s, key, float(raw))
            else:
                setattr(s, key, int(raw))
        except Exception:
            pass


def _load_persisted_block(path: str, key: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        blk = data.get(key)
        return blk if isinstance(blk, dict) else {}
    except Exception:
        return {}


def _save_persisted_block(path: str, key: str, block: dict[str, Any]) -> None:
    # Reads existing file if present; always writes via atomic writer (creates .bak).
    root: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                root = data
        except Exception:
            root = {}
    root[str(key)] = dict(block)
    _atomic_write_json(path, root)


def run_ctypes_struct_editor(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick,
    menu_button: int | None,
    struct_obj: ctypes.Structure,
    title: str,
    persist_path: str,
    persist_key: str,
    field_specs: dict[str, dict[str, Any]] | None = None,
    persist_on_change: bool = False,
) -> bool:
    """Edit a ctypes.Structure using menu-nav bindings.

    Controls:
    - up/down: move field selection
    - left/right: decrement/increment selected field
    - cancel: exit back to menu (auto-saves if changed)
    - menu button: exits to start
    """
    if not isinstance(struct_obj, ctypes.Structure):
        return False

    cfg = load_or_create_joystick_config("joystick.json")
    nav = _get_menu_nav(cfg)
    b_up = nav.get("up")
    b_down = nav.get("down")
    b_left = nav.get("left")
    b_right = nav.get("right")
    b_cancel = nav.get("cancel")

    # Load persisted values (best-effort).
    persisted = _load_persisted_block(str(persist_path), str(persist_key))
    if persisted:
        _apply_dict_to_ctypes_struct(struct_obj, persisted)

    fields = [(str(n), t) for (n, t) in getattr(struct_obj.__class__, "_fields_", [])]
    if not fields:
        return False

    def _fmt_value(name: str, ctype: Any) -> str:
        return _format_ctypes_struct_field_value(struct_obj=struct_obj, field_name=name, field_ctype=ctype, field_specs=field_specs)

    def _fmt_name(name: str) -> str:
        return _ctypes_struct_field_label(field_name=name, field_specs=field_specs)

    import math

    clock = pygame.time.Clock()
    sel = 0
    axes_prev: dict[int, float] = {}
    buttons_prev: set[int] = set()
    hats_prev: dict[int, tuple[int, int]] = {}

    ret_x = 0.04
    ret_y = 0.66

    # Simple key-repeat for left/right.
    last_lr_time = 0.0
    lr_repeat_s = 0.07
    lr_first_delay_s = 0.22
    lr_held_since: float | None = None
    lr_dir = 0

    changed = False
    start_snapshot = _ctypes_struct_to_dict(struct_obj)
    last_saved_snapshot = dict(start_snapshot)

    def _sel_y(idx: int) -> float:
        return float(0.66 - 0.07 * int(idx))

    max_visible = 10

    while True:
        exit_now = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_now = True
                break
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                # Treat as cancel.
                return changed
            if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                # Exit to start.
                exit_now = True
                break

        if exit_now:
            break

        axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

        if _nav_edge(
            b_up,
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            sel = (sel - 1) % max(1, len(fields))
            lr_held_since = None
            lr_dir = 0

        if _nav_edge(
            b_down,
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            sel = (sel + 1) % max(1, len(fields))
            lr_held_since = None
            lr_dir = 0

        if _nav_edge(
            b_cancel,
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            break

        # Left/right adjustments with repeat.
        left_held = _nav_is_held(b_left, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)
        right_held = _nav_is_held(b_right, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)

        now_s = pygame.time.get_ticks() * 0.001
        desired_dir = (-1 if left_held and not right_held else (1 if right_held and not left_held else 0))
        if desired_dir == 0:
            lr_held_since = None
            lr_dir = 0
        else:
            if lr_dir != desired_dir:
                lr_dir = desired_dir
                lr_held_since = float(now_s)
                last_lr_time = 0.0
                # Apply an immediate step on direction change.
                fname, ftype = fields[int(sel)]
                _step_ctypes_struct_field(
                    struct_obj=struct_obj,
                    field_name=fname,
                    field_ctype=ftype,
                    direction=int(lr_dir),
                    field_specs=field_specs,
                )
                if bool(persist_on_change):
                    try:
                        snap_now = _ctypes_struct_to_dict(struct_obj)
                        if snap_now != last_saved_snapshot:
                            _save_persisted_block(str(persist_path), str(persist_key), snap_now)
                            last_saved_snapshot = dict(snap_now)
                    except Exception:
                        pass
            else:
                held_for = float(now_s) - float(lr_held_since or now_s)
                if held_for >= float(lr_first_delay_s):
                    if (float(now_s) - float(last_lr_time)) >= float(lr_repeat_s):
                        fname, ftype = fields[int(sel)]
                        _step_ctypes_struct_field(
                            struct_obj=struct_obj,
                            field_name=fname,
                            field_ctype=ftype,
                            direction=int(lr_dir),
                            field_specs=field_specs,
                        )
                        last_lr_time = float(now_s)
                        if bool(persist_on_change):
                            try:
                                snap_now = _ctypes_struct_to_dict(struct_obj)
                                if snap_now != last_saved_snapshot:
                                    _save_persisted_block(str(persist_path), str(persist_key), snap_now)
                                    last_saved_snapshot = dict(snap_now)
                            except Exception:
                                pass

        # Detect changes.
        if not changed:
            cur_snapshot = _ctypes_struct_to_dict(struct_obj)
            changed = cur_snapshot != start_snapshot

        # Draw
        lines: list[str] = []
        for i, (fname, ftype) in enumerate(fields):
            prefix = "> " if int(i) == int(sel) else "  "
            lines.append(f"{prefix}{_fmt_name(fname)}: {_fmt_value(fname, ftype)}")

        first_idx = 0
        if int(len(lines)) > int(max_visible):
            half = int(max_visible // 2)
            first_idx = int(sel) - int(half)
            first_idx = max(0, min(int(first_idx), int(len(lines) - max_visible)))
        visible_sel = int(sel) - int(first_idx)
        lines_vis = lines[int(first_idx) : int(first_idx) + int(max_visible)] if int(len(lines)) > int(max_visible) else lines

        ret_y_des = _sel_y(visible_sel)
        ret_y += (float(ret_y_des) - float(ret_y)) * 0.35
        _draw_menu_list(
            font=font,
            width=int(width),
            height=int(height),
            title=str(title),
            items=lines_vis,
            selected_idx=int(sel),
            ret_x=float(ret_x),
            ret_y=float(ret_y),
            stage=reticle_sprite.ReticleStage.IDLE,
        )
        pygame.display.flip()
        clock.tick(60)

        axes_prev, buttons_prev, hats_prev = axes_now, set(buttons_now), dict(hats_now)

    # Persist on exit if changed.
    if changed:
        _save_persisted_block(str(persist_path), str(persist_key), _ctypes_struct_to_dict(struct_obj))
    return changed


def draw_pip_lines(
    font: pygame.font.Font,
    width: int,
    height: int,
    lines: list[str],
    *,
    x: float = 0.02,
    y_top: float = 0.96,
    dy: float = 0.03,
    max_lines: int = 14,
    respect_viewport: bool = False,
    draw_box: bool = True,
) -> None:
    """Draw a small text PIP in screen space (no clears).

    Coordinates are normalized [0..1] like the other HUD helpers.
    """
    from OpenGL.GL import (
        GL_LINE_STRIP,
        GL_TRIANGLES,
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
        GL_RGBA,
        glBegin,
        glBlendFunc,
        glColor4f,
        glDisable,
        glDrawPixels,
        glEnable,
        glEnd,
        glIsEnabled,
        glLoadIdentity,
        glLineWidth,
        glMatrixMode,
        glOrtho,
        glPixelStorei,
        glPushMatrix,
        glPopMatrix,
        glRasterPos2f,
        glVertex3f,
        glDepthMask,
        glGetBooleanv,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_UNPACK_ALIGNMENT,
        GL_DEPTH_WRITEMASK,
    )

    if not lines:
        return

    prev_vp = None
    prev_scissor = None
    scissor_was_enabled = False

    if not bool(respect_viewport):
        # Some overlays (minimap/gimbal/etc.) temporarily change viewport/scissor.
        # Ensure this HUD widget always draws in the full window.
        try:
            from OpenGL.GL import (
                glGetIntegerv,
                glViewport,
                glScissor,
                GL_VIEWPORT,
                GL_SCISSOR_TEST,
                GL_SCISSOR_BOX,
            )
            prev_vp = glGetIntegerv(GL_VIEWPORT)
            scissor_was_enabled = bool(glIsEnabled(GL_SCISSOR_TEST))
            prev_scissor = glGetIntegerv(GL_SCISSOR_BOX)
            glViewport(0, 0, int(width), int(height))
            if scissor_was_enabled:
                glDisable(GL_SCISSOR_TEST)
        except Exception:
            prev_vp = None
            prev_scissor = None
            scissor_was_enabled = False

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    depth_mask_prev = bool(glGetBooleanv(GL_DEPTH_WRITEMASK))
    if depth_was_enabled:
        glDisable(GL_DEPTH_TEST)
    glDepthMask(False)

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

    # Clamp the first line so it is fully visible.
    try:
        line_h_px = int(max(1, int(font.get_linesize())))
    except Exception:
        line_h_px = 16
    line_h_ndc = float(line_h_px) / float(max(1, int(height)))
    y0 = float(y_top) - float(line_h_ndc)

    draw_lines = [str(s)[:220] for s in lines[: max(0, int(max_lines))]]
    n = len(draw_lines)

    if bool(draw_box):
        # Backplate sized to text content.
        try:
            pad_px = 8
            pad_x = float(pad_px) / float(max(1, int(width)))
            pad_y = float(pad_px) / float(max(1, int(height)))
            max_w_px = 0
            for s in draw_lines:
                try:
                    w_px, _h_px = font.size(str(s))
                    max_w_px = max(int(max_w_px), int(w_px))
                except Exception:
                    pass
            text_w = float(max_w_px) / float(max(1, int(width)))
            box_left = float(x) - pad_x
            box_right = float(x) + text_w + pad_x
            box_top = float(y0) + float(line_h_ndc) + pad_y
            box_bottom = float(y0) - float(max(0, n - 1)) * float(dy) - pad_y
            box_left = float(max(0.0, box_left))
            box_right = float(min(1.0, box_right))
            box_bottom = float(max(0.0, box_bottom))
            box_top = float(min(1.0, box_top))

            glColor4f(0.0, 0.0, 0.0, 0.35)
            glBegin(GL_TRIANGLES)
            glVertex3f(box_left, box_bottom, 0.0)
            glVertex3f(box_right, box_bottom, 0.0)
            glVertex3f(box_right, box_top, 0.0)
            glVertex3f(box_left, box_bottom, 0.0)
            glVertex3f(box_right, box_top, 0.0)
            glVertex3f(box_left, box_top, 0.0)
            glEnd()

            glLineWidth(1.0)
            glColor4f(1.0, 1.0, 1.0, 0.25)
            glBegin(GL_LINE_STRIP)
            glVertex3f(box_left, box_bottom, 0.0)
            glVertex3f(box_right, box_bottom, 0.0)
            glVertex3f(box_right, box_top, 0.0)
            glVertex3f(box_left, box_top, 0.0)
            glVertex3f(box_left, box_bottom, 0.0)
            glEnd()
        except Exception:
            pass

    y = float(y0)
    for line in draw_lines:
        surf = _render_cell_text(font, str(line))
        text_data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(float(x), float(y))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        y -= float(dy)

    glDisable(GL_BLEND)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glDepthMask(bool(depth_mask_prev))
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    if not bool(respect_viewport):
        # Restore scissor/viewport.
        try:
            from OpenGL.GL import glViewport, glScissor, glEnable, GL_SCISSOR_TEST

            if prev_vp is not None and len(prev_vp) >= 4:
                glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))
            if scissor_was_enabled and prev_scissor is not None and len(prev_scissor) >= 4:
                glEnable(GL_SCISSOR_TEST)
                glScissor(int(prev_scissor[0]), int(prev_scissor[1]), int(prev_scissor[2]), int(prev_scissor[3]))
        except Exception:
            pass


class CtypesStructEditorPip:
    """Non-blocking ctypes struct editor drawn as a small HUD PIP.

    Intended to be ticked each frame from the main loop.
    Navigation uses the existing `menu_nav` bindings in `joystick.json`.
    """

    def __init__(
        self,
        *,
        struct_obj: ctypes.Structure,
        title: str,
        persist_path: str,
        persist_key: str,
        field_specs: dict[str, dict[str, Any]] | None = None,
        allow_cancel_close: bool = True,
        persist_on_change: bool = False,
    ) -> None:
        if not isinstance(struct_obj, ctypes.Structure):
            raise TypeError("struct_obj must be a ctypes.Structure")
        self.struct_obj = struct_obj
        self.title = str(title)
        self.persist_path = str(persist_path)
        self.persist_key = str(persist_key)
        self.field_specs = field_specs
        self.allow_cancel_close = bool(allow_cancel_close)
        self.persist_on_change = bool(persist_on_change)

        self.active = False
        self.sel = 0
        self._axes_prev: dict[int, float] = {}
        self._buttons_prev: set[int] = set()
        self._hats_prev: dict[int, tuple[int, int]] = {}

        self._lr_held_since: float | None = None
        self._lr_dir = 0
        self._last_lr_time = 0.0
        self._lr_repeat_s = 0.07
        self._lr_first_delay_s = 0.22

        self._start_snapshot: dict[str, Any] | None = None

        # Load persisted values once at construction.
        persisted = _load_persisted_block(self.persist_path, self.persist_key)
        if persisted:
            _apply_dict_to_ctypes_struct(self.struct_obj, persisted)

    def _fields(self) -> list[tuple[str, Any]]:
        return [(str(n), t) for (n, t) in getattr(self.struct_obj.__class__, "_fields_", [])]

    def open(self) -> None:
        self.active = True
        self._lr_held_since = None
        self._lr_dir = 0
        self._last_lr_time = 0.0
        self._start_snapshot = _ctypes_struct_to_dict(self.struct_obj)

    def close(self) -> bool:
        """Close the PIP; returns True if persisted a change."""
        self.active = False
        changed = False
        if self._start_snapshot is not None:
            cur = _ctypes_struct_to_dict(self.struct_obj)
            changed = cur != self._start_snapshot
        if changed:
            _save_persisted_block(self.persist_path, self.persist_key, _ctypes_struct_to_dict(self.struct_obj))
        self._start_snapshot = None
        self._lr_held_since = None
        self._lr_dir = 0
        return bool(changed)

    def toggle(self) -> bool:
        if self.active:
            return self.close()
        self.open()
        return False

    def tick(self, *, joystick: pygame.joystick.Joystick) -> bool:
        """Advance the PIP state by one frame.

        Returns True if the struct changed this tick.
        """
        fields = self._fields()
        if not fields:
            return False
        self.sel = int(self.sel) % len(fields)

        cfg = load_or_create_joystick_config("joystick.json")
        nav = _get_menu_nav(cfg)
        b_up = nav.get("up")
        b_down = nav.get("down")
        b_left = nav.get("left")
        b_right = nav.get("right")
        b_cancel = nav.get("cancel")

        axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)
        changed = False

        if self.active:
            if _nav_edge(
                b_up,
                axes_now=axes_now,
                axes_prev=self._axes_prev,
                buttons_now=buttons_now,
                buttons_prev=self._buttons_prev,
                hats_now=hats_now,
                hats_prev=self._hats_prev,
            ):
                self.sel = (int(self.sel) - 1) % len(fields)
                self._lr_held_since = None
                self._lr_dir = 0

            if _nav_edge(
                b_down,
                axes_now=axes_now,
                axes_prev=self._axes_prev,
                buttons_now=buttons_now,
                buttons_prev=self._buttons_prev,
                hats_now=hats_now,
                hats_prev=self._hats_prev,
            ):
                self.sel = (int(self.sel) + 1) % len(fields)
                self._lr_held_since = None
                self._lr_dir = 0

            if _nav_edge(
                b_cancel,
                axes_now=axes_now,
                axes_prev=self._axes_prev,
                buttons_now=buttons_now,
                buttons_prev=self._buttons_prev,
                hats_now=hats_now,
                hats_prev=self._hats_prev,
            ):
                if self.allow_cancel_close:
                    changed = bool(self.close())

            left_held = _nav_is_held(b_left, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)
            right_held = _nav_is_held(b_right, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)
            now_s = pygame.time.get_ticks() * 0.001
            desired_dir = (-1 if left_held and not right_held else (1 if right_held and not left_held else 0))

            if desired_dir == 0:
                self._lr_held_since = None
                self._lr_dir = 0
            else:
                if self._lr_dir != desired_dir:
                    self._lr_dir = int(desired_dir)
                    self._lr_held_since = float(now_s)
                    self._last_lr_time = 0.0
                    fname, ftype = fields[int(self.sel)]
                    _step_ctypes_struct_field(
                        struct_obj=self.struct_obj,
                        field_name=fname,
                        field_ctype=ftype,
                        direction=int(self._lr_dir),
                        field_specs=self.field_specs,
                    )
                    changed = True
                else:
                    held_for = float(now_s) - float(self._lr_held_since or now_s)
                    if held_for >= float(self._lr_first_delay_s):
                        if (float(now_s) - float(self._last_lr_time)) >= float(self._lr_repeat_s):
                            fname, ftype = fields[int(self.sel)]
                            _step_ctypes_struct_field(
                                struct_obj=self.struct_obj,
                                field_name=fname,
                                field_ctype=ftype,
                                direction=int(self._lr_dir),
                                field_specs=self.field_specs,
                            )
                            self._last_lr_time = float(now_s)
                            changed = True

            if changed and self.persist_on_change:
                try:
                    _save_persisted_block(self.persist_path, self.persist_key, _ctypes_struct_to_dict(self.struct_obj))
                    if self._start_snapshot is not None:
                        self._start_snapshot = _ctypes_struct_to_dict(self.struct_obj)
                except Exception:
                    pass

        self._axes_prev, self._buttons_prev, self._hats_prev = axes_now, set(buttons_now), dict(hats_now)
        return bool(changed)

    def draw(self, *, font: pygame.font.Font, width: int, height: int, respect_viewport: bool = False, draw_box: bool = True) -> None:
        fields = self._fields()
        if not fields:
            return

        # Compact window around current selection.
        # Auto-fit line count to the current draw height so it works both fullscreen
        # and inside small HUD viewports.
        try:
            line_px = int(max(1, int(font.get_linesize())))
        except Exception:
            line_px = 16
        gap_px = 4
        max_lines_fit = int(max(2, (int(height) - 4) // max(1, (line_px + gap_px))))
        n_show = int(max(3, min(8, max_lines_fit - 1)))
        sel = int(self.sel) % len(fields)
        start = max(0, min(sel - n_show // 2, max(0, len(fields) - n_show)))
        end = min(len(fields), start + n_show)

        lines: list[str] = []
        hdr = f"[{self.title}]"
        lines.append(hdr)

        def _fmt_value(name: str, ctype: Any) -> str:
            return _format_ctypes_struct_field_value(struct_obj=self.struct_obj, field_name=name, field_ctype=ctype, field_specs=self.field_specs)

        def _fmt_name(name: str) -> str:
            return _ctypes_struct_field_label(field_name=name, field_specs=self.field_specs)

        for i in range(start, end):
            fname, ftype = fields[i]
            prefix = "> " if i == sel else "  "
            lines.append(f"{prefix}{_fmt_name(fname)}: {_fmt_value(fname, ftype)}")

        # Scale spacing with the current draw height so PIP remains readable.
        dy_ndc = float(max(0.03, float(line_px + gap_px) / float(max(1, int(height)))))
        y_top = 1.0 - (2.0 / float(max(1, int(height))))

        draw_pip_lines(
            font,
            int(width),
            int(height),
            lines,
            x=0.02,
            y_top=float(y_top),
            dy=float(dy_ndc),
            max_lines=1 + n_show,
            respect_viewport=bool(respect_viewport),
            draw_box=bool(draw_box),
        )


def _step_ctypes_struct_field(
    *,
    struct_obj: ctypes.Structure,
    field_name: str,
    field_ctype: Any,
    direction: int,
    field_specs: dict[str, dict[str, Any]] | None,
) -> None:
    # Shared stepping helper used by both modal editor and PIP.
    import math

    spec = (field_specs or {}).get(str(field_name), {}) if isinstance(field_specs, dict) else {}

    # Enum/choice fields (integer-like only): cycle through discrete options.
    choices = _normalize_choice_spec(spec.get("choices", None))
    if choices is not None and field_ctype not in (ctypes.c_float, ctypes.c_double):
        try:
            cur_i = int(getattr(struct_obj, field_name))
        except Exception:
            cur_i = 0
        idx = 0
        for i, (val, _lbl) in enumerate(choices):
            if int(val) == int(cur_i):
                idx = int(i)
                break
        n = len(choices)
        if n <= 0:
            return
        wrap = bool(spec.get("wrap", True))
        if wrap:
            idx = (idx + int(direction)) % n
        else:
            idx = max(0, min(n - 1, idx + int(direction)))
        try:
            setattr(struct_obj, field_name, int(choices[idx][0]))
        except Exception:
            pass
        return
    mode = str(spec.get("mode", "lin"))
    step = float(spec.get("step", 1.0 if field_ctype not in (ctypes.c_float, ctypes.c_double) else 0.02))
    vmin = spec.get("min", None)
    vmax = spec.get("max", None)

    try:
        cur = getattr(struct_obj, field_name)
    except Exception:
        return

    if field_ctype in (ctypes.c_float, ctypes.c_double):
        cur_f = float(cur)
        if mode == "exp":
            new_v = cur_f * float(math.exp(step * float(direction)))
        else:
            new_v = cur_f + float(direction) * step
        if vmin is not None:
            try:
                new_v = max(float(vmin), float(new_v))
            except Exception:
                pass
        if vmax is not None:
            try:
                new_v = min(float(vmax), float(new_v))
            except Exception:
                pass
        try:
            setattr(struct_obj, field_name, float(new_v))
        except Exception:
            pass
        return

    try:
        cur_i = int(cur)
    except Exception:
        return
    new_i = int(cur_i + int(direction) * int(max(1, round(step))))
    if vmin is not None:
        try:
            new_i = max(int(vmin), int(new_i))
        except Exception:
            pass
    if vmax is not None:
        try:
            new_i = min(int(vmax), int(new_i))
        except Exception:
            pass
    try:
        setattr(struct_obj, field_name, int(new_i))
    except Exception:
        pass


def _normalize_choice_spec(raw: Any) -> list[tuple[int, str]] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        out: list[tuple[int, str]] = []
        for k, v in raw.items():
            try:
                out.append((int(k), str(v)))
            except Exception:
                pass
        return out if out else None
    if isinstance(raw, list):
        if all(isinstance(x, str) for x in raw):
            return [(int(i), str(lbl)) for i, lbl in enumerate(raw)]
        out2: list[tuple[int, str]] = []
        for x in raw:
            if isinstance(x, dict):
                try:
                    out2.append((int(x.get("value")), str(x.get("label"))))
                except Exception:
                    pass
        return out2 if out2 else None
    return None


def _ctypes_struct_field_label(*, field_name: str, field_specs: dict[str, dict[str, Any]] | None) -> str:
    spec = (field_specs or {}).get(str(field_name), {}) if isinstance(field_specs, dict) else {}
    try:
        lbl = spec.get("label", None)
        if lbl:
            return str(lbl)
    except Exception:
        pass
    return str(field_name)


def _format_ctypes_struct_field_value(
    *,
    struct_obj: ctypes.Structure,
    field_name: str,
    field_ctype: Any,
    field_specs: dict[str, dict[str, Any]] | None,
) -> str:
    try:
        v = getattr(struct_obj, field_name)
    except Exception:
        return "?"

    spec = (field_specs or {}).get(str(field_name), {}) if isinstance(field_specs, dict) else {}
    choices = _normalize_choice_spec(spec.get("choices", None))
    if choices is not None and field_ctype not in (ctypes.c_float, ctypes.c_double):
        try:
            vi = int(v)
        except Exception:
            vi = 0
        for val, lbl in choices:
            try:
                if int(val) == int(vi):
                    return str(lbl)
            except Exception:
                continue
        return str(vi)

    if field_ctype in (ctypes.c_float, ctypes.c_double):
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)

    try:
        return str(int(v))
    except Exception:
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)


def ensure_menu_button_binding(
    *,
    cfg_path: str = "joystick.json",
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    force: bool = False,
) -> tuple[int | None, bool]:
    cfg = load_or_create_joystick_config(cfg_path)
    existing = cfg.get("menu_button", None)
    if (not bool(force)) and isinstance(existing, int) and existing >= 0:
        return int(existing), False

    if joystick is None:
        return None, False

    # Prompt until the user presses any joystick button.
    binding: int | None = None
    clock = pygame.time.Clock()
    while binding is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, False
            if event.type == pygame.JOYBUTTONDOWN:
                binding = int(event.button)
                break

        # Also poll all buttons (autodiscover count) so we work even if
        # JOYBUTTONDOWN isn't emitted for the user's preferred control.
        if binding is None:
            try:
                n_buttons = int(joystick.get_numbuttons())
                for b in range(max(0, n_buttons)):
                    try:
                        if int(joystick.get_button(b)) != 0:
                            binding = int(b)
                    except Exception:
                        pass
            except Exception:
                pass

        _draw_fullscreen_lines(font, width, height, ["press menu button"])
        pygame.display.flip()
        clock.tick(60)

    cfg["menu_button"] = int(binding)
    save_joystick_config(cfg, cfg_path)
    return int(binding), True


def run_main_menu(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    menu_button: int | None,
    menu_context: dict[str, Any] | None = None,
) -> str:
    def _ensure_flight_controls_profiles(cfg: dict) -> dict:
        """Ensure cfg["flight_controls"]["profiles"][profile][mode] skeleton exists."""
        if not isinstance(cfg.get("flight_controls", None), dict):
            cfg["flight_controls"] = {}
        fc = cfg["flight_controls"]
        if not isinstance(fc.get("profiles", None), dict):
            fc["profiles"] = {}
        profiles = fc["profiles"]

        for prof in ("fighter", "bomber"):
            if not isinstance(profiles.get(prof, None), dict):
                profiles[prof] = {}
            pblk = profiles[prof]
            for mode in ("flight", "view"):
                if not isinstance(pblk.get(mode, None), dict):
                    pblk[mode] = {}
                mblk = pblk[mode]
                if not isinstance(mblk.get("craft", None), dict):
                    mblk["craft"] = {}
                if not isinstance(mblk.get("look", None), dict):
                    mblk["look"] = {}
                if not isinstance(mblk.get("reticle_look", None), dict):
                    mblk["reticle_look"] = {}
                if not isinstance(mblk.get("flaps", None), dict):
                    mblk["flaps"] = {}

        if not isinstance(fc.get("toggles", None), dict):
            fc["toggles"] = {}
        if not isinstance(fc.get("state", None), dict):
            fc["state"] = {}
        st = fc["state"]
        if not isinstance(st.get("active_profile", None), str):
            st["active_profile"] = "fighter"
        if not isinstance(st.get("active_mode", None), str):
            st["active_mode"] = "flight"
        if not isinstance(st.get("view_targeting", None), bool):
            st["view_targeting"] = False

        return cfg

    def _ensure_flight_controls_sets(cfg: dict) -> dict:
        """Ensure cfg["flight_controls"]["sets"][set_name] skeleton exists."""
        if not isinstance(cfg.get("flight_controls", None), dict):
            cfg["flight_controls"] = {}
        fc = cfg["flight_controls"]
        if not isinstance(fc.get("sets", None), dict):
            fc["sets"] = {}
        sets = fc["sets"]
        for sname in ("flight", "view_targeting", "bomber", "fighter"):
            if not isinstance(sets.get(sname, None), dict):
                sets[sname] = {}
            blk = sets[sname]
            for group in (
                "craft",
                "look",
                "reticle_look",
                "flaps",
                "triggers",
                "weapons",
                "camera",
                "hud",
            ):
                if not isinstance(blk.get(group, None), dict):
                    blk[group] = {}
        return cfg

    def _write_set_mapping(*, set_name: str, group: str, key: str, mapping: dict[str, Any]) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_flight_controls_sets(cfg)
        fc = cfg["flight_controls"]
        blk = fc["sets"][str(set_name)]
        if not isinstance(blk.get(str(group), None), dict):
            blk[str(group)] = {}
        blk[str(group)][str(key)] = dict(mapping)
        save_joystick_config(cfg, "joystick.json")

    def _write_set_trigger_mapping(*, set_name: str, key: str, mapping: dict[str, Any]) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_flight_controls_sets(cfg)
        fc = cfg["flight_controls"]
        blk = fc["sets"][str(set_name)]
        if not isinstance(blk.get("triggers", None), dict):
            blk["triggers"] = {}
        blk["triggers"][str(key)] = {"axis": int(mapping["axis"]), "sign": int(mapping["sign"])}
        save_joystick_config(cfg, "joystick.json")

    def _ensure_controller_graph(cfg: dict) -> dict:
        return input_graph.ensure_controller_graph(cfg)

    def _pick_from_list(
        *,
        title: str,
        options: list[tuple[str, str]],
    ) -> str | None:
        """Return the selected option key, or None if cancelled."""
        if joystick is None:
            return None
        if not options:
            return None

        cfg_local = load_or_create_joystick_config("joystick.json")
        nav = _get_menu_nav(cfg_local)
        b_up = nav.get("up")
        b_down = nav.get("down")
        b_confirm = nav.get("confirm")
        b_cancel = nav.get("cancel")

        sel = 0
        clock = pygame.time.Clock()

        axes_prev: dict[int, float] = {}
        buttons_prev: set[int] = set()
        hats_prev: dict[int, tuple[int, int]] = {}

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None

            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

            up_edge = _nav_edge(
                b_up,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            )
            down_edge = _nav_edge(
                b_down,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            )
            ok_edge = _nav_edge(
                b_confirm,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            )
            cancel_edge = _nav_edge(
                b_cancel,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            )

            if cancel_edge:
                return None
            if up_edge:
                sel = (sel - 1) % max(1, len(options))
            if down_edge:
                sel = (sel + 1) % max(1, len(options))
            if ok_edge:
                return str(options[int(sel)][0])

            labels = [str(lbl) for _k, lbl in options]
            _draw_menu_list(
                font=font,
                width=int(width),
                height=int(height),
                title=str(title),
                items=labels,
                selected_idx=int(sel),
                ret_x=0.04,
                ret_y=0.66,
                stage=reticle_sprite.ReticleStage.READY,
            )
            pygame.display.flip()
            clock.tick(60)

            axes_prev, buttons_prev, hats_prev = axes_now, set(buttons_now), dict(hats_now)

    def _edit_float(
        *,
        title: str,
        initial: float,
        lo: float = -1.0,
        hi: float = 1.0,
        rate: float = 0.9,
        deadzone: float = 0.18,
    ) -> float | None:
        """Simple float editor: move stick left/right, press any button to confirm."""
        if joystick is None:
            return None
        try:
            joystick.init()
        except Exception:
            pass
        v = float(initial)
        clock = pygame.time.Clock()

        # Snapshot current buttons so confirm is a new press.
        last_buttons: set[int] = set()
        try:
            n_buttons0 = int(joystick.get_numbuttons())
            for b0 in range(max(0, n_buttons0)):
                try:
                    if int(joystick.get_button(int(b0))) != 0:
                        last_buttons.add(int(b0))
                except Exception:
                    pass
        except Exception:
            pass

        while True:
            try:
                pygame.event.pump()
            except Exception:
                pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            # Confirm by new press.
            polled: set[int] = set()
            try:
                n_buttons = int(joystick.get_numbuttons())
            except Exception:
                n_buttons = 0
            for b in range(max(0, n_buttons)):
                try:
                    if int(joystick.get_button(int(b))) != 0:
                        polled.add(int(b))
                except Exception:
                    pass
            new_presses = polled - last_buttons
            if new_presses:
                b = int(sorted(list(new_presses))[0])
                if menu_button is None or int(b) != int(menu_button):
                    return float(v)
            last_buttons = polled

            # Axis 0 as adjustment.
            try:
                ax = float(joystick.get_axis(0))
            except Exception:
                ax = 0.0
            if abs(float(ax)) < float(deadzone):
                ax = 0.0
            v += float(rate) * float(ax) * (1.0 / 60.0)
            v = float(max(float(lo), min(float(hi), float(v))))

            _draw_fullscreen_lines(
                font,
                int(width),
                int(height),
                [
                    str(title),
                    f"value: {v:+.3f}",
                    "move stick left/right to adjust",
                    "press any button to confirm",
                    "(menu button cancels)",
                ],
            )
            pygame.display.flip()
            clock.tick(60)

    def _discover_feature(*, label: str) -> dict[str, int] | None:
        """Listen for any joystick input and return a feature descriptor."""
        if joystick is None:
            return None
        clock = pygame.time.Clock()
        try:
            joystick.init()
        except Exception:
            pass

        init_axis: dict[int, float] = {}
        try:
            n_axes = int(joystick.get_numaxes())
        except Exception:
            n_axes = 0
        for a in range(max(0, n_axes)):
            try:
                init_axis[int(a)] = float(joystick.get_axis(int(a)))
            except Exception:
                init_axis[int(a)] = 0.0

        last_buttons: set[int] = set()
        try:
            n_buttons0 = int(joystick.get_numbuttons())
        except Exception:
            n_buttons0 = 0
        for b0 in range(max(0, n_buttons0)):
            try:
                if int(joystick.get_button(int(b0))) != 0:
                    last_buttons.add(int(b0))
            except Exception:
                pass

        last_hats: dict[int, tuple[int, int]] = {}
        try:
            n_hats0 = int(joystick.get_numhats())
        except Exception:
            n_hats0 = 0
        for h0 in range(max(0, n_hats0)):
            try:
                v0 = joystick.get_hat(int(h0))
                last_hats[int(h0)] = (int(v0[0]), int(v0[1]))
            except Exception:
                last_hats[int(h0)] = (0, 0)

        while True:
            try:
                pygame.event.pump()
            except Exception:
                pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            # New button press.
            polled_buttons: set[int] = set()
            try:
                n_buttons = int(joystick.get_numbuttons())
            except Exception:
                n_buttons = 0
            for b in range(max(0, n_buttons)):
                try:
                    if int(joystick.get_button(int(b))) != 0:
                        polled_buttons.add(int(b))
                except Exception:
                    pass
            new_presses = polled_buttons - last_buttons
            if new_presses:
                b = int(sorted(list(new_presses))[0])
                if menu_button is None or int(b) != int(menu_button):
                    return {"type": "button", "button": int(b)}
            last_buttons = polled_buttons

            # Hat move.
            try:
                n_hats = int(joystick.get_numhats())
            except Exception:
                n_hats = 0
            for h in range(max(0, n_hats)):
                try:
                    v = joystick.get_hat(int(h))
                    now = (int(v[0]), int(v[1]))
                except Exception:
                    now = (0, 0)
                if now != (0, 0) and last_hats.get(int(h), (0, 0)) == (0, 0):
                    return {"type": "hat", "hat": int(h), "x": int(now[0]), "y": int(now[1])}
                last_hats[int(h)] = now

            # Axis motion.
            try:
                n_axes = int(joystick.get_numaxes())
            except Exception:
                n_axes = 0
            for a in range(max(0, n_axes)):
                try:
                    v = float(joystick.get_axis(int(a)))
                except Exception:
                    continue
                v0 = float(init_axis.get(int(a), 0.0))
                if abs(float(v) - float(v0)) >= 0.35:
                    return {"type": "axis", "axis": int(a)}

            _draw_fullscreen_lines(
                font,
                int(width),
                int(height),
                [
                    f"{label}",
                    "move any axis or press any button",
                    "(menu button cancels)",
                ],
            )
            pygame.display.flip()
            clock.tick(60)

    def _controller_add_feature(feature: dict[str, int]) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        feats = ctrl["features"]

        ftype = str(feature.get("type", ""))
        if ftype == "axis" and isinstance(feature.get("axis"), int):
            base = f"axis_{int(feature['axis'])}"
        elif ftype == "button" and isinstance(feature.get("button"), int):
            base = f"button_{int(feature['button'])}"
        elif ftype == "hat" and isinstance(feature.get("hat"), int) and isinstance(feature.get("x"), int) and isinstance(feature.get("y"), int):
            base = f"hat_{int(feature['hat'])}_{int(feature['x'])}_{int(feature['y'])}"
        else:
            return

        fid = base
        k = 2
        while fid in feats:
            fid = f"{base}_{k}"
            k += 1

        feats[fid] = dict(feature)
        ctrl["state"]["last_feature"] = str(fid)
        save_joystick_config(cfg, "joystick.json")

    def _controller_create_signal() -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        feats: dict = ctrl.get("features", {}) if isinstance(ctrl.get("features", {}), dict) else {}
        sigs: dict = ctrl.get("signals", {}) if isinstance(ctrl.get("signals", {}), dict) else {}

        op = _pick_from_list(
            title="SIGNAL OP",
            options=[
                ("raw", "RAW (feature)"),
                ("const", "CONST"),
                ("scale", "SCALE"),
                ("add", "ADD"),
                ("sub", "SUB"),
                ("clamp", "CLAMP"),
                ("deadzone", "DEADZONE"),
            ],
        )
        if op is None:
            return

        def _sources_list() -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for k in sorted(list(feats.keys())):
                out.append((f"feat:{k}", f"FEATURE {k}"))
            for k in sorted(list(sigs.keys())):
                out.append((f"sig:{k}", f"SIGNAL {k}"))
            return out

        def _pick_src(prompt: str) -> str | None:
            return _pick_from_list(title=prompt, options=_sources_list())

        node: dict[str, Any] = {"op": str(op)}
        if op == "raw":
            if not feats:
                return
            src = _pick_from_list(
                title="SELECT FEATURE",
                options=[(k, k) for k in sorted(list(feats.keys()))],
            )
            if src is None:
                return
            node["feature"] = str(src)
        elif op == "const":
            v = _edit_float(title="CONST VALUE", initial=0.0)
            if v is None:
                return
            node["value"] = float(v)
        elif op == "scale":
            src = _pick_src("SCALE INPUT")
            if src is None:
                return
            g = _edit_float(title="GAIN", initial=1.0, lo=-4.0, hi=4.0)
            if g is None:
                return
            node["in"] = str(src)
            node["gain"] = float(g)
        elif op in ("add", "sub"):
            a = _pick_src("INPUT A")
            if a is None:
                return
            b = _pick_src("INPUT B")
            if b is None:
                return
            node["a"] = str(a)
            node["b"] = str(b)
        elif op == "clamp":
            src = _pick_src("CLAMP INPUT")
            if src is None:
                return
            lo = _edit_float(title="CLAMP LO", initial=-1.0, lo=-10.0, hi=10.0)
            if lo is None:
                return
            hi = _edit_float(title="CLAMP HI", initial=1.0, lo=-10.0, hi=10.0)
            if hi is None:
                return
            node["in"] = str(src)
            node["lo"] = float(lo)
            node["hi"] = float(hi)
        elif op == "deadzone":
            src = _pick_src("DEADZONE INPUT")
            if src is None:
                return
            dz = _edit_float(title="DEADZONE", initial=0.08, lo=0.0, hi=0.95)
            if dz is None:
                return
            node["in"] = str(src)
            node["dz"] = float(dz)

        sid_base = "sig"
        i = 1
        sid = f"{sid_base}{i}"
        while sid in sigs:
            i += 1
            sid = f"{sid_base}{i}"

        sigs[sid] = node
        ctrl["state"]["last_signal"] = str(sid)
        save_joystick_config(cfg, "joystick.json")

    def _controller_map_channel() -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs: dict = ctrl.get("signals", {}) if isinstance(ctrl.get("signals", {}), dict) else {}

        ch_key = _pick_from_list(
            title="CHANNEL",
            options=[(str(i), f"channel[{i}]") for i in range(8)],
        )
        if ch_key is None:
            return

        src_kind = _pick_from_list(
            title="MAP TYPE",
            options=[
                ("signal", "FROM SIGNAL"),
                ("const", "CONSTANT"),
                ("clear", "CLEAR"),
            ],
        )
        if src_kind is None:
            return

        if src_kind == "clear":
            try:
                ctrl["channels"].pop(str(ch_key), None)
            except Exception:
                pass
            save_joystick_config(cfg, "joystick.json")
            return

        if src_kind == "const":
            v = _edit_float(title=f"channel[{ch_key}] CONST", initial=0.0)
            if v is None:
                return
            ctrl["channels"][str(ch_key)] = {"source": "const", "value": float(v)}
            save_joystick_config(cfg, "joystick.json")
            return

        if src_kind == "signal":
            if not sigs:
                return
            sid = _pick_from_list(
                title="SELECT SIGNAL",
                options=[(k, k) for k in sorted(list(sigs.keys()))],
            )
            if sid is None:
                return
            ctrl["channels"][str(ch_key)] = {"source": "signal", "id": str(sid)}
            save_joystick_config(cfg, "joystick.json")
            return

    def _bind_axis(
        label: str,
        *,
        joystick: pygame.joystick.Joystick,
        threshold: float,
    ) -> int | None:
        clock = pygame.time.Clock()

        init_axis: dict[int, float] = {}
        axis_woken: set[int] = set()
        try:
            n_axes = int(joystick.get_numaxes())
            for a in range(max(0, n_axes)):
                try:
                    init_axis[a] = float(joystick.get_axis(a))
                except Exception:
                    init_axis[a] = 0.0
        except Exception:
            pass

        if not hasattr(_bind_axis, "_reticle_assets"):
            _bind_axis._reticle_assets = reticle_sprite.load_reticle_assets(size_px=64)
        assets = _bind_axis._reticle_assets
        anim = reticle_sprite.ReticleAnimator(lock_delay_s=0.20, ready_delay_s=0.45)

        x_lbl = 0.40
        y_lbl = 0.52
        start = (x_lbl, y_lbl - 0.16)
        target = (x_lbl, y_lbl)
        ret_x, ret_y = float(start[0]), float(start[1])

        candidate: dict[str, Any] | None = None
        latest_axis: dict[int, float] = {int(a): float(v) for a, v in init_axis.items()}

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            try:
                n_axes = int(joystick.get_numaxes())
                for a in range(max(0, n_axes)):
                    try:
                        latest_axis[int(a)] = float(joystick.get_axis(a))
                    except Exception:
                        pass
            except Exception:
                pass

            for a, v in list(latest_axis.items()):
                v0 = float(init_axis.get(int(a), 0.0))
                if abs(float(v) - v0) >= 0.25:
                    axis_woken.add(int(a))

            if candidate is None:
                for a, v in latest_axis.items():
                    if int(a) not in axis_woken:
                        continue
                    if abs(float(v)) < float(threshold):
                        continue
                    sign = 1 if float(v) > 0.0 else -1
                    candidate = {"type": "axis", "axis": int(a), "sign": int(sign)}
                    break

            candidate_active = False
            if candidate is not None and candidate.get("type") == "axis":
                a = int(candidate.get("axis", -1))
                s = int(candidate.get("sign", 0))
                v = float(latest_axis.get(a, 0.0))
                candidate_active = (abs(v) >= float(threshold)) and ((1 if v > 0.0 else -1) == s)

            desired = target if candidate_active else start
            ret_x += (float(desired[0]) - float(ret_x)) * 0.20
            ret_y += (float(desired[1]) - float(ret_y)) * 0.20
            on_target = (abs(float(ret_x) - float(target[0])) <= 0.015) and (abs(float(ret_y) - float(target[1])) <= 0.020)
            now_s = pygame.time.get_ticks() * 0.001
            stage = anim.update(on_target=bool(on_target and candidate_active), now_s=float(now_s))

            from OpenGL.GL import (
                GL_DEPTH_TEST,
                GL_BLEND,
                GL_COLOR_BUFFER_BIT,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_SRC_ALPHA,
                GL_UNSIGNED_BYTE,
                GL_RGBA,
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
                glPushMatrix,
                glPopMatrix,
                glRasterPos2f,
                GL_MODELVIEW,
                GL_PROJECTION,
                GL_UNPACK_ALIGNMENT,
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

            surf_lbl = _render_cell_text(font, label)
            lbl_data = pygame.image.tostring(surf_lbl, "RGBA", True)
            glRasterPos2f(float(x_lbl), float(y_lbl))
            glDrawPixels(surf_lbl.get_width(), surf_lbl.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, lbl_data)

            surf_ret = assets.surfaces.get(stage)
            if surf_ret is not None:
                r_data = pygame.image.tostring(surf_ret, "RGBA", True)
                glRasterPos2f(float(ret_x), float(ret_y))
                glDrawPixels(surf_ret.get_width(), surf_ret.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, r_data)

            glDisable(GL_BLEND)
            if depth_was_enabled:
                glEnable(GL_DEPTH_TEST)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

            pygame.display.flip()
            clock.tick(60)

            if candidate_active and stage == reticle_sprite.ReticleStage.READY:
                return int(candidate.get("axis", -1))

            if candidate is not None and not candidate_active:
                candidate = None
                anim._on_since = None

    def _bind_axis_calibrated(
        label: str,
        *,
        joystick: pygame.joystick.Joystick,
        threshold: float,
        wake_deadzone: float = 0.25,
        pair_threshold: float | None = 0.85,
    ) -> dict[str, Any] | None:
        """Standardized axis binding: choose axis then calibrate extrema (or optionally pair to 2D)."""
        a0 = _bind_axis(str(label), joystick=joystick, threshold=float(threshold))
        if a0 is None or int(a0) < 0:
            return None

        try:
            joystick.init()
        except Exception:
            pass

        def _poll_axis(idx: int) -> float:
            try:
                return float(joystick.get_axis(int(idx)))
            except Exception:
                return 0.0

        # Track min/max while the user moves the control.
        v0 = _poll_axis(int(a0))
        min0 = float(v0)
        max0 = float(v0)

        a1: int | None = None
        min1: float = 0.0
        max1: float = 0.0

        # Snapshot current button state so "confirm" is a new press.
        last_buttons: set[int] = set()
        try:
            n_buttons0 = int(joystick.get_numbuttons())
            for b0 in range(max(0, n_buttons0)):
                try:
                    if int(joystick.get_button(int(b0))) != 0:
                        last_buttons.add(int(b0))
                except Exception:
                    pass
        except Exception:
            pass

        clock = pygame.time.Clock()
        while True:
            try:
                pygame.event.pump()
            except Exception:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            # Confirm by *new* button press (anything except menu button).
            polled_buttons: set[int] = set()
            try:
                n_buttons = int(joystick.get_numbuttons())
            except Exception:
                n_buttons = 0
            for b in range(max(0, n_buttons)):
                try:
                    if int(joystick.get_button(int(b))) != 0:
                        polled_buttons.add(int(b))
                except Exception:
                    pass
            new_presses = polled_buttons - last_buttons
            if new_presses:
                b = int(sorted(list(new_presses))[0])
                if menu_button is None or int(b) != int(menu_button):
                    if a1 is None:
                        return {
                            "type": "axis1d",
                            "axis": int(a0),
                            "calib": {"min": float(min0), "max": float(max0)},
                        }
                    return {
                        "type": "axis2d",
                        "x": {"axis": int(a0), "calib": {"min": float(min0), "max": float(max0)}},
                        "y": {"axis": int(a1), "calib": {"min": float(min1), "max": float(max1)}},
                    }

            last_buttons = polled_buttons

            v = _poll_axis(int(a0))
            min0 = float(min(min0, float(v)))
            max0 = float(max(max0, float(v)))

            # Optional: detect a second axis for 2D-mode if the user moves another axis to an extreme.
            if pair_threshold is not None and a1 is None:
                try:
                    n_axes = int(joystick.get_numaxes())
                except Exception:
                    n_axes = 0
                for a in range(max(0, n_axes)):
                    if int(a) == int(a0):
                        continue
                    vv = _poll_axis(int(a))
                    if abs(float(vv)) >= float(pair_threshold):
                        a1 = int(a)
                        min1 = float(vv)
                        max1 = float(vv)
                        break

            if a1 is not None:
                v1 = _poll_axis(int(a1))
                min1 = float(min(min1, float(v1)))
                max1 = float(max(max1, float(v1)))

            lines = [
                f"bind {label}",
                f"axis: {int(a0)}  min/max: {min0:+.3f} / {max0:+.3f}",
            ]
            if a1 is None and pair_threshold is not None:
                lines.append(f"move optional 2D axis (abs >= {float(pair_threshold):.2f})")
            if a1 is not None:
                lines.append(f"2D axis: {int(a1)}  min/max: {min1:+.3f} / {max1:+.3f}")
            lines += [
                "move through full range",
                "press any button to confirm",
                "(menu button cancels)",
            ]
            _draw_fullscreen_lines(font, int(width), int(height), lines)
            pygame.display.flip()
            clock.tick(60)

    def _bind_buttonish(label: str, *, joystick: pygame.joystick.Joystick) -> dict[str, int] | None:
        # Reuse the existing mapping capture: button/hat/axis.
        return _bind_fire_mapping(str(label), joystick=joystick, delta_threshold=0.30)

    def _write_toggle_mapping(*, key: str, mapping: dict[str, int]) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_flight_controls_profiles(cfg)
        fc = cfg["flight_controls"]
        if not isinstance(fc.get("toggles", None), dict):
            fc["toggles"] = {}
        t = fc["toggles"]
        mtype = str(mapping.get("type", ""))
        if mtype == "button" and isinstance(mapping.get("button"), int):
            t[str(key)] = {"type": "button", "button": int(mapping["button"])}
        elif (
            mtype == "hat"
            and isinstance(mapping.get("hat"), int)
            and isinstance(mapping.get("x"), int)
            and isinstance(mapping.get("y"), int)
        ):
            t[str(key)] = {"type": "hat", "hat": int(mapping["hat"]), "x": int(mapping["x"]), "y": int(mapping["y"])}
        elif mtype == "axis" and isinstance(mapping.get("axis"), int) and int(mapping.get("sign", 0)) in (-1, 1):
            t[str(key)] = {"type": "axis", "axis": int(mapping["axis"]), "sign": int(mapping["sign"])}
        save_joystick_config(cfg, "joystick.json")

    def _write_profile_mapping(*, profile: str, mode: str, group: str, key: str, mapping: Any) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_flight_controls_profiles(cfg)
        fc = cfg["flight_controls"]
        pblk = fc["profiles"].get(str(profile))
        if not isinstance(pblk, dict):
            return
        mblk = pblk.get(str(mode))
        if not isinstance(mblk, dict):
            return
        if not isinstance(mblk.get(str(group), None), dict):
            mblk[str(group)] = {}
        g = mblk[str(group)]
        g[str(key)] = mapping
        save_joystick_config(cfg, "joystick.json")

    def _sync_all_to(*, src_profile: str, src_mode: str) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = _ensure_flight_controls_profiles(cfg)
        fc = cfg["flight_controls"]
        profiles = fc.get("profiles", {})
        src = profiles.get(str(src_profile), {}).get(str(src_mode), {})
        if not isinstance(src, dict):
            return
        for prof in ("fighter", "bomber"):
            for mode in ("flight", "view"):
                try:
                    profiles[prof][mode] = json.loads(json.dumps(src))
                except Exception:
                    # Fall back to shallow copy if needed.
                    profiles[prof][mode] = dict(src)
        save_joystick_config(cfg, "joystick.json")

    def _bind_trigger_mapping(
        label: str,
        *,
        joystick: pygame.joystick.Joystick,
        delta_threshold: float = 0.30,
        wake_deadzone: float = 0.05,
    ) -> dict[str, int] | None:
        """Bind a trigger-like control.

        Many controllers expose triggers as:
        - separate axes, rest at 0 or -1
        - OR a single shared axis (LT negative, RT positive) with rest near 0

        We therefore bind using delta-from-initial and capture the sign.
        Returns {"axis": <idx>, "sign": (+1|-1)}.
        """
        clock = pygame.time.Clock()

        init_axis: dict[int, float] = {}
        axis_woken: set[int] = set()
        try:
            n_axes = int(joystick.get_numaxes())
            for a in range(max(0, n_axes)):
                try:
                    init_axis[a] = float(joystick.get_axis(a))
                except Exception:
                    init_axis[a] = 0.0
        except Exception:
            pass

        if not hasattr(_bind_trigger_mapping, "_reticle_assets"):
            _bind_trigger_mapping._reticle_assets = reticle_sprite.load_reticle_assets(size_px=64)
        assets = _bind_trigger_mapping._reticle_assets
        anim = reticle_sprite.ReticleAnimator(lock_delay_s=0.20, ready_delay_s=0.45)

        x_lbl = 0.40
        y_lbl = 0.52
        start = (x_lbl, y_lbl - 0.16)
        target = (x_lbl, y_lbl)
        ret_x, ret_y = float(start[0]), float(start[1])

        candidate: dict[str, int] | None = None
        latest_axis: dict[int, float] = {int(a): float(v) for a, v in init_axis.items()}

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            try:
                n_axes = int(joystick.get_numaxes())
                for a in range(max(0, n_axes)):
                    try:
                        latest_axis[int(a)] = float(joystick.get_axis(a))
                    except Exception:
                        pass
            except Exception:
                pass

            for a, v in list(latest_axis.items()):
                v0 = float(init_axis.get(int(a), 0.0))
                if abs(float(v) - v0) >= float(wake_deadzone):
                    axis_woken.add(int(a))

            if candidate is None:
                # Pick the first axis that is woken and has a strong enough delta.
                for a, v in latest_axis.items():
                    if int(a) not in axis_woken:
                        continue
                    v0 = float(init_axis.get(int(a), 0.0))
                    dv = float(v) - v0
                    if abs(dv) < float(delta_threshold):
                        continue
                    sign = 1 if dv > 0.0 else -1
                    candidate = {"axis": int(a), "sign": int(sign)}
                    break

            candidate_active = False
            if candidate is not None:
                a = int(candidate.get("axis", -1))
                s = int(candidate.get("sign", 0))
                v = float(latest_axis.get(a, 0.0))
                v0 = float(init_axis.get(a, 0.0))
                dv = (v - v0) * float(s)
                candidate_active = dv >= float(delta_threshold)

            desired = target if candidate_active else start
            ret_x += (float(desired[0]) - float(ret_x)) * 0.20
            ret_y += (float(desired[1]) - float(ret_y)) * 0.20
            on_target = (abs(float(ret_x) - float(target[0])) <= 0.015) and (abs(float(ret_y) - float(target[1])) <= 0.020)
            now_s = pygame.time.get_ticks() * 0.001
            stage = anim.update(on_target=bool(on_target and candidate_active), now_s=float(now_s))

            from OpenGL.GL import (
                GL_DEPTH_TEST,
                GL_BLEND,
                GL_COLOR_BUFFER_BIT,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_SRC_ALPHA,
                GL_UNSIGNED_BYTE,
                GL_RGBA,
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
                glPushMatrix,
                glPopMatrix,
                glRasterPos2f,
                GL_MODELVIEW,
                GL_PROJECTION,
                GL_UNPACK_ALIGNMENT,
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

            surf_lbl = _render_cell_text(font, label)
            lbl_data = pygame.image.tostring(surf_lbl, "RGBA", True)
            glRasterPos2f(float(x_lbl), float(y_lbl))
            glDrawPixels(surf_lbl.get_width(), surf_lbl.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, lbl_data)

            surf_ret = assets.surfaces.get(stage)
            if surf_ret is not None:
                r_data = pygame.image.tostring(surf_ret, "RGBA", True)
                glRasterPos2f(float(ret_x), float(ret_y))
                glDrawPixels(surf_ret.get_width(), surf_ret.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, r_data)

            glDisable(GL_BLEND)
            if depth_was_enabled:
                glEnable(GL_DEPTH_TEST)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

            pygame.display.flip()
            clock.tick(60)

            if candidate_active and stage == reticle_sprite.ReticleStage.READY:
                a = int(candidate.get("axis", -1))
                s = int(candidate.get("sign", 0))
                if a >= 0 and s in (-1, 1):
                    return {"axis": int(a), "sign": int(s)}
                return None

            if candidate is not None and not candidate_active:
                candidate = None
                anim._on_since = None

    def _bind_fire_mapping(
        label: str,
        *,
        joystick: pygame.joystick.Joystick,
        delta_threshold: float = 0.30,
    ) -> dict[str, int] | None:
        """Bind a fire control to either a button or a trigger-like axis.

        Returns one of:
        - {"type": "button", "button": <idx>}
        - {"type": "axis", "axis": <idx>, "sign": (+1|-1)}
        """
        clock = pygame.time.Clock()

        # Defensive: ensure the joystick is initialized before polling counts.
        try:
            joystick.init()
        except Exception:
            pass

        init_axis: dict[int, float] = {}
        try:
            n_axes = int(joystick.get_numaxes())
            for a in range(max(0, n_axes)):
                try:
                    init_axis[int(a)] = float(joystick.get_axis(a))
                except Exception:
                    init_axis[int(a)] = 0.0
        except Exception:
            init_axis = {}

        # Snapshot initial button/hat state so we can require a deliberate new press,
        # not a button that was already held while entering the bind screen.
        last_buttons: set[int] = set()
        last_hats: dict[int, tuple[int, int]] = {}
        try:
            n_buttons0 = int(joystick.get_numbuttons())
            for b0 in range(max(0, n_buttons0)):
                try:
                    if int(joystick.get_button(int(b0))) != 0:
                        last_buttons.add(int(b0))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            n_hats0 = int(joystick.get_numhats())
            for h0 in range(max(0, n_hats0)):
                try:
                    v0 = joystick.get_hat(int(h0))
                    last_hats[int(h0)] = (int(v0[0]), int(v0[1]))
                except Exception:
                    last_hats[int(h0)] = (0, 0)
        except Exception:
            pass

        while True:
            # Ensure joystick state is refreshed even if there are no events.
            try:
                pygame.event.pump()
            except Exception:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return None

            # Poll ALL buttons (some controllers/drivers don't deliver JOYBUTTONDOWN reliably).
            try:
                n_buttons = int(joystick.get_numbuttons())
            except Exception:
                n_buttons = 0
            polled_buttons: set[int] = set()
            for b in range(max(0, n_buttons)):
                try:
                    if int(joystick.get_button(int(b))) != 0:
                        polled_buttons.add(int(b))
                except Exception:
                    pass
            new_presses = polled_buttons - last_buttons
            if new_presses:
                # Pick the lowest index deterministically.
                b = int(sorted(list(new_presses))[0])
                if menu_button is None or b != int(menu_button):
                    return {"type": "button", "button": int(b)}

            # Poll hats too (D-pad can be a hat on Xbox controllers).
            try:
                n_hats = int(joystick.get_numhats())
            except Exception:
                n_hats = 0
            polled_hats: dict[int, tuple[int, int]] = {}
            for h in range(max(0, n_hats)):
                try:
                    v = joystick.get_hat(int(h))
                    polled_hats[int(h)] = (int(v[0]), int(v[1]))
                except Exception:
                    polled_hats[int(h)] = (0, 0)
            for h, v in polled_hats.items():
                if v != (0, 0) and last_hats.get(int(h), (0, 0)) == (0, 0):
                    return {"type": "hat", "hat": int(h), "x": int(v[0]), "y": int(v[1])}

            # Trigger-like axis delta detection (works for shared-axis triggers).
            try:
                n_axes = int(joystick.get_numaxes())
            except Exception:
                n_axes = 0
            for a in range(max(0, n_axes)):
                try:
                    v = float(joystick.get_axis(int(a)))
                except Exception:
                    continue
                v0 = float(init_axis.get(int(a), 0.0))
                dv = float(v) - float(v0)
                if abs(float(dv)) >= float(delta_threshold):
                    sign = 1 if float(dv) > 0.0 else -1
                    return {"type": "axis", "axis": int(a), "sign": int(sign)}

            # Advance previous snapshots so we detect *new* presses/hat moves.
            last_buttons = polled_buttons
            last_hats = polled_hats

            _draw_fullscreen_lines(
                font,
                int(width),
                int(height),
                [
                    f"bind {label}",
                    "press a button or squeeze a trigger",
                    "(menu button cancels)",
                ],
            )
            pygame.display.flip()
            clock.tick(60)

    def _write_flight_axis(*, section: str, key: str, axis: int) -> None:
        cfg = load_or_create_joystick_config("joystick.json")
        if not isinstance(cfg.get("flight_controls", None), dict):
            cfg["flight_controls"] = {}
        fc = cfg["flight_controls"]
        if not isinstance(fc.get(section, None), dict):
            fc[section] = {}
        fc[section][key] = int(axis)
        save_joystick_config(cfg, "joystick.json")

    def _run_menu_from_json(
        *,
        spec_path: str = MENU_SPEC_PATH,
        start_node: str = "main",
        action_handlers: dict[str, Any] | None = None,
    ) -> str:
        try:
            ensure_menu_navigation_bindings(
                cfg_path="joystick.json",
                font=font,
                width=int(width),
                height=int(height),
                joystick=joystick,
                menu_button=menu_button,
            )
        except Exception:
            pass

        if joystick is None:
            return "start"

        spec = load_or_create_menu_spec(spec_path)
        node_id = str(start_node)
        stack: list[str] = []

        cfg = load_or_create_joystick_config("joystick.json")
        nav = _get_menu_nav(cfg)
        b_up = nav.get("up")
        b_down = nav.get("down")
        b_confirm = nav.get("confirm")
        b_cancel = nav.get("cancel")

        clock = pygame.time.Clock()
        sel = 0

        max_visible = 10

        axes_prev: dict[int, float] = {}
        buttons_prev: set[int] = set()
        hats_prev: dict[int, tuple[int, int]] = {}

        ret_x = 0.04
        ret_y = 0.66
        start_x = 0.04
        target_x = 0.10

        anim = reticle_sprite.ReticleAnimator(lock_delay_s=0.20, ready_delay_s=0.45)

        def _sel_y(idx_visible: int) -> float:
            return float(0.66 - 0.07 * int(idx_visible))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return "start"

            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

            node = _get_menu_node(spec, node_id)
            title = str(node.get("title") or node_id)
            menu_items = _node_items(node)
            labels = [str(it.get("label")) for it in menu_items]
            if not labels:
                labels = ["BACK"]
                menu_items = [{"label": "BACK", "action": "back"}]

            sel = int(sel) % max(1, len(labels))

            if _nav_edge(
                b_up,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                sel = (sel - 1) % max(1, len(labels))
                anim._on_since = None

            if _nav_edge(
                b_down,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                sel = (sel + 1) % max(1, len(labels))
                anim._on_since = None

            if _nav_edge(
                b_cancel,
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                if stack:
                    node_id = stack.pop()
                    sel = 0
                    anim._on_since = None
                else:
                    return "start"

            confirm_held = _nav_is_held(b_confirm, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)

            first_idx = 0
            if int(len(labels)) > int(max_visible):
                half = int(max_visible // 2)
                first_idx = int(sel) - int(half)
                first_idx = max(0, min(int(first_idx), int(len(labels) - max_visible)))
            visible_sel = int(sel) - int(first_idx)
            labels_vis = labels[int(first_idx) : int(first_idx) + int(max_visible)] if int(len(labels)) > int(max_visible) else labels

            ret_y_des = _sel_y(visible_sel)
            ret_y += (float(ret_y_des) - float(ret_y)) * 0.25
            ret_x_des = target_x if confirm_held else start_x
            ret_x += (float(ret_x_des) - float(ret_x)) * 0.25

            on_target = (abs(float(ret_x) - float(target_x)) <= 0.02) and (abs(float(ret_y) - float(ret_y_des)) <= 0.02)
            now_s = pygame.time.get_ticks() * 0.001
            stage = anim.update(on_target=bool(on_target and confirm_held), now_s=float(now_s))

            if confirm_held and stage == reticle_sprite.ReticleStage.READY:
                item = menu_items[int(sel)] if menu_items else {"label": "BACK", "action": "back"}
                submenu = item.get("submenu")
                action = item.get("action")
                edit_struct = item.get("edit_struct")

                if isinstance(submenu, str) and submenu:
                    stack.append(node_id)
                    node_id = submenu
                    sel = 0
                    anim._on_since = None
                else:
                    # Ctypes struct editor hook.
                    if isinstance(edit_struct, dict) and isinstance(edit_struct.get("id"), str):
                        sid = str(edit_struct.get("id"))
                        # Look up struct in context.
                        ctx = menu_context if isinstance(menu_context, dict) else {}
                        structs = ctx.get("ctypes_structs") if isinstance(ctx.get("ctypes_structs"), dict) else {}
                        entry = structs.get(sid) if isinstance(structs, dict) else None
                        if isinstance(entry, dict) and isinstance(entry.get("struct"), ctypes.Structure):
                            persist_path = str(edit_struct.get("persist_path") or entry.get("persist_path") or "world_config.json")
                            persist_key = str(edit_struct.get("persist_key") or entry.get("persist_key") or sid)
                            etitle = str(edit_struct.get("title") or entry.get("title") or sid)
                            specs = entry.get("field_specs") if isinstance(entry.get("field_specs"), dict) else None
                            poc = bool(edit_struct.get("persist_on_change", entry.get("persist_on_change", False)))
                            did = run_ctypes_struct_editor(
                                font=font,
                                width=int(width),
                                height=int(height),
                                joystick=joystick,
                                menu_button=menu_button,
                                struct_obj=entry["struct"],
                                title=etitle,
                                persist_path=persist_path,
                                persist_key=persist_key,
                                field_specs=specs,
                                persist_on_change=bool(poc),
                            )
                            if did:
                                cb = entry.get("on_commit")
                                if callable(cb):
                                    try:
                                        cb(entry["struct"])
                                    except Exception:
                                        pass
                            anim._on_since = None
                        else:
                            # No resolver: return an action-like hook.
                            return f"edit_struct:{sid}"

                    # Built-in actions
                    if action == "back":
                        if stack:
                            node_id = stack.pop()
                            sel = 0
                            anim._on_since = None
                        else:
                            return "start"
                    elif action == "start":
                        return "start"
                    elif isinstance(action, str) and action_handlers and action in action_handlers:
                        try:
                            res = action_handlers[action]()
                        except Exception:
                            res = None

                        # Convention: handler may return a terminal result.
                        if isinstance(res, str) and res in ("start", "quit"):
                            return res
                        # Otherwise stay in menu.
                        anim._on_since = None
                    elif isinstance(action, str) and action:
                        # Hook: caller can interpret this.
                        return str(action)
                    else:
                        anim._on_since = None

                # Reload nav bindings in case the user rebound anything.
                cfg = load_or_create_joystick_config("joystick.json")
                nav = _get_menu_nav(cfg)
                b_up = nav.get("up")
                b_down = nav.get("down")
                b_confirm = nav.get("confirm")
                b_cancel = nav.get("cancel")

            _draw_menu_list(
                font=font,
                width=int(width),
                height=int(height),
                title=title,
                items=labels_vis,
                selected_idx=int(sel),
                ret_x=float(ret_x),
                ret_y=float(ret_y),
                stage=stage,
            )
            pygame.display.flip()
            clock.tick(60)

            axes_prev, buttons_prev, hats_prev = axes_now, set(buttons_now), dict(hats_now)

    if joystick is None:
        return "start"

    def _mk_bind_handler(label: str, *, section: str, key: str, threshold: float) -> Any:
        def _h() -> None:
            a = _bind_axis(label, joystick=joystick, threshold=float(threshold))
            if a is not None and int(a) >= 0:
                _write_flight_axis(section=section, key=key, axis=int(a))
        return _h

    handlers: dict[str, Any] = {
        "bind_rudder": _mk_bind_handler("flight rudder", section="left_stick", key="rudder_axis", threshold=0.85),
        "bind_elevator": _mk_bind_handler("flight elevator", section="left_stick", key="elevator_axis", threshold=0.85),
        "bind_view_yaw": _mk_bind_handler("view yaw", section="right_stick", key="view_yaw_axis", threshold=0.85),
        "bind_view_pitch": _mk_bind_handler("view pitch", section="right_stick", key="view_pitch_axis", threshold=0.85),
    }

    # Controller graph menu actions.
    handlers["controller_discover_feature"] = lambda: (
        (lambda f: _controller_add_feature(f) if isinstance(f, dict) else None)(
            _discover_feature(label="discover feature")
        )
    )
    handlers["controller_create_signal"] = lambda: _controller_create_signal()
    handlers["controller_map_channel"] = lambda: _controller_map_channel()

    def _controller_status() -> None:
        if joystick is None:
            return
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return

            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)
            header = "JOYSTICK STATUS"
            lines: list[str] = []
            try:
                lines.append(f"axes: {int(joystick.get_numaxes())}  buttons: {int(joystick.get_numbuttons())}  hats: {int(joystick.get_numhats())}")
            except Exception:
                pass
            for a in sorted(axes_now.keys()):
                lines.append(f"axis {int(a)}: {float(axes_now[a]):+.3f}")
            if buttons_now:
                lines.append("pressed buttons: " + ", ".join(str(int(b)) for b in sorted(list(buttons_now))))
            if hats_now:
                for h in sorted(hats_now.keys()):
                    hx, hy = hats_now[int(h)]
                    lines.append(f"hat {int(h)}: ({int(hx)}, {int(hy)})")
            if not buttons_now and not hats_now:
                lines.append("(no buttons/hats pressed)")
            lines.append("(menu button / Esc exits)")

            _draw_fullscreen_lines(font, int(width), int(height), [header] + lines)
            pygame.display.flip()
            clock.tick(60)

    handlers["controller_status"] = lambda: _controller_status()

    # Menus: bind menu navigation explicitly.
    handlers["bind_menu_navigation"] = lambda: ensure_menu_navigation_bindings(
        cfg_path="joystick.json",
        font=font,
        width=int(width),
        height=int(height),
        joystick=joystick,
        menu_button=menu_button,
    )

    handlers["bind_menu_button"] = lambda: ensure_menu_button_binding(
        cfg_path="joystick.json",
        font=font,
        width=int(width),
        height=int(height),
        joystick=joystick,
        force=True,
    )

    def _bind_menu_key(k: str) -> None:
        ensure_menu_navigation_bindings(
            cfg_path="joystick.json",
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
            menu_button=menu_button,
            keys=[str(k)],
            force=True,
        )

    handlers["bind_menu_up"] = lambda: _bind_menu_key("up")
    handlers["bind_menu_down"] = lambda: _bind_menu_key("down")
    handlers["bind_menu_left"] = lambda: _bind_menu_key("left")
    handlers["bind_menu_right"] = lambda: _bind_menu_key("right")
    handlers["bind_menu_confirm"] = lambda: _bind_menu_key("confirm")
    handlers["bind_menu_back"] = lambda: _bind_menu_key("cancel")

    # Control-set bindings.
    def _bind_set_axis1d(*, label: str, set_name: str, group: str, key: str) -> None:
        m = _bind_axis_calibrated(label, joystick=joystick, threshold=0.85)
        if isinstance(m, dict):
            _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

    def _bind_set_axis2d(*, label: str, set_name: str, group: str, key: str) -> None:
        m = _bind_axis_calibrated(label, joystick=joystick, threshold=0.85)
        if not isinstance(m, dict):
            return
        if str(m.get("type", "")) == "axis2d":
            out = m
        else:
            out = {
                "type": "axis2d",
                "x": {"axis": int(m.get("axis", -1)), "calib": dict(m.get("calib", {}) if isinstance(m.get("calib", {}), dict) else {})},
                "y": {"axis": -1, "calib": {"min": 0.0, "max": 0.0}},
            }
        _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=out)

    def _bind_set_buttonish(*, label: str, set_name: str, group: str, key: str) -> None:
        m = _bind_buttonish(label, joystick=joystick)
        if isinstance(m, dict):
            _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

    def _bind_set_trigger(*, label: str, set_name: str, key: str) -> None:
        m = _bind_trigger_mapping(label, joystick=joystick, delta_threshold=0.30, wake_deadzone=0.05)
        if isinstance(m, dict) and isinstance(m.get("axis"), int) and isinstance(m.get("sign"), int):
            _write_set_trigger_mapping(set_name=str(set_name), key=str(key), mapping=m)

    def _bind_set_firelike(*, label: str, set_name: str, group: str, key: str) -> None:
        m = _bind_fire_mapping(str(label), joystick=joystick, delta_threshold=0.30)
        if isinstance(m, dict):
            _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

    def _install_set_handlers(prefix: str, set_name: str) -> None:
        # Craft surfaces.
        handlers[f"bind_set_{prefix}_craft_ailerons"] = lambda: _bind_set_axis1d(label=f"{prefix} craft ailerons", set_name=set_name, group="craft", key="ailerons")
        handlers[f"bind_set_{prefix}_craft_rudder"] = lambda: _bind_set_axis1d(label=f"{prefix} craft rudder", set_name=set_name, group="craft", key="rudder")
        handlers[f"bind_set_{prefix}_craft_elevators"] = lambda: _bind_set_axis1d(label=f"{prefix} craft elevators", set_name=set_name, group="craft", key="elevators")

        # Flaps.
        handlers[f"bind_set_{prefix}_flaps_up"] = lambda: _bind_set_buttonish(label=f"{prefix} flaps up", set_name=set_name, group="flaps", key="up")
        handlers[f"bind_set_{prefix}_flaps_down"] = lambda: _bind_set_buttonish(label=f"{prefix} flaps down", set_name=set_name, group="flaps", key="down")

        # Throttle triggers.
        handlers[f"bind_set_{prefix}_throttle_fwd"] = lambda: _bind_set_trigger(label=f"{prefix} throttle fwd", set_name=set_name, key="forward_axis")
        handlers[f"bind_set_{prefix}_throttle_rev"] = lambda: _bind_set_trigger(label=f"{prefix} throttle rev", set_name=set_name, key="reverse_axis")

        # Look + reticle look.
        handlers[f"bind_set_{prefix}_look_2d"] = lambda: _bind_set_axis2d(label=f"{prefix} look", set_name=set_name, group="look", key="axis2d")
        handlers[f"bind_set_{prefix}_reticle_look_2d"] = lambda: _bind_set_axis2d(label=f"{prefix} reticle look", set_name=set_name, group="reticle_look", key="axis2d")

        # Weapons.
        handlers[f"bind_set_{prefix}_fire_1"] = lambda: _bind_set_firelike(label=f"{prefix} fire 1", set_name=set_name, group="weapons", key="fire_1")
        handlers[f"bind_set_{prefix}_fire_2"] = lambda: _bind_set_firelike(label=f"{prefix} fire 2", set_name=set_name, group="weapons", key="fire_2")

        # Camera.
        handlers[f"bind_set_{prefix}_camera_zoom_in"] = lambda: _bind_set_firelike(label=f"{prefix} camera zoom in", set_name=set_name, group="camera", key="zoom_in")
        handlers[f"bind_set_{prefix}_camera_zoom_out"] = lambda: _bind_set_firelike(label=f"{prefix} camera zoom out", set_name=set_name, group="camera", key="zoom_out")

        # HUD.
        handlers[f"bind_set_{prefix}_debug_hud_toggle"] = lambda: _bind_set_firelike(label=f"{prefix} debug hud toggle", set_name=set_name, group="hud", key="toggle")

    _install_set_handlers("flight", "flight")
    _install_set_handlers("view_targeting", "view_targeting")
    _install_set_handlers("bomber", "bomber")
    _install_set_handlers("fighter", "fighter")

    # Note: legacy global weapon/camera/hud/throttle handlers removed from the menu,
    # but older configs are still read as fallback by the runtime.

    return _run_menu_from_json(spec_path=MENU_SPEC_PATH, start_node="main", action_handlers=handlers)


def run_controls_menu(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick,
    menu_button: int | None,
) -> str:
    # Legacy entry point kept for compatibility. This is now powered by menu.json.
    return run_main_menu(font=font, width=int(width), height=int(height), joystick=joystick, menu_button=menu_button)
