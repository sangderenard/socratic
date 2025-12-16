from __future__ import annotations

import json
import os
import shutil
import ctypes
import time
from typing import Any, Iterable
import sys

import pygame

import input_graph
import controller_backend
import reticle_sprite
import scroll_model


MENU_SPEC_PATH = "menu.json"


_CONTROLS_DIRTY = False


def mark_controls_dirty() -> None:
    global _CONTROLS_DIRTY
    _CONTROLS_DIRTY = True


def consume_controls_dirty() -> bool:
    global _CONTROLS_DIRTY
    v = bool(_CONTROLS_DIRTY)
    _CONTROLS_DIRTY = False
    return v


# Binding UX tuning: use the same reticle-confirm timing everywhere.
# The intention is to avoid accidental binds and make capture feel consistent.
BIND_LOCK_DELAY_S = 0.20
BIND_READY_DELAY_S = 0.45


_BIND_RETICLE_ASSETS: reticle_sprite.ReticleAssets | None = None


def _get_bind_reticle_assets(*, size_px: int = 64) -> reticle_sprite.ReticleAssets:
    global _BIND_RETICLE_ASSETS
    if _BIND_RETICLE_ASSETS is None:
        _BIND_RETICLE_ASSETS = reticle_sprite.load_reticle_assets(size_px=int(size_px))
    return _BIND_RETICLE_ASSETS


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
    except Exception as e:
        # If the file exists but is corrupted, try recovering from .bak and
        # preserve a copy for debugging instead of silently nuking it.
        try:
            msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
            print(f"[joystick_menu] Failed to load {path}: {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass

        bak_path = f"{path}.bak"
        if os.path.exists(bak_path):
            try:
                with open(bak_path, "r", encoding="utf-8") as f:
                    bak = json.load(f)
                if isinstance(bak, dict):
                    try:
                        print(f"[joystick_menu] Restoring from {bak_path}", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    _atomic_write_json(path, bak)
                    return bak
            except Exception:
                pass

        # Preserve the corrupt file for later inspection.
        try:
            ts = int(time.time())
            corrupt_path = f"{path}.corrupt.{ts}"
            if not os.path.exists(corrupt_path):
                shutil.copyfile(path, corrupt_path)
                print(f"[joystick_menu] Preserved corrupt config as {corrupt_path}", file=sys.stderr, flush=True)
        except Exception:
            pass

        _atomic_write_json(path, {})
        return {}


def save_joystick_config(cfg: dict[str, Any], path: str = "joystick.json") -> None:
    if not isinstance(cfg, dict):
        cfg = {}
    _atomic_write_json(path, cfg)
    try:
        if os.path.basename(str(path)).lower() == "joystick.json":
            mark_controls_dirty()
    except Exception:
        pass


def _menu_bindings_block(cfg: dict[str, Any]) -> dict[str, Any]:
    blk = cfg.get("menu_bindings")
    return blk if isinstance(blk, dict) else {}


def ensure_menu_bindings_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure cfg['menu_bindings'] exists.

    This block is used for bootstrapping menu input in a fresh config.
    It primarily lives outside the compiled controller graph.
    
    However, we also opportunistically seed a minimal set of *compiled* controller
    graph outputs for menu navigation (DirOR) and menu actions (Confirm/Cancel/Menu)
    so a fresh config can operate menus without running a blocking bootstrap.
    """
    if not isinstance(cfg.get("menu_bindings"), dict):
        cfg["menu_bindings"] = {}
    blk = cfg["menu_bindings"]
    # 2D channel index used for menu directional navigation (via controller graph).
    if not isinstance(blk.get("nav_channel_2d"), int):
        blk["nav_channel_2d"] = 1
    # Optional keyboard keys for menu toggle/accept/back.
    # Store as pygame keycodes (ints), or None if unbound.
    if blk.get("toggle_key") is not None and not isinstance(blk.get("toggle_key"), int):
        blk["toggle_key"] = None
    if blk.get("accept_key") is not None and not isinstance(blk.get("accept_key"), int):
        blk["accept_key"] = None
    if blk.get("back_key") is not None and not isinstance(blk.get("back_key"), int):
        blk["back_key"] = None

    # Also seed controller-graph menu outputs if missing.
    try:
        cfg = input_graph.ensure_controller_graph(cfg)
        fc = cfg.get("flight_controls") if isinstance(cfg.get("flight_controls"), dict) else None
        ctrl = fc.get("controller") if isinstance(fc, dict) else None
        if isinstance(ctrl, dict):
            sigs = ctrl.get("signals") if isinstance(ctrl.get("signals"), dict) else None
            chs = ctrl.get("channels") if isinstance(ctrl.get("channels"), dict) else None
            if isinstance(sigs, dict) and isinstance(chs, dict):
                # Stable channel indices for initial menu controls.
                CH_MENU_CONFIRM = 6
                CH_MENU_CANCEL = 7
                CH_MENU_OPEN = 8
                CH_MENU_RIGHT = 9
                CH_MENU_LEFT = 10
                CH_MENU_UP = 11
                CH_MENU_DOWN = 12

                def _ensure_signal(sid: str, node: dict[str, Any]) -> None:
                    if sid not in sigs or not isinstance(sigs.get(sid), dict):
                        sigs[str(sid)] = dict(node)

                def _ensure_channel(ch_idx: int, sid: str) -> None:
                    k = str(int(ch_idx))
                    if k not in chs or not isinstance(chs.get(k), dict):
                        chs[k] = {"source": "signal", "id": str(sid)}

                def _in_joybtn(b: int) -> dict[str, Any]:
                    return {"source": "input", "device": "joystick", "kind": "button", "id": int(b), "state": "D"}

                def _in_key(kc: int) -> dict[str, Any]:
                    return {"source": "input", "device": "keyboard", "kind": "key", "id": int(kc), "state": "D"}

                def _in_mbtn(b: int) -> dict[str, Any]:
                    return {"source": "input", "device": "mouse", "kind": "mouse_button", "id": int(b), "state": "D"}

                def _ensure_or_signal(name: str, inputs: list[dict[str, Any]]) -> str:
                    # Build an OR-like scalar by summing multiple button/key inputs.
                    # Hook watches use threshold>0.5, so any single active source trips it.
                    if str(name) in sigs and isinstance(sigs.get(str(name)), dict):
                        return str(name)
                    inputs2 = [it for it in inputs if isinstance(it, dict)]
                    if not inputs2:
                        _ensure_signal(str(name), {"op": "const", "dim": 1, "value": 0.0})
                        return str(name)
                    if len(inputs2) == 1:
                        _ensure_signal(str(name), {"op": "passthrough", "dim": 1, "args": [inputs2[0]]})
                        return str(name)
                    prev = f"{name}__0"
                    _ensure_signal(str(prev), {"op": "passthrough", "dim": 1, "args": [inputs2[0]]})
                    for i in range(1, len(inputs2) - 1):
                        sid_i = f"{name}__{i}"
                        _ensure_signal(str(sid_i), {"op": "add", "dim": 1, "args": [{"source": "signal", "id": str(prev)}, inputs2[i]]})
                        prev = str(sid_i)
                    _ensure_signal(str(name), {"op": "add", "dim": 1, "args": [{"source": "signal", "id": str(prev)}, inputs2[-1]]})
                    return str(name)

                # Direction signals derived from DirOR (signed). Hook thresholds are v>=thr,
                # so we invert negative directions via (0 - x/y) to make them positive.
                _ensure_signal("MenuZero", {"op": "const", "dim": 1, "value": 0.0})
                _ensure_signal(
                    "MenuNavRight",
                    {"op": "passthrough", "dim": 1, "args": [{"source": "signal", "id": "DirOR", "comp": "x"}]},
                )
                _ensure_signal(
                    "MenuNavUp",
                    {"op": "passthrough", "dim": 1, "args": [{"source": "signal", "id": "DirOR", "comp": "y"}]},
                )
                _ensure_signal(
                    "MenuNavLeft",
                    {"op": "sub", "dim": 1, "args": [{"source": "signal", "id": "MenuZero"}, {"source": "signal", "id": "DirOR", "comp": "x"}]},
                )
                _ensure_signal(
                    "MenuNavDown",
                    {"op": "sub", "dim": 1, "args": [{"source": "signal", "id": "MenuZero"}, {"source": "signal", "id": "DirOR", "comp": "y"}]},
                )

                # OR controls for the three menu actions.
                menu_btn = 7
                try:
                    if isinstance(cfg.get("menu_button"), int):
                        menu_btn = int(cfg.get("menu_button"))
                except Exception:
                    menu_btn = 7

                _ensure_or_signal(
                    "MenuConfirmOR",
                    [
                        _in_joybtn(0),
                        _in_key(int(pygame.K_RETURN)),
                        _in_key(int(getattr(pygame, "K_KP_ENTER", pygame.K_RETURN))),
                        _in_key(int(pygame.K_SPACE)),
                    ],
                )
                _ensure_or_signal(
                    "MenuCancelOR",
                    [
                        _in_joybtn(1),
                        _in_key(int(pygame.K_ESCAPE)),
                        _in_mbtn(3),
                    ],
                )
                _ensure_or_signal(
                    "MenuOpenOR",
                    [
                        _in_joybtn(int(menu_btn)),
                        _in_key(int(pygame.K_TAB)),
                        _in_key(int(pygame.K_e)),
                        _in_key(int(pygame.K_i)),
                        _in_key(int(pygame.K_m)),
                        _in_mbtn(2),
                    ],
                )

                # Publish controller outputs as channels.
                _ensure_channel(int(CH_MENU_CONFIRM), "MenuConfirmOR")
                _ensure_channel(int(CH_MENU_CANCEL), "MenuCancelOR")
                _ensure_channel(int(CH_MENU_OPEN), "MenuOpenOR")
                _ensure_channel(int(CH_MENU_RIGHT), "MenuNavRight")
                _ensure_channel(int(CH_MENU_LEFT), "MenuNavLeft")
                _ensure_channel(int(CH_MENU_UP), "MenuNavUp")
                _ensure_channel(int(CH_MENU_DOWN), "MenuNavDown")

                # Seed controller hook bindings for semantic menu actions.
                hb = ctrl.get("hook_bindings")
                if not isinstance(hb, list):
                    hb = []
                    ctrl["hook_bindings"] = hb

                existing_actions = {str(it.get("action")) for it in hb if isinstance(it, dict) and isinstance(it.get("action"), str)}

                def _ensure_hook(action: str, ch: int, *, edge: str = "rise", thr: float = 0.5, hyst: float = 0.08) -> None:
                    if str(action) in existing_actions:
                        return
                    hb.append(
                        {
                            "action": str(action),
                            "source": "channel",
                            "channel": int(ch),
                            "comp": 0,
                            "edge": str(edge),
                            "threshold": float(thr),
                            "hysteresis": float(hyst),
                            "dispatch": "async",
                        }
                    )
                    existing_actions.add(str(action))

                _ensure_hook("menu_open", int(CH_MENU_OPEN), edge="rise", thr=0.5, hyst=0.05)
                _ensure_hook("menu_confirm", int(CH_MENU_CONFIRM), edge="rise", thr=0.5, hyst=0.05)
                _ensure_hook("menu_cancel", int(CH_MENU_CANCEL), edge="rise", thr=0.5, hyst=0.05)
                _ensure_hook("menu_right", int(CH_MENU_RIGHT), edge="rise", thr=0.5, hyst=0.15)
                _ensure_hook("menu_left", int(CH_MENU_LEFT), edge="rise", thr=0.5, hyst=0.15)
                _ensure_hook("menu_up", int(CH_MENU_UP), edge="rise", thr=0.5, hyst=0.15)
                _ensure_hook("menu_down", int(CH_MENU_DOWN), edge="rise", thr=0.5, hyst=0.15)
    except Exception:
        pass
    return cfg


def menu_bindings_complete(cfg: dict[str, Any]) -> bool:
    """True if the user has a complete menu control set.

    Requirements (fresh-start):
    - Some way to toggle/open menu: either legacy menu_button, or menu_bindings.toggle_key.
    - Accept/back: either legacy menu_nav confirm/cancel, or menu_bindings.accept_key/back_key.
    - Directional nav: either legacy menu_nav up/down/left/right, or menu_bindings.nav_channel_2d.
    """
    # New path: controller-engine hook bindings can fully define the menu controls.
    ctrl_actions: set[str] = set()
    try:
        fc = cfg.get("flight_controls") if isinstance(cfg.get("flight_controls"), dict) else None
        ctrl = fc.get("controller") if isinstance(fc, dict) else None
        hb = ctrl.get("hook_bindings") if isinstance(ctrl, dict) else None
        if isinstance(hb, list):
            for it in hb:
                if isinstance(it, dict) and isinstance(it.get("action"), str):
                    ctrl_actions.add(str(it.get("action")))
    except Exception:
        ctrl_actions = set()

    # Toggle
    has_toggle = False
    if isinstance(cfg.get("menu_button"), int):
        has_toggle = True
    blk = _menu_bindings_block(cfg)
    if isinstance(blk.get("toggle_key"), int):
        has_toggle = True
    if "menu_open" in ctrl_actions:
        has_toggle = True

    # Accept/back
    nav = cfg.get("menu_nav") if isinstance(cfg.get("menu_nav"), dict) else {}
    has_accept = isinstance(nav.get("confirm"), dict) or isinstance(blk.get("accept_key"), int)
    has_back = isinstance(nav.get("cancel"), dict) or isinstance(blk.get("back_key"), int)
    if "menu_confirm" in ctrl_actions:
        has_accept = True
    if "menu_cancel" in ctrl_actions:
        has_back = True

    # Directions
    has_dirs = all(isinstance(nav.get(k), dict) for k in ("up", "down", "left", "right"))
    if not has_dirs:
        has_dirs = isinstance(blk.get("nav_channel_2d"), int)
    if {"menu_up", "menu_down", "menu_left", "menu_right"}.issubset(ctrl_actions):
        has_dirs = True

    return bool(has_toggle and has_accept and has_back and has_dirs)


def reset_action_bindings_keep_controller(*, cfg_path: str = "joystick.json") -> bool:
    """Wipe action↔input mappings but retain joystick device/signal/channel settings.

    Keeps:
    - flight_controls.controller (signals/channels/device_prefs/features/state)

    Clears:
    - flight_controls camera/triggers/weapons/sets/profiles/toggles/state beyond controller
    - legacy menu_nav + menu_button bindings

    Adds:
    - menu_bindings block (keyboard accept/back/toggle + nav_channel_2d)
    """
    cfg = load_or_create_joystick_config(str(cfg_path))
    if not isinstance(cfg, dict):
        cfg = {}

    # Preserve controller block if present.
    ctrl_blk: dict[str, Any] | None = None
    fc = cfg.get("flight_controls") if isinstance(cfg.get("flight_controls"), dict) else None
    if isinstance(fc, dict):
        cb = fc.get("controller")
        if isinstance(cb, dict):
            # Deep copy through json round-trip to avoid retaining shared references.
            try:
                ctrl_blk = json.loads(json.dumps(cb))
            except Exception:
                ctrl_blk = dict(cb)

    # Rebuild flight_controls with controller only.
    cfg["flight_controls"] = {"controller": (ctrl_blk or {})}

    # Clear legacy menu bindings.
    cfg.pop("menu_nav", None)
    cfg.pop("menu_scroll", None)
    cfg.pop("menu_button", None)

    # Fresh menu bindings scaffold (defaults show off 2D channel nav).
    cfg = ensure_menu_bindings_block(cfg)
    blk = cfg["menu_bindings"]
    blk["nav_channel_2d"] = 1
    blk["toggle_key"] = None
    blk["accept_key"] = None
    blk["back_key"] = None

    save_joystick_config(cfg, str(cfg_path))
    return True


def _draw_reticle_capture_screen(
    *,
    font: pygame.font.Font,
    label: str,
    label_x: float,
    label_y: float,
    ret_x: float,
    ret_y: float,
    stage: reticle_sprite.ReticleStage,
    assets: reticle_sprite.ReticleAssets,
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
        glPopMatrix,
        glPushMatrix,
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

    surf_lbl = _render_cell_text(font, str(label))
    lbl_data = pygame.image.tostring(surf_lbl, "RGBA", True)
    glRasterPos2f(float(label_x), float(label_y))
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


def _reticle_confirm_frame(
    *,
    font: pygame.font.Font,
    label: str,
    assets: reticle_sprite.ReticleAssets,
    anim: reticle_sprite.ReticleAnimator,
    clock: pygame.time.Clock,
    start: tuple[float, float],
    target: tuple[float, float],
    ret_x: float,
    ret_y: float,
    candidate_active: bool,
    label_x: float = 0.40,
    label_y: float = 0.52,
    lerp: float = 0.20,
) -> tuple[float, float, reticle_sprite.ReticleStage]:
    desired = target if bool(candidate_active) else start
    ret_x += (float(desired[0]) - float(ret_x)) * float(lerp)
    ret_y += (float(desired[1]) - float(ret_y)) * float(lerp)
    on_target = (abs(float(ret_x) - float(target[0])) <= 0.015) and (abs(float(ret_y) - float(target[1])) <= 0.020)
    now_s = pygame.time.get_ticks() * 0.001
    stage = anim.update(on_target=bool(on_target and candidate_active), now_s=float(now_s))
    _draw_reticle_capture_screen(
        font=font,
        label=str(label),
        label_x=float(label_x),
        label_y=float(label_y),
        ret_x=float(ret_x),
        ret_y=float(ret_y),
        stage=stage,
        assets=assets,
    )
    pygame.display.flip()
    clock.tick(60)
    return float(ret_x), float(ret_y), stage


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
    except Exception as e:
        try:
            msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
            print(f"[joystick_menu] Failed to load {path}: {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass

        bak_path = f"{path}.bak"
        if os.path.exists(bak_path):
            try:
                with open(bak_path, "r", encoding="utf-8") as f:
                    bak = json.load(f)
                if isinstance(bak, dict):
                    try:
                        print(f"[joystick_menu] Restoring from {bak_path}", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    _atomic_write_json(path, bak)
                    return bak
            except Exception:
                pass

        try:
            ts = int(time.time())
            corrupt_path = f"{path}.corrupt.{ts}"
            if not os.path.exists(corrupt_path):
                shutil.copyfile(path, corrupt_path)
                print(f"[joystick_menu] Preserved corrupt spec as {corrupt_path}", file=sys.stderr, flush=True)
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


def _get_menu_scroll(cfg: dict[str, Any]) -> dict[str, Any]:
        """Optional scroll bindings for long lists.

        If present, cfg['menu_scroll'] may contain binding objects keyed by:
            - scroll_up / scroll_down: line scroll
            - page_up / page_down: page scroll

        Binding objects use the same schema as menu_nav bindings.
        """
        sc = cfg.get("menu_scroll", None)
        return sc if isinstance(sc, dict) else {}


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
        ensure_menu_navigation_bindings._reticle_anim = reticle_sprite.ReticleAnimator(lock_delay_s=float(BIND_LOCK_DELAY_S), ready_delay_s=float(BIND_READY_DELAY_S))

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


def _poll_joystick_snapshot(
    joystick: pygame.joystick.Joystick,
    *,
    axes_ids: Iterable[int] | None = None,
    button_ids: Iterable[int] | None = None,
    hat_ids: Iterable[int] | None = None,
) -> tuple[dict[int, float], set[int], dict[int, tuple[int, int]]]:
    """Return (axes, buttons_down, hats).

    If ids are provided, poll only those ids. This is critical for announce-mode:
    callers can avoid consuming "everything" and instead poll only declared specs.
    """

    axes: dict[int, float] = {}
    buttons: set[int] = set()
    hats: dict[int, tuple[int, int]] = {}

    if axes_ids is None:
        try:
            n_axes = int(joystick.get_numaxes())
            for a in range(max(0, n_axes)):
                try:
                    axes[int(a)] = float(joystick.get_axis(a))
                except Exception:
                    pass
        except Exception:
            pass
    else:
        for a in axes_ids:
            try:
                axes[int(a)] = float(joystick.get_axis(int(a)))
            except Exception:
                pass

    if button_ids is None:
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
    else:
        for b in button_ids:
            try:
                if int(joystick.get_button(int(b))) != 0:
                    buttons.add(int(b))
            except Exception:
                pass

    if hat_ids is None:
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
    else:
        for h in hat_ids:
            try:
                v = joystick.get_hat(int(h))
                hats[int(h)] = (int(v[0]), int(v[1]))
            except Exception:
                hats[int(h)] = (0, 0)

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
    sc = _get_menu_scroll(cfg)
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
    scroll = scroll_model.ScrollModel(first_idx=0)

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

        # Optional manual scrolling (does not change selection).
        if _nav_edge(
            sc.get("scroll_up"),
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            scroll.scroll_lines(delta=-1, total=len(fields), max_visible=max_visible)
        if _nav_edge(
            sc.get("scroll_down"),
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            scroll.scroll_lines(delta=+1, total=len(fields), max_visible=max_visible)
        if _nav_edge(
            sc.get("page_up"),
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            scroll.scroll_pages(delta_pages=-1, total=len(fields), max_visible=max_visible)
        if _nav_edge(
            sc.get("page_down"),
            axes_now=axes_now,
            axes_prev=axes_prev,
            buttons_now=buttons_now,
            buttons_prev=buttons_prev,
            hats_now=hats_now,
            hats_prev=hats_prev,
        ):
            scroll.scroll_pages(delta_pages=+1, total=len(fields), max_visible=max_visible)

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

        scroll.ensure_visible(sel=int(sel), total=len(lines), max_visible=max_visible, center=False)
        first_idx, last_idx, visible_sel = scroll.window(total=len(lines), sel=int(sel), max_visible=max_visible)
        lines_vis = lines[int(first_idx) : int(last_idx)]

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
    start_node: str = "main",
    menu_context: dict[str, Any] | None = None,
) -> str:
    # Optional: use controller-engine-derived channel0 (DirOR) for navigation.
    # This uses the same C signal-kernel inputs as the rest of the program.
    sigk = None
    ctl = None
    ctl_meta = None
    ctl_outputs: dict[int, Any] = {}
    ctl_buf = None
    ctl_ready = False
    ctrl_backend = None
    try:
        from c_physics import signal_kernel_api

        _lib, api = signal_kernel_api.try_load_signal_kernel()
        sigk = api
        if sigk is not None:
            try:
                sigk.gp_sigk_reset()
            except Exception:
                pass

            # Prefer the C controller engine if present.
            try:
                import ctypes

                from c_physics import controller_engine_api
                from c_physics.controller_engine_ctypes import GP_CtlMeta, GP_CtlOutputDesc

                _lib2, ctl = controller_engine_api.try_load_controller_engine(search_dir="c_physics")
                if ctl is not None:
                    ok_load = int(ctl.gp_ctl_load_graph_file(b"controller_graph_final.bin"))
                    if ok_load:
                        # No initial passthrough here; menu nav should work off compiled channels.
                        if hasattr(ctl, "gp_ctl_passthru_clear"):
                            try:
                                ctl.gp_ctl_passthru_clear()
                            except Exception:
                                pass

                        meta = GP_CtlMeta()
                        if int(ctl.gp_ctl_get_meta(ctypes.byref(meta))) == 1:
                            n_out = int(meta.output_count)
                            outs: dict[int, Any] = {}
                            if n_out > 0:
                                arr_t = GP_CtlOutputDesc * n_out
                                arr = arr_t()
                                got = int(ctl.gp_ctl_get_outputs(arr, int(n_out)))
                                for i in range(max(0, got)):
                                    d = arr[int(i)]
                                    outs[int(d.channel)] = d
                            buf = None
                            if int(meta.total_floats) > 0:
                                buf = (ctypes.c_float * int(meta.total_floats))()
                            if buf is not None:
                                ctl_meta = meta
                                ctl_outputs = outs
                                ctl_buf = buf
                                ctl_ready = True
            except Exception:
                ctl = None
                ctl_meta = None
                ctl_outputs = {}
                ctl_buf = None
                ctl_ready = False

            # Python fallback backend (still consumes the C signal kernel).
            if not ctl_ready:
                ctrl_backend = controller_backend.ControllerBackend(sigk)
    except Exception:
        sigk = None
        ctl = None
        ctl_meta = None
        ctl_outputs = {}
        ctl_buf = None
        ctl_ready = False
        ctrl_backend = None

    # Keep ids small (<= 65535) because item_id is stored in 16 bits in the kernel signal_id.
    KBD_AXIS_WASD_X = 0
    KBD_AXIS_WASD_Y = 1
    KBD_AXIS_ARROWS_X = 2
    KBD_AXIS_ARROWS_Y = 3
    HARD_HAT_AXIS_BASE = 50000

    # Selected 2D channel used for menu navigation (updated from cfg["menu_bindings"]).
    nav_channel_2d = 0

    def _backend_nav_vec(*, joystick: pygame.joystick.Joystick) -> tuple[float, float] | None:
        if sigk is None or ctrl_backend is None:
            # Allow controller engine to drive nav even if Python backend is unavailable.
            if not (sigk is not None and ctl_ready and ctl is not None and ctl_meta is not None and ctl_buf is not None):
                return None
        try:
            from c_physics.signal_kernel_ctypes import GP_InputEvent
            from c_physics import signal_kernel_api

            # Input policy: (allow, deny) set model.
            # allow is None => allow all (scan/listen). allow is set() => allow none (quiet announce).
            try:
                import input_interest

                get_effective_set = input_interest.get_effective_set
            except Exception:

                def get_effective_set():  # type: ignore[no-redef]
                    return (None, set())

            now_ns = int(time.monotonic_ns())

            allow, deny = get_effective_set()
            allow_all = allow is None
            want_joy_axes_physical: set[int] | None = None
            want_hat_axes: set[int] | None = None
            want_hat_ids: set[int] | None = None
            want_kbd_axes: set[int] | None = None
            want_mouse_motion: set[int] | None = None

            if not allow_all:
                want_joy_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
                want_joy_axes_physical = {int(iid) for iid in want_joy_axes if 0 <= int(iid) < int(HARD_HAT_AXIS_BASE)}
                want_hat_axes = {int(iid) for iid in want_joy_axes if int(iid) >= int(HARD_HAT_AXIS_BASE)}
                want_hat_ids = {int((int(iid) - int(HARD_HAT_AXIS_BASE)) // 2) for iid in (want_hat_axes or set())}
                want_kbd_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
                want_mouse_motion = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_MOUSE) and int(k) == int(signal_kernel_api.GP_EV_MOUSE_MOTION)}

            if allow_all:
                axes_now, _buttons_now, hats_now = _poll_joystick_snapshot(joystick)
            else:
                axes_now, _buttons_now, hats_now = _poll_joystick_snapshot(
                    joystick,
                    axes_ids=want_joy_axes_physical,
                    button_ids=set(),
                    hat_ids=want_hat_ids,
                )

            evs: list[GP_InputEvent] = []

            # Joystick axes each frame.
            if allow_all:
                try:
                    na = int(joystick.get_numaxes())
                except Exception:
                    na = 0
                axis_iter = [int(a) for a in range(max(0, na))]
            else:
                axis_iter = sorted(list(want_joy_axes_physical or set()))
            for a in axis_iter:
                spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(a))
                if spec in deny:
                    continue
                vv = float((axes_now or {}).get(int(a), 0.0))
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(a),
                        v0=float(vv),
                        v1=0.0,
                        flags=0,
                    )
                )

            # Hat axes (virtual joystick axis ids).
            if allow_all:
                try:
                    nh = int(joystick.get_numhats())
                except Exception:
                    nh = 0
                hat_iter = [int(h) for h in range(max(0, nh))]
            else:
                hat_iter = sorted(list(want_hat_ids or set()))

            for h in hat_iter:
                hx, hy = (hats_now or {}).get(int(h), (0, 0))
                hx_id = int(HARD_HAT_AXIS_BASE + int(h) * 2 + 0)
                hy_id = int(HARD_HAT_AXIS_BASE + int(h) * 2 + 1)
                specx = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(hx_id))
                specy = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(hy_id))
                if specx not in deny and (allow_all or int(hx_id) in (want_hat_axes or set())):
                    evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(hx_id),
                            v0=float(int(hx)),
                            v1=0.0,
                            flags=0,
                        )
                    )
                if specy not in deny and (allow_all or int(hy_id) in (want_hat_axes or set())):
                    evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(hy_id),
                            v0=float(int(hy)),
                            v1=0.0,
                            flags=0,
                        )
                    )

            # Keyboard direction axes (WASD + arrows).
            pressed = None
            if allow_all or (want_kbd_axes and len(want_kbd_axes)):
                try:
                    pressed = pygame.key.get_pressed()
                except Exception:
                    pressed = None

            def _is_down(k: int) -> int:
                if pressed is None:
                    return 0
                try:
                    return 1 if bool(pressed[int(k)]) else 0
                except Exception:
                    return 0

            wasd_x = float(_is_down(pygame.K_d) - _is_down(pygame.K_a))
            wasd_y = float(_is_down(pygame.K_w) - _is_down(pygame.K_s))
            arrows_x = float(_is_down(pygame.K_RIGHT) - _is_down(pygame.K_LEFT))
            arrows_y = float(_is_down(pygame.K_UP) - _is_down(pygame.K_DOWN))

            spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(KBD_AXIS_WASD_X))
            if spec not in deny and (allow_all or int(KBD_AXIS_WASD_X) in (want_kbd_axes or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(KBD_AXIS_WASD_X),
                        v0=float(wasd_x),
                        v1=0.0,
                        flags=0,
                    )
                )
            spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(KBD_AXIS_WASD_Y))
            if spec not in deny and (allow_all or int(KBD_AXIS_WASD_Y) in (want_kbd_axes or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(KBD_AXIS_WASD_Y),
                        v0=float(wasd_y),
                        v1=0.0,
                        flags=0,
                    )
                )
            spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(KBD_AXIS_ARROWS_X))
            if spec not in deny and (allow_all or int(KBD_AXIS_ARROWS_X) in (want_kbd_axes or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(KBD_AXIS_ARROWS_X),
                        v0=float(arrows_x),
                        v1=0.0,
                        flags=0,
                    )
                )
            spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(KBD_AXIS_ARROWS_Y))
            if spec not in deny and (allow_all or int(KBD_AXIS_ARROWS_Y) in (want_kbd_axes or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(KBD_AXIS_ARROWS_Y),
                        v0=float(arrows_y),
                        v1=0.0,
                        flags=0,
                    )
                )

            # Mouse delta (dx/dy).
            mdx, mdy = 0, 0
            if allow_all or (want_mouse_motion and len(want_mouse_motion)):
                try:
                    mdx, mdy = pygame.mouse.get_rel()
                except Exception:
                    mdx, mdy = 0, 0
            spec = (int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), 0)
            if spec not in deny and (allow_all or 0 in (want_mouse_motion or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_MOUSE),
                        kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
                        id=0,
                        v0=float(mdx),
                        v1=0.0,
                        flags=0,
                    )
                )
            spec = (int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), 1)
            if spec not in deny and (allow_all or 1 in (want_mouse_motion or set())):
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_MOUSE),
                        kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
                        id=1,
                        v0=float(mdy),
                        v1=0.0,
                        flags=0,
                    )
                )

            if evs:
                arr_t = GP_InputEvent * len(evs)
                sigk.gp_sigk_push_events(arr_t(*evs), int(len(evs)))

            # Prefer selected 2D channel from the C controller engine.
            if ctl_ready and ctl is not None and ctl_meta is not None and ctl_buf is not None:
                try:
                    import ctypes

                    # Step once on the calling thread (menu runs before the main tick thread).
                    ctl.gp_ctl_step_once(int(now_ns))

                    d0 = ctl_outputs.get(int(nav_channel_2d))
                    if d0 is not None and int(getattr(d0, "dim", 0)) >= 2:
                        seq = ctypes.c_uint64(0)
                        t_ns = ctypes.c_uint64(0)
                        if int(ctl.gp_ctl_peek_latest(ctl_buf, int(ctl_meta.total_floats), ctypes.byref(seq), ctypes.byref(t_ns))) == 1:
                            off = int(getattr(d0, "offset", 0))
                            stride = int(getattr(d0, "stride", 1)) if int(getattr(d0, "stride", 1)) > 0 else 1
                            x = float(ctl_buf[off])
                            y = float(ctl_buf[off + stride])
                            return float(x), float(y)
                except Exception:
                    pass

            # Fallback: evaluate with the Python backend.
            if ctrl_backend is not None:
                _overrides_1d, channels_2d = ctrl_backend.step(now_ns=now_ns)
                if hasattr(sigk, "gp_sigk_clear_pulses"):
                    try:
                        sigk.gp_sigk_clear_pulses()
                    except Exception:
                        pass
                if isinstance(channels_2d, dict) and int(nav_channel_2d) in channels_2d:
                    x, y = channels_2d.get(int(nav_channel_2d), (0.0, 0.0))
                    return float(x), float(y)
            return None
        except Exception:
            return None

    def _vec_edges(v_now: tuple[float, float] | None, v_prev: tuple[float, float], *, thr: float = 0.65) -> tuple[bool, bool, bool, bool, tuple[float, float]]:
        if v_now is None:
            return False, False, False, False, (float(v_prev[0]), float(v_prev[1]))
        x, y = float(v_now[0]), float(v_now[1])
        px, py = float(v_prev[0]), float(v_prev[1])
        up = (y > thr) and (py <= thr)
        down = (y < -thr) and (py >= -thr)
        left = (x < -thr) and (px >= -thr)
        right = (x > thr) and (px <= thr)
        return bool(up), bool(down), bool(left), bool(right), (x, y)

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
        vec_prev = (0.0, 0.0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None

            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

            v_now = _backend_nav_vec(joystick=joystick)
            up_vec, down_vec, _left_vec, _right_vec, vec_prev = _vec_edges(v_now, vec_prev)

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
            if up_edge or up_vec:
                sel = (sel - 1) % max(1, len(options))
            if down_edge or down_vec:
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

    def _controller_report_single_press() -> None:
        """Read-only tool: capture one input actuation and report backend channel output.

        Does not modify joystick.json; purely informational.
        """
        if joystick is None:
            return

        # Ensure we have an up-to-date final graph so the backend can evaluate channels.
        try:
            import controller_graph_compile

            controller_graph_compile.try_build_final_graph(
                joystick_path="joystick.json",
                mixer_path="channel_mixer.json",
                compiled_out_path="controller_graph_compiled.json",
                final_out_path="controller_graph_final.json",
            )
        except Exception:
            pass

        # Load signal kernel + backend.
        try:
            from c_physics import signal_kernel_api
            from c_physics.signal_kernel_ctypes import GP_InputEvent
            import controller_backend
        except Exception:
            return

        _sigk_lib, sigk = signal_kernel_api.try_load_signal_kernel()
        if sigk is None:
            return
        try:
            sigk.gp_sigk_reset()
        except Exception:
            pass

        backend = controller_backend.ControllerBackend(sigk)

        clock = pygame.time.Clock()
        try:
            joystick.init()
        except Exception:
            pass

        # Baseline snapshot (stable reference for axis movement detection).
        try:
            init_axes, buttons_now0, init_hats = _poll_joystick_snapshot(joystick)
            buttons_prev = set(buttons_now0)
        except Exception:
            init_axes, buttons_prev, init_hats = {}, set(), {}

        # Rolling prev snapshot for edges.
        axes_prev = dict(init_axes)
        hats_prev = dict(init_hats)

        # 1) Wait for a new actuation.
        detected: dict[str, Any] | None = None
        detected_ns: int = 0
        while True:
            try:
                pygame.event.pump()
            except Exception:
                pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return

            now_ns = int(time.perf_counter_ns())
            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

            # Button press edge.
            new_presses = set(buttons_now) - set(buttons_prev)
            if new_presses:
                b = int(sorted(list(new_presses))[0])
                if menu_button is None or int(b) != int(menu_button):
                    detected = {"type": "button", "button": int(b)}
                    detected_ns = int(now_ns)
                    break

            # Hat change (from neutral).
            for h, xy in hats_now.items():
                if tuple(xy) != (0, 0) and tuple(hats_prev.get(int(h), (0, 0))) == (0, 0):
                    detected = {"type": "hat", "hat": int(h), "x": int(xy[0]), "y": int(xy[1])}
                    detected_ns = int(now_ns)
                    break
            if detected is not None:
                break

            # Axis moved sufficiently from baseline.
            for a, v in axes_now.items():
                v0 = float(init_axes.get(int(a), 0.0))
                if abs(float(v) - float(v0)) >= 0.35:
                    detected = {"type": "axis", "axis": int(a)}
                    detected_ns = int(now_ns)
                    break
            if detected is not None:
                break

            _draw_fullscreen_lines(
                font,
                int(width),
                int(height),
                [
                    "DISCOVER FEATURE (report)",
                    "actuate one control to sample backend output",
                    "(this does not save anything)",
                    "(menu button / Esc cancels)",
                ],
            )
            pygame.display.flip()
            clock.tick(60)

            axes_prev, buttons_prev, hats_prev = axes_now, set(buttons_now), dict(hats_now)

        if detected is None:
            return

        # 2) For a short window, feed kernel and measure channel changes.
        window_ns = int(350_000_000)  # ~0.35s
        base_overrides_1d: dict[int, float] = {}
        base_overrides_2d: dict[int, tuple[float, float]] = {}
        peak_delta: dict[int, float] = {}
        last_overrides_1d: dict[int, float] = {}
        last_overrides_2d: dict[int, tuple[float, float]] = {}
        start_ns = int(detected_ns)

        # Reset baseline for edges.
        try:
            axes_prev, buttons_now0, hats_prev = _poll_joystick_snapshot(joystick)
            buttons_prev = set(buttons_now0)
        except Exception:
            axes_prev, buttons_prev, hats_prev = {}, set(), {}

        while True:
            now_ns = int(time.perf_counter_ns())
            if int(now_ns) - int(start_ns) > int(window_ns):
                break

            try:
                pygame.event.pump()
            except Exception:
                pass

            axes_now, buttons_now, _hats_now = _poll_joystick_snapshot(joystick)

            evs: list[GP_InputEvent] = []
            # Button edges.
            for b in range(max(0, int(getattr(joystick, "get_numbuttons", lambda: 0)() or 0))):
                b = int(b)
                was = b in buttons_prev
                isd = b in buttons_now
                if was == isd:
                    continue
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                        kind=int(signal_kernel_api.GP_EV_BUTTON),
                        id=int(b),
                        v0=1.0 if isd else 0.0,
                        v1=0.0,
                        flags=0,
                    )
                )
            # Axis values.
            for a in range(max(0, int(getattr(joystick, "get_numaxes", lambda: 0)() or 0))):
                vv = float(axes_now.get(int(a), 0.0))
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(a),
                        v0=float(vv),
                        v1=0.0,
                        flags=0,
                    )
                )
            if evs:
                arr_t = GP_InputEvent * len(evs)
                try:
                    sigk.gp_sigk_push_events(arr_t(*evs), int(len(evs)))
                except Exception:
                    pass

            overrides_1d, channels_2d = backend.step(now_ns=int(now_ns))
            last_overrides_1d = dict(overrides_1d)
            last_overrides_2d = dict(channels_2d)

            if not base_overrides_1d and not base_overrides_2d:
                base_overrides_1d = dict(overrides_1d)
                base_overrides_2d = dict(channels_2d)

            for k, v in overrides_1d.items():
                base = float(base_overrides_1d.get(int(k), 0.0))
                d = abs(float(v) - float(base))
                peak_delta[int(k)] = float(max(float(peak_delta.get(int(k), 0.0)), float(d)))
            for k, (x, y) in channels_2d.items():
                bx, by = base_overrides_2d.get(int(k), (0.0, 0.0))
                d = float(max(abs(float(x) - float(bx)), abs(float(y) - float(by))))
                peak_delta[int(k)] = float(max(float(peak_delta.get(int(k), 0.0)), float(d)))

            axes_prev, buttons_prev = axes_now, set(buttons_now)
            clock.tick(120)

        # 3) Display report.
        lines: list[str] = []
        lines.append(f"input: {detected}")
        lines.append(f"backend: ok={backend.status.ok}  msg={backend.status.msg}")
        if last_overrides_1d or last_overrides_2d:
            parts: list[str] = []
            for k, v in sorted(last_overrides_1d.items()):
                parts.append(f"ch{int(k)}={float(v):+.3f}")
            for k, (x, y) in sorted(last_overrides_2d.items()):
                parts.append(f"ch{int(k)}=({float(x):+.2f},{float(y):+.2f})")
            lines.append("channels (last): " + ", ".join(parts))
        else:
            lines.append("channels (last): (no overrides)")

        if peak_delta:
            act = [(k, d) for k, d in peak_delta.items() if float(d) >= 0.05]
            act.sort(key=lambda kv: kv[1], reverse=True)
            if act:
                lines.append("peak delta (~0.35s): " + ", ".join(f"ch{int(k)}={float(d):.3f}" for k, d in act[:8]))
        lines.append("(menu button / Esc exits)")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return
            _draw_fullscreen_lines(font, int(width), int(height), ["DISCOVER FEATURE (report)"] + lines)
            pygame.display.flip()
            clock.tick(60)

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

        assets = _get_bind_reticle_assets(size_px=64)
        anim = reticle_sprite.ReticleAnimator(lock_delay_s=float(BIND_LOCK_DELAY_S), ready_delay_s=float(BIND_READY_DELAY_S))

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

            ret_x, ret_y, stage = _reticle_confirm_frame(
                font=font,
                label=str(label),
                assets=assets,
                anim=anim,
                clock=clock,
                start=(float(start[0]), float(start[1])),
                target=(float(target[0]), float(target[1])),
                ret_x=float(ret_x),
                ret_y=float(ret_y),
                candidate_active=bool(candidate_active),
                label_x=float(x_lbl),
                label_y=float(y_lbl),
            )

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

        assets = _get_bind_reticle_assets(size_px=64)
        anim = reticle_sprite.ReticleAnimator(lock_delay_s=float(BIND_LOCK_DELAY_S), ready_delay_s=float(BIND_READY_DELAY_S))

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

            ret_x, ret_y, stage = _reticle_confirm_frame(
                font=font,
                label=str(label),
                assets=assets,
                anim=anim,
                clock=clock,
                start=(float(start[0]), float(start[1])),
                target=(float(target[0]), float(target[1])),
                ret_x=float(ret_x),
                ret_y=float(ret_y),
                candidate_active=bool(candidate_active),
                label_x=float(x_lbl),
                label_y=float(y_lbl),
            )

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

        assets = _get_bind_reticle_assets(size_px=64)
        anim = reticle_sprite.ReticleAnimator(lock_delay_s=float(BIND_LOCK_DELAY_S), ready_delay_s=float(BIND_READY_DELAY_S))

        x_lbl = 0.40
        y_lbl = 0.52
        start = (x_lbl, y_lbl - 0.16)
        target = (x_lbl, y_lbl)
        ret_x, ret_y = float(start[0]), float(start[1])

        candidate: dict[str, int] | None = None
        candidate_active = False
        candidate_init_axis: float = 0.0
        
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

            # Candidate selection (only if none chosen yet).
            if candidate is None:
                new_presses = polled_buttons - last_buttons
                if new_presses:
                    b = int(sorted(list(new_presses))[0])
                    if menu_button is None or b != int(menu_button):
                        candidate = {"type": "button", "button": int(b)}

                if candidate is None:
                    for h, v in polled_hats.items():
                        if v != (0, 0) and last_hats.get(int(h), (0, 0)) == (0, 0):
                            candidate = {"type": "hat", "hat": int(h), "x": int(v[0]), "y": int(v[1])}
                            break

                if candidate is None:
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
                            candidate = {"type": "axis", "axis": int(a), "sign": int(sign)}
                            candidate_init_axis = float(v0)
                            break

            # Candidate active tracking.
            candidate_active = False
            if candidate is not None:
                t = str(candidate.get("type", ""))
                if t == "button":
                    btn = int(candidate.get("button", -1))
                    candidate_active = int(btn) in polled_buttons
                elif t == "hat":
                    h = int(candidate.get("hat", -1))
                    cx = int(candidate.get("x", 0))
                    cy = int(candidate.get("y", 0))
                    candidate_active = polled_hats.get(int(h), (0, 0)) == (int(cx), int(cy))
                elif t == "axis":
                    a = int(candidate.get("axis", -1))
                    s = int(candidate.get("sign", 0))
                    try:
                        v = float(joystick.get_axis(int(a)))
                    except Exception:
                        v = float(init_axis.get(int(a), 0.0))
                    dv = (float(v) - float(candidate_init_axis)) * float(s)
                    candidate_active = dv >= float(delta_threshold)

            # Draw + confirm via the standardized reticle lock/ready sequence.
            ret_x, ret_y, stage = _reticle_confirm_frame(
                font=font,
                label=f"bind {label}",
                assets=assets,
                anim=anim,
                clock=clock,
                start=(float(start[0]), float(start[1])),
                target=(float(target[0]), float(target[1])),
                ret_x=float(ret_x),
                ret_y=float(ret_y),
                candidate_active=bool(candidate_active),
                label_x=float(x_lbl),
                label_y=float(y_lbl),
            )

            if candidate is not None and candidate_active and stage == reticle_sprite.ReticleStage.READY:
                return dict(candidate)

            # If they released before confirmation, reset candidate.
            if candidate is not None and not candidate_active:
                candidate = None
                anim._on_since = None

            # Advance previous snapshots so we detect *new* presses/hat moves.
            last_buttons = polled_buttons
            last_hats = polled_hats

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
        if joystick is None:
            return "start"

        spec = load_or_create_menu_spec(spec_path)
        node_id = str(start_node)
        stack: list[str] = []

        cfg = load_or_create_joystick_config("joystick.json")
        cfg = ensure_menu_bindings_block(cfg)
        blk0 = _menu_bindings_block(cfg)

        nonlocal nav_channel_2d
        try:
            nav_channel_2d = int(blk0.get("nav_channel_2d") or 0)
        except Exception:
            nav_channel_2d = 0

        toggle_key: int | None = int(blk0.get("toggle_key")) if isinstance(blk0.get("toggle_key"), int) else None
        accept_key: int | None = int(blk0.get("accept_key")) if isinstance(blk0.get("accept_key"), int) else None
        back_key: int | None = int(blk0.get("back_key")) if isinstance(blk0.get("back_key"), int) else None

        # Fresh-start: if menu bindings are incomplete, open the binding-tasks tool first.
        if not menu_bindings_complete(cfg):
            try:
                if isinstance(action_handlers, dict) and callable(action_handlers.get("menu_binding_tasks")):
                    action_handlers["menu_binding_tasks"]()
            except Exception:
                pass
            cfg = load_or_create_joystick_config("joystick.json")
            cfg = ensure_menu_bindings_block(cfg)
            blk0 = _menu_bindings_block(cfg)
            try:
                nav_channel_2d = int(blk0.get("nav_channel_2d") or 0)
            except Exception:
                nav_channel_2d = 0
            toggle_key = int(blk0.get("toggle_key")) if isinstance(blk0.get("toggle_key"), int) else None
            accept_key = int(blk0.get("accept_key")) if isinstance(blk0.get("accept_key"), int) else None
            back_key = int(blk0.get("back_key")) if isinstance(blk0.get("back_key"), int) else None
            if not menu_bindings_complete(cfg):
                return "start"

        nav = _get_menu_nav(cfg)
        sc = _get_menu_scroll(cfg)
        b_up = nav.get("up")
        b_down = nav.get("down")
        b_confirm = nav.get("confirm")
        b_cancel = nav.get("cancel")

        clock = pygame.time.Clock()
        sel = 0

        max_visible = 10

        scroll = scroll_model.ScrollModel(first_idx=0)

        axes_prev: dict[int, float] = {}
        buttons_prev: set[int] = set()
        hats_prev: dict[int, tuple[int, int]] = {}
        vec_prev = (0.0, 0.0)

        ret_x = 0.04
        ret_y = 0.66
        start_x = 0.04
        target_x = 0.10

        anim = reticle_sprite.ReticleAnimator(lock_delay_s=float(BIND_LOCK_DELAY_S), ready_delay_s=float(BIND_READY_DELAY_S))

        def _sel_y(idx_visible: int) -> float:
            return float(0.66 - 0.07 * int(idx_visible))

        while True:
            toggle_edge = False
            back_edge = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if toggle_key is not None and int(event.key) == int(toggle_key):
                        toggle_edge = True
                    if back_key is not None and int(event.key) == int(back_key):
                        back_edge = True
                if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                    return "start"
            if toggle_edge:
                return "start"

            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)

            v_now = _backend_nav_vec(joystick=joystick)
            up_vec, down_vec, _left_vec, _right_vec, vec_prev = _vec_edges(v_now, vec_prev)

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
            ) or up_vec:
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
            ) or down_vec:
                sel = (sel + 1) % max(1, len(labels))
                anim._on_since = None

            # Optional manual list scrolling (does not change selection).
            if _nav_edge(
                sc.get("scroll_up"),
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                scroll.scroll_lines(delta=-1, total=len(labels), max_visible=max_visible)
            if _nav_edge(
                sc.get("scroll_down"),
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                scroll.scroll_lines(delta=+1, total=len(labels), max_visible=max_visible)
            if _nav_edge(
                sc.get("page_up"),
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                scroll.scroll_pages(delta_pages=-1, total=len(labels), max_visible=max_visible)
            if _nav_edge(
                sc.get("page_down"),
                axes_now=axes_now,
                axes_prev=axes_prev,
                buttons_now=buttons_now,
                buttons_prev=buttons_prev,
                hats_now=hats_now,
                hats_prev=hats_prev,
            ):
                scroll.scroll_pages(delta_pages=+1, total=len(labels), max_visible=max_visible)

            if back_edge or _nav_edge(
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
                    scroll.first_idx = 0
                    anim._on_since = None
                else:
                    return "start"

            confirm_held = _nav_is_held(b_confirm, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)
            if accept_key is not None:
                try:
                    kp = pygame.key.get_pressed()
                    if kp is not None and bool(kp[int(accept_key)]):
                        confirm_held = True
                except Exception:
                    pass

            scroll.ensure_visible(sel=int(sel), total=len(labels), max_visible=max_visible, center=False)
            first_idx, last_idx, visible_sel = scroll.window(total=len(labels), sel=int(sel), max_visible=max_visible)
            labels_vis = labels[int(first_idx) : int(last_idx)]

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
                    scroll.first_idx = 0
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

                # If any handlers wrote joystick.json, let the caller soft-refresh without restarting.
                try:
                    if consume_controls_dirty():
                        ctx = menu_context if isinstance(menu_context, dict) else {}
                        cb = ctx.get("on_controls_changed")
                        if callable(cb):
                            cb()
                except Exception:
                    pass

                # Reload nav bindings in case the user rebound anything.
                cfg = load_or_create_joystick_config("joystick.json")
                cfg = ensure_menu_bindings_block(cfg)
                blk2 = _menu_bindings_block(cfg)
                try:
                    nav_channel_2d = int(blk2.get("nav_channel_2d") or 0)
                except Exception:
                    nav_channel_2d = 0
                toggle_key = int(blk2.get("toggle_key")) if isinstance(blk2.get("toggle_key"), int) else None
                accept_key = int(blk2.get("accept_key")) if isinstance(blk2.get("accept_key"), int) else None
                back_key = int(blk2.get("back_key")) if isinstance(blk2.get("back_key"), int) else None
                nav = _get_menu_nav(cfg)
                sc = _get_menu_scroll(cfg)
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

    def _run_in_scan(fn) -> Any:
        try:
            import input_interest

            with input_interest.scan_mode():
                return fn()
        except Exception:
            return fn()

    def _mk_bind_handler(label: str, *, section: str, key: str, threshold: float) -> Any:
        def _h() -> None:
            def _do() -> None:
                a = _bind_axis(label, joystick=joystick, threshold=float(threshold))
                if a is not None and int(a) >= 0:
                    _write_flight_axis(section=section, key=key, axis=int(a))

            _run_in_scan(_do)
        return _h

    handlers: dict[str, Any] = {
        "bind_rudder": _mk_bind_handler("flight rudder", section="left_stick", key="rudder_axis", threshold=0.85),
        "bind_elevator": _mk_bind_handler("flight elevator", section="left_stick", key="elevator_axis", threshold=0.85),
        "bind_view_yaw": _mk_bind_handler("view yaw", section="right_stick", key="view_yaw_axis", threshold=0.85),
        "bind_view_pitch": _mk_bind_handler("view pitch", section="right_stick", key="view_pitch_axis", threshold=0.85),
    }

    # Controller graph menu actions.
    # DISCOVER FEATURE is repurposed as a read-only report tool (no config writes).
    handlers["controller_discover_feature"] = lambda: _controller_report_single_press()
    handlers["controller_create_signal"] = lambda: _controller_create_signal()
    handlers["controller_map_channel"] = lambda: _controller_map_channel()

    def _controller_view_channel() -> None:
        try:
            import channel_viewer
        except Exception:
            return
        channel_viewer.run_channel_viewer(
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
            menu_button=menu_button,
            load_or_create_joystick_config=load_or_create_joystick_config,
            get_menu_nav=_get_menu_nav,
            get_menu_scroll=_get_menu_scroll,
            nav_edge=_nav_edge,
            poll_joystick_snapshot=_poll_joystick_snapshot,
        )

    def _controller_workbench() -> None:
        try:
            import signal_workbench
        except Exception:
            return

        def _do() -> None:
            signal_workbench.run_signal_workbench(
                font=font,
                width=int(width),
                height=int(height),
                joystick=joystick,
                menu_button=menu_button,
                load_or_create_joystick_config=load_or_create_joystick_config,
                save_joystick_config=save_joystick_config,
                get_menu_nav=_get_menu_nav,
                get_menu_scroll=_get_menu_scroll,
                nav_edge=_nav_edge,
                poll_joystick_snapshot=_poll_joystick_snapshot,
                draw_fullscreen_lines=_draw_fullscreen_lines,
            )

        _do()

    def _controller_channel_mixer() -> None:
        try:
            import channel_mixer_menu
        except Exception:
            return
        channel_mixer_menu.run_channel_mixer_menu(
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
            menu_button=menu_button,
            get_menu_nav=_get_menu_nav,
            nav_edge=_nav_edge,
            poll_joystick_snapshot=_poll_joystick_snapshot,
            draw_fullscreen_lines=_draw_fullscreen_lines,
        )

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
    handlers["controller_workbench"] = lambda: _controller_workbench()
    handlers["controller_channel_mixer"] = lambda: _controller_channel_mixer()
    handlers["controller_view_channel"] = lambda: _controller_view_channel()

    def _menu_binding_tasks() -> None:
        if joystick is None:
            return

        try:
            import ui_tables
        except Exception:
            return

        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_SRC_ALPHA,
            GL_TRIANGLES,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_BYTE,
            GL_RGBA,
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

        def _px_to_nx(x: int) -> float:
            return float(int(x)) / float(max(1, int(width)))

        def _px_to_ny(y: int) -> float:
            return float(max(0, int(height) - int(y))) / float(max(1, int(height)))

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

        def _available_2d_channels() -> list[int]:
            out: list[int] = []
            try:
                for ch, d in (ctl_outputs or {}).items():
                    if int(getattr(d, "dim", 0)) >= 2:
                        out.append(int(ch))
            except Exception:
                out = []
            if not out:
                out = [0]
            out = sorted(list(set(int(x) for x in out)))
            return out

        def _fmt_key(k: int | None) -> str:
            if k is None:
                return "<UNBOUND>"
            try:
                return str(pygame.key.name(int(k))).upper()
            except Exception:
                return str(int(k))

        def _fmt_legacy_binding(b: dict[str, Any] | None) -> str:
            if not isinstance(b, dict):
                return "<UNBOUND>"
            t = b.get("type")
            if t == "button":
                bb = b.get("button")
                return f"JOY BTN {int(bb)}" if isinstance(bb, int) else "<UNBOUND>"
            if t == "axis":
                aa = b.get("axis")
                ss = b.get("sign")
                if isinstance(aa, int) and isinstance(ss, int):
                    return f"JOY AXIS {int(aa)} {'+' if int(ss) > 0 else '-'}"
                return "<UNBOUND>"
            if t == "hat":
                hh = b.get("hat")
                xx = b.get("x")
                yy = b.get("y")
                if isinstance(hh, int) and isinstance(xx, int) and isinstance(yy, int):
                    return f"JOY HAT {int(hh)} ({int(xx)},{int(yy)})"
                return "<UNBOUND>"
            return "<UNBOUND>"

        def _capture_key(*, title: str) -> int | None:
            clock = pygame.time.Clock()
            key: int | None = None
            held_since: float | None = None
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if int(event.key) in (pygame.K_ESCAPE, pygame.K_q):
                            return None
                        key = int(event.key)
                        held_since = float(pygame.time.get_ticks()) * 0.001

                if key is not None:
                    try:
                        kp = pygame.key.get_pressed()
                    except Exception:
                        kp = None
                    if kp is None or not bool(kp[int(key)]):
                        key = None
                        held_since = None
                    else:
                        now_s = float(pygame.time.get_ticks()) * 0.001
                        if held_since is not None and (now_s - float(held_since)) >= float(BIND_READY_DELAY_S):
                            return int(key)

                _draw_fullscreen_lines(font, int(width), int(height), [
                    "BIND KEY (HOLD TO COMMIT)",
                    str(title),
                    "hold a key to commit", 
                    "ESC cancels",
                ])
                pygame.display.flip()
                clock.tick(60)

        def _save_menu_bindings(*, nav_ch: int | None = None, toggle: int | None = None, accept: int | None = None, back: int | None = None) -> None:
            cfg = load_or_create_joystick_config("joystick.json")
            cfg = ensure_menu_bindings_block(cfg)
            blk = cfg["menu_bindings"]
            if nav_ch is not None:
                blk["nav_channel_2d"] = int(nav_ch)
            if toggle is not None:
                blk["toggle_key"] = int(toggle)
            if accept is not None:
                blk["accept_key"] = int(accept)
            if back is not None:
                blk["back_key"] = int(back)
            save_joystick_config(cfg, "joystick.json")

        # Table state
        clock = pygame.time.Clock()
        dropdown_open = False
        dropdown_x0 = 0
        dropdown_y0 = 0
        dropdown_w = 0
        dropdown_row_h = 0

        while True:
            # Load current bindings.
            cfg = load_or_create_joystick_config("joystick.json")
            cfg = ensure_menu_bindings_block(cfg)
            blk = _menu_bindings_block(cfg)
            cur_nav = int(blk.get("nav_channel_2d") or 0) if isinstance(blk.get("nav_channel_2d"), int) else 0
            cur_toggle = int(blk.get("toggle_key")) if isinstance(blk.get("toggle_key"), int) else None
            cur_accept = int(blk.get("accept_key")) if isinstance(blk.get("accept_key"), int) else None
            cur_back = int(blk.get("back_key")) if isinstance(blk.get("back_key"), int) else None

            # Keep menu-nav channel selection consistent while this screen is open.
            nonlocal nav_channel_2d
            nav_channel_2d = int(cur_nav)

            # Poll joystick snapshot for legacy binding status.
            axes_now, buttons_now, hats_now = _poll_joystick_snapshot(joystick)
            legacy_menu_btn = cfg.get("menu_button")
            legacy_nav = _get_menu_nav(cfg)
            legacy_confirm = legacy_nav.get("confirm") if isinstance(legacy_nav, dict) else None
            legacy_cancel = legacy_nav.get("cancel") if isinstance(legacy_nav, dict) else None

            # Feed kernel so channel outputs update (uses nav_channel_2d).
            v_now = _backend_nav_vec(joystick=joystick)
            vx, vy = (0.0, 0.0)
            if isinstance(v_now, tuple) and len(v_now) >= 2:
                vx, vy = float(v_now[0]), float(v_now[1])

            try:
                kp = pygame.key.get_pressed()
            except Exception:
                kp = None

            def _is_key_down(k: int | None) -> bool:
                if kp is None or k is None:
                    return False
                try:
                    return bool(kp[int(k)])
                except Exception:
                    return False

            def _toggle_used() -> bool:
                if cur_toggle is not None:
                    return _is_key_down(cur_toggle)
                if isinstance(legacy_menu_btn, int):
                    return int(legacy_menu_btn) in buttons_now
                return False

            def _accept_used() -> bool:
                if cur_accept is not None:
                    return _is_key_down(cur_accept)
                return _nav_is_held(legacy_confirm if isinstance(legacy_confirm, dict) else None, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)

            def _back_used() -> bool:
                if cur_back is not None:
                    return _is_key_down(cur_back)
                return _nav_is_held(legacy_cancel if isinstance(legacy_cancel, dict) else None, axes_now=axes_now, buttons_now=buttons_now, hats_now=hats_now)

            def _toggle_disp() -> str:
                if cur_toggle is not None:
                    return _fmt_key(cur_toggle)
                if isinstance(legacy_menu_btn, int):
                    return f"JOY BTN {int(legacy_menu_btn)}"
                return "<UNBOUND>"

            def _accept_disp() -> str:
                if cur_accept is not None:
                    return _fmt_key(cur_accept)
                return _fmt_legacy_binding(legacy_confirm if isinstance(legacy_confirm, dict) else None)

            def _back_disp() -> str:
                if cur_back is not None:
                    return _fmt_key(cur_back)
                return _fmt_legacy_binding(legacy_cancel if isinstance(legacy_cancel, dict) else None)

            # Rows
            rows = [
                {"kind": "key", "label": "TOGGLE MENU", "field": "toggle_key", "disp": _toggle_disp(), "used": _toggle_used()},
                {"kind": "key", "label": "ACCEPT", "field": "accept_key", "disp": _accept_disp(), "used": _accept_used()},
                {"kind": "key", "label": "BACK", "field": "back_key", "disp": _back_disp(), "used": _back_used()},
                {"kind": "chan2d", "label": "NAV CHANNEL (2D)", "field": "nav_channel_2d", "value": cur_nav, "used": (abs(float(vx)) > 0.65 or abs(float(vy)) > 0.65)},
            ]

            # Input
            mx, my = pygame.mouse.get_pos()
            mouse_clicked = False
            click_pos = (0, 0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and int(event.key) in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and int(event.button) == 1:
                    mouse_clicked = True
                    click_pos = (int(event.pos[0]), int(event.pos[1]))

            # Render
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
            row_h = max(18, int(font.get_linesize()) + 6)
            geom = ui_tables.TableGeom(x0=0, y0=0, w=int(width), h=int(height), pad=int(pad), header_h=int(header_h), row_h=int(row_h))

            _draw_text_px(pad, pad + header_h, "MENU BINDING TASKS")
            _draw_text_px(pad, pad + header_h + int(row_h * 0.9), "click a row to bind / select")

            col_label_w = int(width * 0.55)
            col_value_w = int(width * 0.25)
            col_used_w = int(width - pad * 2 - col_label_w - col_value_w)

            x_label = pad
            x_value = pad + col_label_w
            x_used = pad + col_label_w + col_value_w

            # Header row
            y_hdr = pad + header_h + 2
            _draw_text_px(x_label, y_hdr, "control")
            _draw_text_px(x_value, y_hdr, "binding")
            _draw_text_px(x_used, y_hdr, "used")

            hit_row: int | None = None
            hit_kind: str | None = None

            for idx, row, y, _mv in ui_tables.iter_visible_rows(rows=rows, sel_idx=0, g=geom, scroll=scroll_model.ScrollModel(first_idx=0), center=False):
                y0 = int(y - row_h + 2)
                _draw_rect_px(2, int(y0), int(width - 4), int(row_h), (0.06, 0.06, 0.06, 1.0))
                _draw_rect_px(2, int(y0), int(width - 4), 1, (0.25, 0.25, 0.25, 1.0))
                _draw_text_px(x_label, int(y), str(row.get("label", "")))
                if str(row.get("kind")) == "key":
                    _draw_text_px(x_value, int(y), str(row.get("disp") or "<UNBOUND>"))
                else:
                    _draw_text_px(x_value, int(y), f"CH {int(row.get('value') or 0)}")

                used = bool(row.get("used"))
                led_col = (0.2, 0.9, 0.2, 1.0) if used else (0.2, 0.2, 0.2, 1.0)
                _draw_rect_px(x_used, int(y0 + 4), 18, 18, led_col)

                if mouse_clicked:
                    cx, cy = int(click_pos[0]), int(click_pos[1])
                    if int(y0) <= cy <= int(y0 + row_h) and 2 <= cx <= int(width - 2):
                        hit_row = int(idx)
                        hit_kind = str(row.get("kind"))

            # DONE button
            done_w = 160
            done_h = 36
            done_x = int(width - done_w - 16)
            done_y = int(height - done_h - 16)
            _draw_rect_px(done_x, done_y, done_w, done_h, (0.10, 0.10, 0.10, 1.0))
            _draw_rect_px(done_x, done_y, done_w, 1, (0.35, 0.35, 0.35, 1.0))
            _draw_text_px(done_x + 16, done_y + 24, "DONE")

            if mouse_clicked:
                cx, cy = int(click_pos[0]), int(click_pos[1])
                if done_x <= cx <= done_x + done_w and done_y <= cy <= done_y + done_h:
                    glDisable(GL_BLEND)
                    if depth_was_enabled:
                        glEnable(GL_DEPTH_TEST)
                    glPopMatrix()
                    glMatrixMode(GL_PROJECTION)
                    glPopMatrix()
                    glMatrixMode(GL_MODELVIEW)
                    return

            # Dropdown for channels (simple list near clicked row)
            if dropdown_open:
                chans = _available_2d_channels()
                box_h = int(len(chans) * dropdown_row_h)
                _draw_rect_px(dropdown_x0, dropdown_y0, dropdown_w, box_h, (0.08, 0.08, 0.08, 1.0))
                for i, ch in enumerate(chans):
                    yy = int(dropdown_y0 + i * dropdown_row_h)
                    _draw_text_px(dropdown_x0 + 8, yy + int(dropdown_row_h * 0.7), f"CH {int(ch)}")
                if mouse_clicked:
                    cx, cy = int(click_pos[0]), int(click_pos[1])
                    if dropdown_x0 <= cx <= dropdown_x0 + dropdown_w and dropdown_y0 <= cy <= dropdown_y0 + box_h:
                        i = int((cy - dropdown_y0) / max(1, dropdown_row_h))
                        if 0 <= i < len(chans):
                            _save_menu_bindings(nav_ch=int(chans[int(i)]))
                    dropdown_open = False

            # Apply click actions (after draw so we can place dropdown relative to layout)
            if mouse_clicked and hit_row is not None:
                row = rows[int(hit_row)]
                if str(hit_kind) == "key":
                    k = _capture_key(title=str(row.get("label", "")))
                    if k is not None:
                        field = str(row.get("field"))
                        if field == "toggle_key":
                            _save_menu_bindings(toggle=int(k))
                        elif field == "accept_key":
                            _save_menu_bindings(accept=int(k))
                        elif field == "back_key":
                            _save_menu_bindings(back=int(k))
                elif str(hit_kind) == "chan2d":
                    dropdown_open = True
                    dropdown_w = int(width * 0.22)
                    dropdown_row_h = int(row_h)
                    dropdown_x0 = int(width * 0.60)
                    dropdown_y0 = int(height * 0.18)

            glDisable(GL_BLEND)
            if depth_was_enabled:
                glEnable(GL_DEPTH_TEST)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

            pygame.display.flip()
            clock.tick(60)

    handlers["menu_binding_tasks"] = lambda: _run_in_scan(_menu_binding_tasks)

    def _reset_action_bindings_keep_controller() -> None:
        reset_action_bindings_keep_controller(cfg_path="joystick.json")
        _draw_fullscreen_lines(font, int(width), int(height), [
            "RESET COMPLETE",
            "action bindings wiped",
            "controller graph preserved",
            "press ESC to return",
        ])
        pygame.display.flip()
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and int(event.key) in (pygame.K_ESCAPE, pygame.K_q):
                    return
            clock.tick(60)

    handlers["reset_action_bindings_keep_controller"] = lambda: _reset_action_bindings_keep_controller()

    # Menus: bind menu navigation explicitly.
    handlers["bind_menu_navigation"] = lambda: _run_in_scan(
        lambda: ensure_menu_navigation_bindings(
            cfg_path="joystick.json",
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
            menu_button=menu_button,
        )
    )

    handlers["bind_menu_button"] = lambda: _run_in_scan(
        lambda: ensure_menu_button_binding(
            cfg_path="joystick.json",
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
            force=True,
        )
    )

    def _bind_menu_key(k: str) -> None:
        def _do() -> None:
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

        _run_in_scan(_do)

    handlers["bind_menu_up"] = lambda: _bind_menu_key("up")
    handlers["bind_menu_down"] = lambda: _bind_menu_key("down")
    handlers["bind_menu_left"] = lambda: _bind_menu_key("left")
    handlers["bind_menu_right"] = lambda: _bind_menu_key("right")
    handlers["bind_menu_confirm"] = lambda: _bind_menu_key("confirm")
    handlers["bind_menu_back"] = lambda: _bind_menu_key("cancel")

    # Control-set bindings.
    def _bind_set_axis1d(*, label: str, set_name: str, group: str, key: str) -> None:
        def _do() -> None:
            m = _bind_axis_calibrated(label, joystick=joystick, threshold=0.85)
            if isinstance(m, dict):
                _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

        _run_in_scan(_do)

    def _bind_set_axis2d(*, label: str, set_name: str, group: str, key: str) -> None:
        def _do() -> None:
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

        _run_in_scan(_do)

    def _bind_set_buttonish(*, label: str, set_name: str, group: str, key: str) -> None:
        def _do() -> None:
            m = _bind_buttonish(label, joystick=joystick)
            if isinstance(m, dict):
                _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

        _run_in_scan(_do)

    def _bind_set_trigger(*, label: str, set_name: str, key: str) -> None:
        def _do() -> None:
            m = _bind_trigger_mapping(label, joystick=joystick, delta_threshold=0.30, wake_deadzone=0.05)
            if isinstance(m, dict) and isinstance(m.get("axis"), int) and isinstance(m.get("sign"), int):
                _write_set_trigger_mapping(set_name=str(set_name), key=str(key), mapping=m)

        _run_in_scan(_do)

    def _bind_set_firelike(*, label: str, set_name: str, group: str, key: str) -> None:
        def _do() -> None:
            m = _bind_fire_mapping(str(label), joystick=joystick, delta_threshold=0.30)
            if isinstance(m, dict):
                _write_set_mapping(set_name=str(set_name), group=str(group), key=str(key), mapping=m)

        _run_in_scan(_do)

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

    return _run_menu_from_json(spec_path=MENU_SPEC_PATH, start_node=str(start_node), action_handlers=handlers)


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
