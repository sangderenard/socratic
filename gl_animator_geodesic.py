"""
Geodesic (unit-sphere) spring/charge animator using the existing OpenGL renderer.
Experimental sandbox: n-D synthetic points, sparse Coulomb/gravity, dynamic bonds,
Lorentz-like term, adaptive integrator, and joystick-driven controls.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import os
import sys
import threading
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pygame
import pygame.font
from pygame.locals import DOUBLEBUF, OPENGL

try:
    import torch
except Exception:  # torch optional
    torch = None

import joystick_menu
import input_graph
import controller_backend
import weapons_structs
import weapon_runtime
import targeting_system
import reticle_depth_finder
import reticle_telemetry
import airplane_structs
import reticle_sprite
import world_config_structs

from flight_camera import PlanetFlightCamera, lookat_up_away_from_planet, clamp_radius_band


def _resolve_effective_control_set(menu: object | None) -> str:
    """Resolve the effective control-set name for this frame.

    Priority:
    - If airplane role is explicitly set (bomber/fighter), use it.
    - Else, if targeting is active, use view_targeting.
    - Else, use flight.
    """
    if menu is None:
        return "flight"
    try:
        airplane = getattr(menu, "airplane", None)
        airplane = airplane if isinstance(airplane, dict) else {}
        role = None
        for k in ("control_set", "controls_set", "control_profile", "craft_role", "role"):
            v = airplane.get(k)
            if isinstance(v, str) and v.strip():
                role = v.strip().lower()
                break
        if role in ("bomber", "fighter"):
            return str(role)
    except Exception:
        pass
    try:
        if bool(getattr(menu, "_targeting_active", False)):
            return "view_targeting"
    except Exception:
        pass
    return "flight"


def _get_set_binding(cfg: dict, *, set_name: str, group: str, key: str) -> dict | None:
    """Return a binding dict from flight_controls.sets[set_name][group][key].

    Falls back to legacy locations when set storage isn't present.
    """
    try:
        fc = cfg.get("flight_controls")
        if not isinstance(fc, dict):
            return None
        sets = fc.get("sets")
        if isinstance(sets, dict):
            blk = sets.get(str(set_name))
            if isinstance(blk, dict):
                g = blk.get(str(group))
                if isinstance(g, dict):
                    b = g.get(str(key))
                    return b if isinstance(b, dict) else None

        # Legacy fallbacks (pre control-set schema).
        legacy = fc.get(str(group))
        if isinstance(legacy, dict):
            b = legacy.get(str(key))
            return b if isinstance(b, dict) else None
        return None
    except Exception:
        return None


def _get_set_trigger_mapping(cfg: dict, *, set_name: str, key: str) -> dict | None:
    try:
        fc = cfg.get("flight_controls")
        if not isinstance(fc, dict):
            return None
        sets = fc.get("sets")
        if isinstance(sets, dict):
            blk = sets.get(str(set_name))
            if isinstance(blk, dict):
                t = blk.get("triggers")
                if isinstance(t, dict):
                    m = t.get(str(key))
                    return m if isinstance(m, dict) else None
        # Legacy fallback.
        t0 = fc.get("triggers")
        if isinstance(t0, dict):
            m = t0.get(str(key))
            return m if isinstance(m, dict) else None
        return None
    except Exception:
        return None


def _get_camera_binding(cfg: dict, key: str) -> dict | None:
    try:
        fc = cfg.get("flight_controls")
        if not isinstance(fc, dict):
            return None
        cam = fc.get("camera")
        if not isinstance(cam, dict):
            return None
        b = cam.get(str(key))
        return b if isinstance(b, dict) else None
    except Exception:
        return None


def _get_hud_binding(cfg: dict, key: str) -> dict | None:
    try:
        fc = cfg.get("flight_controls")
        if not isinstance(fc, dict):
            return None
        hud = fc.get("hud")
        if not isinstance(hud, dict):
            return None
        b = hud.get(str(key))
        return b if isinstance(b, dict) else None
    except Exception:
        return None


def _binding_active(
    binding: dict | None,
    axes_now: dict[int, float],
    buttons_now: set[int],
    hats_now: dict[int, tuple[int, int]] | None = None,
) -> tuple[bool, float]:
    """Return (active, analog_value)."""
    if not isinstance(binding, dict):
        return False, 0.0
    btype = str(binding.get("type") or "")
    if btype == "button":
        try:
            b = int(binding.get("button", -9999))
        except Exception:
            b = -9999
        return (b in buttons_now), (1.0 if (b in buttons_now) else 0.0)
    if btype == "axis":
        try:
            a = int(binding.get("axis", -9999))
            s = int(binding.get("sign", 1))
        except Exception:
            return False, 0.0
        v = float(axes_now.get(int(a), 0.0))
        analog = max(0.0, v * float(s))
        return (analog >= 0.60), float(analog)
    if btype == "hat":
        if not isinstance(hats_now, dict):
            return False, 0.0
        try:
            h = int(binding.get("hat", -9999))
            x = int(binding.get("x", 0))
            y = int(binding.get("y", 0))
        except Exception:
            return False, 0.0
        now = hats_now.get(int(h), (0, 0))
        active = (int(now[0]), int(now[1])) == (int(x), int(y))
        return active, (1.0 if active else 0.0)
    return False, 0.0


def _norm_axis_from_calib(v: float, calib: dict | None) -> float:
    """Normalize raw axis value using stored extrema. Returns [-1, 1]."""
    if not isinstance(calib, dict):
        return float(max(-1.0, min(1.0, float(v))))
    try:
        vmin = float(calib.get("min", -1.0))
        vmax = float(calib.get("max", 1.0))
    except Exception:
        return float(max(-1.0, min(1.0, float(v))))
    if not (vmax > vmin + 1e-6):
        return float(max(-1.0, min(1.0, float(v))))
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return float(max(-1.0, min(1.0, (float(v) - float(center)) / float(max(1e-6, half)))))


def _read_axis1d(mapping: object, axes_now: dict[int, float]) -> float:
    """Read a 1D control from the new mapping schema (or legacy int axis index)."""
    if isinstance(mapping, int):
        return float(axes_now.get(int(mapping), 0.0))
    if not isinstance(mapping, dict):
        return 0.0
    mtype = str(mapping.get("type", ""))
    if mtype == "axis1d":
        try:
            a = int(mapping.get("axis", -1))
        except Exception:
            a = -1
        v = float(axes_now.get(int(a), 0.0)) if int(a) >= 0 else 0.0
        return float(_norm_axis_from_calib(v, mapping.get("calib") if isinstance(mapping.get("calib"), dict) else None))
    # Allow axis mapping objects from older code paths.
    if "axis" in mapping and isinstance(mapping.get("axis"), int):
        v = float(axes_now.get(int(mapping.get("axis")), 0.0))
        return float(max(-1.0, min(1.0, v)))
    return 0.0


def _read_axis2d(mapping: object, axes_now: dict[int, float]) -> tuple[float, float]:
    """Read a 2D control from the new mapping schema."""
    if not isinstance(mapping, dict):
        return 0.0, 0.0
    if str(mapping.get("type", "")) != "axis2d":
        return 0.0, 0.0
    x = mapping.get("x") if isinstance(mapping.get("x"), dict) else {}
    y = mapping.get("y") if isinstance(mapping.get("y"), dict) else {}

    ax = int(x.get("axis", -1)) if isinstance(x.get("axis", -1), int) else -1
    ay = int(y.get("axis", -1)) if isinstance(y.get("axis", -1), int) else -1
    vx = float(axes_now.get(int(ax), 0.0)) if ax >= 0 else 0.0
    vy = float(axes_now.get(int(ay), 0.0)) if ay >= 0 else 0.0

    cx = x.get("calib") if isinstance(x.get("calib"), dict) else None
    cy = y.get("calib") if isinstance(y.get("calib"), dict) else None
    return float(_norm_axis_from_calib(vx, cx)), float(_norm_axis_from_calib(vy, cy))


def _draw_local_ground_grid(*, cam_pos: np.ndarray, planet_r: float) -> None:
    """Draw a small tangent-plane grid near the camera's ground point.

    Purpose: provide an immediate grounded reference (a "multi-point settling plane")
    without changing physics.
    """
    # Ground point directly below the camera on the planet surface.
    up = cam_pos.astype(np.float32, copy=False)
    n = float(np.linalg.norm(up))
    if not (n > 1e-6):
        return
    up = up / n
    ground_center = up * float(planet_r + 1e-3)  # tiny lift to avoid z-fighting

    # Tangent basis (t1, t2) spanning the plane.
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, up))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t1 = np.cross(ref, up)
    t1n = float(np.linalg.norm(t1))
    if not (t1n > 1e-6):
        return
    t1 = t1 / t1n
    t2 = np.cross(up, t1)
    t2n = float(np.linalg.norm(t2))
    if t2n > 1e-6:
        t2 = t2 / t2n

    # Grid parameters: scale with altitude so it reads as a local patch,
    # not something that wraps around the whole view.
    alt = max(0.0, float(n) - float(planet_r))
    half = float(min(0.14, max(0.03, 1.5 * alt)))
    steps = 8
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    glColor4f(0.85, 0.85, 0.85, 0.20)

    for i in range(-steps, steps + 1):
        a = (float(i) / float(steps)) * half
        p0 = ground_center + t1 * (-half) + t2 * a
        p1 = ground_center + t1 * (half) + t2 * a
        glBegin(GL_LINE_STRIP)
        glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
        glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
        glEnd()

        q0 = ground_center + t2 * (-half) + t1 * a
        q1 = ground_center + t2 * (half) + t1 * a
        glBegin(GL_LINE_STRIP)
        glVertex3f(float(q0[0]), float(q0[1]), float(q0[2]))
        glVertex3f(float(q1[0]), float(q1[1]), float(q1[2]))
        glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)


def _draw_weapon_splines_world(*, splines: list[list[tuple[float, float, float]]]) -> None:
    if not splines:
        return
    try:
        glUseProgram(0)
    except Exception:
        pass
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(2.0)
    glColor4f(0.90, 0.90, 0.90, 0.85)
    for pts in splines:
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        glBegin(GL_LINE_STRIP)
        for p in pts:
            try:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            except Exception:
                pass
        glEnd()
    glDisable(GL_BLEND)


def _draw_minimap_overlay(
    *,
    width: int,
    height: int,
    pos_mm_f32: np.ndarray,
    col_mm_f32: np.ndarray,
    type_mm_f32: np.ndarray,
    ang_mm_f32: np.ndarray,
    n_draw: int,
    minimap_mode: str,
    mm_prog: int,
    mm_u_point_px: int | None,
    mm_u_self_px: int | None,
    mm_a_pos: int,
    mm_a_color: int,
    mm_a_type: int,
    mm_a_ang: int,
    mm_vbo_pos: int,
    mm_vbo_col: int,
    mm_vbo_type: int,
    mm_vbo_ang: int,
    override_tex: int | None = None,
) -> None:
    """Draw a small top-right minimap (picture-in-picture).

    It renders an alternate projection of the same point cloud.
    If minimap_mode is a map projection and camera vectors are provided, draws a tiny
    orientation arrow on the map.
    """
    # Draw the minimap frame/background even when there are no points.
    # This keeps the minimap usable as a viewport (e.g., craft cameras).

    w = int(max(140, 0.28 * float(width)))
    h = int(max(140, 0.28 * float(height)))
    pad = 12
    x0 = int(max(0, width - w - pad))
    y0 = int(max(0, height - h - pad))

    # Save viewport.
    prev_vp = glGetIntegerv(GL_VIEWPORT)

    glViewport(x0, y0, w, h)
    glEnable(GL_SCISSOR_TEST)
    glScissor(x0, y0, w, h)

    # 2D-style view for minimap.
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1.05, 1.05, -1.05, 1.05, -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Background plate.
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.0, 0.0, 0.0, 0.35)
    glBegin(GL_TRIANGLES)
    glVertex3f(-1.05, -1.05, 0.0)
    glVertex3f(1.05, -1.05, 0.0)
    glVertex3f(1.05, 1.05, 0.0)
    glVertex3f(-1.05, -1.05, 0.0)
    glVertex3f(1.05, 1.05, 0.0)
    glVertex3f(-1.05, 1.05, 0.0)
    glEnd()

    # Optional camera feed override.
    # When present, it replaces minimap points (minimap box becomes a viewport).
    if override_tex is not None and int(override_tex) > 0:
        try:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, int(override_tex))
            glColor4f(1.0, 1.0, 1.0, 0.98)
            glBegin(GL_TRIANGLES)
            # tri 1
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-1.05, -1.05, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(1.05, -1.05, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(1.05, 1.05, 0.0)
            # tri 2
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-1.05, -1.05, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(1.05, 1.05, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(-1.05, 1.05, 0.0)
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
        except Exception:
            try:
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
            except Exception:
                pass

    # Points (retro signatures).
    if (override_tex is None or int(override_tex) <= 0) and int(n_draw) > 0:
        glDepthMask(False)
        glUseProgram(int(mm_prog))
        min_dim = float(min(w, h))
        point_px = float(max(2.0, min(6.0, 0.012 * min_dim)))
        self_px = float(max(5.0, min(18.0, point_px * 2.6)))
        if mm_u_point_px is not None and int(mm_u_point_px) >= 0:
            glUniform1f(int(mm_u_point_px), float(point_px))
        if mm_u_self_px is not None and int(mm_u_self_px) >= 0:
            glUniform1f(int(mm_u_self_px), float(self_px))

        glBindBuffer(GL_ARRAY_BUFFER, int(mm_vbo_pos))
        glBufferData(GL_ARRAY_BUFFER, pos_mm_f32[:n_draw].nbytes, pos_mm_f32[:n_draw], GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(int(mm_a_pos))
        glVertexAttribPointer(int(mm_a_pos), 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, int(mm_vbo_col))
        glBufferData(GL_ARRAY_BUFFER, col_mm_f32[:n_draw].nbytes, col_mm_f32[:n_draw], GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(int(mm_a_color))
        glVertexAttribPointer(int(mm_a_color), 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, int(mm_vbo_type))
        glBufferData(GL_ARRAY_BUFFER, type_mm_f32[:n_draw].nbytes, type_mm_f32[:n_draw], GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(int(mm_a_type))
        glVertexAttribPointer(int(mm_a_type), 1, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, int(mm_vbo_ang))
        glBufferData(GL_ARRAY_BUFFER, ang_mm_f32[:n_draw].nbytes, ang_mm_f32[:n_draw], GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(int(mm_a_ang))
        glVertexAttribPointer(int(mm_a_ang), 1, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDrawArrays(GL_POINTS, 0, int(n_draw))

        glDisableVertexAttribArray(int(mm_a_pos))
        glDisableVertexAttribArray(int(mm_a_color))
        glDisableVertexAttribArray(int(mm_a_type))
        glDisableVertexAttribArray(int(mm_a_ang))
        glUseProgram(0)
        glDepthMask(True)

    # Frame.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.30)
    glBegin(GL_LINE_STRIP)
    glVertex3f(-1.05, -1.05, 0.0)
    glVertex3f(1.05, -1.05, 0.0)
    glVertex3f(1.05, 1.05, 0.0)
    glVertex3f(-1.05, 1.05, 0.0)
    glVertex3f(-1.05, -1.05, 0.0)
    glEnd()
    glDisable(GL_BLEND)

    # Restore matrices.
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glDisable(GL_SCISSOR_TEST)
    # Restore viewport.
    glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))


def _draw_ship_panel_minimap(
    *,
    eye: np.ndarray,
    ship_right: np.ndarray,
    ship_up: np.ndarray,
    ship_fwd: np.ndarray,
    pos_mm_f32: np.ndarray,
    col_mm_f32: np.ndarray,
    type_mm_f32: np.ndarray,
    ang_mm_f32: np.ndarray,
    n_draw: int,
    # Panel placement in ship space.
    dist: float,
    off_r: float,
    off_u: float,
    half_w: float,
    half_h: float,
    override_tex: int | None = None,
) -> None:
    """Draw the minimap as a thin 3D panel mounted to the ship.

    This is intentionally NOT screen-space: it's in world space, attached to the ship,
    so head-look (right stick) can look away from it.
    """
    # Draw the panel background/frame even when there are no points.
    # This makes the minimap reliable as a generic viewport surface.

    r = ship_right.astype(np.float32, copy=False)
    u = ship_up.astype(np.float32, copy=False)
    f = ship_fwd.astype(np.float32, copy=False)
    origin = (eye.astype(np.float32, copy=False) + f * float(dist) + r * float(off_r) + u * float(off_u)).astype(np.float32, copy=False)
    hw = float(half_w)
    hh = float(half_h)
    inner_w = hw * 0.90
    inner_h = hh * 0.90

    def _wp(x: float, y: float) -> np.ndarray:
        return (origin + r * (float(x) * hw) + u * (float(y) * hh)).astype(np.float32, copy=False)

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Background.
    glColor4f(0.0, 0.0, 0.0, 0.38)
    p00 = _wp(-1.0, -1.0)
    p10 = _wp(1.0, -1.0)
    p11 = _wp(1.0, 1.0)
    p01 = _wp(-1.0, 1.0)
    glBegin(GL_TRIANGLES)
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p10[0]), float(p10[1]), float(p10[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p01[0]), float(p01[1]), float(p01[2]))
    glEnd()

    # Optional camera feed override.
    if override_tex is not None and int(override_tex) > 0:
        try:
            p00i = origin + r * (-inner_w) + u * (-inner_h)
            p10i = origin + r * (inner_w) + u * (-inner_h)
            p11i = origin + r * (inner_w) + u * (inner_h)
            p01i = origin + r * (-inner_w) + u * (inner_h)

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, int(override_tex))
            glColor4f(1.0, 1.0, 1.0, 0.98)
            glBegin(GL_TRIANGLES)
            # tri 1
            glTexCoord2f(0.0, 0.0)
            glVertex3f(float(p00i[0]), float(p00i[1]), float(p00i[2]))
            glTexCoord2f(1.0, 0.0)
            glVertex3f(float(p10i[0]), float(p10i[1]), float(p10i[2]))
            glTexCoord2f(1.0, 1.0)
            glVertex3f(float(p11i[0]), float(p11i[1]), float(p11i[2]))
            # tri 2
            glTexCoord2f(0.0, 0.0)
            glVertex3f(float(p00i[0]), float(p00i[1]), float(p00i[2]))
            glTexCoord2f(1.0, 1.0)
            glVertex3f(float(p11i[0]), float(p11i[1]), float(p11i[2]))
            glTexCoord2f(0.0, 1.0)
            glVertex3f(float(p01i[0]), float(p01i[1]), float(p01i[2]))
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
        except Exception:
            try:
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
            except Exception:
                pass

    self_idx = -1
    if (override_tex is None or int(override_tex) <= 0) and int(n_draw) > 0:
        # Points.
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i in range(int(n_draw)):
            if int(type_mm_f32[i]) == 1:
                self_idx = i
                continue
            x = float(max(-1.0, min(1.0, float(pos_mm_f32[i, 0]))))
            y = float(max(-1.0, min(1.0, float(pos_mm_f32[i, 1]))))
            pw = origin + r * (x * inner_w) + u * (y * inner_h)
            c = col_mm_f32[i]
            glColor4f(float(c[0]), float(c[1]), float(c[2]), 0.85)
            glVertex3f(float(pw[0]), float(pw[1]), float(pw[2]))
        glEnd()

        # Self marker (bigger dot + heading arrow).
        if self_idx >= 0:
            x = float(max(-1.0, min(1.0, float(pos_mm_f32[self_idx, 0]))))
            y = float(max(-1.0, min(1.0, float(pos_mm_f32[self_idx, 1]))))
            center = origin + r * (x * inner_w) + u * (y * inner_h)
            segs = 28
            dot_r = 0.08 * min(hw, hh)
            glColor4f(1.0, 0.95, 0.30, 0.95)
            glBegin(GL_TRIANGLES)
            for k in range(segs):
                a0 = 2.0 * math.pi * (float(k) / float(segs))
                a1 = 2.0 * math.pi * (float(k + 1) / float(segs))
                p0 = center
                p1 = center + r * (dot_r * float(math.cos(a0))) + u * (dot_r * float(math.sin(a0)))
                p2 = center + r * (dot_r * float(math.cos(a1))) + u * (dot_r * float(math.sin(a1)))
                glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))
            glEnd()

            ang = float(ang_mm_f32[self_idx])
            dx = float(math.cos(ang))
            dy = float(math.sin(ang))
            L = 0.55 * min(hw, hh)
            tip = center + r * (dx * L) + u * (dy * L)
            glLineWidth(2.0)
            glColor4f(1.0, 0.98, 0.75, 0.85)
            glBegin(GL_LINES)
            glVertex3f(float(center[0]), float(center[1]), float(center[2]))
            glVertex3f(float(tip[0]), float(tip[1]), float(tip[2]))
            glEnd()

    # Frame.
    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.28)
    glBegin(GL_LINE_STRIP)
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p10[0]), float(p10[1]), float(p10[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p01[0]), float(p01[1]), float(p01[2]))
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


_CAM_FEED_TEX: int | None = None
_CAM_FEED_FBO: int | None = None
_CAM_FEED_W: int = 0
_CAM_FEED_H: int = 0

_RETICLE_DEPTH_FINDER = reticle_depth_finder.CWeaponDepthFinder()

# Reticle auto-targeting tuning (ship view).
_RETICLE_AUTO_CENTER_EPS_DEG = 0.75
_RETICLE_AUTO_SLEW_DEG_PER_S = 35.0
_RETICLE_LOS_POLL_S = 0.25
_RETICLE_LOS_EPS_MULT_RADIUS = 0.60
_RETICLE_LOS_EPS_ABS = 0.15

# Weapon-independent aim latch defaults.
# Guided weapons (e.g., missiles) can keep a latched LOS lock while the target stays within
# an off-boresight cone. Weapon stats may override these via `guidance.independent_aim`
# and `guidance.off_boresight_deg`.
_WEAPON_DEFAULT_OFF_BORESIGHT_DEG = 35.0


def _ensure_camera_feed_target(*, w: int, h: int) -> None:
    global _CAM_FEED_TEX, _CAM_FEED_FBO, _CAM_FEED_W, _CAM_FEED_H

    w = int(max(64, min(2048, w)))
    h = int(max(64, min(2048, h)))
    if _CAM_FEED_TEX is not None and _CAM_FEED_FBO is not None and _CAM_FEED_W == w and _CAM_FEED_H == h:
        return

    try:
        if _CAM_FEED_FBO is not None:
            glDeleteFramebuffers(1, [int(_CAM_FEED_FBO)])
    except Exception:
        pass
    try:
        if _CAM_FEED_TEX is not None:
            glDeleteTextures(1, [int(_CAM_FEED_TEX)])
    except Exception:
        pass

    _CAM_FEED_TEX = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, int(_CAM_FEED_TEX))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(w), int(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    _CAM_FEED_FBO = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, int(_CAM_FEED_FBO))
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, int(_CAM_FEED_TEX), 0)
    status = int(glCheckFramebufferStatus(GL_FRAMEBUFFER))
    if status == int(GL_FRAMEBUFFER_COMPLETE):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != int(GL_FRAMEBUFFER_COMPLETE):
        _CAM_FEED_TEX = None
        _CAM_FEED_FBO = None
        _CAM_FEED_W = 0
        _CAM_FEED_H = 0
        return

    _CAM_FEED_W = int(w)
    _CAM_FEED_H = int(h)


def _camera_fov_y_deg_from_weapon_cfg(weapon_cfg: dict, *, aspect: float) -> float:
    """Compute vertical FOV from camera config.

    FOV_y = 2 * atan(sensor_height / (2*focal_length)).
    """
    cam = weapon_cfg.get("camera") if isinstance(weapon_cfg, dict) else None
    cam = cam if isinstance(cam, dict) else {}
    try:
        focal_mm = float(cam.get("focal_length_mm", 35.0))
    except Exception:
        focal_mm = 35.0
    try:
        sensor_h_mm = float(cam.get("sensor_height_mm", 24.0))
    except Exception:
        sensor_h_mm = 24.0
    try:
        sensor_w_mm = float(cam.get("sensor_width_mm", 36.0))
    except Exception:
        sensor_w_mm = 36.0

    focal_mm = float(max(1e-3, min(2000.0, focal_mm)))
    sensor_h_mm = float(max(1e-3, min(200.0, sensor_h_mm)))
    sensor_w_mm = float(max(1e-3, min(200.0, sensor_w_mm)))

    # If sensor height is missing/unreasonable, approximate it from width and aspect.
    if not np.isfinite(sensor_h_mm) or sensor_h_mm <= 1e-6:
        a = float(aspect) if (np.isfinite(aspect) and aspect > 1e-6) else 1.0
        sensor_h_mm = float(sensor_w_mm / a)

    fov_y = 2.0 * math.atan(sensor_h_mm / (2.0 * focal_mm))
    deg = float(fov_y * (180.0 / math.pi))
    if not np.isfinite(deg):
        deg = 60.0
    return float(max(5.0, min(160.0, deg)))


def _camera_fov_y_deg_from_weapon_cfg_zoomed(weapon_cfg: dict, *, aspect: float, zoom_mul: float) -> float:
    """Compute vertical FOV from camera config with an optical zoom multiplier.

    zoom_mul > 1.0 increases effective focal length (narrows FOV).
    """
    cam = weapon_cfg.get("camera") if isinstance(weapon_cfg, dict) else None
    cam = cam if isinstance(cam, dict) else {}
    try:
        focal_mm = float(cam.get("focal_length_mm", 35.0))
    except Exception:
        focal_mm = 35.0
    try:
        sensor_h_mm = float(cam.get("sensor_height_mm", 24.0))
    except Exception:
        sensor_h_mm = 24.0
    try:
        sensor_w_mm = float(cam.get("sensor_width_mm", 36.0))
    except Exception:
        sensor_w_mm = 36.0

    z = float(zoom_mul)
    if not np.isfinite(z):
        z = 1.0
    z = float(max(1.0, min(40.0, z)))

    focal_mm = float(max(1e-3, min(2000.0, focal_mm * z)))
    sensor_h_mm = float(max(1e-3, min(200.0, sensor_h_mm)))
    sensor_w_mm = float(max(1e-3, min(200.0, sensor_w_mm)))

    if not np.isfinite(sensor_h_mm) or sensor_h_mm <= 1e-6:
        a = float(aspect) if (np.isfinite(aspect) and aspect > 1e-6) else 1.0
        sensor_h_mm = float(sensor_w_mm / a)

    fov_y = 2.0 * math.atan(sensor_h_mm / (2.0 * focal_mm))
    deg = float(fov_y * (180.0 / math.pi))
    if not np.isfinite(deg):
        deg = 60.0
    return float(max(2.5, min(160.0, deg)))


def _render_camera_feed_scene(
    *,
    tex_w: int,
    tex_h: int,
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    fov_y_deg: float,
    z_near: float,
    z_far: float,
    # Scene + environment
    scene: dict,
    sky_rgb: tuple[float, float, float],
    is_ship: bool,
    render_scale: float,
    # Sun program
    sun_prog: int,
    sun_u_color: int | None,
    sun_u_size_px: int | None,
    sun_a_pos: int,
    sun_vbo_pos: int,
    # Atmosphere shell program
    atmo_prog: int,
    atmo_u_radius: int | None,
    atmo_u_color: int | None,
    atmo_u_alpha: int | None,
    atmo_u_power: int | None,
    atmo_a_pos: int,
    ball_prog: int,
    ball_u_mvp: int | None,
    ball_u_light_dir: int | None,
    ball_u_light_color: int | None,
    ball_u_fog_color: int | None,
    ball_u_fog_k: int | None,
    ball_a_pos: int,
    ball_a_color: int,
    ball_a_size: int,
    vbo_pos: int,
    vbo_col: int,
    vbo_size: int,
    pos_f32: np.ndarray,
    col_f32: np.ndarray,
    size_f32: np.ndarray,
    n_active: int,
    # Optional straight-edge rendering
    line_prog: int | None = None,
    line_u_mvp: int | None = None,
    line_u_color: int | None = None,
    line_u_fog_color: int | None = None,
    line_u_fog_k: int | None = None,
    line_a_pos: int | None = None,
    line_vbo_pos: int | None = None,
    edge_verts_f32: np.ndarray | None = None,
    springs_live: np.ndarray | None = None,
    draw_direct_edges: bool = False,
    alt_frac: float | None = None,
    # Optional weapon splines (world-space lines)
    weapon_splines: list[dict] | None = None,
) -> int | None:
    _ensure_camera_feed_target(w=int(tex_w), h=int(tex_h))
    if _CAM_FEED_TEX is None or _CAM_FEED_FBO is None:
        return None

    n = int(n_active)

    # Use caller's sky color for consistency.
    try:
        glClearColor(float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]), 1.0)
    except Exception:
        glClearColor(0.0, 0.0, 0.0, 1.0)

    prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
    prev_vp = glGetIntegerv(GL_VIEWPORT)

    glBindFramebuffer(GL_FRAMEBUFFER, int(_CAM_FEED_FBO))
    glViewport(0, 0, int(_CAM_FEED_W), int(_CAM_FEED_H))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    a = float(_CAM_FEED_W) / float(max(1, int(_CAM_FEED_H)))
    gluPerspective(float(fov_y_deg), float(max(1e-6, a)), float(max(1e-4, z_near)), float(max(z_near + 1e-3, z_far)))

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    gluLookAt(
        float(eye[0]),
        float(eye[1]),
        float(eye[2]),
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
    )

    # --- Distant sun (match main view behavior) ---
    try:
        sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
        sun_dir = np.asarray(sun.get("direction"), dtype=np.float32)
        sd = float(np.linalg.norm(sun_dir))
        if sd > 1e-6:
            sun_dir = (sun_dir / sd).astype(np.float32, copy=False)
        else:
            sun_dir = np.array([-0.25, 0.35, 1.0], dtype=np.float32)

        sun_col = sun.get("color", (1.0, 0.98, 0.92))
        inten = float(sun.get("intensity", 1.0))
        sun_rgb = np.array([float(sun_col[0]), float(sun_col[1]), float(sun_col[2])], dtype=np.float32) * float(max(0.0, inten))
        size_px = float(sun.get("size_px", 160.0))
        dist = float(sun.get("distance", 800.0))
        dist = float(max(10.0, min(dist, 0.9 * float(z_far))))
        eye_np = np.asarray(eye, dtype=np.float32)
        sun_pos = (eye_np + sun_dir * dist).astype(np.float32, copy=False)

        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(int(sun_prog))
        if sun_u_color is not None and int(sun_u_color) >= 0:
            glUniform3f(int(sun_u_color), float(sun_rgb[0]), float(sun_rgb[1]), float(sun_rgb[2]))
        if sun_u_size_px is not None and int(sun_u_size_px) >= 0:
            glUniform1f(int(sun_u_size_px), float(size_px))

        glBindBuffer(GL_ARRAY_BUFFER, int(sun_vbo_pos))
        glBufferData(GL_ARRAY_BUFFER, sun_pos.nbytes, sun_pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(int(sun_a_pos))
        glVertexAttribPointer(int(sun_a_pos), 3, GL_FLOAT, False, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDrawArrays(GL_POINTS, 0, 1)
        glDisableVertexAttribArray(int(sun_a_pos))
        glUseProgram(0)
        glDisable(GL_BLEND)
        glDepthMask(True)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)
    except Exception:
        pass

    # --- Planet + atmosphere boundary + sea + limb haze ---
    try:
        rs = float(render_scale) if bool(is_ship) else 1.0
        sun_dir = None
        try:
            sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
            sun_dir = np.asarray(sun.get("direction"), dtype=np.float32)
        except Exception:
            sun_dir = None

        try:
            paint = scene.get("node_paint", {}) if isinstance(scene.get("node_paint", {}), dict) else {}
            paint_strength = float(paint.get("blend_strength", 1.0))
        except Exception:
            paint_strength = 1.0

        terr = scene.get("terrain", {}) if isinstance(scene.get("terrain", {}), dict) else {}
        terr_hm = None
        terr_hs = 0.0
        terr_hb = 0.5
        terr_shade = 0.0
        try:
            if bool(terr.get("enabled", False)) and bool(terr.get("planet_enabled", False)):
                terr_hm = _load_grayscale_heightmap(str(terr.get("heightmap_path", "") or ""))
                terr_hs = float(terr.get("planet_height_scale", 0.0)) * float(rs)
                terr_hb = float(terr.get("planet_height_bias", 0.5))
                terr_shade = float(terr.get("planet_shade_strength", 0.0))
        except Exception:
            terr_hm = None

        _draw_planet_and_atmosphere(
            planet_r=float(_PLANET_DRAW_R) * float(rs),
            atmosphere_r=float(_ATMOSPHERE_R) * float(rs),
            light_dir=sun_dir,
            paint_tex=_NODE_PAINT_TEX,
            paint_strength=float(max(0.0, paint_strength)),
            terrain_heightmap=terr_hm,
            terrain_height_scale=float(max(0.0, terr_hs)),
            terrain_height_bias=float(max(0.0, min(1.0, terr_hb))),
            terrain_shade_strength=float(max(0.0, terr_shade)),
        )

        try:
            sea = scene.get("sea", {}) if isinstance(scene.get("sea", {}), dict) else {}
            if bool(sea.get("enabled", False)):
                base_r = float(_PLANET_DRAW_R) * float(rs)
                use_hm_level = bool(sea.get("use_heightmap_level", True))
                lvl = float(sea.get("heightmap_level", 0.5))
                lvl = float(max(0.0, min(1.0, lvl)))
                r_off = float(sea.get("radius_offset", 0.0)) * float(rs)
                if not np.isfinite(r_off):
                    r_off = 0.0

                if (
                    use_hm_level
                    and terr_hm is not None
                    and isinstance(terr_hm, np.ndarray)
                    and terr_hm.size > 0
                    and float(terr_hs) > 0.0
                ):
                    sea_r = base_r + (lvl - float(terr_hb)) * float(terr_hs)
                else:
                    sea_r = base_r + r_off

                col = sea.get("color", (0.08, 0.26, 0.52))
                a = float(sea.get("alpha", 0.55))
                dw = bool(sea.get("depth_write", False))
                _draw_sea_sphere(
                    sea_r=float(sea_r),
                    color=_safe_color3(col, (0.08, 0.26, 0.52)),
                    alpha=float(a),
                    depth_write=bool(dw),
                )
        except Exception:
            pass

        try:
            shell = scene.get("atmosphere_shell", {}) if isinstance(scene.get("atmosphere_shell", {}), dict) else {}
            if bool(shell.get("enabled", True)) and _PLANET_VBO_POS is not None and int(_PLANET_VBO_COUNT) > 0:
                rad = float(_PLANET_DRAW_R) * float(rs) * float(shell.get("radius_mult", 1.04))
                col = shell.get("color", (0.55, 0.75, 1.0))
                a = float(shell.get("alpha", 0.28))
                pwr = float(shell.get("power", 2.2))

                depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
                if not depth_was_enabled:
                    glEnable(GL_DEPTH_TEST)
                glDepthMask(False)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glUseProgram(int(atmo_prog))
                if atmo_u_radius is not None and int(atmo_u_radius) >= 0:
                    glUniform1f(int(atmo_u_radius), float(rad))
                if atmo_u_color is not None and int(atmo_u_color) >= 0:
                    glUniform3f(int(atmo_u_color), float(col[0]), float(col[1]), float(col[2]))
                if atmo_u_alpha is not None and int(atmo_u_alpha) >= 0:
                    glUniform1f(int(atmo_u_alpha), float(max(0.0, min(1.0, a))))
                if atmo_u_power is not None and int(atmo_u_power) >= 0:
                    glUniform1f(int(atmo_u_power), float(max(0.25, pwr)))

                glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS))
                glEnableVertexAttribArray(int(atmo_a_pos))
                glVertexAttribPointer(int(atmo_a_pos), 3, GL_FLOAT, False, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDrawArrays(GL_TRIANGLES, 0, int(_PLANET_VBO_COUNT))
                glDisableVertexAttribArray(int(atmo_a_pos))

                glUseProgram(0)
                glDisable(GL_BLEND)
                glDepthMask(True)
                if not depth_was_enabled:
                    glDisable(GL_DEPTH_TEST)
        except Exception:
            pass
    except Exception:
        pass

    # --- Optional weapon splines (world-space lines) ---
    try:
        if weapon_splines:
            _draw_weapon_splines_world(splines=[s.get("pts", []) for s in weapon_splines])
    except Exception:
        pass

    # --- Balls (points) ---
    glDisable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(True)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    mvp = _gl_mvp_matrix_f32()

    glUseProgram(int(ball_prog))
    if ball_u_mvp is not None and int(ball_u_mvp) >= 0:
        glUniformMatrix4fv(int(ball_u_mvp), 1, False, mvp)

    # Reuse the same lighting/haze model as the main pass (best-effort).
    try:
        sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
        ldir = np.asarray(sun.get("direction"), dtype=np.float32)
        ln = float(np.linalg.norm(ldir))
        if ln > 1e-6:
            ldir = (ldir / ln).astype(np.float32, copy=False)
        else:
            ldir = np.array([-0.25, 0.35, 1.0], dtype=np.float32)
        scol = sun.get("color", (1.0, 0.98, 0.92))
        inten = float(sun.get("intensity", 1.0))
        lcol = np.array([float(scol[0]), float(scol[1]), float(scol[2])], dtype=np.float32) * float(max(0.0, inten))

        haze = scene.get("haze", {}) if isinstance(scene.get("haze", {}), dict) else {}
        k0 = float(haze.get("strength_at_ground", 1.0))
        k1 = float(haze.get("strength_at_top", 0.15))
        af = float(alt_frac) if alt_frac is not None and np.isfinite(float(alt_frac)) else 1.0
        fog_strength = (1.0 - af) * k0 + af * k1
        fog_k = float(haze.get("fog_k", 0.018)) * float(max(0.0, fog_strength))
        rs_fog = float(render_scale) if bool(is_ship) else 1.0
        fog_k = float(fog_k) / float(max(1e-6, rs_fog))

        if ball_u_light_dir is not None and int(ball_u_light_dir) >= 0:
            glUniform3f(int(ball_u_light_dir), float(ldir[0]), float(ldir[1]), float(ldir[2]))
        if ball_u_light_color is not None and int(ball_u_light_color) >= 0:
            glUniform3f(int(ball_u_light_color), float(lcol[0]), float(lcol[1]), float(lcol[2]))
        if ball_u_fog_color is not None and int(ball_u_fog_color) >= 0:
            glUniform3f(int(ball_u_fog_color), float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]))
        if ball_u_fog_k is not None and int(ball_u_fog_k) >= 0:
            glUniform1f(int(ball_u_fog_k), float(max(0.0, fog_k)))
    except Exception:
        pass

    glBindBuffer(GL_ARRAY_BUFFER, int(vbo_pos))
    glBufferData(GL_ARRAY_BUFFER, pos_f32[:n].nbytes, pos_f32[:n], GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(int(ball_a_pos))
    glVertexAttribPointer(int(ball_a_pos), 3, GL_FLOAT, False, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, int(vbo_col))
    glBufferData(GL_ARRAY_BUFFER, col_f32[:n].nbytes, col_f32[:n], GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(int(ball_a_color))
    glVertexAttribPointer(int(ball_a_color), 3, GL_FLOAT, False, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, int(vbo_size))
    glBufferData(GL_ARRAY_BUFFER, size_f32[:n].nbytes, size_f32[:n], GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(int(ball_a_size))
    glVertexAttribPointer(int(ball_a_size), 1, GL_FLOAT, False, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glDrawArrays(GL_POINTS, 0, int(n))

    glDisableVertexAttribArray(int(ball_a_pos))
    glDisableVertexAttribArray(int(ball_a_color))
    glDisableVertexAttribArray(int(ball_a_size))
    glUseProgram(0)

    glDisable(GL_BLEND)
    glDepthMask(True)

    # --- Optional straight edges ---
    try:
        if (
            bool(draw_direct_edges)
            and springs_live is not None
            and isinstance(springs_live, np.ndarray)
            and springs_live.size
            and line_prog is not None
            and line_a_pos is not None
            and line_vbo_pos is not None
        ):
            i_idx = springs_live[:, 0].astype(np.int64, copy=False)
            j_idx = springs_live[:, 1].astype(np.int64, copy=False)
            valid = (i_idx >= 0) & (j_idx >= 0) & (i_idx < n) & (j_idx < n) & (i_idx != j_idx)
            if valid.size:
                i_idx = i_idx[valid]
                j_idx = j_idx[valid]
            m = int(i_idx.shape[0])
            if m > 0:
                if edge_verts_f32 is None or edge_verts_f32.shape[0] < 2 * m:
                    edge_verts_f32 = np.empty((2 * m, 3), dtype=np.float32)
                edge_verts_f32[0 : 2 * m : 2, :] = pos_f32[i_idx]
                edge_verts_f32[1 : 2 * m : 2, :] = pos_f32[j_idx]

                glUseProgram(int(line_prog))
                if line_u_mvp is not None and int(line_u_mvp) >= 0:
                    glUniformMatrix4fv(int(line_u_mvp), 1, False, mvp)
                if line_u_color is not None and int(line_u_color) >= 0:
                    glUniform4f(int(line_u_color), 0.8, 0.85, 0.95, 0.85)

                # Match haze to points if uniforms exist.
                try:
                    if line_u_fog_color is not None and int(line_u_fog_color) >= 0:
                        glUniform3f(int(line_u_fog_color), float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]))
                    if line_u_fog_k is not None and int(line_u_fog_k) >= 0:
                        haze = scene.get("haze", {}) if isinstance(scene.get("haze", {}), dict) else {}
                        k0 = float(haze.get("strength_at_ground", 1.0))
                        k1 = float(haze.get("strength_at_top", 0.15))
                        af = float(alt_frac) if alt_frac is not None and np.isfinite(float(alt_frac)) else 1.0
                        fog_strength = (1.0 - af) * k0 + af * k1
                        fog_k = float(haze.get("fog_k", 0.018)) * float(max(0.0, fog_strength))
                        rs_fog = float(render_scale) if bool(is_ship) else 1.0
                        fog_k = float(fog_k) / float(max(1e-6, rs_fog))
                        glUniform1f(int(line_u_fog_k), float(max(0.0, fog_k)))
                except Exception:
                    pass

                glBindBuffer(GL_ARRAY_BUFFER, int(line_vbo_pos))
                glBufferData(GL_ARRAY_BUFFER, edge_verts_f32[: 2 * m].nbytes, edge_verts_f32[: 2 * m], GL_DYNAMIC_DRAW)
                glEnableVertexAttribArray(int(line_a_pos))
                glVertexAttribPointer(int(line_a_pos), 3, GL_FLOAT, False, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                glLineWidth(2.0)
                glDrawArrays(GL_LINES, 0, 2 * m)

                glDisableVertexAttribArray(int(line_a_pos))
                glUseProgram(0)
    except Exception:
        pass

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))
    glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))
    return int(_CAM_FEED_TEX)


def _draw_ship_panel_attitude_ball(
    *,
    eye: np.ndarray,
    ship_right: np.ndarray,
    ship_up: np.ndarray,
    ship_fwd: np.ndarray,
    dist: float,
    off_r: float,
    off_u: float,
    half_w: float,
    half_h: float,
    altitude: float = 0.0,
    altitude_max: float = 1.0,
    ship_heading_rad: float = 0.0,
    ground_heading_rad: float = 0.0,
    airspeed: float = 0.0,
    groundspeed: float = 0.0,
    throttle: float = 0.0,
) -> None:
    """Draw the blue/black attitude ball as a thin 3D panel mounted to the ship."""
    r = ship_right.astype(np.float32, copy=False)
    u = ship_up.astype(np.float32, copy=False)
    f = ship_fwd.astype(np.float32, copy=False)
    origin = (eye.astype(np.float32, copy=False) + f * float(dist) + r * float(off_r) + u * float(off_u)).astype(np.float32, copy=False)
    hw = float(half_w)
    hh = float(half_h)

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _v(x: float, y: float) -> np.ndarray:
        return (origin + r * (float(x) * hw) + u * (float(y) * hh)).astype(np.float32, copy=False)

    def _draw_7seg_text(
        *,
        x: float,
        y: float,
        s: float,
        text: str,
        color: tuple[float, float, float, float],
    ) -> None:
        # Minimal 7-seg digits (no textures), good enough for a cockpit readout.
        segs = {
            "0": "abcfed",
            "1": "bc",
            "2": "abged",
            "3": "abgcd",
            "4": "fgbc",
            "5": "afgcd",
            "6": "afgcde",
            "7": "abc",
            "8": "abcdefg",
            "9": "abfgcd",
            "-": "g",
        }

        def _seg_line(seg: str, ox: float, oy: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
            # Digit cell in local coords: x in [0,1], y in [0,2].
            if seg == "a":
                return ((ox + 0.10, oy + 1.85), (ox + 0.90, oy + 1.85))
            if seg == "b":
                return ((ox + 0.90, oy + 1.80), (ox + 0.90, oy + 1.10))
            if seg == "c":
                return ((ox + 0.90, oy + 0.90), (ox + 0.90, oy + 0.20))
            if seg == "d":
                return ((ox + 0.10, oy + 0.15), (ox + 0.90, oy + 0.15))
            if seg == "e":
                return ((ox + 0.10, oy + 0.20), (ox + 0.10, oy + 0.90))
            if seg == "f":
                return ((ox + 0.10, oy + 1.10), (ox + 0.10, oy + 1.80))
            if seg == "g":
                return ((ox + 0.10, oy + 1.00), (ox + 0.90, oy + 1.00))
            return None

        glColor4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        glLineWidth(2.0)
        glBegin(GL_LINES)
        cx = float(x)
        cy = float(y)
        for ch in str(text):
            if ch == " ":
                cx += 0.65 * float(s)
                continue
            if ch == ".":
                # Dot: a tiny vertical segment.
                p0 = _v(cx + 0.92 * s, cy + 0.20 * s)
                p1 = _v(cx + 0.92 * s, cy + 0.30 * s)
                glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                cx += 0.35 * float(s)
                continue
            mask = segs.get(ch)
            if not mask:
                cx += 0.65 * float(s)
                continue
            for seg in mask:
                ln = _seg_line(seg, 0.0, 0.0)
                if ln is None:
                    continue
                (x0, y0), (x1, y1) = ln
                p0 = _v(cx + float(x0) * s, cy + float(y0) * s)
                p1 = _v(cx + float(x1) * s, cy + float(y1) * s)
                glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
            cx += 1.25 * float(s)
        glEnd()

    # Backplate.
    glColor4f(0.0, 0.0, 0.0, 0.22)
    p00 = _v(-1.05, -1.05)
    p10 = _v(1.05, -1.05)
    p11 = _v(1.05, 1.05)
    p01 = _v(-1.05, 1.05)
    glBegin(GL_TRIANGLES)
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p10[0]), float(p10[1]), float(p10[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p01[0]), float(p01[1]), float(p01[2]))
    glEnd()

    # Down vector in world, then expressed in ship coordinates.
    down_w = (-eye.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    dn = float(np.linalg.norm(down_w))
    if dn > 1e-6:
        down_w = down_w / dn
    else:
        down_w = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    dx = float(np.dot(down_w, r))
    dy = float(np.dot(down_w, u))
    dz = float(np.dot(down_w, f))

    horizon_ang = float(math.atan2(-dx, dy))
    pitch = float(math.asin(max(-1.0, min(1.0, -dz))))
    pitch_offset = float(max(-0.75, min(0.75, (pitch / (0.5 * math.pi)) * 0.75)))
    ca = float(math.cos(horizon_ang))
    sa = float(math.sin(horizon_ang))

    def _sky_ground_color(x: float, y: float) -> tuple[float, float, float, float]:
        yr = (-x * sa + y * ca) + pitch_offset
        if yr >= 0.0:
            return (0.18, 0.42, 0.92, 0.78)
        return (0.02, 0.02, 0.02, 0.82)

    segs = 96
    glBegin(GL_TRIANGLES)
    for i in range(segs):
        a0 = 2.0 * math.pi * (float(i) / float(segs))
        a1 = 2.0 * math.pi * (float(i + 1) / float(segs))
        x0, y0u = float(math.cos(a0)), float(math.sin(a0))
        x1, y1u = float(math.cos(a1)), float(math.sin(a1))

        c0 = _sky_ground_color(0.0, 0.0)
        c1 = _sky_ground_color(x0, y0u)
        c2 = _sky_ground_color(x1, y1u)

        glColor4f(float(c0[0]), float(c0[1]), float(c0[2]), float(c0[3]))
        p0 = _v(0.0, 0.0)
        glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))

        glColor4f(float(c1[0]), float(c1[1]), float(c1[2]), float(c1[3]))
        p1 = _v(x0, y0u)
        glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))

        glColor4f(float(c2[0]), float(c2[1]), float(c2[2]), float(c2[3]))
        p2 = _v(x1, y1u)
        glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))
    glEnd()

    # Horizon line (white).
    yh = -pitch_offset
    L = 1.35
    xh0, xh1 = -L, L
    hx0 = xh0 * ca - yh * sa
    hy0 = xh0 * sa + yh * ca
    hx1 = xh1 * ca - yh * sa
    hy1 = xh1 * sa + yh * ca
    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.85)
    q0 = _v(hx0, hy0)
    q1 = _v(hx1, hy1)
    glBegin(GL_LINES)
    glVertex3f(float(q0[0]), float(q0[1]), float(q0[2]))
    glVertex3f(float(q1[0]), float(q1[1]), float(q1[2]))
    glEnd()

    # Fixed aircraft reference (wings).
    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.65)
    w0 = _v(-0.28, 0.0)
    w1 = _v(-0.08, 0.0)
    w2 = _v(0.08, 0.0)
    w3 = _v(0.28, 0.0)
    glBegin(GL_LINES)
    glVertex3f(float(w0[0]), float(w0[1]), float(w0[2]))
    glVertex3f(float(w1[0]), float(w1[1]), float(w1[2]))
    glVertex3f(float(w2[0]), float(w2[1]), float(w2[2]))
    glVertex3f(float(w3[0]), float(w3[1]), float(w3[2]))
    glEnd()

    # Outer ring.
    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.35)
    glBegin(GL_LINE_STRIP)
    for i in range(segs + 1):
        a = 2.0 * math.pi * (float(i) / float(segs))
        p = _v(float(math.cos(a)), float(math.sin(a)))
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()

    # --- Border tapes + numeric readouts ---
    # Right edge: altitude tape (land=0 is red, ceiling=max is black).
    # Bottom edge: compass tape (north=0 is red, south=pi is black).
    try:
        alt = float(max(0.0, float(altitude)))
    except Exception:
        alt = 0.0
    try:
        alt_max = float(max(1e-6, float(altitude_max)))
    except Exception:
        alt_max = 1.0

    # Alt tape window is in absolute altitude units.
    alt_win = 0.35 * max(0.25, alt_max)
    alt_win = float(max(0.10, min(0.75, alt_win)))
    step = 0.05
    span = 0.78
    x_edge = 1.02

    def _alt_tick_len(t: float) -> float:
        if abs((t / 0.20) - round(t / 0.20)) < 1e-6:
            return 0.12
        if abs((t / 0.10) - round(t / 0.10)) < 1e-6:
            return 0.085
        return 0.055

    glLineWidth(2.0)
    # Reference mark.
    glColor4f(0.98, 0.98, 0.98, 0.70)
    glBegin(GL_LINES)
    a0 = _v(x_edge, 0.0)
    a1 = _v(x_edge - 0.16, 0.0)
    glVertex3f(float(a0[0]), float(a0[1]), float(a0[2]))
    glVertex3f(float(a1[0]), float(a1[1]), float(a1[2]))
    glEnd()

    t0 = math.floor((alt - alt_win) / step) * step
    t1 = math.ceil((alt + alt_win) / step) * step
    t = t0
    glBegin(GL_LINES)
    while t <= t1 + 1e-9:
        y = (t - alt) / alt_win
        if -1.0 <= y <= 1.0:
            yn = float(y) * span
            L = _alt_tick_len(t)
            # Special ticks.
            if abs(t - 0.0) < 0.5 * step:
                glColor4f(0.92, 0.12, 0.12, 0.85)  # land (red)
            elif abs(t - alt_max) < 0.5 * step:
                glColor4f(0.0, 0.0, 0.0, 0.85)     # space ceiling (black)
            else:
                glColor4f(0.98, 0.98, 0.98, 0.55)
            p0 = _v(x_edge, yn)
            p1 = _v(x_edge - L, yn)
            glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
            glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
        t += step
    glEnd()

    # Altitude readout (top-right, small).
    try:
        alt_txt = f"{alt:.2f}"
    except Exception:
        alt_txt = "0.00"
    _draw_7seg_text(x=0.20, y=0.58, s=0.12, text=alt_txt, color=(0.98, 0.98, 0.98, 0.75))

    # Speed readouts (top-left corner of the gimbal): airspeed + groundspeed.
    try:
        as_txt = f"{max(0.0, float(airspeed)):.2f}"
    except Exception:
        as_txt = "0.00"
    try:
        gs_txt = f"{max(0.0, float(groundspeed)):.2f}"
    except Exception:
        gs_txt = "0.00"
    _draw_7seg_text(x=-0.98, y=0.58, s=0.11, text=as_txt, color=(0.45, 0.80, 1.00, 0.78))
    _draw_7seg_text(x=-0.98, y=0.40, s=0.11, text=gs_txt, color=(1.00, 0.92, 0.25, 0.78))

    # Throttle readout (below speeds). Signed percent in [-100, 100].
    try:
        thr_pct = float(max(-1.0, min(1.0, float(throttle)))) * 100.0
        thr_txt = f"{thr_pct:.0f}"
    except Exception:
        thr_txt = "0"
    _draw_7seg_text(x=-0.98, y=0.22, s=0.11, text=thr_txt, color=(0.98, 0.98, 0.98, 0.72))

    # Compass tape: centered on ship heading.
    hdg_ship = _wrap_angle_pi(float(ship_heading_rad))
    hdg_ground = _wrap_angle_pi(float(ground_heading_rad))
    hdg_win = math.radians(65.0)
    step_h = math.radians(10.0)
    y_edge = -1.02

    def _hdg_tick_len(a: float) -> float:
        if abs((a / math.radians(90.0)) - round(a / math.radians(90.0))) < 1e-6:
            return 0.12
        if abs((a / math.radians(30.0)) - round(a / math.radians(30.0))) < 1e-6:
            return 0.085
        return 0.055

    glLineWidth(2.0)
    # Ship heading reference mark (blue).
    glColor4f(0.35, 0.70, 1.00, 0.85)
    glBegin(GL_LINES)
    c0 = _v(0.0, y_edge)
    c1 = _v(0.0, y_edge + 0.16)
    glVertex3f(float(c0[0]), float(c0[1]), float(c0[2]))
    glVertex3f(float(c1[0]), float(c1[1]), float(c1[2]))
    glEnd()

    # Ground heading marker (yellow) moves relative to ship heading.
    try:
        dgh = _wrap_angle_pi(hdg_ground - hdg_ship)
        xg = float(max(-1.0, min(1.0, dgh / float(hdg_win))))
        glColor4f(1.00, 0.92, 0.25, 0.90)
        glBegin(GL_LINES)
        g0 = _v(xg * span, y_edge)
        g1 = _v(xg * span, y_edge + 0.14)
        glVertex3f(float(g0[0]), float(g0[1]), float(g0[2]))
        glVertex3f(float(g1[0]), float(g1[1]), float(g1[2]))
        glEnd()
    except Exception:
        pass

    h0 = math.floor((hdg_ship - hdg_win) / step_h) * step_h
    h1 = math.ceil((hdg_ship + hdg_win) / step_h) * step_h
    a = h0
    glBegin(GL_LINES)
    while a <= h1 + 1e-9:
        d = _wrap_angle_pi(a - hdg_ship)
        x = d / hdg_win
        if -1.0 <= x <= 1.0:
            xn = float(x) * span
            L = _hdg_tick_len(a)
            # Special ticks.
            if abs(_wrap_angle_pi(a - 0.0)) < 0.5 * step_h:
                glColor4f(0.92, 0.12, 0.12, 0.85)  # north (red)
            elif abs(_wrap_angle_pi(a - math.pi)) < 0.5 * step_h:
                glColor4f(0.0, 0.0, 0.0, 0.85)     # south (black)
            else:
                glColor4f(0.98, 0.98, 0.98, 0.55)
            p0 = _v(xn, y_edge)
            p1 = _v(xn, y_edge + L)
            glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
            glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
        a += step_h
    glEnd()

    # Ship heading readout (bottom-center).
    try:
        hdg_deg = (math.degrees(float(hdg_ship)) + 360.0) % 360.0
        hdg_txt = f"{hdg_deg:05.1f}"
    except Exception:
        hdg_txt = "000.0"
    _draw_7seg_text(x=-0.35, y=-0.88, s=0.11, text=hdg_txt, color=(0.98, 0.98, 0.98, 0.75))

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


def _wrap_angle_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    x = float(a)
    x = (x + math.pi) % (2.0 * math.pi) - math.pi
    return float(x)


def _ship_heading_rad(*, pos: np.ndarray, fwd: np.ndarray) -> float:
    """Compute a stable compass heading around local radial up.

    Heading is measured relative to a projected +Y "north" direction in the local tangent plane.
    """
    up = np.asarray(pos, dtype=np.float32)
    n_up = float(np.linalg.norm(up))
    if not np.isfinite(n_up) or n_up <= 1e-9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = (up / n_up).astype(np.float32, copy=False)

    f = np.asarray(fwd, dtype=np.float32)
    # Project forward into tangent plane.
    f_t = f - up * float(np.dot(f, up))
    n_f = float(np.linalg.norm(f_t))
    if not np.isfinite(n_f) or n_f <= 1e-9:
        return 0.0
    f_t = (f_t / n_f).astype(np.float32, copy=False)

    # Build a stable tangent north/east frame.
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, up))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    east = np.cross(ref, up).astype(np.float32, copy=False)
    n_e = float(np.linalg.norm(east))
    if not np.isfinite(n_e) or n_e <= 1e-9:
        return 0.0
    east = (east / n_e).astype(np.float32, copy=False)
    north = np.cross(up, east).astype(np.float32, copy=False)
    n_n = float(np.linalg.norm(north))
    if not np.isfinite(n_n) or n_n <= 1e-9:
        return 0.0
    north = (north / n_n).astype(np.float32, copy=False)

    x = float(np.dot(f_t, east))
    y = float(np.dot(f_t, north))
    return float(math.atan2(x, y))


def _draw_ship_panel_altimeter_compass(
    *,
    eye: np.ndarray,
    ship_right: np.ndarray,
    ship_up: np.ndarray,
    ship_fwd: np.ndarray,
    dist: float,
    half_w: float,
    half_h: float,
    altitude: float,
    heading_rad: float,
) -> None:
    """Draw simple hash-line altimeter (left) and compass (bottom).

    The marks slide with the measured value, similar to a tape indicator.
    """
    r = ship_right.astype(np.float32, copy=False)
    u = ship_up.astype(np.float32, copy=False)
    f = ship_fwd.astype(np.float32, copy=False)
    origin = (eye.astype(np.float32, copy=False) + f * float(dist)).astype(np.float32, copy=False)
    hw = float(half_w)
    hh = float(half_h)

    def _v(x_ndc: float, y_ndc: float) -> np.ndarray:
        return (origin + r * (float(x_ndc) * hw) + u * (float(y_ndc) * hh)).astype(np.float32, copy=False)

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Common styling (match existing cockpit panel line tone).
    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.55)

    x_edge = -0.92
    y_edge = -0.92
    span = 0.85

    # --- Altimeter tape (left) ---
    alt = float(max(0.0, altitude))
    alt_win = 0.50  # show ±0.5 units
    step = 0.05
    t0 = math.floor((alt - alt_win) / step) * step
    t1 = math.ceil((alt + alt_win) / step) * step

    def _alt_tick_len(t: float) -> float:
        # Major/medium/minor tick by fractional unit.
        if abs((t / 0.20) - round(t / 0.20)) < 1e-6:
            return 0.12
        if abs((t / 0.10) - round(t / 0.10)) < 1e-6:
            return 0.085
        return 0.055

    glBegin(GL_LINES)
    # Reference mark at current value.
    p0 = _v(x_edge, 0.0)
    p1 = _v(x_edge + 0.14, 0.0)
    glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
    glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))

    t = t0
    while t <= t1 + 1e-9:
        y = (t - alt) / alt_win
        if -1.0 <= y <= 1.0:
            yn = float(y) * span
            L = _alt_tick_len(t)
            a0 = _v(x_edge, yn)
            a1 = _v(x_edge + L, yn)
            glVertex3f(float(a0[0]), float(a0[1]), float(a0[2]))
            glVertex3f(float(a1[0]), float(a1[1]), float(a1[2]))
        t += step
    glEnd()

    # --- Compass tape (bottom) ---
    hdg = _wrap_angle_pi(float(heading_rad))
    hdg_win = math.radians(90.0)  # show ±90°
    step_h = math.radians(10.0)
    h0 = math.floor((hdg - hdg_win) / step_h) * step_h
    h1 = math.ceil((hdg + hdg_win) / step_h) * step_h

    def _hdg_tick_len(a: float) -> float:
        # Major at 90°, medium at 30°, minor at 10°.
        if abs((a / math.radians(90.0)) - round(a / math.radians(90.0))) < 1e-6:
            return 0.12
        if abs((a / math.radians(30.0)) - round(a / math.radians(30.0))) < 1e-6:
            return 0.085
        return 0.055

    glBegin(GL_LINES)
    # Reference mark at current heading.
    c0 = _v(0.0, y_edge)
    c1 = _v(0.0, y_edge + 0.14)
    glVertex3f(float(c0[0]), float(c0[1]), float(c0[2]))
    glVertex3f(float(c1[0]), float(c1[1]), float(c1[2]))

    a = h0
    while a <= h1 + 1e-9:
        d = _wrap_angle_pi(a - hdg)
        x = d / hdg_win
        if -1.0 <= x <= 1.0:
            xn = float(x) * span
            L = _hdg_tick_len(a)
            b0 = _v(xn, y_edge)
            b1 = _v(xn, y_edge + L)
            glVertex3f(float(b0[0]), float(b0[1]), float(b0[2]))
            glVertex3f(float(b1[0]), float(b1[1]), float(b1[2]))
        a += step_h
    glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


def _draw_bubble_gimbal_overlay(
    *,
    width: int,
    height: int,
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    heading_world: np.ndarray | None,
) -> None:
    """Draw a small airplane-style attitude indicator ("blue/white/black ball").

    Visualization intent:
    - Blue sky / black ground split by a white horizon line.
    - Horizon line orientation is derived from planet-down in camera coordinates.
    - Horizon vertical offset is derived from pitch relative to the local horizon.
    """
    w = int(max(140, 0.18 * float(width)))
    h = int(w)
    pad = 12
    x0 = int(max(0, pad))
    # OpenGL viewport coords are bottom-left origin; place at bottom-left.
    y0 = int(max(0, pad))

    # Save viewport.
    prev_vp = glGetIntegerv(GL_VIEWPORT)

    glViewport(x0, y0, w, h)
    glEnable(GL_SCISSOR_TEST)
    glScissor(x0, y0, w, h)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1.05, 1.05, -1.05, 1.05, -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Backplate (subtle).
    glColor4f(0.0, 0.0, 0.0, 0.22)
    glBegin(GL_TRIANGLES)
    glVertex3f(-1.05, -1.05, 0.0)
    glVertex3f(1.05, -1.05, 0.0)
    glVertex3f(1.05, 1.05, 0.0)
    glVertex3f(-1.05, -1.05, 0.0)
    glVertex3f(1.05, 1.05, 0.0)
    glVertex3f(-1.05, 1.05, 0.0)
    glEnd()

    # Camera basis from (eye, center, up) matching gluLookAt usage.
    fwd = center.astype(np.float32, copy=False) - eye.astype(np.float32, copy=False)
    fn = float(np.linalg.norm(fwd))
    if fn > 1e-6:
        fwd = fwd / fn
    else:
        fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    upn = up.astype(np.float32, copy=False)
    un = float(np.linalg.norm(upn))
    if un > 1e-6:
        upn = upn / un
    else:
        upn = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(fwd, upn).astype(np.float32, copy=False)
    rn = float(np.linalg.norm(right))
    if rn > 1e-6:
        right = right / rn
    else:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    upo = np.cross(right, fwd).astype(np.float32, copy=False)
    un2 = float(np.linalg.norm(upo))
    if un2 > 1e-6:
        upo = upo / un2

    # Planet core direction (down) in world.
    down_w = -eye.astype(np.float32, copy=False)
    dn = float(np.linalg.norm(down_w))
    if dn > 1e-6:
        down_w = down_w / dn
    else:
        down_w = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    # Convert to camera coordinates.
    dx = float(np.dot(down_w, right))
    dy = float(np.dot(down_w, upo))
    dz = float(np.dot(down_w, fwd))

    # Attitude model from planet-down in camera coordinates.
    # Horizon is the intersection of the local horizon plane (perp to down) with the camera's view plane.
    horizon_ang = float(math.atan2(-dx, dy))
    # Pitch relative to the local horizon: positive when looking "up" (away from planet).
    pitch = float(math.asin(max(-1.0, min(1.0, -dz))))
    pitch_offset = float(max(-0.75, min(0.75, (pitch / (0.5 * math.pi)) * 0.75)))
    ca = float(math.cos(horizon_ang))
    sa = float(math.sin(horizon_ang))

    def _sky_ground_color(x: float, y: float) -> tuple[float, float, float, float]:
        # Rotate by -ang so the horizon is horizontal in "attitude space", then apply pitch offset.
        yr = (-x * sa + y * ca) + pitch_offset
        if yr >= 0.0:
            # Sky (blue)
            return (0.18, 0.42, 0.92, 0.78)
        # Ground (black)
        return (0.02, 0.02, 0.02, 0.82)

    # Fill the ball with per-vertex sky/ground shading.
    segs = 96
    glBegin(GL_TRIANGLES)
    for i in range(segs):
        a0 = 2.0 * math.pi * (float(i) / float(segs))
        a1 = 2.0 * math.pi * (float(i + 1) / float(segs))
        x0, y0u = float(math.cos(a0)), float(math.sin(a0))
        x1, y1u = float(math.cos(a1)), float(math.sin(a1))

        c0 = _sky_ground_color(0.0, 0.0)
        c1 = _sky_ground_color(x0, y0u)
        c2 = _sky_ground_color(x1, y1u)

        glColor4f(float(c0[0]), float(c0[1]), float(c0[2]), float(c0[3]))
        glVertex3f(0.0, 0.0, 0.0)

        glColor4f(float(c1[0]), float(c1[1]), float(c1[2]), float(c1[3]))
        glVertex3f(x0, y0u, 0.0)

        glColor4f(float(c2[0]), float(c2[1]), float(c2[2]), float(c2[3]))
        glVertex3f(x1, y1u, 0.0)
    glEnd()

    # Horizon line (white) with pitch offset and bank.
    # In attitude space the horizon is y = -pitch_offset.
    yh = -pitch_offset
    # Endpoints far enough to span the circle; we don't strictly clip.
    L = 1.35
    xh0, xh1 = -L, L
    # Rotate back by +ang.
    hx0 = xh0 * ca - yh * sa
    hy0 = xh0 * sa + yh * ca
    hx1 = xh1 * ca - yh * sa
    hy1 = xh1 * sa + yh * ca
    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.85)
    glBegin(GL_LINES)
    glVertex3f(float(hx0), float(hy0), 0.0)
    glVertex3f(float(hx1), float(hy1), 0.0)
    glEnd()

    # Fixed aircraft reference (white "wings").
    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.65)
    glBegin(GL_LINES)
    glVertex3f(-0.28, 0.0, 0.0)
    glVertex3f(-0.08, 0.0, 0.0)
    glVertex3f(0.08, 0.0, 0.0)
    glVertex3f(0.28, 0.0, 0.0)
    glEnd()

    # Outer ring.
    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.35)
    glBegin(GL_LINE_STRIP)
    for i in range(segs + 1):
        a = 2.0 * math.pi * (float(i) / float(segs))
        glVertex3f(float(math.cos(a)), float(math.sin(a)), 0.0)
    glEnd()

    glDisable(GL_BLEND)
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_SCISSOR_TEST)
    glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))


## NOTE: Legacy loadout overlay removed.
## The loadout HUD is rendered from the existing ctypes PIP editor via
## `_draw_loadout_menu_overlay()` so it remains editable with the `menu_nav`
## bindings.


def _draw_loadout_menu_overlay(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    pip: joystick_menu.CtypesStructEditorPip,
) -> None:
    """Draw the existing loadout menu (ctypes PIP editor) as a top-left HUD overlay.

    This matches the other HUD widgets: viewport + scissor, then draw content.
    """
    w = int(max(220, 0.28 * float(width)))
    h = int(max(150, 0.22 * float(height)))
    pad = 12
    x0 = int(max(0, pad))
    y0 = int(max(0, height - h - pad))

    prev_vp = glGetIntegerv(GL_VIEWPORT)

    glViewport(x0, y0, w, h)
    glEnable(GL_SCISSOR_TEST)
    glScissor(x0, y0, w, h)

    # Fixed panel (minimap-style container), then draw the existing menu text inside.
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    if depth_was_enabled:
        glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glColor4f(0.0, 0.0, 0.0, 0.35)
    glBegin(GL_TRIANGLES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glEnd()

    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.25)
    glBegin(GL_LINE_STRIP)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    # Draw inside this viewport (do NOT override viewport to fullscreen).
    # `draw_box=False` prevents the PIP text renderer from drawing its own
    # content-sized box inside the already-existing panel.
    pip.draw(font=font, width=int(w), height=int(h), respect_viewport=True, draw_box=False)

    glDisable(GL_SCISSOR_TEST)
    glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))

try:
    from OpenGL.GL import (
        glBegin,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor4f,
        glColor3f,
        glDisable,
        glEnable,
        glEnd,
        glIsEnabled,
        glLoadIdentity,
        glMatrixMode,
        glPointSize,
        glLineWidth,
        glVertex3f,
        glBindBuffer,
        glBufferData,
        glBufferSubData,
        glGenBuffers,
        glEnableClientState,
        glDisableClientState,
        glVertexPointer,
        glColorPointer,
        glDrawArrays,
        glUseProgram,
        glCreateShader,
        glShaderSource,
        glCompileShader,
        glGetShaderiv,
        glGetShaderInfoLog,
        glCreateProgram,
        glAttachShader,
        glLinkProgram,
        glGetProgramiv,
        glGetProgramInfoLog,
        glDeleteShader,
        glDeleteProgram,
        glGetUniformLocation,
        glUniformMatrix4fv,
        glUniform1f,
        glUniform1i,
        glUniform2f,
        glUniform3f,
        glUniform4f,
        glGetAttribLocation,
        glEnableVertexAttribArray,
        glDisableVertexAttribArray,
        glVertexAttribPointer,
        glDepthMask,
        glPushMatrix,
        glPopMatrix,
        glScalef,
        glGetDoublev,
        glGetIntegerv,
        glViewport,
        glScissor,
        glOrtho,
        glPixelStorei,
        glRasterPos2f,
        glDrawPixels,
        glActiveTexture,
        glBindTexture,
        glDeleteTextures,
        glGenTextures,
        glTexImage2D,
        glTexCoord2f,
        glTexParameteri,
        glGenFramebuffers,
        glBindFramebuffer,
        glDeleteFramebuffers,
        glFramebufferTexture2D,
        glCheckFramebufferStatus,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LIGHTING,
        GL_MODELVIEW,
        GL_POINTS,
        GL_LINES,
        GL_LINE_STRIP,
        GL_TRIANGLES,
        GL_PROJECTION,
        GL_TEXTURE0,
        GL_TEXTURE_2D,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        GL_LINEAR,
        GL_NEAREST,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_REPEAT,
        GL_CLAMP,
        GL_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_BINDING,
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_ARRAY_BUFFER,
        GL_DYNAMIC_DRAW,
        GL_VERTEX_ARRAY,
        GL_COLOR_ARRAY,
        GL_FLOAT,
        GL_DOUBLE,
        GL_VERTEX_SHADER,
        GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS,
        GL_LINK_STATUS,
        GL_PROGRAM_POINT_SIZE,
        GL_POINT_SPRITE,
        GL_MODELVIEW_MATRIX,
        GL_PROJECTION_MATRIX,
        GL_VIEWPORT,
        GL_SCISSOR_TEST,
        GL_CULL_FACE,
        GL_PROGRAM_POINT_SIZE,
        GL_POINT_SPRITE,
    )
    from OpenGL.GLU import gluPerspective, gluLookAt
except Exception as exc:  # pragma: no cover - runtime guard
    print("PyOpenGL is required to run this animator.")
    raise

import gl_animator as base

# NOTE: HUD overlays are rendered via dedicated helpers (minimap, etc.) and the
# loadout HUD uses the existing ctypes PIP editor (see `_draw_loadout_menu_overlay`).

# Pinning is only meaningful when a CUDA device is active; toggled in run().
PIN_ENABLED = bool(torch is not None and torch.cuda.is_available())


def _gl_compile_program(vertex_src: str, fragment_src: str) -> int:
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, vertex_src)
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(vs)
        glDeleteShader(vs)
        raise RuntimeError(f"Vertex shader compile failed:\n{log.decode('utf-8', errors='replace') if isinstance(log, (bytes, bytearray)) else log}")

    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, fragment_src)
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(fs)
        glDeleteShader(vs)
        glDeleteShader(fs)
        raise RuntimeError(f"Fragment shader compile failed:\n{log.decode('utf-8', errors='replace') if isinstance(log, (bytes, bytearray)) else log}")

    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        glDeleteProgram(prog)
        raise RuntimeError(f"Program link failed:\n{log.decode('utf-8', errors='replace') if isinstance(log, (bytes, bytearray)) else log}")

    glDeleteShader(vs)
    glDeleteShader(fs)
    return int(prog)


def _gl_mvp_matrix_f32() -> np.ndarray:
    """Return current MVP (projection * modelview) as float32 4x4, column-major."""
    mv = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4, 4), order="F")
    pr = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float32).reshape((4, 4), order="F")
    mvp = pr @ mv
    return np.asarray(mvp, dtype=np.float32, order="F")


_MAP_PROJ_MODES = ("map-equirect", "map-mercator", "map-lambert")

# "Planet" rendering constants (render-only).
# The simulation generally lives around ~unit radius; the planet surface MUST be smaller so
# nodes/balls stay above ground. The flight camera is constrained to an atmosphere band
# above the surface (never inside the planet, never below ground).
_PLANET_SURFACE_R = 0.78
_FLIGHT_ALT_MIN = 0.06
# Allow climbing well above the atmosphere band ("to space").
_FLIGHT_ALT_MAX = 3.00
_FLIGHT_R_MIN = _PLANET_SURFACE_R + _FLIGHT_ALT_MIN
_FLIGHT_R_MAX = _PLANET_SURFACE_R + _FLIGHT_ALT_MAX

# Keep the visible atmosphere shell close to the planet even if flight can go much higher.
# By default, bind the atmosphere/sky transition to the "node layer" radius (~1.0),
# so flying above the nodes trends toward black space.
_ATMOSPHERE_ALT_MAX = float(max(0.05, 1.0 - _PLANET_SURFACE_R))
_ATMOSPHERE_R = _PLANET_SURFACE_R + _ATMOSPHERE_ALT_MAX
_PLANET_DRAW_R = _PLANET_SURFACE_R


def _safe_render_scale(v: object | None) -> float:
    """Planet/world scaling factor.

    The underlying physics model (particles + ship flight) operates in a unit-sphere
    coordinate system.

    `render_scale` converts that unit-sphere into larger/smaller *world units* for:
    - planet radius + atmosphere thickness
    - ship/camera position and velocities in world space

    IMPORTANT: this does *not* scale the ship's body geometry or actuator/arm layout.
    Arms and thrusters remain in ship-local units.
    """
    try:
        s = float(v) if v is not None else 1.0
    except Exception:
        s = 1.0
    if not np.isfinite(s):
        s = 1.0
    return float(max(1e-4, min(1e6, s)))

_SPHERE_MESH_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_PLANET_VBO_POS: int | None = None
_PLANET_VBO_COL: int | None = None
_PLANET_VBO_COUNT: int = 0
_PLANET_LIGHT_DIR_KEY: tuple[float, float, float] | None = None
_PLANET_TERRAIN_KEY: tuple[object, ...] | None = None
_PLANET_VBO_POS_TERRAIN: int | None = None
_PLANET_PROG: int | None = None
_PLANET_A_POS: int | None = None
_PLANET_A_COL: int | None = None
_PLANET_U_SCALE: int | None = None
_PLANET_U_PAINT_TEX: int | None = None
_PLANET_U_PAINT_STRENGTH: int | None = None

_SEA_PROG: int | None = None
_SEA_A_POS: int | None = None
_SEA_U_SCALE: int | None = None
_SEA_U_COLOR: int | None = None

_NODE_PAINT_TEX: int | None = None
_NODE_PAINT_FBO: int | None = None
_NODE_PAINT_W: int = 0
_NODE_PAINT_H: int = 0

_HUD_PLANE_TEX: int | None = None
_HUD_PLANE_TEX_WH: tuple[int, int] | None = None
_TERRAIN_HEIGHTMAP_CACHE: dict[str, dict[str, object]] = {}

_DEFAULT_TERRAIN_HEIGHTMAP_PATH = "assets/terrain/default_heightmap.png"


def _ensure_default_terrain_heightmap_file(*, path: str, w: int = 1024, h: int = 512) -> None:
    """Create a default grayscale equirectangular heightmap if it doesn't exist.

    Pattern: smooth spherical sinusoid in lon/lat (seamless in u / lon).
    Saved as an 8-bit PNG.
    """
    p = str(path or "").strip()
    if not p:
        return
    try:
        if os.path.exists(p):
            return
    except Exception:
        return

    try:
        folder = os.path.dirname(p)
        if folder:
            os.makedirs(folder, exist_ok=True)
    except Exception:
        pass

    try:
        if not pygame.get_init():
            pygame.init()
    except Exception:
        pass

    try:
        w_i = int(max(64, min(8192, w)))
        h_i = int(max(32, min(4096, h)))

        u = (np.linspace(0.0, 1.0, w_i, endpoint=False, dtype=np.float32))[None, :]
        v = (np.linspace(0.0, 1.0, h_i, endpoint=True, dtype=np.float32))[:, None]
        lon = (u - 0.5) * (2.0 * math.pi)
        lat = (v - 0.5) * math.pi

        # "Spherical sin" procedural terrain. Seamless at u wrap; well-behaved at poles.
        a = np.sin(4.0 * lon) * np.cos(3.0 * lat)
        b = np.sin(2.0 * lon + 1.7 * np.sin(lat)) * np.cos(5.0 * lat)
        c = np.sin(2.0 * lat)
        h01 = 0.5 + 0.5 * (0.55 * a + 0.35 * b + 0.10 * c)
        h01 = np.clip(h01, 0.0, 1.0)
        img = (h01 * 255.0).astype(np.uint8, copy=False)

        rgb = np.repeat(img[:, :, None], 3, axis=2)
        rgb_whc = np.transpose(rgb, (1, 0, 2))  # (w,h,3) for make_surface
        surf = pygame.surfarray.make_surface(rgb_whc)
        pygame.image.save(surf, p)
    except Exception:
        # Best-effort; if saving fails, runtime can still proceed without terrain.
        return


def _safe_color3(v: object | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            r, g, b = float(v[0]), float(v[1]), float(v[2])
        else:
            r, g, b = default
    except Exception:
        r, g, b = default
    if not np.isfinite(r):
        r = default[0]
    if not np.isfinite(g):
        g = default[1]
    if not np.isfinite(b):
        b = default[2]
    return (float(max(0.0, min(1.0, r))), float(max(0.0, min(1.0, g))), float(max(0.0, min(1.0, b))))


def _safe_unit_vec3(v: object | None, default: tuple[float, float, float]) -> np.ndarray:
    try:
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            a = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
        else:
            a = np.array([float(default[0]), float(default[1]), float(default[2])], dtype=np.float32)
    except Exception:
        a = np.array([float(default[0]), float(default[1]), float(default[2])], dtype=np.float32)
    n = float(np.linalg.norm(a))
    if not (n > 1e-6) or not np.isfinite(n):
        return np.array([float(default[0]), float(default[1]), float(default[2])], dtype=np.float32)
    return (a / n).astype(np.float32, copy=False)


def _load_scene_spec(path: object | None) -> dict:
    """Load a simple scene spec JSON (sun + sky + haze).

    This is intentionally tiny and tolerant: missing/invalid fields fall back to defaults.
    """
    scene: dict = {}
    if isinstance(path, str) and path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                scene = obj
        except Exception:
            scene = {}

    sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
    sky = scene.get("sky", {}) if isinstance(scene.get("sky", {}), dict) else {}
    haze = scene.get("haze", {}) if isinstance(scene.get("haze", {}), dict) else {}
    shell = scene.get("atmosphere_shell", {}) if isinstance(scene.get("atmosphere_shell", {}), dict) else {}
    paint = scene.get("node_paint", {}) if isinstance(scene.get("node_paint", {}), dict) else {}
    terrain = scene.get("terrain", {}) if isinstance(scene.get("terrain", {}), dict) else {}
    sea = scene.get("sea", {}) if isinstance(scene.get("sea", {}), dict) else {}

    out = {
        "sun": {
            "direction": _safe_unit_vec3(sun.get("direction"), (-0.25, 0.35, 1.0)),
            "color": _safe_color3(sun.get("color"), (1.0, 0.98, 0.92)),
            "intensity": float(sun.get("intensity", 1.0)) if sun.get("intensity", None) is not None else 1.0,
            "distance": float(sun.get("distance", 800.0)) if sun.get("distance", None) is not None else 800.0,
            "size_px": float(sun.get("size_px", 160.0)) if sun.get("size_px", None) is not None else 160.0,
        },
        "sky": {
            "space_color": _safe_color3(sky.get("space_color"), (0.02, 0.03, 0.06)),
            "atmo_color": _safe_color3(sky.get("atmo_color"), (0.32, 0.52, 0.86)),
            "curve": float(sky.get("curve", 1.8)) if sky.get("curve", None) is not None else 1.8,
        },
        "haze": {
            "fog_k": float(haze.get("fog_k", 0.018)) if haze.get("fog_k", None) is not None else 0.018,
            "strength_at_ground": float(haze.get("strength_at_ground", 1.0)) if haze.get("strength_at_ground", None) is not None else 1.0,
            "strength_at_top": float(haze.get("strength_at_top", 0.15)) if haze.get("strength_at_top", None) is not None else 0.15,
        },
        "atmosphere_shell": {
            "enabled": bool(shell.get("enabled", True)),
            "radius_mult": float(shell.get("radius_mult", 1.04)) if shell.get("radius_mult", None) is not None else 1.04,
            "color": _safe_color3(shell.get("color"), (0.55, 0.75, 1.0)),
            "alpha": float(shell.get("alpha", 0.28)) if shell.get("alpha", None) is not None else 0.28,
            "power": float(shell.get("power", 2.2)) if shell.get("power", None) is not None else 2.2,
        },
        "sea": {
            "enabled": bool(sea.get("enabled", False)),
            # If true and terrain is enabled, compute sea radius using the heightmap level.
            "use_heightmap_level": bool(sea.get("use_heightmap_level", True)),
            # Heightmap grayscale value in [0..1] treated as sea level.
            # Effective radius: r_sea = planet_r + (heightmap_level - planet_height_bias) * planet_height_scale
            "heightmap_level": float(sea.get("heightmap_level", 0.5)) if sea.get("heightmap_level", None) is not None else 0.5,
            # Fallback when no terrain is available: additive radius offset (scaled by render_scale in ship mode).
            "radius_offset": float(sea.get("radius_offset", 0.0)) if sea.get("radius_offset", None) is not None else 0.0,
            "color": _safe_color3(sea.get("color"), (0.08, 0.26, 0.52)),
            "alpha": float(sea.get("alpha", 0.55)) if sea.get("alpha", None) is not None else 0.55,
            # If true, sea writes depth. Default false so submerged terrain can remain visible through transparency.
            "depth_write": bool(sea.get("depth_write", False)),
        },
        "node_paint": {
            "enabled": bool(paint.get("enabled", True)),
            # Texture resolution (w,h) in pixels. Keep modest for speed.
            "resolution": paint.get("resolution", (512, 256)),
            # Fade fraction per second toward black. 0 => permanent.
            "decay_per_sec": float(paint.get("decay_per_sec", 0.0)) if paint.get("decay_per_sec", None) is not None else 0.0,
            # Stamp radius in texture pixels.
            "radius_px": float(paint.get("radius_px", 6.0)) if paint.get("radius_px", None) is not None else 6.0,
            # Stamp application rate (alpha per second, clamped per frame).
            "strength_per_sec": float(paint.get("strength_per_sec", 5.0)) if paint.get("strength_per_sec", None) is not None else 5.0,
            # Final blend strength when sampling in the planet shader.
            "blend_strength": float(paint.get("blend_strength", 1.0)) if paint.get("blend_strength", None) is not None else 1.0,
        },
        # Optional terrain heightmap used by the ground-track arc instrument.
        # This is intentionally simple: a monochrome equirectangular map sampled as
        # u=lon/(2π)+0.5, v=lat/π+0.5.
        "terrain": {
            "enabled": bool(terrain.get("enabled", False)),
            # Monochrome, equirectangular (lon/lat) map used for terrain height.
            "heightmap_path": str(terrain.get("heightmap_path", _DEFAULT_TERRAIN_HEIGHTMAP_PATH) or _DEFAULT_TERRAIN_HEIGHTMAP_PATH).strip(),

            # --- Planet terrain (world geometry + shading) ---
            # If true, displace planet vertices radially using the heightmap.
            "planet_enabled": bool(terrain.get("planet_enabled", False)),
            # Height scale in "world units" relative to the planet radius passed to the renderer.
            # (In ship mode, this is also scaled by render_scale.)
            "planet_height_scale": float(terrain.get("planet_height_scale", 0.06)) if terrain.get("planet_height_scale", None) is not None else 0.06,
            # Bias in [0..1] treated as the zero-displacement value.
            "planet_height_bias": float(terrain.get("planet_height_bias", 0.5)) if terrain.get("planet_height_bias", None) is not None else 0.5,
            # Optional height-based brightness modulation for the procedural planet colors.
            "planet_shade_strength": float(terrain.get("planet_shade_strength", 0.85)) if terrain.get("planet_shade_strength", None) is not None else 0.85,

            # --- Ground-track arc instrument (HUD panel) ---
            # Visual deformation amplitude (in instrument local coordinates ~[-1..1]).
            "arc_amp": float(terrain.get("arc_amp", 0.18)) if terrain.get("arc_amp", None) is not None else 0.18,
            # Great-circle half-span (degrees). Total arc coverage is ~2*span_deg.
            "span_deg": float(terrain.get("span_deg", 60.0)) if terrain.get("span_deg", None) is not None else 60.0,
            # Plane icon sprite path (PNG). If missing, an in-memory fallback is generated.
            "plane_icon_path": str(terrain.get("plane_icon_path", "assets/hud/plane_icon.png") or "assets/hud/plane_icon.png").strip(),
            # Plane icon size in panel-local units.
            "plane_w": float(terrain.get("plane_w", 0.42)) if terrain.get("plane_w", None) is not None else 0.42,
            "plane_h": float(terrain.get("plane_h", 0.16)) if terrain.get("plane_h", None) is not None else 0.16,
        },
    }

    # sanitize numerics
    try:
        out["sun"]["intensity"] = float(max(0.0, out["sun"]["intensity"]))
        out["sun"]["distance"] = float(max(1.0, out["sun"]["distance"]))
        out["sun"]["size_px"] = float(max(1.0, min(2000.0, out["sun"]["size_px"])))
    except Exception:
        pass
    try:
        out["sky"]["curve"] = float(max(0.25, min(8.0, out["sky"]["curve"])))
    except Exception:
        pass
    try:
        out["haze"]["fog_k"] = float(max(0.0, out["haze"]["fog_k"]))
        out["haze"]["strength_at_ground"] = float(max(0.0, out["haze"]["strength_at_ground"]))
        out["haze"]["strength_at_top"] = float(max(0.0, out["haze"]["strength_at_top"]))
    except Exception:
        pass
    try:
        out["atmosphere_shell"]["radius_mult"] = float(max(1.0, min(2.0, out["atmosphere_shell"]["radius_mult"])))
        out["atmosphere_shell"]["alpha"] = float(max(0.0, min(1.0, out["atmosphere_shell"]["alpha"])))
        out["atmosphere_shell"]["power"] = float(max(0.25, min(12.0, out["atmosphere_shell"]["power"])))
    except Exception:
        pass

    # sanitize sea
    try:
        out["sea"]["heightmap_level"] = float(max(0.0, min(1.0, float(out["sea"]["heightmap_level"]))))
        out["sea"]["radius_offset"] = float(out["sea"]["radius_offset"])
        if not np.isfinite(out["sea"]["radius_offset"]):
            out["sea"]["radius_offset"] = 0.0
        out["sea"]["alpha"] = float(max(0.0, min(1.0, float(out["sea"]["alpha"]))))
    except Exception:
        pass

    # sanitize node paint
    try:
        res = out.get("node_paint", {}).get("resolution", (512, 256))
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            w, h = int(res[0]), int(res[1])
        else:
            w, h = 512, 256
        w = max(64, min(4096, w))
        h = max(64, min(4096, h))
        out["node_paint"]["resolution"] = (w, h)
    except Exception:
        out.setdefault("node_paint", {})["resolution"] = (512, 256)
    try:
        out["node_paint"]["decay_per_sec"] = float(max(0.0, out["node_paint"]["decay_per_sec"]))
        out["node_paint"]["radius_px"] = float(max(0.5, min(256.0, out["node_paint"]["radius_px"])))
        out["node_paint"]["strength_per_sec"] = float(max(0.0, out["node_paint"]["strength_per_sec"]))
        out["node_paint"]["blend_strength"] = float(max(0.0, out["node_paint"]["blend_strength"]))
    except Exception:
        pass

    # sanitize terrain instrument config
    try:
        out["terrain"]["arc_amp"] = float(max(0.0, min(0.35, float(out["terrain"]["arc_amp"]))))
        out["terrain"]["span_deg"] = float(max(10.0, min(180.0, float(out["terrain"]["span_deg"]))))
        out["terrain"]["plane_w"] = float(max(0.10, min(0.90, float(out["terrain"]["plane_w"]))))
        out["terrain"]["plane_h"] = float(max(0.06, min(0.60, float(out["terrain"]["plane_h"]))))
        out["terrain"]["planet_height_scale"] = float(max(0.0, min(1.0, float(out["terrain"]["planet_height_scale"]))))
        out["terrain"]["planet_height_bias"] = float(max(0.0, min(1.0, float(out["terrain"]["planet_height_bias"]))))
        out["terrain"]["planet_shade_strength"] = float(max(0.0, min(2.0, float(out["terrain"]["planet_shade_strength"]))))
    except Exception:
        pass
    return out


def _ensure_plane_icon_surface(*, path: str, w_px: int = 96, h_px: int = 32) -> pygame.Surface:
    """Load a plane icon PNG, or generate a squat/long placeholder surface."""
    p = str(path or "").strip()
    if p:
        try:
            if os.path.exists(p):
                surf = pygame.image.load(p).convert_alpha()
                return surf
        except Exception:
            pass

    # Fallback: squat long rectangle with a tiny tail.
    surf = pygame.Surface((int(w_px), int(h_px)), flags=pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    body = pygame.Rect(int(0.10 * w_px), int(0.38 * h_px), int(0.78 * w_px), int(0.28 * h_px))
    tail = pygame.Rect(int(0.08 * w_px), int(0.28 * h_px), int(0.10 * w_px), int(0.44 * h_px))
    pygame.draw.rect(surf, (255, 255, 255, 235), body, border_radius=int(max(1, 0.10 * h_px)))
    pygame.draw.rect(surf, (255, 255, 255, 235), tail, border_radius=int(max(1, 0.10 * h_px)))
    # A small nose notch for directionality.
    pygame.draw.circle(surf, (0, 0, 0, 0), (int(0.90 * w_px), int(0.52 * h_px)), int(0.10 * h_px))

    # Best-effort: write to disk so users can replace it with their own PNG.
    try:
        if p:
            folder = os.path.dirname(p)
            if folder:
                os.makedirs(folder, exist_ok=True)
            pygame.image.save(surf, p)
    except Exception:
        pass
    return surf


def _ensure_hud_plane_texture(*, path: str) -> tuple[int | None, tuple[int, int] | None]:
    """Create (or reuse) an OpenGL texture for the HUD plane icon."""
    global _HUD_PLANE_TEX, _HUD_PLANE_TEX_WH

    # If we already created a texture, keep it (texture content changes only when path file changes;
    # for now we keep it simple and require restart to refresh).
    if _HUD_PLANE_TEX is not None and _HUD_PLANE_TEX_WH is not None:
        return _HUD_PLANE_TEX, _HUD_PLANE_TEX_WH

    try:
        surf = _ensure_plane_icon_surface(path=str(path or ""))
        w, h = surf.get_size()
        raw = pygame.image.tostring(surf, "RGBA", True)
        tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(w), int(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)
        _HUD_PLANE_TEX = tex
        _HUD_PLANE_TEX_WH = (int(w), int(h))
        return _HUD_PLANE_TEX, _HUD_PLANE_TEX_WH
    except Exception:
        return None, None


def _load_grayscale_heightmap(path: str) -> np.ndarray | None:
    """Load a grayscale heightmap image as float32 array in [0,1]."""
    p = str(path or "").strip()
    if not p:
        p = str(_DEFAULT_TERRAIN_HEIGHTMAP_PATH)

    # Auto-generate the default heightmap if it is requested but missing.
    try:
        if os.path.normpath(p) == os.path.normpath(str(_DEFAULT_TERRAIN_HEIGHTMAP_PATH)):
            _ensure_default_terrain_heightmap_file(path=p)
    except Exception:
        pass
    try:
        if not os.path.exists(p):
            return None
    except Exception:
        return None

    try:
        mtime = float(os.path.getmtime(p))
    except Exception:
        mtime = -1.0

    cache = _TERRAIN_HEIGHTMAP_CACHE.get(p)
    if isinstance(cache, dict) and float(cache.get("mtime", -2.0)) == mtime:
        arr = cache.get("arr")
        if isinstance(arr, np.ndarray):
            return arr

    try:
        surf = pygame.image.load(p)
        try:
            surf = surf.convert()  # no alpha needed
        except Exception:
            pass
        w, h = surf.get_size()
        raw = pygame.image.tostring(surf, "RGB", True)
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        g = (rgb[:, :, 0].astype(np.float32) + rgb[:, :, 1].astype(np.float32) + rgb[:, :, 2].astype(np.float32)) / (3.0 * 255.0)
        g = np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)
        _TERRAIN_HEIGHTMAP_CACHE[p] = {"mtime": mtime, "arr": g}
        return g
    except Exception:
        return None


def _heightmap_sample_bilinear(hm: np.ndarray, u: float, v: float) -> float:
    """Bilinear sample from an equirectangular map; u wraps, v clamps."""
    if hm is None or hm.size == 0:
        return 0.5
    H, W = hm.shape[0], hm.shape[1]
    if W <= 1 or H <= 1:
        return float(hm[0, 0])

    uu = float(u) % 1.0
    vv = float(max(0.0, min(1.0, v)))

    x = uu * float(W - 1)
    y = vv * float(H - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = (x0 + 1) % int(W)
    y1 = min(int(H - 1), y0 + 1)
    tx = float(x - float(x0))
    ty = float(y - float(y0))

    a = float(hm[y0, x0])
    b = float(hm[y0, x1])
    c = float(hm[y1, x0])
    d = float(hm[y1, x1])
    ab = a + (b - a) * tx
    cd = c + (d - c) * tx
    return float(ab + (cd - ab) * ty)


def _heightmap_sample_bilinear_vec(hm: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sampling from an equirectangular map; u wraps, v clamps."""
    H, W = int(hm.shape[0]), int(hm.shape[1])
    if W <= 1 or H <= 1:
        return np.full_like(u, float(hm[0, 0]), dtype=np.float32)

    uu = np.mod(u.astype(np.float32, copy=False), 1.0)
    vv = np.clip(v.astype(np.float32, copy=False), 0.0, 1.0)
    x = uu * float(W - 1)
    y = vv * float(H - 1)
    x0 = np.floor(x).astype(np.int64, copy=False)
    y0 = np.floor(y).astype(np.int64, copy=False)
    x1 = (x0 + 1) % int(W)
    y1 = np.minimum(int(H - 1), y0 + 1)
    tx = (x - x0.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    ty = (y - y0.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    a = hm[y0, x0].astype(np.float32, copy=False)
    b = hm[y0, x1].astype(np.float32, copy=False)
    c = hm[y1, x0].astype(np.float32, copy=False)
    d = hm[y1, x1].astype(np.float32, copy=False)
    ab = a + (b - a) * tx
    cd = c + (d - c) * tx
    return (ab + (cd - ab) * ty).astype(np.float32, copy=False)


def _draw_ship_panel_groundtrack_arc(
    *,
    eye: np.ndarray,
    ship_right: np.ndarray,
    ship_up: np.ndarray,
    ship_fwd: np.ndarray,
    dist: float,
    off_r: float,
    off_u: float,
    half_w: float,
    half_h: float,
    pos_world: np.ndarray,
    vel_world: np.ndarray,
    ship_heading_rad: float,
    ground_heading_rad: float,
    camera_pitch_rad: float,
    altitude: float,
    atmosphere_thickness: float,
    groundspeed: float,
    gravity_g: float,
    surface_r: float,
    terrain_cfg: dict,
) -> None:
    """Draw a bottom-right great-circle / terrain arc instrument as a thin 3D panel."""
    r = ship_right.astype(np.float32, copy=False)
    u = ship_up.astype(np.float32, copy=False)
    f = ship_fwd.astype(np.float32, copy=False)
    origin = (eye.astype(np.float32, copy=False) + f * float(dist) + r * float(off_r) + u * float(off_u)).astype(np.float32, copy=False)
    hw = float(half_w)
    hh = float(half_h)

    def _v(x_ndc: float, y_ndc: float) -> np.ndarray:
        return (origin + r * (float(x_ndc) * hw) + u * (float(y_ndc) * hh)).astype(np.float32, copy=False)

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Backplate.
    glColor4f(0.0, 0.0, 0.0, 0.22)
    p00 = _v(-1.05, -1.05)
    p10 = _v(1.05, -1.05)
    p11 = _v(1.05, 1.05)
    p01 = _v(-1.05, 1.05)
    glBegin(GL_TRIANGLES)
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p10[0]), float(p10[1]), float(p10[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p01[0]), float(p01[1]), float(p01[2]))
    glEnd()

    # Great-circle basis from current position and ground velocity.
    pos = np.asarray(pos_world, dtype=np.float32)
    n_pos = float(np.linalg.norm(pos))
    if not (np.isfinite(n_pos) and n_pos > 1e-6):
        glDisable(GL_BLEND)
        glDepthMask(True)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)
        return
    up_rad = (pos / n_pos).astype(np.float32, copy=False)

    vel = np.asarray(vel_world, dtype=np.float32)
    # Project velocity into tangent plane.
    vel_t = (vel - up_rad * float(np.dot(vel, up_rad))).astype(np.float32, copy=False)
    n_vt = float(np.linalg.norm(vel_t))
    if not (np.isfinite(n_vt) and n_vt > 1e-6):
        # Fall back to ship forward projected into tangent.
        sf = np.asarray(ship_fwd, dtype=np.float32)
        vel_t = (sf - up_rad * float(np.dot(sf, up_rad))).astype(np.float32, copy=False)
        n_vt = float(np.linalg.norm(vel_t))
        if not (np.isfinite(n_vt) and n_vt > 1e-6):
            glDisable(GL_BLEND)
            glDepthMask(True)
            if depth_was_enabled:
                glEnable(GL_DEPTH_TEST)
            return
    vel_t = (vel_t / n_vt).astype(np.float32, copy=False)

    # Terrain config.
    terrain_enabled = bool(terrain_cfg.get("enabled", False))
    hm = None
    if terrain_enabled:
        hm = _load_grayscale_heightmap(str(terrain_cfg.get("heightmap_path", "") or ""))
        if hm is None:
            terrain_enabled = False
    arc_amp = float(terrain_cfg.get("arc_amp", 0.08))
    # Arc sampling span along the ground-speed great circle.
    # `span_deg` is treated as the max half-span; we scale the effective half-span
    # down at low ground speed so the arc reads terrain mostly "right under us".
    span_deg_max = float(terrain_cfg.get("span_deg", 60.0))
    span_deg_min = float(terrain_cfg.get("span_deg_min", 15.0))
    span_deg_max = float(max(5.0, min(180.0, span_deg_max)))
    span_deg_min = float(max(0.5, min(span_deg_max, span_deg_min)))

    gs = float(max(0.0, groundspeed))
    gs_full = float(terrain_cfg.get("groundspeed_full", 1.25))
    gs_full = float(max(1e-6, gs_full))
    gs_frac = float(max(0.0, min(1.0, gs / gs_full)))

    span_deg_eff = span_deg_min + (span_deg_max - span_deg_min) * gs_frac
    span = float(max(0.01, min(math.pi, math.radians(span_deg_eff))))

    # As altitude increases through the atmosphere band, drop the arc downward in the panel.
    # Only the arc shifts; the plane marker stays fixed.
    alt = float(max(0.0, altitude))
    atmo = float(max(1e-6, atmosphere_thickness))
    alt_frac = float(max(0.0, min(1.0, alt / atmo)))
    arc_drop_max = 0.55
    arc_y_shift = -float(arc_drop_max) * alt_frac

    # Instrument arc (top semicircle) in local panel coords.
    # Slightly smaller radius so the plane can sit above the arc within the panel.
    base_R = 0.72
    segs = 96
    pts_xy: list[tuple[float, float]] = []

    # Visual arc span (in panel space) also scales with ground speed:
    # slow => short/flat segment near the top, showing terrain "right under" the craft
    # fast => approaches a full half-circle.
    theta_span_min = float(terrain_cfg.get("theta_span_min", 0.45))
    theta_span_min = float(max(0.10, min(1.20, theta_span_min)))
    theta_span = float(theta_span_min + (0.5 * math.pi - theta_span_min) * gs_frac)
    theta0 = float((0.5 * math.pi) - theta_span)
    theta1 = float((0.5 * math.pi) + theta_span)

    # Plane position bias along the arc (speed-based): as ground speed increases, bias the
    # displayed window so more of it is "ahead" (to the right) and less is "behind".
    # This makes the plane marker slide left (counterclockwise) at high speed.
    plane_shift = float(terrain_cfg.get("plane_shift", 0.85))
    plane_shift = float(max(0.0, min(1.0, plane_shift)))
    bias = float(max(0.0, min(1.0, plane_shift * gs_frac)))
    # Note: the panel arc parameterization runs from right->left as theta increases.
    # We want "forward/ahead" to be on the right side, so we assign the larger span
    # to the right side as speed rises.
    span_back = float(max(1e-4, span * (1.0 + bias)))
    span_fwd = float(max(1e-4, span * (1.0 - bias)))
    span_total = float(span_back + span_fwd)
    zero_frac = float(max(0.0, min(1.0, span_back / max(1e-6, span_total))))
    theta_plane = float(theta0 + zero_frac * (theta1 - theta0))

    def _terrain_height_norm(dir_unit: np.ndarray) -> float:
        if not terrain_enabled or hm is None:
            return 0.0
        x, y, z = float(dir_unit[0]), float(dir_unit[1]), float(dir_unit[2])
        # Keep consistent with the planet UV convention used elsewhere:
        # lon = atan2(z, x), lat = asin(y).
        lon = math.atan2(z, x)
        lat = math.asin(max(-1.0, min(1.0, y)))
        u0 = 0.5 + (lon / (2.0 * math.pi))
        v0 = 0.5 + (lat / math.pi)
        h01 = _heightmap_sample_bilinear(hm, u0, v0)
        # Map [0,1] -> [-1,1]
        return float(2.0 * (h01 - 0.5))

    glLineWidth(2.0)
    glColor4f(0.98, 0.98, 0.98, 0.85)
    glBegin(GL_LINE_STRIP)
    for i in range(segs + 1):
        t = float(i) / float(segs)
        theta = theta0 + t * (theta1 - theta0)
        # Great-circle angular offset s with the plane marker as the zero reference.
        # t spans [-span_back, +span_fwd] so wherever the marker sits is "straight down".
        # Sign convention: positive s points "forward" and should appear on the +X (right) side.
        s = float(span_back - t * span_total)
        # Surface direction along the ground-track great circle.
        dir_u = (math.cos(s) * up_rad + math.sin(s) * vel_t).astype(np.float32, copy=False)
        n_du = float(np.linalg.norm(dir_u))
        if n_du > 1e-6:
            dir_u = (dir_u / n_du).astype(np.float32, copy=False)
        h_norm = _terrain_height_norm(dir_u)
        R = float(base_R + arc_amp * h_norm)
        x = float(math.cos(theta) * R)
        y0 = float(math.sin(theta) * R)
        pts_xy.append((x, y0))
        p = _v(x, y0 + arc_y_shift)
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()

    # Outer frame (subtle).
    glLineWidth(1.0)
    glColor4f(1.0, 1.0, 1.0, 0.30)
    glBegin(GL_LINE_STRIP)
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glVertex3f(float(p10[0]), float(p10[1]), float(p10[2]))
    glVertex3f(float(p11[0]), float(p11[1]), float(p11[2]))
    glVertex3f(float(p01[0]), float(p01[1]), float(p01[2]))
    glVertex3f(float(p00[0]), float(p00[1]), float(p00[2]))
    glEnd()

    # Interpolate arc point at the plane marker (theta_plane).
    if len(pts_xy) >= 2:
        t_idx = float(max(0.0, min(1.0, zero_frac)))
        idx_f = t_idx * float(len(pts_xy) - 1)
        idx0 = int(max(0, min(len(pts_xy) - 1, math.floor(idx_f))))
        idx1 = int(max(0, min(len(pts_xy) - 1, idx0 + 1)))
        a = float(idx_f - float(idx0))
        x0, y0 = pts_xy[idx0]
        x1, y1 = pts_xy[idx1]
        px = float((1.0 - a) * x0 + a * x1)
        py = float((1.0 - a) * y0 + a * y1)
    else:
        px, py = (0.0, float(base_R))
    # When touching the ground (altitude ~ 0), the plane should touch the arc.
    # We accomplish the "airborne separation" by dropping only the arc with altitude.
    px_o, py_o = px, py

    # Gravity-only drop projection curve: show the predicted impact point on the arc.
    # Uses a constant-g approximation with along-track distance = groundspeed * t.
    g = float(max(0.0, gravity_g))
    if g > 1e-6 and alt > 1e-6 and span_total > 1e-6 and surface_r > 1e-6:
        t_imp = float(math.sqrt((2.0 * alt) / g))
        s_imp = float((gs * t_imp) / float(surface_r))
        s_imp = float(max(-span_fwd, min(span_back, s_imp)))
        # Invert the mapping used above: s = span_back - t*span_total.
        t_imp_frac = float((span_back - s_imp) / span_total)
        if len(pts_xy) >= 2:
            idx_f = t_imp_frac * float(len(pts_xy) - 1)
            idx0 = int(max(0, min(len(pts_xy) - 1, math.floor(idx_f))))
            idx1 = int(max(0, min(len(pts_xy) - 1, idx0 + 1)))
            a = float(idx_f - float(idx0))
            x0, y0 = pts_xy[idx0]
            x1, y1 = pts_xy[idx1]
            ix = float((1.0 - a) * x0 + a * x1)
            iy = float((1.0 - a) * y0 + a * y1 + arc_y_shift)

            sag = float(0.25 * abs((py_o) - iy))
            glLineWidth(2.0)
            glColor4f(1.0, 1.0, 1.0, 0.45)
            glBegin(GL_LINE_STRIP)
            nseg = 32
            for j in range(nseg + 1):
                tt = float(j) / float(nseg)
                x = float((1.0 - tt) * px_o + tt * ix)
                y_lin = float((1.0 - tt) * py_o + tt * iy)
                y = float(y_lin + sag * 4.0 * tt * (1.0 - tt))
                p = _v(x, y)
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

    # Draw the plane icon as a textured quad oriented tangentially to the arc.
    plane_tex, _ = _ensure_hud_plane_texture(path=str(terrain_cfg.get("plane_icon_path", "assets/hud/plane_icon.png")))
    plane_w = float(terrain_cfg.get("plane_w", 0.42))
    plane_h = float(terrain_cfg.get("plane_h", 0.16))

    # Orient the plane along the arc and tilt by camera pitch.
    # We choose the tangent direction that points "nose-right" at the top of the arc.
    base_ang = float(math.atan2(-math.cos(theta_plane), math.sin(theta_plane)))
    ang = float(base_ang + camera_pitch_rad)
    ang = float(max(-1.25, min(1.25, ang)))
    ca = float(math.cos(ang))
    sa = float(math.sin(ang))
    sx, sy = ca, sa
    ux, uy = -sa, ca

    c = _v(px_o, py_o)
    # Convert 2D panel offsets to world offsets.
    def _offset(dx: float, dy: float) -> np.ndarray:
        return (r * (dx * hw) + u * (dy * hh)).astype(np.float32, copy=False)

    half_pw = 0.5 * plane_w
    half_ph = 0.5 * plane_h
    o1 = _offset(sx * (-half_pw) + ux * (-half_ph), sy * (-half_pw) + uy * (-half_ph))
    o2 = _offset(sx * (half_pw) + ux * (-half_ph), sy * (half_pw) + uy * (-half_ph))
    o3 = _offset(sx * (half_pw) + ux * (half_ph), sy * (half_pw) + uy * (half_ph))
    o4 = _offset(sx * (-half_pw) + ux * (half_ph), sy * (-half_pw) + uy * (half_ph))

    if plane_tex is not None and int(plane_tex) > 0:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, int(plane_tex))
        glColor4f(1.0, 1.0, 1.0, 0.92)
        glBegin(GL_TRIANGLES)
        # tri 1
        glTexCoord2f(0.0, 0.0)
        glVertex3f(float((c + o1)[0]), float((c + o1)[1]), float((c + o1)[2]))
        glTexCoord2f(1.0, 0.0)
        glVertex3f(float((c + o2)[0]), float((c + o2)[1]), float((c + o2)[2]))
        glTexCoord2f(1.0, 1.0)
        glVertex3f(float((c + o3)[0]), float((c + o3)[1]), float((c + o3)[2]))
        # tri 2
        glTexCoord2f(0.0, 0.0)
        glVertex3f(float((c + o1)[0]), float((c + o1)[1]), float((c + o1)[2]))
        glTexCoord2f(1.0, 1.0)
        glVertex3f(float((c + o3)[0]), float((c + o3)[1]), float((c + o3)[2]))
        glTexCoord2f(0.0, 1.0)
        glVertex3f(float((c + o4)[0]), float((c + o4)[1]), float((c + o4)[2]))
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
    else:
        # Fallback: simple white quad outline.
        glLineWidth(2.0)
        glColor4f(0.98, 0.98, 0.98, 0.85)
        glBegin(GL_LINE_STRIP)
        for o in (o1, o2, o3, o4, o1):
            p = c + o
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


def _ensure_node_paint_target(*, w: int, h: int) -> None:
    global _NODE_PAINT_TEX, _NODE_PAINT_FBO, _NODE_PAINT_W, _NODE_PAINT_H

    w = int(max(64, min(4096, w)))
    h = int(max(64, min(4096, h)))
    if _NODE_PAINT_TEX is not None and _NODE_PAINT_FBO is not None and _NODE_PAINT_W == w and _NODE_PAINT_H == h:
        return

    # Tear down old resources (best-effort; GL context may already be closing).
    try:
        if _NODE_PAINT_FBO is not None:
            glDeleteFramebuffers(1, [int(_NODE_PAINT_FBO)])
    except Exception:
        pass
    try:
        if _NODE_PAINT_TEX is not None:
            glDeleteTextures(1, [int(_NODE_PAINT_TEX)])
    except Exception:
        pass

    _NODE_PAINT_TEX = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, int(_NODE_PAINT_TEX))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(w), int(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    _NODE_PAINT_FBO = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, int(_NODE_PAINT_FBO))
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, int(_NODE_PAINT_TEX), 0)
    status = int(glCheckFramebufferStatus(GL_FRAMEBUFFER))
    if status == int(GL_FRAMEBUFFER_COMPLETE):
        # Initialize to black/transparent.
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != int(GL_FRAMEBUFFER_COMPLETE):
        # Disable paint if FBO creation fails.
        _NODE_PAINT_TEX = None
        _NODE_PAINT_FBO = None
        _NODE_PAINT_W = 0
        _NODE_PAINT_H = 0
        return

    _NODE_PAINT_W = int(w)
    _NODE_PAINT_H = int(h)


def _update_node_paint(
    *,
    scene: dict,
    node_dirs_unit_f32: np.ndarray,
    node_colors_f32: np.ndarray,
    n_active: int,
    dt_s: float,
) -> None:
    """Accumulate node trails into a planet-wrapped paint texture.

    Paint is stored in an RGBA texture in equirectangular UV space:
    u = lon/(2π)+0.5, v = lat/π+0.5.
    """
    paint = scene.get("node_paint", {}) if isinstance(scene.get("node_paint", {}), dict) else {}
    if not bool(paint.get("enabled", True)):
        return

    try:
        w, h = paint.get("resolution", (512, 256))
        w_i, h_i = int(w), int(h)
    except Exception:
        w_i, h_i = 512, 256
    _ensure_node_paint_target(w=w_i, h=h_i)
    if _NODE_PAINT_TEX is None or _NODE_PAINT_FBO is None:
        return

    n = int(n_active)
    if n <= 0:
        return

    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.0
    dt = float(min(0.25, dt))

    decay_per_sec = float(paint.get("decay_per_sec", 0.0))
    decay_per_sec = float(max(0.0, decay_per_sec))
    decay_alpha = float(min(1.0, decay_per_sec * dt))

    radius_px = float(paint.get("radius_px", 6.0))
    radius_px = float(max(0.5, min(256.0, radius_px)))

    strength_per_sec = float(paint.get("strength_per_sec", 5.0))
    strength_per_sec = float(max(0.0, strength_per_sec))
    stamp_alpha = float(min(1.0, strength_per_sec * dt))
    if stamp_alpha <= 0.0 and decay_alpha <= 0.0:
        return

    du = float(radius_px) / float(max(1, _NODE_PAINT_W))
    dv = float(radius_px) / float(max(1, _NODE_PAINT_H))
    du = float(max(1e-6, du))
    dv = float(max(1e-6, dv))

    prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
    prev_vp = glGetIntegerv(GL_VIEWPORT)

    glBindFramebuffer(GL_FRAMEBUFFER, int(_NODE_PAINT_FBO))
    glViewport(0, 0, int(_NODE_PAINT_W), int(_NODE_PAINT_H))

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Fade existing paint toward black in-place.
    if decay_alpha > 0.0:
        glColor4f(0.0, 0.0, 0.0, float(decay_alpha))
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glEnd()

    # Stamp node colors (charge-derived) into UV space.
    if stamp_alpha > 0.0:
        glBegin(GL_TRIANGLES)
        for i in range(n):
            x = float(node_dirs_unit_f32[i, 0])
            y = float(node_dirs_unit_f32[i, 1])
            z = float(node_dirs_unit_f32[i, 2])
            nn = math.sqrt(x * x + y * y + z * z)
            if not (nn > 1e-9):
                continue
            x /= nn
            y /= nn
            z /= nn

            lon = math.atan2(y, x)
            lat = math.asin(max(-1.0, min(1.0, z)))
            u0 = 0.5 + (lon / (2.0 * math.pi))
            v0 = 0.5 + (lat / math.pi)
            v0 = float(max(0.0, min(1.0, v0)))

            r = float(node_colors_f32[i, 0])
            g = float(node_colors_f32[i, 1])
            b = float(node_colors_f32[i, 2])
            glColor4f(r, g, b, float(stamp_alpha))

            # Handle seam wrap by stamping an extra copy when near u=0/1.
            u_list = [u0]
            if u0 < du:
                u_list.append(u0 + 1.0)
            elif u0 > 1.0 - du:
                u_list.append(u0 - 1.0)

            for u in u_list:
                x0 = float(u - du)
                x1 = float(u + du)
                y0 = float(v0 - dv)
                y1 = float(v0 + dv)
                if y1 < 0.0 or y0 > 1.0:
                    continue
                y0 = float(max(0.0, min(1.0, y0)))
                y1 = float(max(0.0, min(1.0, y1)))

                glVertex3f(x0, y0, 0.0)
                glVertex3f(x1, y0, 0.0)
                glVertex3f(x1, y1, 0.0)
                glVertex3f(x0, y0, 0.0)
                glVertex3f(x1, y1, 0.0)
                glVertex3f(x0, y1, 0.0)
        glEnd()

    glDisable(GL_BLEND)
    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glViewport(int(prev_vp[0]), int(prev_vp[1]), int(prev_vp[2]), int(prev_vp[3]))
    glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))


def _sphere_tri_mesh(stacks: int = 12, slices: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts, norms) as float32 arrays shaped (T*3,3)."""
    key = (int(stacks), int(slices))
    if key in _SPHERE_MESH_CACHE:
        return _SPHERE_MESH_CACHE[key]

    stacks = max(4, int(stacks))
    slices = max(8, int(slices))
    verts: list[list[float]] = []
    norms: list[list[float]] = []

    for i in range(stacks):
        lat0 = math.pi * (-0.5 + (i / stacks))
        lat1 = math.pi * (-0.5 + ((i + 1) / stacks))
        z0 = math.sin(lat0)
        z1 = math.sin(lat1)
        r0 = math.cos(lat0)
        r1 = math.cos(lat1)
        for j in range(slices):
            lon0 = 2.0 * math.pi * (j / slices)
            lon1 = 2.0 * math.pi * ((j + 1) / slices)
            x00 = math.cos(lon0) * r0
            y00 = math.sin(lon0) * r0
            x01 = math.cos(lon1) * r0
            y01 = math.sin(lon1) * r0
            x10 = math.cos(lon0) * r1
            y10 = math.sin(lon0) * r1
            x11 = math.cos(lon1) * r1
            y11 = math.sin(lon1) * r1

            v00 = (x00, y00, z0)
            v01 = (x01, y01, z0)
            v10 = (x10, y10, z1)
            v11 = (x11, y11, z1)

            # Two triangles: (v00, v10, v11) and (v00, v11, v01)
            for a, b, c in ((v00, v10, v11), (v00, v11, v01)):
                verts.append([a[0], a[1], a[2]])
                verts.append([b[0], b[1], b[2]])
                verts.append([c[0], c[1], c[2]])
                norms.append([a[0], a[1], a[2]])
                norms.append([b[0], b[1], b[2]])
                norms.append([c[0], c[1], c[2]])

    v_arr = np.asarray(verts, dtype=np.float32)
    n_arr = np.asarray(norms, dtype=np.float32)
    _SPHERE_MESH_CACHE[key] = (v_arr, n_arr)
    return v_arr, n_arr


def _draw_planet_and_atmosphere(
    *,
    planet_r: float = _PLANET_DRAW_R,
    atmosphere_r: float = _ATMOSPHERE_R,
    light_dir: np.ndarray | None = None,
    paint_tex: int | None = None,
    paint_strength: float = 1.0,
    terrain_heightmap: np.ndarray | None = None,
    terrain_height_scale: float = 0.0,
    terrain_height_bias: float = 0.5,
    terrain_shade_strength: float = 0.0,
) -> None:
    """Draw an interior planet sphere plus a faint atmosphere boundary.

    Performance note: this uses a cached VBO to avoid Python per-vertex loops.
    """
    global _PLANET_VBO_POS, _PLANET_VBO_COL, _PLANET_VBO_COUNT, _PLANET_VBO_POS_TERRAIN
    global _PLANET_PROG, _PLANET_A_POS, _PLANET_A_COL, _PLANET_U_SCALE
    global _PLANET_U_PAINT_TEX, _PLANET_U_PAINT_STRENGTH
    global _PLANET_TERRAIN_KEY

    glDisable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    # Write depth for the planet so it can occlude points/edges behind the horizon.
    glDepthMask(True)

    global _PLANET_LIGHT_DIR_KEY

    # Allow the scene to drive the directional light used for the planet's baked vertex colors.
    if light_dir is None:
        light = np.array([-0.25, 0.35, 1.0], dtype=np.float32)
    else:
        light = np.asarray(light_dir, dtype=np.float32)
    ln = float(np.linalg.norm(light))
    if ln > 1e-6:
        light = light / ln
    else:
        light = np.array([-0.25, 0.35, 1.0], dtype=np.float32)

    light_key = (float(light[0]), float(light[1]), float(light[2]))

    if _PLANET_VBO_POS is None:
        verts, norms = _sphere_tri_mesh(32, 64)
        _PLANET_VBO_COUNT = int(verts.shape[0])

        # Procedural "texture": subtle shading + lat/lon banding for scale cues.
        _PLANET_VBO_POS = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, _PLANET_VBO_POS)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Seed the color buffer on first creation.
        _PLANET_VBO_COL = int(glGenBuffers(1))
        _PLANET_LIGHT_DIR_KEY = None

        # Optional terrain-displaced position VBO.
        _PLANET_VBO_POS_TERRAIN = int(glGenBuffers(1))
        _PLANET_TERRAIN_KEY = None

    # Compile a tiny planet program on demand (avoids legacy client-state arrays).
    if _PLANET_PROG is None:
        try:
            planet_vs = """
            #version 120
            attribute vec3 a_pos;
            attribute vec3 a_color;
            varying vec3 v_color;
            varying vec3 v_n;
            uniform float u_scale;
            void main() {
                v_color = a_color;
                v_n = normalize(a_pos);
                vec4 eyePos = gl_ModelViewMatrix * vec4(a_pos * u_scale, 1.0);
                gl_Position = gl_ProjectionMatrix * eyePos;
            }
            """
            planet_fs = """
            #version 120
            varying vec3 v_color;
            varying vec3 v_n;
            uniform sampler2D u_paint_tex;
            uniform float u_paint_strength;
            void main() {
                vec3 n = normalize(v_n);
                float pi = 3.141592653589793;
                float lon = atan(n.y, n.x);
                float lat = asin(clamp(n.z, -1.0, 1.0));
                float u = 0.5 + (lon / (2.0 * pi));
                float v = 0.5 + (lat / pi);
                vec4 p = texture2D(u_paint_tex, vec2(u, v));
                float a = clamp(p.a * u_paint_strength, 0.0, 1.0);
                vec3 outc = mix(v_color, p.rgb, a);
                gl_FragColor = vec4(outc, 1.0);
            }
            """
            _PLANET_PROG = int(_gl_compile_program(planet_vs, planet_fs))
            _PLANET_A_POS = int(glGetAttribLocation(_PLANET_PROG, "a_pos"))
            _PLANET_A_COL = int(glGetAttribLocation(_PLANET_PROG, "a_color"))
            _PLANET_U_SCALE = int(glGetUniformLocation(_PLANET_PROG, "u_scale"))
            _PLANET_U_PAINT_TEX = int(glGetUniformLocation(_PLANET_PROG, "u_paint_tex"))
            _PLANET_U_PAINT_STRENGTH = int(glGetUniformLocation(_PLANET_PROG, "u_paint_strength"))
        except Exception:
            _PLANET_PROG = None

    if _PLANET_VBO_COL is None:
        _PLANET_VBO_COL = int(glGenBuffers(1))

    # Terrain displacement key (changes when heightmap or params change).
    terr_enabled = (
        terrain_heightmap is not None
        and isinstance(terrain_heightmap, np.ndarray)
        and terrain_heightmap.size > 0
        and float(terrain_height_scale) > 0.0
    )
    terr_key: tuple[object, ...] = (
        bool(terr_enabled),
        int(terrain_heightmap.shape[1]) if terr_enabled else 0,
        int(terrain_heightmap.shape[0]) if terr_enabled else 0,
        float(planet_r),
        float(terrain_height_scale),
        float(terrain_height_bias),
        # include a cheap checksum so swapping files without restarting still updates
        float(terrain_heightmap.mean()) if terr_enabled else 0.0,
        float(terrain_heightmap.std()) if terr_enabled else 0.0,
    )

    # (Re)build the position VBO for terrain displacement when needed.
    if terr_enabled and (_PLANET_TERRAIN_KEY is None or _PLANET_TERRAIN_KEY != terr_key):
        _, norms = _sphere_tri_mesh(32, 64)
        n = norms.astype(np.float32, copy=False)
        lon = np.arctan2(n[:, 1], n[:, 0]).astype(np.float32, copy=False)
        lat = np.arcsin(np.clip(n[:, 2], -1.0, 1.0)).astype(np.float32, copy=False)
        u0 = (0.5 + (lon / (2.0 * math.pi))).astype(np.float32, copy=False)
        v0 = (0.5 + (lat / math.pi)).astype(np.float32, copy=False)
        h01 = _heightmap_sample_bilinear_vec(terrain_heightmap, u0, v0)
        disp = (h01 - float(terrain_height_bias)).astype(np.float32, copy=False) * float(terrain_height_scale)
        rr = (float(planet_r) + disp).astype(np.float32, copy=False)
        rr = np.maximum(rr, 1e-6).astype(np.float32, copy=False)
        verts_terr = (n * rr[:, None]).astype(np.float32, copy=False)

        if _PLANET_VBO_POS_TERRAIN is None:
            _PLANET_VBO_POS_TERRAIN = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS_TERRAIN))
        glBufferData(GL_ARRAY_BUFFER, verts_terr.nbytes, verts_terr, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        _PLANET_TERRAIN_KEY = terr_key

    # (Re)build the color VBO when light direction OR terrain shading inputs change.
    color_key: tuple[object, ...] = (
        light_key,
        float(terrain_shade_strength),
        terr_key,
    )
    if _PLANET_LIGHT_DIR_KEY is None or _PLANET_LIGHT_DIR_KEY != color_key:
        _, norms = _sphere_tri_mesh(32, 64)
        n = norms.astype(np.float32, copy=False)
        diff = np.maximum(0.0, (n @ light).astype(np.float32, copy=False))
        lon = np.arctan2(n[:, 1], n[:, 0]).astype(np.float32, copy=False)
        lat = np.arcsin(np.clip(n[:, 2], -1.0, 1.0)).astype(np.float32, copy=False)
        band = 0.5 + 0.5 * np.sin(4.0 * lon + 3.0 * lat) * np.cos(2.0 * lat)
        shade = (0.48 + 0.28 * diff + 0.14 * band).astype(np.float32, copy=False)
        shade = np.clip(shade, 0.22, 0.98)

        tropics = (0.5 + 0.5 * np.cos(2.0 * lat)).astype(np.float32, copy=False)
        noise = (0.5 + 0.5 * np.sin(3.0 * lon + 1.7 * lat) * np.cos(2.3 * lat)).astype(np.float32, copy=False)

        r = np.clip(shade * (1.05 + 0.20 * noise + 0.05 * tropics), 0.0, 1.0)
        g = np.clip(shade * (0.62 + 0.10 * noise + 0.06 * tropics), 0.0, 1.0)
        b = np.clip(shade * (0.36 + 0.06 * noise - 0.04 * tropics), 0.0, 1.0)
        cols = np.stack([r, g, b], axis=1).astype(np.float32, copy=False)

        # Height-based shading modulation (optional).
        if terr_enabled and float(terrain_shade_strength) > 0.0:
            u0 = (0.5 + (lon / (2.0 * math.pi))).astype(np.float32, copy=False)
            v0 = (0.5 + (lat / math.pi)).astype(np.float32, copy=False)
            h01 = _heightmap_sample_bilinear_vec(terrain_heightmap, u0, v0)
            h_norm = (h01 - 0.5).astype(np.float32, copy=False) * 2.0
            mul = (1.0 + float(terrain_shade_strength) * 0.18 * h_norm).astype(np.float32, copy=False)
            cols = np.clip(cols * mul[:, None], 0.0, 1.0).astype(np.float32, copy=False)

        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_COL))
        glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        _PLANET_LIGHT_DIR_KEY = color_key

    # Solid-ish ground with subtle vertex color variation for scale cues.
    glDisable(GL_LIGHTING)
    glDisable(GL_BLEND)

    if _PLANET_PROG is not None and _PLANET_A_POS is not None and _PLANET_A_COL is not None:
        glUseProgram(int(_PLANET_PROG))
        # If terrain is enabled, we upload already-scaled vertex positions, so u_scale=1.
        if _PLANET_U_SCALE is not None and int(_PLANET_U_SCALE) >= 0:
            glUniform1f(int(_PLANET_U_SCALE), float(1.0 if terr_enabled else planet_r))

        # Node paint overlay (optional).
        if (
            paint_tex is not None
            and int(paint_tex) > 0
            and _PLANET_U_PAINT_TEX is not None
            and int(_PLANET_U_PAINT_TEX) >= 0
        ):
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, int(paint_tex))
            glUniform1i(int(_PLANET_U_PAINT_TEX), 0)
            if _PLANET_U_PAINT_STRENGTH is not None and int(_PLANET_U_PAINT_STRENGTH) >= 0:
                glUniform1f(int(_PLANET_U_PAINT_STRENGTH), float(max(0.0, paint_strength)))
        else:
            # Bind a null texture; shader samples black.
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)
            if _PLANET_U_PAINT_TEX is not None and int(_PLANET_U_PAINT_TEX) >= 0:
                glUniform1i(int(_PLANET_U_PAINT_TEX), 0)
            if _PLANET_U_PAINT_STRENGTH is not None and int(_PLANET_U_PAINT_STRENGTH) >= 0:
                glUniform1f(int(_PLANET_U_PAINT_STRENGTH), 0.0)

        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS_TERRAIN if terr_enabled and _PLANET_VBO_POS_TERRAIN is not None else _PLANET_VBO_POS))
        glEnableVertexAttribArray(int(_PLANET_A_POS))
        glVertexAttribPointer(int(_PLANET_A_POS), 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_COL))
        glEnableVertexAttribArray(int(_PLANET_A_COL))
        glVertexAttribPointer(int(_PLANET_A_COL), 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDrawArrays(GL_TRIANGLES, 0, int(_PLANET_VBO_COUNT))

        glDisableVertexAttribArray(int(_PLANET_A_POS))
        glDisableVertexAttribArray(int(_PLANET_A_COL))
        glUseProgram(0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

    # Surface graticule (lat/lon lines) for additional texture.
    # Depth-test ON so far-side lines are naturally occluded by the planet.
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    glColor4f(0.10, 0.10, 0.10, 0.18)
    rings = 10
    segs = 72
    r_line = float(planet_r) * 1.001
    for ri in range(1, rings):
        lat = math.pi * (-0.5 + (ri / rings))
        z = math.sin(lat)
        rr = math.cos(lat)
        glBegin(GL_LINE_STRIP)
        for sj in range(segs + 1):
            lon = 2.0 * math.pi * (sj / segs)
            x = math.cos(lon) * rr
            y = math.sin(lon) * rr
            glVertex3f(float(x * r_line), float(y * r_line), float(z * r_line))
        glEnd()
    meridians = 10
    for mj in range(meridians):
        lon = 2.0 * math.pi * (mj / meridians)
        glBegin(GL_LINE_STRIP)
        for ri in range(segs + 1):
            lat = math.pi * (-0.5 + (ri / segs))
            z = math.sin(lat)
            rr = math.cos(lat)
            x = math.cos(lon) * rr
            y = math.sin(lon) * rr
            glVertex3f(float(x * r_line), float(y * r_line), float(z * r_line))
        glEnd()
    glDisable(GL_BLEND)
    glDepthMask(True)

    # Atmosphere boundary: faint wireframe lat/lon rings.
    # Don't write depth for the rings so they don't punch holes in the world.
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    glColor4f(0.70, 0.80, 0.95, 0.08)
    rings = 6
    segs = 48
    for ri in range(1, rings):
        lat = math.pi * (-0.5 + (ri / rings))
        z = math.sin(lat)
        rr = math.cos(lat)
        glBegin(GL_LINE_STRIP)
        for sj in range(segs + 1):
            lon = 2.0 * math.pi * (sj / segs)
            x = math.cos(lon) * rr
            y = math.sin(lon) * rr
            glVertex3f(float(x * atmosphere_r), float(y * atmosphere_r), float(z * atmosphere_r))
        glEnd()
    glDisable(GL_BLEND)
    glDepthMask(True)


def _draw_sea_sphere(
    *,
    sea_r: float,
    color: tuple[float, float, float] = (0.08, 0.26, 0.52),
    alpha: float = 0.55,
    depth_write: bool = False,
) -> None:
    """Draw a simple sea surface as a sphere at radius sea_r.

    This is intentionally "cheap": draw a full sphere and rely on depth testing
    against the planet terrain to naturally mask it.
    """
    global _SEA_PROG, _SEA_A_POS, _SEA_U_SCALE, _SEA_U_COLOR
    global _PLANET_VBO_POS, _PLANET_VBO_COUNT

    try:
        r = float(sea_r)
    except Exception:
        return
    if not np.isfinite(r) or r <= 1e-6:
        return

    # Ensure we have a unit-sphere position VBO to render from.
    if _PLANET_VBO_POS is None or int(_PLANET_VBO_COUNT) <= 0:
        verts, _norms = _sphere_tri_mesh(32, 64)
        _PLANET_VBO_COUNT = int(verts.shape[0])
        _PLANET_VBO_POS = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS))
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Compile sea shader lazily.
    if _SEA_PROG is None:
        try:
            sea_vs = """
            #version 120
            attribute vec3 a_pos;
            uniform float u_scale;
            void main() {
                vec4 eyePos = gl_ModelViewMatrix * vec4(a_pos * u_scale, 1.0);
                gl_Position = gl_ProjectionMatrix * eyePos;
            }
            """
            sea_fs = """
            #version 120
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
            }
            """
            _SEA_PROG = int(_gl_compile_program(sea_vs, sea_fs))
            _SEA_A_POS = int(glGetAttribLocation(_SEA_PROG, "a_pos"))
            _SEA_U_SCALE = int(glGetUniformLocation(_SEA_PROG, "u_scale"))
            _SEA_U_COLOR = int(glGetUniformLocation(_SEA_PROG, "u_color"))
        except Exception:
            _SEA_PROG = None

    if _SEA_PROG is None or _SEA_A_POS is None:
        return

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    if not depth_was_enabled:
        glEnable(GL_DEPTH_TEST)

    glDisable(GL_CULL_FACE)

    a = float(max(0.0, min(1.0, alpha)))
    if a < 1.0:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    else:
        glDisable(GL_BLEND)

    # Usually keep sea from writing depth so submerged terrain stays visible through it.
    glDepthMask(bool(depth_write))

    glUseProgram(int(_SEA_PROG))
    if _SEA_U_SCALE is not None and int(_SEA_U_SCALE) >= 0:
        # Slight lift to reduce z-fighting at coastlines.
        glUniform1f(int(_SEA_U_SCALE), float(r * 1.00015))
    if _SEA_U_COLOR is not None and int(_SEA_U_COLOR) >= 0:
        glUniform4f(int(_SEA_U_COLOR), float(color[0]), float(color[1]), float(color[2]), float(a))

    glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS))
    glEnableVertexAttribArray(int(_SEA_A_POS))
    glVertexAttribPointer(int(_SEA_A_POS), 3, GL_FLOAT, False, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glDrawArrays(GL_TRIANGLES, 0, int(_PLANET_VBO_COUNT))
    glDisableVertexAttribArray(int(_SEA_A_POS))
    glUseProgram(0)

    # Restore common defaults expected by the rest of the renderer.
    glDepthMask(True)
    glDisable(GL_BLEND)
    if not depth_was_enabled:
        glDisable(GL_DEPTH_TEST)


def _clamp_inside_radius(p: np.ndarray, r_max: float) -> np.ndarray:
    nrm = float(np.linalg.norm(p))
    if not (nrm > 1e-9):
        return p
    if nrm > r_max:
        p *= (r_max / nrm)
    return p


def _clamp_outside_radius(p: np.ndarray, r_min: float) -> np.ndarray:
    nrm = float(np.linalg.norm(p))
    if not (nrm > 1e-9):
        # Avoid degenerate origin; pick an arbitrary direction.
        p[:] = (0.0, 0.0, float(r_min))
        return p
    if nrm < r_min:
        p *= (r_min / nrm)
    return p


def _proj_mode_norm(mode: str | None) -> str:
    return str(mode or "").strip().lower()


def _load_airplane_spec(path: object | None) -> dict:
    """Load a simple airplane spec JSON.

    This is renderer-only: it defines control mappings, rates, and integer offsets
    for centers of thrust/lift/actuation so the "flight sim" concepts are pinned to
    geometry.
    """
    default = {
        "name": "default",
        "centers": {"mass": [0, 0, 0], "lift": [0, 0, 1], "thrust": [0, 0, -2]},
        "actuation": {"rudder": [0, 0, -3], "elevator": [0, 0, -3]},
        "rigid_body": {"mass": 1.0, "inertia_diag": [200.0, 200.0, 200.0]},
        "control_system": {"mode": "auto", "auto_kp": 6.0, "auto_kd": 2.5},
        # Optional arm definitions used by the C flight sim.
        # Keep these symmetric so they produce torque with ~0 net force.
        "arms": [
            {"type": "thruster", "input_idx": 2, "pos_b": [0, 0, -3], "dir_b": [0, 1, 0], "max_force": 0.02},
            {"type": "thruster", "input_idx": 2, "pos_b": [0, 0, 3], "dir_b": [0, -1, 0], "max_force": 0.02},
            {"type": "thruster", "input_idx": 3, "pos_b": [0, 0, -3], "dir_b": [-1, 0, 0], "max_force": 0.02},
            {"type": "thruster", "input_idx": 3, "pos_b": [0, 0, 3], "dir_b": [1, 0, 0], "max_force": 0.02},
            {"type": "thruster", "input_idx": 4, "pos_b": [3, 0, 0], "dir_b": [0, 1, 0], "max_force": 0.02},
            {"type": "thruster", "input_idx": 4, "pos_b": [-3, 0, 0], "dir_b": [0, -1, 0], "max_force": 0.02},
        ],
        "controls": {
            "deadzone": 0.08,
            "curve_exp": 2.5,
            "left_stick": {"rudder_axis": 0, "rudder_invert": True, "elevator_axis": 1, "elevator_invert": False},
            "right_stick": {"view_yaw_axis": 2, "view_yaw_invert": True, "view_pitch_axis": 3, "view_pitch_invert": True},
            "triggers": {"reverse_axis": 4, "forward_axis": 5},
        },
        "rates": {"craft_yaw_rate": 1.6, "craft_pitch_rate": 1.2, "view_yaw_rate": 1.25, "view_pitch_rate": 1.10},
        "motion": {"base_speed": 0.38, "thrust_accel": 0.12, "max_speed": 2.50, "auto_heading_on_move": False},
    }

    if path is None:
        return default
    p = str(path).strip()
    if not p:
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        out = default
        # Shallow merge is plenty for this use.
        out.update(data)

        # Optional flight-physics overrides (kept distinct from airplane.json).
        # These override only the "motion" subsection at runtime.
        try:
            fp_path = "flight_physics.json"
            if os.path.exists(fp_path):
                with open(fp_path, "r", encoding="utf-8") as ff:
                    fp_root = json.load(ff)
                fp_blk = fp_root.get("flight_physics") if isinstance(fp_root, dict) else None
                if isinstance(fp_blk, dict):
                    motion = out.get("motion", None)
                    if not isinstance(motion, dict):
                        motion = {}
                        out["motion"] = motion
                    motion.update(fp_blk)
        except Exception:
            pass

        # Optional joystick.json overrides for control axis indices.
        # This keeps flight-control bindings decoupled from airplane.json.
        try:
            with open("joystick.json", "r", encoding="utf-8") as jf:
                jcfg = json.load(jf)
            if isinstance(jcfg, dict):
                fc = jcfg.get("flight_controls", None)
                if isinstance(fc, dict):
                    controls = out.get("controls", None)
                    if not isinstance(controls, dict):
                        controls = {}
                        out["controls"] = controls

                    # Merge nested groups if provided.
                    for group in ("left_stick", "right_stick", "triggers"):
                        src = fc.get(group, None)
                        if not isinstance(src, dict):
                            continue
                        dst = controls.get(group, None)
                        if not isinstance(dst, dict):
                            dst = {}
                            controls[group] = dst
                        for k, v in src.items():
                            # Most control entries are integer axis indices.
                            if isinstance(v, int):
                                dst[str(k)] = int(v)
                                continue

                            # Triggers may be stored as {"axis": <int>, "sign": (+1|-1)}
                            # to support controllers where both triggers share one axis.
                            if group == "triggers" and isinstance(v, dict):
                                a = v.get("axis")
                                s = v.get("sign")
                                if isinstance(a, int) and isinstance(s, int) and int(s) in (-1, 1):
                                    dst[str(k)] = {"axis": int(a), "sign": int(s)}
        except Exception:
            pass

        return out
    except Exception:
        return default



def _scene_sky_color(scene: dict, alt_frac: float) -> tuple[float, float, float]:
    sky = scene.get("sky", {}) if isinstance(scene.get("sky", {}), dict) else {}
    space = sky.get("space_color", (0.02, 0.03, 0.06))
    atmo = sky.get("atmo_color", (0.32, 0.52, 0.86))
    curve = float(sky.get("curve", 1.8))
    t = float(max(0.0, min(1.0, alt_frac)))
    # At ground: atmo. At top: space.
    w = float(t ** max(0.25, curve))
    r = (1.0 - w) * float(atmo[0]) + w * float(space[0])
    g = (1.0 - w) * float(atmo[1]) + w * float(space[1])
    b = (1.0 - w) * float(atmo[2]) + w * float(space[2])
    return (float(max(0.0, min(1.0, r))), float(max(0.0, min(1.0, g))), float(max(0.0, min(1.0, b))))


def _is_map_proj_mode(mode: str | None) -> bool:
    return _proj_mode_norm(mode) in _MAP_PROJ_MODES


def _is_ship_proj_mode(mode: str | None) -> bool:
    return _proj_mode_norm(mode) == "ship"


def _basis_mode_for_display(mode: str | None) -> str:
    """Map/ship are nonlinear post-projections; pick a linear basis mode first."""
    m = _proj_mode_norm(mode)
    if m in _MAP_PROJ_MODES or m == "ship":
        return "pca"
    return m or "pca"


def _apply_map_projection_np(pos3_unit: np.ndarray, mode: str) -> None:
    """In-place: unit-sphere xyz -> 2D map coords embedded in 3D (z=0)."""
    if pos3_unit.size == 0:
        return
    m = _proj_mode_norm(mode)
    x = pos3_unit[:, 0]
    y = pos3_unit[:, 1]
    z = np.clip(pos3_unit[:, 2], -1.0, 1.0)
    lon = np.arctan2(y, x)  # [-pi, pi]
    lat = np.arcsin(z)      # [-pi/2, pi/2]
    u = lon / np.pi
    if m == "map-equirect":
        v = (2.0 * lat) / np.pi
    elif m == "map-mercator":
        lat_cap = np.deg2rad(85.0)
        lat_c = np.clip(lat, -lat_cap, lat_cap)
        v_raw = np.log(np.tan((np.pi / 4.0) + (lat_c / 2.0)))
        v_max = float(np.log(np.tan((np.pi / 4.0) + (lat_cap / 2.0))))
        v = v_raw / max(v_max, 1e-6)
    else:  # "map-lambert" (Lambert cylindrical equal-area)
        v = np.sin(lat)

    pos3_unit[:, 0] = u.astype(np.float32, copy=False)
    pos3_unit[:, 1] = v.astype(np.float32, copy=False)
    pos3_unit[:, 2] = 0.0


def _apply_ship_projection_np(pos3_unit: np.ndarray, *, out_visible: np.ndarray | None = None) -> np.ndarray:
    """In-place: unit-sphere xyz -> interior 'ship' perspective plane (z=0).

    Returns a boolean visibility mask (same length as pos3_unit).
    """
    n = int(pos3_unit.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if out_visible is None or out_visible.shape[0] != n:
        vis = np.empty((n,), dtype=bool)
    else:
        vis = out_visible

    cam_r = 0.85
    # Camera is slightly inside the unit sphere at +Z, looking toward +Z.
    # This yields an "overhead dome" view with a clear horizon.
    cx, cy, cz = 0.0, 0.0, cam_r
    relx = pos3_unit[:, 0] - cx
    rely = pos3_unit[:, 1] - cy
    relz = pos3_unit[:, 2] - cz
    denom = relz
    np.greater(denom, 1e-3, out=vis)

    inv = np.zeros_like(denom, dtype=np.float32)
    inv[vis] = (1.0 / denom[vis]).astype(np.float32, copy=False)
    sx = (relx * inv).astype(np.float32, copy=False)
    sy = (rely * inv).astype(np.float32, copy=False)

    # Fit to view and cap near-horizon blowup.
    scale = 0.9
    clip = 4.0
    sx = np.clip(sx * scale, -clip, clip)
    sy = np.clip(sy * scale, -clip, clip)

    pos3_unit[:, 0] = 0.0
    pos3_unit[:, 1] = 0.0
    pos3_unit[:, 2] = 0.0
    pos3_unit[vis, 0] = sx[vis]
    pos3_unit[vis, 1] = sy[vis]
    return vis


def _apply_map_projection_torch(pos3_unit: "torch.Tensor", mode: str) -> "torch.Tensor":
    """Torch CPU tensor: unit-sphere xyz -> 2D map coords embedded in 3D."""
    m = _proj_mode_norm(mode)
    x = pos3_unit[:, 0]
    y = pos3_unit[:, 1]
    z = torch.clamp(pos3_unit[:, 2], -1.0, 1.0)
    lon = torch.atan2(y, x)
    lat = torch.asin(z)
    u = lon / math.pi
    if m == "map-equirect":
        v = (2.0 * lat) / math.pi
    elif m == "map-mercator":
        lat_cap = float(math.radians(85.0))
        lat_c = torch.clamp(lat, -lat_cap, lat_cap)
        v_raw = torch.log(torch.tan((math.pi / 4.0) + (lat_c / 2.0)))
        v_max = float(math.log(math.tan((math.pi / 4.0) + (lat_cap / 2.0))))
        v = v_raw / max(v_max, 1e-6)
    else:  # map-lambert
        v = torch.sin(lat)
    pos3_unit[:, 0] = u
    pos3_unit[:, 1] = v
    pos3_unit[:, 2] = 0.0
    return pos3_unit


def _apply_ship_projection_torch(pos3_unit: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch CPU tensor: unit-sphere xyz -> interior ship perspective.

    Returns (pos3_proj, visible_mask).
    """
    cam_r = 0.85
    rel = pos3_unit - torch.tensor([0.0, 0.0, cam_r], dtype=pos3_unit.dtype, device=pos3_unit.device)
    denom = rel[:, 2]
    vis = denom > 1e-3
    inv = torch.zeros_like(denom)
    inv[vis] = 1.0 / denom[vis]
    sx = rel[:, 0] * inv
    sy = rel[:, 1] * inv
    scale = 0.9
    clip = 4.0
    sx = torch.clamp(sx * scale, -clip, clip)
    sy = torch.clamp(sy * scale, -clip, clip)
    pos3_unit.zero_()
    pos3_unit[vis, 0] = sx[vis]
    pos3_unit[vis, 1] = sy[vis]
    return pos3_unit, vis


class _JoystickSideMenu:
    """Shared joystick side-menu used by both Python and C-physics paths.

    This is extracted from the original Python-path logic so the C-physics mode
    can reuse the exact same menu contents and adjustment semantics.
    """

    CONTROL_NAMES = ["k_spring", "k_coulomb", "G", "south", "damping", "temp", "bond_shear", "max_speed", "proj_mode", "n_dim"]
    PROJECTION_MODES = ["pca", "axes", "random", "hyperbolic", "map-equirect", "map-mercator", "map-lambert", "ship"]

    def __init__(self, joystick: pygame.joystick.Joystick, *, params: Dict[str, object]):
        self.joystick = joystick
        self.axis_state: Dict[int, float] = {}

        # Optional C signal kernel and controller backend evaluator.
        # This is the bridge step: gameplay can read controls from the backend even if
        # the graph executor is still Python-orchestrated.
        self._sigk = None
        self._ctrl_backend: controller_backend.ControllerBackend | None = None
        self._sigk_buttons_prev: set[int] = set()
        self._sigk_hats_prev: dict[int, tuple[int, int]] = {}

        # Keyboard axes for safe-mode navigation (DirOR).
        # Keep ids small (<= 65535) because item_id is stored in 16 bits in the kernel signal_id.
        self._KBD_AXIS_WASD_X = 0
        self._KBD_AXIS_WASD_Y = 1
        self._KBD_AXIS_ARROWS_X = 2
        self._KBD_AXIS_ARROWS_Y = 3

        # Virtual id ranges used elsewhere (workbench); keep consistent.
        self._HARD_HAT_AXIS_BASE = 50000
        self._HARD_HAT_BTN_BASE = 51000

        try:
            from c_physics import signal_kernel_api

            _lib, api = signal_kernel_api.try_load_signal_kernel()
            self._sigk = api
            if self._sigk is not None:
                try:
                    self._sigk.gp_sigk_reset()
                except Exception:
                    pass
                self._ctrl_backend = controller_backend.ControllerBackend(self._sigk)
        except Exception:
            self._sigk = None
            self._ctrl_backend = None

        self.control_names = list(self.CONTROL_NAMES)
        self.projection_modes = list(self.PROJECTION_MODES)

        self.adjust_vel = {name: 0.0 for name in self.control_names}
        self.control_amp = {name: 1.0 for name in self.control_names}
        self.control_freq = {name: 1.0 for name in self.control_names}

        self.selector_idx = 0
        self.selector_cooldown = 0.25
        self.last_selector_time = 0.0
        self.adjust_locked = False

        self.proj_mode_idx = 0
        try:
            pm = str(params.get("proj_mode", "pca"))
            if pm in self.projection_modes:
                self.proj_mode_idx = self.projection_modes.index(pm)
        except Exception:
            self.proj_mode_idx = 0
        self.last_proj_toggle_time = 0.0

        self.last_dim_axis_sign = 0
        self.last_dim_change_time = 0.0

        # Minimap (picture-in-picture) cycles through non-ship projections.
        # Start with the 2D map projections (default), then 3D basis modes.
        self.minimap_modes = list(_MAP_PROJ_MODES) + [m for m in self.PROJECTION_MODES if (m not in _MAP_PROJ_MODES and m != "ship")]
        self.minimap_idx = 0
        self.minimap_cycle_button = 2
        self._last_minimap_cycle_time = 0.0

        # Render-only flight camera state (ship view). Never affects particle physics.
        self.flight_enabled = True
        self.flight_toggle_button = 1  # button 0 is already used for "adjust lock"

        # Planet/world scale factor: scales the unit-sphere world to a desired size.
        # This affects the planet radius and camera position/velocity in world space,
        # but NOT ship geometry/arms/thrust.
        self.render_scale = _safe_render_scale(params.get("render_scale", 10.0))

        # Airplane spec (renderer-only): defines control mapping and integer offsets for
        # thrust/lift/actuation centers.
        self.airplane = _load_airplane_spec(params.get("airplane_path", "airplane.json"))

        planet_surface_r = float(_PLANET_SURFACE_R) * float(self.render_scale)
        flight_r_min = float(_FLIGHT_R_MIN) * float(self.render_scale)
        flight_r_max = float(_FLIGHT_R_MAX) * float(self.render_scale)
        self.flight_cam = PlanetFlightCamera(
            planet_surface_r=float(planet_surface_r),
            flight_r_min=float(flight_r_min),
            flight_r_max=float(flight_r_max),
        )

        # Configure C flight controller + rigid-body + arms from airplane spec.
        try:
            airplane = self.airplane if isinstance(self.airplane, dict) else {}
            rb = airplane.get("rigid_body", {}) if isinstance(airplane.get("rigid_body", {}), dict) else {}
            cs = airplane.get("control_system", {}) if isinstance(airplane.get("control_system", {}), dict) else {}
            arms = airplane.get("arms", None)

            mass = rb.get("mass", None)
            inertia = rb.get("inertia_diag", None)
            inertia_diag = None
            if isinstance(inertia, (list, tuple)) and len(inertia) == 3:
                inertia_diag = (float(inertia[0]), float(inertia[1]), float(inertia[2]))

            self.flight_cam.configure_rigid_body(
                mass=float(mass) if isinstance(mass, (int, float)) else None,
                inertia_diag=inertia_diag,
            )

            mode = cs.get("mode", None)
            auto_kp = cs.get("auto_kp", None)
            auto_kd = cs.get("auto_kd", None)
            dbg = cs.get("debug_print", None)
            self.flight_cam.configure_control_system(
                mode=str(mode) if isinstance(mode, str) else None,
                auto_kp=float(auto_kp) if isinstance(auto_kp, (int, float)) else None,
                auto_kd=float(auto_kd) if isinstance(auto_kd, (int, float)) else None,
                debug_print=bool(dbg) if isinstance(dbg, (bool, int, float)) else None,
            )

            if isinstance(arms, list):
                # Arms are ship-local geometry; do NOT scale them with world/planet size.
                self.flight_cam.configure_arms(arms=arms, scale=1.0)
                self._airplane_arms_ref = arms
            else:
                self._airplane_arms_ref = None
        except Exception:
            self._airplane_arms_ref = None

        # Atmosphere model for inertial speed envelope.
        # Bind to the visible atmosphere shell thickness, not the max flight altitude.
        try:
            self.flight_cam.atmosphere_alt_max = float(_ATMOSPHERE_ALT_MAX) * float(self.render_scale)
        except Exception:
            self.flight_cam.atmosphere_alt_max = None
        try:
            self.flight_cam._sync_config_to_sim()
            self.flight_cam._sync_state_to_sim()
        except Exception:
            pass

        # Persistent camera zoom state (optical zoom; affects camera weapon FOV).
        self.camera_zoom_mul = 1.0

        # View offsets ("head look") layered on top of ship orientation.
        # Right stick controls these; they should NOT change the craft heading.
        self.view_yaw = 0.0
        self.view_pitch = 0.0

        # Reticle look offsets (independent of view). Used by manual targeting.
        self.reticle_yaw = 0.0
        self.reticle_pitch = 0.0

        # Control mode state (driven by bindable toggles in joystick.json).
        self._active_control_set = "flight"  # flight|view_targeting|bomber|fighter
        self._targeting_active = False

        self._flaps_deflection = 0.0
        self._prev_flaps_up = False
        self._prev_flaps_down = False

    def apply_render_scale(self, scale: float) -> None:
        """Apply a new planet/world scale.

        This rescales planet radii and the ship/camera position/velocity in world
        coordinates so the view stays consistent.

        It does NOT rescale ship geometry, actuator arms, or thrust magnitudes.
        """
        s = _safe_render_scale(scale)
        prev = float(self.render_scale)
        if abs(s - prev) <= 1e-9:
            return
        ratio = float(s / max(1e-9, prev))
        self.render_scale = float(s)

        # Scale the camera position and any radius target.
        try:
            self.flight_cam.pos = (np.asarray(self.flight_cam.pos, dtype=np.float32) * ratio).astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            if hasattr(self.flight_cam, "vel"):
                self.flight_cam.vel = (np.asarray(self.flight_cam.vel, dtype=np.float32) * ratio).astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            if getattr(self.flight_cam, "radius_target", None) is not None:
                self.flight_cam.radius_target = float(self.flight_cam.radius_target) * ratio
        except Exception:
            pass

        # Update bands.
        self.flight_cam.planet_surface_r = float(_PLANET_SURFACE_R) * float(s)
        self.flight_cam.flight_r_min = float(_FLIGHT_R_MIN) * float(s)
        self.flight_cam.flight_r_max = float(_FLIGHT_R_MAX) * float(s)
        try:
            self.flight_cam.atmosphere_alt_max = float(_ATMOSPHERE_ALT_MAX) * float(s)
        except Exception:
            self.flight_cam.atmosphere_alt_max = None
        try:
            clamp_radius_band(self.flight_cam.pos, r_min=float(self.flight_cam.flight_r_min), r_max=float(self.flight_cam.flight_r_max))
        except Exception:
            pass
        try:
            # Recompute attitude after scaling.
            self.flight_cam._rebuild_attitude()
        except Exception:
            pass
        try:
            self.flight_cam._sync_config_to_sim()
            self.flight_cam._sync_state_to_sim()
        except Exception:
            pass

        # Arms are ship-local geometry; never re-scale them.
        # (If airplane arms change, we reconfigure elsewhere.)

    def flight_active(self, proj_mode: str) -> bool:
        return bool(self.flight_enabled) and _is_ship_proj_mode(proj_mode)

    def flight_camera(self, *, dt: float, proj_mode: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return (eye, center) for gluLookAt.

        In ship mode:
        - Left stick controls craft rudder + elevator (yaw/pitch).
        - Triggers control throttle forward/reverse over the surface.
        - Right stick changes view direction (yaw/pitch) without changing craft heading.
        """
        if not self.flight_active(proj_mode):
            # Ship view default camera: above ground and within the atmosphere band.
            if _is_ship_proj_mode(proj_mode):
                return self.flight_cam.default_eye_center()
            eye = (0.0, 0.0, 3.0)
            center = (0.0, 0.0, 0.0)
            return eye, center

        # Read mapping/rates from airplane spec.
        airplane = self.airplane if isinstance(self.airplane, dict) else {}
        controls = airplane.get("controls", {}) if isinstance(airplane.get("controls", {}), dict) else {}
        left = controls.get("left_stick", {}) if isinstance(controls.get("left_stick", {}), dict) else {}
        right = controls.get("right_stick", {}) if isinstance(controls.get("right_stick", {}), dict) else {}
        triggers = controls.get("triggers", {}) if isinstance(controls.get("triggers", {}), dict) else {}
        rates = airplane.get("rates", {}) if isinstance(airplane.get("rates", {}), dict) else {}
        motion = airplane.get("motion", {}) if isinstance(airplane.get("motion", {}), dict) else {}
        rb = airplane.get("rigid_body", {}) if isinstance(airplane.get("rigid_body", {}), dict) else {}
        cs = airplane.get("control_system", {}) if isinstance(airplane.get("control_system", {}), dict) else {}

        dz = float(controls.get("deadzone", 0.08))
        curve_exp = float(controls.get("curve_exp", 2.5))

        def _curve(val: float) -> float:
            v = float(val)
            if abs(v) < dz:
                return 0.0
            sign = 1.0 if v > 0.0 else -1.0
            raw = (abs(v) - dz) / max(1e-6, (1.0 - dz))
            raw = float(max(0.0, min(1.0, raw)))
            return sign * (raw ** float(max(1.0, curve_exp)))

        def _trigger_unit(v: float | None) -> float:
            if v is None:
                return 0.0
            vf = float(v)
            # Common pygame conventions:
            # - Some controllers report triggers in [-1, +1] with rest at -1.
            # - Others report in [0, 1] with rest at 0.
            if vf < -0.2:
                u = 0.5 * (vf + 1.0)
            else:
                u = vf
            return float(max(0.0, min(1.0, u)))

        # Poll full joystick snapshot so we can evaluate button-like bindings and toggles.
        axes_now: dict[int, float] = {}
        buttons_now: set[int] = set()
        hats_now: dict[int, tuple[int, int]] = {}
        try:
            axes_now, buttons_now, hats_now = joystick_menu._poll_joystick_snapshot(self.joystick)
        except Exception:
            axes_now, buttons_now, hats_now = {}, set(), {}

        # Backend funnel (step 1): push raw inputs into the C signal kernel.
        # This allows controller graphs/channels to be sourced from the backend.
        if self._sigk is not None:
            try:
                from c_physics.signal_kernel_ctypes import GP_InputEvent
                from c_physics import signal_kernel_api

                now_ns = int(time.monotonic_ns())

                # Button edges.
                b_evs: list[GP_InputEvent] = []
                try:
                    nb = int(self.joystick.get_numbuttons())
                except Exception:
                    nb = 0
                for b in range(max(0, nb)):
                    was_down = int(b) in self._sigk_buttons_prev
                    is_down = int(b) in (buttons_now or set())
                    if was_down == is_down:
                        continue
                    b_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_BUTTON),
                            id=int(b),
                            v0=1.0 if is_down else 0.0,
                            v1=0.0,
                            flags=0,
                        )
                    )
                if b_evs:
                    arrb_t = GP_InputEvent * len(b_evs)
                    self._sigk.gp_sigk_push_events(arrb_t(*b_evs), int(len(b_evs)))

                # Axes each frame.
                a_evs: list[GP_InputEvent] = []
                try:
                    na = int(self.joystick.get_numaxes())
                except Exception:
                    na = 0
                for a in range(max(0, na)):
                    vv = float((axes_now or {}).get(int(a), 0.0))
                    a_evs.append(
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
                if a_evs:
                    arra_t = GP_InputEvent * len(a_evs)
                    self._sigk.gp_sigk_push_events(arra_t(*a_evs), int(len(a_evs)))

                # Keyboard direction axes (WASD + arrows) each frame.
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

                k_evs: list[GP_InputEvent] = [
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(self._KBD_AXIS_WASD_X),
                        v0=float(wasd_x),
                        v1=0.0,
                        flags=0,
                    ),
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(self._KBD_AXIS_WASD_Y),
                        v0=float(wasd_y),
                        v1=0.0,
                        flags=0,
                    ),
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(self._KBD_AXIS_ARROWS_X),
                        v0=float(arrows_x),
                        v1=0.0,
                        flags=0,
                    ),
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                        kind=int(signal_kernel_api.GP_EV_AXIS),
                        id=int(self._KBD_AXIS_ARROWS_Y),
                        v0=float(arrows_y),
                        v1=0.0,
                        flags=0,
                    ),
                ]
                arrk_t = GP_InputEvent * len(k_evs)
                self._sigk.gp_sigk_push_events(arrk_t(*k_evs), int(len(k_evs)))

                # Mouse delta each frame (dx/dy). Use two motion ids: 0=dx, 1=dy.
                try:
                    mdx, mdy = pygame.mouse.get_rel()
                except Exception:
                    mdx, mdy = 0, 0
                m_evs: list[GP_InputEvent] = [
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_MOUSE),
                        kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
                        id=0,
                        v0=float(mdx),
                        v1=0.0,
                        flags=0,
                    ),
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_MOUSE),
                        kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
                        id=1,
                        v0=float(mdy),
                        v1=0.0,
                        flags=0,
                    ),
                ]
                arrm_t = GP_InputEvent * len(m_evs)
                self._sigk.gp_sigk_push_events(arrm_t(*m_evs), int(len(m_evs)))

                # Hat axes + direction buttons as virtual joystick ids.
                def _dir_idx_8(x: int, y: int) -> int | None:
                    if int(x) == 0 and int(y) == 0:
                        return None
                    dirs = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
                    for i, (dx, dy) in enumerate(dirs):
                        if int(dx) == int(x) and int(dy) == int(y):
                            return int(i)
                    return None

                def _hard_hat_axis_id(hat_idx: int, comp: str) -> int:
                    return int(self._HARD_HAT_AXIS_BASE + int(max(0, int(hat_idx))) * 2 + (1 if str(comp) == "y" else 0))

                def _hard_hat_btn_id(hat_idx: int, dir_idx: int) -> int:
                    return int(self._HARD_HAT_BTN_BASE + int(max(0, int(hat_idx))) * 8 + int(dir_idx))

                h_evs: list[GP_InputEvent] = []
                hb_evs: list[GP_InputEvent] = []
                try:
                    nh = int(self.joystick.get_numhats())
                except Exception:
                    nh = 0
                for h in range(max(0, nh)):
                    hx, hy = (hats_now or {}).get(int(h), (0, 0))
                    phx, phy = self._sigk_hats_prev.get(int(h), (0, 0))
                    cur_dir = _dir_idx_8(int(hx), int(hy))
                    prev_dir = _dir_idx_8(int(phx), int(phy))

                    if cur_dir != prev_dir:
                        if prev_dir is not None:
                            hb_evs.append(
                                GP_InputEvent(
                                    t_mono_ns=now_ns,
                                    device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                    kind=int(signal_kernel_api.GP_EV_BUTTON),
                                    id=int(_hard_hat_btn_id(int(h), int(prev_dir))),
                                    v0=0.0,
                                    v1=0.0,
                                    flags=0,
                                )
                            )
                        if cur_dir is not None:
                            hb_evs.append(
                                GP_InputEvent(
                                    t_mono_ns=now_ns,
                                    device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                    kind=int(signal_kernel_api.GP_EV_BUTTON),
                                    id=int(_hard_hat_btn_id(int(h), int(cur_dir))),
                                    v0=1.0,
                                    v1=0.0,
                                    flags=0,
                                )
                            )

                    h_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(_hard_hat_axis_id(int(h), "x")),
                            v0=float(int(hx)),
                            v1=0.0,
                            flags=0,
                        )
                    )
                    h_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(_hard_hat_axis_id(int(h), "y")),
                            v0=float(int(hy)),
                            v1=0.0,
                            flags=0,
                        )
                    )

                if hb_evs:
                    arrhb_t = GP_InputEvent * len(hb_evs)
                    self._sigk.gp_sigk_push_events(arrhb_t(*hb_evs), int(len(hb_evs)))
                if h_evs:
                    arrh_t = GP_InputEvent * len(h_evs)
                    self._sigk.gp_sigk_push_events(arrh_t(*h_evs), int(len(h_evs)))

                # Backend funnel (step 2): evaluate controller graph and publish channels.
                if self._ctrl_backend is not None:
                    overrides_1d, channels_2d = self._ctrl_backend.step(now_ns=now_ns)
                    if overrides_1d:
                        try:
                            self.flight_cam.controller_channels_override = overrides_1d
                        except Exception:
                            pass
                    try:
                        self.flight_cam.controller_channels_2d = channels_2d
                    except Exception:
                        pass

                if hasattr(self._sigk, "gp_sigk_clear_pulses"):
                    try:
                        self._sigk.gp_sigk_clear_pulses()
                    except Exception:
                        pass

                self._sigk_buttons_prev = set(int(b) for b in (buttons_now or set()))
                self._sigk_hats_prev = {int(k): (int(v[0]), int(v[1])) for k, v in (hats_now or {}).items()}
            except Exception:
                pass

        # Select which control-set is active.
        # Priority:
        # - If airplane role is explicitly set (bomber/fighter), use it.
        # - Else, if targeting is active, use view_targeting.
        # - Else, use flight.
        cfg = joystick_menu.load_or_create_joystick_config("joystick.json")
        role = None
        try:
            airplane = self.airplane if isinstance(self.airplane, dict) else {}
            for k in ("control_set", "controls_set", "control_profile", "craft_role", "role"):
                v = airplane.get(k)
                if isinstance(v, str) and v.strip():
                    role = v.strip().lower()
                    break
        except Exception:
            role = None

        if role in ("bomber", "fighter"):
            set_name = str(role)
        else:
            set_name = "view_targeting" if bool(getattr(self, "_targeting_active", False)) else "flight"

        self._active_control_set = str(set_name)

        # Load set bindings.
        try:
            fc = cfg.get("flight_controls") if isinstance(cfg, dict) else None
            sets = fc.get("sets") if isinstance(fc, dict) else None
            set_blk = sets.get(str(set_name)) if isinstance(sets, dict) else None
            craft_blk = set_blk.get("craft") if isinstance(set_blk, dict) else None
            look_blk = set_blk.get("look") if isinstance(set_blk, dict) else None
            ret_blk = set_blk.get("reticle_look") if isinstance(set_blk, dict) else None
            flaps_blk = set_blk.get("flaps") if isinstance(set_blk, dict) else None
            trig_blk = set_blk.get("triggers") if isinstance(set_blk, dict) else None
        except Exception:
            craft_blk = None
            look_blk = None
            ret_blk = None
            flaps_blk = None
            trig_blk = None

        # Fallback to legacy airplane.json axis indices if no new bindings exist.
        rudder_axis = int(left.get("rudder_axis", 0))
        elevator_axis = int(left.get("elevator_axis", 1))
        rudder_inv = bool(left.get("rudder_invert", True))
        elevator_inv = bool(left.get("elevator_invert", False))

        view_yaw_axis = int(right.get("view_yaw_axis", 2))
        view_pitch_axis = int(right.get("view_pitch_axis", 3))
        view_yaw_inv = bool(right.get("view_yaw_invert", True))
        view_pitch_inv = bool(right.get("view_pitch_invert", True))

        # Throttle trigger mapping can be per-set (preferred) or legacy global.
        rev_axis_raw = None
        fwd_axis_raw = None
        try:
            if isinstance(trig_blk, dict):
                rev_axis_raw = trig_blk.get("reverse_axis")
                fwd_axis_raw = trig_blk.get("forward_axis")
        except Exception:
            pass
        if rev_axis_raw is None:
            rev_axis_raw = triggers.get("reverse_axis", 4)
        if fwd_axis_raw is None:
            fwd_axis_raw = triggers.get("forward_axis", 5)

        def _axis_mapping(v: object) -> tuple[int | None, int]:
            # Returns (axis_index, sign).
            if v is None:
                return None, 1
            if isinstance(v, int):
                return int(v), 1
            if isinstance(v, dict):
                a = v.get("axis")
                s = v.get("sign", 1)
                if isinstance(a, int) and isinstance(s, int) and int(s) in (-1, 1):
                    return int(a), int(s)
            return None, 1

        rev_axis, rev_sign = _axis_mapping(rev_axis_raw)
        fwd_axis, fwd_sign = _axis_mapping(fwd_axis_raw)

        # Craft controls (ailerons/rudder/elevators).
        if isinstance(craft_blk, dict):
            rudder_raw = float(_read_axis1d(craft_blk.get("rudder"), axes_now))
            elevator_raw = float(_read_axis1d(craft_blk.get("elevators"), axes_now))
            aileron_raw = float(_read_axis1d(craft_blk.get("ailerons"), axes_now))
        else:
            rudder_raw = float(axes_now.get(rudder_axis, float(self.axis_state.get(rudder_axis, 0.0))))
            elevator_raw = float(axes_now.get(elevator_axis, float(self.axis_state.get(elevator_axis, 0.0))))
            aileron_raw = 0.0
        if rudder_inv and (not isinstance(craft_blk, dict)):
            rudder_raw = -rudder_raw
        if elevator_inv and (not isinstance(craft_blk, dict)):
            elevator_raw = -elevator_raw
        rudder = _curve(rudder_raw)
        elevator = _curve(elevator_raw)
        aileron = _curve(aileron_raw)

        # Look controls (2D) for view offsets.
        if isinstance(look_blk, dict):
            lx, ly = _read_axis2d(look_blk.get("axis2d"), axes_now)
            look_x_raw = float(lx)
            look_y_raw = float(ly)
        else:
            look_x_raw = float(axes_now.get(view_yaw_axis, float(self.axis_state.get(view_yaw_axis, 0.0))))
            look_y_raw = float(axes_now.get(view_pitch_axis, float(self.axis_state.get(view_pitch_axis, 0.0))))
            if view_yaw_inv:
                look_x_raw = -look_x_raw
            if view_pitch_inv:
                look_y_raw = -look_y_raw
        look_x = _curve(look_x_raw)
        look_y = _curve(look_y_raw)

        # Reticle look controls (2D) used by manual targeting.
        if isinstance(ret_blk, dict):
            rx, ry = _read_axis2d(ret_blk.get("axis2d"), axes_now)
            ret_x = _curve(float(rx))
            ret_y = _curve(float(ry))
        else:
            ret_x = 0.0
            ret_y = 0.0

        trig_back = _trigger_unit(float(rev_sign) * float(axes_now.get(rev_axis, float(self.axis_state.get(rev_axis, 0.0))))) if rev_axis is not None else 0.0
        trig_fwd = _trigger_unit(float(fwd_sign) * float(axes_now.get(fwd_axis, float(self.axis_state.get(fwd_axis, 0.0))))) if fwd_axis is not None else 0.0
        throttle_nudge = float(trig_fwd - trig_back)

        # Rates are split: craft movement uses `speed`; view uses view yaw/pitch rates.
        view_yaw_rate = float(rates.get("view_yaw_rate", 1.25))
        view_pitch_rate = float(rates.get("view_pitch_rate", 1.10))
        speed = float(motion.get("base_speed", 0.38))
        thrust_accel = float(motion.get("thrust_accel", 4.0 * speed))
        throttle_rate = float(motion.get("throttle_rate", 1.80))  # 1/sec to move from 0->1
        lift_k = float(motion.get("lift_k", 0.0))
        drag_k = float(motion.get("drag_k", 0.0))
        gravity_g = float(motion.get("gravity_g", 0.0))
        max_speed = float(motion.get("max_speed", 0.0))
        inertial = bool(motion.get("inertial", False))
        craft_auto_heading = bool(motion.get("auto_heading_on_move", False))
        craft_yaw_rate = float(rates.get("craft_yaw_rate", self.flight_cam.yaw_rate))
        craft_pitch_rate = float(rates.get("craft_pitch_rate", self.flight_cam.pitch_rate))

        # Flight radius band (altitude clamps) and atmosphere thickness.
        # Values are in unit-sphere sim units, scaled by render_scale at runtime.
        try:
            alt_min_u = float(motion.get("flight_alt_min", _FLIGHT_ALT_MIN))
        except Exception:
            alt_min_u = float(_FLIGHT_ALT_MIN)
        try:
            alt_max_u = float(motion.get("flight_alt_max", _FLIGHT_ALT_MAX))
        except Exception:
            alt_max_u = float(_FLIGHT_ALT_MAX)
        try:
            atmo_u = float(motion.get("atmosphere_alt_max", _ATMOSPHERE_ALT_MAX))
        except Exception:
            atmo_u = float(_ATMOSPHERE_ALT_MAX)

        if not np.isfinite(alt_min_u):
            alt_min_u = float(_FLIGHT_ALT_MIN)
        if not np.isfinite(alt_max_u):
            alt_max_u = float(_FLIGHT_ALT_MAX)
        if not np.isfinite(atmo_u):
            atmo_u = float(_ATMOSPHERE_ALT_MAX)

        alt_min_u = float(max(0.0, alt_min_u))
        alt_max_u = float(max(alt_min_u + 1e-5, alt_max_u))
        atmo_u = float(max(0.0, atmo_u))

        s = float(self.render_scale)
        r_surface = float(self.flight_cam.planet_surface_r)
        rmin = float(r_surface + alt_min_u * s)
        rmax = float(r_surface + alt_max_u * s)
        atmo_alt_max_val = (float(atmo_u * s) if atmo_u > 1e-9 else None)

        # Apply immediately so config changes take effect without restarting.
        try:
            band_changed = (
                abs(float(self.flight_cam.flight_r_min) - rmin) > 1e-6
                or abs(float(self.flight_cam.flight_r_max) - rmax) > 1e-6
            )
            atmo_changed = (
                (self.flight_cam.atmosphere_alt_max is None) != (atmo_alt_max_val is None)
                or (
                    self.flight_cam.atmosphere_alt_max is not None
                    and atmo_alt_max_val is not None
                    and abs(float(self.flight_cam.atmosphere_alt_max) - float(atmo_alt_max_val)) > 1e-6
                )
            )
            if band_changed or atmo_changed:
                self.flight_cam.flight_r_min = float(rmin)
                self.flight_cam.flight_r_max = float(max(rmin, rmax))
                self.flight_cam.atmosphere_alt_max = atmo_alt_max_val
                clamp_radius_band(self.flight_cam.pos, r_min=float(self.flight_cam.flight_r_min), r_max=float(self.flight_cam.flight_r_max))
        except Exception:
            pass

        # Integrate view offsets (look) unless in the view/targeting control-set.
        if str(self._active_control_set) == "view_targeting":
            self.view_yaw = 0.0
            self.view_pitch = 0.0
        else:
            self.view_yaw += float(view_yaw_rate) * float(dt) * float(look_x)
            self.view_pitch += float(view_pitch_rate) * float(dt) * float(look_y)
            # Keep yaw bounded to avoid precision loss after many orbits/turns.
            self.view_yaw = _wrap_angle_pi(float(self.view_yaw))
            self.view_pitch = float(max(-1.25, min(1.25, float(self.view_pitch))))

        # Auto-recenter view when the user releases the look stick.
        # This is a simple exponential spring-to-zero (stable, no oscillation).
        if (str(self._active_control_set) != "view_targeting") and abs(float(look_x)) <= 1e-9 and abs(float(look_y)) <= 1e-9:
            recenter_rate = 6.0  # higher = snaps back faster
            k = float(math.exp(-float(recenter_rate) * float(dt)))
            self.view_yaw *= k
            self.view_pitch *= k
            if abs(float(self.view_yaw)) < 1e-4:
                self.view_yaw = 0.0
            if abs(float(self.view_pitch)) < 1e-4:
                self.view_pitch = 0.0

        # Reticle look offsets: always independent and spring back when released.
        ret_yaw_rate = float(rates.get("view_yaw_rate", 1.25))
        ret_pitch_rate = float(rates.get("view_pitch_rate", 1.10))
        self.reticle_yaw += float(ret_yaw_rate) * float(dt) * float(ret_x)
        self.reticle_pitch += float(ret_pitch_rate) * float(dt) * float(ret_y)
        self.reticle_yaw = _wrap_angle_pi(float(self.reticle_yaw))
        self.reticle_pitch = float(max(-1.25, min(1.25, float(self.reticle_pitch))))
        if abs(float(ret_x)) <= 1e-9 and abs(float(ret_y)) <= 1e-9:
            recenter_rate = 6.0
            k = float(math.exp(-float(recenter_rate) * float(dt)))
            self.reticle_yaw *= k
            self.reticle_pitch *= k
            if abs(float(self.reticle_yaw)) < 1e-4:
                self.reticle_yaw = 0.0
            if abs(float(self.reticle_pitch)) < 1e-4:
                self.reticle_pitch = 0.0

        # Delegate the craft motion integration to the extracted camera module.
        # Craft heading/pitch comes from left stick; right stick is view-only.
        self.flight_cam.speed = float(speed)
        self.flight_cam.climb_speed = float(speed)
        self.flight_cam.deadzone = float(dz)
        # In flight-sim mode, disable auto-heading so throttle doesn't snap the craft heading.
        self.flight_cam.auto_heading_on_move = bool(craft_auto_heading)
        self.flight_cam.yaw_rate = float(craft_yaw_rate)
        self.flight_cam.pitch_rate = float(craft_pitch_rate)
        self.flight_cam.inertial = bool(inertial)

        # Control-system mode/gains from airplane spec.
        try:
            mode = cs.get("mode", motion.get("control_mode", self.flight_cam.control_mode))
            if isinstance(mode, str):
                self.flight_cam.control_mode = str(mode)
            akp = cs.get("auto_kp", None)
            akd = cs.get("auto_kd", None)
            dbg = cs.get("debug_print", None)
            if isinstance(akp, (int, float)):
                self.flight_cam.auto_kp = float(akp)
            if isinstance(akd, (int, float)):
                self.flight_cam.auto_kd = float(akd)
            if isinstance(dbg, (bool, int, float)):
                self.flight_cam.debug_print = bool(dbg)
        except Exception:
            pass

        # Rigid-body params for the C sim.
        try:
            mass = rb.get("mass", None)
            inertia = rb.get("inertia_diag", None)
            mass_v: float | None = float(mass) if isinstance(mass, (int, float)) else None
            inertia_diag: tuple[float, float, float] | None = None
            if isinstance(inertia, (list, tuple)) and len(inertia) == 3:
                inertia_diag = (float(inertia[0]), float(inertia[1]), float(inertia[2]))

            cur_mass = float(getattr(self.flight_cam, "mass", 1.0))
            cur_inertia = tuple(getattr(self.flight_cam, "inertia_diag", (1.0, 1.0, 1.0)))
            need = False
            if mass_v is not None and abs(float(cur_mass) - float(mass_v)) > 1e-9:
                need = True
            if inertia_diag is not None and tuple(cur_inertia) != tuple(inertia_diag):
                need = True
            if need:
                self.flight_cam.configure_rigid_body(mass=mass_v, inertia_diag=inertia_diag)
        except Exception:
            pass

        # Arms: configure once (or after airplane reload/scale changes).
        try:
            arms = airplane.get("arms", None)
            if isinstance(arms, list) and arms is not getattr(self, "_airplane_arms_ref", None):
                # Arms are ship-local geometry; do NOT scale them with planet/world size.
                self.flight_cam.configure_arms(arms=arms, scale=1.0)
                self._airplane_arms_ref = arms
        except Exception:
            pass
        self.flight_cam.thrust_accel = float(thrust_accel)
        self.flight_cam.lift_k = float(lift_k)
        self.flight_cam.drag_k = float(drag_k)
        self.flight_cam.gravity_g = float(gravity_g)
        self.flight_cam.max_speed = float(max_speed)

        # Persistent engine throttle (integrator): forward/reverse triggers nudge it up/down.
        try:
            thr = float(getattr(self.flight_cam, "engine_throttle", 0.0))
        except Exception:
            thr = 0.0
        thr += float(throttle_rate) * float(dt) * float(throttle_nudge)
        thr = float(max(-1.0, min(1.0, thr)))
        try:
            self.flight_cam.engine_throttle = float(thr)
        except Exception:
            pass

        # Flaps (discrete up/down bindings) -> persistent deflection in [-1, 1].
        try:
            flaps_up_binding = flaps_blk.get("up") if isinstance(flaps_blk, dict) else None
            flaps_down_binding = flaps_blk.get("down") if isinstance(flaps_blk, dict) else None
            a_up, _ = _binding_active(flaps_up_binding if isinstance(flaps_up_binding, dict) else None, axes_now, buttons_now, hats_now)
            a_dn, _ = _binding_active(flaps_down_binding if isinstance(flaps_down_binding, dict) else None, axes_now, buttons_now, hats_now)

            if bool(a_up) and (not bool(self._prev_flaps_up)):
                self._flaps_deflection = float(max(-1.0, min(1.0, float(self._flaps_deflection) - 0.2)))
            if bool(a_dn) and (not bool(self._prev_flaps_down)):
                self._flaps_deflection = float(max(-1.0, min(1.0, float(self._flaps_deflection) + 0.2)))

            self._prev_flaps_up = bool(a_up)
            self._prev_flaps_down = bool(a_dn)
        except Exception:
            pass

        try:
            self.flight_cam.flaps = float(self._flaps_deflection)
        except Exception:
            pass

        # Controller graph override: now sourced from the backend funnel (signal kernel + compiled final graph).

        eye, _ship_center = self.flight_cam.step(
            dt=float(dt),
            # No strafing in this mapping.
            move_x=0.0,
            move_y=float(thr),
            look_x=float(rudder),
            look_y=float(elevator),
            roll_in=float(aileron),
            climb=0.0,
        )

        # Compute view center from ship orientation plus view offsets.
        # Yaw about local radial up, then pitch about local horizon-right.
        eye_v = np.asarray(self.flight_cam.pos, dtype=np.float32)
        right_s, up_s, fwd_s = self.flight_cam.basis()
        up_rad = self.flight_cam.planet_up()

        # Rodrigues rotation in-place (local helper, keeps this file self-contained).
        def _rot(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
            a = axis.astype(np.float32, copy=False)
            an = float(np.linalg.norm(a))
            if not (an > 1e-6):
                return v
            a = a / an
            c = float(math.cos(float(ang)))
            s = float(math.sin(float(ang)))
            return (v * c + np.cross(a, v) * s + a * float(np.dot(a, v)) * (1.0 - c)).astype(np.float32, copy=False)

        fwd_view = _rot(fwd_s.astype(np.float32, copy=False), up_rad, float(self.view_yaw))
        r_axis = np.cross(up_rad, fwd_view).astype(np.float32, copy=False)
        rn = float(np.linalg.norm(r_axis))
        if rn > 1e-6:
            r_axis = r_axis / rn
        else:
            r_axis = right_s.astype(np.float32, copy=False)
        fwd_view = _rot(fwd_view, r_axis, float(self.view_pitch))
        fn = float(np.linalg.norm(fwd_view))
        if fn > 1e-6:
            fwd_view = fwd_view / fn

        center_v = eye_v + fwd_view
        center = (float(center_v[0]), float(center_v[1]), float(center_v[2]))
        return eye, center

    @classmethod
    def create(cls, *, params: Dict[str, object]) -> "_JoystickSideMenu | None":
        pygame.joystick.init()
        js = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        if not js:
            return None
        js.init()
        print(f"Joystick detected: {js.get_name()}")
        return cls(js, params=params)

    @property
    def selected_name(self) -> str:
        return self.control_names[self.selector_idx % max(1, len(self.control_names))]

    def handle_event(self, event) -> None:
        if event.type == pygame.JOYAXISMOTION:
            self.axis_state[event.axis] = float(event.value)
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == int(self.flight_toggle_button):
                self.flight_enabled = not self.flight_enabled
                if self.flight_enabled:
                    self.flight_cam.reset_north_pole()
            if event.button == int(self.minimap_cycle_button):
                now_t = pygame.time.get_ticks() * 0.001
                if now_t - float(self._last_minimap_cycle_time) > float(self.selector_cooldown):
                    self.minimap_idx = (self.minimap_idx + 1) % max(1, len(self.minimap_modes))
                    self._last_minimap_cycle_time = now_t
            if event.button == 0:
                self.adjust_locked = True
                for name in self.control_names:
                    self.adjust_vel[name] = 0.0
                self.axis_state[0] = 0.0
                self.axis_state[1] = self.axis_state.get(1, 0.0)
        elif event.type == pygame.JOYBUTTONUP:
            if event.button == 0:
                self.adjust_locked = False

    def poll_axes(self) -> None:
        # Refresh all axes to avoid relying solely on events.
        try:
            n_axes = int(self.joystick.get_numaxes())
        except Exception:
            n_axes = 0
        for a in range(max(0, int(n_axes))):
            try:
                self.axis_state[int(a)] = float(self.joystick.get_axis(int(a)))
            except Exception:
                pass

    def minimap_mode(self) -> str:
        if not self.minimap_modes:
            return "pca"
        return str(self.minimap_modes[self.minimap_idx % len(self.minimap_modes)])

    def update_params(
        self,
        params: Dict[str, object],
        *,
        now_t: float,
        accel_gain: float = 2.0,
        dt_controls: float = 1.0 / 60.0,
        dead_zone: float = 0.6,
        allow_proj_mode: bool = True,
        allow_n_dim: bool = True,
        on_proj_mode_changed=None,
        on_dim_change_requested=None,
    ) -> bool:
        """Apply the side-menu update semantics from the Python path.

        Returns True if params were mutated.
        """
        changed = False

        nav_axis = float(self.axis_state.get(1, 0.0))
        if abs(nav_axis) > dead_zone and now_t - self.last_selector_time > self.selector_cooldown:
            step = 1 if nav_axis > 0 else -1
            self.selector_idx = (self.selector_idx + step) % len(self.control_names)
            self.last_selector_time = now_t

        sel_name = self.selected_name
        adj_axis = float(self.axis_state.get(0, 0.0))
        if abs(adj_axis) < dead_zone:
            adj_axis = 0.0

        if self.adjust_locked:
            return False

        if sel_name == "proj_mode":
            if allow_proj_mode and abs(adj_axis) > dead_zone and now_t - self.last_proj_toggle_time > self.selector_cooldown:
                self.proj_mode_idx = (self.proj_mode_idx + (1 if adj_axis > 0 else -1)) % len(self.projection_modes)
                params["proj_mode"] = self.projection_modes[self.proj_mode_idx]
                self.last_proj_toggle_time = now_t
                changed = True
                if on_proj_mode_changed is not None:
                    on_proj_mode_changed(params["proj_mode"])
            return changed

        if sel_name == "n_dim":
            if allow_n_dim and abs(adj_axis) > dead_zone:
                sign = 1 if adj_axis > 0 else -1
                if (sign != self.last_dim_axis_sign) or (now_t - self.last_dim_change_time > self.selector_cooldown):
                    if on_dim_change_requested is not None:
                        on_dim_change_requested(sign)
                    self.last_dim_change_time = now_t
                    self.last_dim_axis_sign = sign
            else:
                self.last_dim_axis_sign = 0
            return False

        eff_dt = dt_controls * float(self.control_freq.get(sel_name, 1.0))
        accel = accel_gain * adj_axis
        self.adjust_vel[sel_name] += accel * eff_dt
        vel_damp = math.exp(-1.5 * dt_controls)
        for name in self.control_names:
            self.adjust_vel[name] *= vel_damp

        for name, vel_val in self.adjust_vel.items():
            if abs(vel_val) < 1e-5:
                continue
            dv = float(vel_val) * dt_controls
            if name in ("k_spring", "k_coulomb", "G"):
                params[name] = max(1e-5, float(params[name]) * math.exp(dv))
                changed = True
            elif name == "south":
                new_strength = max(0.0, float(params.get("south_strength", 0.0)) + dv * 0.5)
                params["south_strength"] = new_strength
                params["south_enabled"] = new_strength > 1e-5
                changed = True
            elif name == "damping":
                params[name] = min(0.5, max(0.0, float(params[name]) + dv * 0.2))
                changed = True
            elif name == "temp":
                params[name] = float(params[name]) + dv * 1.2
                changed = True
            elif name == "bond_shear":
                params[name] = max(0.5, min(4.0, float(params[name]) + dv * 0.6))
                changed = True
            elif name == "max_speed":
                params[name] = max(1e-4, min(0.5, float(params[name]) * math.exp(dv)))
                changed = True
        return changed

    def side_menu_lines(self, params: Dict[str, object], *, live_dim: int, current_n_dim: int) -> List[str]:
        """Return the exact side-menu lines used in the Python path."""
        sel_name = self.selected_name
        ks_live = float(params.get("k_spring", 0.0))
        kc_live = float(params.get("k_coulomb", 0.0))
        gg_live = float(params.get("G", 0.0))
        south_live = float(params.get("south_strength", 0.0))
        south_on = bool(params.get("south_enabled", False))
        south_axis_live = int(params.get("south_axis", current_n_dim - 1))
        dp_live = float(params.get("damping", 0.0))
        tp_live = float(params.get("temp", 0.0))

        return [
            ("> " if sel_name == "k_spring" else "  ") + f"k_spring: {ks_live:.3f}",
            ("> " if sel_name == "k_coulomb" else "  ") + f"k_charge: {kc_live:.3f}",
            ("> " if sel_name == "G" else "  ") + f"G:        {gg_live:.3f}",
            ("> " if sel_name == "south" else "  ") + f"south:    {'on' if south_on else 'off'} ({south_live:.3f} @ axis {int(south_axis_live)})",
            ("> " if sel_name == "damping" else "  ") + f"damping:  {dp_live:.4f}",
            ("> " if sel_name == "temp" else "  ") + f"temp:     {tp_live:.3f}",
            ("> " if sel_name == "bond_shear" else "  ") + f"bond_shear: {float(params.get('bond_shear', 0.0)):.2f}",
            ("> " if sel_name == "max_speed" else "  ") + f"max_speed: {float(params.get('max_speed', 0.0)):.4f}",
            ("> " if sel_name == "proj_mode" else "  ") + f"proj:     {params.get('proj_mode', 'pca')}",
            ("> " if sel_name == "n_dim" else "  ") + f"n_dim:    {int(live_dim)} / {int(current_n_dim)}",
        ]


def _run_c_physics_only(
    *,
    pos_np: np.ndarray,
    vel_np: np.ndarray,
    springs_np: np.ndarray,
    masses_np: np.ndarray,
    charges_np: np.ndarray,
    colors_np: np.ndarray,
    radius_scale: float,
    collide_gain: float,
    merge_speed_frac: float,
    merge_size_frac: float,
    shatter_speed_frac: float,
    shatter_size_frac: float,
    shatter_k_max: int,
    collision_pair_limit: int,
    accel_shatter_thresh: float,
    accel_shatter_fraction: float,
    accel_shatter_k: int,
    accel_shatter_kick: float,
    mass_vapor_thresh: float,
    condense_temp_thresh: float,
    condense_chunk_mass: float,
    dt: float,
    dt_mode: str,
    min_dt: float,
    max_speed: float,
    c_substep_dt_max: float | None,
    c_substeps_max: int,
    k_spring: float,
    k_coulomb: float,
    G: float,
    lorentz_k: float,
    B_vec: Tuple[float, float, float],
    temp: float,
    damping: float,
    softening: float,
    bond_max_per_node: int,
    bond_link_angle: float,
    bond_shear_ratio: float,
    bond_k: float,
    ionic_bonds: bool,
    ionic_valence: int,
    south_strength: float,
    south_axis: int,
    south_enabled: bool,
    draw_edges: bool,
    max_edges: int | None = None,
    phys_fps_limit: float,
    ghost_history: int,
    ghost_hue_cycles: float,
    draw_direct_edges: bool,
    c_n_cap: int | None = None,
    proj_mode: str = "ship",
    freeze_pca: bool = True,
    fov_deg: float = 10.0,
    z_near: float = 0.1,
    z_far: float = 1000.0,
    render_scale: float = 10.0,
    airplane_path: str = "airplane.json",
    scene_path: str = "scene.json",
):
    """Minimal C-backed physics path.

    Goals:
    - Physics runs in the C DLL over a contiguous double-buffer state block.
    - Rendering uploads directly from the ctypes buffer to OpenGL (no torch).

    Notes:
    - Physics runs in D dimensions; rendering projects to 3D in the render loop.
    - Projection is intentionally NOT part of the physics hot loop.
        - Lifecycle consequences (collisions/shatter/vapor/condense) run in the C tick.
            The C implementation uses fixed-size scratch vectors and currently supports
            dimensions up to a safety cap.
    - Disables ghost history (existing implementation uses torch).
    """

    if pos_np.ndim != 2 or pos_np.shape[1] < 2:
        raise SystemExit("--c-physics requires pos as (n, d) with d>=2")

    # Lazy import so normal mode doesn't require the C DLL.
    from c_physics.geodesic_ctypes import GeodesicPhysicsState

    target_sys = targeting_system.TargetingSystem(folder="assets/reticle", size_px=64)
    reticle_id = "center"
    target_focus = targeting_system.ReticleFocus(on_target=False, victim_id=0)
    target_view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    n = int(pos_np.shape[0])
    d = int(pos_np.shape[1])

    # Match Python-mode rendering: colors from charge and per-node point sizes
    # from cube-root mass scaling.
    masses_in = np.asarray(masses_np, dtype=np.float32)
    mass_ref = float(np.median(masses_in)) if masses_in.size and np.isfinite(masses_in).any() else 1.0
    mass_ref_cuberoot = float(np.cbrt(max(mass_ref, 1e-6)))
    radius_scale_local = float(radius_scale)
    n_cap = int(c_n_cap) if (c_n_cap is not None) else n
    if n_cap < n:
        n_cap = n
    spring_cap_in = int(springs_np.shape[0]) if springs_np is not None else 0

    # Dynamic bonding needs spare spring capacity beyond any initial edges.
    # If the input has no edges, spring_cap_in would be 0 which would prevent
    # any bond formation in C. Choose a conservative cap based on per-node limits.
    per_node_cap = 0
    if ionic_bonds:
        per_node_cap = max(1, int(ionic_valence))
    else:
        per_node_cap = max(0, int(bond_max_per_node))
    pair_cap = int((n * (n - 1)) // 2)
    suggested = int((n * per_node_cap) // 2) if per_node_cap > 0 else 0
    spring_cap_default = max(spring_cap_in, min(pair_cap, suggested))

    # Optional cap for C spring storage (affects both initial edges and dynamic bonds).
    # If provided, this overrides the default heuristic (e.g. 128 nodes * 3 bonds/node / 2 = 192).
    if max_edges is not None:
        cap_user = int(max_edges)
        if cap_user < 0:
            cap_user = 0
        spring_cap = min(pair_cap, cap_user)
    else:
        spring_cap = spring_cap_default

    # If the cap is below the initial edge count, truncate the initial edge list.
    if springs_np is not None and spring_cap_in > spring_cap:
        springs_np = springs_np[:spring_cap]
        spring_cap_in = int(springs_np.shape[0])

    state = GeodesicPhysicsState(n, d, spring_cap=spring_cap, n_cap=n_cap)
    state.set_params(k_spring=float(k_spring), k_coulomb=float(k_coulomb), G=float(G), damping=float(damping), softening=float(softening))

    # South-pole field (now supported in the C DLL).
    state.set_south(enabled=bool(south_enabled), strength=float(south_strength), axis=int(south_axis))

    # Lorentz + temperature live in the C DLL now.
    bx, by, bz = (float(B_vec[0]), float(B_vec[1]), float(B_vec[2]))
    state.set_lorentz(lorentz_k=float(lorentz_k), Bx=bx, By=by, Bz=bz)

    # Temperature model:
    # - Treat `temp` as initial node temperature AND ambient "atmosphere" temperature.
    # - Use |temp| as a simple knob for thermal agitation (noise).
    # - Convection + damping-heating happen in C; these are conservative defaults.
    tp = float(temp)
    state.set_temp(ambient=tp, convection=0.5, noise=abs(tp), heat_gain=1.0)
    state.seed(int(time.time()) & 0xFFFFFFFF)

    # Lifecycle consequences in C use fixed-size temporary vectors (MSVC-safe).
    # Keep Python and C behavior consistent by disabling lifecycle above this cap.
    C_LIFECYCLE_MAX_D = 128
    lifecycle_ok = (d <= C_LIFECYCLE_MAX_D)
    if not lifecycle_ok and (
        float(collide_gain) != 0.0
        or float(mass_vapor_thresh) != 0.0
        or float(condense_chunk_mass) != 0.0
        or float(accel_shatter_thresh) != 0.0
    ):
        print(f"[warn] C lifecycle consequences support d<= {C_LIFECYCLE_MAX_D}; auto-disabling for d={d}", file=sys.stderr, flush=True)

    state.set_lifecycle(
        max_speed=float(max_speed),
        radius_scale=float(radius_scale_local),
        mass_ref_cuberoot=float(mass_ref_cuberoot),
        collide_gain=float(collide_gain if lifecycle_ok else 0.0),
        merge_speed_frac=float(merge_speed_frac),
        merge_size_frac=float(merge_size_frac),
        shatter_speed_frac=float(shatter_speed_frac),
        shatter_size_frac=float(shatter_size_frac),
        shatter_k_max=int(shatter_k_max),
        collision_pair_limit=int(collision_pair_limit),
        mass_vapor_thresh=float(mass_vapor_thresh if lifecycle_ok else 0.0),
        condense_temp_thresh=float(condense_temp_thresh if lifecycle_ok else 0.0),
        condense_chunk_mass=float(condense_chunk_mass if lifecycle_ok else 0.0),
        accel_shatter_thresh=float(accel_shatter_thresh if lifecycle_ok else 0.0),
        accel_shatter_fraction=float(accel_shatter_fraction),
        accel_shatter_k=int(accel_shatter_k),
        accel_shatter_kick=float(accel_shatter_kick),
    )

    # Dynamic bond forming/breaking is now supported in C.
    bonds_enabled = (float(bond_link_angle) > 0.0) and (int(bond_max_per_node) > 0 or bool(ionic_bonds))
    state.set_bonds(
        enabled=bool(bonds_enabled),
        max_per_node=int(bond_max_per_node),
        link_angle=float(bond_link_angle),
        shear_ratio=float(bond_shear_ratio),
        k=float(bond_k),
        ionic=bool(ionic_bonds),
        ionic_valence=int(ionic_valence),
    )
    if spring_cap:
        state.set_springs(springs_np)

    # Initialize both buffers; the writer advances the back buffer (front^1).
    # State views are sized to capacity, but only the first n_active nodes are simulated.
    p0 = np.asarray(pos_np, dtype=np.float64)
    v0 = np.asarray(vel_np, dtype=np.float64)
    if state.n_cap == n:
        state.pos[0][:] = p0
        state.vel[0][:] = v0
        state.pos[1][:] = p0
        state.vel[1][:] = v0
        state.mass[:] = np.asarray(masses_np, dtype=np.float64)
        state.charge[:] = np.asarray(charges_np, dtype=np.float64)
        state.temps[:] = float(tp)
    else:
        # Fill inactive slots with wraparound copies (so future activation has sane defaults).
        idx = (np.arange(state.n_cap, dtype=np.int64) % n)
        state.pos[0][:] = p0[idx]
        state.vel[0][:] = v0[idx]
        state.pos[1][:] = p0[idx]
        state.vel[1][:] = v0[idx]
        state.mass[:] = np.asarray(masses_np, dtype=np.float64)[idx]
        state.charge[:] = np.asarray(charges_np, dtype=np.float64)[idx]
        state.temps[:] = float(tp)

    pygame.init()
    width, height = 1280, 720
    # Ensure we actually get a depth buffer; otherwise the planet can't occlude far-side objects.
    try:
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    except Exception:
        pass
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    base.init_gl(width, height, float(fov_deg), float(z_near), float(z_far))
    try:
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
    except Exception:
        pass
    try:
        from OpenGL.GL import glGetIntegerv, GL_DEPTH_BITS

        print(f"GL depth bits: {int(glGetIntegerv(GL_DEPTH_BITS))}")
    except Exception:
        pass
    font = pygame.font.SysFont("consolas", 18)
    try:
        pip_font = pygame.font.SysFont("consolas", 14)
    except Exception:
        pip_font = font
    clock = pygame.time.Clock()

    # --- Minimal shader renderer (sticks + balls) ---
    # Points: shaded discs ("balls") using gl_PointCoord.
    # Lines: simple colored lines ("sticks").
    ball_vs = """
    #version 120
    attribute vec3 a_pos;
    attribute vec3 a_color;
    attribute float a_size;
    varying vec3 v_color;
    varying float v_depth;
    void main() {
        v_color = a_color;
        vec4 eyePos = gl_ModelViewMatrix * vec4(a_pos, 1.0);
        v_depth = max(0.0, -eyePos.z);
        gl_Position = gl_ProjectionMatrix * eyePos;
        // Perspective-correct point size so balls sit visually in 3D with sticks.
        float w = max(0.001, gl_Position.w);
        gl_PointSize = clamp(a_size / w, 1.0, 512.0);
    }
    """
    ball_fs = """
    #version 120
    varying vec3 v_color;
    varying float v_depth;
    uniform vec3 u_light_dir;
    uniform vec3 u_light_color;
    uniform vec3 u_fog_color;
    uniform float u_fog_k;
    void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        float z = sqrt(max(0.0, 1.0 - r2));
        vec3 n = normalize(vec3(p.x, p.y, z));
        vec3 l = normalize(u_light_dir);
        float diff = max(0.0, dot(n, l));
        float rim = pow(1.0 - z, 2.0);
        vec3 lit = v_color * (0.35 + 0.65 * diff) * u_light_color + vec3(0.08) * rim;
        float edge = smoothstep(0.85, 1.0, sqrt(r2));
        float alpha = 1.0 - edge;
        float fog = 1.0 - exp(-max(0.0, u_fog_k) * v_depth);
        vec3 col = mix(lit, u_fog_color, clamp(fog, 0.0, 1.0));
        gl_FragColor = vec4(col, alpha);
    }
    """
    line_vs = """
    #version 120
    attribute vec3 a_pos;
    varying float v_depth;
    void main() {
        vec4 eyePos = gl_ModelViewMatrix * vec4(a_pos, 1.0);
        v_depth = max(0.0, -eyePos.z);
        gl_Position = gl_ProjectionMatrix * eyePos;
    }
    """
    line_fs = """
    #version 120
    uniform vec4 u_color;
    varying float v_depth;
    uniform vec3 u_fog_color;
    uniform float u_fog_k;
    void main() {
        float fog = 1.0 - exp(-max(0.0, u_fog_k) * v_depth);
        vec3 rgb = mix(u_color.rgb, u_fog_color, clamp(fog, 0.0, 1.0));
        gl_FragColor = vec4(rgb, u_color.a);
    }
    """

    # --- Sun sprite (single point sprite, constant pixel size) ---
    sun_vs = """
    #version 120
    attribute vec3 a_pos;
    uniform float u_size_px;
    void main() {
        gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);
        gl_PointSize = u_size_px;
    }
    """
    sun_fs = """
    #version 120
    uniform vec3 u_color;
    void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        float r = sqrt(r2);
        // Bright core + soft glow falloff
        float core = smoothstep(0.60, 0.0, r);
        float glow = smoothstep(1.00, 0.0, r);
        float a = clamp(0.35 * glow + 0.85 * core, 0.0, 1.0);
        vec3 col = u_color * (0.55 + 0.75 * core);
        gl_FragColor = vec4(col, a);
    }
    """

    # --- Atmosphere shell (limb haze) ---
    atmo_vs = """
    #version 120
    attribute vec3 a_pos;
    uniform float u_radius;
    varying float v_edge;
    void main() {
        vec3 world = a_pos * u_radius;
        vec4 eyePos = gl_ModelViewMatrix * vec4(world, 1.0);
        vec3 V = normalize(-eyePos.xyz);
        // Normal in eye space (unit sphere positions act like normals).
        vec3 N = normalize((gl_ModelViewMatrix * vec4(a_pos, 0.0)).xyz);
        float ndv = clamp(dot(N, V), 0.0, 1.0);
        v_edge = 1.0 - ndv;
        gl_Position = gl_ProjectionMatrix * eyePos;
    }
    """
    atmo_fs = """
    #version 120
    varying float v_edge;
    uniform vec3 u_color;
    uniform float u_alpha;
    uniform float u_power;
    void main() {
        float a = u_alpha * pow(clamp(v_edge, 0.0, 1.0), max(0.25, u_power));
        gl_FragColor = vec4(u_color, clamp(a, 0.0, 1.0));
    }
    """

    # --- Minimap shader (fast, 2D retro signatures) ---
    # Draws circles for objects and a heading-rotated triangle for "self".
    mm_vs = """
    #version 120
    attribute vec3 a_pos;
    attribute vec3 a_color;
    attribute float a_type;
    attribute float a_ang;
    varying vec3 v_color;
    varying float v_type;
    varying float v_ang;
    uniform float u_point_px;
    uniform float u_self_px;
    void main() {
        v_color = a_color;
        v_type = a_type;
        v_ang = a_ang;
        gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);
        float s = (a_type > 0.5) ? u_self_px : u_point_px;
        gl_PointSize = s;
    }
    """
    mm_fs = """
    #version 120
    varying vec3 v_color;
    varying float v_type;
    varying float v_ang;
    float edge(vec2 a, vec2 b, vec2 p) {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    }
    void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        // Rotate by heading for the self-triangle.
        float c = cos(v_ang);
        float s = sin(v_ang);
        vec2 pr = vec2(c * p.x - s * p.y, s * p.x + c * p.y);

        if (v_type > 0.5) {
            // Upright triangle in point-sprite space.
            vec2 a = vec2(0.0, 1.0);
            vec2 b = vec2(-0.9, -0.8);
            vec2 c2 = vec2(0.9, -0.8);
            float e0 = edge(a, b, pr);
            float e1 = edge(b, c2, pr);
            float e2 = edge(c2, a, pr);
            if (!((e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0))) discard;
            gl_FragColor = vec4(v_color, 1.0);
        } else {
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float alpha = smoothstep(1.0, 0.85, sqrt(r2));
            gl_FragColor = vec4(v_color, alpha);
        }
    }
    """

    ball_prog = _gl_compile_program(ball_vs, ball_fs)
    line_prog = _gl_compile_program(line_vs, line_fs)
    mm_prog = _gl_compile_program(mm_vs, mm_fs)
    sun_prog = _gl_compile_program(sun_vs, sun_fs)
    atmo_prog = _gl_compile_program(atmo_vs, atmo_fs)

    # Some drivers require this for gl_PointSize from the vertex shader.
    try:
        glEnable(GL_PROGRAM_POINT_SIZE)
    except Exception:
        pass

    # Some drivers require GL_POINT_SPRITE for gl_PointCoord to work.
    try:
        glEnable(GL_POINT_SPRITE)
    except Exception:
        pass

    ball_u_mvp = glGetUniformLocation(ball_prog, "u_mvp")
    line_u_mvp = glGetUniformLocation(line_prog, "u_mvp")
    line_u_color = glGetUniformLocation(line_prog, "u_color")

    ball_u_light_dir = glGetUniformLocation(ball_prog, "u_light_dir")
    ball_u_light_color = glGetUniformLocation(ball_prog, "u_light_color")
    ball_u_fog_color = glGetUniformLocation(ball_prog, "u_fog_color")
    ball_u_fog_k = glGetUniformLocation(ball_prog, "u_fog_k")

    line_u_fog_color = glGetUniformLocation(line_prog, "u_fog_color")
    line_u_fog_k = glGetUniformLocation(line_prog, "u_fog_k")

    sun_u_color = glGetUniformLocation(sun_prog, "u_color")
    sun_u_size_px = glGetUniformLocation(sun_prog, "u_size_px")

    atmo_u_radius = glGetUniformLocation(atmo_prog, "u_radius")
    atmo_u_color = glGetUniformLocation(atmo_prog, "u_color")
    atmo_u_alpha = glGetUniformLocation(atmo_prog, "u_alpha")
    atmo_u_power = glGetUniformLocation(atmo_prog, "u_power")

    mm_u_point_px = glGetUniformLocation(mm_prog, "u_point_px")
    mm_u_self_px = glGetUniformLocation(mm_prog, "u_self_px")

    ball_a_pos = glGetAttribLocation(ball_prog, "a_pos")
    ball_a_color = glGetAttribLocation(ball_prog, "a_color")
    ball_a_size = glGetAttribLocation(ball_prog, "a_size")
    line_a_pos = glGetAttribLocation(line_prog, "a_pos")

    sun_a_pos = glGetAttribLocation(sun_prog, "a_pos")
    atmo_a_pos = glGetAttribLocation(atmo_prog, "a_pos")

    mm_a_pos = glGetAttribLocation(mm_prog, "a_pos")
    mm_a_color = glGetAttribLocation(mm_prog, "a_color")
    mm_a_type = glGetAttribLocation(mm_prog, "a_type")
    mm_a_ang = glGetAttribLocation(mm_prog, "a_ang")

    vbos = {
        "pt_pos": glGenBuffers(1),
        "pt_col": glGenBuffers(1),
        "pt_size": glGenBuffers(1),
        "ln_pos": glGenBuffers(1),
        "sun_pos": glGenBuffers(1),
        "mm_pos": glGenBuffers(1),
        "mm_col": glGenBuffers(1),
        "mm_type": glGenBuffers(1),
        "mm_ang": glGenBuffers(1),
    }

    pos_f32 = np.empty((int(state.n_cap), 3), dtype=np.float32)
    pos_unit_f32 = np.empty((int(state.n_cap), 3), dtype=np.float32)
    pos_mm_f32 = np.empty((int(state.n_cap), 3), dtype=np.float32)
    pos_mm_draw_f32 = np.empty((int(state.n_cap) + 1, 3), dtype=np.float32)
    # Work buffer for projection (cast + center) to avoid per-frame allocations.
    pos_work_f32 = np.empty((int(state.n_cap), int(d)), dtype=np.float32)
    pos_work_mm_f32 = np.empty((int(state.n_cap), int(d)), dtype=np.float32)
    col_f32 = np.empty((int(state.n_cap), 3), dtype=np.float32)
    col_mm_draw_f32 = np.empty((int(state.n_cap) + 1, 3), dtype=np.float32)
    type_mm_draw_f32 = np.zeros((int(state.n_cap) + 1,), dtype=np.float32)
    ang_mm_draw_f32 = np.zeros((int(state.n_cap) + 1,), dtype=np.float32)
    size_f32 = np.empty((int(state.n_cap),), dtype=np.float32)
    edge_verts = np.empty((max(2, 2 * int(state.header.spring_cap)), 3), dtype=np.float32)

    # --- Shared joystick side-menu (exact same contents as Python path) ---
    params_lock = threading.Lock()
    params_seq = 0
    params_menu: Dict[str, object] = {
        "k_spring": float(k_spring),
        "k_coulomb": float(k_coulomb),
        "G": float(G),
        "south_strength": 0.0,
        "south_enabled": False,
        "south_axis": 2,
        "damping": float(damping),
        "temp": float(temp),
        "bond_shear": float(bond_shear_ratio),
        "max_speed": float(max_speed),
        "proj_mode": str(proj_mode) if proj_mode else "ship",
        "n_dim": int(d),
        "render_scale": float(_safe_render_scale(render_scale)),
        "airplane_path": str(airplane_path or "airplane.json"),
        "scene_path": str(scene_path or "scene.json"),
    }

    # --- Physics config split: World (planet/env) vs Particle (node sim) ---
    # Migrate legacy world_config.json:{world_config:{...}} into:
    # - particle_config.json:{particle_config:{...}}
    # - world_config.json:{world_env:{...}}
    try:
        legacy_wc = joystick_menu._load_persisted_block("world_config.json", "world_config")
        if legacy_wc:
            p_blk, w_blk = world_config_structs.split_legacy_world_config_block(legacy_wc)
            try:
                if p_blk and not joystick_menu._load_persisted_block("particle_config.json", "particle_config"):
                    joystick_menu._save_persisted_block("particle_config.json", "particle_config", p_blk)
            except Exception:
                pass
            try:
                if w_blk and not joystick_menu._load_persisted_block("world_config.json", "world_env"):
                    joystick_menu._save_persisted_block("world_config.json", "world_env", w_blk)
            except Exception:
                pass
    except Exception:
        pass

    # Apply persisted particle/world values to params_menu before menu init.
    particle_cfg = world_config_structs.particle_config_from_params(params_menu)
    world_env = world_config_structs.world_env_from_params(params_menu)
    try:
        p_persist = joystick_menu._load_persisted_block("particle_config.json", "particle_config")
        if p_persist:
            world_config_structs.particle_config_update_from_dict(particle_cfg, p_persist)
    except Exception:
        pass
    try:
        w_persist = joystick_menu._load_persisted_block("world_config.json", "world_env")
        if w_persist:
            world_config_structs.world_env_update_from_dict(world_env, w_persist)
        else:
            # Back-compat: pull render_scale from legacy block if world_env absent.
            legacy_wc = joystick_menu._load_persisted_block("world_config.json", "world_config")
            if isinstance(legacy_wc, dict) and "render_scale" in legacy_wc:
                world_env.render_scale = float(legacy_wc.get("render_scale"))
    except Exception:
        pass
    try:
        world_config_structs.apply_particle_config_to_params(particle_cfg, params_menu)
    except Exception:
        pass
    try:
        world_config_structs.apply_world_env_to_params(world_env, params_menu)
    except Exception:
        pass

    menu = _JoystickSideMenu.create(params=params_menu)
    joystick = menu.joystick if menu else None

    particle_field_specs = {
        "k_spring": {"step": 0.05},
        "k_coulomb": {"step": 0.05},
        "G": {"step": 0.05},
        "damping": {"step": 0.01, "min": 0.0},
        "temp": {"step": 0.05, "min": 0.0},
        "bond_shear": {"step": 0.05, "min": 0.0},
        "max_speed": {"step": 0.002, "min": 0.0},
        "south_strength": {"step": 0.05, "min": 0.0},
        "south_enabled": {"step": 1, "min": 0, "max": 1},
        "south_axis": {"step": 1, "min": 0},
    }

    world_field_specs = {
        # Keep key name for compatibility, but present it as planet/world scale in UI.
        "render_scale": {"label": "planet_scale", "step": 0.25, "min": 0.01},
        "sea_level_pressure_kpa": {"step": 1.0, "min": 0.0},
        "sea_level_temp_k": {"step": 1.0, "min": 0.0},
    }

    # Flight physics is applied as an overlay to airplane.json "motion".
    flight_motion = {}
    try:
        if menu is not None and isinstance(getattr(menu, "airplane", None), dict):
            flight_motion = menu.airplane.get("motion", {}) if isinstance(menu.airplane.get("motion", {}), dict) else {}
    except Exception:
        flight_motion = {}
    flight_cfg = world_config_structs.flight_physics_from_motion(flight_motion)
    flight_field_specs = {
        "inertial": {"step": 1, "min": 0, "max": 1},
        "auto_heading_on_move": {"step": 1, "min": 0, "max": 1},
        "base_speed": {"step": 0.01, "min": 0.0},
        "thrust_accel": {"step": 0.05, "min": 0.0},
        "throttle_rate": {"step": 0.05, "min": 0.0},
        "lift_k": {"step": 0.05, "min": 0.0},
        "drag_k": {"step": 0.01, "min": 0.0},
        "gravity_g": {"step": 0.01},
        "max_speed": {"step": 0.1, "min": 0.0},
        # Altitude clamps and atmosphere thickness in unit-sphere units.
        "flight_alt_min": {"step": 0.01, "min": 0.0},
        "flight_alt_max": {"step": 0.05, "min": 0.0},
        "atmosphere_alt_max": {"step": 0.01, "min": 0.0},
    }
    try:
        fp_persist = joystick_menu._load_persisted_block("flight_physics.json", "flight_physics")
        if fp_persist:
            world_config_structs.flight_physics_update_from_dict(flight_cfg, fp_persist)
    except Exception:
        pass

    ball_cfg = world_config_structs.ballistics_default()
    ball_field_specs = {
        "max_points": {"step": 1, "min": 2, "max": 256},
    }
    try:
        b_persist = joystick_menu._load_persisted_block("ballistics.json", "ballistics")
        if b_persist:
            world_config_structs.ballistics_update_from_dict(ball_cfg, b_persist)
    except Exception:
        pass

    # Airplane tuning (persisted in airplane.json under a separate key).
    airplane_tuning = airplane_structs.airplane_tuning_default()
    try:
        airplane_tuning = airplane_structs.load_airplane_tuning_json(params_menu.get("airplane_path", "airplane.json"))
        airplane_structs.save_airplane_tuning_json(airplane_tuning, params_menu.get("airplane_path", "airplane.json"))
    except Exception:
        airplane_tuning = airplane_structs.airplane_tuning_default()
    airplane_field_specs = airplane_structs.airplane_tuning_field_specs()

    # Airplane tuning (persisted in airplane.json under a separate key).
    airplane_tuning = airplane_structs.airplane_tuning_default()
    try:
        airplane_tuning = airplane_structs.load_airplane_tuning_json(params_menu.get("airplane_path", "airplane.json"))
        airplane_structs.save_airplane_tuning_json(airplane_tuning, params_menu.get("airplane_path", "airplane.json"))
    except Exception:
        airplane_tuning = airplane_structs.airplane_tuning_default()
    airplane_field_specs = airplane_structs.airplane_tuning_field_specs()

    def _commit_particle_config(cfg_obj: ctypes.Structure) -> None:
        nonlocal params_seq
        try:
            with params_lock:
                world_config_structs.apply_particle_config_to_params(cfg_obj, params_menu)
                params_seq += 1
        except Exception:
            pass

    def _commit_world_env(cfg_obj: ctypes.Structure) -> None:
        nonlocal params_seq
        try:
            with params_lock:
                world_config_structs.apply_world_env_to_params(cfg_obj, params_menu)
                params_seq += 1
        except Exception:
            pass
        try:
            if menu is not None and hasattr(menu, "apply_render_scale"):
                menu.apply_render_scale(float(params_menu.get("render_scale", 10.0)))
        except Exception:
            pass

    def _commit_flight_physics(cfg_obj: ctypes.Structure) -> None:
        # Apply to airplane spec's motion dict (consumed every frame).
        try:
            if menu is None or not isinstance(getattr(menu, "airplane", None), dict):
                return
            motion = menu.airplane.get("motion", None)
            if not isinstance(motion, dict):
                motion = {}
                menu.airplane["motion"] = motion
            # Copy across by field name.
            for fname, _ft in getattr(cfg_obj.__class__, "_fields_", []):
                key = str(fname)
                try:
                    motion[key] = float(getattr(cfg_obj, key)) if _ft in (ctypes.c_float, ctypes.c_double) else int(getattr(cfg_obj, key))
                except Exception:
                    pass
        except Exception:
            pass

    def _commit_ballistics(cfg_obj: ctypes.Structure) -> None:
        try:
            if weap_rt is not None and getattr(weap_rt, "sim", None) is not None:
                weap_rt.sim.max_points = int(getattr(cfg_obj, "max_points", 16) or 16)
        except Exception:
            pass

    # Always-on weapon loadout PIP (top-left), separate from world config.
    loadout = weapons_structs.weapon_loadout_default()
    # Migrate any legacy persisted loadout into the new schema before PIP init.
    try:
        loadout = weapons_structs.load_weapon_loadout_json("weapon_loadout.json")
        try:
            joystick_menu._save_persisted_block(
                "weapon_loadout.json",
                "weapon_loadout",
                joystick_menu._ctypes_struct_to_dict(loadout),
            )
        except Exception:
            pass
    except Exception:
        pass
    loadout_field_specs = weapons_structs.weapon_loadout_field_specs()
    cfg_pip: joystick_menu.CtypesStructEditorPip | None = None
    weap_rt: weapon_runtime.WeaponRuntime | None = None
    fire1_prev = False
    fire2_prev = False
    weapon_splines: list[dict] = []
    try:
        cfg_pip = joystick_menu.CtypesStructEditorPip(
            struct_obj=loadout,
            title="LOADOUT",
            persist_path="weapon_loadout.json",
            persist_key="weapon_loadout",
            field_specs=loadout_field_specs,
            allow_cancel_close=False,
            persist_on_change=True,
        )
        cfg_pip.open()
    except Exception:
        cfg_pip = None

    try:
        weap_rt = weapon_runtime.WeaponRuntime(stats_path="weapon_stats.json")
    except Exception:
        weap_rt = None

    def _commit_airplane_tuning(cfg_obj: ctypes.Structure) -> None:
        # Persist and apply to the running subsystems.
        try:
            airplane_structs.save_airplane_tuning_json(
                cfg_obj, params_menu.get("airplane_path", "airplane.json"), key="airplane_tuning"
            )
        except Exception:
            pass

        try:
            if menu is not None and hasattr(menu, "flight_cam"):
                mode = "auto" if int(getattr(cfg_obj, "control_mode", 1) or 0) != 0 else "manual"
                auto_kp = float(getattr(cfg_obj, "auto_kp", 0.0) or 0.0)
                auto_kd = float(getattr(cfg_obj, "auto_kd", 0.0) or 0.0)
                dbg = bool(int(getattr(cfg_obj, "flight_debug_print", 0) or 0) != 0)
                menu.flight_cam.configure_control_system(mode=mode, auto_kp=auto_kp, auto_kd=auto_kd, debug_print=dbg)

                # If enabled, append generated arms/centers from tuning.
                try:
                    base_air = menu.airplane if isinstance(getattr(menu, "airplane", None), dict) else {}
                    base_arms = base_air.get("arms", None)
                    arms_out = list(base_arms) if isinstance(base_arms, list) else []
                    arms_out.extend(airplane_structs.build_extra_arms_from_tuning(cfg_obj))
                    # Arms are ship-local geometry; do NOT scale them with planet/world size.
                    menu.flight_cam.configure_arms(arms=arms_out, scale=1.0)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if weap_rt is not None:
                weap_rt.debug_print = bool(int(getattr(cfg_obj, "weapon_debug_print", 0) or 0) != 0)
        except Exception:
            pass

    def _commit_airplane_tuning(cfg_obj: ctypes.Structure) -> None:
        # Persist and apply to the running subsystems.
        try:
            airplane_structs.save_airplane_tuning_json(
                cfg_obj, params_menu.get("airplane_path", "airplane.json"), key="airplane_tuning"
            )
        except Exception:
            pass

        try:
            if menu is not None and hasattr(menu, "flight_cam"):
                mode = "auto" if int(getattr(cfg_obj, "control_mode", 1) or 0) != 0 else "manual"
                auto_kp = float(getattr(cfg_obj, "auto_kp", 0.0) or 0.0)
                auto_kd = float(getattr(cfg_obj, "auto_kd", 0.0) or 0.0)
                dbg = bool(int(getattr(cfg_obj, "flight_debug_print", 0) or 0) != 0)
                menu.flight_cam.configure_control_system(mode=mode, auto_kp=auto_kp, auto_kd=auto_kd, debug_print=dbg)

                # If enabled, append generated arms/centers from tuning.
                try:
                    base_air = menu.airplane if isinstance(getattr(menu, "airplane", None), dict) else {}
                    base_arms = base_air.get("arms", None)
                    arms_out = list(base_arms) if isinstance(base_arms, list) else []
                    arms_out.extend(airplane_structs.build_extra_arms_from_tuning(cfg_obj))
                    # Arms are ship-local geometry; do NOT scale them with planet/world size.
                    menu.flight_cam.configure_arms(arms=arms_out, scale=1.0)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if weap_rt is not None:
                weap_rt.debug_print = bool(int(getattr(cfg_obj, "weapon_debug_print", 0) or 0) != 0)
        except Exception:
            pass

    # Ensure joystick.json exists and bind a menu button if missing.
    menu_button_c: int | None = None
    try:
        menu_button_c, did_bind_c = joystick_menu.ensure_menu_button_binding(
            cfg_path="joystick.json",
            font=font,
            width=int(width),
            height=int(height),
            joystick=joystick,
        )
        if did_bind_c and joystick is not None and menu_button_c is not None:
            action = joystick_menu.run_main_menu(
                font=font,
                width=int(width),
                height=int(height),
                joystick=joystick,
                menu_button=int(menu_button_c),
                menu_context={
                    "ctypes_structs": {
                                "airplane_tuning": {
                                    "struct": airplane_tuning,
                                    "title": "AIRPLANE",
                                    "persist_path": params_menu.get("airplane_path", "airplane.json"),
                                    "persist_key": "airplane_tuning",
                                    "field_specs": airplane_field_specs,
                                    "on_commit": _commit_airplane_tuning,
                                    "persist_on_change": True,
                                },
                        "weapon_loadout": {
                            "struct": loadout,
                            "title": "LOADOUT",
                            "persist_path": "weapon_loadout.json",
                            "persist_key": "weapon_loadout",
                            "field_specs": loadout_field_specs,
                                    "persist_on_change": True,
                        },
                        "particle_config": {
                            "struct": particle_cfg,
                            "title": "PARTICLE",
                            "persist_path": "particle_config.json",
                            "persist_key": "particle_config",
                            "on_commit": _commit_particle_config,
                            "field_specs": particle_field_specs,
                            "persist_on_change": True,
                        },
                        "world_env": {
                            "struct": world_env,
                            "title": "WORLD",
                            "persist_path": "world_config.json",
                            "persist_key": "world_env",
                            "on_commit": _commit_world_env,
                            "field_specs": world_field_specs,
                            "persist_on_change": True,
                        },
                        "flight_physics": {
                            "struct": flight_cfg,
                            "title": "FLIGHT",
                            "persist_path": "flight_physics.json",
                            "persist_key": "flight_physics",
                            "on_commit": _commit_flight_physics,
                            "field_specs": flight_field_specs,
                            "persist_on_change": True,
                        },
                        "ballistics": {
                            "struct": ball_cfg,
                            "title": "BALLISTICS",
                            "persist_path": "ballistics.json",
                            "persist_key": "ballistics",
                            "on_commit": _commit_ballistics,
                            "field_specs": ball_field_specs,
                            "persist_on_change": True,
                        },
                    }
                },
            )
            if action == "quit":
                try:
                    if weap_rt is not None:
                        weap_rt.stop()
                except Exception:
                    pass
                stop_evt.set()
                pygame.quit()
                return
    except Exception:
        menu_button_c = None

    scene = _load_scene_spec(params_menu.get("scene_path", "scene.json"))

    # If terrain is enabled, attach the heightmap to the flight camera so altitude + ground clamp
    # reflect local terrain rather than a single global radius.
    try:
        if menu is not None and hasattr(menu, "flight_cam"):
            terr = scene.get("terrain", {}) if isinstance(scene.get("terrain", {}), dict) else {}
            hm = None
            if bool(terr.get("enabled", False)) and bool(terr.get("planet_enabled", False)):
                hm = _load_grayscale_heightmap(str(terr.get("heightmap_path", "") or ""))
            # Values in the scene are in unscaled units; the ship world uses render_scale.
            hs = float(terr.get("planet_height_scale", 0.0)) * float(menu.render_scale)
            hb = float(terr.get("planet_height_bias", 0.5))
            try:
                menu.flight_cam.terrain_heightmap = hm
                menu.flight_cam.terrain_height_scale = float(max(0.0, hs))
                menu.flight_cam.terrain_height_bias = float(max(0.0, min(1.0, hb)))
            except Exception:
                pass
    except Exception:
        pass

    stop_evt = threading.Event()
    phys_fps = 0.0
    phys_dt = float(dt)
    phys_substeps = 1

    base_dt = float(dt)
    adaptive_dt = float(dt)
    min_dt_local = float(min_dt) if (min_dt is not None) else float(np.finfo(np.float64).tiny)
    if not np.isfinite(min_dt_local) or min_dt_local <= 0.0:
        min_dt_local = float(np.finfo(np.float64).tiny)
    max_speed_local = float(max_speed)
    if not np.isfinite(max_speed_local) or max_speed_local <= 0.0:
        max_speed_local = 0.02
    substep_dt_max = float(c_substep_dt_max) if (c_substep_dt_max is not None) else base_dt
    if not np.isfinite(substep_dt_max) or substep_dt_max <= 0.0:
        substep_dt_max = base_dt
    substep_dt_max = max(min_dt_local, substep_dt_max)
    substeps_cap = max(1, int(c_substeps_max))
    # Match the Torch adaptive controller: dt grows back up to (but not above) base_dt.
    dt_max = base_dt

    # Posthoc metric scratch: previous sim positions (used to measure true displacement).
    pos_prev = np.empty((int(state.n_cap), d), dtype=np.float64)

    # Physics thread limiter (Hz). 0 or negative => uncapped.
    phys_hz = float(phys_fps_limit)
    phys_hz = phys_hz if phys_hz > 0.0 else 0.0
    phys_period = (1.0 / phys_hz) if phys_hz > 0.0 else 0.0

    def physics_worker():
        nonlocal phys_fps, phys_dt, phys_substeps, adaptive_dt, max_speed_local
        last = time.perf_counter()
        frames = 0
        next_tick = last
        substep_pressure = 0.0
        last_applied_seq = -1
        while not stop_evt.is_set():
            # Apply any live-edited params at a safe boundary (between steps).
            seq_now = None
            snap = None
            with params_lock:
                seq_now = params_seq
                if seq_now != last_applied_seq:
                    snap = dict(params_menu)
            if snap is not None:
                try:
                    state.set_params(
                        k_spring=float(snap.get("k_spring", k_spring)),
                        k_coulomb=float(snap.get("k_coulomb", k_coulomb)),
                        G=float(snap.get("G", G)),
                        damping=float(snap.get("damping", damping)),
                        softening=float(softening),
                    )
                    # Lorentz + B are controlled via CLI in C-physics mode (not in the Python side-menu).
                    state.set_lorentz(lorentz_k=float(lorentz_k), Bx=bx, By=by, Bz=bz)
                    tp = float(snap.get("temp", temp))
                    state.set_temp(ambient=tp, convection=0.5, noise=abs(tp), heat_gain=1.0)
                    state.set_bonds(
                        enabled=bool(bonds_enabled),
                        max_per_node=int(bond_max_per_node),
                        link_angle=float(bond_link_angle),
                        shear_ratio=float(snap.get("bond_shear", bond_shear_ratio)),
                        k=float(bond_k),
                        ionic=bool(ionic_bonds),
                        ionic_valence=int(ionic_valence),
                    )
                    max_speed_local = float(snap.get("max_speed", max_speed_local))
                except Exception as exc:
                    # Keep sim running if a bad edit slipped through.
                    print(f"[warn] failed to apply live C params: {exc}", file=sys.stderr, flush=True)
                last_applied_seq = int(seq_now)

            step_t0 = time.perf_counter()

            # Snapshot the sim buffer before stepping so we can compute a true
            # posthoc displacement metric.
            sim_before = int(state.header.sim_idx & 1)
            n_active_local = int(state.header.n_active)
            if n_active_local < 0:
                n_active_local = 0
            if n_active_local > int(state.n_cap):
                n_active_local = int(state.n_cap)
            pos_prev[:n_active_local, :] = state.pos[sim_before][:n_active_local]

            dt_try = float(adaptive_dt)
            if not np.isfinite(dt_try) or dt_try <= 0.0:
                dt_try = base_dt
            dt_try = max(min_dt_local, min(dt_try, dt_max))

            substeps_base = int(math.ceil(dt_try / substep_dt_max))
            substeps_base = max(1, substeps_base)
            # If min_dt_local is extremely small, dt_try/min_dt_local can be huge.
            # We only ever use this value to clamp to substeps_cap, so avoid
            # creating a massive Python int.
            if dt_try <= min_dt_local:
                substeps_dt_cap = 1
            else:
                ratio = dt_try / min_dt_local
                substeps_dt_cap = substeps_cap if ratio >= float(substeps_cap) else int(ratio)
            substeps_dt_cap = max(1, int(substeps_dt_cap))
            substeps_cap_eff = max(1, min(substeps_cap, substeps_dt_cap))

            # Substep pressure: when the FPS limiter is engaged and we have slack,
            # spend some of that time budget on additional substeps for precision.
            max_add = max(0, substeps_cap_eff - substeps_base)
            add = int(round(substep_pressure * float(max_add))) if max_add else 0
            substeps = min(substeps_cap_eff, substeps_base + add)
            dt_sub = dt_try / float(substeps)
            dt_sub = max(min_dt_local, dt_sub)

            state.step(dt_sub, steps=substeps)
            frames += substeps
            phys_dt = dt_try
            phys_substeps = substeps

            if dt_mode == "single-posthoc":
                try:
                    sim_after = int(state.header.sim_idx & 1)
                    pos_after = state.pos[sim_after][:n_active_local]
                    disp_est = float(np.linalg.norm(pos_after - pos_prev[:n_active_local], axis=1).max()) if pos_after.size else 0.0
                except Exception:
                    disp_est = 0.0

                # Torch-like thresholds
                subdiv_margin_hi = 0.05
                grow_margin_lo = 0.20
                grow_rate = 2.0
                shrink_rate = 0.50
                dim_scale = math.sqrt(max(d, 1) / 3.0)
                disp_base = min(max_speed_local * base_dt * dim_scale, 1.8)
                d_hi = disp_base * (1.0 + subdiv_margin_hi)
                d_lo = disp_base * (1.0 - grow_margin_lo)

                # When things go unstable, shrinking by 0.5 per frame can take a
                # long time to reach the tiny dt values the Python path will
                # converge to (especially after a parameter jump). Use a
                # ratio-based shrink so we can drop dt much faster when the
                # displacement overshoots badly.
                denom = max(float(d_hi), 1e-12)
                ratio = float(disp_est) / denom if np.isfinite(disp_est) else float("inf")
                if ratio > 1.0:
                    if ratio > 8.0:
                        shrink_factor = 0.01
                    elif ratio > 3.0:
                        shrink_factor = 0.10
                    else:
                        shrink_factor = shrink_rate
                    adaptive_dt = max(min_dt_local, dt_try * float(shrink_factor))
                elif disp_est < d_lo:
                    adaptive_dt = min(dt_max, dt_try * grow_rate)
                else:
                    adaptive_dt = dt_try
            now = time.perf_counter()
            if now - last >= 0.5:
                phys_fps = frames / (now - last)
                frames = 0
                last = now

            if phys_period > 0.0:
                # Pace the physics loop to ~phys_hz.
                next_tick += phys_period
                sleep_s = next_tick - time.perf_counter()

                # Update substep pressure for the next iteration.
                # - If we're sleeping, increase pressure toward more substeps.
                # - If we're behind, decrease pressure so we don't spiral.
                if sleep_s > 0.0:
                    # Normalize slack into a gentle [0,1] signal.
                    slack_frac = min(1.0, sleep_s / max(phys_period, 1e-9))
                    substep_pressure = min(1.0, substep_pressure + 0.25 * slack_frac)
                else:
                    substep_pressure = max(0.0, substep_pressure * 0.5)

                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    # If we fell behind, resync so we don't accumulate drift.
                    next_tick = time.perf_counter()

    phys_thread = threading.Thread(target=physics_worker, daemon=True)
    phys_thread.start()

    # Reusable edge buffer for renderer (avoid per-frame allocations).
    # Used for straight edges and/or geodesic arcs.
    edge_buf = np.empty((int(state.header.spring_cap), 4), dtype=np.float32) if (draw_direct_edges or draw_edges) else None

    # --- Render-thread-only projection basis (D -> 3) ---
    # Keep this OUT of the physics thread; it's purely for display.
    proj_mean = np.zeros((1, d), dtype=np.float32)
    proj_mat = np.zeros((d, 3), dtype=np.float32)

    # Separate cached basis for minimap rendering (may differ from main).
    proj_mm_mean = np.zeros((1, d), dtype=np.float32)
    proj_mm_mat = np.zeros((d, 3), dtype=np.float32)
    proj_mm_last_mode = ""

    def _sample_points_f32(pos_nd_view: np.ndarray, n_use: int, cap: int = 4096) -> np.ndarray:
        if n_use <= 0:
            return np.zeros((0, d), dtype=np.float32)
        m = int(min(n_use, cap))
        if m == n_use:
            # Copy/cast without allocating a new array each call.
            np.copyto(pos_work_f32[:m, :], pos_nd_view[:m, :], casting="unsafe")
            return pos_work_f32[:m, :]
        idx = np.linspace(0, n_use - 1, num=m, dtype=np.int64)
        np.copyto(pos_work_f32[:m, :], pos_nd_view[idx, :], casting="unsafe")
        return pos_work_f32[:m, :]

    def _compute_projection_basis(mode: str, pts_nd_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (mean (1,D), P (D,3)), float32.
        if pts_nd_f32.size == 0:
            mean_local = np.zeros((1, d), dtype=np.float32)
            P = np.zeros((d, 3), dtype=np.float32)
            for i in range(min(d, 3)):
                P[i, i] = 1.0
            return mean_local, P

        mode = _basis_mode_for_display(mode)
        mean_local = pts_nd_f32.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        D = int(pts_nd_f32.shape[1])
        if mode == "axes" or D <= 3:
            P = np.zeros((D, 3), dtype=np.float32)
            for i in range(min(D, 3)):
                P[i, i] = 1.0
            return mean_local, P

        if mode == "random":
            R = np.random.standard_normal(size=(D, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(R)
            return mean_local, Q[:, :3].astype(np.float32, copy=False)

        # default: PCA via randomized SVD (fast top-3)
        Xc = (pts_nd_f32 - mean_local).astype(np.float32, copy=False)
        # small oversampling improves stability when spectrum is flat
        l = 6
        R = np.random.standard_normal(size=(D, l)).astype(np.float32)
        Y = Xc @ R  # (N,l)
        Q, _ = np.linalg.qr(Y, mode="reduced")  # (N,l)
        B = Q.T @ Xc  # (l,D)
        _, _, Vt = np.linalg.svd(B, full_matrices=False)
        P = Vt[:3, :].T.astype(np.float32, copy=False)  # (D,3)
        return mean_local, P

    # Initialize basis from the initial positions.
    if d == 3:
        proj_mean[:] = 0.0
        proj_mat[:] = np.eye(3, dtype=np.float32)
    else:
        pts0 = _sample_points_f32(np.asarray(pos_np, dtype=np.float32), int(pos_np.shape[0]))
        proj_mean, proj_mat = _compute_projection_basis(str(proj_mode), pts0)

    proj_last_t = time.perf_counter()
    proj_refresh_s = 0.5  # only used when freeze_pca=False

    # NOTE: Unlike the original C path, we derive per-frame colors and per-node
    # point sizes from the live C-state charge/mass, matching Python mode.

    try:
        running = True
        t0 = time.perf_counter()
        last_render_t = t0
        while running:
            now_render_t = time.perf_counter()
            frame_dt_s = float(now_render_t - last_render_t)
            last_render_t = now_render_t
            if not np.isfinite(frame_dt_s) or frame_dt_s < 0.0:
                frame_dt_s = 0.0
            frame_dt_s = float(min(0.25, frame_dt_s))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif menu_button_c is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button_c):
                    # Menu button opens the full menu. PIP is always-on.
                    try:
                        action = joystick_menu.run_main_menu(
                            font=font,
                            width=int(width),
                            height=int(height),
                            joystick=joystick,
                            menu_button=int(menu_button_c),
                            menu_context={
                                "ctypes_structs": {
                                    "airplane_tuning": {
                                        "struct": airplane_tuning,
                                        "title": "AIRPLANE",
                                        "persist_path": params_menu.get("airplane_path", "airplane.json"),
                                        "persist_key": "airplane_tuning",
                                        "field_specs": airplane_field_specs,
                                        "on_commit": _commit_airplane_tuning,
                                        "persist_on_change": True,
                                    },
                                    "weapon_loadout": {
                                        "struct": loadout,
                                        "title": "LOADOUT",
                                        "persist_path": "weapon_loadout.json",
                                        "persist_key": "weapon_loadout",
                                        "field_specs": loadout_field_specs,
                                        "persist_on_change": True,
                                    },
                                    "particle_config": {
                                        "struct": particle_cfg,
                                        "title": "PARTICLE",
                                        "persist_path": "particle_config.json",
                                        "persist_key": "particle_config",
                                        "on_commit": _commit_particle_config,
                                        "field_specs": particle_field_specs,
                                        "persist_on_change": True,
                                    },
                                    "world_env": {
                                        "struct": world_env,
                                        "title": "WORLD",
                                        "persist_path": "world_config.json",
                                        "persist_key": "world_env",
                                        "on_commit": _commit_world_env,
                                        "field_specs": world_field_specs,
                                        "persist_on_change": True,
                                    },
                                    "flight_physics": {
                                        "struct": flight_cfg,
                                        "title": "FLIGHT",
                                        "persist_path": "flight_physics.json",
                                        "persist_key": "flight_physics",
                                        "on_commit": _commit_flight_physics,
                                        "field_specs": flight_field_specs,
                                        "persist_on_change": True,
                                    },
                                    "ballistics": {
                                        "struct": ball_cfg,
                                        "title": "BALLISTICS",
                                        "persist_path": "ballistics.json",
                                        "persist_key": "ballistics",
                                        "on_commit": _commit_ballistics,
                                        "field_specs": ball_field_specs,
                                        "persist_on_change": True,
                                    },
                                }
                            },
                        )
                        if action == "quit":
                            running = False
                    except Exception:
                        pass
                elif menu and event.type in (pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                    # Forward joystick events to the shared side-menu handler *except*
                    # the bound menu button (handled above).
                    if not (
                        menu_button_c is not None
                        and event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP)
                        and int(getattr(event, "button", -9999)) == int(menu_button_c)
                    ):
                        menu.handle_event(event)

            # If the window is closing, don't attempt any further GL calls.
            if not running:
                break

            # Apply joystick side-menu adjustments every frame.
            if menu:
                menu.poll_axes()
                now_t = pygame.time.get_ticks() * 0.001
                mode_now_for_menu = str(params_menu.get("proj_mode", proj_mode) or "pca")
                if not menu.flight_active(mode_now_for_menu):
                    did_change = menu.update_params(
                        params_menu,
                        now_t=now_t,
                        allow_proj_mode=True,
                        allow_n_dim=False,
                    )
                    if did_change:
                        with params_lock:
                            params_seq += 1

            # Config PIP (menu-nav bindings) tick.
            if cfg_pip is not None and joystick is not None:
                try:
                    cfg_pip.tick(joystick=joystick)
                except Exception:
                    pass

            # Enforce loadout constraints:
            # - bay_inner is only usable when bay_outer is deployed (non-zero).
            try:
                if int(getattr(loadout, "bay_outer", 0)) == 0 and int(getattr(loadout, "bay_inner", 0)) != 0:
                    loadout.bay_inner = 0
                    try:
                        joystick_menu._save_persisted_block(
                            "weapon_loadout.json",
                            "weapon_loadout",
                            joystick_menu._ctypes_struct_to_dict(loadout),
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            # Fire dispatch (Weapon 1 / Weapon 2) -> projectile simulator queue.
            if joystick is not None and weap_rt is not None:
                try:
                    cfg = joystick_menu.load_or_create_joystick_config("joystick.json")
                    if menu is not None:
                        try:
                            targeting_mode = int(getattr(loadout, "targeting", 0) or 0)
                        except Exception:
                            targeting_mode = 0
                        try:
                            menu._targeting_active = bool(int(targeting_mode) != 0)
                        except Exception:
                            pass
                        try:
                            menu._active_control_set = _resolve_effective_control_set(menu)
                        except Exception:
                            pass

                    set_name = _resolve_effective_control_set(menu)
                    b1 = _get_set_binding(cfg, set_name=set_name, group="weapons", key="fire_1")
                    b2 = _get_set_binding(cfg, set_name=set_name, group="weapons", key="fire_2")
                    axes_now, buttons_now, hats_now = joystick_menu._poll_joystick_snapshot(joystick)
                    f1_active, f1_analog = _binding_active(b1, axes_now, buttons_now, hats_now)
                    f2_active, f2_analog = _binding_active(b2, axes_now, buttons_now, hats_now)

                    ship_snap = None
                    try:
                        mode_ship = str(params_menu.get("proj_mode", proj_mode) or "pca")
                        need_snap = (f1_active and (not fire1_prev)) or (f2_active and (not fire2_prev))
                        if need_snap and menu is not None and menu.flight_active(mode_ship):
                            sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                            sf = np.asarray(menu.flight_cam.basis()[2], dtype=np.float32)
                            sv = np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)

                            hm_info = None
                            hm_ref = None
                            try:
                                fc = menu.flight_cam
                                hm = getattr(fc, "terrain_heightmap", None)
                                hs = float(getattr(fc, "terrain_height_scale", 0.0) or 0.0)
                                hb = float(getattr(fc, "terrain_height_bias", 0.5) if getattr(fc, "terrain_height_bias", None) is not None else 0.5)
                                if hm is not None and hs != 0.0:
                                    hm_np = np.ascontiguousarray(np.asarray(hm, dtype=np.float32))
                                    h_h = int(hm_np.shape[0])
                                    h_w = int(hm_np.shape[1])
                                    if h_h > 1 and h_w > 1:
                                        hm_ref = hm_np
                                        ptr = int(hm_np.__array_interface__["data"][0])
                                        hm_info = {
                                            "ptr": ptr,
                                            "w": h_w,
                                            "h": h_h,
                                            "stride": h_w,
                                            "height_scale": float(hs),
                                            "height_bias": float(hb),
                                        }
                            except Exception:
                                hm_info = None
                                hm_ref = None

                            nodes_info = None
                            nodes_ref = None
                            try:
                                front_i = int(state.front_idx)
                                n_nodes = int(state.header.n_active)
                                if n_nodes < 0:
                                    n_nodes = 0
                                if n_nodes > int(state.n_cap):
                                    n_nodes = int(state.n_cap)
                                if n_nodes > 0:
                                    pos_nodes = np.ascontiguousarray(np.asarray(state.pos[front_i][:n_nodes, :3], dtype=np.float32))
                                    masses_nodes = np.ascontiguousarray(np.asarray(state.mass[:n_nodes], dtype=np.float32))
                                    radii_nodes = radius_scale_local * np.cbrt(np.maximum(masses_nodes, 1e-6)) / mass_ref_cuberoot
                                    radii_nodes = np.ascontiguousarray(np.asarray(radii_nodes, dtype=np.float32))
                                    nodes_ref = (pos_nodes, radii_nodes)
                                    nodes_info = {
                                        "pos_ptr": int(pos_nodes.__array_interface__["data"][0]),
                                        "rad_ptr": int(radii_nodes.__array_interface__["data"][0]),
                                        "count": int(n_nodes),
                                        "pos_stride": 3,
                                        "rad_stride": 1,
                                    }
                            except Exception:
                                nodes_info = None
                                nodes_ref = None

                            ship_snap = weapon_runtime.ShipSnapshot(
                                pos=(float(sp[0]), float(sp[1]), float(sp[2])),
                                vel=(float(sv[0]), float(sv[1]), float(sv[2])),
                                fwd=(float(sf[0]), float(sf[1]), float(sf[2])),
                                planet_surface_r=float(getattr(menu.flight_cam, "planet_surface_r", 0.0) or 0.0),
                                gravity_g=float(getattr(menu.flight_cam, "gravity_g", 0.0) or 0.0),
                                terrain_heightmap=hm_info,
                                terrain_heightmap_ref=hm_ref,
                                nodes=nodes_info,
                                nodes_ref=nodes_ref,
                            )
                    except Exception:
                        ship_snap = None

                    if f1_active and (not fire1_prev):
                        resolved = weapons_structs.resolve_virtual_weapon(loadout, 1)
                        for source_name, weapon_type in resolved:
                            wo = None
                            wd = None
                            try:
                                mode_ship = str(params_menu.get("proj_mode", proj_mode) or "pca")
                                in_flight = bool(menu is not None and menu.flight_active(mode_ship))
                                if in_flight and str(source_name) == "nose gun" and ship_snap is not None:
                                    sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                    sr, su, sf = menu.flight_cam.basis()
                                    wo_t = targeting_system.compute_nose_gun_origin_world(
                                        ship_pos=sp,
                                        ship_right=np.asarray(sr, dtype=np.float32),
                                        ship_up=np.asarray(su, dtype=np.float32),
                                        ship_fwd=np.asarray(sf, dtype=np.float32),
                                    )
                                    wo = wo_t
                                    wd = targeting_system.compute_weapon_dir_from_reticle(
                                        weapon_origin=np.asarray(wo_t, dtype=np.float32),
                                        fallback_view_dir=np.asarray(target_view_dir, dtype=np.float32),
                                        reticle=target_sys.get_reticle(reticle_id),
                                    )
                            except Exception:
                                wo = None
                                wd = None
                            weap_rt.enqueue_fire(
                                loadout=loadout,
                                weapon_slot=1,
                                resolved=[(source_name, weapon_type)],
                                ship=ship_snap,
                                analog=float(f1_analog),
                                weapon_origin=wo,
                                weapon_dir=wd,
                            )
                    if f2_active and (not fire2_prev):
                        resolved = weapons_structs.resolve_virtual_weapon(loadout, 2)
                        for source_name, weapon_type in resolved:
                            wo = None
                            wd = None
                            try:
                                mode_ship = str(params_menu.get("proj_mode", proj_mode) or "pca")
                                in_flight = bool(menu is not None and menu.flight_active(mode_ship))
                                if in_flight and str(source_name) == "nose gun" and ship_snap is not None:
                                    sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                    sr, su, sf = menu.flight_cam.basis()
                                    wo_t = targeting_system.compute_nose_gun_origin_world(
                                        ship_pos=sp,
                                        ship_right=np.asarray(sr, dtype=np.float32),
                                        ship_up=np.asarray(su, dtype=np.float32),
                                        ship_fwd=np.asarray(sf, dtype=np.float32),
                                    )
                                    wo = wo_t
                                    wd = targeting_system.compute_weapon_dir_from_reticle(
                                        weapon_origin=np.asarray(wo_t, dtype=np.float32),
                                        fallback_view_dir=np.asarray(target_view_dir, dtype=np.float32),
                                        reticle=target_sys.get_reticle(reticle_id),
                                    )
                            except Exception:
                                wo = None
                                wd = None
                            weap_rt.enqueue_fire(
                                loadout=loadout,
                                weapon_slot=2,
                                resolved=[(source_name, weapon_type)],
                                ship=ship_snap,
                                analog=float(f2_analog),
                                weapon_origin=wo,
                                weapon_dir=wd,
                            )

                    fire1_prev = bool(f1_active)
                    fire2_prev = bool(f2_active)

                    evts = weap_rt.poll_consequences(max_events=32)
                    now_s = pygame.time.get_ticks() * 0.001
                    try:
                        weapon_splines[:] = [s for s in weapon_splines if float(s.get("t_end", 0.0)) >= now_s]
                    except Exception:
                        pass
                    for evt in evts:
                        try:
                            if evt.kind != "info" or not isinstance(evt.payload, dict):
                                continue
                            if evt.payload.get("event") != "dll_processed":
                                continue
                            # Hard requirement: only draw paths returned by the C bullet simulator.
                            if not bool(evt.payload.get("from_c", False)):
                                continue
                            if int(evt.payload.get("ok", 0) or 0) == 0:
                                continue
                            pts = evt.payload.get("spline_points")
                            if isinstance(pts, list) and len(pts) >= 2:
                                weapon_splines.append({"t_end": float(now_s + 1.25), "pts": pts})
                        except Exception:
                            pass
                except Exception:
                    pass

            # Flip-on-read: ask the writer to swap buffers at a step boundary.
            # Springs are double-buffered in C, so the front spring list is safe to read.
            state.request_swap()

            # Snapshot which buffer is front once per frame.
            front = state.front_idx
            n_active_frame = int(state.header.n_active)
            if n_active_frame < 0:
                n_active_frame = 0
            if n_active_frame > int(state.n_cap):
                n_active_frame = int(state.n_cap)
            pos_front = state.pos[front][:n_active_frame]
            mode_now = str(params_menu.get("proj_mode", proj_mode) or "pca")
            charges_live = np.asarray(state.charge[:n_active_frame], dtype=np.float32)
            masses_live = np.asarray(state.mass[:n_active_frame], dtype=np.float32)
            colors_live = _colors_from_charges(charges_live)
            radii_live = radius_scale_local * np.cbrt(np.maximum(masses_live, 1e-6)) / mass_ref_cuberoot
            # Base size is in pixels-at-unit-distance; the vertex shader applies 1/w.
            point_sizes = np.maximum(8.0, 700.0 * radii_live)

            # Update projection basis in the render thread (never in physics).
            if d != 3:
                nowp = time.perf_counter()
                mode_basis = _basis_mode_for_display(mode_now)
                if not freeze_pca:
                    if nowp - proj_last_t >= proj_refresh_s:
                        pts_s = _sample_points_f32(pos_front, n_active_frame)
                        proj_mean, proj_mat = _compute_projection_basis(mode_basis, pts_s)
                        proj_last_t = nowp
                else:
                    # If frozen, only recompute when mode changes.
                    # (very cheap to check by caching the last mode string)
                    if not hasattr(_run_c_physics_only, "_c_last_proj_mode"):
                        _run_c_physics_only._c_last_proj_mode = ""
                    if mode_basis != _run_c_physics_only._c_last_proj_mode:
                        pts_s = _sample_points_f32(pos_front, n_active_frame)
                        proj_mean, proj_mat = _compute_projection_basis(mode_basis, pts_s)
                        _run_c_physics_only._c_last_proj_mode = mode_basis

            # Minimap projection basis (independent of main mode).
            minimap_mode = menu.minimap_mode() if menu is not None else "pca"
            if d != 3:
                mm_basis = _basis_mode_for_display(minimap_mode)
                if mm_basis != proj_mm_last_mode:
                    pts_s = _sample_points_f32(pos_front, n_active_frame)
                    proj_mm_mean, proj_mm_mat = _compute_projection_basis(mm_basis, pts_s)
                    proj_mm_last_mode = mm_basis
            else:
                minimap_mode = str(minimap_mode)

            # Camera
            if _is_ship_proj_mode(mode_now) and menu is not None:
                # Render-only scale factor (independent of physics scale).
                render_scale = _safe_render_scale(params_menu.get("render_scale", 10.0))
                try:
                    menu.apply_render_scale(render_scale)
                except Exception:
                    pass

                # Flight camera (toggle with joystick button 1)
                dt_cam = 1.0 / 60.0
                try:
                    dt_cam = max(1e-4, float(clock.get_time()) * 0.001)
                except Exception:
                    pass
                eye, center_cam = menu.flight_camera(dt=dt_cam, proj_mode=mode_now)
                center = np.asarray(center_cam, dtype=np.float32)
            elif _is_map_proj_mode(mode_now):
                eye = (0.0, 0.0, 4.0)
                center = np.zeros(3, dtype=np.float32)
            else:
                eye = _orbit_eye((0.0, 0.0, 0.0), 4.0, time.perf_counter() - t0)
                center = np.zeros(3, dtype=np.float32)

            # Snapshot springs for the renderer.
            springs_live = None
            if edge_buf is not None:
                sp = state.springs_view()
                if sp.size:
                    m = int(sp.shape[0])
                    springs_live = edge_buf[:m]
                    springs_live[:, 0] = sp["i"].astype(np.float32, copy=False)
                    springs_live[:, 1] = sp["j"].astype(np.float32, copy=False)
                    springs_live[:, 2] = sp["rest_angle"].astype(np.float32, copy=False)
                    springs_live[:, 3] = sp["k"].astype(np.float32, copy=False)
                else:
                    springs_live = edge_buf[:0]

            hud_lines = [
                "C physics mode (no torch)",
                f"n={n_active_frame}/{int(state.n_cap)} d={int(d)} dt={phys_dt:.6f} substeps={phys_substeps}",
                f"phys_fps={phys_fps:.1f}",
                f"front={front} swap_seq={state.swap_seq}",
                f"springs={0 if springs_live is None else int(springs_live.shape[0])} cap={spring_cap}",
                (
                    f"cam_alt={menu.flight_cam.altitude():.3f} "
                    f"axes(0..5)=({menu.axis_state.get(0,0.0):+.2f},{menu.axis_state.get(1,0.0):+.2f},"
                    f"{menu.axis_state.get(2,0.0):+.2f},{menu.axis_state.get(3,0.0):+.2f},"
                    f"{menu.axis_state.get(4,0.0):+.2f},{menu.axis_state.get(5,0.0):+.2f}) "
                    f"minimap={menu.minimap_mode()}"
                )
                if (menu is not None and _is_ship_proj_mode(mode_now))
                else "",
            ]

            hud_right: list[str] = []
            try:
                is_ship = bool(_is_ship_proj_mode(mode_now) and menu is not None)
                if is_ship and menu is not None:
                    sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                    sr, su, sf = menu.flight_cam.basis()
                    wo = targeting_system.compute_nose_gun_origin_world(
                        ship_pos=sp,
                        ship_right=np.asarray(sr, dtype=np.float32),
                        ship_up=np.asarray(su, dtype=np.float32),
                        ship_fwd=np.asarray(sf, dtype=np.float32),
                    )
                    rt_st = target_sys.get_reticle(reticle_id)
                    tel = reticle_telemetry.build_reticle_telemetry(
                        weap_rt=weap_rt,
                        reticle=rt_st,
                        weapon_origin=np.asarray(wo, dtype=np.float32),
                    )
                    if tel.left_lines:
                        hud_lines.extend(tel.left_lines)
                    if tel.right_lines:
                        hud_right.extend(tel.right_lines)
            except Exception:
                hud_right = []

            # Clear + set camera for this frame.
            # In ship mode, use altitude-based sky color for a simple atmosphere feel.
            is_ship = bool(_is_ship_proj_mode(mode_now) and menu is not None)
            alt_frac = 1.0
            if is_ship:
                try:
                    rs_alt = _safe_render_scale(params_menu.get("render_scale", 10.0))
                except Exception:
                    rs_alt = 10.0
                alt_max = float(_ATMOSPHERE_ALT_MAX) * float(rs_alt)
                alt_now = float(menu.flight_cam.altitude())
                if alt_max > 1e-9:
                    alt_frac = float(max(0.0, min(1.0, alt_now / alt_max)))
            if is_ship:
                sky_rgb = _scene_sky_color(scene, float(alt_frac))
            else:
                # Outside ship mode, default to space color so the world is readable.
                try:
                    sky = scene.get("sky", {}) if isinstance(scene.get("sky", {}), dict) else {}
                    sky_rgb = _safe_color3(sky.get("space_color"), (0.02, 0.03, 0.06))
                except Exception:
                    sky_rgb = (0.02, 0.03, 0.06)
            glClearColor(float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]), 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            upx, upy, upz = (0.0, 1.0, 0.0)
            if _is_ship_proj_mode(mode_now) and menu is not None:
                eye_v = np.asarray(eye, dtype=np.float32)
                upx, upy, upz = lookat_up_away_from_planet(eye=eye_v, center=center)
            gluLookAt(
                float(eye[0]),
                float(eye[1]),
                float(eye[2]),
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(upx),
                float(upy),
                float(upz),
            )

            # Distant sun (all non-map modes). Draw it first without depth so the planet can occlude it.
            if not _is_map_proj_mode(mode_now):
                try:
                    sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
                    sun_dir = np.asarray(sun.get("direction"), dtype=np.float32)
                    sd = float(np.linalg.norm(sun_dir))
                    if sd > 1e-6:
                        sun_dir = (sun_dir / sd).astype(np.float32, copy=False)
                    else:
                        sun_dir = np.array([-0.25, 0.35, 1.0], dtype=np.float32)

                    sun_col = sun.get("color", (1.0, 0.98, 0.92))
                    inten = float(sun.get("intensity", 1.0))
                    sun_rgb = np.array([float(sun_col[0]), float(sun_col[1]), float(sun_col[2])], dtype=np.float32) * float(max(0.0, inten))
                    size_px = float(sun.get("size_px", 160.0))
                    dist = float(sun.get("distance", 800.0))
                    dist = float(max(10.0, min(dist, 0.9 * float(z_far))))
                    eye_np = np.asarray(eye, dtype=np.float32)
                    sun_pos = (eye_np + sun_dir * dist).astype(np.float32, copy=False)

                    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
                    glDisable(GL_DEPTH_TEST)
                    glDepthMask(False)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glUseProgram(sun_prog)
                    if sun_u_color is not None and int(sun_u_color) >= 0:
                        glUniform3f(int(sun_u_color), float(sun_rgb[0]), float(sun_rgb[1]), float(sun_rgb[2]))
                    if sun_u_size_px is not None and int(sun_u_size_px) >= 0:
                        glUniform1f(int(sun_u_size_px), float(size_px))

                    glBindBuffer(GL_ARRAY_BUFFER, vbos["sun_pos"])
                    glBufferData(GL_ARRAY_BUFFER, sun_pos.nbytes, sun_pos, GL_DYNAMIC_DRAW)
                    glEnableVertexAttribArray(sun_a_pos)
                    glVertexAttribPointer(sun_a_pos, 3, GL_FLOAT, False, 0, None)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    glDrawArrays(GL_POINTS, 0, 1)

                    glDisableVertexAttribArray(sun_a_pos)
                    glUseProgram(0)
                    glDisable(GL_BLEND)
                    glDepthMask(True)
                    if depth_was_enabled:
                        glEnable(GL_DEPTH_TEST)
                except Exception:
                    pass

            # Planet ground + atmosphere shell (all non-map modes).
            if not _is_map_proj_mode(mode_now):
                try:
                    rs = _safe_render_scale(params_menu.get("render_scale", 10.0)) if is_ship else 1.0
                except Exception:
                    rs = 1.0
                try:
                    sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
                    sun_dir = np.asarray(sun.get("direction"), dtype=np.float32)
                except Exception:
                    sun_dir = None
                try:
                    paint = scene.get("node_paint", {}) if isinstance(scene.get("node_paint", {}), dict) else {}
                    paint_strength = float(paint.get("blend_strength", 1.0))
                except Exception:
                    paint_strength = 1.0
                # Optional terrain heightmap used to displace the planet.
                terr = scene.get("terrain", {}) if isinstance(scene.get("terrain", {}), dict) else {}
                terr_hm = None
                terr_hs = 0.0
                terr_hb = 0.5
                terr_shade = 0.0
                try:
                    if bool(terr.get("enabled", False)) and bool(terr.get("planet_enabled", False)):
                        terr_hm = _load_grayscale_heightmap(str(terr.get("heightmap_path", "") or ""))
                        terr_hs = float(terr.get("planet_height_scale", 0.0)) * float(rs)
                        terr_hb = float(terr.get("planet_height_bias", 0.5))
                        terr_shade = float(terr.get("planet_shade_strength", 0.0))
                except Exception:
                    terr_hm = None
                _draw_planet_and_atmosphere(
                    planet_r=float(_PLANET_DRAW_R) * float(rs),
                    atmosphere_r=float(_ATMOSPHERE_R) * float(rs),
                    light_dir=sun_dir,
                    paint_tex=_NODE_PAINT_TEX,
                    paint_strength=float(max(0.0, paint_strength)),
                    terrain_heightmap=terr_hm,
                    terrain_height_scale=float(max(0.0, terr_hs)),
                    terrain_height_bias=float(max(0.0, min(1.0, terr_hb))),
                    terrain_shade_strength=float(max(0.0, terr_shade)),
                )

                # Sea surface: draw a simple sphere and rely on depth testing
                # against the planet/terrain to automatically fill low regions.
                try:
                    sea = scene.get("sea", {}) if isinstance(scene.get("sea", {}), dict) else {}
                    if bool(sea.get("enabled", False)):
                        base_r = float(_PLANET_DRAW_R) * float(rs)
                        use_hm_level = bool(sea.get("use_heightmap_level", True))
                        lvl = float(sea.get("heightmap_level", 0.5))
                        lvl = float(max(0.0, min(1.0, lvl)))
                        r_off = float(sea.get("radius_offset", 0.0)) * float(rs)
                        if not np.isfinite(r_off):
                            r_off = 0.0

                        # If terrain is available, interpret `heightmap_level` in the
                        # same space as the displacement: rr = base + (h01 - bias) * scale.
                        if (
                            use_hm_level
                            and terr_hm is not None
                            and isinstance(terr_hm, np.ndarray)
                            and terr_hm.size > 0
                            and float(terr_hs) > 0.0
                        ):
                            sea_r = base_r + (lvl - float(terr_hb)) * float(terr_hs)
                        else:
                            sea_r = base_r + r_off

                        col = sea.get("color", (0.08, 0.26, 0.52))
                        a = float(sea.get("alpha", 0.55))
                        dw = bool(sea.get("depth_write", False))
                        _draw_sea_sphere(
                            sea_r=float(sea_r),
                            color=_safe_color3(col, (0.08, 0.26, 0.52)),
                            alpha=float(a),
                            depth_write=bool(dw),
                        )
                except Exception:
                    pass

                # Limb haze: translucent atmosphere shell (shader-based).
                try:
                    shell = scene.get("atmosphere_shell", {}) if isinstance(scene.get("atmosphere_shell", {}), dict) else {}
                    if bool(shell.get("enabled", True)) and _PLANET_VBO_POS is not None and int(_PLANET_VBO_COUNT) > 0:
                        rad = float(_PLANET_DRAW_R) * float(rs) * float(shell.get("radius_mult", 1.04))
                        col = shell.get("color", (0.55, 0.75, 1.0))
                        a = float(shell.get("alpha", 0.28))
                        pwr = float(shell.get("power", 2.2))

                        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
                        if not depth_was_enabled:
                            glEnable(GL_DEPTH_TEST)
                        glDepthMask(False)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glUseProgram(atmo_prog)
                        if atmo_u_radius is not None and int(atmo_u_radius) >= 0:
                            glUniform1f(int(atmo_u_radius), float(rad))
                        if atmo_u_color is not None and int(atmo_u_color) >= 0:
                            glUniform3f(int(atmo_u_color), float(col[0]), float(col[1]), float(col[2]))
                        if atmo_u_alpha is not None and int(atmo_u_alpha) >= 0:
                            glUniform1f(int(atmo_u_alpha), float(max(0.0, min(1.0, a))))
                        if atmo_u_power is not None and int(atmo_u_power) >= 0:
                            glUniform1f(int(atmo_u_power), float(max(0.25, pwr)))

                        glBindBuffer(GL_ARRAY_BUFFER, int(_PLANET_VBO_POS))
                        glEnableVertexAttribArray(atmo_a_pos)
                        glVertexAttribPointer(atmo_a_pos, 3, GL_FLOAT, False, 0, None)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        glDrawArrays(GL_TRIANGLES, 0, int(_PLANET_VBO_COUNT))
                        glDisableVertexAttribArray(atmo_a_pos)

                        glUseProgram(0)
                        glDisable(GL_BLEND)
                        glDepthMask(True)
                        if not depth_was_enabled:
                            glDisable(GL_DEPTH_TEST)
                except Exception:
                    pass

            # Prepare buffers (float32) for shader draw.
            n_draw = 0
            if n_active_frame:
                if d == 3:
                    # Normalize for render-time parity (does not mutate sim state).
                    np.copyto(pos_f32[:n_active_frame, :], pos_front[:, :3], casting="unsafe")
                else:
                    # Project D->3 without allocating per frame.
                    np.copyto(pos_work_f32[:n_active_frame, :], pos_front[:, :], casting="unsafe")
                    pos_work_f32[:n_active_frame, :] -= proj_mean
                    pos_f32[:n_active_frame, :] = pos_work_f32[:n_active_frame, :] @ proj_mat

                # Display-space adjustments:
                # - map modes: normalize to unit sphere then project to 2D
                # - ship mode: show an N-D->3D projection normalized onto the sphere
                # - other modes: keep legacy unit-sphere normalization for consistent look
                if _is_map_proj_mode(mode_now):
                    norms = np.linalg.norm(pos_f32[:n_active_frame, :], axis=1, keepdims=True)
                    np.maximum(norms, 1e-12, out=norms)
                    pos_f32[:n_active_frame, :] /= norms
                    # Preserve unit directions for planet paint (before 2D map projection).
                    pos_unit_f32[:n_active_frame, :] = pos_f32[:n_active_frame, :]
                    _apply_map_projection_np(pos_f32[:n_active_frame, :], mode_now)
                elif _is_ship_proj_mode(mode_now):
                    norms = np.linalg.norm(pos_f32[:n_active_frame, :], axis=1, keepdims=True)
                    np.maximum(norms, 1e-12, out=norms)
                    pos_f32[:n_active_frame, :] /= norms
                    pos_unit_f32[:n_active_frame, :] = pos_f32[:n_active_frame, :]
                    # Render-only scaling: keep physics on the unit sphere, but draw the ship world scaled.
                    try:
                        rs = _safe_render_scale(params_menu.get("render_scale", 10.0))
                    except Exception:
                        rs = 10.0
                    pos_f32[:n_active_frame, :] *= float(rs)
                else:
                    norms = np.linalg.norm(pos_f32[:n_active_frame, :], axis=1, keepdims=True)
                    np.maximum(norms, 1e-12, out=norms)
                    pos_f32[:n_active_frame, :] /= norms
                    pos_unit_f32[:n_active_frame, :] = pos_f32[:n_active_frame, :]

            # Reticle focus (ship view only): update once per frame after positions are in ship world.
            try:
                if is_ship and n_active_frame > 0:
                    eye_v = np.asarray(eye, dtype=np.float32)
                    now_s = pygame.time.get_ticks() * 0.001
                    view_dir_cam = np.asarray(center, dtype=np.float32) - eye_v
                    vn = float(np.linalg.norm(view_dir_cam))
                    if vn > 1e-6:
                        view_dir_cam = (view_dir_cam / vn).astype(np.float32, copy=False)
                    # Scale radii to match ship-world scaling (pos_f32 already includes render_scale).
                    rs_now = float(_safe_render_scale(params_menu.get("render_scale", 10.0)))
                    radii_world = np.asarray(radii_live[:n_active_frame], dtype=np.float32) * float(rs_now)

                    nodes_w = np.asarray(pos_f32[:n_active_frame, :], dtype=np.float32)

                    # Auto targeting is anything other than manual.
                    try:
                        targeting_mode = int(getattr(loadout, "targeting", 0) or 0)
                    except Exception:
                        targeting_mode = 0
                    auto_targeting = bool(targeting_mode != 0)
                    if menu is not None:
                        try:
                            menu._targeting_active = bool(targeting_mode != 0)
                        except Exception:
                            pass

                    # Decide which weapon family this reticle's LOS status represents.
                    # Keep it lightweight: use Weapon Slot 1 as "primary".
                    primary_weapon_type = ""
                    try:
                        resolved_primary = weapons_structs.resolve_virtual_weapon(loadout, 1)
                        if resolved_primary:
                            primary_weapon_type = str(resolved_primary[0][1])
                        else:
                            resolved_primary = weapons_structs.resolve_virtual_weapon(loadout, 2)
                            if resolved_primary:
                                primary_weapon_type = str(resolved_primary[0][1])
                    except Exception:
                        primary_weapon_type = ""

                    # Weapon-independent aim policy (defaults: guided weapons latch).
                    weapon_independent_aim = False
                    weapon_off_boresight_deg = float(_WEAPON_DEFAULT_OFF_BORESIGHT_DEG)
                    try:
                        if primary_weapon_type:
                            cfg_primary = weapon_runtime.get_weapon_type_stats(weap_rt.stats, str(primary_weapon_type))
                            guidance = cfg_primary.get("guidance") if isinstance(cfg_primary, dict) else None
                            tags = cfg_primary.get("mechanics_tags") if isinstance(cfg_primary, dict) else None
                            if isinstance(guidance, dict) and ("independent_aim" in guidance):
                                weapon_independent_aim = bool(guidance.get("independent_aim"))
                            else:
                                weapon_independent_aim = bool(isinstance(tags, list) and ("guided" in tags))
                            if isinstance(guidance, dict) and (guidance.get("off_boresight_deg") is not None):
                                weapon_off_boresight_deg = float(guidance.get("off_boresight_deg"))
                    except Exception:
                        weapon_independent_aim = False
                        weapon_off_boresight_deg = float(_WEAPON_DEFAULT_OFF_BORESIGHT_DEG)

                    st_prev = target_sys.get_reticle(reticle_id)
                    ret_dir = np.asarray(st_prev.view_dir, dtype=np.float32).reshape((3,))
                    if float(np.linalg.norm(ret_dir)) <= 1e-6:
                        ret_dir = np.asarray(view_dir_cam, dtype=np.float32)

                    tracked_id = 0
                    tracked_pos = None
                    centered_on_track = False

                    def _df(o_df: np.ndarray, d_df: np.ndarray, tmax_df: float):
                        hm_np = getattr(menu.flight_cam, "terrain_heightmap", None) if menu is not None else None
                        hm_scale = getattr(menu.flight_cam, "terrain_height_scale", None) if menu is not None else None
                        hm_bias = getattr(menu.flight_cam, "terrain_height_bias", None) if menu is not None else None
                        # LOS depth finder is always the laser probe.
                        hit = _RETICLE_DEPTH_FINDER.probe_laser_impact(
                            weap_rt=weap_rt,
                            ray_origin=o_df,
                            ray_dir=d_df,
                            max_dist=float(tmax_df),
                            planet_surface_r=(getattr(menu.flight_cam, "planet_surface_r", None) if menu is not None else None),
                            gravity_g=(getattr(menu.flight_cam, "gravity_g", None) if menu is not None else None),
                            terrain_heightmap=hm_np,
                            terrain_height_scale=hm_scale,
                            terrain_height_bias=hm_bias,
                            nodes_pos=nodes_w,
                            nodes_radius=radii_world,
                        )
                        return hit.point if hit.valid else None

                    # Electronic tracking: choose the node closest to screen center (max dot).
                    if auto_targeting and nodes_w.size:
                        v = nodes_w - eye_v.reshape((1, 3))
                        dist = np.linalg.norm(v, axis=1)
                        dist = np.maximum(dist, 1e-6)
                        vhat = (v / dist.reshape((-1, 1))).astype(np.float32, copy=False)
                        dots = (vhat @ view_dir_cam.reshape((3, 1))).reshape((-1,))
                        mask = dots > 0.0
                        if bool(np.any(mask)):
                            dmax = float(np.max(dots[mask]))
                            cand = np.where((dots >= (dmax - 1e-5)) & mask)[0]
                            if cand.size:
                                best_i = int(cand[np.argmin(dist[cand])])
                            else:
                                best_i = int(np.argmax(dots))
                            tracked_id = int(best_i + 1)
                            tp = nodes_w[best_i]
                            tracked_pos = (float(tp[0]), float(tp[1]), float(tp[2]))

                    # Drift reticle direction toward tracked target (rate-limited).
                    if auto_targeting and tracked_pos is not None:
                        desired = np.asarray(tracked_pos, dtype=np.float32).reshape((3,)) - eye_v
                        dn = float(np.linalg.norm(desired))
                        if dn > 1e-6:
                            desired = (desired / dn).astype(np.float32, copy=False)
                            cur = (ret_dir / float(max(1e-6, np.linalg.norm(ret_dir)))).astype(np.float32, copy=False)
                            dot_cd = float(np.clip(float(np.dot(cur, desired)), -1.0, 1.0))
                            ang = float(math.acos(dot_cd))
                            max_ang = float(_RETICLE_AUTO_SLEW_DEG_PER_S) * (math.pi / 180.0) * float(max(0.0, frame_dt_s))
                            if ang <= 1e-6 or max_ang <= 1e-9:
                                ret_dir = desired
                            else:
                                t = float(min(1.0, max_ang / ang))
                                s = float(math.sin(ang))
                                a = float(math.sin((1.0 - t) * ang) / s)
                                b = float(math.sin(t * ang) / s)
                                ret_dir = (a * cur + b * desired).astype(np.float32, copy=False)

                            eps_ang = float(_RETICLE_AUTO_CENTER_EPS_DEG) * (math.pi / 180.0)
                            centered_on_track = bool(ang <= eps_ang)
                    else:
                        # Manual targeting: reticle direction can be offset from the camera via bindable reticle-look.
                        ret_dir = np.asarray(view_dir_cam, dtype=np.float32)
                        try:
                            if menu is not None:
                                ry = float(getattr(menu, "reticle_yaw", 0.0))
                                rp = float(getattr(menu, "reticle_pitch", 0.0))
                            else:
                                ry, rp = 0.0, 0.0
                            if abs(float(ry)) > 1e-9 or abs(float(rp)) > 1e-9:
                                up_rad = np.asarray(menu.flight_cam.planet_up(), dtype=np.float32) if menu is not None else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                                right_axis = np.cross(up_rad, ret_dir).astype(np.float32, copy=False)
                                rn = float(np.linalg.norm(right_axis))
                                if rn > 1e-6:
                                    right_axis = right_axis / rn

                                # Local Rodrigues rotation.
                                def _rot(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
                                    a = axis.astype(np.float32, copy=False)
                                    an = float(np.linalg.norm(a))
                                    if not (an > 1e-6):
                                        return v
                                    a = a / an
                                    c = float(math.cos(float(ang)))
                                    s = float(math.sin(float(ang)))
                                    return (v * c + np.cross(a, v) * s + a * float(np.dot(a, v)) * (1.0 - c)).astype(np.float32, copy=False)

                                # Positive yaw rotates about radial up; pitch about camera-right.
                                ret_dir = _rot(ret_dir, up_rad, float(ry))
                                ret_dir = _rot(ret_dir, right_axis, float(rp))
                                dn = float(np.linalg.norm(ret_dir))
                                if dn > 1e-6:
                                    ret_dir = (ret_dir / dn).astype(np.float32, copy=False)
                        except Exception:
                            ret_dir = np.asarray(view_dir_cam, dtype=np.float32)

                    # LOS poll gate: only run the C depth finder periodically in auto mode.
                    prev_tracked = int(getattr(st_prev, "tracked_victim_id", 0) or 0)
                    prev_last = float(getattr(st_prev, "los_last_check_s", 0.0) or 0.0)
                    prev_primary = str(getattr(st_prev, "weapon_type_primary", "") or "")
                    if (int(tracked_id) != int(prev_tracked)) or (str(primary_weapon_type) != str(prev_primary)):
                        prev_last = 0.0

                    # Precompute focus so we can gate the expensive depth finder.
                    focus_pre = targeting_system.ReticleFocus(on_target=False, victim_id=0)
                    try:
                        if nodes_w is not None and radii_world is not None:
                            focus_pre = targeting_system.raycast_nodes(
                                ray_origin=eye_v,
                                ray_dir=ret_dir,
                                nodes_pos=np.asarray(nodes_w, dtype=np.float32),
                                nodes_radius=np.asarray(radii_world, dtype=np.float32),
                                t_max=float(z_far),
                            )
                    except Exception:
                        focus_pre = targeting_system.ReticleFocus(on_target=False, victim_id=0)

                    # Airplane-driven LOS probe gating.
                    # 0=off, 1=on, 2=lock_yellow (or higher), 3=lock_red (ready only).
                    try:
                        los_probe_mode = int(getattr(airplane_tuning, "los_probe_mode", 1) or 0)
                    except Exception:
                        los_probe_mode = 1

                    allow_probe = True
                    if los_probe_mode == 0:
                        allow_probe = False
                    elif los_probe_mode == 1:
                        allow_probe = True
                    elif los_probe_mode == 2:
                        # Yellow lock or higher.
                        if auto_targeting:
                            allow_probe = bool(centered_on_track)
                        else:
                            # Use the manual reticle animator once per frame.
                            try:
                                _stage_now = target_sys.reticle_stage(on_target=bool(focus_pre.on_target), now_s=float(now_s))
                            except Exception:
                                _stage_now = reticle_sprite.ReticleStage.IDLE
                            allow_probe = bool(_stage_now in (reticle_sprite.ReticleStage.LOCKED, reticle_sprite.ReticleStage.READY))
                    else:
                        # Red lock only (auto: requires confirmed LOS; manual: READY stage).
                        if auto_targeting:
                            allow_probe = bool(getattr(st_prev, "los_confirmed", False))
                        else:
                            try:
                                _stage_now = target_sys.reticle_stage(on_target=bool(focus_pre.on_target), now_s=float(now_s))
                            except Exception:
                                _stage_now = reticle_sprite.ReticleStage.IDLE
                            allow_probe = bool(_stage_now == reticle_sprite.ReticleStage.READY)

                    do_los_poll = bool(
                        auto_targeting
                        and allow_probe
                        and centered_on_track
                        and tracked_pos is not None
                        and ((now_s - prev_last) >= float(_RETICLE_LOS_POLL_S))
                    )

                    depth_finder_cb = None
                    if allow_probe and (not auto_targeting or do_los_poll):
                        depth_finder_cb = _df

                    st = target_sys.solve_reticle(
                        ray_origin=eye_v,
                        ray_dir=ret_dir,
                        t_max=float(z_far),
                        nodes_pos=nodes_w,
                        nodes_radius=radii_world,
                        depth_finder=depth_finder_cb,
                        focus_override=focus_pre,
                    )

                    # Preserve the last computed intercept between LOS polls.
                    # This keeps weapon aiming + telemetry stable even when we only probe periodically.
                    try:
                        if st.aim_point is None:
                            st.aim_point = getattr(st_prev, "aim_point", None)
                    except Exception:
                        pass

                    st.tracked_victim_id = int(tracked_id)
                    st.tracked_pos = tracked_pos
                    st.centered_on_track = bool(centered_on_track)

                    st.weapon_type_primary = str(primary_weapon_type)
                    st.weapon_independent_aim = bool(weapon_independent_aim)
                    st.weapon_off_boresight_deg = float(weapon_off_boresight_deg)

                    # LOS confirmation poll (depth finder vs tracked location).
                    los_ok = bool(getattr(st_prev, "los_confirmed", False))
                    los_last = float(prev_last)

                    # Persistent latch identity (weapon-independent aim only).
                    latched_tid = int(getattr(st_prev, "los_confirmed_target_id", 0) or 0)
                    latched_wt = str(getattr(st_prev, "los_confirmed_weapon_type", "") or "")

                    if do_los_poll:
                        los_last = float(now_s)
                        los_ok = False
                        if st.aim_point is not None:
                            ap = np.asarray(st.aim_point, dtype=np.float32)
                            tp = np.asarray(st.tracked_pos, dtype=np.float32)
                            dist_err = float(np.linalg.norm(ap - tp))
                            ridx = int(max(0, int(st.tracked_victim_id) - 1))
                            rad = float(radii_world[ridx]) if (0 <= ridx < int(radii_world.shape[0])) else 0.0
                            eps = max(float(_RETICLE_LOS_EPS_ABS), float(_RETICLE_LOS_EPS_MULT_RADIUS) * float(rad))
                            los_ok = bool(dist_err <= eps)

                    if auto_targeting:
                        # Strict: nose/reticle-aimed weapons must stay centered.
                        # Independent: allow latch if target remains within weapon cone.
                        if not bool(weapon_independent_aim):
                            if not (st.centered_on_track and st.tracked_pos is not None):
                                los_ok = False
                            latched_tid = 0
                            latched_wt = ""
                        else:
                            # If we just confirmed, latch to the tracked target at that instant.
                            if do_los_poll and bool(los_ok):
                                latched_tid = int(tracked_id)
                                latched_wt = str(primary_weapon_type)

                            # Maintain latch as long as the latched target stays within the off-boresight cone.
                            # This intentionally does NOT require the reticle to stay centered.
                            if int(latched_tid) > 0 and str(latched_wt) == str(primary_weapon_type):
                                within_cone = False
                                try:
                                    if menu is not None and 0 <= int(latched_tid) - 1 < int(nodes_w.shape[0]):
                                        sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                        bore = np.asarray(menu.flight_cam.basis()[2], dtype=np.float32)
                                        bn = float(np.linalg.norm(bore))
                                        if bn > 1e-6:
                                            bore = (bore / bn).astype(np.float32, copy=False)
                                        else:
                                            bore = np.asarray(view_dir_cam, dtype=np.float32)

                                        tgt = nodes_w[int(latched_tid) - 1]
                                        to_t = (np.asarray(tgt, dtype=np.float32) - sp).astype(np.float32, copy=False)
                                        tn = float(np.linalg.norm(to_t))
                                        if tn > 1e-6:
                                            to_t = (to_t / tn).astype(np.float32, copy=False)
                                            dot_bt = float(np.clip(float(np.dot(bore, to_t)), -1.0, 1.0))
                                            ang_bt = float(math.acos(dot_bt))
                                            within_cone = bool(ang_bt <= (float(weapon_off_boresight_deg) * (math.pi / 180.0)))
                                except Exception:
                                    within_cone = False

                                los_ok = bool(within_cone)
                                if not bool(los_ok):
                                    latched_tid = 0
                                    latched_wt = ""
                            else:
                                los_ok = False
                                latched_tid = 0
                                latched_wt = ""

                    st.los_confirmed = bool(los_ok)
                    st.los_last_check_s = float(los_last)
                    st.los_confirmed_target_id = int(latched_tid) if bool(los_ok) else 0
                    st.los_confirmed_weapon_type = str(latched_wt) if bool(los_ok) else ""

                    # Cache a stage for this frame so HUD rendering can reuse it without
                    # advancing the animator twice.
                    try:
                        if auto_targeting:
                            if bool(los_ok):
                                st.stage = reticle_sprite.ReticleStage.READY
                            elif bool(centered_on_track):
                                st.stage = reticle_sprite.ReticleStage.LOCKED
                            else:
                                st.stage = reticle_sprite.ReticleStage.IDLE
                        else:
                            st.stage = _stage_now if "_stage_now" in locals() else None
                    except Exception:
                        pass

                    target_sys.set_reticle(reticle_id=reticle_id, state=st)
                    target_focus = st.focus
                    target_view_dir = np.asarray(st.view_dir, dtype=np.float32)
                else:
                    target_focus = targeting_system.ReticleFocus(on_target=False, victim_id=0)
                    st = targeting_system.ReticleState(focus=target_focus, view_dir=(0.0, 0.0, 1.0), aim_point=None)
                    target_sys.set_reticle(reticle_id=reticle_id, state=st)
            except Exception:
                target_focus = targeting_system.ReticleFocus(on_target=False, victim_id=0)
                st = targeting_system.ReticleState(focus=target_focus, view_dir=(0.0, 0.0, 1.0), aim_point=None)
                target_sys.set_reticle(reticle_id=reticle_id, state=st)

            # Colors/sizes + minimap buffers must be updated every frame (not only on reticle exceptions).
            if n_active_frame:
                col_f32[:n_active_frame, :] = colors_live
                size_f32[:n_active_frame] = point_sizes

                # Update node paint (planet-wrapped trails) once per frame.
                try:
                    _update_node_paint(
                        scene=scene,
                        node_dirs_unit_f32=pos_unit_f32,
                        node_colors_f32=colors_live,
                        n_active=n_active_frame,
                        dt_s=frame_dt_s,
                    )
                except Exception:
                    pass

                # Prepare minimap positions.
                if d == 3:
                    np.copyto(pos_mm_f32[:n_active_frame, :], pos_front[:, :3], casting="unsafe")
                else:
                    np.copyto(pos_work_mm_f32[:n_active_frame, :], pos_front[:, :], casting="unsafe")
                    pos_work_mm_f32[:n_active_frame, :] -= proj_mm_mean
                    pos_mm_f32[:n_active_frame, :] = pos_work_mm_f32[:n_active_frame, :] @ proj_mm_mat

                norms_mm = np.linalg.norm(pos_mm_f32[:n_active_frame, :], axis=1, keepdims=True)
                np.maximum(norms_mm, 1e-12, out=norms_mm)
                pos_mm_f32[:n_active_frame, :] /= norms_mm
                if _is_map_proj_mode(minimap_mode):
                    _apply_map_projection_np(pos_mm_f32[:n_active_frame, :], minimap_mode)

                # Build minimap draw buffers: objects + self marker.
                n_draw = int(n_active_frame)
                pos_mm_draw_f32[:n_draw, :] = pos_mm_f32[:n_active_frame, :]
                # Use a dim, consistent color for minimap points (retro, readable).
                col_mm_draw_f32[:n_draw, :] = (0.85 * col_f32[:n_active_frame, :] + 0.15).astype(np.float32, copy=False)
                type_mm_draw_f32[:n_draw] = 0.0
                ang_mm_draw_f32[:n_draw] = 0.0

                # Self marker: position from camera eye direction, with heading from camera forward.
                self_pos = None
                self_ang = 0.0
                if menu is not None:
                    eye_v = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                    fwd_v = np.asarray(menu.flight_cam.basis()[2], dtype=np.float32)
                    en = float(np.linalg.norm(eye_v))
                    if en > 1e-6:
                        p0 = (eye_v / en).astype(np.float32, copy=False)
                        if _is_map_proj_mode(minimap_mode):
                            up = p0
                            fwd_t = fwd_v - up * float(np.dot(fwd_v, up))
                            fn = float(np.linalg.norm(fwd_t))
                            if fn > 1e-6:
                                fwd_t = fwd_t / fn
                                eps = 0.04
                                p1 = p0 + eps * fwd_t
                                p1n = float(np.linalg.norm(p1))
                                if p1n > 1e-6:
                                    p1 = p1 / p1n
                                    pts = np.vstack([p0, p1]).astype(np.float32, copy=False)
                                    _apply_map_projection_np(pts, minimap_mode)
                                    self_pos = pts[0]
                                    du = float(pts[1, 0] - pts[0, 0])
                                    dv = float(pts[1, 1] - pts[0, 1])
                                    self_ang = float(math.atan2(dv, du))
                        else:
                            self_pos = p0

                if self_pos is not None and n_draw + 1 <= pos_mm_draw_f32.shape[0]:
                    pos_mm_draw_f32[n_draw, :] = np.asarray(self_pos, dtype=np.float32)
                    col_mm_draw_f32[n_draw, :] = np.array([1.0, 0.95, 0.30], dtype=np.float32)
                    type_mm_draw_f32[n_draw] = 1.0
                    ang_mm_draw_f32[n_draw] = np.float32(self_ang)
                    n_draw += 1

            mvp = _gl_mvp_matrix_f32()

            # --- Balls (points) ---
            # Depth-test balls so they sit at the correct 3D radius.
            # Disable depth writes so blended edges don't punch holes.
            glDisable(GL_LIGHTING)
            depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
            if not depth_was_enabled:
                glEnable(GL_DEPTH_TEST)
            glDepthMask(False)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(ball_prog)
            if ball_u_mvp is not None and int(ball_u_mvp) >= 0:
                glUniformMatrix4fv(ball_u_mvp, 1, False, mvp)

            # Scene lighting + haze.
            try:
                sun = scene.get("sun", {}) if isinstance(scene.get("sun", {}), dict) else {}
                ldir = np.asarray(sun.get("direction"), dtype=np.float32)
                ln = float(np.linalg.norm(ldir))
                if ln > 1e-6:
                    ldir = (ldir / ln).astype(np.float32, copy=False)
                else:
                    ldir = np.array([-0.25, 0.35, 1.0], dtype=np.float32)
                scol = sun.get("color", (1.0, 0.98, 0.92))
                inten = float(sun.get("intensity", 1.0))
                lcol = np.array([float(scol[0]), float(scol[1]), float(scol[2])], dtype=np.float32) * float(max(0.0, inten))

                haze = scene.get("haze", {}) if isinstance(scene.get("haze", {}), dict) else {}
                k0 = float(haze.get("strength_at_ground", 1.0))
                k1 = float(haze.get("strength_at_top", 0.15))
                fog_strength = (1.0 - float(alt_frac)) * k0 + float(alt_frac) * k1
                fog_k = float(haze.get("fog_k", 0.018)) * float(max(0.0, fog_strength))
                # Depth-based fog scales with world units; in ship mode we often scale the
                # world up via render_scale. Counteract that so node colors remain readable.
                try:
                    rs_fog = float(_safe_render_scale(params_menu.get("render_scale", 10.0))) if _is_ship_proj_mode(mode_now) else 1.0
                except Exception:
                    rs_fog = 1.0
                fog_k = float(fog_k) / float(max(1e-6, rs_fog))

                if ball_u_light_dir is not None and int(ball_u_light_dir) >= 0:
                    glUniform3f(int(ball_u_light_dir), float(ldir[0]), float(ldir[1]), float(ldir[2]))
                if ball_u_light_color is not None and int(ball_u_light_color) >= 0:
                    glUniform3f(int(ball_u_light_color), float(lcol[0]), float(lcol[1]), float(lcol[2]))
                if ball_u_fog_color is not None and int(ball_u_fog_color) >= 0:
                    glUniform3f(int(ball_u_fog_color), float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]))
                if ball_u_fog_k is not None and int(ball_u_fog_k) >= 0:
                    glUniform1f(int(ball_u_fog_k), float(max(0.0, fog_k)))
            except Exception:
                pass

            glBindBuffer(GL_ARRAY_BUFFER, vbos["pt_pos"])
            glBufferData(GL_ARRAY_BUFFER, pos_f32[:n_active_frame].nbytes, pos_f32[:n_active_frame], GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(ball_a_pos)
            glVertexAttribPointer(ball_a_pos, 3, GL_FLOAT, False, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, vbos["pt_col"])
            glBufferData(GL_ARRAY_BUFFER, col_f32[:n_active_frame].nbytes, col_f32[:n_active_frame], GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(ball_a_color)
            glVertexAttribPointer(ball_a_color, 3, GL_FLOAT, False, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, vbos["pt_size"])
            glBufferData(GL_ARRAY_BUFFER, size_f32[:n_active_frame].nbytes, size_f32[:n_active_frame], GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(ball_a_size)
            glVertexAttribPointer(ball_a_size, 1, GL_FLOAT, False, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDrawArrays(GL_POINTS, 0, n_active_frame)

            glDisableVertexAttribArray(ball_a_pos)
            glDisableVertexAttribArray(ball_a_color)
            glDisableVertexAttribArray(ball_a_size)
            glUseProgram(0)
            glDisable(GL_BLEND)
            glDepthMask(True)
            if not depth_was_enabled:
                glDisable(GL_DEPTH_TEST)

            # --- Sticks (straight edges) ---
            if springs_live is not None and springs_live.size and draw_direct_edges:
                depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
                glDisable(GL_LIGHTING)
                if _is_ship_proj_mode(mode_now):
                    glEnable(GL_DEPTH_TEST)
                else:
                    glDisable(GL_DEPTH_TEST)

                i_idx = springs_live[:, 0].astype(np.int64, copy=False)
                j_idx = springs_live[:, 1].astype(np.int64, copy=False)
                # Defensive filtering: spring endpoints must be in-range for this frame.
                # (Prevents artifacts if we ever observe a transient inconsistent spring list.)
                valid = (i_idx >= 0) & (j_idx >= 0) & (i_idx < n_active_frame) & (j_idx < n_active_frame) & (i_idx != j_idx)
                if valid.size:
                    i_idx = i_idx[valid]
                    j_idx = j_idx[valid]
                m = int(i_idx.shape[0])
                if m:
                    if edge_verts.shape[0] < 2 * m:
                        edge_verts = np.empty((2 * m, 3), dtype=np.float32)
                    edge_verts[0 : 2 * m : 2, :] = pos_f32[i_idx]
                    edge_verts[1 : 2 * m : 2, :] = pos_f32[j_idx]

                    glUseProgram(line_prog)
                    if line_u_mvp is not None and int(line_u_mvp) >= 0:
                        glUniformMatrix4fv(line_u_mvp, 1, False, mvp)
                    glUniform4f(line_u_color, 0.8, 0.85, 0.95, 0.85)

                    # Match haze to point shader.
                    try:
                        haze = scene.get("haze", {}) if isinstance(scene.get("haze", {}), dict) else {}
                        k0 = float(haze.get("strength_at_ground", 1.0))
                        k1 = float(haze.get("strength_at_top", 0.15))
                        fog_strength = (1.0 - float(alt_frac)) * k0 + float(alt_frac) * k1
                        fog_k = float(haze.get("fog_k", 0.018)) * float(max(0.0, fog_strength))
                        try:
                            rs_fog = float(_safe_render_scale(params_menu.get("render_scale", 10.0))) if _is_ship_proj_mode(mode_now) else 1.0
                        except Exception:
                            rs_fog = 1.0
                        fog_k = float(fog_k) / float(max(1e-6, rs_fog))
                        if line_u_fog_color is not None and int(line_u_fog_color) >= 0:
                            glUniform3f(int(line_u_fog_color), float(sky_rgb[0]), float(sky_rgb[1]), float(sky_rgb[2]))
                        if line_u_fog_k is not None and int(line_u_fog_k) >= 0:
                            glUniform1f(int(line_u_fog_k), float(max(0.0, fog_k)))
                    except Exception:
                        pass

                    glBindBuffer(GL_ARRAY_BUFFER, vbos["ln_pos"])
                    glBufferData(GL_ARRAY_BUFFER, edge_verts[: 2 * m].nbytes, edge_verts[: 2 * m], GL_DYNAMIC_DRAW)
                    glEnableVertexAttribArray(line_a_pos)
                    glVertexAttribPointer(line_a_pos, 3, GL_FLOAT, False, 0, None)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)

                    glLineWidth(2.0)
                    glDrawArrays(GL_LINES, 0, 2 * m)

                    glDisableVertexAttribArray(line_a_pos)
                    glUseProgram(0)

                if depth_was_enabled:
                    glEnable(GL_DEPTH_TEST)

            # Draw edges AFTER points so they appear on top.
            if springs_live is not None and springs_live.size:
                depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
                glDisable(GL_LIGHTING)
                glDisable(GL_DEPTH_TEST)
                if draw_edges and (not _is_map_proj_mode(mode_now)) and (not _is_ship_proj_mode(mode_now)):
                    # Use the native-D positions for geodesic slerp, then project for display.
                    # (pos_f32 is already projected and normalized, but slerp in 3D is not
                    # equivalent to slerp in the embedding when D>3.)
                    if d == 3:
                        _draw_geodesic_edges(pos_f32[:n_active_frame, :], springs_live, np.zeros((1, 3), dtype=np.float32), np.eye(3, dtype=np.float32), samples=24, alpha=0.75)
                    else:
                        _draw_geodesic_edges(np.asarray(pos_front, dtype=np.float32), springs_live, proj_mean, proj_mat, samples=24, alpha=0.75)
                if depth_was_enabled:
                    glEnable(GL_DEPTH_TEST)

            # Weapon debug splines from bullet-sim DLL (world-space).
            try:
                now_s = pygame.time.get_ticks() * 0.001
                weapon_splines[:] = [s for s in weapon_splines if float(s.get("t_end", 0.0)) >= now_s]
                _draw_weapon_splines_world(splines=[s.get("pts", []) for s in weapon_splines])
            except Exception:
                pass

            # Minimap feed mixer:
            # - default: classic minimap points
            # - override: camera feed when camera nose-weapon trigger is held
            minimap_override_tex = None
            try:
                mode_ship = str(params_menu.get("proj_mode", proj_mode) or "pca")
                in_flight = bool(menu is not None and menu.flight_active(mode_ship))
                if in_flight and menu is not None:
                    best_prio = None
                    best_cfg = None
                    best_origin = None
                    best_dir = None
                    for slot, is_active in ((1, bool(f1_active)), (2, bool(f2_active))):
                        if not is_active:
                            continue
                        resolved = weapons_structs.resolve_virtual_weapon(loadout, int(slot))
                        for source_name, weapon_type in resolved:
                            if str(source_name) != "nose gun":
                                continue
                            if str(weapon_type) != "camera":
                                continue
                            cfg = weapon_runtime.get_weapon_type_stats(weap_rt.stats, str(weapon_type))
                            cam = cfg.get("camera") if isinstance(cfg, dict) else None
                            cam = cam if isinstance(cam, dict) else {}
                            try:
                                prio = int(cam.get("priority", 10))
                            except Exception:
                                prio = 10

                            sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                            sr, su, sf = menu.flight_cam.basis()
                            wo_t = targeting_system.compute_nose_gun_origin_world(
                                ship_pos=sp,
                                ship_right=np.asarray(sr, dtype=np.float32),
                                ship_up=np.asarray(su, dtype=np.float32),
                                ship_fwd=np.asarray(sf, dtype=np.float32),
                            )
                            wd_t = targeting_system.compute_weapon_dir_from_reticle(
                                weapon_origin=np.asarray(wo_t, dtype=np.float32),
                                fallback_view_dir=np.asarray(target_view_dir, dtype=np.float32),
                                reticle=target_sys.get_reticle(reticle_id),
                            )
                            if best_prio is None or int(prio) > int(best_prio):
                                best_prio = int(prio)
                                best_cfg = cfg
                                best_origin = wo_t
                                best_dir = wd_t

                    if best_cfg is not None and best_origin is not None and best_dir is not None:
                        # Match the minimap viewport size so texel density feels consistent.
                        mm_tw = int(max(140, 0.28 * float(width)))
                        mm_th = int(max(140, 0.28 * float(height)))
                        mm_aspect = float(mm_tw) / float(max(1, mm_th))
                        # Camera zoom controls (optical/FOV). Defaults to 1.0 if unbound.
                        try:
                            zoom_mul = float(getattr(menu, "camera_zoom_mul", 1.0))
                        except Exception:
                            zoom_mul = 1.0

                        try:
                            if joystick is not None:
                                cfg_zoom = joystick_menu.load_or_create_joystick_config("joystick.json")
                                set_name = _resolve_effective_control_set(menu)
                                z_in = _get_set_binding(cfg_zoom, set_name=set_name, group="camera", key="zoom_in")
                                z_out = _get_set_binding(cfg_zoom, set_name=set_name, group="camera", key="zoom_out")
                                axes_z, buttons_z, hats_z = joystick_menu._poll_joystick_snapshot(joystick)
                                _a_in, v_in = _binding_active(z_in, axes_z, buttons_z, hats_z)
                                _a_out, v_out = _binding_active(z_out, axes_z, buttons_z, hats_z)
                                dz = float(max(0.0, float(v_in)) - max(0.0, float(v_out)))
                                if abs(dz) > 1e-6:
                                    zoom_rate = 2.2
                                    zoom_mul *= float(math.exp(float(zoom_rate) * float(frame_dt_s) * float(dz)))
                        except Exception:
                            pass

                        zoom_mul = float(max(1.0, min(40.0, float(zoom_mul))))
                        try:
                            menu.camera_zoom_mul = float(zoom_mul)
                        except Exception:
                            pass

                        fov_y = _camera_fov_y_deg_from_weapon_cfg_zoomed(best_cfg, aspect=mm_aspect, zoom_mul=float(zoom_mul))

                        eye_cam = np.asarray(best_origin, dtype=np.float32).reshape((3,))
                        dir_cam = np.asarray(best_dir, dtype=np.float32).reshape((3,))
                        dn = float(np.linalg.norm(dir_cam))
                        if dn > 1e-6:
                            dir_cam = (dir_cam / dn).astype(np.float32, copy=False)
                        aim_pt = target_sys.get_reticle(reticle_id).aim_point
                        if aim_pt is not None:
                            center_cam = np.asarray(aim_pt, dtype=np.float32).reshape((3,))
                        else:
                            center_cam = (eye_cam + dir_cam * 2.0).astype(np.float32, copy=False)
                        upx, upy, upz = lookat_up_away_from_planet(eye=eye_cam, center=center_cam)
                        up_cam = np.asarray([upx, upy, upz], dtype=np.float32)

                        minimap_override_tex = _render_camera_feed_scene(
                            tex_w=int(mm_tw),
                            tex_h=int(mm_th),
                            eye=eye_cam,
                            center=center_cam,
                            up=up_cam,
                            fov_y_deg=float(fov_y),
                            z_near=float(max(1e-4, float(z_near) if "z_near" in locals() else 0.02)),
                            z_far=float(max(1.0, float(z_far) if "z_far" in locals() else 250.0)),
                            scene=scene,
                            sky_rgb=tuple(sky_rgb),
                            is_ship=True,
                            render_scale=float(_safe_render_scale(params_menu.get("render_scale", 10.0))),
                            sun_prog=int(sun_prog),
                            sun_u_color=sun_u_color,
                            sun_u_size_px=sun_u_size_px,
                            sun_a_pos=int(sun_a_pos),
                            sun_vbo_pos=int(vbos["sun_pos"]),
                            atmo_prog=int(atmo_prog),
                            atmo_u_radius=atmo_u_radius,
                            atmo_u_color=atmo_u_color,
                            atmo_u_alpha=atmo_u_alpha,
                            atmo_u_power=atmo_u_power,
                            atmo_a_pos=int(atmo_a_pos),
                            ball_prog=int(ball_prog),
                            ball_u_mvp=ball_u_mvp,
                            ball_u_light_dir=ball_u_light_dir,
                            ball_u_light_color=ball_u_light_color,
                            ball_u_fog_color=ball_u_fog_color,
                            ball_u_fog_k=ball_u_fog_k,
                            ball_a_pos=int(ball_a_pos),
                            ball_a_color=int(ball_a_color),
                            ball_a_size=int(ball_a_size),
                            vbo_pos=int(vbos["pt_pos"]),
                            vbo_col=int(vbos["pt_col"]),
                            vbo_size=int(vbos["pt_size"]),
                            pos_f32=np.asarray(pos_f32[:n_active_frame], dtype=np.float32),
                            col_f32=np.asarray(col_f32[:n_active_frame], dtype=np.float32),
                            size_f32=np.asarray(size_f32[:n_active_frame], dtype=np.float32),
                            n_active=int(n_active_frame),
                            line_prog=int(line_prog) if "line_prog" in locals() and line_prog is not None else None,
                            line_u_mvp=line_u_mvp,
                            line_u_color=line_u_color,
                            line_u_fog_color=line_u_fog_color,
                            line_u_fog_k=line_u_fog_k,
                            line_a_pos=line_a_pos,
                            line_vbo_pos=int(vbos["ln_pos"]) if "ln_pos" in vbos else None,
                            edge_verts_f32=(edge_verts if "edge_verts" in locals() else None),
                            springs_live=(springs_live if "springs_live" in locals() else None),
                            draw_direct_edges=bool(draw_direct_edges) if "draw_direct_edges" in locals() else False,
                            alt_frac=float(alt_frac) if "alt_frac" in locals() else None,
                            weapon_splines=(weapon_splines if "weapon_splines" in locals() else None),
                        )
            except Exception:
                minimap_override_tex = None

            # Ship cockpit instruments:
            # - In ship mode, render the minimap + attitude ball as thin panels mounted to the ship.
            #   They are NOT screen-space overlays, so view-look can look away from them.
            # - In other modes, keep the classic picture-in-picture minimap.
            if _is_ship_proj_mode(mode_now) and menu is not None:
                try:
                    eye_v = np.asarray(eye, dtype=np.float32)
                    # IMPORTANT: Build a stable, no-roll ship basis for cockpit panels.
                    # Using the camera quaternion basis can occasionally roll-flip 180° (a valid but
                    # discontinuous basis choice), which makes the two panels swap corners.
                    center_v = np.asarray(center, dtype=np.float32)
                    sf = (center_v - eye_v).astype(np.float32, copy=False)
                    n_sf = float(np.linalg.norm(sf))
                    if not np.isfinite(n_sf) or n_sf <= 1e-9:
                        sf = np.asarray(menu.flight_cam.basis()[2], dtype=np.float32)
                        n_sf = float(np.linalg.norm(sf))
                    if np.isfinite(n_sf) and n_sf > 1e-9:
                        sf = (sf / n_sf).astype(np.float32, copy=False)

                    up_rad = eye_v.astype(np.float32, copy=False)
                    n_up = float(np.linalg.norm(up_rad))
                    if np.isfinite(n_up) and n_up > 1e-9:
                        up_rad = (up_rad / n_up).astype(np.float32, copy=False)
                    else:
                        up_rad = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                    # Right-handed ship/camera basis (matches gluLookAt convention):
                    # right = forward x up, up = right x forward.
                    sr = np.cross(sf, up_rad).astype(np.float32, copy=False)
                    n_sr = float(np.linalg.norm(sr))
                    if not np.isfinite(n_sr) or n_sr <= 1e-9:
                        sr = np.asarray(menu.flight_cam.basis()[0], dtype=np.float32)
                        n_sr = float(np.linalg.norm(sr))
                    if np.isfinite(n_sr) and n_sr > 1e-9:
                        sr = (sr / n_sr).astype(np.float32, copy=False)

                    su = np.cross(sr, sf).astype(np.float32, copy=False)
                    n_su = float(np.linalg.norm(su))
                    if np.isfinite(n_su) and n_su > 1e-9:
                        su = (su / n_su).astype(np.float32, copy=False)

                    # Match the original 2D overlay layout at load by converting
                    # pixel sizes/positions into world-space offsets on a plane at distance `d_panel`.
                    W = float(max(1, int(width)))
                    H = float(max(1, int(height)))
                    pad = 12.0
                    # Panel distance should be screen-relative (HUD-like), not scaled with render_scale.
                    # Scaling this by render_scale pushes panels beyond the camera far plane at large
                    # render scales (e.g. 1000), making the minimap appear to disappear.
                    d_panel = 0.92

                    # Perspective: vertical FOV is the `fov_deg` used at init.
                    fov_y = float(fov_deg) * (math.pi / 180.0)
                    tan_y = float(math.tan(0.5 * fov_y))
                    aspect = W / H
                    tan_x = tan_y * aspect

                    # --- Minimap panel (top-right) ---
                    mm_w = float(max(140.0, 0.28 * W))
                    mm_h = float(max(140.0, 0.28 * H))
                    mm_cx = W - pad - 0.5 * mm_w
                    mm_cy = H - pad - 0.5 * mm_h
                    mm_ndc_x = (2.0 * mm_cx / W) - 1.0
                    mm_ndc_y = (2.0 * mm_cy / H) - 1.0
                    mm_off_r = mm_ndc_x * tan_x * d_panel
                    mm_off_u = mm_ndc_y * tan_y * d_panel
                    mm_half_w = (mm_w / W) * tan_x * d_panel
                    mm_half_h = (mm_h / H) * tan_y * d_panel

                    # --- Attitude ball panel (bottom-left) ---
                    gb_w = float(max(140.0, 0.18 * W))
                    gb_h = gb_w
                    gb_cx = pad + 0.5 * gb_w
                    gb_cy = pad + 0.5 * gb_h
                    gb_ndc_x = (2.0 * gb_cx / W) - 1.0
                    gb_ndc_y = (2.0 * gb_cy / H) - 1.0
                    gb_off_r = gb_ndc_x * tan_x * d_panel
                    gb_off_u = gb_ndc_y * tan_y * d_panel
                    gb_half_w = (gb_w / W) * tan_x * d_panel
                    gb_half_h = (gb_h / H) * tan_y * d_panel

                    # --- Ground-track arc panel (bottom-right, same size as attitude ball) ---
                    gt_w = gb_w
                    # Slightly shorter panel; also lift it up so it doesn't hug the bottom.
                    gt_h = gb_h * 0.80
                    gt_cx = (W - pad) - 0.5 * gt_w
                    gt_cy = (pad + 0.12 * gb_h) + 0.5 * gt_h
                    gt_ndc_x = (2.0 * gt_cx / W) - 1.0
                    gt_ndc_y = (2.0 * gt_cy / H) - 1.0
                    gt_off_r = gt_ndc_x * tan_x * d_panel
                    gt_off_u = gt_ndc_y * tan_y * d_panel
                    gt_half_w = (gt_w / W) * tan_x * d_panel
                    gt_half_h = (gt_h / H) * tan_y * d_panel

                    ship_hdg = float(_ship_heading_rad(pos=np.asarray(menu.flight_cam.pos, dtype=np.float32), fwd=np.asarray(menu.flight_cam.basis()[2], dtype=np.float32)))
                    ground_hdg = float(
                        _ship_heading_rad(
                            pos=np.asarray(menu.flight_cam.pos, dtype=np.float32),
                            fwd=np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32),
                        )
                    )

                    # Ground speed proxy:
                    # - inertial mode: use tangent velocity magnitude
                    # - non-inertial surface mode: `vel` may be ~0; fall back to camera speed.
                    try:
                        v_tmp = np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                        p_tmp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                        pn = float(np.linalg.norm(p_tmp))
                        if pn > 1e-6:
                            up_tmp = p_tmp / pn
                            v_t = v_tmp - up_tmp * float(np.dot(v_tmp, up_tmp))
                            gs_val = float(np.linalg.norm(v_t))
                        else:
                            gs_val = float(np.linalg.norm(v_tmp))
                    except Exception:
                        gs_val = 0.0
                    if not np.isfinite(gs_val) or gs_val <= 1e-6:
                        try:
                            gs_val = float(getattr(menu.flight_cam, "speed", 0.0))
                        except Exception:
                            gs_val = 0.0

                    _draw_ship_panel_minimap(
                        eye=eye_v,
                        ship_right=np.asarray(sr, dtype=np.float32),
                        ship_up=np.asarray(su, dtype=np.float32),
                        ship_fwd=np.asarray(sf, dtype=np.float32),
                        pos_mm_f32=pos_mm_draw_f32,
                        col_mm_f32=col_mm_draw_f32,
                        type_mm_f32=type_mm_draw_f32,
                        ang_mm_f32=ang_mm_draw_f32,
                        n_draw=int(n_draw),
                        dist=float(d_panel),
                        off_r=float(mm_off_r),
                        off_u=float(mm_off_u),
                        half_w=float(mm_half_w),
                        half_h=float(mm_half_h),
                        override_tex=(None if minimap_override_tex is None else int(minimap_override_tex)),
                    )
                    _draw_ship_panel_attitude_ball(
                        eye=eye_v,
                        ship_right=np.asarray(sr, dtype=np.float32),
                        ship_up=np.asarray(su, dtype=np.float32),
                        ship_fwd=np.asarray(sf, dtype=np.float32),
                        dist=float(d_panel),
                        off_r=float(gb_off_r),
                        off_u=float(gb_off_u),
                        half_w=float(gb_half_w),
                        half_h=float(gb_half_h),
                        altitude=float(menu.flight_cam.altitude()),
                        altitude_max=float(max(1e-6, float(menu.flight_cam.flight_r_max) - float(menu.flight_cam.planet_surface_r))),
                        ship_heading_rad=float(ship_hdg),
                        ground_heading_rad=float(ground_hdg),
                        airspeed=float(np.linalg.norm(np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32))),
                        groundspeed=float(
                            np.linalg.norm(
                                np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                                - (
                                    (np.asarray(menu.flight_cam.pos, dtype=np.float32) / max(1e-9, float(np.linalg.norm(np.asarray(menu.flight_cam.pos, dtype=np.float32)))))
                                    * float(
                                        np.dot(
                                            np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32),
                                            (np.asarray(menu.flight_cam.pos, dtype=np.float32) / max(1e-9, float(np.linalg.norm(np.asarray(menu.flight_cam.pos, dtype=np.float32))))),
                                        )
                                    )
                                )
                            )
                        ),
                        throttle=float(getattr(menu.flight_cam, "engine_throttle", 0.0)),
                    )

                    # Bottom-right ground-track / terrain arc instrument.
                    try:
                        try:
                            gravity_g_val = float(getattr(menu.flight_cam, "gravity_g", 0.0))
                        except Exception:
                            gravity_g_val = 0.0
                        try:
                            surf_fn = getattr(menu.flight_cam, "terrain_surface_r_at_pos", None)
                            if callable(surf_fn):
                                surface_r_val = float(surf_fn(np.asarray(menu.flight_cam.pos, dtype=np.float32)))
                            else:
                                surface_r_val = float(getattr(menu.flight_cam, "planet_surface_r", 1.0))
                        except Exception:
                            surface_r_val = float(getattr(menu.flight_cam, "planet_surface_r", 1.0))

                        _draw_ship_panel_groundtrack_arc(
                            eye=eye_v,
                            ship_right=np.asarray(sr, dtype=np.float32),
                            ship_up=np.asarray(su, dtype=np.float32),
                            ship_fwd=np.asarray(sf, dtype=np.float32),
                            dist=float(d_panel),
                            off_r=float(gt_off_r),
                            off_u=float(gt_off_u),
                            half_w=float(gt_half_w),
                            half_h=float(gt_half_h),
                            pos_world=np.asarray(menu.flight_cam.pos, dtype=np.float32),
                            vel_world=np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32),
                            ship_heading_rad=float(ship_hdg),
                            ground_heading_rad=float(ground_hdg),
                            camera_pitch_rad=float(getattr(menu.flight_cam, "pitch", 0.0)) + float(getattr(menu, "view_pitch", 0.0)),
                            altitude=float(menu.flight_cam.altitude()),
                            atmosphere_thickness=float(_ATMOSPHERE_ALT_MAX) * float(getattr(menu, "render_scale", 1.0)),
                            groundspeed=float(gs_val),
                            gravity_g=float(gravity_g_val),
                            surface_r=float(surface_r_val),
                            terrain_cfg=(scene.get("terrain", {}) if isinstance(scene.get("terrain", {}), dict) else {}),
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                try:
                    _draw_minimap_overlay(
                        width=int(width),
                        height=int(height),
                        pos_mm_f32=pos_mm_draw_f32,
                        col_mm_f32=col_mm_draw_f32,
                        type_mm_f32=type_mm_draw_f32,
                        ang_mm_f32=ang_mm_draw_f32,
                        n_draw=int(n_draw),
                        minimap_mode=str(minimap_mode),
                        mm_prog=int(mm_prog),
                        mm_u_point_px=mm_u_point_px,
                        mm_u_self_px=mm_u_self_px,
                        mm_a_pos=int(mm_a_pos),
                        mm_a_color=int(mm_a_color),
                        mm_a_type=int(mm_a_type),
                        mm_a_ang=int(mm_a_ang),
                        mm_vbo_pos=int(vbos["mm_pos"]),
                        mm_vbo_col=int(vbos["mm_col"]),
                        mm_vbo_type=int(vbos["mm_type"]),
                        mm_vbo_ang=int(vbos["mm_ang"]),
                        override_tex=(None if minimap_override_tex is None else int(minimap_override_tex)),
                    )
                except Exception:
                    pass

            # Draw HUD last so it remains visible.
            try:
                proj_now = str(params_menu.get("proj_mode", "pca") or "pca")
                in_flight = bool(menu is not None and menu.flight_active(proj_now))
                if in_flight:
                    try:
                        now_s = pygame.time.get_ticks() * 0.001
                        st = target_sys.get_reticle(reticle_id)
                        try:
                            targeting_mode = int(getattr(loadout, "targeting", 0) or 0)
                        except Exception:
                            targeting_mode = 0
                        auto_targeting = bool(targeting_mode != 0)

                        if auto_targeting:
                            if bool(getattr(st, "los_confirmed", False)):
                                stage = reticle_sprite.ReticleStage.READY
                            elif bool(getattr(st, "centered_on_track", False)):
                                stage = reticle_sprite.ReticleStage.LOCKED
                            else:
                                stage = reticle_sprite.ReticleStage.IDLE
                        else:
                            # Manual targeting: the reticle animator may have already been advanced
                            # earlier in the frame for LOS probe gating; reuse that stage if present.
                            try:
                                stage = getattr(st, "stage", None)
                            except Exception:
                                stage = None
                            if stage is None:
                                stage = target_sys.reticle_stage(on_target=bool(target_focus.on_target), now_s=float(now_s))

                        center_px = None
                        if auto_targeting and bool(is_ship):
                            try:
                                W = float(max(1, int(width)))
                                H = float(max(1, int(height)))
                                fov_y = float(fov_deg) * (math.pi / 180.0)
                                tan_y = float(math.tan(0.5 * fov_y))
                                aspect = W / H
                                tan_x = tan_y * aspect

                                eye_v = np.asarray(eye, dtype=np.float32)
                                center_v = np.asarray(center, dtype=np.float32)
                                sf = (center_v - eye_v).astype(np.float32, copy=False)
                                n_sf = float(np.linalg.norm(sf))
                                if n_sf > 1e-6:
                                    sf = (sf / n_sf).astype(np.float32, copy=False)
                                else:
                                    sf = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                                upx, upy, upz = lookat_up_away_from_planet(eye=eye_v, center=center)
                                up_rad = np.array([float(upx), float(upy), float(upz)], dtype=np.float32)
                                n_up = float(np.linalg.norm(up_rad))
                                if n_up > 1e-6:
                                    up_rad = (up_rad / n_up).astype(np.float32, copy=False)
                                else:
                                    up_rad = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                                sr = np.cross(sf, up_rad).astype(np.float32, copy=False)
                                n_sr = float(np.linalg.norm(sr))
                                if n_sr > 1e-6:
                                    sr = (sr / n_sr).astype(np.float32, copy=False)
                                su = np.cross(sr, sf).astype(np.float32, copy=False)
                                n_su = float(np.linalg.norm(su))
                                if n_su > 1e-6:
                                    su = (su / n_su).astype(np.float32, copy=False)

                                view_dir_ret = np.asarray(st.view_dir, dtype=np.float32).reshape((3,))
                                dn = float(np.linalg.norm(view_dir_ret))
                                if dn > 1e-6:
                                    view_dir_ret = (view_dir_ret / dn).astype(np.float32, copy=False)

                                x = float(np.dot(view_dir_ret, sr))
                                y = float(np.dot(view_dir_ret, su))
                                z = float(np.dot(view_dir_ret, sf))
                                if z > 1e-6 and tan_x > 1e-9 and tan_y > 1e-9:
                                    ndc_x = (x / z) / tan_x
                                    ndc_y = (y / z) / tan_y
                                    ndc_x = float(max(-1.0, min(1.0, ndc_x)))
                                    ndc_y = float(max(-1.0, min(1.0, ndc_y)))
                                    cx = (ndc_x + 1.0) * 0.5 * W
                                    cy = (ndc_y + 1.0) * 0.5 * H
                                    center_px = (float(cx), float(cy))
                            except Exception:
                                center_px = None

                        target_sys.draw_center_reticle(width=int(width), height=int(height), stage=stage, center_px=center_px)

                        # Reticle telemetry: draw in screen-space near the reticle so it's visible during flight.
                        try:
                            sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                            sr_b, su_b, sf_b = menu.flight_cam.basis()
                            wo = targeting_system.compute_nose_gun_origin_world(
                                ship_pos=sp,
                                ship_right=np.asarray(sr_b, dtype=np.float32),
                                ship_up=np.asarray(su_b, dtype=np.float32),
                                ship_fwd=np.asarray(sf_b, dtype=np.float32),
                            )
                            tel = reticle_telemetry.build_reticle_telemetry(
                                weap_rt=weap_rt,
                                reticle=st,
                                weapon_origin=np.asarray(wo, dtype=np.float32),
                            )
                            cp = center_px
                            if cp is None:
                                cp = (0.5 * float(width), 0.5 * float(height))
                            reticle_telemetry.draw_reticle_telemetry(
                                font=font,
                                width=int(width),
                                height=int(height),
                                center_px=(float(cp[0]), float(cp[1])),
                                telemetry=tel,
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                if cfg_pip is not None:
                    try:
                        _draw_loadout_menu_overlay(font=pip_font, width=int(width), height=int(height), pip=cfg_pip)
                    except Exception:
                        pass
                if (not in_flight) or bool(flight_debug_hud_visible):
                    base.draw_hud(font=font, fps_render=clock.get_fps(), fps_phys=phys_fps, text_lines=hud_lines, text_lines_right=hud_right)
            except Exception:
                pass

            pygame.display.flip()
            clock.tick(60)
    finally:
        try:
            if menu is not None and hasattr(menu, "flight_cam"):
                menu.flight_cam.shutdown()
        except Exception:
            pass
        stop_evt.set()
        phys_thread.join(timeout=1.0)
        pygame.quit()


def _load_prephysics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    # guard against runaway magnitudes that can overflow downstream
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, 1e6)
    return mat / norms


def _sobol_like_sphere(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, dim)).astype(np.float32)
    return _normalize_rows(x)


def _relax_repulsion(pos: np.ndarray, iters: int = 12, step: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng()
    P = pos.copy()
    N, D = P.shape
    for _ in range(max(1, iters)):
        noise = 0.01 * rng.standard_normal(size=P.shape).astype(np.float32)
        for i in range(N):
            diff = P[i] - P
            dist2 = np.sum(diff * diff, axis=1, keepdims=True) + 1e-6
            inv = 1.0 / dist2
            force = (diff * inv).sum(axis=0)
            P[i] = P[i] + step * force + noise[i]
        P = _normalize_rows(P)
    return P


def _orbit_eye(center, radius, t):
    cx, cy, cz = center
    ang = 0.2 * t
    return (
        cx + radius * math.cos(ang),
        cy + radius * 0.35 + radius * 0.15 * math.sin(0.5 * ang),
        cz + radius * math.sin(ang),
    )


def _colors_from_charges(charges_np: np.ndarray, sat: float = 0.8, light: float = 0.55) -> np.ndarray:
    """Map signed charges to RGB (cool for negative, warm for positive)."""
    if charges_np.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    q_abs = np.abs(charges_np)
    q_max = float(np.max(q_abs)) if np.isfinite(q_abs).all() else 1.0
    if q_max < 1e-6 or not np.isfinite(q_max):
        q_max = 1.0
    t = 0.5 + 0.5 * np.clip(charges_np / q_max, -1.0, 1.0)
    # Hue blend: blue (~0.58) for negative to red (~0.02) for positive
    hue_neg = 0.58
    hue_pos = 0.02
    hue = hue_neg * (1.0 - t) + hue_pos * t
    rgb = _hsl_to_rgb_np(hue.astype(np.float32), sat, light)
    return rgb.astype(np.float32)


def _hsl_to_rgb_np(h, s, l):
    # vectorized HSL to RGB for numpy arrays
    c = (1 - np.abs(2 * l - 1)) * s
    h6 = (h * 6.0) % 6
    x = c * (1 - np.abs(h6 % 2 - 1))
    zeros = np.zeros_like(h)
    rgbp = np.stack(
        [
            np.where((0 <= h6) & (h6 < 1), c, np.where((1 <= h6) & (h6 < 2), x, np.where((4 <= h6) & (h6 < 5), x, zeros))),
            np.where((1 <= h6) & (h6 < 2), c, np.where((2 <= h6) & (h6 < 3), c, np.where((5 <= h6) & (h6 < 6), x, zeros))),
            np.where((2 <= h6) & (h6 < 3), c, np.where((3 <= h6) & (h6 < 4), x, np.where((0 <= h6) & (h6 < 1), x, zeros))),
        ],
        axis=-1,
    )
    m = l - 0.5 * c
    return rgbp + m[..., None]


def _to_torch_pinned_float(arr) -> "torch.Tensor":
    """Convert numpy or tensor to float CPU tensor, pinning only when enabled."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if isinstance(arr, torch.Tensor):
        t = arr.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_floating_point():
            t = t.float()
        return t.pin_memory() if PIN_ENABLED else t.contiguous()
    t_np = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return t_np.pin_memory() if PIN_ENABLED else t_np


def _skew_from_flat(flat: List[float], dim: int) -> np.ndarray:
    mat = np.zeros((dim, dim), dtype=np.float32)
    upper = min(len(flat), dim * (dim - 1) // 2)
    idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if idx >= upper:
                break
            mat[i, j] = flat[idx]
            mat[j, i] = -flat[idx]
            idx += 1
    return mat


def _random_skew(dim: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    M = rng.standard_normal(size=(dim, dim)).astype(np.float32)
    S = M - M.T
    return scale * S


def _slerp_strip(p_i: np.ndarray, p_j: np.ndarray, samples: int = 24) -> np.ndarray:
    """Great-circle strip between two unit vectors (works for any dim >=2)."""
    cos_t = float(np.clip(np.dot(p_i, p_j), -1.0, 1.0))
    theta = math.acos(cos_t)
    if theta < 1e-6 or not math.isfinite(theta):
        return np.vstack([p_i, p_j]).astype(np.float32)
    t_vals = np.linspace(0.0, 1.0, num=max(2, samples), dtype=np.float32)
    sin_t = math.sin(theta)
    a = np.sin((1.0 - t_vals) * theta) / sin_t
    b = np.sin(t_vals * theta) / sin_t
    strip = a[:, None] * p_i[None, :] + b[:, None] * p_j[None, :]
    strip = _normalize_rows(strip.astype(np.float32))
    return strip


def _expmap_np(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map on the unit sphere for numpy arrays."""
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8 or not np.isfinite(v_norm):
        return _normalize_rows(p[None, :])[0]
    w_hat = v / v_norm
    cos_t = math.cos(v_norm)
    sin_t = math.sin(v_norm)
    x = cos_t * p + sin_t * w_hat
    return _normalize_rows(x[None, :])[0]


def _tangent_basis(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return two orthonormal tangent vectors (second may be None if dim<3)."""
    D = p.shape[0]
    axes = np.eye(D, dtype=np.float32)
    # pick axis least aligned with p for stability
    align = np.abs(axes @ p)
    order = np.argsort(align)
    t1 = None
    for idx in order:
        candidate = axes[idx]
        t = candidate - np.dot(candidate, p) * p
        n = np.linalg.norm(t)
        if n > 1e-6:
            t1 = t / n
            break
    if t1 is None:
        return p, None
    t2 = None
    for idx in order:
        candidate = axes[idx]
        t = candidate - np.dot(candidate, p) * p - np.dot(candidate, t1) * t1
        n = np.linalg.norm(t)
        if n > 1e-6:
            t2 = t / n
            break
    return t1, t2


def _orbit_strip(p: np.ndarray, samples: int = 64, arc: float = 0.35) -> np.ndarray:
    """Small geodesic loop through p on the unit sphere (works for any dim>=2)."""
    D = p.shape[0]
    if D == 2:
        theta = np.linspace(0.0, 2.0 * math.pi, num=max(4, samples), endpoint=True, dtype=np.float32)
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return pts.astype(np.float32)
    t1, t2 = _tangent_basis(p)
    phi = np.linspace(0.0, 2.0 * math.pi, num=max(8, samples), endpoint=True, dtype=np.float32)
    pts = []
    for ang in phi:
        if t2 is not None:
            v = arc * (math.cos(ang) * t1 + math.sin(ang) * t2)
        else:
            v = arc * math.cos(ang) * t1
        pts.append(_expmap_np(p, v))
    return np.asarray(pts, dtype=np.float32)


def _draw_orbitals(pos_nd: np.ndarray, mean: np.ndarray, proj: np.ndarray, samples: int = 48, arc: float = 0.35, alpha: float = 0.08):
    if pos_nd.size == 0:
        return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
    for p in pos_nd:
        ring_nd = _orbit_strip(p, samples=samples, arc=arc)
        ring_3d = (ring_nd - mean) @ proj
        glColor4f(0.7, 0.85, 1.0, alpha)
        glBegin(GL_LINE_STRIP)
        for x, y, z in ring_3d:
            glVertex3f(float(x), float(y), float(z))
        glEnd()


def _tangent_great_circle(p: np.ndarray, v: np.ndarray, samples: int = 160) -> np.ndarray | None:
    """Great-circle path aligned with current tangent velocity; None if speed tiny."""
    v_tan = v - np.dot(v, p) * p
    n = np.linalg.norm(v_tan)
    if n < 1e-8 or not np.isfinite(n):
        return None
    u = v_tan / n
    t_vals = np.linspace(-math.pi, math.pi, num=max(16, samples), dtype=np.float32)
    pts = [_expmap_np(p, u * float(t)) for t in t_vals]
    return np.asarray(pts, dtype=np.float32)


def _draw_tangent_paths(pos_nd: np.ndarray, vel_nd: np.ndarray, mean: np.ndarray, proj: np.ndarray, samples: int = 160, alpha: float = 0.10, indices: list[int] | None = None):
    if pos_nd.size == 0 or vel_nd is None or vel_nd.size == 0:
        return
        if pos_nd.shape[0] != vel_nd.shape[0]:
            n = min(pos_nd.shape[0], vel_nd.shape[0])
            pos_nd = pos_nd[:n]
            vel_nd = vel_nd[:n]
    if pos_nd.shape[1] != vel_nd.shape[1]:
        d = min(pos_nd.shape[1], vel_nd.shape[1])
        pos_nd = pos_nd[:, :d]
        vel_nd = vel_nd[:, :d]
    if indices is None:
        idx_iter = range(pos_nd.shape[0])
    else:
        idx_iter = indices
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.2)
    for idx in idx_iter:
        if idx < 0 or idx >= pos_nd.shape[0]:
            continue
        p = pos_nd[idx]
        v = vel_nd[idx]
        strip = _tangent_great_circle(p, v, samples=samples)
        if strip is None:
            continue
        # align projection shapes to current strip dimension to avoid broadcast mismatches after dim changes
        d_strip = strip.shape[1]
        d_mean = mean.shape[1] if mean.ndim == 2 else d_strip
        d_proj = proj.shape[0] if proj.ndim == 2 else d_strip
        d_use = min(d_strip, d_mean, d_proj)
        strip_use = strip[:, :d_use]
        mean_use = mean[:, :d_use] if mean.ndim == 2 else np.zeros((1, d_use), dtype=np.float32)
        proj_use = proj[:d_use, :] if proj.ndim == 2 else np.zeros((d_use, 3), dtype=np.float32)
        strip_3d = (strip_use - mean_use) @ proj_use
        glColor4f(0.9, 0.8, 0.35, alpha)
        glBegin(GL_LINE_STRIP)
        for x, y, z in strip_3d:
            glVertex3f(float(x), float(y), float(z))
        glEnd()


def _draw_geodesic_edges(pos_nd: np.ndarray, springs_np: np.ndarray, mean: np.ndarray, proj: np.ndarray, samples: int = 20, alpha: float = 0.22):
    """Render springs as great-circle arcs projected to 3D; supports any source dim."""
    if springs_np.size == 0:
        return
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.5)
    i_idx = springs_np[:, 0].astype(np.int64)
    j_idx = springs_np[:, 1].astype(np.int64)
    for ii, jj in zip(i_idx, j_idx):
        if ii >= len(pos_nd) or jj >= len(pos_nd):
            continue
        strip_nd = _slerp_strip(pos_nd[ii], pos_nd[jj], samples=samples)
        # align projection shapes to current strip dimension to avoid broadcast mismatches after dim changes
        d_strip = strip_nd.shape[1]
        d_mean = mean.shape[1] if mean.ndim == 2 else d_strip
        d_proj = proj.shape[0] if proj.ndim == 2 else d_strip
        d_use = min(d_strip, d_mean, d_proj)
        strip_use = strip_nd[:, :d_use]
        mean_use = mean[:, :d_use] if mean.ndim == 2 else np.zeros((1, d_use), dtype=np.float32)
        proj_use = proj[:d_use, :] if proj.ndim == 2 else np.zeros((d_use, 3), dtype=np.float32)
        strip_3d = (strip_use - mean_use) @ proj_use
        glColor4f(0.8, 0.85, 0.95, alpha)
        glBegin(GL_LINE_STRIP)
        for x, y, z in strip_3d:
            glVertex3f(float(x), float(y), float(z))
        glEnd()


def _build_backdrop_lines(dim: int, proj_np: np.ndarray, samples: int = 96, max_dims: int = 6):
    dims = min(dim, max_dims, proj_np.shape[0])
    if dims < 2:
        return [], [], [], []
    theta = np.linspace(0.0, 2.0 * math.pi, num=samples, endpoint=True, dtype=np.float32)
    lines = []
    line_colors = []
    for i, j in combinations(range(dims), 2):
        circle = np.zeros((samples, dims), dtype=np.float32)
        circle[:, i] = np.cos(theta)
        circle[:, j] = np.sin(theta)
        lines.append(circle @ proj_np[:dims, :])
        hue = (0.13 * i + 0.17 * j) % 1.0
        line_colors.append(_hsl_to_rgb_np(np.array(hue, dtype=np.float32), 0.6, 0.55))

    axes = []
    axis_colors = []
    for i in range(dims):
        axis = np.zeros((2, dims), dtype=np.float32)
        axis[1, i] = 1.0
        axes.append(axis @ proj_np[:dims, :])
        hue = (0.14 * i) % 1.0
        axis_colors.append(_hsl_to_rgb_np(np.array(hue, dtype=np.float32), 0.7, 0.6))

    return lines, line_colors, axes, axis_colors


def _draw_backdrop(lines, line_colors, axes, axis_colors, alpha_grid: float = 0.18, alpha_axis: float = 0.35):
    if (not lines) and (not axes):
        return
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    if lines:
        glLineWidth(1.0)
        for pts, col in zip(lines, line_colors):
            col_np = np.asarray(col, dtype=np.float32).reshape(-1)
            glColor4f(float(col_np[0]), float(col_np[1]), float(col_np[2]), alpha_grid)
            glBegin(GL_LINE_STRIP)
            for x, y, z in pts:
                glVertex3f(float(x), float(y), float(z))
            glEnd()
    if axes:
        glLineWidth(2.0)
        for pts, col in zip(axes, axis_colors):
            col_np = np.asarray(col, dtype=np.float32).reshape(-1)
            glColor4f(float(col_np[0]), float(col_np[1]), float(col_np[2]), alpha_axis)
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            for x, y, z in pts[1:]:
                glVertex3f(float(x), float(y), float(z))
            glEnd()


class _ForceBuffers:
    def __init__(self, device: torch.device, n: int, dim: int, batch: int = 1):
        self.device = device
        self.shape = (batch, n, dim)
        self.force = torch.empty(self.shape, device=self.device)
        self.noise = torch.empty(self.shape, device=self.device)
        self.lor_full = torch.empty(self.shape, device=self.device)

    def ensure(self, n: int, dim: int, batch: int = 1):
        desired = (batch, n, dim)
        if getattr(self, "shape", None) != desired or self.force.shape != desired:
            self.shape = desired
            self.force = torch.empty(desired, device=self.device)
            self.noise = torch.empty(desired, device=self.device)
            self.lor_full = torch.empty(desired, device=self.device)
        return self


class _StepBuffers:
    """Per-depth integration scratch to avoid allocations during adaptive retries."""

    def __init__(self, device: torch.device, n: int, dim: int, batch: int = 1):
        self.device = device
        self.ensure(n, dim, batch)

    def ensure(self, n: int, dim: int, batch: int = 1):
        shape = (batch, n, dim)
        self.v_mid = torch.empty(shape, device=self.device)
        self.p_mid = torch.empty(shape, device=self.device)
        self.v_new = torch.empty(shape, device=self.device)
        self.p_new = torch.empty(shape, device=self.device)
        self.a0 = torch.empty(shape, device=self.device)
        self.a1 = torch.empty(shape, device=self.device)
        return self


def _build_rest_angles(pos_unit: np.ndarray, springs: np.ndarray) -> np.ndarray:
    if springs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    rest_angles = np.zeros(springs.shape[0], dtype=np.float32)
    for k, (i, j, _, _) in enumerate(springs):
        ui = pos_unit[int(i)]
        uj = pos_unit[int(j)]
        cos_t = float(np.clip(np.dot(ui, uj), -1.0, 1.0))
        rest_angles[k] = math.acos(cos_t)
    return rest_angles


def _all_pairs(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pairs = np.stack([i_idx, j_idx], axis=1)
    return pairs.astype(np.int64)


def _geodesic_forces(
    pos: torch.Tensor,
    vel: torch.Tensor,
    springs: torch.Tensor,
    rest_angles: torch.Tensor,
    pairs: torch.Tensor,
    charges: torch.Tensor,
    masses: torch.Tensor,
    k_spring: float,
    k_coulomb: float,
    G: float,
    softening: float,
    lorentz_k: float,
    temp: float,
    B_vec: torch.Tensor,
    B_mat: torch.Tensor | None,
    buffers: _ForceBuffers | None = None,
    out: torch.Tensor | None = None,
    core_radius: float = 0.06,
    core_stiffness: float = 800.0,
    # angular gate: soften far-pair work when expected force is below tolerance
    gate_band: float = 0.2,
    dense_pairs: bool = False,
) -> torch.Tensor:
    """Supports pos/vel as (N,D) or batched (B,N,D)."""
    if out is not None:
        F = out
        F.zero_()
    elif buffers is not None:
        if pos.dim() == 2:
            buffers.ensure(pos.shape[0], pos.shape[1], 1)
        else:
            buffers.ensure(pos.shape[1], pos.shape[2], pos.shape[0])
        F = buffers.force
        F.zero_()
    else:
        F = torch.zeros_like(pos)

    batched = pos.dim() == 3
    if batched:
        B, N, D = pos.shape
        batch_offset = torch.arange(B, device=pos.device)[:, None] * N
    else:
        N = pos.shape[0]
        D = pos.shape[1]

    if springs.numel() > 0 and k_spring != 0.0:
        i_idx = springs[:, 0].long()
        j_idx = springs[:, 1].long()
        k_edge = springs[:, 3] if springs.shape[1] > 3 else torch.ones_like(i_idx, dtype=pos.dtype, device=pos.device)
        if batched:
            ui = pos[:, i_idx]
            uj = pos[:, j_idx]
            cos_t = torch.clamp((ui * uj).sum(dim=2), -1.0, 1.0)
            theta = torch.acos(cos_t)
            sin_t = torch.sin(theta)
            mask = sin_t > 1e-6
            if mask.any():
                rest_m = rest_angles.unsqueeze(0).expand_as(theta)[mask]
                k_edge_m = k_edge.unsqueeze(0).expand_as(theta)[mask]
                ui_m = ui[mask]
                uj_m = uj[mask]
                cos_t_m = cos_t[mask]
                theta_m = theta[mask]
                dir_ij = uj_m - cos_t_m[:, None] * ui_m
                nrm = torch.linalg.norm(dir_ij, dim=1, keepdim=True).clamp(min=1e-9)
                dir_hat = dir_ij / nrm
                delta = theta_m - rest_m
                fmag = (k_spring * k_edge_m) * delta
                Fi = fmag[:, None] * dir_hat
                Fj = -Fi
                Fj = Fj - (Fj * uj_m).sum(dim=1, keepdim=True) * uj_m
                i_flat = (i_idx.unsqueeze(0).expand(B, -1).reshape(-1) + batch_offset.reshape(-1, 1).expand(-1, i_idx.shape[0]).reshape(-1))[mask.reshape(-1)]
                j_flat = (j_idx.unsqueeze(0).expand(B, -1).reshape(-1) + batch_offset.reshape(-1, 1).expand(-1, j_idx.shape[0]).reshape(-1))[mask.reshape(-1)]
                Fi_flat = Fi.reshape(-1, D)
                Fj_flat = Fj.reshape(-1, D)
                F_flat = F.reshape(-1, D)
                F_flat.index_add_(0, i_flat, Fi_flat)
                F_flat.index_add_(0, j_flat, Fj_flat)
        else:
            ui = pos[i_idx]
            uj = pos[j_idx]
            cos_t = torch.clamp((ui * uj).sum(dim=1), -1.0, 1.0)
            theta = torch.acos(cos_t)
            sin_t = torch.sin(theta)
            mask = sin_t > 1e-6
            if mask.any():
                ui_m = ui[mask]
                uj_m = uj[mask]
                cos_t_m = cos_t[mask]
                theta_m = theta[mask]
                rest_m = rest_angles[mask]
                k_edge_m = k_edge[mask]
                dir_ij = uj_m - cos_t_m[:, None] * ui_m
                nrm = torch.linalg.norm(dir_ij, dim=1, keepdim=True).clamp(min=1e-9)
                dir_hat = dir_ij / nrm
                delta = theta_m - rest_m
                fmag = (k_spring * k_edge_m) * delta
                Fi = fmag[:, None] * dir_hat
                Fj = -Fi
                Fj = Fj - (Fj * uj_m).sum(dim=1, keepdim=True) * uj_m
                i_s = i_idx[mask]
                j_s = j_idx[mask]
                F.index_add_(0, i_s, Fi)
                F.index_add_(0, j_s, Fj)

    use_dense_pairs = dense_pairs and (k_coulomb != 0.0 or G != 0.0)
    use_sparse_pairs = (pairs.numel() > 0) and (k_coulomb != 0.0 or G != 0.0) and not use_dense_pairs

    if use_dense_pairs:
        # Dense all-pairs Coulomb/Gravity; optimal for small N to avoid scatter overhead.
        P = pos if batched else pos.unsqueeze(0)
        V = vel if batched else vel.unsqueeze(0)
        _, N_dense, D_dense = P.shape
        diff = P[:, :, None, :] - P[:, None, :, :]  # (B,N,N,D)
        raw_dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        dist = raw_dist + softening
        inv_dist = 1.0 / dist
        inv_r2 = inv_dist.pow(2)

        qmax = charges.abs().max()
        mmax = masses.max()
        f_scale = torch.max(
            torch.tensor(abs(k_coulomb), device=pos.device, dtype=pos.dtype) * (qmax * qmax),
            torch.tensor(abs(G), device=pos.device, dtype=pos.dtype) * (mmax * mmax),
        )
        f_tol = torch.tensor(max(abs(float(temp)) * 2.0, 0.05), device=pos.device, dtype=pos.dtype)
        theta_cut = torch.sqrt(torch.clamp(f_scale / torch.clamp(f_tol, min=1e-6), max=(math.pi * math.pi)))
        theta_cut = torch.clamp(theta_cut, max=math.pi)
        band = max(1e-4, float(gate_band))
        cos_cut = torch.cos(theta_cut)
        cos_soft = torch.cos(torch.clamp(theta_cut - band, min=0.0))

        dot_p = torch.matmul(P, P.transpose(1, 2)).unsqueeze(-1)
        cos_t = torch.clamp(dot_p, -1.0, 1.0)
        denom = (cos_soft - cos_cut).clamp(min=1e-6)
        weight = ((cos_t - cos_cut) / denom).clamp(min=0.0, max=1.0)

        Q_prod = charges.view(1, N_dense, 1) * charges.view(1, 1, N_dense)
        M_prod = masses.view(1, N_dense, 1) * masses.view(1, 1, N_dense)
        mag = (k_coulomb * Q_prod - G * M_prod).unsqueeze(-1) * inv_r2

        overlap = core_radius - raw_dist
        repulsion = torch.where(overlap > 0, core_stiffness * overlap, torch.zeros_like(overlap))
        mag = mag + repulsion

        mag = mag * weight
        inv_raw = torch.where(raw_dist > 1e-9, 1.0 / raw_dist, torch.zeros_like(raw_dist))
        f_ij = (mag * inv_raw) * diff

        # zero self-interaction on the diagonal
        eye = torch.eye(N_dense, device=pos.device, dtype=pos.dtype).view(1, N_dense, N_dense, 1)
        f_ij = f_ij * (1.0 - eye)

        F_dense = f_ij.sum(dim=2)
        rad_comp = (F_dense * P).sum(dim=-1, keepdim=True)
        F_dense = F_dense - rad_comp * P

        if batched:
            F = F + F_dense
        else:
            F = F + F_dense.squeeze(0)

    if use_sparse_pairs:
        i_idx = pairs[:, 0].long()
        j_idx = pairs[:, 1].long()
        # derive a cosine cutoff from a force tolerance tied to temperature/limits
        qmax = charges.abs().max()
        mmax = masses.max()
        f_scale = torch.max(
            torch.tensor(abs(k_coulomb), device=pos.device, dtype=pos.dtype) * (qmax * qmax),
            torch.tensor(abs(G), device=pos.device, dtype=pos.dtype) * (mmax * mmax),
        )
        f_tol = torch.tensor(max(abs(float(temp)) * 2.0, 0.05), device=pos.device, dtype=pos.dtype)
        theta_cut = torch.sqrt(torch.clamp(f_scale / torch.clamp(f_tol, min=1e-6), max=(math.pi * math.pi)))
        theta_cut = torch.clamp(theta_cut, max=math.pi)
        band = max(1e-4, float(gate_band))
        cos_cut = torch.cos(theta_cut)
        cos_soft = torch.cos(torch.clamp(theta_cut - band, min=0.0))
        if batched:
            ui = pos[:, i_idx]
            uj = pos[:, j_idx]
            diff = uj - ui
            raw_dist = torch.linalg.norm(diff, dim=2, keepdim=True)
            dist = raw_dist + softening
            inv_dist = 1.0 / dist
            dir_hat = diff * inv_dist
            cos_t = torch.clamp((ui * uj).sum(dim=2), -1.0, 1.0)
            qprod = charges.unsqueeze(0)[:, i_idx] * charges.unsqueeze(0)[:, j_idx]
            mprod = masses.unsqueeze(0)[:, i_idx] * masses.unsqueeze(0)[:, j_idx]
            inv_r2 = (inv_dist.squeeze(-1)) ** 2
            f_c = k_coulomb * qprod * inv_r2
            f_g = G * mprod * inv_r2
            mag = f_c - f_g

            overlap = core_radius - raw_dist.squeeze(-1)
            repulsion = torch.where(overlap > 0, core_stiffness * overlap, torch.zeros_like(overlap))
            mag = mag + repulsion

            denom = (cos_soft - cos_cut).clamp(min=1e-6)
            weight = ((cos_t - cos_cut) / denom).clamp(min=0.0, max=1.0)
            mag = mag * weight

            pair_force = mag[:, :, None] * dir_hat
            Fi = -pair_force
            Fj = pair_force
            Fi = Fi - (Fi * ui).sum(dim=2, keepdim=True) * ui
            Fj = Fj - (Fj * uj).sum(dim=2, keepdim=True) * uj
            F_flat = F.reshape(-1, D)
            i_flat = (i_idx.unsqueeze(0).expand(B, -1) + batch_offset).reshape(-1)
            j_flat = (j_idx.unsqueeze(0).expand(B, -1) + batch_offset).reshape(-1)
            F_flat.index_add_(0, i_flat, Fi.reshape(-1, D))
            F_flat.index_add_(0, j_flat, Fj.reshape(-1, D))
        else:
            ui = pos[i_idx]
            uj = pos[j_idx]
            diff = uj - ui
            raw_dist = torch.linalg.norm(diff, dim=1, keepdim=True)
            dist = raw_dist + softening
            inv_dist = 1.0 / dist
            dir_hat = diff * inv_dist
            cos_t = torch.clamp((ui * uj).sum(dim=1), -1.0, 1.0)
            qprod = charges[i_idx] * charges[j_idx]
            mprod = masses[i_idx] * masses[j_idx]
            inv_r2 = (inv_dist.squeeze()) ** 2
            f_c = k_coulomb * qprod * inv_r2
            f_g = G * mprod * inv_r2
            mag = f_c - f_g

            overlap = core_radius - raw_dist.squeeze()
            repulsion = torch.where(overlap > 0, core_stiffness * overlap, torch.zeros_like(overlap))
            mag = mag + repulsion

            denom = (cos_soft - cos_cut).clamp(min=1e-6)
            weight = ((cos_t - cos_cut) / denom).clamp(min=0.0, max=1.0)
            mag = mag * weight

            pair_force = mag[:, None] * dir_hat
            Fi = -pair_force
            Fj = pair_force
            Fi = Fi - (Fi * ui).sum(dim=1, keepdim=True) * ui
            Fj = Fj - (Fj * uj).sum(dim=1, keepdim=True) * uj
            F.index_add_(0, i_idx, Fi)
            F.index_add_(0, j_idx, Fj)
    if lorentz_k != 0.0:
        # full n-D skew field if provided, else legacy 3D cross-product
        if B_mat is not None and B_mat.numel() > 0:
            if batched:
                lor_full = torch.matmul(vel, B_mat.T)
            else:
                lor_full = torch.matmul(vel, B_mat.T)
            F = F + lorentz_k * lor_full
        else:
            Bv = B_vec
            if Bv.shape[0] < 3:
                Bv = torch.nn.functional.pad(Bv, (0, 3 - Bv.shape[0]))
            if pos.shape[-1] >= 3:
                if batched:
                    v3 = vel[:, :, :3]
                    Bv3 = Bv[:3].view(1, 1, 3).expand_as(v3)
                    lor3 = torch.cross(v3, Bv3, dim=2)
                    if buffers is not None:
                        buffers.ensure(N, D, B)
                        lor_full = buffers.lor_full
                        lor_full.zero_()
                    else:
                        lor_full = torch.zeros_like(pos)
                    lor_full[:, :, :3] = lor3
                    F = F + lorentz_k * lor_full
                else:
                    v3 = vel[:, :3]
                    Bv3 = Bv[:3]
                    if Bv3.dim() == 1:
                        Bv3 = Bv3.unsqueeze(0).expand_as(v3)
                    lor3 = torch.cross(v3, Bv3, dim=1)
                    if buffers is not None:
                        buffers.ensure(N, D, 1)
                        lor_full = buffers.lor_full[0]
                        lor_full.zero_()
                    else:
                        lor_full = torch.zeros_like(pos)
                    lor_full[:, :3] = lor3
                    F = F + lorentz_k * lor_full

    masses_exp = masses if batched else masses
    if batched:
        masses_exp = masses.unsqueeze(0).expand(B, -1)
        F = F / masses_exp[:, :, None]
        F = F - (F * pos).sum(dim=2, keepdim=True) * pos
    else:
        F = F / masses[:, None]
        F = F - (F * pos).sum(dim=1, keepdim=True) * pos
    return F


def run(
    path: str | None,
    dt: float,
    k_spring: float,
    k_coulomb: float,
    G: float,
    lorentz_k: float,
    B_vec: Tuple[float, float, float],
    B_matrix_flat: List[float] | None,
    B_skew_random: bool,
    B_skew_auto: bool,
    B_scale: float,
    B_seed: int | None,
    temp: float,
    damping: float,
    softening: float,
    n_dim: int,
    n_points: int,
    max_neighbors: int | None,
    neighbor_refresh: int,
    bond_max_per_node: int,
    bond_link_angle: float,
    bond_shear_ratio: float,
    bond_k: float,
    ionic_bonds: bool,
    ionic_valence: int,
    ghost_history: int,
    ghost_hue_cycles: float,
    min_dt_override: float | None,
    max_speed: float,
    max_recursion: int,
    radius_scale: float = 0.08,
    diff_mass_rate: float = 0.05,
    diff_charge_rate: float = 0.05,
    collide_gain: float = 1.1,
    merge_speed_frac: float = 0.4,
    shatter_speed_frac: float = 0.9,
    merge_size_frac: float = 0.5,
    shatter_size_frac: float = 0.25,
    shatter_k_max: int = 6,
    collision_pair_limit: int = 256,
    dense_pair_threshold: int = 512,
    accel_shatter_thresh: float = 0.0,
    accel_shatter_fraction: float = 0.35,
    accel_shatter_k: int = 6,
    accel_shatter_kick: float = 0.02,
    inflate_mult: float = 1.2,
    mass_vapor_thresh: float = 0.015,
    condense_temp_thresh: float = 0.5,
    condense_chunk_mass: float = 0.15,
    temp_cool_rate: float = 0.35,
    temp_heat_rate: float = 0.1,
    dt_mode: str = "single-posthoc",
    c_substep_dt_max: float | None = None,
    c_substeps_max: int = 64,
    c_n_cap: int | None = None,
    draw_direct_edges: bool = False,
    freeze_pca: bool = True,
    draw_edges: bool = True,
    c_max_edges: int | None = None,
    phys_fps_limit: float = 60.0,
    enable_watchdog: bool = False,
    device: str = "auto",
    nodes_override: List[Dict] | None = None,
    edges_override: List[Dict] | None = None,
    seed: int = 0,
    south_strength: float = 0.0,
    south_axis: int | None = None,
    south_enabled: bool | None = None,
    fov_deg: float = 10.0,
    z_near: float = 0.1,
    z_far: float = 1000.0,
    draw_backdrop: bool = False,
    draw_velocity_paths: bool = False,
    use_c_physics: bool = True,
    render_scale: float = 10.0,
    airplane_path: str = "airplane.json",
    scene_path: str = "scene.json",
):
    rng = np.random.default_rng(seed)
    B_vec = np.asarray(B_vec, dtype=np.float32)
    # Hard-disable the non-C backend: only C-physics is supported.
    if not use_c_physics:
        raise SystemExit("Non-C physics path is disabled. Use the C backend (--c-physics).")

    # Torch is optional; C-physics mode must run without it.
    if use_c_physics:
        torch_device = None
        local_pin_enabled = False
    else:
        if torch is None:
            raise SystemExit("torch is required unless --c-physics is enabled")
        # device selection (clean)
        if device not in {"auto", "cuda", "cpu"}:
            raise SystemExit("device must be one of: auto, cuda, cpu")
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit("CUDA requested but not available")
            torch_device = torch.device("cuda")
        else:  # cpu
            torch_device = torch.device("cpu")
        local_pin_enabled = bool(torch_device.type == "cuda" and torch.cuda.is_available())
    def build_B_mat(dim: int):
        if B_matrix_flat is not None:
            return _skew_from_flat(B_matrix_flat, dim) * float(B_scale)
        if B_skew_random or (B_skew_auto and dim > 3):
            local_rng = np.random.default_rng(seed if B_seed is None else B_seed)
            return _random_skew(dim, local_rng, scale=float(B_scale))
        return None
    B_mat_np = build_B_mat(n_dim)
    if torch_device is None:
        B_vec_torch = None
        B_mat_torch = None
    else:
        B_vec_torch = torch.as_tensor(B_vec, device=torch_device)
        B_mat_torch = torch.as_tensor(B_mat_np, device=torch_device) if B_mat_np is not None else None
    if nodes_override is not None and edges_override is not None:
        nodes, edges = nodes_override, edges_override
    elif path is not None:
        nodes, edges = _load_prephysics(path)
    else:
        pos_init = _sobol_like_sphere(n_points, n_dim, seed)
        pos_init = _relax_repulsion(pos_init, iters=24, step=0.06)
        nodes = [
            {"id": f"p{i}", "embedding": pos_init[i].tolist(), "kind": "concept", "mass": float(1.0 + 0.2 * rng.standard_normal())}
            for i in range(n_points)
        ]
        edges = []
    node_ids, pos0, vel0, masses, colors, springs, mean, proj, fixed_mask = base.build_sim_state(nodes, edges)

    def _projection_for(mode: str, pts) -> tuple[np.ndarray, np.ndarray]:
        if torch is not None and isinstance(pts, torch.Tensor):
            pts_np = pts.detach().cpu().numpy()
        else:
            pts_np = np.asarray(pts)
        D = pts_np.shape[1]
        if mode == "axes":
            mean_local = np.zeros((1, D), dtype=np.float32)
            P = np.zeros((D, 3), dtype=np.float32)
            for i in range(min(D, 3)):
                P[i, i] = 1.0
            return mean_local, P
        if mode == "random":
            mean_local = pts_np.mean(axis=0, keepdims=True).astype(np.float32)
            if D >= 3:
                R = np.random.standard_normal(size=(D, 3)).astype(np.float32)
                Q, _ = np.linalg.qr(R)
                return mean_local, Q[:, :3]
            # D < 3: build an orthobasis for available dims and pad remaining columns with zeros
            R = np.random.standard_normal(size=(D, D)).astype(np.float32)
            Q, _ = np.linalg.qr(R)
            P = np.zeros((D, 3), dtype=np.float32)
            P[:, :D] = Q[:, :D]
            return mean_local, P
        if mode == "hyperbolic":
            if D < 2:
                mean_local = np.zeros((1, D), dtype=np.float32)
                P = np.zeros((D, 3), dtype=np.float32)
                for i in range(min(D, 3)):
                    P[i, i] = 1.0
                return mean_local, P
            denom = np.clip(1.0 - pts_np[:, 0:1], 1e-6, None)
            ball = pts_np[:, 1:] / denom
            mean_ball, P_ball = base.compute_projection_basis(ball)
            if P_ball.shape[1] < 3:
                pad_cols = 3 - P_ball.shape[1]
                P_ball = np.hstack([P_ball, np.zeros((P_ball.shape[0], pad_cols), dtype=P_ball.dtype)])
            P_full = np.zeros((D, 3), dtype=np.float32)
            P_full[1:, :] = P_ball
            mean_full = np.zeros((1, D), dtype=np.float32)
            return mean_full, P_full
        # default PCA
        mean_local, P = base.compute_projection_basis(pts_np)
        if P.shape[1] < 3:
            # pad projection to 3 columns if data is lower dimensional
            pad_cols = 3 - P.shape[1]
            P = np.hstack([P, np.zeros((P.shape[0], pad_cols), dtype=P.dtype)])
        return mean_local, P

    mean, proj = _projection_for("pca", pos0)
    if pos0.shape[1] != n_dim:
        D = pos0.shape[1]
        if D >= n_dim:
            pos0 = pos0[:, :n_dim]
        else:
            pad = np.zeros((pos0.shape[0], n_dim - D), dtype=np.float32)
            pos0 = np.hstack([pos0, pad])
    pos_np = _normalize_rows(pos0.astype(np.float32))
    vel_np = np.zeros_like(pos_np)
    springs_np = springs.astype(np.float32)
    rest_angles_np = _build_rest_angles(pos_np, springs_np)
    # Assign discrete ionic species for a reactive soup (integers in {-2,-1,1,2})
    species_choices = np.array([-2, -1, 1, 2], dtype=np.int64)
    species = rng.choice(species_choices, size=pos_np.shape[0])
    charges_np = species.astype(np.float32)
    masses_base_np = np.maximum(0.1, masses.astype(np.float32))
    masses_np = masses_base_np * (1.0 + 0.4 * np.abs(charges_np))
    mass_ref = float(np.median(masses_np)) if np.isfinite(masses_np).any() else 1.0
    mass_ref_cuberoot = np.cbrt(max(mass_ref, 1e-6))
    # Color by species
    palette = {
        -2: (0.9, 0.2, 0.2),   # strong negative: red
        -1: (0.9, 0.6, 0.3),   # mild negative: orange
         1: (0.3, 0.7, 1.0),   # mild positive: cyan
         2: (0.8, 0.4, 0.9),   # strong positive: magenta
    }
    colors = _colors_from_charges(charges_np)

    # C physics fast-path: no torch, no projection, direct ctypes pointer -> OpenGL.
    if use_c_physics:
        if min_dt_override is not None:
            min_dt_local = max(1e-12, float(min_dt_override))
        else:
            # Do not impose a Python-side dt floor in C-physics mode.
            # The C backend owns stability constraints; we only need a tiny positive
            # value to avoid divide-by-zero in the Python control logic.
            min_dt_local = float(np.finfo(np.float64).tiny)
        _run_c_physics_only(
            pos_np=pos_np,
            vel_np=vel_np,
            springs_np=springs_np,
            masses_np=masses_np,
            charges_np=charges_np,
            colors_np=colors,
            radius_scale=float(radius_scale),
            collide_gain=float(collide_gain),
            merge_speed_frac=float(merge_speed_frac),
            merge_size_frac=float(merge_size_frac),
            shatter_speed_frac=float(shatter_speed_frac),
            shatter_size_frac=float(shatter_size_frac),
            shatter_k_max=int(shatter_k_max),
            collision_pair_limit=int(collision_pair_limit),
            accel_shatter_thresh=float(accel_shatter_thresh),
            accel_shatter_fraction=float(accel_shatter_fraction),
            accel_shatter_k=int(accel_shatter_k),
            accel_shatter_kick=float(accel_shatter_kick),
            mass_vapor_thresh=float(mass_vapor_thresh),
            condense_temp_thresh=float(condense_temp_thresh),
            condense_chunk_mass=float(condense_chunk_mass),
            dt=float(dt),
            dt_mode=str(dt_mode),
            min_dt=float(min_dt_local),
            max_speed=float(max_speed),
            c_substep_dt_max=(None if c_substep_dt_max is None else float(c_substep_dt_max)),
            c_substeps_max=int(c_substeps_max),
            c_n_cap=(None if c_n_cap is None else int(c_n_cap)),
            k_spring=float(k_spring),
            k_coulomb=float(k_coulomb),
            G=float(G),
            lorentz_k=float(lorentz_k),
            B_vec=tuple(B_vec),
            temp=float(temp),
            damping=float(damping),
            softening=float(softening),
            bond_max_per_node=int(bond_max_per_node),
            bond_link_angle=float(bond_link_angle),
            bond_shear_ratio=float(bond_shear_ratio),
            bond_k=float(bond_k),
            ionic_bonds=bool(ionic_bonds),
            ionic_valence=int(ionic_valence),
            south_strength=float(south_strength),
            south_axis=int(south_axis if south_axis is not None else n_dim - 1),
            south_enabled=bool(south_enabled if south_enabled is not None else float(south_strength) > 0.0),
            draw_edges=bool(draw_edges),
            max_edges=(None if (c_max_edges is None or int(c_max_edges) <= 0) else int(c_max_edges)),
            phys_fps_limit=float(phys_fps_limit),
            ghost_history=int(ghost_history),
            ghost_hue_cycles=float(ghost_hue_cycles),
            draw_direct_edges=bool(draw_direct_edges),
            freeze_pca=bool(freeze_pca),
            fov_deg=float(fov_deg),
            z_near=float(z_near),
            z_far=float(z_far),
            render_scale=float(render_scale),
            airplane_path=str(airplane_path or "airplane.json"),
            scene_path=str(scene_path or "scene.json"),
        )
        return
    pos = torch.as_tensor(pos_np, device=torch_device)
    vel = torch.as_tensor(vel_np, device=torch_device)
    springs = torch.as_tensor(springs_np, device=torch_device)
    rest_angles = torch.as_tensor(rest_angles_np, device=torch_device)
    charges = torch.as_tensor(charges_np, device=torch_device)
    masses = torch.as_tensor(masses_np, device=torch_device)
    temps_np = np.full((pos_np.shape[0],), float(temp), dtype=np.float32)
    temps = torch.as_tensor(temps_np, device=torch_device)
    degree_np = np.zeros(pos.shape[0], dtype=np.int64)
    if springs_np.size:
        for i, j, *_ in springs_np:
            degree_np[int(i)] += 1
            degree_np[int(j)] += 1
    degree = torch.as_tensor(degree_np, device=torch_device)
    ionic_slots = torch.full((pos.shape[0],), ionic_valence, dtype=torch.int64, device=torch_device) if ionic_bonds else None
    bond_adj = torch.zeros((pos.shape[0], pos.shape[0]), dtype=torch.bool, device=torch_device)
    if springs_np.size:
        si = springs_np[:, 0].astype(int)
        sj = springs_np[:, 1].astype(int)
        bond_adj[si, sj] = True
        bond_adj[sj, si] = True
    force_buffers = _ForceBuffers(torch_device, pos.shape[0], pos.shape[1])
    # multiple integration buffers to allow parallel dt candidates without reallocating
    step_buffers = [_StepBuffers(torch_device, pos.shape[0], pos.shape[1]) for _ in range(3)]
    state_pos_buf = [torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
    state_vel_buf = [torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
    for buf_p, buf_v in zip(state_pos_buf, state_vel_buf):
        buf_p.copy_(pos, non_blocking=False)
        buf_v.copy_(vel, non_blocking=False)
    state_pos = [buf.numpy() for buf in state_pos_buf]
    state_vel = [buf.numpy() for buf in state_vel_buf]
    state_masses = [masses_np.copy(), masses_np.copy()]
    state_charges = [charges_np.copy(), charges_np.copy()]
    state_temps = [temps_np.copy(), temps_np.copy()]
    state_springs = [springs_np.copy(), springs_np.copy()]
    springs_dirty = True
    ionic_vapor_mass = 0.0
    ionic_vapor_charge = 0.0
    state_idx = 0
    state_gen = 0
    phys_fps = 0.0
    phys_dt_display = dt
    phys_steps_display = 0
    phys_depth_display = 0
    phys_dt_min_display = dt
    base_dt = dt
    fast_dt = base_dt
    slow_dt = max(5e-324, base_dt * 0.5)
    accept_ema = base_dt
    fail_ema = base_dt
    last_fail = False
    adaptive_dt = dt
    base_max_speed = float(max_speed)

    # projection buffers and helper (must be defined before physics thread starts)
    proj_mean_cpu = _to_torch_pinned_float(mean)
    proj_cpu = _to_torch_pinned_float(proj)
    pos3d_buf = [torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED) for _ in range(2)]
    ship_vis_mask_np: np.ndarray | None = None

    def _adapt_projection_dim(mean_np: np.ndarray, proj_np: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Slice/pad projection basis to a new dimension without recomputing PCA."""
        if mean_np is None or mean_np.ndim != 2:
            mean_np = np.zeros((1, target_dim), dtype=np.float32)
        elif mean_np.shape[1] != target_dim:
            if mean_np.shape[1] > target_dim:
                mean_np = mean_np[:, :target_dim]
            else:
                pad = np.zeros((mean_np.shape[0], target_dim - mean_np.shape[1]), dtype=mean_np.dtype)
                mean_np = np.concatenate([mean_np, pad], axis=1)
        if proj_np is None or proj_np.ndim != 2:
            proj_np = np.zeros((target_dim, 3), dtype=np.float32)
        elif proj_np.shape[0] != target_dim:
            if proj_np.shape[0] > target_dim:
                proj_np = proj_np[:target_dim, :]
            else:
                pad = np.zeros((target_dim - proj_np.shape[0], proj_np.shape[1]), dtype=proj_np.dtype)
                proj_np = np.concatenate([proj_np, pad], axis=0)
        return mean_np.astype(np.float32), proj_np.astype(np.float32)

    def _get_projection_np(pos_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if params.get("freeze_pca", False):
            return _adapt_projection_dim(proj_mean_cpu.cpu().numpy(), proj_cpu.cpu().numpy(), pos_np.shape[1])
        return _projection_for(params.get("proj_mode", "pca"), pos_np)

    def _ensure_projection(pos_like: torch.Tensor, out_buf: torch.Tensor | None = None) -> torch.Tensor:
        """Project CPU state to 3D, rebuilding basis/buffers if needed."""
        nonlocal proj_mean_cpu, proj_cpu, pos3d_buf, backdrop_dirty, ship_vis_mask_np
        D = pos_like.shape[1]

        def _rebuild_basis():
            nonlocal proj_mean_cpu, proj_cpu, backdrop_dirty
            if params.get("freeze_pca", False):
                m_local, p_local = _adapt_projection_dim(proj_mean_cpu.cpu().numpy(), proj_cpu.cpu().numpy(), D)
            else:
                m_local, p_local = _projection_for(params.get("proj_mode", "pca"), pos_like.cpu().numpy())
            proj_mean_cpu = torch.from_numpy(m_local).float()
            proj_cpu = torch.from_numpy(p_local).float()
            if PIN_ENABLED:
                proj_mean_cpu = proj_mean_cpu.pin_memory()
                proj_cpu = proj_cpu.pin_memory()
            backdrop_dirty = True

        if proj_mean_cpu.shape[1] != D or proj_cpu.shape[0] != D:
            _rebuild_basis()
        if out_buf is None or out_buf.shape[0] != pos_like.shape[0] or out_buf.shape[1] != 3:
            out_buf = torch.empty((pos_like.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED)
        try:
            torch.matmul(pos_like - proj_mean_cpu, proj_cpu, out=out_buf)
        except RuntimeError:
            _rebuild_basis()
            if out_buf.shape[0] != pos_like.shape[0] or out_buf.shape[1] != 3:
                out_buf = torch.empty((pos_like.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED)
            torch.matmul(pos_like - proj_mean_cpu, proj_cpu, out=out_buf)

        # Optional nonlinear display projections.
        # Map modes: normalize then project to 2D.
        # Ship mode: normalize the N-D->3D projection onto the sphere (display-only).
        mode_now = str(params.get("proj_mode", "pca") or "pca")
        if _is_map_proj_mode(mode_now):
            norms = torch.linalg.norm(out_buf, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            out_buf.div_(norms)
            _apply_map_projection_torch(out_buf, mode_now)
            ship_vis_mask_np = None
        elif _is_ship_proj_mode(mode_now):
            norms = torch.linalg.norm(out_buf, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            out_buf.div_(norms)
            ship_vis_mask_np = None
        return out_buf
    # allow aggressive halving, but keep a sensible floor so dt doesn't underflow and stall motion
    # default floor scales with the requested dt; override still respected
    if min_dt_override is not None:
        min_dt = max(1e-12, float(min_dt_override))
    else:
        min_dt = max(1e-6, float(dt) * 1e-3)
    subdiv_margin_hi = 0.05   # only subdivide/shrink when vmax exceeds (1+margin)*max_speed
    grow_margin_lo = 0.20     # only grow when vmax is below (1-grow_margin)*max_speed
    grow_rate = 2.0           # expand dt quickly when safe (double per tick)
    shrink_rate = 0.50        # shrink factor when overspeeding (halve)
    force_limit = 500.0      # shrink dt if max force magnitude exceeds this
    
    move_eps = 1e-6
    pos_last_change = pos.clone()
    energy_prev = None
    pair_idx = torch.as_tensor(_all_pairs(pos.shape[0]), device=torch_device)
    diag = {"dt": base_dt, "dt_min": min_dt, "reason": "init", "vmax": 0.0, "f_peak": 0.0, "rel_e": 0.0}
    min_dt_tensor = torch.tensor(min_dt, device=torch_device, dtype=pos.dtype)
    pos_lock = threading.Lock()
    # sanitize potentially tuple/list inputs from argparse
    if isinstance(dense_pair_threshold, (list, tuple, np.ndarray)):
        dense_pair_threshold = dense_pair_threshold[0] if len(dense_pair_threshold) else 0
    if isinstance(accel_shatter_thresh, (list, tuple, np.ndarray)):
        accel_shatter_thresh = accel_shatter_thresh[0] if len(accel_shatter_thresh) else 0.0
    if isinstance(accel_shatter_fraction, (list, tuple, np.ndarray)):
        accel_shatter_fraction = accel_shatter_fraction[0] if len(accel_shatter_fraction) else 0.0
    if isinstance(accel_shatter_k, (list, tuple, np.ndarray)):
        accel_shatter_k = accel_shatter_k[0] if len(accel_shatter_k) else 1
    if isinstance(accel_shatter_kick, (list, tuple, np.ndarray)):
        accel_shatter_kick = accel_shatter_kick[0] if len(accel_shatter_kick) else 0.0
    if isinstance(inflate_mult, (list, tuple, np.ndarray)):
        inflate_mult = inflate_mult[0] if len(inflate_mult) else 1.0
    if isinstance(collision_pair_limit, (list, tuple, np.ndarray)):
        collision_pair_limit = int(collision_pair_limit[0]) if len(collision_pair_limit) else 0
    if isinstance(shatter_k_max, (list, tuple, np.ndarray)):
        shatter_k_max = int(shatter_k_max[0]) if len(shatter_k_max) else 0
    if isinstance(merge_size_frac, (list, tuple, np.ndarray)):
        merge_size_frac = merge_size_frac[0] if len(merge_size_frac) else 0.0
    if isinstance(shatter_size_frac, (list, tuple, np.ndarray)):
        shatter_size_frac = shatter_size_frac[0] if len(shatter_size_frac) else 0.0
    if isinstance(merge_speed_frac, (list, tuple, np.ndarray)):
        merge_speed_frac = merge_speed_frac[0] if len(merge_speed_frac) else 0.0
    if isinstance(shatter_speed_frac, (list, tuple, np.ndarray)):
        shatter_speed_frac = shatter_speed_frac[0] if len(shatter_speed_frac) else 0.0
    if isinstance(mass_vapor_thresh, (list, tuple, np.ndarray)):
        mass_vapor_thresh = float(mass_vapor_thresh[0]) if len(mass_vapor_thresh) else 0.0
    if isinstance(condense_temp_thresh, (list, tuple, np.ndarray)):
        condense_temp_thresh = float(condense_temp_thresh[0]) if len(condense_temp_thresh) else 0.0
    if isinstance(condense_chunk_mass, (list, tuple, np.ndarray)):
        condense_chunk_mass = float(condense_chunk_mass[0]) if len(condense_chunk_mass) else 0.0
    if isinstance(temp_cool_rate, (list, tuple, np.ndarray)):
        temp_cool_rate = float(temp_cool_rate[0]) if len(temp_cool_rate) else 0.0
    if isinstance(temp_heat_rate, (list, tuple, np.ndarray)):
        temp_heat_rate = float(temp_heat_rate[0]) if len(temp_heat_rate) else 0.0
    if isinstance(south_axis, (list, tuple, np.ndarray)):
        south_axis = int(south_axis[0]) if len(south_axis) else -1

    params = {
        "k_spring_base": float(k_spring),
        "k_coulomb_base": float(k_coulomb),
        "G_base": float(G),
        "temp_base": float(temp),
        "damping_base": float(damping),
        "bond_shear_base": float(bond_shear_ratio),
        "max_speed_base": float(base_max_speed),
        "k_spring": float(k_spring),
        "k_coulomb": float(k_coulomb),
        "G": float(G),
        "temp": float(temp),
        "damping": float(damping),
        "bond_shear": float(bond_shear_ratio),
        "max_speed": float(base_max_speed),
        "lorentz_k": float(lorentz_k),
        "ghost_history": int(max(0, ghost_history)),
        "ghost_hue_cycles": float(max(0.0, ghost_hue_cycles)),
        "proj_mode": "ship",
        "dense_pair_threshold": int(max(0, dense_pair_threshold)),
        "south_strength": float(max(0.0, south_strength)),
        "south_enabled": bool(south_enabled if south_enabled is not None else south_strength > 0.0),
        "south_axis": int(south_axis if south_axis is not None else n_dim - 1),
        "accel_shatter_thresh": float(accel_shatter_thresh),
        "accel_shatter_fraction": float(accel_shatter_fraction),
        "accel_shatter_k": int(max(1, accel_shatter_k)),
        "accel_shatter_kick": float(accel_shatter_kick),
        "inflate_mult": float(inflate_mult),
        "mass_vapor_thresh": float(mass_vapor_thresh),
        "condense_temp_thresh": float(condense_temp_thresh),
        "condense_chunk_mass": float(condense_chunk_mass),
        "temp_cool_rate": float(max(0.0, temp_cool_rate)),
        "temp_heat_rate": float(max(0.0, temp_heat_rate)),
        "dt_mode": str(dt_mode),
        "draw_direct_edges": bool(draw_direct_edges),
        "freeze_pca": bool(freeze_pca),
        "draw_edges": bool(draw_edges),
        "draw_backdrop": bool(draw_backdrop),
        "draw_velocity_paths": bool(draw_velocity_paths),
        "render_scale": float(_safe_render_scale(render_scale)),
        "airplane_path": str(airplane_path or "airplane.json"),
        "scene_path": str(scene_path or "scene.json"),
    }

    def _build_south_dir(dim: int) -> torch.Tensor:
        axis = params.get("south_axis", dim - 1)
        axis = dim - 1 if axis is None or axis < 0 else min(int(axis), dim - 1)
        params["south_axis"] = axis
        vec = np.zeros((dim,), dtype=np.float32)
        vec[axis] = -1.0
        return torch.as_tensor(vec, device=torch_device, dtype=pos.dtype)
    watchdog_enabled = bool(enable_watchdog)
    stop_evt = threading.Event()
    current_n_dim = n_dim
    target_n_dim = n_dim
    pending_dim_change: int | None = None
    south_dir_torch = _build_south_dir(current_n_dim)
    pending_node_add: int = 0
    pending_node_remove: list[int] = []
    fly_state = {
        "active": False,
        "idx": 0,
        "vec_cpu": np.zeros((current_n_dim,), dtype=np.float32),
        "vec_torch": None,
        "dirty": False,
        "gain": 1.5,
    }
    phys_thread: threading.Thread | None = None

    def rebuild_state(new_dim: int, with_lock: bool = True):
        nonlocal pos, vel, springs, rest_angles, charges, masses, temps, degree, pair_idx, state_pos, state_pos_buf, state_vel, state_vel_buf, state_springs, state_idx, state_gen, pos_last_change, adaptive_dt, phys_dt_display, phys_dt_min_display, phys_steps_display, phys_depth_display, ionic_slots, current_n_dim, mean, proj, energy_prev, force_buffers, step_buffers, colors, bond_adj, backdrop_dirty, B_mat_np, B_mat_torch, node_selector_idx, fly_state, target_n_dim, pending_dim_change, pos3d_buf, proj_mean_cpu, proj_cpu, fast_dt, slow_dt, accept_ema, fail_ema, last_fail, diag, state_charges, state_temps, ionic_vapor_mass, ionic_vapor_charge, south_dir_torch
        new_dim = max(2, int(new_dim))
        target_n_dim = new_dim

    def queue_add_nodes(count: int = 1):
        nonlocal pending_node_add
        c = max(0, int(count))
        if c == 0:
            return
        with pos_lock:
            pending_node_add += c

    def queue_remove_nodes(indices: list[int]):
        nonlocal pending_node_remove
        if not indices:
            return
        with pos_lock:
            pending_node_remove.extend(int(i) for i in indices)

        def _do_rebuild():
            nonlocal current_n_dim, fly_state
            current_n_dim = new_dim
            prev_gain = 1.5
            try:
                prev_gain = float(fly_state.get("gain", 1.5))
            except Exception:
                prev_gain = 1.5
            rng_local = np.random.default_rng(seed + int(time.time()))
            if nodes_override is not None and edges_override is not None:
                base_nodes, base_edges = nodes_override, edges_override
            elif path is not None:
                base_nodes, base_edges = _load_prephysics(path)
            else:
                pos_init = _sobol_like_sphere(n_points, new_dim, rng_local.integers(0, 1_000_000))
                pos_init = _relax_repulsion(pos_init, iters=24, step=0.06)
                base_nodes = [
                    {"id": f"p{i}", "embedding": pos_init[i].tolist(), "kind": "concept", "mass": float(1.0 + 0.2 * rng_local.standard_normal())}
                    for i in range(n_points)
                ]
                base_edges = []

            node_ids_new, pos0_new, vel0_new, masses_new, colors_new, springs_new, mean0_new, proj0_new, fixed_mask_new = base.build_sim_state(base_nodes, base_edges)
            if pos0_new.shape[1] != new_dim:
                D = pos0_new.shape[1]
                if D >= new_dim:
                    pos0_new = pos0_new[:, :new_dim]
                else:
                    pad = np.zeros((pos0_new.shape[0], new_dim - D), dtype=np.float32)
                    pos0_new = np.hstack([pos0_new, pad])
            pos_np_new = _normalize_rows(pos0_new.astype(np.float32))
            vel_np_new = np.zeros_like(pos_np_new)
            springs_np_new = springs_new.astype(np.float32)
            rest_angles_np_new = _build_rest_angles(pos_np_new, springs_np_new)
            species_choices = np.array([-2, -1, 1, 2], dtype=np.int64)
            species_new = rng_local.choice(species_choices, size=pos_np_new.shape[0])
            charges_np_new = species_new.astype(np.float32)
            masses_base_np_new = np.maximum(0.1, masses_new.astype(np.float32))
            masses_np_new = masses_base_np_new * (1.0 + 0.4 * np.abs(charges_np_new))
            palette = {
                -2: (0.9, 0.2, 0.2),
                -1: (0.9, 0.6, 0.3),
                 1: (0.3, 0.7, 1.0),
                 2: (0.8, 0.4, 0.9),
            }
            colors_new = np.array([palette[int(s)] for s in species_new], dtype=np.float32)
            colors = colors_new
            pos = torch.empty((pos_np_new.shape[0], current_n_dim), device=torch_device, dtype=torch.float32)
            pos.copy_(torch.from_numpy(pos_np_new), non_blocking=False)
            vel = torch.empty_like(pos)
            vel.copy_(torch.from_numpy(vel_np_new), non_blocking=False)
            springs = torch.as_tensor(springs_np_new, device=torch_device)
            rest_angles = torch.as_tensor(rest_angles_np_new, device=torch_device)
            charges = torch.as_tensor(charges_np_new, device=torch_device)
            masses = torch.as_tensor(masses_np_new, device=torch_device)
            temps = torch.full((pos_np_new.shape[0],), float(params.get("temp", temp)), device=torch_device, dtype=torch.float32)
            degree_np_new = np.zeros(pos_np_new.shape[0], dtype=np.int64)
            if springs_np_new.size:
                for i, j, *_ in springs_np_new:
                    degree_np_new[int(i)] += 1
                    degree_np_new[int(j)] += 1
            degree = torch.as_tensor(degree_np_new, device=torch_device)
            ionic_slots = torch.full((pos_np_new.shape[0],), ionic_valence, dtype=torch.int64, device=torch_device) if ionic_bonds else None
            bond_adj = torch.zeros((pos_np_new.shape[0], pos_np_new.shape[0]), dtype=torch.bool, device=torch_device)
            if springs_np_new.size:
                si = springs_np_new[:, 0].astype(int)
                sj = springs_np_new[:, 1].astype(int)
                bond_adj[si, sj] = True
                bond_adj[sj, si] = True
            pair_idx = torch.as_tensor(_all_pairs(pos_np_new.shape[0]), device=torch_device)
            force_buffers.ensure(pos.shape[0], pos.shape[1])
            buf_count = max(3, max_recursion + 1)
            step_buffers = [_StepBuffers(torch_device, pos.shape[0], pos.shape[1]) for _ in range(buf_count)]
            state_pos_buf = [torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
            state_vel_buf = [torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
            for i, (buf_p, buf_v) in enumerate(zip(state_pos_buf, state_vel_buf)):
                if buf_p.shape != pos.shape:
                    buf_p = torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED)
                    state_pos_buf[i] = buf_p
                if buf_v.shape != vel.shape:
                    buf_v = torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED)
                    state_vel_buf[i] = buf_v
                buf_p.copy_(pos, non_blocking=False)
                buf_v.copy_(vel, non_blocking=False)
            state_pos = [buf.numpy() for buf in state_pos_buf]
            state_vel = [buf.numpy() for buf in state_vel_buf]
            state_springs = [springs_np_new.copy(), springs_np_new.copy()]
            state_masses = [masses_np_new.copy(), masses_np_new.copy()]
            state_charges = [charges_np_new.copy(), charges_np_new.copy()]
            temps_np_new = np.full((pos_np_new.shape[0],), float(params.get("temp", temp)), dtype=np.float32)
            state_temps = [temps_np_new.copy(), temps_np_new.copy()]
            springs_dirty = True
            state_idx = 0
            state_gen = 0
            node_selector_idx = 0
            fly_state = {
                "active": False,
                "idx": 0,
                "vec_cpu": np.zeros((current_n_dim,), dtype=np.float32),
                "vec_torch": None,
                "dirty": False,
                "gain": prev_gain,
            }
            south_dir_torch = _build_south_dir(current_n_dim)
            pos_last_change = pos.clone()
            adaptive_dt = base_dt
            phys_dt_display = base_dt
            phys_dt_min_display = base_dt
            mean, proj = _get_projection_np(pos_np_new)
            proj_mean_cpu = _to_torch_pinned_float(mean)
            proj_cpu = _to_torch_pinned_float(proj)
            # ensure basis matches the current dimension
            if proj_mean_cpu.shape[1] != current_n_dim or proj_cpu.shape[0] != current_n_dim:
                if params.get("freeze_pca", False):
                    mean, proj = _adapt_projection_dim(mean, proj, current_n_dim)
                else:
                    mean, proj = _get_projection_np(pos_np_new)
                proj_mean_cpu = _to_torch_pinned_float(mean)
                proj_cpu = _to_torch_pinned_float(proj)

            pos3d_buf = [
                torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED)
                for _ in range(2)
            ]
            try:
                torch.matmul(state_pos_buf[state_idx] - proj_mean_cpu, proj_cpu, out=pos3d_buf[0])
            except RuntimeError:
                # rebuild basis and retry if shapes raced
                if params.get("freeze_pca", False):
                    mean, proj = _adapt_projection_dim(mean, proj, state_pos_buf[state_idx].shape[1])
                else:
                    mean, proj = _get_projection_np(state_pos_buf[state_idx].cpu().numpy())
                proj_mean_cpu = _to_torch_pinned_float(mean)
                proj_cpu = _to_torch_pinned_float(proj)
                torch.matmul(state_pos_buf[state_idx] - proj_mean_cpu, proj_cpu, out=pos3d_buf[0])
            pos3d_buf[1] = torch.empty_like(pos3d_buf[0])
            energy_prev = None
            # reset adaptive integrator/bookkeeping to safe defaults
            adaptive_dt = base_dt
            phys_dt_display = base_dt
            phys_dt_min_display = base_dt
            phys_steps_display = 0
            phys_depth_display = 0
            fast_dt = base_dt
            slow_dt = max(5e-324, base_dt * 0.5)
            accept_ema = base_dt
            fail_ema = base_dt
            last_fail = False
            diag = {"dt": base_dt, "dt_min": min_dt, "reason": "rebuild", "vmax": 0.0, "f_peak": 0.0, "rel_e": 0.0}
            backdrop_dirty = True
            B_mat_np = build_B_mat(current_n_dim)
            B_mat_torch = torch.as_tensor(B_mat_np, device=torch_device) if B_mat_np is not None else None
            print(
                f"[debug] rebuild complete: current_n_dim={current_n_dim} pos_np_new_shape={pos_np_new.shape} proj_mean_shape={proj_mean_cpu.shape}",
                file=sys.stderr,
                flush=True,
            )

        if with_lock:
            with pos_lock:
                _do_rebuild()
        else:
            _do_rebuild()

    def physics_worker():
        nonlocal pos, vel, springs, rest_angles, charges, masses, temps, state_idx, state_gen, phys_fps, phys_dt_display, phys_dt_min_display, phys_steps_display, phys_depth_display, adaptive_dt, pos_last_change, ionic_slots, pair_idx, diag, current_n_dim, step_buffers, springs_dirty, bond_adj, degree, state_pos_buf, state_vel_buf, state_pos, state_vel, state_springs, state_masses, state_charges, state_temps, pos3d_buf, proj_mean_cpu, proj_cpu, fast_dt, slow_dt, last_fail, accept_ema, fail_ema, B_mat_torch, fly_state, pending_dim_change, target_n_dim, pending_node_add, pending_node_remove, colors, energy_prev, last_render_gen, backdrop_dirty, node_selector_idx, mean, proj, mass_ref, mass_ref_cuberoot, radius_scale, diff_mass_rate, diff_charge_rate, collide_gain, merge_speed_frac, shatter_speed_frac, merge_size_frac, shatter_size_frac, shatter_k_max, collision_pair_limit, ionic_vapor_mass, ionic_vapor_charge, south_dir_torch
        torch.set_grad_enabled(False)
        step_count = 0
        last_depth_used = 0
        last_time = time.perf_counter()
        copy_interval = 1.0 / 240.0  # throttle CPU pinned copies to ~240 Hz
        last_copy_time = last_time

        def force_at(p_local, v_local, dt_scale, out_buf=None):
            nonlocal south_dir_torch
            ks = params["k_spring"]
            kc = params["k_coulomb"]
            gg = params["G"]
            tp = params["temp"]
            dp_local = params["damping"]
            lk_local = params.get("lorentz_k", 0.0)
            dense_pairs = False
            dense_thresh = params.get("dense_pair_threshold", 0)
            if dense_thresh and p_local.shape[-2] <= dense_thresh:
                dense_pairs = True
            if p_local.dim() == 2:
                force_buffers.ensure(p_local.shape[0], p_local.shape[1], 1)
            else:
                force_buffers.ensure(p_local.shape[1], p_local.shape[2], p_local.shape[0])
            F_local = _geodesic_forces(
                p_local,
                v_local,
                springs,
                rest_angles,
                pair_idx,
                charges,
                masses,
                ks,
                kc,
                gg,
                softening,
                tp,
                lk_local,
                B_vec_torch,
                B_mat_torch,
                buffers=force_buffers,
                out=out_buf,
                dense_pairs=dense_pairs,
            )
            if tp != 0:
                tp_abs = abs(tp)
                noise = force_buffers.noise
                noise.normal_()
                if isinstance(dt_scale, float):
                    noise_scale = math.sqrt(max(dt_scale, min_dt)) * tp_abs
                    noise.mul_(noise_scale)
                else:
                    dt_scale_max = torch.maximum(dt_scale.max(), min_dt_tensor)
                    noise.mul_(dt_scale_max.sqrt() * tp_abs)
                if tp > 0:
                    F_local = F_local + noise
                else:
                    F_local = F_local - noise - 0.3 * tp_abs * v_local
            if dp_local > 0:
                # Linear-per-frame damping: damping=0.5 removes 50% of velocity per frame (scaled by dt).
                damp_clamped = max(0.0, min(1.0, float(dp_local)))
                if isinstance(dt_scale, float):
                    dt_eff = max(dt_scale, float(min_dt))
                    lambda_d = damp_clamped / dt_eff
                    F_local = F_local - lambda_d * v_local
                else:
                    dt_eff = torch.clamp(dt_scale.max(), min=min_dt_tensor)
                    lambda_d = damp_clamped / dt_eff
                    F_local = F_local - lambda_d * v_local

            south_strength = float(params.get("south_strength", 0.0))
            if params.get("south_enabled", False) and south_strength != 0.0:
                try:
                    dir_vec = south_dir_torch
                    if dir_vec is None or dir_vec.shape[0] != p_local.shape[-1]:
                        dir_vec = _build_south_dir(p_local.shape[-1])
                        south_dir_torch = dir_vec
                    dir_vec = dir_vec.to(p_local.device, dtype=p_local.dtype)
                    if p_local.dim() == 3:
                        dir_b = dir_vec.view(1, 1, -1)
                        dot = (dir_b * p_local).sum(dim=2, keepdim=True)
                        tang = dir_b - dot * p_local
                        F_local = F_local + south_strength * tang
                    else:
                        dir_b = dir_vec.view(1, -1)
                        dot = (dir_b * p_local).sum(dim=1, keepdim=True)
                        tang = dir_b - dot * p_local
                        F_local = F_local + south_strength * tang
                except Exception:
                    pass

            if fly_state.get("active", False):
                try:
                    if fly_state.get("vec_torch") is None or fly_state.get("dirty", False) or fly_state["vec_torch"].shape[0] != p_local.shape[-1]:
                        fly_state["vec_torch"] = torch.as_tensor(fly_state.get("vec_cpu", np.zeros((p_local.shape[-1],), dtype=np.float32)), device=pos.device, dtype=pos.dtype)
                        fly_state["dirty"] = False
                    thrust = fly_state["vec_torch"]
                    if thrust.shape[0] == p_local.shape[-1]:
                        gain = float(fly_state.get("gain", 1.5))
                        if p_local.dim() == 3:
                            B = p_local.shape[0]
                            N = p_local.shape[1]
                            idx = int(fly_state.get("idx", 0)) % max(1, N)
                            p_sel = p_local[:, idx]
                            thrust_b = thrust.view(1, -1).expand_as(p_sel)
                            dot = (thrust_b * p_sel).sum(dim=1, keepdim=True)
                            t_tan = thrust_b - dot * p_sel
                            F_local[:, idx] = F_local[:, idx] + gain * t_tan
                        else:
                            N = p_local.shape[0]
                            idx = int(fly_state.get("idx", 0)) % max(1, N)
                            p_sel = p_local[idx]
                            dot = torch.dot(thrust, p_sel)
                            t_tan = thrust - dot * p_sel
                            F_local[idx] = F_local[idx] + gain * t_tan
                except Exception:
                    pass
            return F_local

        def apply_bonds(p_local):
            nonlocal springs, rest_angles, degree, ionic_slots, springs_dirty, bond_adj, charges, masses
            if bond_max_per_node <= 0 or bond_link_angle <= 0:
                return

            i_idx = pair_idx[:, 0].long()
            j_idx = pair_idx[:, 1].long()
            ui = p_local[i_idx]
            uj = p_local[j_idx]
            cos_t = torch.clamp((ui * uj).sum(dim=1), -1.0, 1.0)
            theta = torch.acos(cos_t)

            cand = theta < bond_link_angle
            if ionic_bonds:
                opposite = (charges[i_idx] * charges[j_idx]) < 0
                slots_ok = (ionic_slots[i_idx] > 0) & (ionic_slots[j_idx] > 0)
                cand = cand & opposite & slots_ok

            # always allow bonding when only two nodes remain to avoid stalling the last merge
            if p_local.shape[0] == 2:
                cand = torch.ones_like(cand, dtype=torch.bool)

            # fallback: if no candidate bonds formed, force the closest pair (within a looser angle)
            if not cand.any() and p_local.shape[0] >= 2:
                theta_min, idx_min = torch.min(theta, dim=0)
                loose_thresh = max(bond_link_angle * 2.0, 0.6)
                if theta_min < loose_thresh:
                    forced = torch.zeros_like(cand, dtype=torch.bool)
                    forced[idx_min] = True
                    cand = forced

            if bond_max_per_node > 0:
                deg_ok = (degree[i_idx] < bond_max_per_node) & (degree[j_idx] < bond_max_per_node)
                cand = cand & deg_ok

            cand = cand & (~bond_adj[i_idx, j_idx])

            if cand.any():
                add_i = i_idx[cand]
                add_j = j_idx[cand]
                add_theta = theta[cand]

                # update degree and ionic slots in bulk
                degree.index_add_(0, add_i, torch.ones_like(add_i, dtype=degree.dtype))
                degree.index_add_(0, add_j, torch.ones_like(add_j, dtype=degree.dtype))
                if ionic_bonds:
                    ionic_slots.index_add_(0, add_i, -torch.ones_like(add_i, dtype=ionic_slots.dtype))
                    ionic_slots.index_add_(0, add_j, -torch.ones_like(add_j, dtype=ionic_slots.dtype))

                bond_adj[add_i, add_j] = True
                bond_adj[add_j, add_i] = True

                new_edges = torch.stack(
                    [add_i.float(), add_j.float(), add_theta.float(), torch.full_like(add_theta, bond_k, dtype=torch.float32)],
                    dim=1,
                )
                springs = torch.cat([springs, new_edges], dim=0) if springs.numel() else new_edges
                rest_angles = torch.cat([rest_angles, add_theta.float()]) if rest_angles.numel() else add_theta.float()
                springs_dirty = True

            if springs.numel():
                i_all = springs[:, 0].long()
                j_all = springs[:, 1].long()
                ui_all = p_local[i_all]
                uj_all = p_local[j_all]
                cos_t_all = torch.clamp((ui_all * uj_all).sum(dim=1), -1.0, 1.0)
                theta_all = torch.acos(cos_t_all)
                # make shear tolerance shrink for oversized, hot nodes so big/heated blobs shed mass
                size_ref = torch.tensor(max(mass_ref, 1e-6), device=pos.device, dtype=pos.dtype)
                mass_avg = 0.5 * (masses[i_all] + masses[j_all])
                size_factor = mass_avg / size_ref
                abs_temp = abs(params.get("temp", temp))
                temp_factor = 1.0 + 0.6 * abs_temp
                soften = 1.0 + 0.4 * torch.clamp(size_factor - 1.0, min=0.0)
                shear_eff = params.get("bond_shear", bond_shear_ratio) / (soften * temp_factor)
                overstretch = theta_all > (rest_angles * shear_eff)

                # high temperature adds a stochastic bond failure channel to force destabilization
                if abs_temp >= 5.0:
                    hot_prob = min(0.9, 0.08 * max(0.0, abs_temp - 5.0))
                    rand_break = torch.rand(theta_all.shape, device=theta_all.device) < hot_prob
                    overstretch = overstretch | rand_break
                if p_local.shape[0] == 2:
                    overstretch = overstretch & torch.zeros_like(overstretch, dtype=torch.bool)
                if overstretch.any():
                    keep = ~overstretch
                    rem_i = i_all[overstretch]
                    rem_j = j_all[overstretch]

                    degree.index_add_(0, rem_i, -torch.ones_like(rem_i, dtype=degree.dtype))
                    degree.index_add_(0, rem_j, -torch.ones_like(rem_j, dtype=degree.dtype))
                    if ionic_bonds:
                        ionic_slots.index_add_(0, rem_i, torch.ones_like(rem_i, dtype=ionic_slots.dtype))
                        ionic_slots.index_add_(0, rem_j, torch.ones_like(rem_j, dtype=ionic_slots.dtype))

                    bond_adj[rem_i, rem_j] = False
                    bond_adj[rem_j, rem_i] = False

                    # bleed mass/charge from overstressed nodes into newly spawned shard nodes (not into nothing)
                    bleed_base = 0.02 + 0.04 * abs_temp
                    bleed_size = torch.clamp(size_factor[overstretch] - 1.0, min=0.0)
                    bleed = torch.clamp(bleed_base + 0.03 * bleed_size, max=0.25)
                    if bleed.numel():
                        pos_np_live = pos.detach().cpu().numpy()
                        vel_np_live = vel.detach().cpu().numpy()
                        masses_np_live = masses.detach().cpu().numpy()
                        charges_np_live = charges.detach().cpu().numpy()
                        temps_np_live = temps.detach().cpu().numpy()
                        springs_np_live = springs.detach().cpu().numpy()
                        springs_keep_mask = keep.detach().cpu().numpy().astype(bool)
                        springs_np_live = springs_np_live[springs_keep_mask]
                        ionic_slots_np_live = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None
                        rem_all = torch.cat([rem_i, rem_j])
                        bleed_all = torch.cat([bleed, bleed])
                        # iterate per unique node to avoid double spawning per bond pair
                        for idx_t, bleed_frac in zip(rem_all.tolist(), bleed_all.tolist()):
                            if idx_t < 0 or idx_t >= masses_np_live.shape[0]:
                                continue
                            m_loss = masses_np_live[idx_t] * bleed_frac
                            if m_loss <= 1e-8:
                                continue
                            q_loss = charges_np_live[idx_t] * bleed_frac
                            masses_np_live[idx_t] = max(1e-8, masses_np_live[idx_t] - m_loss)
                            charges_np_live[idx_t] = charges_np_live[idx_t] - q_loss
                            p_v = pos_np_live[idx_t]
                            v_v = vel_np_live[idx_t]
                            jitter = np.random.standard_normal(size=p_v.shape).astype(np.float32)
                            jitter = jitter - np.dot(jitter, p_v) * p_v
                            jit_norm = np.linalg.norm(jitter)
                            if jit_norm > 1e-6:
                                jitter = jitter / jit_norm
                            offset = 0.35 * radius_from_mass_np(np.array([m_loss], dtype=np.float32))[0] * jitter
                            p_new = _normalize_rows((p_v + offset)[None, :])[0]
                            v_new = v_v + jitter * 0.1 * np.linalg.norm(v_v)
                            pos_np_live = np.concatenate([pos_np_live, p_new[None, :]], axis=0)
                            vel_np_live = np.concatenate([vel_np_live, v_new[None, :]], axis=0)
                            masses_np_live = np.concatenate([masses_np_live, np.array([m_loss], dtype=np.float32)], axis=0)
                            charges_np_live = np.concatenate([charges_np_live, np.array([q_loss], dtype=np.float32)], axis=0)
                            if ionic_slots_np_live is not None:
                                ionic_slots_np_live = np.concatenate([ionic_slots_np_live, np.array([ionic_valence], dtype=np.int64)], axis=0)
                        colors_np_live = _colors_from_charges(charges_np_live)
                        rebuild_from_numpy(pos_np_live, vel_np_live, charges_np_live, masses_np_live, colors_np_live, springs_np_live, ionic_slots_np_live, temps_np_live)
                    else:
                        springs = springs[keep]
                        rest_angles = rest_angles[keep]
                        springs_dirty = True
        def project_tangent_inplace(v_local, p_local, out):
            dot = (v_local * p_local).sum(dim=-1, keepdim=True)
            out.copy_(v_local)
            out.sub_(dot * p_local)
            return out

        def normalize_inplace(t):
            inv = torch.rsqrt(torch.clamp(torch.sum(t * t, dim=-1, keepdim=True), min=1e-12))
            t.mul_(inv)
            return t

        def expmap_step(p_local, v_local, dt_scale, out):
            # Move along the great circle using the exponential map; stays on-sphere numerically.
            w = v_local * dt_scale
            w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)
            eps = 1e-8
            cos_t = torch.cos(w_norm)
            sin_t = torch.sin(w_norm)
            w_hat = w / torch.clamp(w_norm, min=eps)
            out.copy_(p_local)
            out.mul_(cos_t)
            out.add_(w_hat * sin_t)
            # series fallback for very small steps to reduce cancellation
            small = (w_norm.squeeze(-1) < eps)
            if small.any():
                w_small = w[small]
                p_small = p_local[small]
                corr = w_small - 0.5 * (torch.sum(w_small * w_small, dim=-1, keepdim=True)) * p_small
                out[small].copy_(p_small + corr)
            normalize_inplace(out)
            return out

        def step_midpoint_batch(p_local, v_local, dt_local, buf):
            # p_local, v_local: (B,N,D); dt_local: (B,1,1)
            buf.ensure(p_local.shape[1], p_local.shape[2], p_local.shape[0])
            a0 = force_at(p_local, v_local, dt_local, out_buf=buf.a0)
            buf.v_mid.copy_(v_local)
            buf.v_mid.add_(a0 * (0.5 * dt_local))
            project_tangent_inplace(buf.v_mid, p_local, out=buf.v_mid)
            expmap_step(p_local, buf.v_mid, 0.5 * dt_local, out=buf.p_mid)

            a1 = force_at(buf.p_mid, buf.v_mid, dt_local, out_buf=buf.a1)
            buf.v_new.copy_(v_local)
            buf.v_new.add_(a1 * dt_local)
            project_tangent_inplace(buf.v_new, buf.p_mid, out=buf.v_new)

            expmap_step(p_local, buf.v_mid, dt_local, out=buf.p_new)
            project_tangent_inplace(buf.v_new, buf.p_new, out=buf.v_new)

            f_peak = torch.maximum(torch.linalg.norm(a0, dim=2).amax(dim=1), torch.linalg.norm(a1, dim=2).amax(dim=1))
            return buf.p_new, buf.v_new, f_peak

        def radius_from_mass_np(m_np: np.ndarray) -> np.ndarray:
            base = np.maximum(m_np, 1e-6)
            return radius_scale * np.cbrt(base) / mass_ref_cuberoot

        def rebuild_from_numpy(pos_np_live, vel_np_live, charges_np_live, masses_np_live, colors_np_live, springs_np_live, ionic_slots_np_live=None, temps_np_live=None):
            # colors must track node count; include in nonlocals so render has matching length
            nonlocal pos, vel, charges, masses, temps, springs, rest_angles, degree, bond_adj, pair_idx, force_buffers, step_buffers, state_pos_buf, state_vel_buf, state_pos, state_vel, state_springs, state_masses, state_charges, state_temps, springs_dirty, state_idx, state_gen, last_render_gen, pos_last_change, pos3d_buf, proj_mean_cpu, proj_cpu, backdrop_dirty, ionic_slots, mean, proj, colors
            pos = torch.as_tensor(pos_np_live, device=torch_device)
            vel = torch.as_tensor(vel_np_live, device=torch_device)
            charges = torch.as_tensor(charges_np_live, device=torch_device)
            masses = torch.as_tensor(masses_np_live, device=torch_device)
            if temps_np_live is None:
                temps_np_live = np.full((pos_np_live.shape[0],), float(params.get("temp", temp)), dtype=np.float32)
            temps = torch.as_tensor(temps_np_live, device=torch_device)
            colors_np_live = _colors_from_charges(charges_np_live)
            colors = colors_np_live.astype(np.float32)
            ionic_slots = torch.as_tensor(ionic_slots_np_live, device=torch_device) if ionic_slots_np_live is not None else None
            springs_np_live = springs_np_live.astype(np.float32) if springs_np_live.size else np.zeros((0, 4), dtype=np.float32)
            if ionic_bonds and ionic_slots is None:
                ionic_slots = torch.full((pos.shape[0],), ionic_valence, dtype=torch.int64, device=torch_device)
            degree_np_live = np.zeros(pos.shape[0], dtype=np.int64)
            if springs_np_live.size:
                for i, j, *_ in springs_np_live:
                    degree_np_live[int(i)] += 1
                    degree_np_live[int(j)] += 1
            springs = torch.as_tensor(springs_np_live, device=torch_device)
            rest_angles_np_live = _build_rest_angles(pos_np_live, springs_np_live)
            rest_angles = torch.as_tensor(rest_angles_np_live, device=torch_device)
            degree = torch.as_tensor(degree_np_live, device=torch_device)
            bond_adj = torch.zeros((pos.shape[0], pos.shape[0]), dtype=torch.bool, device=torch_device)
            if springs_np_live.size:
                si = springs_np_live[:, 0].astype(int)
                sj = springs_np_live[:, 1].astype(int)
                bond_adj[si, sj] = True
                bond_adj[sj, si] = True
            pair_idx = torch.as_tensor(_all_pairs(pos.shape[0]), device=torch_device)
            force_buffers.ensure(pos.shape[0], pos.shape[1])
            buf_count = max(3, max_recursion + 1)
            step_buffers = [_StepBuffers(torch_device, pos.shape[0], pos.shape[1]) for _ in range(buf_count)]
            state_pos_buf = [torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
            state_vel_buf = [torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
            for buf_p, buf_v in zip(state_pos_buf, state_vel_buf):
                buf_p.copy_(pos, non_blocking=False)
                buf_v.copy_(vel, non_blocking=False)
            state_pos = [buf.numpy() for buf in state_pos_buf]
            state_vel = [buf.numpy() for buf in state_vel_buf]
            state_springs = [springs_np_live.copy(), springs_np_live.copy()]
            state_masses = [masses_np_live.copy(), masses_np_live.copy()]
            state_charges = [charges_np_live.copy(), charges_np_live.copy()]
            state_temps = [temps_np_live.copy(), temps_np_live.copy()]
            state_temps = [temps_np_live.copy(), temps_np_live.copy()]
            springs_dirty = True
            state_idx = 0
            state_gen += 1
            last_render_gen = -1
            pos_last_change = pos.clone()
            mean, proj = _get_projection_np(pos_np_live)
            proj_mean_cpu = _to_torch_pinned_float(mean)
            proj_cpu = _to_torch_pinned_float(proj)
            pos3d_buf = [torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED) for _ in range(2)]
            pos3d_buf[0] = _ensure_projection(state_pos_buf[state_idx], pos3d_buf[0])
            pos3d_buf[1] = torch.empty_like(pos3d_buf[0])
            backdrop_dirty = True

        def diffuse_bonds(dt_local: float):
            nonlocal masses, charges, temps
            if springs.numel() == 0:
                return
            if diff_mass_rate <= 0 and diff_charge_rate <= 0:
                return
            i_idx = springs[:, 0].long()
            j_idx = springs[:, 1].long()
            if i_idx.numel() == 0:
                return
            if diff_mass_rate > 0:
                delta_m = (masses[j_idx] - masses[i_idx]) * (diff_mass_rate * dt_local)
                masses = masses.clone()
                masses.index_add_(0, i_idx, delta_m)
                masses.index_add_(0, j_idx, -delta_m)
                masses = torch.clamp(masses, min=1e-5)
            if diff_charge_rate > 0:
                delta_q = (charges[j_idx] - charges[i_idx]) * (diff_charge_rate * dt_local)
                charges = charges.clone()
                charges.index_add_(0, i_idx, delta_q)
                charges.index_add_(0, j_idx, -delta_q)

        def handle_accel_shatter(dt_local: float):
            """Atomize nodes whose acceleration exceeds the threshold."""
            nonlocal pos, vel, masses, charges, temps, ionic_slots, springs, rest_angles, degree, bond_adj, springs_dirty
            if accel_shatter_thresh <= 0:
                return
            try:
                acc = force_at(pos, vel, dt_local)
            except Exception:
                return
            acc_mag = torch.linalg.norm(acc, dim=1)
            temp_live = float(params.get("temp", temp))
            hot_temp = max(0.0, temp_live)
            cold_temp = max(0.0, -temp_live)
            hot_gain = 1.0 + 0.5 * hot_temp
            cold_gain = 1.0 + 0.4 * cold_temp
            mass_scale = torch.clamp(masses / max(mass_ref, 1e-6), min=0.25)
            # heavy nodes become easier to pop; very small nodes stay at baseline
            frag_gain = 1.0 + 0.75 * torch.sqrt(mass_scale)
            # Cold makes shatter less likely (raise threshold), heat makes it more likely (lower threshold)
            eff_thresh = accel_shatter_thresh * cold_gain / (hot_gain * frag_gain)
            # make temperature and acceleration collaborate: only hot temps add random pop chance
            hot_prob = min(0.98, 0.10 * hot_temp)
            hot_prob = max(hot_prob, 0.02 if hot_temp > 0.05 else 0.0)
            # additional chance rises with how far accel exceeds the per-node threshold
            overdrive = torch.clamp((acc_mag / torch.clamp(eff_thresh, min=1e-9)) - 1.0, min=0.0)
            over_prob = torch.clamp(0.15 * overdrive, max=0.65)
            rand_hot = (hot_prob > 0.0) & (torch.rand_like(acc_mag) < hot_prob)
            rand_over = (over_prob > 0.0) & (torch.rand_like(acc_mag) < over_prob)
            hit_mask = (acc_mag > eff_thresh) | rand_hot | rand_over
            if not hit_mask.any():
                return
            hit_idx = torch.nonzero(hit_mask, as_tuple=False).squeeze(-1).detach().cpu().numpy()
            if hit_idx.size == 0:
                return
            pos_np = pos.detach().cpu().numpy()
            vel_np = vel.detach().cpu().numpy()
            masses_np = masses.detach().cpu().numpy()
            charges_np = charges.detach().cpu().numpy()
            temps_np = temps.detach().cpu().numpy()
            springs_np = springs.detach().cpu().numpy()
            ionic_slots_np = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None
            for idx in hit_idx:
                if idx < 0 or idx >= pos_np.shape[0]:
                    continue
                m_v = masses_np[idx]
                q_v = charges_np[idx]
                if m_v <= 1e-6:
                    continue
                split_frac = np.clip(accel_shatter_fraction, 0.05, 0.8)
                m_split = m_v * split_frac
                if m_split <= 1e-6:
                    continue
                m_remain = max(1e-6, m_v - m_split)
                k = max(2, int(accel_shatter_k))
                m_share = np.full((k,), m_split / k, dtype=np.float32)
                q_split = q_v * split_frac
                sign_q = 1.0 if q_split >= 0 else -1.0
                abs_q = np.abs(q_split)
                if abs_q < 1e-12:
                    q_raw = np.zeros((k,), dtype=np.float32)
                else:
                    weights = np.random.dirichlet(alpha=np.ones(k, dtype=np.float32))
                    q_raw = (weights * abs_q * sign_q).astype(np.float32)
                # exact conservation against fp drift
                q_raw[0] += (q_split - float(q_raw.sum()))
                acc_vec = acc[idx].detach().cpu().numpy().astype(np.float32)
                p_v = pos_np[idx]
                v_v = vel_np[idx]
                acc_tan = acc_vec - np.dot(acc_vec, p_v) * p_v
                acc_norm = np.linalg.norm(acc_tan)
                if acc_norm < 1e-6:
                    acc_tan = np.random.standard_normal(size=p_v.shape).astype(np.float32)
                    acc_tan = acc_tan - np.dot(acc_tan, p_v) * p_v
                    acc_norm = np.linalg.norm(acc_tan)
                acc_dir = acc_tan / (acc_norm + 1e-6)
                offset_scale = 0.3 * radius_from_mass_np(np.array([m_v], dtype=np.float32))[0]
                for kk in range(k):
                    jitter = np.random.standard_normal(size=p_v.shape).astype(np.float32)
                    jitter = jitter - np.dot(jitter, p_v) * p_v
                    jit_norm = np.linalg.norm(jitter)
                    if jit_norm > 1e-6:
                        jitter = jitter / jit_norm
                    dir_vec = acc_dir + 0.4 * jitter
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm < 1e-6:
                        dir_vec = jitter if np.linalg.norm(jitter) > 1e-6 else acc_dir
                        dir_norm = np.linalg.norm(dir_vec)
                    dir_vec = dir_vec / (dir_norm + 1e-6)
                    offset = offset_scale * dir_vec
                    p_new = _normalize_rows((p_v + offset)[None, :])[0]
                    v_new = v_v + dir_vec * (accel_shatter_kick * np.linalg.norm(acc_vec))
                    v_new = v_new - np.dot(v_new, p_new) * p_new
                    pos_np = np.concatenate([pos_np, p_new[None, :]], axis=0)
                    vel_np = np.concatenate([vel_np, v_new[None, :]], axis=0)
                    masses_np = np.concatenate([masses_np, np.array([m_share[kk]], dtype=np.float32)], axis=0)
                    charges_np = np.concatenate([charges_np, np.array([q_raw[kk]], dtype=np.float32)], axis=0)
                    if ionic_slots_np is not None:
                        ionic_slots_np = np.concatenate([ionic_slots_np, np.array([ionic_valence], dtype=np.int64)], axis=0)
                    temps_np = np.concatenate([temps_np, np.array([temps_np[idx]], dtype=np.float32)], axis=0)
                masses_np[idx] = m_remain
                charges_np[idx] = q_v - q_split
                temps_np[idx] = temps_np[idx]  # preserve parent temp for remnant
            colors_np = _colors_from_charges(charges_np)
            rebuild_from_numpy(pos_np, vel_np, charges_np, masses_np, colors_np, springs_np, ionic_slots_np, temps_np)

        def handle_collisions(dt_local: float):
            nonlocal springs, node_selector_idx
            if collide_gain <= 0:
                return
            n_live = pos.shape[0]
            if n_live < 2:
                return
            max_speed_local = params.get("max_speed", base_max_speed)
            v_merge = merge_speed_frac * max_speed_local
            v_shatter = shatter_speed_frac * max_speed_local
            # choose candidate pairs
            if n_live <= collision_pair_limit and pair_idx.numel():
                pairs_np = pair_idx.detach().cpu().numpy()
            elif springs.numel():
                pairs_np = springs[:, :2].long().detach().cpu().numpy()
            else:
                return
            if pairs_np.size == 0:
                return
            pos_np_live = pos.detach().cpu().numpy()
            vel_np_live = vel.detach().cpu().numpy()
            masses_np_live = masses.detach().cpu().numpy()
            charges_np_live = charges.detach().cpu().numpy()
            temps_np_live = temps.detach().cpu().numpy()
            colors_np_live = np.asarray(colors, dtype=np.float32)
            # keep temps aligned with live positions to avoid mask/shape mismatches after prior rebuilds
            if temps_np_live.shape[0] != pos_np_live.shape[0]:
                n_live = pos_np_live.shape[0]
                if temps_np_live.shape[0] > n_live:
                    temps_np_live = temps_np_live[:n_live]
                else:
                    pad = np.full((n_live - temps_np_live.shape[0],), temps_np_live[-1] if temps_np_live.size else 0.0, dtype=np.float32)
                    temps_np_live = np.concatenate([temps_np_live, pad], axis=0)
            if colors_np_live.shape[0] != pos_np_live.shape[0]:
                n_live = pos_np_live.shape[0]
                if colors_np_live.shape[0] > n_live:
                    colors_np_live = colors_np_live[:n_live]
                else:
                    pad = np.tile(colors_np_live[-1], (n_live - colors_np_live.shape[0], 1)) if colors_np_live.size else np.zeros((n_live, 3), dtype=np.float32)
                    colors_np_live = np.concatenate([colors_np_live, pad], axis=0)
            springs_np_live = springs.detach().cpu().numpy()
            ionic_slots_np_live = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None
            radii_np = radius_from_mass_np(masses_np_live)
            to_remove = set()
            shards = []
            speeds = np.linalg.norm(vel_np_live[pairs_np[:, 0]] - vel_np_live[pairs_np[:, 1]], axis=1)
            order = np.argsort(-speeds)
            for idx in order:
                i = int(pairs_np[idx, 0])
                j = int(pairs_np[idx, 1])
                if i in to_remove or j in to_remove:
                    continue
                d = np.linalg.norm(pos_np_live[i] - pos_np_live[j])
                r_sum = (radii_np[i] + radii_np[j]) * collide_gain
                if d >= r_sum:
                    continue
                v_rel = speeds[idx]
                m_i = masses_np_live[i]
                m_j = masses_np_live[j]
                size_ratio = min(m_i, m_j) / max(m_i, m_j)
                if (v_rel <= v_merge) or (size_ratio >= merge_size_frac):
                    m_new = m_i + m_j
                    q_new = charges_np_live[i] + charges_np_live[j]
                    w_i = m_i / max(m_new, 1e-6)
                    w_j = 1.0 - w_i
                    p_new = _normalize_rows((w_i * pos_np_live[i] + w_j * pos_np_live[j])[None, :])[0]
                    v_new = w_i * vel_np_live[i] + w_j * vel_np_live[j]
                    v_new = v_new - np.dot(v_new, p_new) * p_new
                    c_new = w_i * colors_np_live[i] + w_j * colors_np_live[j]
                    pos_np_live[i] = p_new
                    vel_np_live[i] = v_new.astype(np.float32)
                    masses_np_live[i] = m_new
                    charges_np_live[i] = q_new
                    colors_np_live[i] = c_new.astype(np.float32)
                    to_remove.add(j)
                elif (v_rel >= v_shatter) and (size_ratio <= shatter_size_frac):
                    victim = i if m_i >= m_j else j
                    attacker = j if victim == i else i
                    if victim in to_remove:
                        continue
                    m_v = masses_np_live[victim]
                    q_v = charges_np_live[victim]
                    t_v = temps_np_live[victim]
                    k = max(2, min(shatter_k_max, int(2 + (v_rel / max_speed_local) * 2)))
                    m_share = np.full((k,), m_v / k, dtype=np.float32)
                    # split charge conservatively: keep shard signs aligned with victim
                    sign_q = 1.0 if q_v >= 0 else -1.0
                    abs_q = np.abs(q_v)
                    if abs_q < 1e-12:
                        q_share = np.zeros((k,), dtype=np.float32)
                    else:
                        weights = np.random.dirichlet(alpha=np.ones(k, dtype=np.float32))
                        q_share = (weights * abs_q * sign_q).astype(np.float32)
                    # ensure exact conservation despite fp noise
                    q_share[0] += (q_v - float(q_share.sum()))
                    p_v = pos_np_live[victim]
                    v_att = vel_np_live[attacker]
                    dir_att = v_att - np.dot(v_att, p_v) * p_v
                    if np.linalg.norm(dir_att) < 1e-6:
                        dir_att = np.random.standard_normal(size=p_v.shape).astype(np.float32)
                    dir_att = dir_att / (np.linalg.norm(dir_att) + 1e-6)
                    for kk in range(k):
                        jitter = np.random.standard_normal(size=p_v.shape).astype(np.float32)
                        jitter = jitter - np.dot(jitter, p_v) * p_v
                        jitter_norm = np.linalg.norm(jitter)
                        if jitter_norm > 1e-6:
                            jitter = jitter / jitter_norm
                        offset = 0.5 * radii_np[victim] * jitter
                        p_new = _normalize_rows((p_v + offset)[None, :])[0]
                        v_bias = dir_att * (v_rel * 0.25)
                        v_new = vel_np_live[victim] + v_bias
                        v_new = v_new - np.dot(v_new, p_new) * p_new
                        shards.append((p_new.astype(np.float32), v_new.astype(np.float32), m_share[kk], q_share[kk], colors_np_live[victim].astype(np.float32), t_v))
                    to_remove.add(victim)
                else:
                    continue

            if not to_remove and not shards:
                return
            keep_mask = np.ones(pos_np_live.shape[0], dtype=bool)
            if to_remove:
                safe_remove = [idx for idx in to_remove if 0 <= idx < keep_mask.shape[0]]
                if safe_remove:
                    keep_mask[safe_remove] = False
            pos_np_live = pos_np_live[keep_mask]
            vel_np_live = vel_np_live[keep_mask]
            masses_np_live = masses_np_live[keep_mask]
            charges_np_live = charges_np_live[keep_mask]
            colors_np_live = colors_np_live[keep_mask]
            if ionic_slots_np_live is not None:
                ionic_slots_np_live = ionic_slots_np_live[keep_mask]
            temps_np_live = temps_np_live[keep_mask]
            # remove springs touching removed nodes and remap indices
            if springs_np_live.size:
                keep_edges_mask = keep_mask[springs_np_live[:, 0].astype(int)] & keep_mask[springs_np_live[:, 1].astype(int)]
                springs_np_live = springs_np_live[keep_edges_mask]
                remap = -np.ones(keep_mask.shape[0] + len(to_remove), dtype=np.int64)
                remap[np.nonzero(keep_mask)[0]] = np.arange(np.count_nonzero(keep_mask))
                if springs_np_live.size:
                    springs_np_live[:, 0] = remap[springs_np_live[:, 0].astype(int)]
                    springs_np_live[:, 1] = remap[springs_np_live[:, 1].astype(int)]
            # append shards (start without springs)
            if shards:
                for p_new, v_new, m_new, q_new, c_new, t_new in shards:
                    pos_np_live = np.concatenate([pos_np_live, p_new[None, :]], axis=0)
                    vel_np_live = np.concatenate([vel_np_live, v_new[None, :]], axis=0)
                    masses_np_live = np.concatenate([masses_np_live, np.array([m_new], dtype=np.float32)], axis=0)
                    charges_np_live = np.concatenate([charges_np_live, np.array([q_new], dtype=np.float32)], axis=0)
                    colors_np_live = np.concatenate([colors_np_live, c_new[None, :]], axis=0)
                    if ionic_slots_np_live is not None:
                        ionic_slots_np_live = np.concatenate([ionic_slots_np_live, np.array([ionic_valence], dtype=np.int64)], axis=0)
                    temps_np_live = np.concatenate([temps_np_live, np.array([t_new], dtype=np.float32)], axis=0)
            rebuild_from_numpy(pos_np_live, vel_np_live, charges_np_live, masses_np_live, colors_np_live, springs_np_live, ionic_slots_np_live, temps_np_live)
            node_selector_idx = min(node_selector_idx, max(0, pos.shape[0] - 1))

        def update_temperatures(dt_local: float):
            nonlocal temps
            ambient = float(params.get("temp", temp))
            cool_rate = float(params.get("temp_cool_rate", 0.0))
            heat_rate = float(params.get("temp_heat_rate", 0.0))
            # realign temps to live node count to tolerate recent add/remove
            n_live = pos.shape[0]
            if temps.shape[0] != n_live:
                if temps.shape[0] > n_live:
                    temps = temps[:n_live]
                else:
                    fill_val = temps[-1] if temps.numel() else torch.tensor(ambient, device=temps.device, dtype=temps.dtype)
                    pad = torch.full((n_live - temps.shape[0],), float(fill_val), device=temps.device, dtype=temps.dtype)
                    temps = torch.cat([temps, pad], dim=0)
            if cool_rate > 0:
                temps = temps + (ambient - temps) * (cool_rate * dt_local)
            if heat_rate > 0:
                v_norm = torch.linalg.norm(vel, dim=1)
                temps = temps + heat_rate * v_norm * dt_local
            temps = torch.clamp(temps, -1e6, 1e6)

        def vaporize_small_nodes():
            nonlocal pos, vel, masses, charges, temps, ionic_slots, springs, rest_angles, degree, bond_adj, pair_idx, springs_dirty, ionic_vapor_mass, ionic_vapor_charge, state_idx
            thresh = float(params.get("mass_vapor_thresh", 0.0))
            if thresh <= 0:
                return
            mask = masses < thresh
            if not mask.any():
                return
            idx_small = torch.nonzero(mask, as_tuple=False).squeeze(-1).cpu().numpy()
            if idx_small.size == 0:
                return
            pos_np = pos.detach().cpu().numpy()
            vel_np = vel.detach().cpu().numpy()
            masses_np = masses.detach().cpu().numpy()
            charges_np = charges.detach().cpu().numpy()
            temps_np = temps.detach().cpu().numpy()
            springs_np = springs.detach().cpu().numpy()
            ionic_slots_np = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None
            ionic_vapor_mass += float(np.maximum(masses_np[idx_small], 0.0).sum())
            ionic_vapor_charge += float(charges_np[idx_small].sum())
            keep_mask = np.ones(pos_np.shape[0], dtype=bool)
            keep_mask[idx_small] = False
            pos_np = pos_np[keep_mask]
            vel_np = vel_np[keep_mask]
            masses_np = masses_np[keep_mask]
            charges_np = charges_np[keep_mask]
            temps_np = temps_np[keep_mask]
            colors_np = _colors_from_charges(charges_np)
            if ionic_slots_np is not None:
                ionic_slots_np = ionic_slots_np[keep_mask]
            if springs_np.size:
                keep_edges_mask = keep_mask[springs_np[:, 0].astype(int)] & keep_mask[springs_np[:, 1].astype(int)]
                springs_np = springs_np[keep_edges_mask]
                remap = -np.ones(keep_mask.shape[0] + idx_small.size, dtype=np.int64)
                remap[np.nonzero(keep_mask)[0]] = np.arange(np.count_nonzero(keep_mask))
                if springs_np.size:
                    springs_np[:, 0] = remap[springs_np[:, 0].astype(int)]
                    springs_np[:, 1] = remap[springs_np[:, 1].astype(int)]
            rebuild_from_numpy(pos_np, vel_np, charges_np, masses_np, colors_np, springs_np, ionic_slots_np, temps_np)
            state_idx = 0

        def condense_vapor():
            nonlocal ionic_vapor_mass, ionic_vapor_charge, pos, vel, charges, masses, temps, ionic_slots
            ambient = float(params.get("temp", temp))
            temp_gate = float(params.get("condense_temp_thresh", 0.0))
            # scale condensation chance with cold: colder temps condense more mass
            if ionic_vapor_mass <= 1e-9:
                return
            if ambient > temp_gate:
                return
            cold_gain = 1.0 + max(0.0, -ambient) * 1.5
            base_chunk = float(params.get("condense_chunk_mass", 0.0))
            chunk = base_chunk * cold_gain
            chunk = float(min(chunk, ionic_vapor_mass))
            if chunk <= 0:
                return
            if ionic_vapor_mass <= 1e-9:
                return
            if chunk <= 0:
                return
            q_chunk = ionic_vapor_charge * (chunk / ionic_vapor_mass) if ionic_vapor_mass > 0 else 0.0
            ionic_vapor_mass -= chunk
            ionic_vapor_charge -= q_chunk
            pos_np = pos.detach().cpu().numpy()
            vel_np = vel.detach().cpu().numpy()
            charges_np = charges.detach().cpu().numpy()
            masses_np = masses.detach().cpu().numpy()
            temps_np = temps.detach().cpu().numpy()
            springs_np = springs.detach().cpu().numpy()
            ionic_slots_np = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None
            if pos_np.shape[0] > 0:
                anchor_idx = np.random.randint(0, pos_np.shape[0])
                p_anchor = pos_np[anchor_idx]
                jitter = np.random.standard_normal(size=p_anchor.shape).astype(np.float32)
                jitter = jitter - np.dot(jitter, p_anchor) * p_anchor
                jit_norm = np.linalg.norm(jitter)
                if jit_norm > 1e-6:
                    jitter = jitter / jit_norm
                offset = 0.4 * radius_from_mass_np(np.array([chunk], dtype=np.float32))[0] * jitter
                # keep as 2D (1, dim) so concatenation with pos_np (N, dim) works
                p_new = _normalize_rows((p_anchor + offset)[None, :])
            else:
                jitter = np.random.standard_normal(size=(pos.shape[1],)).astype(np.float32)
                jitter = jitter / (np.linalg.norm(jitter) + 1e-6)
                p_new = jitter[None, :]
            v_new = np.zeros_like(p_new, dtype=np.float32)
            pos_np = np.concatenate([pos_np, p_new], axis=0)
            vel_np = np.concatenate([vel_np, v_new], axis=0)
            masses_np = np.concatenate([masses_np, np.array([chunk], dtype=np.float32)], axis=0)
            charges_np = np.concatenate([charges_np, np.array([q_chunk], dtype=np.float32)], axis=0)
            temps_np = np.concatenate([temps_np, np.array([ambient], dtype=np.float32)], axis=0)
            if ionic_slots_np is not None:
                ionic_slots_np = np.concatenate([ionic_slots_np, np.array([ionic_valence], dtype=np.int64)], axis=0)
            colors_np = _colors_from_charges(charges_np)
            rebuild_from_numpy(pos_np, vel_np, charges_np, masses_np, colors_np, springs_np, ionic_slots_np, temps_np)

        try:
            prev_dt = adaptive_dt
            stagnant_steps = 0
            while not stop_evt.is_set():
                with pos_lock:
                    # If shapes drift, trust the tensor shape as the live dimension and realign fields
                    if pos.shape[1] != current_n_dim:
                        current_n_dim = pos.shape[1]
                        target_n_dim = current_n_dim
                        B_mat_np = build_B_mat(current_n_dim)
                        B_mat_torch = torch.as_tensor(B_mat_np, device=torch_device) if B_mat_np is not None else None
                    # Apply pending node mutations (add/remove) to avoid full rebuild
                    if pending_node_add or pending_node_remove:
                        # consolidate removals
                        if pending_node_remove:
                            remove_idx = sorted(set(int(i) for i in pending_node_remove if i >= 0))
                        else:
                            remove_idx = []
                        add_count = int(pending_node_add)
                        pending_node_add = 0
                        pending_node_remove = []

                        # Current state to numpy for manipulations
                        pos_np_live = pos.detach().cpu().numpy()
                        vel_np_live = vel.detach().cpu().numpy()
                        charges_np_live = charges.detach().cpu().numpy()
                        masses_np_live = masses.detach().cpu().numpy()
                        temps_np_live = temps.detach().cpu().numpy()
                        springs_np_live = springs.detach().cpu().numpy()
                        colors_np_live = colors
                        ionic_slots_np_live = ionic_slots.detach().cpu().numpy() if ionic_slots is not None else None

                        # Remove nodes if requested
                        if remove_idx:
                            n_cur = pos_np_live.shape[0]
                            keep_mask = np.ones(n_cur, dtype=bool)
                            keep_mask[remove_idx] = False
                            keep_ids = np.nonzero(keep_mask)[0]
                            # remap indices for springs
                            remap = -np.ones(n_cur, dtype=np.int64)
                            remap[keep_ids] = np.arange(len(keep_ids), dtype=np.int64)
                            if springs_np_live.size:
                                keep_edges_mask = keep_mask[springs_np_live[:, 0].astype(int)] & keep_mask[springs_np_live[:, 1].astype(int)]
                                springs_np_live = springs_np_live[keep_edges_mask]
                                springs_np_live[:, 0] = remap[springs_np_live[:, 0].astype(int)]
                                springs_np_live[:, 1] = remap[springs_np_live[:, 1].astype(int)]
                            pos_np_live = pos_np_live[keep_mask]
                            vel_np_live = vel_np_live[keep_mask]
                            charges_np_live = charges_np_live[keep_mask]
                            masses_np_live = masses_np_live[keep_mask]
                            temps_np_live = temps_np_live[keep_mask]
                            colors_np_live = colors_np_live[keep_mask]
                            if ionic_slots_np_live is not None:
                                ionic_slots_np_live = ionic_slots_np_live[keep_mask]

                        # Add nodes if requested
                        if add_count > 0:
                            rng_local = np.random.default_rng(seed + state_gen + int(time.time()))
                            new_pos = _sobol_like_sphere(add_count, current_n_dim, rng_local.integers(0, 1_000_000))
                            new_pos = _relax_repulsion(new_pos, iters=6, step=0.05)
                            new_vel = np.zeros_like(new_pos, dtype=np.float32)
                            species_choices = np.array([-2, -1, 1, 2], dtype=np.int64)
                            species_new = rng_local.choice(species_choices, size=add_count)
                            charges_new = species_new.astype(np.float32)
                            masses_new = np.maximum(0.1, 1.0 + 0.4 * np.abs(charges_new))
                            colors_new = _colors_from_charges(charges_new)
                            pos_np_live = np.concatenate([pos_np_live, new_pos], axis=0)
                            vel_np_live = np.concatenate([vel_np_live, new_vel], axis=0)
                            charges_np_live = np.concatenate([charges_np_live, charges_new], axis=0)
                            masses_np_live = np.concatenate([masses_np_live, masses_new], axis=0)
                            temps_np_live = np.concatenate([temps_np_live, np.full((add_count,), float(params.get("temp", temp)), dtype=np.float32)], axis=0)
                            colors_np_live = np.concatenate([colors_np_live, colors_new], axis=0)
                            if ionic_slots_np_live is not None:
                                ionic_slots_np_live = np.concatenate([ionic_slots_np_live, np.full((add_count,), ionic_valence, dtype=np.int64)], axis=0)

                        # normalize positions back to sphere
                        pos_np_live = _normalize_rows(pos_np_live.astype(np.float32))
                        vel_np_live = vel_np_live.astype(np.float32)

                        # regenerate charge-based colors for consistency
                        colors_np_live = _colors_from_charges(charges_np_live)

                        # rebuild tensors
                        pos = torch.as_tensor(pos_np_live, device=torch_device)
                        vel = torch.as_tensor(vel_np_live, device=torch_device)
                        charges = torch.as_tensor(charges_np_live, device=torch_device)
                        masses = torch.as_tensor(masses_np_live, device=torch_device)
                        temps = torch.as_tensor(temps_np_live, device=torch_device)
                        colors = colors_np_live.astype(np.float32)
                        if ionic_slots_np_live is not None:
                            ionic_slots = torch.as_tensor(ionic_slots_np_live, device=torch_device)
                        else:
                            ionic_slots = None

                        # rebuild springs/rest/degree/bond adjacency
                        springs_np_live = springs_np_live.astype(np.float32) if springs_np_live.size else np.zeros((0, 4), dtype=np.float32)
                        if ionic_bonds and ionic_slots is None:
                            ionic_slots = torch.full((pos.shape[0],), ionic_valence, dtype=torch.int64, device=torch_device)
                        degree_np_live = np.zeros(pos.shape[0], dtype=np.int64)
                        if springs_np_live.size:
                            for i, j, *_ in springs_np_live:
                                degree_np_live[int(i)] += 1
                                degree_np_live[int(j)] += 1
                        springs = torch.as_tensor(springs_np_live, device=torch_device)
                        rest_angles_np_live = _build_rest_angles(pos_np_live, springs_np_live)
                        rest_angles = torch.as_tensor(rest_angles_np_live, device=torch_device)
                        degree = torch.as_tensor(degree_np_live, device=torch_device)
                        bond_adj = torch.zeros((pos.shape[0], pos.shape[0]), dtype=torch.bool, device=torch_device)
                        if springs_np_live.size:
                            si = springs_np_live[:, 0].astype(int)
                            sj = springs_np_live[:, 1].astype(int)
                            bond_adj[si, sj] = True
                            bond_adj[sj, si] = True

                        # projection basis and buffers
                        mean, proj = _get_projection_np(pos_np_live)
                        pair_idx = torch.as_tensor(_all_pairs(pos.shape[0]), device=torch_device)
                        force_buffers.ensure(pos.shape[0], pos.shape[1])
                        buf_count = max(3, max_recursion + 1)
                        step_buffers = [_StepBuffers(torch_device, pos.shape[0], pos.shape[1]) for _ in range(buf_count)]
                        state_pos_buf = [torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
                        state_vel_buf = [torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
                        for buf_p, buf_v in zip(state_pos_buf, state_vel_buf):
                            buf_p.copy_(pos, non_blocking=False)
                            buf_v.copy_(vel, non_blocking=False)
                        state_pos = [buf.numpy() for buf in state_pos_buf]
                        state_vel = [buf.numpy() for buf in state_vel_buf]
                        state_springs = [springs_np_live.copy(), springs_np_live.copy()]
                        state_masses = [masses_np_live.copy(), masses_np_live.copy()]
                        state_charges = [charges_np_live.copy(), charges_np_live.copy()]
                        state_temps = [temps_np_live.copy(), temps_np_live.copy()]
                        state_temps = [temps_np_live.copy(), temps_np_live.copy()]
                        springs_dirty = True
                        state_idx = 0
                        state_gen += 1
                        last_render_gen = -1
                        pos_last_change = pos.clone()
                        pos3d_buf = [torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED) for _ in range(2)]
                        pos3d_buf[0] = _ensure_projection(state_pos_buf[state_idx], pos3d_buf[0])
                        pos3d_buf[1] = torch.empty_like(pos3d_buf[0])
                        proj_mean_cpu = _to_torch_pinned_float(mean)
                        proj_cpu = _to_torch_pinned_float(proj)
                        backdrop_dirty = True
                        node_selector_idx = min(node_selector_idx, max(0, pos.shape[0] - 1))
                    # Apply queued dimension changes once, then clear
                    if pending_dim_change is not None:
                        new_dim = pending_dim_change
                        pending_dim_change = None
                        rebuild_state(new_dim, with_lock=False)
                        # After rebuild, enforce tensors to match the applied dimension; pad with zeros when growing
                        current_n_dim = new_dim
                        target_n_dim = new_dim
                        if pos.shape[1] < current_n_dim:
                            pad = current_n_dim - pos.shape[1]
                            pos = torch.cat(
                                [pos, torch.zeros((pos.shape[0], pad), device=pos.device, dtype=pos.dtype)], dim=1
                            )
                            vel = torch.cat(
                                [vel, torch.zeros((vel.shape[0], pad), device=vel.device, dtype=vel.dtype)], dim=1
                            )
                        elif pos.shape[1] > current_n_dim:
                            pos = pos[:, :current_n_dim].contiguous()
                            vel = vel[:, :current_n_dim].contiguous()
                        # Rebuild Lorentz matrix for the new dimension to avoid stale shapes
                        B_mat_np = build_B_mat(current_n_dim)
                        B_mat_torch = torch.as_tensor(B_mat_np, device=torch_device) if B_mat_np is not None else None
                        # hard-reset CPU buffers to the new shape to avoid stale dims
                        state_pos_buf = [torch.empty((pos.shape[0], pos.shape[1]), device="cpu", dtype=pos.dtype, pin_memory=PIN_ENABLED) for _ in range(2)]
                        state_vel_buf = [torch.empty((vel.shape[0], vel.shape[1]), device="cpu", dtype=vel.dtype, pin_memory=PIN_ENABLED) for _ in range(2)]
                        for buf_p, buf_v in zip(state_pos_buf, state_vel_buf):
                            buf_p.copy_(pos, non_blocking=False)
                            buf_v.copy_(vel, non_blocking=False)
                        state_pos = [buf.numpy() for buf in state_pos_buf]
                        state_vel = [buf.numpy() for buf in state_vel_buf]
                        state_temps = [temps.detach().cpu().numpy(), temps.detach().cpu().numpy()]
                        state_idx = 0
                        state_gen += 1
                        last_render_gen = -1
                        pos3d_buf = [torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED) for _ in range(2)]
                        pos3d_buf[0] = _ensure_projection(state_pos_buf[state_idx], pos3d_buf[0])
                        pos3d_buf[1] = torch.empty_like(pos3d_buf[0])
                        print(
                            f"[info] dimension change applied; live_dim={current_n_dim} pos_shape={tuple(pos.shape)} state_pos_shape={state_pos[0].shape if state_pos else 'n/a'}",
                            file=sys.stderr,
                            flush=True,
                        )
                    now = time.perf_counter()
                    elapsed = now - last_time
                    last_time = now

                    # clamp dt away from denorms; if we ever fell below the floor, snap back so motion can resume
                    if adaptive_dt < min_dt:
                        adaptive_dt = min_dt
                    dt_try = adaptive_dt
                    max_speed_local = params.get("max_speed", base_max_speed)
                    # avoid denormal/zero underflow which would freeze motion
                    tiny_norm = np.finfo(np.float64).tiny
                    if dt_try < tiny_norm:
                        dt_try = max(min_dt, tiny_norm)
                        adaptive_dt = dt_try
                        dt_change_reason = "denorm clamp"
                    peak_disp = 0.0
                    peak_f = 0.0
                    dt_change_reason = "hold"

                    dim_scale = math.sqrt(max(current_n_dim, 1) / 3.0)
                    disp_base = min(max_speed_local * base_dt * dim_scale, 1.8)
                    d_hi = disp_base * (1.0 + subdiv_margin_hi)
                    d_lo = disp_base * (1.0 - grow_margin_lo)
                    f_impulse_cap = force_limit * base_dt
                    # self-tuning two-candidate set; only widen after a failure
                    dt_mode = params.get("dt_mode", "single-posthoc")
                    if dt_mode == "two-level":
                        dt_candidates = torch.as_tensor(
                            [base_dt, max(min_dt, base_dt * 0.5)], device=torch_device, dtype=pos.dtype
                        )
                        dt_candidates, _ = torch.sort(dt_candidates.unique())
                    elif dt_mode == "single-posthoc":
                        dt_candidates = torch.as_tensor([adaptive_dt], device=torch_device, dtype=pos.dtype)
                    else:
                        dt_pair = torch.as_tensor([fast_dt, slow_dt], device=torch_device, dtype=pos.dtype)
                        dt_pair, _ = torch.sort(dt_pair.unique())
                        if last_fail:
                            wide_candidates = torch.as_tensor(
                                [
                                    dt_try,
                                    min(base_dt, dt_try * grow_rate),
                                    max(min_dt, dt_try * shrink_rate),
                                    max(min_dt, dt_try * 0.1),
                                    max(min_dt, dt_try * 0.01),
                                ],
                                device=torch_device,
                                dtype=pos.dtype,
                            )
                            dt_candidates, _ = torch.sort(torch.cat([dt_pair, wide_candidates]).unique())
                        else:
                            dt_candidates = dt_pair
                    dt_batch = dt_candidates.view(-1, 1, 1)
                    pos_batch = pos.unsqueeze(0).expand(dt_batch.shape[0], -1, -1)
                    vel_batch = vel.unsqueeze(0).expand(dt_batch.shape[0], -1, -1)
                    buf = step_buffers[0]
                    p_trial_batch, v_trial_batch, f_peak_batch = step_midpoint_batch(pos_batch, vel_batch, dt_batch, buf)
                    disp_batch = torch.linalg.norm(p_trial_batch - pos_batch, dim=2).amax(dim=1)
                    f_impulse_batch = f_peak_batch * dt_candidates
                    acceptable = (disp_batch <= d_hi) & (f_impulse_batch <= f_impulse_cap)
                dt_min_view_t = dt_candidates.min()

                dt_vals = dt_candidates
                if dt_mode == "single-posthoc":
                    chosen_idx_t = 0
                    dt_chosen = float(dt_vals[0].item())
                    disp_chosen = float(disp_batch[0].item())
                    imp_chosen = float(f_impulse_batch[0].item())
                    ok = bool(acceptable[0].item())
                    if ok:
                        slack_disp = d_lo / max(disp_chosen, 1e-9)
                        slack_imp = f_impulse_cap / max(imp_chosen, 1e-9)
                        slack = min(slack_disp, slack_imp)
                        boost = 0.1 * min(max(slack - 1.0, 0.0), 1.0)
                        adaptive_dt = min(base_dt, max(min_dt, dt_chosen * (1.0 + boost)))
                        dt_change_reason = "accept_single"
                        last_fail = False
                    else:
                        adaptive_dt = max(min_dt, dt_chosen * shrink_rate)
                        dt_change_reason = "shrink_single"
                        last_fail = True
                    fast_dt = adaptive_dt
                    slow_dt = adaptive_dt
                else:
                    if acceptable.any():
                        scores = torch.where(acceptable, dt_vals, torch.full_like(dt_vals, -1.0))
                        chosen_idx_t = torch.argmax(scores)
                        dt_chosen = float(dt_vals[chosen_idx_t].item())
                        disp_chosen = float(disp_batch[chosen_idx_t].item())
                        imp_chosen = float(f_impulse_batch[chosen_idx_t].item())
                        # update EMA toward the chosen dt
                        alpha = 0.2
                        accept_ema = (1.0 - alpha) * accept_ema + alpha * dt_chosen
                        # compute slack to push upward when we have margin
                        slack_disp = d_lo / max(disp_chosen, 1e-9)
                        slack_imp = f_impulse_cap / max(imp_chosen, 1e-9)
                        slack = min(slack_disp, slack_imp)
                        boost = 0.1 * min(max(slack - 1.0, 0.0), 1.0)
                        if dt_mode == "two-level":
                            fast_dt = dt_vals.max().item()
                            slow_dt = dt_vals.min().item()
                        else:
                            fast_dt = min(base_dt, accept_ema * (1.0 + boost))
                            slow_dt = max(min_dt, fast_dt * 0.5)
                        last_fail = False
                        dt_change_reason = "accept_multi"
                    else:
                        # if all failed, fall back to smallest and widen next round; bias EMAs lower
                        chosen_idx_t = torch.argmin(disp_batch)
                        dt_chosen = float(dt_vals[chosen_idx_t].item())
                        alpha_fail = 0.3
                        fail_ema = (1.0 - alpha_fail) * fail_ema + alpha_fail * dt_chosen
                        if dt_mode == "two-level":
                            fast_dt = dt_vals.max().item()
                            slow_dt = dt_vals.min().item()
                        else:
                            fast_dt = max(min_dt, min(fast_dt, fail_ema) * 0.8)
                            slow_dt = max(min_dt, fast_dt * 0.5)
                        last_fail = True
                        dt_change_reason = "fallback_wide"

                dt_try_t = dt_vals[chosen_idx_t]
                p_trial = p_trial_batch[chosen_idx_t]
                v_trial = v_trial_batch[chosen_idx_t]
                peak_disp_t = disp_batch[chosen_idx_t]
                peak_f_t = f_peak_batch[chosen_idx_t]
                work_est_t = f_impulse_batch[chosen_idx_t] * disp_batch[chosen_idx_t]

                    # ensure dimensionality stays consistent; rebuild buffers on mismatch
                if p_trial.shape[1] != pos.shape[1]:
                    current_n_dim = p_trial.shape[1]
                    pos = p_trial.clone()
                    vel = v_trial.clone()
                    # keep Lorentz matrix aligned to live dimension
                    B_mat_np = build_B_mat(current_n_dim)
                    B_mat_torch = torch.as_tensor(B_mat_np, device=torch_device) if B_mat_np is not None else None
                    force_buffers.ensure(pos.shape[0], pos.shape[1])
                    buf_count = max(3, max_recursion + 1)
                    step_buffers = [_StepBuffers(torch_device, pos.shape[0], pos.shape[1]) for _ in range(buf_count)]
                    state_pos_buf = [torch.empty_like(pos, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
                    state_vel_buf = [torch.empty_like(vel, device="cpu", pin_memory=PIN_ENABLED) for _ in range(2)]
                    for buf_p, buf_v in zip(state_pos_buf, state_vel_buf):
                        buf_p.copy_(pos, non_blocking=False)
                        buf_v.copy_(vel, non_blocking=False)
                    state_pos = [buf_p.numpy() for buf_p in state_pos_buf]
                    state_vel = [buf_v.numpy() for buf_v in state_vel_buf]
                    pos_last_change = pos.clone()
                    state_idx = 0
                    last_copy_time = now
                    proj_mean_cpu_np, proj_cpu_np = _get_projection_np(state_pos_buf[state_idx].cpu().numpy())
                    proj_mean_cpu = _to_torch_pinned_float(proj_mean_cpu_np)
                    proj_cpu = _to_torch_pinned_float(proj_cpu_np)
                    pos3d_buf = [torch.empty((pos.shape[0], 3), device="cpu", dtype=torch.float32, pin_memory=PIN_ENABLED) for _ in range(2)]
                    _ensure_projection(state_pos_buf[state_idx], pos3d_buf[state_idx])
                else:
                    pos.copy_(p_trial)
                    vel.copy_(v_trial)
                step_count += 1
                energy_prev = None
                dt_try = float(dt_try_t.item())
                phys_dt_display = dt_try
                if dt_mode != "single-posthoc":
                    adaptive_dt = min(base_dt, max(min_dt, dt_try))

                # form or break bonds based on the latest positions before copying to render buffers
                apply_bonds(pos)
                # diffuse mass/charge along bonds and resolve collisions
                diffuse_bonds(dt_try)
                handle_accel_shatter(dt_try)
                handle_collisions(dt_try)
                update_temperatures(dt_try)
                vaporize_small_nodes()
                condense_vapor()

                # detect dt changes and prolonged stagnation
                if dt_try != prev_dt:
                    prev_dt = dt_try
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1

                time_since_copy = now - last_copy_time
                do_copy = (time_since_copy >= copy_interval) or springs_dirty
                if do_copy:
                    next_idx = 1 - state_idx
                    state_pos_buf[next_idx].copy_(pos, non_blocking=True)
                    state_vel_buf[next_idx].copy_(vel, non_blocking=True)
                    state_masses[next_idx] = masses.detach().cpu().numpy()
                    state_charges[next_idx] = charges.detach().cpu().numpy()
                    state_temps[next_idx] = temps.detach().cpu().numpy()
                    # keep projection buffers in sync with current dimensionality/count
                    try:
                        _ensure_projection(state_pos_buf[next_idx], pos3d_buf[next_idx])
                    except Exception:
                        # brute-force rebuild on any mismatch to avoid breaking the render loop
                        if params.get("freeze_pca", False):
                            m_local, p_local = _adapt_projection_dim(mean, proj, state_pos_buf[next_idx].shape[1])
                        else:
                            m_local, p_local = _get_projection_np(state_pos_buf[next_idx].cpu().numpy())
                        proj_mean_cpu = _to_torch_pinned_float(m_local)
                        proj_cpu = _to_torch_pinned_float(p_local)
                        pos3d_buf[next_idx] = _ensure_projection(state_pos_buf[next_idx], pos3d_buf[next_idx])
                    # only sync lightweight diagnostics when we already copy to CPU
                    if pos_last_change.shape != pos.shape:
                        pos_last_change = pos.clone()
                        changed = True
                        max_delta = float("inf")
                    else:
                        max_delta = float(torch.linalg.norm(pos - pos_last_change, dim=1).max().item())
                        changed = max_delta > move_eps
                        if changed:
                            if pos_last_change.shape != pos.shape:
                                pos_last_change = pos.clone()
                            else:
                                pos_last_change.copy_(pos)
                    phys_fps = 1.0 / elapsed if (elapsed > 0 and changed) else 0.0
                    phys_dt_min_display = float(dt_min_view_t.item())
                    phys_steps_display = 1 if changed else 0
                    phys_depth_display = 0
                    peak_disp = float(peak_disp_t.item())
                    peak_f = float(peak_f_t.item())
                    f_limit_hi = force_limit * base_dt
                    f_impulse = peak_f * dt_try
                    work_est = float(work_est_t.item())
                    diag = {
                        "dt": dt_try,
                        "dt_min": min_dt,
                        "reason": dt_change_reason,
                        "dmax": peak_disp,
                        "d_hi": d_hi,
                        "d_lo": d_lo,
                        "d_cap": max_speed_local * adaptive_dt * math.sqrt(max(current_n_dim, 1) / 3.0) * 3.0,
                        "f_peak": peak_f,
                        "f_impulse": f_impulse,
                        "f_impulse_tol_hi": f_limit_hi,
                        "work_est": work_est,
                        "work_tol": work_est,
                    }
                    if springs_dirty:
                        state_springs[next_idx] = springs.detach().cpu().numpy()
                        springs_dirty = False
                    else:
                        state_springs[next_idx] = state_springs[state_idx]
                    state_vel = [buf.numpy() for buf in state_vel_buf]
                    state_idx = next_idx
                    last_copy_time = now
                state_gen += 1

        except Exception as exc:
            import traceback

            print(f"[phys thread error] {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            stop_evt.set()

    def watchdog(thread_ref: threading.Thread, stop_evt_ref: threading.Event):
        last_gen = -1
        stagnant = 0
        while not stop_evt_ref.is_set():
            time.sleep(1.0)
            if state_gen == last_gen:
                stagnant += 1
                if stagnant >= 2 and thread_ref.is_alive():
                    frames = sys._current_frames()
                    ftid = thread_ref.ident
                    if ftid in frames:
                        import traceback

                        stack_str = "".join(traceback.format_stack(frames[ftid]))
                        print(
                            f"[phys watchdog] stagnant={stagnant}s gen={state_gen} dt={adaptive_dt:.3e} display_dt={phys_dt_display:.3e} last_reason={diag.get('reason','?')}\n{stack_str}",
                            file=sys.stderr,
                            flush=True,
                        )
            else:
                last_gen = state_gen
                stagnant = 0

    def stop_physics_thread(timeout: float = 5.0) -> bool:
        """Signal physics to stop and wait; return True if it halted."""
        nonlocal phys_thread, stop_evt
        if phys_thread is None:
            return True
        stop_evt.set()
        phys_thread.join(timeout=timeout)
        if phys_thread.is_alive():
            return False
        phys_thread = None
        return True

    def start_physics_threads():
        nonlocal phys_thread, stop_evt
        if phys_thread is not None and phys_thread.is_alive():
            return
        phys_thread = threading.Thread(target=physics_worker, daemon=True)
        phys_thread.start()
        if watchdog_enabled:
            threading.Thread(target=watchdog, args=(phys_thread, stop_evt), daemon=True).start()

    # Initialize selection state before launching physics to avoid unbound closures
    node_selector_idx = 0
    start_physics_threads()

    pygame.init()
    # Ensure we actually get a depth buffer; otherwise the planet can't occlude far-side objects.
    try:
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    except Exception:
        pass
    pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    base.init_gl(1280, 720, float(fov_deg), float(z_near), float(z_far))
    try:
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
    except Exception:
        pass
    try:
        from OpenGL.GL import glGetIntegerv, GL_DEPTH_BITS

        print(f"GL depth bits: {int(glGetIntegerv(GL_DEPTH_BITS))}")
    except Exception:
        pass

    font = pygame.font.SysFont("consolas", 16)

    # --- Physics config split: World (planet/env) vs Particle (node sim) ---
    # Migrate legacy world_config.json:{world_config:{...}} into:
    # - particle_config.json:{particle_config:{...}}
    # - world_config.json:{world_env:{...}}
    try:
        legacy_wc = joystick_menu._load_persisted_block("world_config.json", "world_config")
        if legacy_wc:
            p_blk, w_blk = world_config_structs.split_legacy_world_config_block(legacy_wc)
            try:
                if p_blk and not joystick_menu._load_persisted_block("particle_config.json", "particle_config"):
                    joystick_menu._save_persisted_block("particle_config.json", "particle_config", p_blk)
            except Exception:
                pass
            try:
                if w_blk and not joystick_menu._load_persisted_block("world_config.json", "world_env"):
                    joystick_menu._save_persisted_block("world_config.json", "world_env", w_blk)
            except Exception:
                pass
    except Exception:
        pass

    # Apply persisted particle/world values to params before menu init.
    particle_cfg = world_config_structs.particle_config_from_params(params)
    world_env = world_config_structs.world_env_from_params(params)
    try:
        p_persist = joystick_menu._load_persisted_block("particle_config.json", "particle_config")
        if p_persist:
            world_config_structs.particle_config_update_from_dict(particle_cfg, p_persist)
    except Exception:
        pass
    try:
        w_persist = joystick_menu._load_persisted_block("world_config.json", "world_env")
        if w_persist:
            world_config_structs.world_env_update_from_dict(world_env, w_persist)
        else:
            legacy_wc = joystick_menu._load_persisted_block("world_config.json", "world_config")
            if isinstance(legacy_wc, dict) and "render_scale" in legacy_wc:
                world_env.render_scale = float(legacy_wc.get("render_scale"))
    except Exception:
        pass
    try:
        world_config_structs.apply_particle_config_to_params(particle_cfg, params)
    except Exception:
        pass
    try:
        world_config_structs.apply_world_env_to_params(world_env, params)
    except Exception:
        pass

    menu = _JoystickSideMenu.create(params=params)
    joystick = menu.joystick if menu else None
    axis_state: Dict[int, float] = menu.axis_state if menu else {}

    try:
        if menu is not None and hasattr(menu, "apply_render_scale"):
            menu.apply_render_scale(float(params.get("render_scale", 10.0)))
    except Exception:
        pass

    particle_field_specs = {
        "k_spring": {"step": 0.05},
        "k_coulomb": {"step": 0.05},
        "G": {"step": 0.05},
        "damping": {"step": 0.01, "min": 0.0},
        "temp": {"step": 0.05, "min": 0.0},
        "bond_shear": {"step": 0.05, "min": 0.0},
        "max_speed": {"step": 0.002, "min": 0.0},
        "south_strength": {"step": 0.05, "min": 0.0},
        "south_enabled": {"step": 1, "min": 0, "max": 1},
        "south_axis": {"step": 1, "min": 0},
    }

    world_field_specs = {
        "render_scale": {"step": 0.25, "min": 0.01},
        "sea_level_pressure_kpa": {"step": 1.0, "min": 0.0},
        "sea_level_temp_k": {"step": 1.0, "min": 0.0},
    }

    flight_motion = {}
    try:
        if menu is not None and isinstance(getattr(menu, "airplane", None), dict):
            flight_motion = menu.airplane.get("motion", {}) if isinstance(menu.airplane.get("motion", {}), dict) else {}
    except Exception:
        flight_motion = {}
    flight_cfg = world_config_structs.flight_physics_from_motion(flight_motion)
    flight_field_specs = {
        "inertial": {"step": 1, "min": 0, "max": 1},
        "auto_heading_on_move": {"step": 1, "min": 0, "max": 1},
        "base_speed": {"step": 0.01, "min": 0.0},
        "thrust_accel": {"step": 0.05, "min": 0.0},
        "throttle_rate": {"step": 0.05, "min": 0.0},
        "lift_k": {"step": 0.05, "min": 0.0},
        "drag_k": {"step": 0.01, "min": 0.0},
        "gravity_g": {"step": 0.01},
        "max_speed": {"step": 0.1, "min": 0.0},
    }
    try:
        fp_persist = joystick_menu._load_persisted_block("flight_physics.json", "flight_physics")
        if fp_persist:
            world_config_structs.flight_physics_update_from_dict(flight_cfg, fp_persist)
    except Exception:
        pass

    ball_cfg = world_config_structs.ballistics_default()
    ball_field_specs = {
        "max_points": {"step": 1, "min": 2, "max": 256},
    }
    try:
        b_persist = joystick_menu._load_persisted_block("ballistics.json", "ballistics")
        if b_persist:
            world_config_structs.ballistics_update_from_dict(ball_cfg, b_persist)
    except Exception:
        pass

    def _commit_particle_config(cfg_obj: ctypes.Structure) -> None:
        try:
            world_config_structs.apply_particle_config_to_params(cfg_obj, params)
        except Exception:
            pass

    def _commit_world_env(cfg_obj: ctypes.Structure) -> None:
        try:
            world_config_structs.apply_world_env_to_params(cfg_obj, params)
        except Exception:
            pass
        try:
            if menu is not None and hasattr(menu, "apply_render_scale"):
                menu.apply_render_scale(float(params.get("render_scale", 10.0)))
        except Exception:
            pass

    def _commit_flight_physics(cfg_obj: ctypes.Structure) -> None:
        try:
            if menu is None or not isinstance(getattr(menu, "airplane", None), dict):
                return
            motion = menu.airplane.get("motion", None)
            if not isinstance(motion, dict):
                motion = {}
                menu.airplane["motion"] = motion
            for fname, _ft in getattr(cfg_obj.__class__, "_fields_", []):
                key = str(fname)
                try:
                    motion[key] = float(getattr(cfg_obj, key)) if _ft in (ctypes.c_float, ctypes.c_double) else int(getattr(cfg_obj, key))
                except Exception:
                    pass
        except Exception:
            pass

    def _commit_ballistics(cfg_obj: ctypes.Structure) -> None:
        try:
            if weap_rt is not None and getattr(weap_rt, "sim", None) is not None:
                weap_rt.sim.max_points = int(getattr(cfg_obj, "max_points", 16) or 16)
        except Exception:
            pass

    # Always-on weapon loadout PIP (top-left), separate from world config.
    loadout = weapons_structs.weapon_loadout_default()
    # Migrate any legacy persisted loadout into the new schema before PIP init.
    try:
        loadout = weapons_structs.load_weapon_loadout_json("weapon_loadout.json")
        try:
            joystick_menu._save_persisted_block(
                "weapon_loadout.json",
                "weapon_loadout",
                joystick_menu._ctypes_struct_to_dict(loadout),
            )
        except Exception:
            pass
    except Exception:
        pass
    loadout_field_specs = weapons_structs.weapon_loadout_field_specs()
    cfg_pip: joystick_menu.CtypesStructEditorPip | None = None
    weap_rt: weapon_runtime.WeaponRuntime | None = None
    fire1_prev = False
    fire2_prev = False
    weapon_splines: list[dict] = []
    try:
        cfg_pip = joystick_menu.CtypesStructEditorPip(
            struct_obj=loadout,
            title="LOADOUT",
            persist_path="weapon_loadout.json",
            persist_key="weapon_loadout",
            field_specs=loadout_field_specs,
            allow_cancel_close=False,
            persist_on_change=True,
        )
        cfg_pip.open()
    except Exception:
        cfg_pip = None

    try:
        weap_rt = weapon_runtime.WeaponRuntime(stats_path="weapon_stats.json")
    except Exception:
        weap_rt = None

    # Ensure joystick.json exists and bind a menu button if missing.
    # If the menu button is not yet bound, show "press menu button", bind it,
    # save joystick.json, then open the main menu.
    menu_button: int | None = None
    try:
        menu_button, did_bind = joystick_menu.ensure_menu_button_binding(
            cfg_path="joystick.json",
            font=font,
            width=int(1280),
            height=int(720),
            joystick=joystick,
        )
        if did_bind and joystick is not None and menu_button is not None:
            action = joystick_menu.run_main_menu(
                font=font,
                width=int(1280),
                height=int(720),
                joystick=joystick,
                menu_button=int(menu_button),
                menu_context={
                    "ctypes_structs": {
                        "weapon_loadout": {
                            "struct": loadout,
                            "title": "LOADOUT",
                            "persist_path": "weapon_loadout.json",
                            "persist_key": "weapon_loadout",
                            "field_specs": loadout_field_specs,
                        },
                        "particle_config": {
                            "struct": particle_cfg,
                            "title": "PARTICLE",
                            "persist_path": "particle_config.json",
                            "persist_key": "particle_config",
                            "field_specs": particle_field_specs,
                            "on_commit": _commit_particle_config,
                            "persist_on_change": True,
                        },
                        "world_env": {
                            "struct": world_env,
                            "title": "WORLD",
                            "persist_path": "world_config.json",
                            "persist_key": "world_env",
                            "field_specs": world_field_specs,
                            "on_commit": _commit_world_env,
                            "persist_on_change": True,
                        },
                        "flight_physics": {
                            "struct": flight_cfg,
                            "title": "FLIGHT",
                            "persist_path": "flight_physics.json",
                            "persist_key": "flight_physics",
                            "field_specs": flight_field_specs,
                            "on_commit": _commit_flight_physics,
                            "persist_on_change": True,
                        },
                        "ballistics": {
                            "struct": ball_cfg,
                            "title": "BALLISTICS",
                            "persist_path": "ballistics.json",
                            "persist_key": "ballistics",
                            "field_specs": ball_field_specs,
                            "on_commit": _commit_ballistics,
                            "persist_on_change": True,
                        },
                    }
                },
            )
            if action == "quit":
                try:
                    if weap_rt is not None:
                        weap_rt.stop()
                except Exception:
                    pass
                pygame.quit()
                return
    except Exception:
        menu_button = None

    node_selector_idx = 0
    node_selector_cooldown = 0.25
    last_node_selector_time = 0.0
    fly_mode = False
    # projection helper already defined above before physics threads start

    orbit_enabled = False
    clock = pygame.time.Clock()
    last_render_gen = -1
    ghost_history = deque(maxlen=max(0, int(params.get("ghost_history", 0))))
    ghost_hue_cycles = params.get("ghost_hue_cycles", 1.0)
    backdrop_cache = {"fp": None, "dim": None, "lines": [], "line_colors": [], "axes": [], "axis_colors": []}
    backdrop_dirty = True
    springs_empty = np.zeros((0, 4), dtype=np.float32)

    # init overlay metrics
    peak_vmax = 0.0
    peak_f = 0.0
    rel_e = 0.0
    dt_change_reason = "init"

    # In ship flight-camera mode, Tab toggles the debug HUD.
    # Outside flight mode, the debug HUD must always be visible.
    flight_debug_hud_visible = True
    debug_hud_toggle_prev = False

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    orbit_enabled = not orbit_enabled
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    # Only toggle while flight camera is active (ship mode).
                    try:
                        proj_now = str(params.get("proj_mode", "pca") or "pca")
                        if menu is not None and menu.flight_active(proj_now):
                            flight_debug_hud_visible = not bool(flight_debug_hud_visible)
                    except Exception:
                        pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_y:
                    adaptive_dt = base_dt
                    dt_change_reason = "manual reset"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    # inflate the active node mass via mouse
                    with pos_lock:
                        if pos.shape[0] > 0:
                            idx = node_selector_idx % pos.shape[0]
                            mult = params.get("inflate_mult", 1.2)
                            masses[idx] = torch.clamp(masses[idx] * mult, min=1e-6)
                            state_masses[state_idx] = masses.detach().cpu().numpy()
                            state_masses[1 - state_idx] = state_masses[state_idx].copy()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_i, pygame.K_KP_PLUS):
                    # keyboard inflate for accessibility
                    with pos_lock:
                        if pos.shape[0] > 0:
                            idx = node_selector_idx % pos.shape[0]
                            mult = params.get("inflate_mult", 1.2)
                            masses[idx] = torch.clamp(masses[idx] * mult, min=1e-6)
                            state_masses[state_idx] = masses.detach().cpu().numpy()
                            state_masses[1 - state_idx] = state_masses[state_idx].copy()
                elif joystick and event.type == pygame.JOYAXISMOTION:
                    if menu:
                        menu.handle_event(event)
                elif joystick and event.type == pygame.JOYBUTTONDOWN:
                    # Menu button opens the main menu and must not be forwarded to
                    # the side-menu handler (which historically used button 0).
                    if menu_button is not None and int(event.button) == int(menu_button):
                        try:
                            action = joystick_menu.run_main_menu(
                                font=font,
                                width=int(1280),
                                height=int(720),
                                joystick=joystick,
                                menu_button=int(menu_button),
                                menu_context={
                                    "ctypes_structs": {
                                            "weapon_loadout": {
                                                "struct": loadout,
                                                "title": "LOADOUT",
                                                "persist_path": "weapon_loadout.json",
                                                "persist_key": "weapon_loadout",
                                                "field_specs": loadout_field_specs,
                                            },
                                        "particle_config": {
                                            "struct": particle_cfg,
                                            "title": "PARTICLE",
                                            "persist_path": "particle_config.json",
                                            "persist_key": "particle_config",
                                            "field_specs": particle_field_specs,
                                            "on_commit": _commit_particle_config,
                                            "persist_on_change": True,
                                        },
                                        "world_env": {
                                            "struct": world_env,
                                            "title": "WORLD",
                                            "persist_path": "world_config.json",
                                            "persist_key": "world_env",
                                            "field_specs": world_field_specs,
                                            "on_commit": _commit_world_env,
                                            "persist_on_change": True,
                                        },
                                        "flight_physics": {
                                            "struct": flight_cfg,
                                            "title": "FLIGHT",
                                            "persist_path": "flight_physics.json",
                                            "persist_key": "flight_physics",
                                            "field_specs": flight_field_specs,
                                            "on_commit": _commit_flight_physics,
                                            "persist_on_change": True,
                                        },
                                        "ballistics": {
                                            "struct": ball_cfg,
                                            "title": "BALLISTICS",
                                            "persist_path": "ballistics.json",
                                            "persist_key": "ballistics",
                                            "field_specs": ball_field_specs,
                                            "on_commit": _commit_ballistics,
                                            "persist_on_change": True,
                                        },
                                    }
                                },
                            )
                            if action == "quit":
                                running = False
                        except Exception:
                            pass
                        continue
                    if menu:
                        menu.handle_event(event)
                    if event.button == 0:
                        # handled by menu (lock + velocity reset)
                        pass
                    elif event.button == 1:
                        # In ship camera mode, button 1 is reserved for the render-only flight camera.
                        if not _is_ship_proj_mode(str(params.get("proj_mode", "pca") or "pca")):
                            fly_mode = not fly_mode
                            fly_state["active"] = fly_mode
                            fly_state["dirty"] = True
                            if not fly_mode:
                                fly_state["vec_cpu"] = np.zeros((current_n_dim,), dtype=np.float32)
                                fly_state["vec_torch"] = None
                    elif event.button == 2:
                        # reset fly vector and re-lock adjustments
                        fly_state["vec_cpu"] = np.zeros((current_n_dim,), dtype=np.float32)
                        fly_state["vec_torch"] = None
                        fly_state["dirty"] = True
                    elif event.button == 3:
                        # joystick button to inflate active node
                        with pos_lock:
                            if pos.shape[0] > 0:
                                idx = node_selector_idx % pos.shape[0]
                                mult = params.get("inflate_mult", 1.2)
                                masses[idx] = torch.clamp(masses[idx] * mult, min=1e-6)
                                state_masses[state_idx] = masses.detach().cpu().numpy()
                                state_masses[1 - state_idx] = state_masses[state_idx].copy()
                elif joystick and event.type == pygame.JOYBUTTONUP:
                    if menu_button is not None and int(event.button) == int(menu_button):
                        continue
                    if menu:
                        menu.handle_event(event)

            # If the window is closing, don't attempt any further GL calls.
            if not running:
                break

            # apply joystick adjustments every frame (poll axis to stay responsive)
            if joystick and menu:
                menu.poll_axes()

                dt_controls = 1.0 / 60.0
                accel_gain = 2.0
                now_t = pygame.time.get_ticks() * 0.001
                dead_zone = 0.6

                if fly_mode:
                    # axes drive thrust mapped to dimensions
                    n_axes = joystick.get_numaxes()
                    dim_live = state_pos[state_idx].shape[1] if state_pos else current_n_dim
                    fly_vec = np.zeros((dim_live,), dtype=np.float32)
                    for d in range(min(dim_live, n_axes)):
                        val = axis_state.get(d, 0.0)
                        if abs(val) < dead_zone:
                            val = 0.0
                        fly_vec[d] = float(val)
                    fly_state["vec_cpu"] = fly_vec
                    fly_state["vec_torch"] = None
                    fly_state["dirty"] = True
                    fly_state["active"] = True
                    fly_state["idx"] = node_selector_idx % max(1, state_pos[state_idx].shape[0] if state_pos else 1)
                else:
                    node_axis = float(axis_state.get(2, 0.0))
                    if abs(node_axis) > dead_zone and now_t - last_node_selector_time > node_selector_cooldown:
                        n_nodes = state_pos[state_idx].shape[0] if state_pos else 0
                        if n_nodes:
                            step = 1 if node_axis > 0 else -1
                            node_selector_idx = (node_selector_idx + step) % n_nodes
                            last_node_selector_time = now_t

                    def _on_proj_mode_changed(new_mode):
                        nonlocal mean, proj, proj_mean_cpu, proj_cpu, backdrop_dirty
                        mean2, proj2 = _projection_for(str(new_mode), pos)
                        mean, proj = mean2, proj2
                        proj_mean_cpu = _to_torch_pinned_float(mean)
                        proj_cpu = _to_torch_pinned_float(proj)
                        _ensure_projection(state_pos_buf[state_idx], pos3d_buf[state_idx])
                        backdrop_dirty = True

                    def _on_dim_change_requested(sign: int):
                        nonlocal target_n_dim, pending_dim_change, backdrop_dirty
                        delta_dim = int(sign)
                        new_target = max(2, int(target_n_dim) + delta_dim)
                        if pending_dim_change is None and new_target != current_n_dim:
                            target_n_dim = new_target
                            pending_dim_change = new_target
                            backdrop_dirty = True
                            print(
                                f"[info] dimension change requested to {new_target}; queued for physics thread",
                                file=sys.stderr,
                                flush=True,
                            )

                    mode_now_for_menu = str(params.get("proj_mode", "pca") or "pca")
                    if not menu.flight_active(mode_now_for_menu):
                        menu.update_params(
                            params,
                            now_t=now_t,
                            accel_gain=accel_gain,
                            dt_controls=dt_controls,
                            dead_zone=dead_zone,
                            allow_proj_mode=True,
                            allow_n_dim=True,
                            on_proj_mode_changed=_on_proj_mode_changed,
                            on_dim_change_requested=_on_dim_change_requested,
                        )

                # Config PIP (menu-nav bindings) tick.
                if cfg_pip is not None and joystick is not None:
                    try:
                        cfg_pip.tick(joystick=joystick)
                    except Exception:
                        pass

                # Enforce loadout constraints:
                # - bay_inner is only usable when bay_outer is deployed (non-zero).
                try:
                    if int(getattr(loadout, "bay_outer", 0)) == 0 and int(getattr(loadout, "bay_inner", 0)) != 0:
                        loadout.bay_inner = 0
                        try:
                            joystick_menu._save_persisted_block(
                                "weapon_loadout.json",
                                "weapon_loadout",
                                joystick_menu._ctypes_struct_to_dict(loadout),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

                # Fire dispatch (Weapon 1 / Weapon 2) -> projectile simulator queue.
                if joystick is not None and weap_rt is not None:
                    try:
                        cfg = joystick_menu.load_or_create_joystick_config("joystick.json")
                        if menu is not None:
                            try:
                                targeting_mode = int(getattr(loadout, "targeting", 0) or 0)
                            except Exception:
                                targeting_mode = 0
                            try:
                                menu._targeting_active = bool(int(targeting_mode) != 0)
                            except Exception:
                                pass
                            try:
                                menu._active_control_set = _resolve_effective_control_set(menu)
                            except Exception:
                                pass

                        set_name = _resolve_effective_control_set(menu)
                        b1 = _get_set_binding(cfg, set_name=set_name, group="weapons", key="fire_1")
                        b2 = _get_set_binding(cfg, set_name=set_name, group="weapons", key="fire_2")
                        axes_now, buttons_now, hats_now = joystick_menu._poll_joystick_snapshot(joystick)
                        f1_active, f1_analog = _binding_active(b1, axes_now, buttons_now, hats_now)
                        f2_active, f2_analog = _binding_active(b2, axes_now, buttons_now, hats_now)

                        # Bindable debug HUD toggle (joystick.json -> flight_controls.hud.toggle).
                        try:
                            proj_now = str(params_menu.get("proj_mode", proj_mode) or "pca")
                            in_flight_dbg = bool(menu is not None and menu.flight_active(proj_now))
                            set_name = _resolve_effective_control_set(menu)
                            b_dbg = _get_set_binding(cfg, set_name=set_name, group="hud", key="toggle")
                            dbg_active, _dbg_analog = _binding_active(b_dbg, axes_now, buttons_now, hats_now)
                            if in_flight_dbg:
                                if bool(dbg_active) and (not bool(debug_hud_toggle_prev)):
                                    flight_debug_hud_visible = not bool(flight_debug_hud_visible)
                            else:
                                flight_debug_hud_visible = True
                            debug_hud_toggle_prev = bool(dbg_active)
                        except Exception:
                            pass

                        ship_snap = None
                        focus_now = targeting_system.ReticleFocus(on_target=False, victim_id=0)
                        view_dir_now = None
                        try:
                            mode_ship = str(params.get("proj_mode", "pca") or "pca")
                            need_snap = (f1_active and (not fire1_prev)) or (f2_active and (not fire2_prev))
                            if need_snap and menu is not None and menu.flight_active(mode_ship):
                                sp = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                sr_b, su_b, sf_b = menu.flight_cam.basis()
                                sf = np.asarray(sf_b, dtype=np.float32)
                                sv = np.asarray(getattr(menu.flight_cam, "vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)

                                hm_info = None
                                hm_ref = None
                                try:
                                    fc = menu.flight_cam
                                    hm = getattr(fc, "terrain_heightmap", None)
                                    hs = float(getattr(fc, "terrain_height_scale", 0.0) or 0.0)
                                    hb = float(getattr(fc, "terrain_height_bias", 0.5) if getattr(fc, "terrain_height_bias", None) is not None else 0.5)
                                    if hm is not None and hs != 0.0:
                                        hm_np = np.ascontiguousarray(np.asarray(hm, dtype=np.float32))
                                        h_h = int(hm_np.shape[0])
                                        h_w = int(hm_np.shape[1])
                                        if h_h > 1 and h_w > 1:
                                            hm_ref = hm_np
                                            ptr = int(hm_np.__array_interface__["data"][0])
                                            hm_info = {
                                                "ptr": ptr,
                                                "w": h_w,
                                                "h": h_h,
                                                "stride": h_w,
                                                "height_scale": float(hs),
                                                "height_bias": float(hb),
                                            }
                                except Exception:
                                    hm_info = None
                                    hm_ref = None

                                nodes_info = None
                                nodes_ref = None
                                try:
                                    idx = int(state_idx)
                                    pos_src = None
                                    if 'pos3d_buf' in locals() and pos3d_buf and len(pos3d_buf) > idx and pos3d_buf[idx] is not None:
                                        pos_src = pos3d_buf[idx]
                                    elif 'state_pos' in locals() and state_pos and len(state_pos) > idx:
                                        pos_src = state_pos[idx]

                                    if pos_src is not None:
                                        if hasattr(pos_src, "detach"):
                                            pos_nodes = pos_src.detach().cpu().numpy()
                                        else:
                                            pos_nodes = np.asarray(pos_src)
                                        pos_nodes = np.ascontiguousarray(np.asarray(pos_nodes[:, :3], dtype=np.float32))

                                        masses_src = None
                                        if 'state_masses' in locals() and state_masses and len(state_masses) > idx:
                                            masses_src = state_masses[idx]
                                        if masses_src is not None:
                                            masses_nodes = np.ascontiguousarray(np.asarray(masses_src, dtype=np.float32))
                                        else:
                                            masses_nodes = np.ones((pos_nodes.shape[0],), dtype=np.float32)

                                        n_nodes = int(min(int(pos_nodes.shape[0]), int(masses_nodes.shape[0])))
                                        if n_nodes > 0:
                                            pos_nodes = pos_nodes[:n_nodes]
                                            masses_nodes = masses_nodes[:n_nodes]
                                            radii_nodes = radius_scale * np.cbrt(np.maximum(masses_nodes, 1e-6)) / mass_ref_cuberoot
                                            radii_nodes = np.ascontiguousarray(np.asarray(radii_nodes, dtype=np.float32))
                                            nodes_ref = (pos_nodes, radii_nodes)
                                            nodes_info = {
                                                "pos_ptr": int(pos_nodes.__array_interface__["data"][0]),
                                                "rad_ptr": int(radii_nodes.__array_interface__["data"][0]),
                                                "count": int(n_nodes),
                                                "pos_stride": 3,
                                                "rad_stride": 1,
                                            }

                                            # Reticle focus ray (view-only) for aiming.
                                            try:
                                                eye_c, center_c = menu.flight_camera(dt=0.0, proj_mode=mode_ship)
                                                eye_v = np.asarray(eye_c, dtype=np.float32)
                                                center_v = np.asarray(center_c, dtype=np.float32)
                                                view_dir_now = center_v - eye_v
                                                vn = float(np.linalg.norm(view_dir_now))
                                                if vn > 1e-6:
                                                    view_dir_now = (view_dir_now / vn).astype(np.float32, copy=False)
                                                else:
                                                    view_dir_now = np.asarray(sf, dtype=np.float32)
                                            except Exception:
                                                view_dir_now = np.asarray(sf, dtype=np.float32)

                                            def _df(o_df: np.ndarray, d_df: np.ndarray, tmax_df: float):
                                                hm_np = getattr(menu.flight_cam, "terrain_heightmap", None) if menu is not None else None
                                                hm_scale = getattr(menu.flight_cam, "terrain_height_scale", None) if menu is not None else None
                                                hm_bias = getattr(menu.flight_cam, "terrain_height_bias", None) if menu is not None else None
                                                hit = _RETICLE_DEPTH_FINDER.probe_laser_impact(
                                                    weap_rt=weap_rt,
                                                    ray_origin=o_df,
                                                    ray_dir=d_df,
                                                    max_dist=float(tmax_df),
                                                    planet_surface_r=(getattr(menu.flight_cam, "planet_surface_r", None) if menu is not None else None),
                                                    gravity_g=(getattr(menu.flight_cam, "gravity_g", None) if menu is not None else None),
                                                    terrain_heightmap=hm_np,
                                                    terrain_height_scale=hm_scale,
                                                    terrain_height_bias=hm_bias,
                                                    nodes_pos=np.asarray(pos_nodes, dtype=np.float32),
                                                    nodes_radius=np.asarray(radii_nodes, dtype=np.float32),
                                                )
                                                return hit.point if hit.valid else None

                                            st_now = target_sys.solve_reticle(
                                                ray_origin=np.asarray(eye_v, dtype=np.float32),
                                                ray_dir=np.asarray(view_dir_now, dtype=np.float32),
                                                t_max=float(z_far),
                                                nodes_pos=np.asarray(pos_nodes, dtype=np.float32),
                                                nodes_radius=np.asarray(radii_nodes, dtype=np.float32),
                                                depth_finder=_df,
                                            )
                                            target_sys.set_reticle(reticle_id=reticle_id, state=st_now)
                                            focus_now = st_now.focus
                                except Exception:
                                    nodes_info = None
                                    nodes_ref = None

                                ship_snap = weapon_runtime.ShipSnapshot(
                                    pos=(float(sp[0]), float(sp[1]), float(sp[2])),
                                    vel=(float(sv[0]), float(sv[1]), float(sv[2])),
                                    fwd=(float(sf[0]), float(sf[1]), float(sf[2])),
                                    planet_surface_r=float(getattr(menu.flight_cam, "planet_surface_r", 0.0) or 0.0),
                                    gravity_g=float(getattr(menu.flight_cam, "gravity_g", 0.0) or 0.0),
                                    terrain_heightmap=hm_info,
                                    terrain_heightmap_ref=hm_ref,
                                    nodes=nodes_info,
                                    nodes_ref=nodes_ref,
                                )
                        except Exception:
                            ship_snap = None

                        if f1_active and (not fire1_prev):
                            resolved = weapons_structs.resolve_virtual_weapon(loadout, 1)
                            for source_name, weapon_type in resolved:
                                wo = None
                                wd = None
                                try:
                                    if menu is not None and menu.flight_active(str(params.get("proj_mode", "pca") or "pca")) and str(source_name) == "nose gun" and ship_snap is not None:
                                        sp0 = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                        sr0, su0, sf0 = menu.flight_cam.basis()
                                        wo_t = targeting_system.compute_nose_gun_origin_world(
                                            ship_pos=sp0,
                                            ship_right=np.asarray(sr0, dtype=np.float32),
                                            ship_up=np.asarray(su0, dtype=np.float32),
                                            ship_fwd=np.asarray(sf0, dtype=np.float32),
                                        )
                                        wo = wo_t
                                        wd = targeting_system.compute_weapon_dir_from_reticle(
                                            weapon_origin=np.asarray(wo_t, dtype=np.float32),
                                            fallback_view_dir=np.asarray(view_dir_now if view_dir_now is not None else sf, dtype=np.float32),
                                            reticle=target_sys.get_reticle(reticle_id),
                                        )
                                except Exception:
                                    wo = None
                                    wd = None
                                weap_rt.enqueue_fire(
                                    loadout=loadout,
                                    weapon_slot=1,
                                    resolved=[(source_name, weapon_type)],
                                    ship=ship_snap,
                                    analog=float(f1_analog),
                                    weapon_origin=wo,
                                    weapon_dir=wd,
                                )
                        if f2_active and (not fire2_prev):
                            resolved = weapons_structs.resolve_virtual_weapon(loadout, 2)
                            for source_name, weapon_type in resolved:
                                wo = None
                                wd = None
                                try:
                                    if menu is not None and menu.flight_active(str(params.get("proj_mode", "pca") or "pca")) and str(source_name) == "nose gun" and ship_snap is not None:
                                        sp0 = np.asarray(menu.flight_cam.pos, dtype=np.float32)
                                        sr0, su0, sf0 = menu.flight_cam.basis()
                                        wo_t = targeting_system.compute_nose_gun_origin_world(
                                            ship_pos=sp0,
                                            ship_right=np.asarray(sr0, dtype=np.float32),
                                            ship_up=np.asarray(su0, dtype=np.float32),
                                            ship_fwd=np.asarray(sf0, dtype=np.float32),
                                        )
                                        wo = wo_t
                                        wd = targeting_system.compute_weapon_dir_from_reticle(
                                            weapon_origin=np.asarray(wo_t, dtype=np.float32),
                                            fallback_view_dir=np.asarray(view_dir_now if view_dir_now is not None else sf, dtype=np.float32),
                                            reticle=target_sys.get_reticle(reticle_id),
                                        )
                                except Exception:
                                    wo = None
                                    wd = None
                                weap_rt.enqueue_fire(
                                    loadout=loadout,
                                    weapon_slot=2,
                                    resolved=[(source_name, weapon_type)],
                                    ship=ship_snap,
                                    analog=float(f2_analog),
                                    weapon_origin=wo,
                                    weapon_dir=wd,
                                )

                        fire1_prev = bool(f1_active)
                        fire2_prev = bool(f2_active)

                        evts = weap_rt.poll_consequences(max_events=32)
                        now_s = pygame.time.get_ticks() * 0.001
                        try:
                            weapon_splines[:] = [s for s in weapon_splines if float(s.get("t_end", 0.0)) >= now_s]
                        except Exception:
                            pass
                        for evt in evts:
                            try:
                                if evt.kind != "info" or not isinstance(evt.payload, dict):
                                    continue
                                if evt.payload.get("event") != "dll_processed":
                                    continue
                                # Hard requirement: only draw paths returned by the C bullet simulator.
                                if not bool(evt.payload.get("from_c", False)):
                                    continue
                                if int(evt.payload.get("ok", 0) or 0) == 0:
                                    continue
                                pts = evt.payload.get("spline_points")
                                if isinstance(pts, list) and len(pts) >= 2:
                                    weapon_splines.append({"t_end": float(now_s + 1.25), "pts": pts})
                            except Exception:
                                pass
                    except Exception:
                        pass

            t_now = pygame.time.get_ticks() / 1000.0
            mode_now = str(params.get("proj_mode", "pca") or "pca")
            if _is_ship_proj_mode(mode_now) and menu is not None:
                dt_cam = 1.0 / 60.0
                try:
                    dt_cam = max(1e-4, float(clock.get_time()) * 0.001)
                except Exception:
                    pass
                eye, center = menu.flight_camera(dt=dt_cam, proj_mode=mode_now)
            elif _is_map_proj_mode(mode_now):
                eye = (0.0, 0.0, 3.0)
                center = (0.0, 0.0, 0.0)
            else:
                eye = _orbit_eye((0, 0, 0), 3.0, t_now) if orbit_enabled else (0.0, 0.0, 3.0)
                center = (0.0, 0.0, 0.0)
            current_gen = state_gen
            current_idx = state_idx
            pos_live = state_pos[current_idx]
            vel_live = state_vel[current_idx] if len(state_vel) > current_idx else None
            charges_live = state_charges[current_idx] if len(state_charges) > current_idx else None
            live_dim = pos_live.shape[1] if pos_live.ndim == 2 else 0
            ks_live = params["k_spring"]
            kc_live = params["k_coulomb"]
            gg_live = params["G"]
            south_live = params.get("south_strength", 0.0)
            south_on = params.get("south_enabled", False)
            south_axis_live = params.get("south_axis", current_n_dim - 1)
            dp_live = params["damping"]
            tp_live = params["temp"]
            # ensure projection basis matches current dimensionality
            if mean.shape[1] != pos_live.shape[1] or proj.shape[0] != pos_live.shape[1]:
                if params.get("freeze_pca", False):
                    mean, proj = _adapt_projection_dim(mean, proj, pos_live.shape[1])
                else:
                    mean, proj = _get_projection_np(pos_live)
                proj_mean_cpu = _to_torch_pinned_float(mean)
                proj_cpu = _to_torch_pinned_float(proj)
                backdrop_dirty = True

            try:
                _ensure_projection(state_pos_buf[current_idx], pos3d_buf[current_idx])
            except Exception:
                # rebuild projection defensively if a shape mismatch slipped through
                if params.get("freeze_pca", False):
                    mean, proj = _adapt_projection_dim(mean, proj, state_pos_buf[current_idx].shape[1])
                else:
                    mean, proj = _get_projection_np(state_pos_buf[current_idx].cpu().numpy())
                proj_mean_cpu = _to_torch_pinned_float(mean)
                proj_cpu = _to_torch_pinned_float(proj)
                pos3d_buf[current_idx] = _ensure_projection(state_pos_buf[current_idx], pos3d_buf[current_idx])
                backdrop_dirty = True
            pos3d_t = pos3d_buf[current_idx]
            pos3d = pos3d_t.numpy()

            # if render state dim drifts from target, queue a rebuild
            if pos_live.shape[1] != current_n_dim and pending_dim_change is None:
                pending_dim_change = current_n_dim

            proj_rows = int(proj_cpu.shape[0])
            proj_cols = int(proj_cpu.shape[1])
            backdrop_dim = min(live_dim, proj_rows)
            proj_fp = (proj_rows, proj_cols, backdrop_dim, float(torch.sum(torch.abs(proj_cpu)).item()))
            if backdrop_dirty or backdrop_cache["fp"] != proj_fp:
                proj_np = proj_cpu.cpu().numpy()
                lines, line_cols, axes, axis_cols = _build_backdrop_lines(backdrop_dim, proj_np)
                backdrop_cache = {"fp": proj_fp, "dim": backdrop_dim, "lines": lines, "line_colors": line_cols, "axes": axes, "axis_colors": axis_cols}
                backdrop_dirty = False
            diag_view = diag
            overlay = [
                f"phys fps: {phys_fps:.1f}",
                f"dt: {diag_view.get('dt', adaptive_dt):.3e}",
                f"max speed: {params['max_speed']:.3e}",
                f"state gen: {state_gen}",
                f"n_dim (live): {live_dim} / target: {target_n_dim}",
                f"N (live): {pos_live.shape[0]}",
                f"node sel: {node_selector_idx % max(1, pos_live.shape[0])} / {max(0, pos_live.shape[0]-1)}",
                f"fly mode: {'on' if fly_mode else 'off'}",
            ]
            last_render_gen = current_gen

            # append ghost trail snapshot (3D projected positions)
            if ghost_history.maxlen and pos3d is not None:
                ghost_history.append(pos3d_t.clone())
            sel_idx = (node_selector_idx % pos_live.shape[0]) if pos_live.shape[0] else None
            sel_list = [sel_idx] if sel_idx is not None else None

            # derive colors and sizes from live charges/masses
            colors_live = colors
            point_sizes = None
            if charges_live is not None:
                colors_live = _colors_from_charges(charges_live)
            if 'state_masses' in locals() and state_masses and len(state_masses) > current_idx:
                masses_live = state_masses[current_idx]
                radii_live = radius_scale * np.cbrt(np.maximum(masses_live, 1e-6)) / mass_ref_cuberoot
                # drop the upper cap so very massive nodes render proportionally
                point_sizes = np.maximum(3.0, 150.0 * radii_live)

            springs_live = state_springs[current_idx] if state_springs and len(state_springs) > current_idx else springs_empty
            # straight edges (base draw) are optional; geodesic arcs are drawn below
            springs_draw = springs_live if params.get("draw_direct_edges", False) else springs_empty

            base.draw_scene(
                pos_live,
                pos3d,
                colors_live,
                springs_draw,
                center=np.asarray(center, dtype=np.float32),
                eye=eye,
                font=font,
                fps_render=clock.get_fps(),
                fps_phys=phys_fps,
                text_lines=overlay,
                node_labels=None,
                proj_mats=None,
                highlight_idx=sel_list,
                edge_colors=None,
                thrust_lines=None,
                sentence_triangles=None,
                point_sizes=point_sizes,
                pre_draw_fn=(_draw_planet_and_atmosphere if _is_ship_proj_mode(mode_now) else None),
                draw_hud_now=False,
            )

            # Weapon debug splines from bullet-sim DLL (world-space).
            try:
                now_s = pygame.time.get_ticks() * 0.001
                weapon_splines[:] = [s for s in weapon_splines if float(s.get("t_end", 0.0)) >= now_s]
                _draw_weapon_splines_world(splines=[s.get("pts", []) for s in weapon_splines])
            except Exception:
                pass

            if params.get("draw_velocity_paths", False):
                _draw_tangent_paths(pos_live, vel_live, proj_mean_cpu.numpy(), proj_cpu.numpy(), samples=144, alpha=0.12, indices=sel_list)

            if params.get("draw_edges", True) and (not _is_map_proj_mode(mode_now)) and (not _is_ship_proj_mode(mode_now)):
                _draw_geodesic_edges(pos_live, state_springs[current_idx], proj_mean_cpu.numpy(), proj_cpu.numpy(), samples=24, alpha=0.26)

            if params.get("draw_backdrop", False):
                _draw_backdrop(
                    backdrop_cache["lines"],
                    backdrop_cache["line_colors"],
                    backdrop_cache["axes"],
                    backdrop_cache["axis_colors"],
                    alpha_grid=0.14,
                    alpha_axis=0.32,
                )

            # draw fading rainbow ghost points with depth darkening (VBO, single draw)
            if ghost_history:
                if not hasattr(run, "_ghost_vbos"):
                    run._ghost_vbos = {
                        "verts": glGenBuffers(1),
                        "colors": glGenBuffers(1),
                    }

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPointSize(3.0)

                # Align history snapshots to current live node count by truncating/padding
                hist_list = list(ghost_history)
                if hist_list:
                    current_n = pos3d_t.shape[0]
                    aligned = []
                    for snap in hist_list:
                        if snap.shape[0] == current_n:
                            aligned.append(snap)
                        elif snap.shape[0] > current_n:
                            aligned.append(snap[:current_n])
                        else:
                            pad = torch.zeros((current_n - snap.shape[0], snap.shape[1]), dtype=snap.dtype)
                            aligned.append(torch.cat([snap, pad], dim=0))
                    hist_t = torch.stack(aligned)
                else:
                    hist_t = torch.empty((0, pos3d_t.shape[0], 3), dtype=pos3d_t.dtype)
                H, N, _ = hist_t.shape if hist_t.numel() else (0, pos3d_t.shape[0], 3)

                if H > 0:
                    t_hue = (pygame.time.get_ticks() * 0.001) * ghost_hue_cycles
                    hue = (t_hue + torch.arange(H, dtype=torch.float32) / max(1, H)) % 1.0
                    ages = torch.linspace(0.0, 1.0, steps=max(2, H), dtype=torch.float32)[:H]
                    alpha = torch.clamp(1.0 - ages, min=0.05, max=1.0)
                else:
                    hue = torch.empty((0,), dtype=torch.float32)
                    alpha = torch.empty((0,), dtype=torch.float32)

                z_vals = pos3d_t[:, 2] if pos3d_t.numel() else torch.zeros((1,), dtype=torch.float32)
                z_min = float(z_vals.min().item()) if len(z_vals) else 0.0
                z_max = float(z_vals.max().item()) if len(z_vals) else 1.0
                z_span = max(1e-6, z_max - z_min)
                if H > 0:
                    z_norm = (hist_t[:, :, 2] - z_min) / z_span
                    shade = 0.35 + 0.65 * (1.0 - z_norm)
                else:
                    shade = torch.zeros((0, 1), dtype=torch.float32)

                if H > 0:
                    hue_full = hue[:, None].expand(H, N)
                    rgb = torch.as_tensor(_hsl_to_rgb_np(hue_full.numpy(), 0.9, 0.55))
                    rgb = rgb * shade.unsqueeze(-1)
                    alpha_full = alpha[:, None].expand(H, N).unsqueeze(-1)
                    rgba = torch.cat([rgb, alpha_full], dim=-1).reshape(-1, 4)

                    verts = hist_t.reshape(-1, 3).contiguous()
                    rgba = rgba.contiguous()

                    v_np = verts.numpy().astype(np.float32, copy=False)
                    c_np = rgba.numpy().astype(np.float32, copy=False)

                    glBindBuffer(GL_ARRAY_BUFFER, run._ghost_vbos["verts"])
                    glBufferData(GL_ARRAY_BUFFER, v_np.nbytes, v_np, GL_DYNAMIC_DRAW)
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glVertexPointer(3, GL_FLOAT, 0, None)

                    glBindBuffer(GL_ARRAY_BUFFER, run._ghost_vbos["colors"])
                    glBufferData(GL_ARRAY_BUFFER, c_np.nbytes, c_np, GL_DYNAMIC_DRAW)
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointer(4, GL_FLOAT, 0, None)

                    glDrawArrays(GL_POINTS, 0, verts.shape[0])

                    glDisableClientState(GL_COLOR_ARRAY)
                    glDisableClientState(GL_VERTEX_ARRAY)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)

                glBindBuffer(GL_ARRAY_BUFFER, run._ghost_vbos["verts"])
                glBufferData(GL_ARRAY_BUFFER, v_np.nbytes, v_np, GL_DYNAMIC_DRAW)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, None)

                glBindBuffer(GL_ARRAY_BUFFER, run._ghost_vbos["colors"])
                glBufferData(GL_ARRAY_BUFFER, c_np.nbytes, c_np, GL_DYNAMIC_DRAW)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(4, GL_FLOAT, 0, None)

                glDrawArrays(GL_POINTS, 0, verts.shape[0])

                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

            # Draw HUD/menu last so it always renders on top of any later overlays.
            try:
                try:
                    _surf = pygame.display.get_surface()
                    if _surf is not None:
                        _w, _h = _surf.get_size()
                    else:
                        _w, _h = 1280, 720
                    if cfg_pip is not None:
                        _draw_loadout_menu_overlay(font=pip_font, width=int(_w), height=int(_h), pip=cfg_pip)
                except Exception:
                    pass
                base.draw_hud(font=font, fps_render=clock.get_fps(), fps_phys=phys_fps, text_lines=overlay)
            except Exception:
                pass
            pygame.display.flip()
            clock.tick(60)
    finally:
        try:
            if menu is not None and hasattr(menu, "flight_cam"):
                menu.flight_cam.shutdown()
        except Exception:
            pass
        stop_evt.set()
        phys_thread.join(timeout=1.0)
        pygame.quit()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Geodesic spring-mass-charge animator (unit sphere)")
    p.add_argument("path", help="Path to prephysics JSON", nargs="?")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--k-spring", type=float, default=1.0)
    p.add_argument("--k-charge", type=float, default=0.2, help="Coulomb-like coefficient")
    p.add_argument("--G", type=float, default=0.05, help="Gravity-like coefficient")
    p.add_argument("--south-strength", type=float, default=0.0, help="Uniform southward field magnitude (tangent pull toward chosen pole)")
    p.add_argument("--south-axis", type=int, default=-1, help="Axis index treated as 'south' (default: last dimension)")
    p.add_argument("--south-enabled", action="store_true", default=None, help="Force the uniform south field on at start")
    p.add_argument("--no-south", action="store_false", dest="south_enabled", help="Force the uniform south field off at start")
    p.add_argument("--lorentz-k", type=float, default=0.0, help="Lorentz term strength (uses first 3 dims)")
    p.add_argument("--B", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="Magnetic field vector for Lorentz term")
    p.add_argument("--B-matrix", type=float, nargs="*", default=None, help="Optional flattened upper-triangular skew matrix for n-D magnetic field")
    p.add_argument("--B-skew-random", action="store_true", help="Generate a random n-D skew-symmetric magnetic field")
    p.add_argument("--B-skew-auto", action="store_true", default=True, help="Auto-generate an n-D skew field when dim>3 and none provided")
    p.add_argument("--no-B-skew-auto", action="store_false", dest="B_skew_auto", help="Disable auto n-D skew generation")
    p.add_argument("--B-scale", type=float, default=1.0, help="Scale for provided or random skew magnetic field")
    p.add_argument("--B-seed", type=int, default=None, help="Seed for random skew magnetic field")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--n-dim", type=int, default=4, help="Simulation dimensionality")
    p.add_argument("--n-points", type=int, default=128, help="Number of particles when no path provided")
    p.add_argument("--max-neighbors", type=int, default=None, help="Sparse neighbors per particle (default: N-1)")
    p.add_argument("--neighbor-refresh", type=int, default=4, help="Steps between neighbor reselection")
    p.add_argument("--bond-max-per-node", type=int, default=3, help="Max active bonds per node")
    p.add_argument("--bond-link-angle", type=float, default=0.22, help="Geodesic angle threshold to form bond (radians)")
    p.add_argument("--bond-shear-ratio", type=float, default=1.6, help="Break bond if angle exceeds rest * ratio")
    p.add_argument("--bond-k", type=float, default=0.8, help="Spring constant for new bonds")
    p.add_argument("--ionic-bonds", action="store_true", help="Enable ionic-style bonding (opposite charges with limited valence)")
    p.add_argument("--ionic-valence", type=int, default=2, help="Max bonds per node when ionic bonding is enabled")
    p.add_argument("--accel-shatter-thresh", type=float, default=0.0, help="Acceleration magnitude threshold to atomize nodes (0 to disable)")
    p.add_argument("--accel-shatter-fraction", type=float, default=0.35, help="Fraction of mass/charge shed when acceleration shatter triggers")
    p.add_argument("--accel-shatter-k", type=int, default=6, help="Number of shards produced in an acceleration shatter event")
    p.add_argument("--accel-shatter-kick", type=float, default=0.02, help="Velocity scale per acceleration magnitude applied to shards")
    p.add_argument("--ghost-history", type=int, default=96, help="Number of past frames to render as ghost points")
    p.add_argument("--ghost-hue-cycles", type=float, default=1.5, help="Hue wheel cycles per second for ghost points")
    p.add_argument("--dense-pair-threshold", type=int, default=512, help="Use dense all-pairs force when N <= threshold (0 disables)")
    p.add_argument(
        "--min-dt",
        type=float,
        default=None,
        help="Optional minimum timestep floor (default: float64 tiny; C backend enforces real stability limits)",
    )
    p.add_argument("--max-speed", type=float, default=.02, help="Adaptive step velocity limit")
    p.add_argument("--max-recursion", type=int, default=3, help="Adaptive step recursion depth")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to run physics on")
    p.add_argument("--radius-scale", type=float, default=0.08, help="Base radius scale for mass-based sizing")
    p.add_argument("--diff-mass", type=float, default=0.05, help="Mass diffusion rate along bonds per second")
    p.add_argument("--diff-charge", type=float, default=0.05, help="Charge diffusion rate along bonds per second")
    p.add_argument("--collide-gain", type=float, default=1.1, help="Collision distance multiplier vs summed radii")
    p.add_argument("--merge-speed-frac", type=float, default=0.4, help="<= frac*max_speed triggers merge")
    p.add_argument("--shatter-speed-frac", type=float, default=0.9, help=">= frac*max_speed triggers shatter when sizes differ")
    p.add_argument("--merge-size-frac", type=float, default=0.5, help="Minimum size ratio to favor merge when speeds moderate")
    p.add_argument("--shatter-size-frac", type=float, default=0.25, help="Max size ratio (small/large) to allow shatter")
    p.add_argument("--shatter-k-max", type=int, default=6, help="Max shards when shattering a large node")
    p.add_argument("--collision-pair-limit", type=int, default=256, help="Use all-pairs collision checks only when N <= limit; otherwise bonded pairs")
    p.add_argument("--inflate-mult", type=float, default=1.2, help="Multiplier applied to active node mass when button 3 is pressed")
    p.add_argument("--mass-vapor-thresh", type=float, default=0.015, help="Mass threshold below which nodes vaporize into the ionic cache")
    p.add_argument("--condense-temp-thresh", type=float, default=0.5, help="Max |temp| to allow condensation from vapor cache")
    p.add_argument("--condense-chunk-mass", type=float, default=0.15, help="Mass chunk condensed per event from vapor cache")
    p.add_argument("--temp-cool-rate", type=float, default=0.35, help="Rate temps relax toward ambient per second")
    p.add_argument("--temp-heat-rate", type=float, default=0.1, help="Heating rate per unit speed per second")
    p.add_argument("--dt-mode", type=str, choices=["single-posthoc", "two-level", "multi"], default="single-posthoc", help="Adaptive step mode: single-posthoc (one dt, adjust after), two-level (base/half), multi (current multi-candidate)")
    p.add_argument("--c-substep-dt", type=float, default=None, help="(C-physics only) Max dt per internal substep; dt_try is split into ceil(dt_try/this) steps (default: --dt)")
    p.add_argument("--c-substeps-max", type=int, default=64, help="(C-physics only) Cap on internal substeps per outer tick")
    p.add_argument("--c-n-cap", type=int, default=None, help="(C-physics only) Allocate a larger node capacity for future dynamic node count (NOTE: memory scales ~O(n^2))")
    p.add_argument("--draw-direct-edges", action="store_true", default=False, help="Render straight edges between nodes (off by default; geodesic arcs still draw)")
    p.add_argument("--freeze-pca", action="store_true", default=True, help="Keep PCA projection basis fixed across dimension changes")
    p.add_argument("--no-freeze-pca", action="store_false", dest="freeze_pca", help="Recompute PCA basis on dimension changes")
    p.add_argument("--draw-edges", action="store_true", default=True, help="Render geodesic edge arcs")
    p.add_argument("--no-draw-edges", action="store_false", dest="draw_edges", help="Disable edge rendering")
    p.add_argument("--max-edges", type=int, default=0, help="(C-physics only) Cap total springs/bonds stored (0 for unlimited)")
    p.add_argument("--phys-fps", type=float, default=60.0, help="Cap physics loop rate in Hz (0 to uncap)")
    p.add_argument("--backdrop", action="store_true", default=False, help="Enable backdrop grid/axes rendering (off by default)")
    p.add_argument("--velocity-rings", action="store_true", default=False, help="Show velocity tangent rings (off by default)")
    p.add_argument("--watchdog", action="store_true", default=False, help="Enable physics stagnation watchdog (disabled by default)")
    p.add_argument("--c-physics", action="store_true", default=False, help="Use the C DLL physics loop and upload positions to OpenGL directly (requires --n-dim 3)")
    p.add_argument("--fov", type=float, default=10.0, help="Vertical field of view in degrees (narrow FOV makes the world feel larger/farther)")
    p.add_argument("--render-scale", type=float, default=10.0, help="Render-only scale factor for ship/planet (physics stays unit sphere)")
    p.add_argument("--airplane", type=str, default="airplane.json", help="Path to airplane spec JSON (controls + thrust/lift/actuation centers)")
    p.add_argument("--scene", type=str, default="scene.json", help="Path to scene spec JSON (sun + sky + haze)")
    p.add_argument("--z-near", type=float, default=0.1, help="Near clip plane")
    p.add_argument("--z-far", type=float, default=1000.0, help="Far clip plane (increase if the planet/points clip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        run(
            path=args.path,
            dt=args.dt,
            k_spring=args.k_spring,
            k_coulomb=args.k_charge,
            G=args.G,
            south_strength=args.south_strength,
            south_axis=args.south_axis,
            south_enabled=args.south_enabled,
            lorentz_k=args.lorentz_k,
            B_vec=tuple(args.B),
            B_matrix_flat=args.B_matrix,
            B_skew_random=args.B_skew_random,
            B_skew_auto=args.B_skew_auto,
            B_scale=args.B_scale,
            B_seed=args.B_seed,
            temp=args.temp,
            damping=args.damping,
            draw_backdrop=bool(args.backdrop),
            draw_velocity_paths=bool(args.velocity_rings),
            softening=args.softening,
            n_dim=args.n_dim,
            n_points=args.n_points,
            max_neighbors=args.max_neighbors,
            neighbor_refresh=args.neighbor_refresh,
            bond_max_per_node=args.bond_max_per_node,
            bond_link_angle=args.bond_link_angle,
            bond_shear_ratio=args.bond_shear_ratio,
            bond_k=args.bond_k,
            ionic_bonds=args.ionic_bonds,
            ionic_valence=args.ionic_valence,
            ghost_history=args.ghost_history,
            ghost_hue_cycles=args.ghost_hue_cycles,
            dense_pair_threshold=args.dense_pair_threshold,
            device=args.device,
            render_scale=args.render_scale,
            airplane_path=args.airplane,
            scene_path=args.scene,
            fov_deg=args.fov,
            z_near=args.z_near,
            z_far=args.z_far,
            min_dt_override=args.min_dt,
            max_speed=args.max_speed,
            max_recursion=args.max_recursion,
            radius_scale=args.radius_scale,
            diff_mass_rate=args.diff_mass,
            diff_charge_rate=args.diff_charge,
            collide_gain=args.collide_gain,
            merge_speed_frac=args.merge_speed_frac,
            shatter_speed_frac=args.shatter_speed_frac,
            merge_size_frac=args.merge_size_frac,
            shatter_size_frac=args.shatter_size_frac,
            shatter_k_max=args.shatter_k_max,
            collision_pair_limit=args.collision_pair_limit,
            accel_shatter_thresh=args.accel_shatter_thresh,
            accel_shatter_fraction=args.accel_shatter_fraction,
            accel_shatter_k=args.accel_shatter_k,
            accel_shatter_kick=args.accel_shatter_kick,
            inflate_mult=args.inflate_mult,
            mass_vapor_thresh=args.mass_vapor_thresh,
            condense_temp_thresh=args.condense_temp_thresh,
            condense_chunk_mass=args.condense_chunk_mass,
            temp_cool_rate=args.temp_cool_rate,
            temp_heat_rate=args.temp_heat_rate,
            dt_mode=args.dt_mode,
            c_substep_dt_max=args.c_substep_dt,
            c_substeps_max=args.c_substeps_max,
            c_n_cap=args.c_n_cap,
            draw_direct_edges=args.draw_direct_edges,
            freeze_pca=args.freeze_pca,
            draw_edges=args.draw_edges,
            c_max_edges=(None if int(args.max_edges) <= 0 else int(args.max_edges)),
            phys_fps_limit=args.phys_fps,
            enable_watchdog=args.watchdog,
            use_c_physics=True,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as exc:
        # Treat common pygame shutdown errors as clean exit when the window closes.
        msg = str(exc).lower()
        if "video system not initialized" in msg or "display surface quit" in msg:
            raise SystemExit(0)
        raise
