from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame

import targeting_system
import weapon_runtime
import reticle_text_overlay


@dataclass(frozen=True)
class ReticleTelemetry:
    left_lines: list[str]
    right_lines: list[str]


def draw_reticle_telemetry(
    *,
    font,
    width: int,
    height: int,
    center_px: tuple[float, float],
    telemetry: ReticleTelemetry,
) -> None:
    """Draw telemetry near a reticle center in screen space.

    This is intentionally independent from the global HUD so it remains visible during flight.
    """
    try:
        from OpenGL.GL import (
            GL_DEPTH_TEST,
            GL_BLEND,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_SRC_ALPHA,
            GL_UNSIGNED_BYTE,
            GL_RGBA,
            GL_UNPACK_ALIGNMENT,
            glBlendFunc,
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
            glDepthMask,
            GL_MODELVIEW,
            GL_PROJECTION,
        )
    except Exception:
        return

    if telemetry is None:
        return
    left_lines = list(telemetry.left_lines or [])
    right_lines = list(telemetry.right_lines or [])
    if not left_lines and not right_lines:
        return

    w = int(max(1, int(width)))
    h = int(max(1, int(height)))
    cx = float(center_px[0])
    cy = float(center_px[1])

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    if depth_was_enabled:
        glDisable(GL_DEPTH_TEST)
    glDepthMask(False)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, float(w), 0.0, float(h), -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def _draw_lines(lines: list[str], *, x0: float, y0: float) -> None:
        yy = float(y0)
        for line in lines[:6]:
            surf = reticle_text_overlay.render_cell_text(font, str(line)[:180])
            data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(float(x0), float(yy))
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)
            yy -= float(max(12, surf.get_height() + 2))

    # Place text slightly above and to the sides of the reticle.
    y_top = float(cy) + 28.0
    x_left = float(cx) - 260.0
    x_right = float(cx) + 44.0
    x_left = float(max(6.0, min(float(w) - 6.0, x_left)))
    x_right = float(max(6.0, min(float(w) - 6.0, x_right)))
    y_top = float(max(6.0, min(float(h) - 6.0, y_top)))

    if left_lines:
        _draw_lines(left_lines, x0=x_left, y0=y_top)
    if right_lines:
        _draw_lines(right_lines, x0=x_right, y0=y_top)

    glDisable(GL_BLEND)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glDepthMask(True)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


def _safe_norm(v: np.ndarray) -> float:
    try:
        n = float(np.linalg.norm(v))
    except Exception:
        return 0.0
    if not np.isfinite(n):
        return 0.0
    return n


def _safe_normalize(v: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    n = _safe_norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32, copy=False)


def _angle_deg_between(u: np.ndarray, v: np.ndarray) -> float:
    uu = _safe_normalize(np.asarray(u, dtype=np.float32).reshape((3,)))
    vv = _safe_normalize(np.asarray(v, dtype=np.float32).reshape((3,)))
    if _safe_norm(uu) <= 1e-6 or _safe_norm(vv) <= 1e-6:
        return float("nan")
    d = float(np.clip(float(np.dot(uu, vv)), -1.0, 1.0))
    return float(math.acos(d) * (180.0 / math.pi))


def format_dms(angle_deg: float) -> str:
    """Format an angle in degrees as D°M′S″.

    This is always non-negative; callers should add a sign if needed.
    """
    if not np.isfinite(angle_deg):
        return "—"
    a = float(abs(angle_deg))
    d = int(math.floor(a))
    m_full = (a - float(d)) * 60.0
    m = int(math.floor(m_full))
    s = (m_full - float(m)) * 60.0
    # Keep seconds in [0,60) with rounding.
    s_ri = int(round(s))
    if s_ri >= 60:
        s_ri -= 60
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return f"{d}°{m:02d}′{s_ri:02d}″"


def _estimate_muzzle_speed(weapon_cfg: dict | None) -> Optional[float]:
    if not isinstance(weapon_cfg, dict):
        return None
    kin = weapon_cfg.get("kinematics") if isinstance(weapon_cfg.get("kinematics"), dict) else None
    if not isinstance(kin, dict):
        return None
    mode = str(kin.get("mode") or "")
    if mode == "beam":
        return None
    try:
        v = kin.get("add_velocity")
        if v is None:
            return None
        vf = float(v)
        if not np.isfinite(vf) or abs(vf) <= 1e-6:
            return None
        return float(abs(vf))
    except Exception:
        return None


def build_reticle_telemetry(
    *,
    weap_rt: weapon_runtime.WeaponRuntime,
    reticle: targeting_system.ReticleState,
    weapon_origin: np.ndarray,
) -> ReticleTelemetry:
    """Build reticle telemetry lines.

    Left side:
    - target angular separation from reticle ray (DMS)
    - target Euclidean range

    Right side (only if LOS confirmed):
    - impact angular separation from reticle ray (DMS)
    - impact Euclidean range
    - a simple TTI-like metric (range / muzzle speed) when speed is known
    """
    o = np.asarray(weapon_origin, dtype=np.float32).reshape((3,))
    vdir = _safe_normalize(np.asarray(reticle.view_dir, dtype=np.float32).reshape((3,)))

    left: list[str] = []
    right: list[str] = []

    # Prefer auto-tracked target, then the manual focus target.
    tgt = None
    tgt_id = 0
    try:
        tgt_id = int(getattr(reticle, "tracked_victim_id", 0) or 0)
    except Exception:
        tgt_id = 0

    if getattr(reticle, "tracked_pos", None) is not None:
        tgt = np.asarray(reticle.tracked_pos, dtype=np.float32).reshape((3,))
    elif reticle.focus.on_target and reticle.focus.target_pos is not None:
        tgt = np.asarray(reticle.focus.target_pos, dtype=np.float32).reshape((3,))
        tgt_id = int(reticle.focus.victim_id)

    if tgt is not None:
        to_tgt = (tgt - o).astype(np.float32, copy=False)
        rng = _safe_norm(to_tgt)
        ang = _angle_deg_between(vdir, to_tgt)
        left.append(f"TGT Δθ {format_dms(ang)}")
        if tgt_id:
            left.append(f"TGT id {int(tgt_id)}  R {rng:.3f}")
        else:
            left.append(f"TGT R {rng:.3f}")

    # Impact telemetry (only when LOS is confirmed).
    los_ok = bool(getattr(reticle, "los_confirmed", False))
    if los_ok and getattr(reticle, "aim_point", None) is not None:
        ip = np.asarray(reticle.aim_point, dtype=np.float32).reshape((3,))
        to_ip = (ip - o).astype(np.float32, copy=False)
        rng_i = _safe_norm(to_ip)
        ang_i = _angle_deg_between(vdir, to_ip)
        right.append(f"IMP Δθ {format_dms(ang_i)}")
        right.append(f"IMP R {rng_i:.3f}")

        # TTI-like estimate from weapon family.
        weapon_type = str(getattr(reticle, "weapon_type_primary", "") or "")
        cfg = weapon_runtime.get_weapon_type_stats(weap_rt.stats, weapon_type) if weapon_type else None
        v0 = _estimate_muzzle_speed(cfg)
        if v0 is not None and v0 > 1e-6:
            right.append(f"TTI ~ {rng_i / float(v0):.2f}s")

    return ReticleTelemetry(left_lines=left, right_lines=right)
