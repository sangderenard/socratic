from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import reticle_sprite


Vec3 = Tuple[float, float, float]


@dataclass
class ReticleFocus:
    on_target: bool
    victim_id: int  # 1-based, 0 means none
    target_pos: Optional[Vec3] = None
    t_hit: Optional[float] = None


@dataclass
class ReticleState:
    """All targeting outputs bound to one reticle.

    - `focus`: UI-ish target lock info (currently node-sphere raycast)
    - `view_dir`: the normalized ray direction in world space
    - `aim_point`: authoritative depth finder point (e.g. C beam intercept)
    """

    focus: ReticleFocus
    view_dir: Vec3
    aim_point: Optional[Vec3] = None

    # Optional electronic tracking (auto targeting). This is not derived from simulation hits;
    # it is an external "where the system thinks the target is" signal.
    tracked_victim_id: int = 0
    tracked_pos: Optional[Vec3] = None
    centered_on_track: bool = False

    # Line-of-sight confirmation (periodic). When true, the reticle can show "ready".
    los_confirmed: bool = False
    los_last_check_s: float = 0.0

    # Lightweight policy + latch metadata for future multi-weapon support.
    # `weapon_type_primary` describes which weapon family this reticle state is being evaluated for.
    # If `weapon_independent_aim` is true, LOS confirmation may be latched and remain valid as long
    # as the target stays within that weapon family's angles-of-fire envelope.
    weapon_type_primary: str = ""
    weapon_independent_aim: bool = False
    weapon_off_boresight_deg: float = 0.0
    los_confirmed_target_id: int = 0
    los_confirmed_weapon_type: str = ""


def _safe_normalize(v: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32, copy=False)


def raycast_nodes(
    *,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    nodes_pos: np.ndarray,
    nodes_radius: np.ndarray,
    t_max: float,
) -> ReticleFocus:
    """Return the closest node hit along the reticle ray.

    This mirrors the C-side ray-sphere logic (choose the nearest positive hit).
    Victim IDs are 1-based.
    """
    o = np.asarray(ray_origin, dtype=np.float32).reshape((3,))
    d = _safe_normalize(np.asarray(ray_dir, dtype=np.float32).reshape((3,)))
    if float(np.linalg.norm(d)) <= 1e-9:
        return ReticleFocus(on_target=False, victim_id=0)

    p = np.asarray(nodes_pos, dtype=np.float32)
    r = np.asarray(nodes_radius, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        return ReticleFocus(on_target=False, victim_id=0)
    if r.ndim != 1 or r.shape[0] != p.shape[0]:
        return ReticleFocus(on_target=False, victim_id=0)

    best_t: float = float(t_max) + 1.0
    best_i: int = -1

    a = float(np.dot(d, d))
    if not (a > 1e-12):
        return ReticleFocus(on_target=False, victim_id=0)

    for i in range(int(p.shape[0])):
        rad = float(r[i])
        if not (rad > 0.0):
            continue
        c = p[i]
        oc = o - c
        b = float(np.dot(oc, d))
        cc = float(np.dot(oc, oc) - rad * rad)
        disc = b * b - a * cc
        if disc < 0.0:
            continue
        s = float(np.sqrt(disc))
        inva = 1.0 / a
        t0 = (-b - s) * inva
        t1 = (-b + s) * inva
        t = t0 if t0 >= 0.0 else t1
        if t < 0.0 or t > float(t_max):
            continue
        if t < best_t:
            best_t = t
            best_i = i

    if best_i < 0:
        return ReticleFocus(on_target=False, victim_id=0)

    hit = o + best_t * d
    return ReticleFocus(
        on_target=True,
        victim_id=int(best_i + 1),
        target_pos=(float(p[best_i, 0]), float(p[best_i, 1]), float(p[best_i, 2])),
        t_hit=float(best_t),
    )


def compute_nose_gun_origin_world(
    *,
    ship_pos: np.ndarray,
    ship_right: np.ndarray,
    ship_up: np.ndarray,
    ship_fwd: np.ndarray,
    local_offset: Optional[Tuple[float, float, float]] = None,
) -> Vec3:
    """Compute a muzzle origin in world space.

    `local_offset` is interpreted in ship-local axes (right, up, forward).
    The default includes a small lateral/down offset so shots don't originate
    exactly on the view axis.
    """
    if local_offset is None:
        local_offset = (0.35, -0.15, 2.0)

    sp = np.asarray(ship_pos, dtype=np.float32).reshape((3,))
    sr = _safe_normalize(np.asarray(ship_right, dtype=np.float32).reshape((3,)))
    su = _safe_normalize(np.asarray(ship_up, dtype=np.float32).reshape((3,)))
    sf = _safe_normalize(np.asarray(ship_fwd, dtype=np.float32).reshape((3,)))

    ox, oy, oz = float(local_offset[0]), float(local_offset[1]), float(local_offset[2])
    o = sp + ox * sr + oy * su + oz * sf
    return (float(o[0]), float(o[1]), float(o[2]))


def compute_weapon_dir_world(
    *,
    weapon_origin: np.ndarray,
    view_dir: np.ndarray,
    focus: ReticleFocus,
) -> Vec3:
    """Aim at focused target when available; otherwise aim down view ray."""
    o = np.asarray(weapon_origin, dtype=np.float32).reshape((3,))
    if focus.on_target and focus.target_pos is not None:
        t = np.asarray(focus.target_pos, dtype=np.float32)
        d = _safe_normalize(t - o)
    else:
        d = _safe_normalize(np.asarray(view_dir, dtype=np.float32).reshape((3,)))
    return (float(d[0]), float(d[1]), float(d[2]))


def compute_weapon_dir_from_reticle(
    *,
    weapon_origin: np.ndarray,
    fallback_view_dir: np.ndarray,
    reticle: ReticleState,
) -> Vec3:
    """Compute a weapon direction based on a reticle's best available aim target.

    Precedence:
    1) `reticle.aim_point` (depth finder / intercept)
    2) `reticle.focus.target_pos` (node focus)
    3) `fallback_view_dir`
    """
    o = np.asarray(weapon_origin, dtype=np.float32).reshape((3,))
    if reticle.aim_point is not None:
        a = np.asarray(reticle.aim_point, dtype=np.float32).reshape((3,))
        d = _safe_normalize(a - o)
    elif reticle.focus.on_target and reticle.focus.target_pos is not None:
        a = np.asarray(reticle.focus.target_pos, dtype=np.float32).reshape((3,))
        d = _safe_normalize(a - o)
    else:
        d = _safe_normalize(np.asarray(fallback_view_dir, dtype=np.float32).reshape((3,)))
    return (float(d[0]), float(d[1]), float(d[2]))


class TargetingSystem:
    def __init__(self, *, folder: str = "assets/reticle", size_px: int = 64) -> None:
        self.folder = str(folder)
        self.size_px = int(size_px)
        self.assets: reticle_sprite.ReticleAssets | None = None
        self.anim = reticle_sprite.ReticleAnimator()
        self._reticles: Dict[str, ReticleState] = {}

    def get_reticle(self, reticle_id: str = "center") -> ReticleState:
        rid = str(reticle_id)
        st = self._reticles.get(rid)
        if st is not None:
            return st
        # Default neutral state.
        st = ReticleState(
            focus=ReticleFocus(on_target=False, victim_id=0),
            view_dir=(0.0, 0.0, 1.0),
            aim_point=None,
            tracked_victim_id=0,
            tracked_pos=None,
            centered_on_track=False,
            los_confirmed=False,
            los_last_check_s=0.0,
            weapon_type_primary="",
            weapon_independent_aim=False,
            weapon_off_boresight_deg=0.0,
            los_confirmed_target_id=0,
            los_confirmed_weapon_type="",
        )
        self._reticles[rid] = st
        return st

    def set_reticle(self, *, reticle_id: str = "center", state: ReticleState) -> ReticleState:
        rid = str(reticle_id)
        self._reticles[rid] = state
        return state

    def solve_reticle(
        self,
        *,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        t_max: float,
        nodes_pos: Optional[np.ndarray] = None,
        nodes_radius: Optional[np.ndarray] = None,
        depth_finder: Optional[Callable[[np.ndarray, np.ndarray, float], Optional[Vec3]]] = None,
    ) -> ReticleState:
        o = np.asarray(ray_origin, dtype=np.float32).reshape((3,))
        d = _safe_normalize(np.asarray(ray_dir, dtype=np.float32).reshape((3,)))
        view_dir = (float(d[0]), float(d[1]), float(d[2]))

        focus = ReticleFocus(on_target=False, victim_id=0)
        if nodes_pos is not None and nodes_radius is not None:
            try:
                focus = raycast_nodes(
                    ray_origin=o,
                    ray_dir=d,
                    nodes_pos=np.asarray(nodes_pos, dtype=np.float32),
                    nodes_radius=np.asarray(nodes_radius, dtype=np.float32),
                    t_max=float(t_max),
                )
            except Exception:
                focus = ReticleFocus(on_target=False, victim_id=0)

        aim_point: Optional[Vec3] = None
        if depth_finder is not None:
            try:
                aim_point = depth_finder(o, d, float(t_max))
            except Exception:
                aim_point = None

        return ReticleState(focus=focus, view_dir=view_dir, aim_point=aim_point)

    def update_reticle(
        self,
        *,
        reticle_id: str = "center",
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        t_max: float,
        nodes_pos: Optional[np.ndarray] = None,
        nodes_radius: Optional[np.ndarray] = None,
        depth_finder: Optional[Callable[[np.ndarray, np.ndarray, float], Optional[Vec3]]] = None,
    ) -> ReticleState:
        st = self.solve_reticle(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            t_max=t_max,
            nodes_pos=nodes_pos,
            nodes_radius=nodes_radius,
            depth_finder=depth_finder,
        )
        return self.set_reticle(reticle_id=str(reticle_id), state=st)

    def ensure_loaded(self) -> None:
        if self.assets is None:
            self.assets = reticle_sprite.load_reticle_assets(self.folder, size_px=self.size_px)

    def reticle_stage(self, *, on_target: bool, now_s: float) -> reticle_sprite.ReticleStage:
        return self.anim.update(on_target=bool(on_target), now_s=float(now_s))

    def draw_center_reticle(
        self,
        *,
        width: int,
        height: int,
        stage: reticle_sprite.ReticleStage,
        center_px: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Draw the reticle in screen space (final HUD overlay layer)."""
        self.ensure_loaded()
        assert self.assets is not None

        try:
            from OpenGL.GL import (
                glIsEnabled,
                glMatrixMode,
                glPushMatrix,
                glPopMatrix,
                glLoadIdentity,
                glOrtho,
                glDisable,
                glEnable,
                glBlendFunc,
                glDepthMask,
                glRasterPos2f,
                glDrawPixels,
                glPixelStorei,
                GL_PROJECTION,
                GL_MODELVIEW,
                GL_BLEND,
                GL_DEPTH_TEST,
                GL_SRC_ALPHA,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                GL_UNPACK_ALIGNMENT,
            )
        except Exception:
            return

        arr = self.assets.rgba_u8.get(stage)
        if arr is None:
            return

        w = int(width)
        h = int(height)
        sz = int(self.assets.size_px)
        if center_px is None:
            cx = 0.5 * float(w)
            cy = 0.5 * float(h)
        else:
            cx = float(center_px[0])
            cy = float(center_px[1])

        x = float(cx) - 0.5 * float(sz)
        y = float(cy) - 0.5 * float(sz)

        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, float(w), 0.0, float(h), -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        try:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        except Exception:
            pass
        glRasterPos2f(float(x), float(y))
        glDrawPixels(int(sz), int(sz), GL_RGBA, GL_UNSIGNED_BYTE, arr)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_BLEND)
        glDepthMask(True)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)
