from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import weapon_runtime


@dataclass
class DepthHit:
    valid: bool
    point: Optional[tuple[float, float, float]] = None


class CWeaponDepthFinder:
    """C-backed reticle depth finder.

    This is intentionally reticle-centric: given a world-space ray (origin+dir),
    it asks the C weapon sim (beam/laser mode) where that ray would intercept
    world geometry (terrain) or node spheres.

    The caller supplies scene context (planet params, terrain heightmap ptr info,
    nodes ptr info) so this module stays UI/renderer agnostic.
    """

    def __init__(self) -> None:
        self._lib = None
        self._mod = None

    def _ensure_loaded(self) -> bool:
        if self._lib is not None and self._mod is not None:
            return True
        try:
            from c_physics import weapon_sim_ctypes as _wsc

            self._mod = _wsc
            self._lib = _wsc.load_lib()
            return True
        except Exception:
            self._lib = None
            self._mod = None
            return False

    def probe_weapon_impact(
        self,
        *,
        weap_rt: "weapon_runtime.WeaponRuntime",
        weapon_type: str,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        max_dist: float,
        planet_surface_r: float | None,
        gravity_g: float | None,
        terrain_heightmap: np.ndarray | None,
        terrain_height_scale: float | None,
        terrain_height_bias: float | None,
        nodes_pos: np.ndarray | None,
        nodes_radius: np.ndarray | None,
    ) -> DepthHit:
        if not self._ensure_loaded():
            return DepthHit(valid=False, point=None)

        import weapon_runtime

        wtype = str(weapon_type or "")
        if not wtype:
            return DepthHit(valid=False, point=None)

        # Pull weapon stats so the probe uses the same codepath as firing.
        cfg = weapon_runtime.get_weapon_type_stats(weap_rt.stats, wtype)
        kin = cfg.get("kinematics") if isinstance(cfg.get("kinematics"), dict) else None
        sim = cfg.get("sim") if isinstance(cfg.get("sim"), dict) else None

        def _mode_enum(m: str) -> int:
            m = str(m or "")
            if m == "beam":
                return 0
            if m == "fire":
                return 1
            if m == "fall":
                return 2
            return -1

        try:
            k_mode = _mode_enum(kin.get("mode") if kin else "")
            inherit = bool(kin.get("inherit_ship_velocity")) if kin else False
            add_v = kin.get("add_velocity") if kin else 0.0
            if add_v is None:
                add_v = 0.0
            sim_points_cfg = int(sim.get("sim_points", 2) if sim else 2)
            sim_points = int(max(2, min(16, sim_points_cfg)))
            t_end = float(sim.get("t_end", 0.0) if sim else 0.0)
            # Depth-finder wants the longest meaningful beam.
            beam_len = float(sim.get("beam_len", max_dist) if sim else max_dist)
            if k_mode == 0:  # beam
                beam_len = float(max(1.0, min(float(max_dist), beam_len)))
                t_end = 0.0
                sim_points = int(max(2, min(16, sim_points)))
            else:
                # For ballistic, clamp t_end to a sane range.
                if not np.isfinite(t_end) or t_end <= 0.0:
                    # Rough fallback: time to traverse max_dist at muzzle speed.
                    v0 = float(abs(float(add_v)))
                    t_end = float(max_dist) / float(max(1e-3, v0))
                t_end = float(max(0.01, min(30.0, t_end)))
                beam_len = float(0.0)
            drop_off = float(sim.get("drop_off", 0.0) if sim else 0.0)
            if k_mode < 0:
                return DepthHit(valid=False, point=None)
        except Exception:
            return DepthHit(valid=False, point=None)

        ro = np.asarray(ray_origin, dtype=np.float32).reshape((3,))
        rd = np.asarray(ray_dir, dtype=np.float32).reshape((3,))
        dn = float(np.linalg.norm(rd))
        if not (dn > 1e-6):
            return DepthHit(valid=False, point=None)
        rd = (rd / dn).astype(np.float32, copy=False)

        hm_info = None
        hm_ref = None
        try:
            if (
                terrain_heightmap is not None
                and isinstance(terrain_heightmap, np.ndarray)
                and terrain_heightmap.size
                and terrain_height_scale is not None
                and float(terrain_height_scale) != 0.0
            ):
                hm_np = np.ascontiguousarray(np.asarray(terrain_heightmap, dtype=np.float32))
                h_h = int(hm_np.shape[0])
                h_w = int(hm_np.shape[1])
                if h_h > 1 and h_w > 1:
                    hm_ref = hm_np
                    hm_info = {
                        "ptr": int(hm_np.__array_interface__["data"][0]),
                        "w": h_w,
                        "h": h_h,
                        "stride": h_w,
                        "height_scale": float(terrain_height_scale),
                        "height_bias": float(0.5 if terrain_height_bias is None else terrain_height_bias),
                    }
        except Exception:
            hm_info = None
            hm_ref = None

        nodes_info = None
        nodes_ref = None
        try:
            if nodes_pos is not None and nodes_radius is not None:
                p = np.ascontiguousarray(np.asarray(nodes_pos, dtype=np.float32))
                r = np.ascontiguousarray(np.asarray(nodes_radius, dtype=np.float32))
                if p.ndim == 2 and p.shape[1] == 3 and r.ndim == 1 and r.shape[0] == p.shape[0] and p.shape[0] > 0:
                    nodes_ref = (p, r)
                    nodes_info = {
                        "pos_ptr": int(p.__array_interface__["data"][0]),
                        "rad_ptr": int(r.__array_interface__["data"][0]),
                        "count": int(p.shape[0]),
                        "pos_stride": 3,
                        "rad_stride": 1,
                    }
        except Exception:
            nodes_info = None
            nodes_ref = None

        batch = self._mod.make_batch(
            requests=[
                {
                    "request_id": 0,
                    "weapon_slot": 0,
                    "analog": 0.0,
                    "source": "reticle",
                    "weapon_type": str(wtype),
                    "kinematics_mode": int(k_mode),
                    "inherit_ship_velocity": bool(inherit),
                    "add_velocity": float(add_v),
                    "sim_points": int(sim_points),
                    "sim_t_end": float(t_end),
                    "sim_beam_len": float(beam_len),
                    "sim_drop_off": float(drop_off),
                    "ship_pos": (float(ro[0]), float(ro[1]), float(ro[2])),
                    "ship_vel": (0.0, 0.0, 0.0),
                    "ship_fwd": (float(rd[0]), float(rd[1]), float(rd[2])),
                    "weapon_origin": (float(ro[0]), float(ro[1]), float(ro[2])),
                    "weapon_dir": (float(rd[0]), float(rd[1]), float(rd[2])),
                    "planet_surface_r": (None if planet_surface_r is None else float(planet_surface_r)),
                    "gravity_g": (None if gravity_g is None else float(gravity_g)),
                    "terrain_heightmap": hm_info,
                    "nodes": nodes_info,
                }
            ],
            max_points=16,
        )

        # Ensure refs are kept alive for the duration of the call.
        _keepalive = (hm_ref, nodes_ref)
        _ = _keepalive

        ok = bool(self._mod.process_batch(lib=self._lib, batch=batch))
        if not ok:
            return DepthHit(valid=False, point=None)
        out0 = batch.out[0]
        if int(getattr(out0, "ok", 0) or 0) == 0:
            return DepthHit(valid=False, point=None)
        if int(getattr(out0, "impact_valid", 0) or 0) == 0:
            return DepthHit(valid=False, point=None)
        ip = out0.impact_point
        return DepthHit(valid=True, point=(float(ip[0]), float(ip[1]), float(ip[2])))

    def probe_laser_impact(
        self,
        *,
        weap_rt: "weapon_runtime.WeaponRuntime",
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        max_dist: float,
        planet_surface_r: float | None,
        gravity_g: float | None,
        terrain_heightmap: np.ndarray | None,
        terrain_height_scale: float | None,
        terrain_height_bias: float | None,
        nodes_pos: np.ndarray | None,
        nodes_radius: np.ndarray | None,
    ) -> DepthHit:
        # Compatibility wrapper: prior callers used a fixed laser probe.
        return self.probe_weapon_impact(
            weap_rt=weap_rt,
            weapon_type="laser",
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            max_dist=max_dist,
            planet_surface_r=planet_surface_r,
            gravity_g=gravity_g,
            terrain_heightmap=terrain_heightmap,
            terrain_height_scale=terrain_height_scale,
            terrain_height_bias=terrain_height_bias,
            nodes_pos=nodes_pos,
            nodes_radius=nodes_radius,
        )
