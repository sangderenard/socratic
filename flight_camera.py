from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _safe_normalize(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    n = _safe_norm(v)
    if not np.isfinite(n) or n <= eps:
        return np.zeros_like(v)
    return v / n


def clamp_radius_band(p: np.ndarray, *, r_min: float, r_max: float) -> np.ndarray:
    r = _safe_norm(p)
    if not np.isfinite(r) or r <= 1e-12:
        p[:] = np.array([0.0, r_min, 0.0], dtype=p.dtype)
        return p
    r_clamped = min(max(r, float(r_min)), float(r_max))
    if abs(r_clamped - r) > 0.0:
        p *= (r_clamped / r)
    return p


# Quaternion helpers: q = [w, x, y, z]

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = _safe_norm(q)
    if not np.isfinite(n) or n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32, copy=False)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = _safe_normalize(axis.astype(np.float32, copy=False))
    half = 0.5 * float(angle_rad)
    s = float(math.sin(half))
    return np.array([math.cos(half), axis_n[0] * s, axis_n[1] * s, axis_n[2] * s], dtype=np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])], dtype=np.float32)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # v' = q * [0,v] * conj(q)
    qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:].astype(np.float32, copy=False)


def _rotate_vec_axis_angle(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of vector v about unit axis by angle_rad."""
    a = _safe_normalize(axis.astype(np.float32, copy=False))
    if _safe_norm(a) <= 1e-6:
        return v
    th = float(angle_rad)
    c = float(math.cos(th))
    s = float(math.sin(th))
    return (v * c + np.cross(a, v) * s + a * float(np.dot(a, v)) * (1.0 - c)).astype(np.float32, copy=False)


def quat_from_basis(right: np.ndarray, up: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Create a quaternion from an orthonormal basis.

    Basis convention matches `basis()`:
    - local +X is `right`
    - local +Y is `up`
    - local +Z is `fwd`
    """
    r = _safe_normalize(right.astype(np.float32, copy=False))
    u = _safe_normalize(up.astype(np.float32, copy=False))
    f = _safe_normalize(fwd.astype(np.float32, copy=False))
    if _safe_norm(r) <= 1e-6 or _safe_norm(u) <= 1e-6 or _safe_norm(f) <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Rotation matrix with columns = basis vectors in world space.
    # M maps local -> world.
    m00, m10, m20 = float(r[0]), float(r[1]), float(r[2])
    m01, m11, m21 = float(u[0]), float(u[1]), float(u[2])
    m02, m12, m22 = float(f[0]), float(f[1]), float(f[2])

    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=np.float32))


@dataclass
class PlanetFlightCamera:
    planet_surface_r: float
    flight_r_min: float
    flight_r_max: float

    # Optional terrain heightmap (equirectangular lon/lat, grayscale in [0..1]).
    # When provided, the local terrain surface radius is:
    #   r_surface(pos) = planet_surface_r + (h(pos) - terrain_height_bias) * terrain_height_scale
    terrain_heightmap: np.ndarray | None = None
    terrain_height_scale: float = 0.0
    terrain_height_bias: float = 0.5

    yaw_rate: float = 1.6
    pitch_rate: float = 1.2
    roll_rate: float = 1.4
    speed: float = 0.65
    climb_speed: float = 0.65
    deadzone: float = 0.08

    # Inertial flight dynamics.
    # When enabled, camera motion uses a simple integrator with velocity state.
    inertial: bool = False
    thrust_accel: float = 0.65
    strafe_accel: float = 0.0
    lift_k: float = 0.0
    drag_k: float = 0.0
    gravity_g: float = 0.0
    max_speed: float = 0.0

    # Optional atmosphere model for inertial flight.
    # If provided (>0), atmospheric influence decreases with altitude and reaches ~0 at this altitude.
    # This is used to scale drag and the effective max-speed clamp so that:
    # - in atmosphere: speed is limited (drag + clamp)
    # - in vacuum: drag ~ 0 and clamp is disabled
    atmosphere_alt_max: float | None = None

    # Persistent engine throttle in [-1, 1] (reverse..forward). This is integrated
    # by the caller (controller mapping) and used as the thrust scalar.
    engine_throttle: float = 0.0

    # If enabled, when there is movement input but no active look input,
    # steer heading toward the movement direction on the local horizon.
    # This makes surface motion feel like "you go where you point" rather than strafing.
    auto_heading_on_move: bool = True
    auto_heading_rate: float = 8.0

    # Angular traverse rate (radians/sec) for "moving over the planet".
    angular_speed: float = 0.9

    # Radial attractor: pulls camera toward a target radius (grounded feel).
    radius_target: float | None = None
    radius_attractor_k: float = 2.0

    def __post_init__(self) -> None:
        self.pos = np.array([0.0, float(self.flight_r_min), 0.0], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        # Camera attitude is expressed as:
        # - `heading_t`: unit tangent direction ("forward" projected onto local horizon)
        # - `pitch`: radians, positive pitches "up" (away from planet)
        # We also keep a quaternion `q` for compatibility with the renderer and any callers.
        self.heading_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # Continuity helpers: last stable tangent basis.
        # These prevent large visual flips when the instantaneous tangent projection becomes
        # numerically degenerate (e.g., heading accumulates radial drift).
        self._fwd_level_last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._right_last = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.pitch = float(-0.25)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if self.radius_target is None:
            self.radius_target = float(self.flight_r_min)
        rmin, rmax = self._dynamic_radius_band_at_pos(self.pos)
        self.radius_target = max(float(self.radius_target), float(rmin))
        clamp_radius_band(self.pos, r_min=float(rmin), r_max=float(rmax))
        self._rebuild_attitude()

    def reset_north_pole(self) -> None:
        self.pos[:] = np.array([0.0, float(self.flight_r_min), 0.0], dtype=np.float32)
        self.vel[:] = 0.0
        # Start with a slight nose-down pitch so the ground/horizon is visible.
        self.heading_t[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._fwd_level_last[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._right_last[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.pitch = float(-0.25)
        if self.radius_target is None:
            self.radius_target = float(self.flight_r_min)
        rmin, rmax = self._dynamic_radius_band_at_pos(self.pos)
        self.radius_target = max(float(self.radius_target), float(rmin))
        clamp_radius_band(self.pos, r_min=float(rmin), r_max=float(rmax))
        self._rebuild_attitude()

    def default_eye_center(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        r = float(self.flight_r_min)
        eye = (0.0, r, 0.0)
        center = (0.0, r, 1.0)
        return eye, center

    def altitude(self) -> float:
        return max(0.0, _safe_norm(self.pos) - float(self.terrain_surface_r_at_pos(self.pos)))

    def atmosphere_ratio(self) -> float:
        """Return a [0..1] atmospheric pressure/density ratio at the current altitude."""
        try:
            alt_max = self.atmosphere_alt_max
            if alt_max is None:
                return 1.0
            alt_max_f = float(alt_max)
            if not np.isfinite(alt_max_f) or alt_max_f <= 1e-9:
                return 1.0
            alt = float(self.altitude())
            x = float(max(0.0, min(1.0, alt / alt_max_f)))
            # Simple smooth falloff to 0 at x=1.
            rho = (1.0 - x)
            rho = float(max(0.0, min(1.0, rho * rho)))
            return rho
        except Exception:
            return 1.0

    def _heightmap_sample_bilinear(self, u: float, v: float) -> float:
        hm = self.terrain_heightmap
        if hm is None or not isinstance(hm, np.ndarray) or hm.size <= 0 or hm.ndim != 2:
            return 0.0

        h, w = int(hm.shape[0]), int(hm.shape[1])
        if h <= 0 or w <= 0:
            return 0.0
        if h == 1 or w == 1:
            return float(hm[0, 0])

        uu = float(u) % 1.0
        vv = max(0.0, min(1.0, float(v)))

        x = uu * float(w - 1)
        y = vv * float(h - 1)
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        sx = float(x - float(x0))
        sy = float(y - float(y0))

        h00 = float(hm[y0, x0])
        h10 = float(hm[y0, x1])
        h01 = float(hm[y1, x0])
        h11 = float(hm[y1, x1])
        h0 = (1.0 - sx) * h00 + sx * h10
        h1 = (1.0 - sx) * h01 + sx * h11
        return (1.0 - sy) * h0 + sy * h1

    def terrain_surface_r_at_pos(self, pos: np.ndarray) -> float:
        base = float(self.planet_surface_r)
        if (
            self.terrain_heightmap is None
            or not isinstance(self.terrain_heightmap, np.ndarray)
            or self.terrain_heightmap.size <= 0
            or float(self.terrain_height_scale) <= 0.0
        ):
            return base

        r = _safe_norm(pos)
        if not np.isfinite(r) or r <= 1e-9:
            return base

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        lat = math.asin(max(-1.0, min(1.0, y / float(r))))
        lon = math.atan2(z, x)
        u = lon / (2.0 * math.pi) + 0.5
        v = lat / math.pi + 0.5

        h01 = self._heightmap_sample_bilinear(u, v)
        disp = (float(h01) - float(self.terrain_height_bias)) * float(self.terrain_height_scale)
        return base + disp

    def _dynamic_radius_band_at_pos(self, pos: np.ndarray) -> tuple[float, float]:
        base = float(self.planet_surface_r)
        clearance_min = max(0.0, float(self.flight_r_min) - base)
        clearance_max = max(0.0, float(self.flight_r_max) - base)
        surf = float(self.terrain_surface_r_at_pos(pos))
        r_min = max(0.0, surf + clearance_min)
        r_max = max(r_min, surf + clearance_max)
        return r_min, r_max

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = quat_normalize(self.q)
        right = quat_rotate(q, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        up = quat_rotate(q, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        fwd = quat_rotate(q, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        return right, up, fwd

    def _planet_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, up_radial, fwd_level) where fwd_level is tangent (horizon forward)."""
        up_rad = self.planet_up()

        # Ensure heading is tangent; if it degenerates, pick a stable tangent.
        fwd_level = (self.heading_t - up_rad * float(np.dot(self.heading_t, up_rad))).astype(np.float32, copy=False)
        fwd_level = _safe_normalize(fwd_level)
        if _safe_norm(fwd_level) <= 1e-6:
            # First try to reuse last stable tangent forward, projected into the current tangent plane.
            fwd_level = (self._fwd_level_last - up_rad * float(np.dot(self._fwd_level_last, up_rad))).astype(np.float32, copy=False)
            fwd_level = _safe_normalize(fwd_level)
        if _safe_norm(fwd_level) <= 1e-6:
            # As a last resort, fall back to a fixed world reference (still projected tangent).
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(np.dot(ref, up_rad))) > 0.95:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            fwd_level = _safe_normalize(ref - up_rad * float(np.dot(ref, up_rad)))

        # Right-handed tangent basis: right = up x forward.
        right = _safe_normalize(np.cross(up_rad, fwd_level))
        if _safe_norm(right) <= 1e-6:
            # Prefer the last right projected tangent for continuity.
            right = (self._right_last - up_rad * float(np.dot(self._right_last, up_rad))).astype(np.float32, copy=False)
            right = _safe_normalize(right)
        if _safe_norm(right) <= 1e-6:
            right = _safe_normalize(np.cross(up_rad, np.array([1.0, 0.0, 0.0], dtype=np.float32)))

        # Update continuity cache.
        if _safe_norm(fwd_level) > 1e-6:
            self._fwd_level_last = fwd_level.astype(np.float32, copy=False)
        if _safe_norm(right) > 1e-6:
            self._right_last = right.astype(np.float32, copy=False)
        return right, up_rad, fwd_level

    def _rebuild_attitude(self) -> None:
        right, up_rad, fwd_level = self._planet_frame()
        # Clamp pitch away from singularities.
        self.pitch = float(max(-1.45, min(1.45, float(self.pitch))))
        cp = float(math.cos(float(self.pitch)))
        sp = float(math.sin(float(self.pitch)))
        fwd = _safe_normalize(fwd_level * cp + up_rad * sp)
        if _safe_norm(fwd) <= 1e-6:
            fwd = fwd_level
        # Build a proper right-handed basis.
        # For (right, up, fwd) we require: right x up = fwd, so up = fwd x right.
        up = _safe_normalize(np.cross(fwd, right))
        if _safe_norm(up) <= 1e-6:
            up = up_rad
        self.q = quat_from_basis(right, up, fwd)

    def planet_up(self) -> np.ndarray:
        """Radial up away from the planet center at the camera position."""
        return _safe_normalize(self.pos.astype(np.float32, copy=False))

    def step(
        self,
        *,
        dt: float,
        move_x: float,
        move_y: float,
        look_x: float,
        look_y: float,
        roll_in: float,
        climb: float = 0.0,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        dz = float(self.deadzone)
        if abs(move_x) < dz:
            move_x = 0.0
        if abs(move_y) < dz:
            move_y = 0.0
        if abs(look_x) < dz:
            look_x = 0.0
        if abs(look_y) < dz:
            look_y = 0.0
        if abs(roll_in) < dz:
            roll_in = 0.0
        if abs(climb) < dz:
            climb = 0.0

        dt_f = float(max(1e-4, dt))

        # Build the current local planet frame.
        right, up_rad, fwd_level = self._planet_frame()

        # Ship-relative look: yaw changes heading on the horizon; pitch changes look pitch.
        if look_x != 0.0:
            d_yaw = float(self.yaw_rate) * dt_f * float(look_x)
            self.heading_t = _rotate_vec_axis_angle(self.heading_t, up_rad, d_yaw)
            # Keep heading strictly tangent to avoid numerical drift that can cause discrete flips.
            self.heading_t = (self.heading_t - up_rad * float(np.dot(self.heading_t, up_rad))).astype(np.float32, copy=False)
            self.heading_t = _safe_normalize(self.heading_t)

        if look_y != 0.0:
            self.pitch += float(self.pitch_rate) * dt_f * float(look_y)

        # Roll is intentionally ignored for a steady horizon (renderer up is radial).
        _ = roll_in

        # Recompute after look changes.
        right, up_rad, fwd_level = self._planet_frame()

        # Inertial flight model: integrate velocity with thrust + lift + drag + gravity.
        if self.inertial:
            throttle = float(max(-1.0, min(1.0, float(move_y))))
            strafe = float(max(-1.0, min(1.0, float(move_x))))

            # Ensure attitude is current before applying thrust direction.
            self._rebuild_attitude()
            right_b, up_b, fwd_b = self.basis()
            up_rad = self.planet_up()

            a = np.zeros(3, dtype=np.float32)
            if abs(throttle) > 1e-9 and float(self.thrust_accel) != 0.0:
                a += fwd_b.astype(np.float32, copy=False) * (float(self.thrust_accel) * throttle)
            if abs(strafe) > 1e-9 and float(self.strafe_accel) != 0.0:
                a += right_b.astype(np.float32, copy=False) * (float(self.strafe_accel) * strafe)

            v = self.vel.astype(np.float32, copy=False)
            speed = _safe_norm(v)

            # Atmosphere factor (0..1): scales drag and the effective speed clamp.
            rho = float(self.atmosphere_ratio())
            if speed > 1e-6:
                vhat = (v / float(speed)).astype(np.float32, copy=False)
                # Lift direction: component of the craft's up axis perpendicular to velocity.
                lift_dir = (up_b - vhat * float(np.dot(up_b, vhat))).astype(np.float32, copy=False)
                lift_dir = _safe_normalize(lift_dir)
                if _safe_norm(lift_dir) > 1e-6 and float(self.lift_k) != 0.0:
                    a += lift_dir * (float(self.lift_k) * float(speed) * float(speed))

                # Quadratic drag opposing velocity.
                if float(self.drag_k) != 0.0 and rho > 0.0:
                    a += (-vhat) * (float(self.drag_k) * rho * float(speed) * float(speed))

            # Simple gravity toward planet center.
            if float(self.gravity_g) != 0.0:
                a += (-up_rad) * float(self.gravity_g)

            # Semi-implicit Euler.
            self.vel = (self.vel + a * dt_f).astype(np.float32, copy=False)

            # Atmosphere-limited max speed: in vacuum (rho ~ 0) this effectively disables
            # clamping; in atmosphere it enforces a finite speed envelope.
            ms = float(self.max_speed)
            if ms > 1e-6:
                if rho <= 1e-4:
                    ms_eff = 0.0
                else:
                    ms_eff = float(ms / math.sqrt(float(max(1e-6, rho))))
                if ms_eff > 1e-6:
                    sp2 = _safe_norm(self.vel)
                    if sp2 > ms_eff:
                        self.vel = (self.vel * (ms_eff / float(sp2))).astype(np.float32, copy=False)

            self.pos = (self.pos + self.vel * dt_f).astype(np.float32, copy=False)

            # Constrain to flight radius band; if we hit a boundary, remove radial velocity.
            r_before = _safe_norm(self.pos)
            rmin, rmax = self._dynamic_radius_band_at_pos(self.pos)
            if self.radius_target is not None:
                self.radius_target = max(float(self.radius_target), float(rmin))
            clamp_radius_band(self.pos, r_min=float(rmin), r_max=float(rmax))
            r_after = _safe_norm(self.pos)
            if abs(float(r_after) - float(r_before)) > 1e-7:
                up_new = self.planet_up()
                self.vel = (self.vel - up_new * float(np.dot(self.vel, up_new))).astype(np.float32, copy=False)

            # Update quaternion from the stable heading/pitch.
            self._rebuild_attitude()

            eye = (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
            _, _, fwd = self.basis()
            center_v = self.pos + fwd
            center = (float(center_v[0]), float(center_v[1]), float(center_v[2]))
            return eye, center

        # Surface translation: rotate position about axis = up x dir (great-circle motion).
        r = _safe_norm(self.pos)
        if r <= 1e-6:
            r = float(self.flight_r_min)
            self.pos[:] = up_rad * r

        move_mag = float(math.sqrt(float(move_x) * float(move_x) + float(move_y) * float(move_y)))
        if move_mag > 1e-6:
            dir_t = _safe_normalize((right * float(move_x) + fwd_level * float(move_y)).astype(np.float32, copy=False))
            if _safe_norm(dir_t) > 1e-6:
                # Optional: steer heading toward movement direction when the user isn't actively looking.
                # Keeps the view oriented with the arc you are traversing.
                if (
                    self.auto_heading_on_move
                    and abs(float(look_x)) <= 1e-6
                    and abs(float(look_y)) <= 1e-6
                ):
                    a = 1.0 - math.exp(-float(self.auto_heading_rate) * dt_f)
                    up_use = up_rad
                    # Blend in tangent space and renormalize.
                    h_new = (1.0 - float(a)) * fwd_level + float(a) * dir_t
                    h_new = (h_new - up_use * float(np.dot(h_new, up_use))).astype(np.float32, copy=False)
                    h_new = _safe_normalize(h_new)
                    if _safe_norm(h_new) > 1e-6:
                        self.heading_t = h_new
                        # Recompute local frame so translation uses the steered heading.
                        right, up_rad, fwd_level = self._planet_frame()
                        dir_t = _safe_normalize((right * float(move_x) + fwd_level * float(move_y)).astype(np.float32, copy=False))

                axis = _safe_normalize(np.cross(up_rad, dir_t))
                dtheta = (float(self.speed) * dt_f * move_mag) / float(max(1e-6, r))
                # Rotate position and heading together for stable, non-spinning planet-relative motion.
                self.pos = _rotate_vec_axis_angle(self.pos, axis, dtheta)
                self.heading_t = _rotate_vec_axis_angle(self.heading_t, axis, dtheta)
                # Reproject heading onto the new tangent plane.
                up_new = self.planet_up()
                self.heading_t = _safe_normalize(self.heading_t - up_new * float(np.dot(self.heading_t, up_new)))

        # Radial climb/descend.
        if climb != 0.0:
            up_rad = self.planet_up()
            self.pos += up_rad * (float(self.climb_speed) * dt_f * float(climb))
            self.radius_target = float(_safe_norm(self.pos))

        # Gentle attractor toward a target radius to keep you "settled" over the surface.
        if self.radius_target is not None and self.radius_attractor_k > 0.0:
            r = _safe_norm(self.pos)
            if np.isfinite(r) and r > 1e-9:
                dr = float(self.radius_target) - float(r)
                k = float(self.radius_attractor_k)
                self.pos += (self.pos / r) * (dr * k * dt_f)

        rmin, rmax = self._dynamic_radius_band_at_pos(self.pos)
        if self.radius_target is not None:
            self.radius_target = max(float(self.radius_target), float(rmin))
        clamp_radius_band(self.pos, r_min=float(rmin), r_max=float(rmax))

        # Update quaternion from the stable heading/pitch.
        self._rebuild_attitude()

        eye = (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        # Look direction follows the camera attitude (heading + pitch).
        _, _, fwd = self.basis()
        center_v = self.pos + fwd
        center = (float(center_v[0]), float(center_v[1]), float(center_v[2]))
        return eye, center


def lookat_up_away_from_planet(
    *,
    eye: np.ndarray,
    center: np.ndarray,
    fallback_up: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Build a stable up vector for gluLookAt.

    Requirement: up points away from the planet center (radial) rather than a fixed world axis.
    We also re-orthogonalize so up is not parallel to forward.
    """
    if fallback_up is None:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    fwd = _safe_normalize((center - eye).astype(np.float32, copy=False))
    up_radial = _safe_normalize(eye.astype(np.float32, copy=False))
    if _safe_norm(up_radial) <= 1e-6:
        up_radial = _safe_normalize(fallback_up.astype(np.float32, copy=False))

    # If forward is nearly parallel to up, fall back.
    if abs(float(np.dot(fwd, up_radial))) > 0.98:
        up_radial = _safe_normalize(fallback_up.astype(np.float32, copy=False))

    right = np.cross(fwd, up_radial)
    right = _safe_normalize(right)
    if _safe_norm(right) <= 1e-6:
        return float(up_radial[0]), float(up_radial[1]), float(up_radial[2])

    up = np.cross(right, fwd)
    up = _safe_normalize(up)
    return float(up[0]), float(up[1]), float(up[2])
