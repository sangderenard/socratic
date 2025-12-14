from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass

import numpy as np

from .flight_buffer_ctypes import GP_FlightBuffer, try_load_flight_buffer_lib


@dataclass
class FlightSnapshot:
    pos: np.ndarray
    vel: np.ndarray
    q: np.ndarray
    rho: float
    altitude: float
    r_surface: float
    last_requested_ang: float
    last_applied_ang: float
    last_ang_clamp_ratio: float
    omega_b: np.ndarray
    ctrl_mode: int
    ctrl_channel_cmd: np.ndarray
    ctrl_tau_des_b: np.ndarray
    ctrl_tau_est_b: np.ndarray
    ctrl_err_axis_b: np.ndarray
    ctrl_err_angle: float
    arm_temp: np.ndarray
    arm_eff: np.ndarray


class FlightSimRuntime:
    def __init__(self, *, tick_hz: float = 120.0, dll_path: str | None = None) -> None:
        self._tick_hz = float(max(10.0, tick_hz))
        self._dt = 1.0 / self._tick_hz

        lib, api = try_load_flight_buffer_lib(dll_path)
        if lib is None or api is None:
            raise RuntimeError("Failed to load flight buffer API from geodesic_physics.dll")

        self._lib = lib
        self._api = api

        nbytes = int(self._api.required_bytes())
        self._mem = ctypes.create_string_buffer(nbytes)
        ok = int(self._api.init(ctypes.byref(self._mem)))
        if not ok:
            raise RuntimeError("gp_flight_init failed")

        self._buf = ctypes.cast(ctypes.byref(self._mem), ctypes.POINTER(GP_FlightBuffer))

        self._running = False
        self._thread: threading.Thread | None = None

        self._swap_seq_last = int(self._api.get_swap_seq(ctypes.byref(self._mem)))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="FlightSimRuntime", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout: float = 0.5) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=float(join_timeout))
        self._thread = None

    def _loop(self) -> None:
        dt = float(self._dt)
        next_t = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            # Catch up at most a few ticks to avoid spiral of death.
            lag = now - next_t
            steps = int(min(4, max(1, round(lag / dt) + 1)))
            self._api.step(ctypes.byref(self._mem), float(dt), int(steps))
            next_t = next_t + dt * steps

    # ---- Buffer writers ----

    def set_config(
        self,
        *,
        planet_surface_r: float,
        flight_r_min: float,
        flight_r_max: float,
        atmosphere_alt_max: float | None,
        mass: float = 1.0,
        inertia_diag: tuple[float, float, float] = (1.0, 1.0, 1.0),
        terrain_heightmap: np.ndarray | None,
        terrain_height_scale: float,
        terrain_height_bias: float,
    ) -> None:
        b = self._buf.contents
        b.cfg.planet_surface_r = float(planet_surface_r)
        b.cfg.flight_r_min = float(flight_r_min)
        b.cfg.flight_r_max = float(flight_r_max)
        # Semantics:
        # - atmosphere_alt_max > 0: density falls off to 0 by that altitude
        # - atmosphere_alt_max == 0: vacuum (rho=0)
        # - atmosphere_alt_max is None: uniform atmosphere (rho=1)
        b.cfg.atmosphere_alt_max = float(atmosphere_alt_max) if atmosphere_alt_max is not None else -1.0

        b.cfg.mass = float(mass)
        b.cfg.inertia_diag[:] = [float(inertia_diag[0]), float(inertia_diag[1]), float(inertia_diag[2])]

        if isinstance(terrain_heightmap, np.ndarray) and terrain_heightmap.ndim == 2 and terrain_heightmap.size > 0:
            hm = np.asarray(terrain_heightmap, dtype=np.float32, order="C")
            # Keep a reference so the pointer stays valid.
            self._terrain_hm = hm
            ptr = hm.ctypes.data
            b.cfg.terrain_hm_ptr = int(ptr)
            b.cfg.terrain_hm_h = int(hm.shape[0])
            b.cfg.terrain_hm_w = int(hm.shape[1])
            b.cfg.terrain_hm_stride = int(hm.strides[0] // hm.itemsize)
            b.cfg.terrain_height_scale = float(terrain_height_scale)
            b.cfg.terrain_height_bias = float(terrain_height_bias)
        else:
            self._terrain_hm = None
            b.cfg.terrain_hm_ptr = 0
            b.cfg.terrain_hm_w = 0
            b.cfg.terrain_hm_h = 0
            b.cfg.terrain_hm_stride = 0
            b.cfg.terrain_height_scale = 0.0
            b.cfg.terrain_height_bias = float(terrain_height_bias)

    def reset_state(self, *, pos: np.ndarray, vel: np.ndarray, q: np.ndarray) -> None:
        p = np.asarray(pos, dtype=np.float32).reshape(3)
        v = np.asarray(vel, dtype=np.float32).reshape(3)
        qq = np.asarray(q, dtype=np.float32).reshape(4)
        b = self._buf.contents
        for i in range(2):
            b.state[i].pos[:] = [float(p[0]), float(p[1]), float(p[2])]
            b.state[i].vel[:] = [float(v[0]), float(v[1]), float(v[2])]
            b.state[i].q[:] = [float(qq[0]), float(qq[1]), float(qq[2]), float(qq[3])]
            b.state[i].omega_b[:] = [0.0, 0.0, 0.0]

    def set_arms(self, arms: list[dict], *, scale: float = 1.0) -> None:
        """Configure the lever-arm list.

        `scale` is an optional multiplier applied to body-space arm positions.
        It should NOT be tied to the planet/world `render_scale` (unit-sphere -> world).
        Arms/thrusters are ship-local geometry.
        """
        b = self._buf.contents
        # Best-effort coherence: bump controls seq around updates.
        b.hdr.controls_seq = int(b.hdr.controls_seq + 1) & 0xFFFFFFFF
        cap = int(len(b.arms))
        n = int(min(cap, len(arms)))
        b.arm_count = int(n)

        def _v3(x, default=(0.0, 0.0, 0.0)):
            if isinstance(x, (list, tuple)) and len(x) == 3:
                return float(x[0]), float(x[1]), float(x[2])
            return float(default[0]), float(default[1]), float(default[2])

        for i in range(cap):
            a = b.arms[i]
            if i >= n:
                a.type = 0
                a.input_idx = 0
                a.flags = 0
                a.pos_b[:] = [0.0, 0.0, 0.0]
                a.dir_b[:] = [0.0, 0.0, 1.0]
                a.axis_b[:] = [0.0, 1.0, 0.0]
                a.max_force = 0.0
                a.k_lift = 0.0
                a.k_drag = 0.0
                a.eff_rho_pow = 0.0
                a.eff_speed_ref = 0.0
                a.eff_speed_pow = 0.0
                a.temp_heat_rate = 0.0
                a.temp_cool_rate = 0.0
                a.temp_overheat_start = 0.0
                a.temp_overheat_end = 0.0
                continue

            spec = arms[i] if isinstance(arms[i], dict) else {}
            t = spec.get("type", "")
            if t == "thruster":
                a.type = 1
            elif t == "aero_surface":
                a.type = 2
            elif t == "drag_center":
                a.type = 3
            else:
                a.type = int(spec.get("type_id", 0))

            a.input_idx = int(spec.get("input_idx", 0))
            a.flags = int(spec.get("flags", 0))
            px, py, pz = _v3(spec.get("pos_b"))
            a.pos_b[:] = [float(px) * float(scale), float(py) * float(scale), float(pz) * float(scale)]
            dx, dy, dz = _v3(spec.get("dir_b"), default=(0.0, 0.0, 1.0))
            a.dir_b[:] = [float(dx), float(dy), float(dz)]
            ax, ay, az = _v3(spec.get("axis_b"), default=(0.0, 1.0, 0.0))
            a.axis_b[:] = [float(ax), float(ay), float(az)]
            a.max_force = float(spec.get("max_force", 0.0))
            a.k_lift = float(spec.get("k_lift", 0.0))
            a.k_drag = float(spec.get("k_drag", 0.0))

            eff = spec.get("eff") if isinstance(spec.get("eff"), dict) else spec
            a.eff_rho_pow = float(eff.get("eff_rho_pow", 0.0))
            a.eff_speed_ref = float(eff.get("eff_speed_ref", 0.0))
            a.eff_speed_pow = float(eff.get("eff_speed_pow", 0.0))
            a.temp_heat_rate = float(eff.get("temp_heat_rate", 0.0))
            a.temp_cool_rate = float(eff.get("temp_cool_rate", 0.0))
            a.temp_overheat_start = float(eff.get("temp_overheat_start", 0.0))
            a.temp_overheat_end = float(eff.get("temp_overheat_end", 0.0))

        b.hdr.controls_seq = int(b.hdr.controls_seq + 1) & 0xFFFFFFFF

    def set_controls(
        self,
        *,
        throttle: float,
        strafe: float,
        desired_q: np.ndarray,
        max_ang_rate: float,
        mode: str = "auto",
        auto_kp: float = 6.0,
        auto_kd: float = 2.5,
        debug_print: bool = False,
        channels: list[float] | tuple[float, ...] | None = None,
        thrust_accel: float,
        strafe_accel: float,
        lift_k: float,
        drag_k: float,
        gravity_g: float,
        max_speed: float,
    ) -> None:
        b = self._buf.contents
        # Write controls with a sequence bump for best-effort coherence.
        b.hdr.controls_seq = int(b.hdr.controls_seq + 1) & 0xFFFFFFFF
        b.ctl.throttle = float(throttle)
        b.ctl.strafe = float(strafe)
        dq = np.asarray(desired_q, dtype=np.float32).reshape(4)
        b.ctl.desired_q[:] = [float(dq[0]), float(dq[1]), float(dq[2]), float(dq[3])]
        b.ctl.max_ang_rate = float(max_ang_rate)
        b.ctl.mode = 1 if str(mode).lower().startswith("auto") else 0
        b.ctl.auto_kp = float(auto_kp)
        b.ctl.auto_kd = float(auto_kd)
        b.ctl.thrust_accel = float(thrust_accel)
        b.ctl.strafe_accel = float(strafe_accel)
        b.ctl.lift_k = float(lift_k)
        b.ctl.drag_k = float(drag_k)
        b.ctl.gravity_g = float(gravity_g)
        b.ctl.max_speed = float(max_speed)
        if channels is None:
            ch = [0.0] * 8
        else:
            ch = [float(x) for x in channels][:8]
            if len(ch) < 8:
                ch = ch + [0.0] * (8 - len(ch))
        b.ctl.channels[:] = ch
        b.ctl.flags = 1 | (2 if bool(debug_print) else 0)
        b.hdr.controls_seq = int(b.hdr.controls_seq + 1) & 0xFFFFFFFF

    # ---- Snapshot reader ----

    def snapshot(self, *, request_swap: bool = True) -> FlightSnapshot:
        if request_swap:
            self._api.request_swap(ctypes.byref(self._mem))

        seq = int(self._api.get_swap_seq(ctypes.byref(self._mem)))
        self._swap_seq_last = seq

        b = self._buf.contents
        fi = int(b.hdr.front_idx) & 1
        st = b.state[fi]

        pos = np.array([float(st.pos[0]), float(st.pos[1]), float(st.pos[2])], dtype=np.float32)
        vel = np.array([float(st.vel[0]), float(st.vel[1]), float(st.vel[2])], dtype=np.float32)
        q = np.array([float(st.q[0]), float(st.q[1]), float(st.q[2]), float(st.q[3])], dtype=np.float32)
        omega_b = np.array([float(st.omega_b[0]), float(st.omega_b[1]), float(st.omega_b[2])], dtype=np.float32)
        ctrl = st.ctrl
        ctrl_mode = int(ctrl.mode)
        ctrl_channel_cmd = np.array([float(x) for x in ctrl.channel_cmd], dtype=np.float32)
        ctrl_tau_des_b = np.array([float(x) for x in ctrl.tau_des_b], dtype=np.float32)
        ctrl_tau_est_b = np.array([float(x) for x in ctrl.tau_est_b], dtype=np.float32)
        ctrl_err_axis_b = np.array([float(x) for x in ctrl.err_axis_b], dtype=np.float32)
        ctrl_err_angle = float(ctrl.err_angle)
        arm_temp = np.array([float(x) for x in st.arm_temp], dtype=np.float32)
        arm_eff = np.array([float(x) for x in st.arm_eff], dtype=np.float32)
        return FlightSnapshot(
            pos=pos,
            vel=vel,
            q=q,
            rho=float(st.rho),
            altitude=float(st.altitude),
            r_surface=float(st.r_surface),
            last_requested_ang=float(st.last_requested_ang),
            last_applied_ang=float(st.last_applied_ang),
            last_ang_clamp_ratio=float(st.last_ang_clamp_ratio),
            omega_b=omega_b,
            ctrl_mode=ctrl_mode,
            ctrl_channel_cmd=ctrl_channel_cmd,
            ctrl_tau_des_b=ctrl_tau_des_b,
            ctrl_tau_est_b=ctrl_tau_est_b,
            ctrl_err_axis_b=ctrl_err_axis_b,
            ctrl_err_angle=ctrl_err_angle,
            arm_temp=arm_temp,
            arm_eff=arm_eff,
        )
