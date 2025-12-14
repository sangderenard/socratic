from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


Vec3 = Tuple[float, float, float]


def load_weapon_stats(path: str = "weapon_stats.json") -> dict[str, Any]:
    if not os.path.exists(path):
        return {"version": 0, "weapon_types": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"version": 0, "weapon_types": {}}
    except Exception:
        return {"version": 0, "weapon_types": {}}


def get_weapon_type_stats(stats_root: dict[str, Any], weapon_type: str) -> dict[str, Any]:
    wt = stats_root.get("weapon_types")
    if not isinstance(wt, dict):
        return {}
    v = wt.get(str(weapon_type))
    return v if isinstance(v, dict) else {}


@dataclass(frozen=True)
class ShipSnapshot:
    pos: Optional[Vec3] = None
    vel: Optional[Vec3] = None
    fwd: Optional[Vec3] = None
    planet_surface_r: Optional[float] = None
    gravity_g: Optional[float] = None
    terrain_heightmap: Optional[Dict[str, Any]] = None
    # Keep-alive refs for any in-process pointers we hand to C.
    terrain_heightmap_ref: Any | None = None
    nodes_ref: Any | None = None
    nodes: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class WeaponFireRequest:
    request_id: int
    t_submit: float
    weapon_slot: int  # 1 or 2
    source: str
    weapon_type: str
    weapon_cfg: dict[str, Any]
    ship: ShipSnapshot
    analog: float = 1.0
    weapon_origin: Optional[Vec3] = None
    weapon_dir: Optional[Vec3] = None


@dataclass(frozen=True)
class WeaponConsequence:
    t_emit: float
    kind: str  # "heat" | "kickback" | "hit" | "info"
    request_id: int
    weapon_slot: int
    payload: dict[str, Any]


class ProjectileSimulator:
    """Prototype simulator thread.

    For now it does not do projectile integration. It simply turns fire requests
    into a small set of consequences (heat/kickback stubs) so the rest of the
    program can be wired around queues.
    """

    def __init__(self) -> None:
        self._req_q: "queue.Queue[WeaponFireRequest]" = queue.Queue()
        self._evt_q: "queue.Queue[WeaponConsequence]" = queue.Queue()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name="ProjectileSimulator", daemon=True)

        # Optional C-DLL backend (prints requests + writes placeholder outputs).
        self._c_lib = None
        self._c_mod = None
        try:
            from c_physics import weapon_sim_ctypes as _wsc

            self._c_mod = _wsc
            self._c_lib = _wsc.load_lib()
        except Exception:
            self._c_lib = None
            self._c_mod = None

        # When true, request batches ask the C simulator to print details.
        self.debug_print: bool = False

        # Max spline points per request (C output buffer cap).
        # Exposed via the in-game BALLISTICS menu.
        self.max_points: int = 16

    def start(self) -> None:
        if not self._thr.is_alive():
            self._thr.start()

    def stop(self, timeout_s: float = 0.5) -> None:
        self._stop.set()
        try:
            self._thr.join(timeout=float(timeout_s))
        except Exception:
            pass

    @property
    def requests(self) -> "queue.Queue[WeaponFireRequest]":
        return self._req_q

    @property
    def events(self) -> "queue.Queue[WeaponConsequence]":
        return self._evt_q

    def _emit(self, evt: WeaponConsequence) -> None:
        try:
            self._evt_q.put_nowait(evt)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._req_q.get(timeout=0.05)
            except Exception:
                continue

            cfg = req.weapon_cfg if isinstance(req.weapon_cfg, dict) else {}

            # Always emit an acknowledgement event for debugging/telemetry.
            self._emit(
                WeaponConsequence(
                    t_emit=time.time(),
                    kind="info",
                    request_id=req.request_id,
                    weapon_slot=req.weapon_slot,
                    payload={
                        "event": "fired",
                        "weapon_type": req.weapon_type,
                        "source": req.source,
                        "analog": req.analog,
                    },
                )
            )

            # Optional: push the request through the C DLL so we can validate
            # the packed-buffer ABI now (DLL prints details and writes outputs).
            if self._c_lib is not None and self._c_mod is not None:
                try:
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

                    k_mode = _mode_enum(kin.get("mode") if kin else "")
                    inherit = bool(kin.get("inherit_ship_velocity")) if kin else False
                    add_v = kin.get("add_velocity") if kin else None
                    sim_points = sim.get("sim_points") if sim else None
                    t_end = sim.get("t_end") if sim else None
                    beam_len = sim.get("beam_len") if sim else None
                    drop_off = sim.get("drop_off") if sim else None

                    # Enforce: weapon behavior params must come from stats/config.
                    # If they're missing, skip the DLL call but keep running the rest
                    # of the prototype consequence pipeline (heat/kickback stubs).
                    can_call_dll = True
                    if k_mode < 0 or sim is None or kin is None:
                        can_call_dll = False
                    if add_v is None:
                        # For beam weapons, add_velocity can be null; C ignores it.
                        add_v = 0.0
                    if sim_points is None or t_end is None or beam_len is None or drop_off is None:
                        can_call_dll = False

                    if can_call_dll:
                        batch = self._c_mod.make_batch(
                            requests=[
                                {
                                    "request_id": int(req.request_id),
                                    "weapon_slot": int(req.weapon_slot),
                                    "analog": float(req.analog),
                                    "source": str(req.source),
                                    "weapon_type": str(req.weapon_type),

                                    # Debug / instrumentation
                                    "debug_print": bool(self.debug_print),

                                    # Weapon behavior (purely from stats/config)
                                    "kinematics_mode": int(k_mode),
                                    "inherit_ship_velocity": bool(inherit),
                                    "add_velocity": float(add_v),
                                    "sim_points": int(sim_points),
                                    "sim_t_end": float(t_end),
                                    "sim_beam_len": float(beam_len),
                                    "sim_drop_off": float(drop_off),

                                    "ship_pos": req.ship.pos,
                                    "ship_vel": req.ship.vel,
                                    "ship_fwd": req.ship.fwd,

                                    # Optional weapon frame: keep ship_fwd intact.
                                    "weapon_origin": req.weapon_origin,
                                    "weapon_dir": req.weapon_dir,
                                    "planet_surface_r": getattr(req.ship, "planet_surface_r", None),
                                    "gravity_g": getattr(req.ship, "gravity_g", None),
                                    "terrain_heightmap": getattr(req.ship, "terrain_heightmap", None),
                                    "nodes": getattr(req.ship, "nodes", None),
                                }
                            ],
                            max_points=int(max(2, min(256, int(getattr(self, "max_points", 16) or 16)))),
                        )
                        ok = bool(self._c_mod.process_batch(lib=self._c_lib, batch=batch))
                        if ok:
                            out0 = batch.out[0]
                            ip = out0.impact_point
                            # Only publish splines that were explicitly produced by the C simulator.
                            # (Avoid drawing uninitialized/placeholder output buffers.)
                            c_ok = int(getattr(out0, "ok", 0) or 0)
                            n_pts = int(max(0, min(int(getattr(out0, "spline_n", 0) or 0), 16)))
                            pts: List[Vec3] = []
                            if c_ok != 0 and n_pts >= 2:
                                try:
                                    for j in range(n_pts):
                                        p = out0.spline_points[j]
                                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                                        if not (x == x and y == y and z == z):
                                            continue
                                        pts.append((x, y, z))
                                except Exception:
                                    pts = []
                            self._emit(
                                WeaponConsequence(
                                    t_emit=time.time(),
                                    kind="info",
                                    request_id=req.request_id,
                                    weapon_slot=req.weapon_slot,
                                    payload={
                                        "event": "dll_processed",
                                        "from_c": True,
                                        "ok": c_ok,
                                        "impact_valid": int(out0.impact_valid),
                                        "impact_point": (float(ip[0]), float(ip[1]), float(ip[2])),
                                        "spline_n": int(out0.spline_n),
                                        "spline_points": pts,
                                        "victim_id": int(out0.victim_id),
                                    },
                                )
                            )

                            # Promote C-reported node strikes into a first-class gameplay event.
                            try:
                                impact_valid = int(getattr(out0, "impact_valid", 0) or 0)
                                victim_id_1b = int(getattr(out0, "victim_id", 0) or 0)
                                if c_ok != 0 and impact_valid != 0 and victim_id_1b > 0:
                                    self._emit(
                                        WeaponConsequence(
                                            t_emit=time.time(),
                                            kind="hit",
                                            request_id=req.request_id,
                                            weapon_slot=req.weapon_slot,
                                            payload={
                                                "from_c": True,
                                                "victim_id": victim_id_1b,
                                                "victim_index": victim_id_1b - 1,
                                                "impact_point": (float(ip[0]), float(ip[1]), float(ip[2])),
                                                "weapon_type": req.weapon_type,
                                                "source": req.source,
                                            },
                                        )
                                    )
                            except Exception:
                                pass
                except Exception:
                    pass

            # Heat stub: if config has a heat block, treat this as a single-shot
            # heat addition. Later, fire_modes/trigger_behavior will decide whether
            # to use heat_per_shot vs heat_per_sec.
            heat = cfg.get("heat")
            if isinstance(heat, dict):
                hps = heat.get("heat_per_shot")
                try:
                    delta = float(hps) if hps is not None else 0.0
                except Exception:
                    delta = 0.0
                if abs(delta) > 0.0:
                    self._emit(
                        WeaponConsequence(
                            t_emit=time.time(),
                            kind="heat",
                            request_id=req.request_id,
                            weapon_slot=req.weapon_slot,
                            payload={"delta": delta, "weapon_type": req.weapon_type, "source": req.source},
                        )
                    )

            # Kickback stub: opposing forward direction if present.
            kin = cfg.get("kinematics")
            if isinstance(kin, dict):
                try:
                    add_v = kin.get("add_velocity")
                    kick_mag = 0.02 * float(add_v) if add_v is not None else 0.0
                except Exception:
                    kick_mag = 0.0
                fwd = req.ship.fwd
                if kick_mag > 0.0 and isinstance(fwd, tuple) and len(fwd) == 3:
                    fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
                    self._emit(
                        WeaponConsequence(
                            t_emit=time.time(),
                            kind="kickback",
                            request_id=req.request_id,
                            weapon_slot=req.weapon_slot,
                            payload={"dv": (-kick_mag * fx, -kick_mag * fy, -kick_mag * fz), "source": req.source},
                        )
                    )

            # Hit confirmation is not simulated yet. The queue pathway exists
            # and will be used once we implement projectile integration.


@dataclass
class WeaponRuntimeState:
    heat: float = 0.0
    last_kickback_dv: Optional[Vec3] = None
    last_hit: Optional[dict[str, Any]] = None


class WeaponRuntime:
    """Owns weapon stats + fire dispatch + simulator thread queues."""

    def __init__(self, *, stats_path: str = "weapon_stats.json") -> None:
        self.stats_path = str(stats_path)
        self.stats = load_weapon_stats(self.stats_path)
        self.sim = ProjectileSimulator()
        self.sim.start()
        self.state = WeaponRuntimeState()

        # Controls C-side request logging (when the DLL backend is enabled).
        self.debug_print: bool = False

        self._req_id = 1

    def stop(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass

    def refresh_stats(self) -> None:
        self.stats = load_weapon_stats(self.stats_path)

    def enqueue_fire(
        self,
        *,
        loadout: Any,
        weapon_slot: int,
        resolved: Iterable[tuple[str, str]],
        ship: Optional[ShipSnapshot] = None,
        analog: float = 1.0,
        weapon_origin: Optional[Vec3] = None,
        weapon_dir: Optional[Vec3] = None,
    ) -> List[int]:
        """Enqueue fire requests.

        `resolved` is an iterable of (source_name, weapon_type).
        """
        ids: List[int] = []
        try:
            self.sim.debug_print = bool(self.debug_print)
        except Exception:
            pass
        ship_s = ship if ship is not None else ShipSnapshot()
        for source_name, weapon_type in resolved:
            cfg = get_weapon_type_stats(self.stats, str(weapon_type))
            rid = int(self._req_id)
            self._req_id += 1
            req = WeaponFireRequest(
                request_id=rid,
                t_submit=time.time(),
                weapon_slot=int(weapon_slot),
                source=str(source_name),
                weapon_type=str(weapon_type),
                weapon_cfg=dict(cfg) if isinstance(cfg, dict) else {},
                ship=ship_s,
                analog=float(analog),
                weapon_origin=weapon_origin,
                weapon_dir=weapon_dir,
            )
            try:
                self.sim.requests.put_nowait(req)
                ids.append(rid)
            except Exception:
                pass
        return ids

    def poll_consequences(self, *, max_events: int = 64) -> List[WeaponConsequence]:
        evts: List[WeaponConsequence] = []
        for _ in range(max(0, int(max_events))):
            try:
                evt = self.sim.events.get_nowait()
            except Exception:
                break
            evts.append(evt)
            if evt.kind == "heat":
                try:
                    self.state.heat += float(evt.payload.get("delta", 0.0))
                except Exception:
                    pass
            elif evt.kind == "kickback":
                dv = evt.payload.get("dv")
                if isinstance(dv, tuple) and len(dv) == 3:
                    try:
                        self.state.last_kickback_dv = (float(dv[0]), float(dv[1]), float(dv[2]))
                    except Exception:
                        pass
            elif evt.kind == "hit":
                if isinstance(evt.payload, dict):
                    self.state.last_hit = dict(evt.payload)
        return evts
