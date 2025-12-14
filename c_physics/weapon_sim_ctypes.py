from __future__ import annotations

import ctypes
from ctypes import c_uint32, c_float, c_char, c_size_t

from . import geodesic_ctypes


GP_WPN_MAGIC = 0x514E5057  # 'WPNQ'


class GP_WpnHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("count", c_uint32),
        ("max_points", c_uint32),
    ]


class GP_WpnRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("request_id", c_uint32),
        ("weapon_slot", c_uint32),
        ("analog", c_float),
        ("source", c_char * 32),
        ("weapon_type", c_char * 32),

        # Weapon kinematics + sim parameters (MUST be provided by config)
        ("kinematics_mode", c_uint32),
        ("inherit_ship_velocity", c_uint32),
        ("add_velocity", c_float),
        ("sim_points", c_uint32),
        ("sim_t_end", c_float),
        ("sim_beam_len", c_float),
        ("sim_drop_off", c_float),
        ("ship_pos", c_float * 3),
        ("ship_vel", c_float * 3),
        ("ship_fwd", c_float * 3),

        # Optional weapon frame override (keeps ship_fwd intact):
        # - bit0: weapon_origin/weapon_dir are valid and should be used.
        ("weapon_flags", c_uint32),
        ("weapon_origin", c_float * 3),
        ("weapon_dir", c_float * 3),

        # World snapshot (optional)
        ("world_flags", c_uint32),
        ("planet_surface_r", c_float),
        ("gravity_g", c_float),

        # Terrain heightmap pointer (optional)
        ("terrain_hm_ptr", c_size_t),
        ("terrain_hm_w", c_uint32),
        ("terrain_hm_h", c_uint32),
        ("terrain_hm_stride", c_uint32),
        ("terrain_height_scale", c_float),
        ("terrain_height_bias", c_float),

        # Optional node positions pointer (future)
        ("nodes_pos_ptr", c_size_t),
        ("nodes_count", c_uint32),
        ("nodes_pos_stride", c_uint32),
        ("nodes_radius_ptr", c_size_t),
        ("nodes_radius_stride", c_uint32),
    ]


class GP_WpnResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ok", c_uint32),
        ("impact_valid", c_uint32),
        ("spline_n", c_uint32),
        ("victim_id", c_uint32),
        ("impact_point", c_float * 3),
        ("spline_points", (c_float * 3) * 16),
    ]


class GP_WpnBatch(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("hdr", GP_WpnHeader),
        ("req", GP_WpnRequest * 8),
        ("out", GP_WpnResult * 8),
    ]


def load_lib(search_dir: str | None = None) -> ctypes.CDLL:
    lib = geodesic_ctypes.load_lib(search_dir=search_dir)

    if not hasattr(lib, "gp_weapon_process_batch"):
        raise AttributeError("geodesic_physics DLL missing gp_weapon_process_batch; rebuild c_physics")

    lib.gp_weapon_process_batch.argtypes = [ctypes.POINTER(GP_WpnBatch)]
    lib.gp_weapon_process_batch.restype = ctypes.c_int
    return lib


def process_batch(*, lib: ctypes.CDLL, batch: GP_WpnBatch) -> bool:
    try:
        rc = int(lib.gp_weapon_process_batch(ctypes.byref(batch)))
    except Exception:
        return False
    return rc != 0


def _cstr32(s: str) -> bytes:
    # ctypes exposes c_char * N fields as immutable bytes on access; assignment must
    # be done via the field itself.
    return (s or "").encode("utf-8", errors="ignore")[:31]


def make_batch(
    *,
    requests: list[dict],
    max_points: int = 16,
) -> GP_WpnBatch:
    b = GP_WpnBatch()
    b.hdr.magic = GP_WPN_MAGIC
    b.hdr.version = 6
    b.hdr.count = min(8, max(0, len(requests)))
    b.hdr.max_points = int(max(0, max_points))

    for i in range(int(b.hdr.count)):
        r = requests[i] if isinstance(requests[i], dict) else {}
        req = b.req[i]
        req.request_id = int(r.get("request_id", 0))
        req.weapon_slot = int(r.get("weapon_slot", 0))
        req.analog = float(r.get("analog", 1.0))
        req.source = _cstr32(str(r.get("source", "")))
        req.weapon_type = _cstr32(str(r.get("weapon_type", "")))

        # Weapon kinematics + sim parameters.
        # IMPORTANT: Callers should populate these from weapon stats/config; this layer
        # only assigns if explicitly present.
        try:
            if "kinematics_mode" in r:
                req.kinematics_mode = int(r.get("kinematics_mode", 0))
            if "inherit_ship_velocity" in r:
                req.inherit_ship_velocity = 1 if bool(r.get("inherit_ship_velocity")) else 0
            if "add_velocity" in r:
                req.add_velocity = float(r.get("add_velocity", 0.0))
            if "sim_points" in r:
                req.sim_points = int(r.get("sim_points", 0))
            if "sim_t_end" in r:
                req.sim_t_end = float(r.get("sim_t_end", 0.0))
            if "sim_beam_len" in r:
                req.sim_beam_len = float(r.get("sim_beam_len", 0.0))
            if "sim_drop_off" in r:
                req.sim_drop_off = float(r.get("sim_drop_off", 0.0))
        except Exception:
            pass

        sp = r.get("ship_pos")
        sv = r.get("ship_vel")
        sf = r.get("ship_fwd")
        if isinstance(sp, (tuple, list)) and len(sp) == 3:
            req.ship_pos[:] = (float(sp[0]), float(sp[1]), float(sp[2]))
        if isinstance(sv, (tuple, list)) and len(sv) == 3:
            req.ship_vel[:] = (float(sv[0]), float(sv[1]), float(sv[2]))
        if isinstance(sf, (tuple, list)) and len(sf) == 3:
            req.ship_fwd[:] = (float(sf[0]), float(sf[1]), float(sf[2]))

        # Optional weapon origin/direction.
        req.weapon_flags = 0
        wo = r.get("weapon_origin")
        wd = r.get("weapon_dir")
        if isinstance(wo, (tuple, list)) and len(wo) == 3 and isinstance(wd, (tuple, list)) and len(wd) == 3:
            try:
                req.weapon_origin[:] = (float(wo[0]), float(wo[1]), float(wo[2]))
                req.weapon_dir[:] = (float(wd[0]), float(wd[1]), float(wd[2]))
                req.weapon_flags |= 1
            except Exception:
                req.weapon_flags = 0

        # Optional debug printing (C side).
        try:
            if bool(r.get("debug_print", False)):
                req.weapon_flags |= 2
        except Exception:
            pass

        # Optional world + heightmap pointers.
        req.world_flags = 0
        try:
            planet_surface_r = r.get("planet_surface_r", None)
            gravity_g = r.get("gravity_g", None)
            if planet_surface_r is not None:
                req.planet_surface_r = float(planet_surface_r)
            if gravity_g is not None:
                req.gravity_g = float(gravity_g)
        except Exception:
            pass

        hm = r.get("terrain_heightmap")
        if isinstance(hm, dict):
            try:
                ptr = int(hm.get("ptr", 0) or 0)
                w = int(hm.get("w", 0) or 0)
                h = int(hm.get("h", 0) or 0)
                stride = int(hm.get("stride", 0) or 0)
                hs = float(hm.get("height_scale", 0.0) or 0.0)
                hb = float(hm.get("height_bias", 0.5) if hm.get("height_bias", None) is not None else 0.5)
                if ptr != 0 and w > 1 and h > 1 and stride >= w and hs != 0.0:
                    req.terrain_hm_ptr = ptr
                    req.terrain_hm_w = w
                    req.terrain_hm_h = h
                    req.terrain_hm_stride = stride
                    req.terrain_height_scale = float(hs)
                    req.terrain_height_bias = float(hb)
                    req.world_flags |= 1
            except Exception:
                pass

        nodes = r.get("nodes")
        if isinstance(nodes, dict):
            try:
                pos_ptr = int(nodes.get("pos_ptr", 0) or 0)
                rad_ptr = int(nodes.get("rad_ptr", 0) or 0)
                n_cnt = int(nodes.get("count", 0) or 0)
                pos_stride = int(nodes.get("pos_stride", 3) or 3)
                rad_stride = int(nodes.get("rad_stride", 1) or 1)
                if pos_ptr != 0 and rad_ptr != 0 and n_cnt > 0 and pos_stride >= 3 and rad_stride >= 1:
                    req.nodes_pos_ptr = pos_ptr
                    req.nodes_count = n_cnt
                    req.nodes_pos_stride = pos_stride
                    req.nodes_radius_ptr = rad_ptr
                    req.nodes_radius_stride = rad_stride
                    req.world_flags |= 2
            except Exception:
                pass

    return b
