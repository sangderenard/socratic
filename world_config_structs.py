from __future__ import annotations

import ctypes
from typing import Any


class WorldConfig(ctypes.Structure):
    """Small persistent world/config struct.

    This is intentionally ctypes-based so we can:
    - introspect fields via _fields_
    - edit it generically in-game
    - mirror values into the existing params dicts

    It is not tied to the C physics header ABI.
    """

    _pack_ = 1
    _fields_ = [
        ("k_spring", ctypes.c_double),
        ("k_coulomb", ctypes.c_double),
        ("G", ctypes.c_double),
        ("damping", ctypes.c_double),
        ("temp", ctypes.c_double),
        ("bond_shear", ctypes.c_double),
        ("max_speed", ctypes.c_double),
        ("render_scale", ctypes.c_double),
    ]


def world_config_from_params(params: dict[str, Any]) -> WorldConfig:
    cfg = WorldConfig()
    cfg.k_spring = float(params.get("k_spring", 1.0))
    cfg.k_coulomb = float(params.get("k_coulomb", 1.0))
    cfg.G = float(params.get("G", 0.0))
    cfg.damping = float(params.get("damping", 0.0))
    cfg.temp = float(params.get("temp", 0.0))
    cfg.bond_shear = float(params.get("bond_shear", 1.0))
    cfg.max_speed = float(params.get("max_speed", 0.02))
    cfg.render_scale = float(params.get("render_scale", 10.0))
    return cfg


def apply_world_config_to_params(cfg: WorldConfig, params: dict[str, Any]) -> None:
    params["k_spring"] = float(cfg.k_spring)
    params["k_coulomb"] = float(cfg.k_coulomb)
    params["G"] = float(cfg.G)
    params["damping"] = float(cfg.damping)
    params["temp"] = float(cfg.temp)
    params["bond_shear"] = float(cfg.bond_shear)
    params["max_speed"] = float(cfg.max_speed)
    params["render_scale"] = float(cfg.render_scale)


def world_config_update_from_dict(cfg: WorldConfig, data: dict[str, Any]) -> None:
    for name, _ctype in getattr(WorldConfig, "_fields_", []):
        if name in data:
            try:
                setattr(cfg, name, float(data[name]))
            except Exception:
                pass


def world_config_to_dict(cfg: WorldConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, _ctype in getattr(WorldConfig, "_fields_", []):
        try:
            out[name] = float(getattr(cfg, name))
        except Exception:
            pass
    return out
