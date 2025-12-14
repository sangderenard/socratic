from __future__ import annotations

import ctypes
from typing import Any


class ParticleConfig(ctypes.Structure):
    """Particle simulator tuning (nodes/edges physics).

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
        ("south_strength", ctypes.c_double),
        ("south_enabled", ctypes.c_int),
        ("south_axis", ctypes.c_int),
    ]


class WorldEnvConfig(ctypes.Structure):
    """World/planet environment tuning.

    This is distinct from particle-sim parameters.
    Example: particle sim needs damping; world env has sea-level pressure.

    Note: `render_scale` is the planet/world scale factor used to map the unit-sphere
    simulation coordinates into larger/smaller world units. It does NOT scale ship
    geometry, arms, or thrust.
    """

    _pack_ = 1
    _fields_ = [
        ("render_scale", ctypes.c_double),
        ("sea_level_pressure_kpa", ctypes.c_double),
        ("sea_level_temp_k", ctypes.c_double),
    ]


class FlightPhysicsConfig(ctypes.Structure):
    """Flight dynamics tuning for inertial ship flight."""

    _pack_ = 1
    _fields_ = [
        ("inertial", ctypes.c_int),
        ("base_speed", ctypes.c_double),
        ("thrust_accel", ctypes.c_double),
        ("throttle_rate", ctypes.c_double),
        ("lift_k", ctypes.c_double),
        ("drag_k", ctypes.c_double),
        ("gravity_g", ctypes.c_double),
        ("max_speed", ctypes.c_double),
        # Altitude clamps in unit-sphere simulation units (NOT scaled).
        # These are applied as:
        #   flight_r_min = planet_surface_r + flight_alt_min * render_scale
        #   flight_r_max = planet_surface_r + flight_alt_max * render_scale
        ("flight_alt_min", ctypes.c_double),
        ("flight_alt_max", ctypes.c_double),
        # Atmosphere thickness in unit-sphere simulation units (NOT scaled).
        # 0 => vacuum (no lift/drag/max-speed clamp).
        ("atmosphere_alt_max", ctypes.c_double),
        ("auto_heading_on_move", ctypes.c_int),
    ]


class BallisticsConfig(ctypes.Structure):
    """Ballistics / projectile-sim tuning."""

    _pack_ = 1
    _fields_ = [
        ("max_points", ctypes.c_int),
    ]


# Backwards compatibility: the old name "WorldConfig" actually held particle-sim
# tuning (k_spring, damping, etc). Keep the symbol so existing imports continue
# to work.
WorldConfig = ParticleConfig


def particle_config_from_params(params: dict[str, Any]) -> ParticleConfig:
    cfg = ParticleConfig()
    cfg.k_spring = float(params.get("k_spring", 1.0))
    cfg.k_coulomb = float(params.get("k_coulomb", 1.0))
    cfg.G = float(params.get("G", 0.0))
    cfg.damping = float(params.get("damping", 0.0))
    cfg.temp = float(params.get("temp", 0.0))
    cfg.bond_shear = float(params.get("bond_shear", 1.0))
    cfg.max_speed = float(params.get("max_speed", 0.02))
    cfg.south_strength = float(params.get("south_strength", 0.0))
    cfg.south_axis = int(params.get("south_axis", 2))
    se = params.get("south_enabled", None)
    if se is None:
        cfg.south_enabled = int(float(cfg.south_strength) > 1e-9)
    else:
        cfg.south_enabled = int(bool(se))
    return cfg


def world_env_from_params(params: dict[str, Any]) -> WorldEnvConfig:
    cfg = WorldEnvConfig()
    cfg.render_scale = float(params.get("render_scale", 10.0))
    cfg.sea_level_pressure_kpa = float(params.get("sea_level_pressure_kpa", 101.325))
    cfg.sea_level_temp_k = float(params.get("sea_level_temp_k", 288.15))
    return cfg


def flight_physics_from_motion(motion: dict[str, Any]) -> FlightPhysicsConfig:
    cfg = FlightPhysicsConfig()
    cfg.inertial = int(bool(motion.get("inertial", False)))
    cfg.base_speed = float(motion.get("base_speed", 0.38))
    cfg.thrust_accel = float(motion.get("thrust_accel", 4.0 * float(cfg.base_speed)))
    cfg.throttle_rate = float(motion.get("throttle_rate", 1.80))
    cfg.lift_k = float(motion.get("lift_k", 0.0))
    cfg.drag_k = float(motion.get("drag_k", 0.0))
    cfg.gravity_g = float(motion.get("gravity_g", 0.0))
    cfg.max_speed = float(motion.get("max_speed", 0.0))
    # Defaults are aligned with the renderer constants in gl_animator_geodesic.py
    # (expressed in unit-sphere sim units).
    cfg.flight_alt_min = float(motion.get("flight_alt_min", 0.06))
    cfg.flight_alt_max = float(motion.get("flight_alt_max", 3.00))
    cfg.atmosphere_alt_max = float(motion.get("atmosphere_alt_max", 0.22))
    cfg.auto_heading_on_move = int(bool(motion.get("auto_heading_on_move", False)))
    return cfg


def ballistics_default() -> BallisticsConfig:
    cfg = BallisticsConfig()
    cfg.max_points = 16
    return cfg


# Back-compat wrappers (old callers).
def world_config_from_params(params: dict[str, Any]) -> WorldConfig:
    return particle_config_from_params(params)


def apply_particle_config_to_params(cfg: ParticleConfig, params: dict[str, Any]) -> None:
    params["k_spring"] = float(cfg.k_spring)
    params["k_coulomb"] = float(cfg.k_coulomb)
    params["G"] = float(cfg.G)
    params["damping"] = float(cfg.damping)
    params["temp"] = float(cfg.temp)
    params["bond_shear"] = float(cfg.bond_shear)
    params["max_speed"] = float(cfg.max_speed)
    params["south_strength"] = float(cfg.south_strength)
    params["south_axis"] = int(cfg.south_axis)
    params["south_enabled"] = bool(int(cfg.south_enabled) != 0)


def apply_world_env_to_params(cfg: WorldEnvConfig, params: dict[str, Any]) -> None:
    params["render_scale"] = float(cfg.render_scale)
    params["sea_level_pressure_kpa"] = float(cfg.sea_level_pressure_kpa)
    params["sea_level_temp_k"] = float(cfg.sea_level_temp_k)


# Back-compat wrapper (old callers).
def apply_world_config_to_params(cfg: WorldConfig, params: dict[str, Any]) -> None:
    apply_particle_config_to_params(cfg, params)


def _update_from_dict(cfg: ctypes.Structure, data: dict[str, Any]) -> None:
    for name, _ctype in getattr(cfg.__class__, "_fields_", []):
        if name in data:
            try:
                if _ctype in (ctypes.c_float, ctypes.c_double):
                    setattr(cfg, name, float(data[name]))
                else:
                    setattr(cfg, name, int(data[name]))
            except Exception:
                pass


def world_config_update_from_dict(cfg: WorldConfig, data: dict[str, Any]) -> None:
    _update_from_dict(cfg, data)


def particle_config_update_from_dict(cfg: ParticleConfig, data: dict[str, Any]) -> None:
    _update_from_dict(cfg, data)


def world_env_update_from_dict(cfg: WorldEnvConfig, data: dict[str, Any]) -> None:
    _update_from_dict(cfg, data)


def flight_physics_update_from_dict(cfg: FlightPhysicsConfig, data: dict[str, Any]) -> None:
    _update_from_dict(cfg, data)


def ballistics_update_from_dict(cfg: BallisticsConfig, data: dict[str, Any]) -> None:
    _update_from_dict(cfg, data)


def _to_dict(cfg: ctypes.Structure) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, _ctype in getattr(cfg.__class__, "_fields_", []):
        try:
            val = getattr(cfg, name)
            if _ctype in (ctypes.c_float, ctypes.c_double):
                out[name] = float(val)
            else:
                out[name] = int(val)
        except Exception:
            pass
    return out


def world_config_to_dict(cfg: WorldConfig) -> dict[str, Any]:
    return _to_dict(cfg)


def particle_config_to_dict(cfg: ParticleConfig) -> dict[str, Any]:
    return _to_dict(cfg)


def world_env_to_dict(cfg: WorldEnvConfig) -> dict[str, Any]:
    return _to_dict(cfg)


def flight_physics_to_dict(cfg: FlightPhysicsConfig) -> dict[str, Any]:
    return _to_dict(cfg)


def ballistics_to_dict(cfg: BallisticsConfig) -> dict[str, Any]:
    return _to_dict(cfg)


def split_legacy_world_config_block(block: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the legacy world_config block into (particle_block, world_env_block)."""
    particle_keys = {"k_spring", "k_coulomb", "G", "damping", "temp", "bond_shear", "max_speed", "south_strength", "south_enabled", "south_axis"}
    world_keys = {"render_scale", "sea_level_pressure_kpa", "sea_level_temp_k"}
    p: dict[str, Any] = {}
    w: dict[str, Any] = {}
    for k, v in (block.items() if isinstance(block, dict) else []):
        if k in particle_keys:
            p[k] = v
        elif k in world_keys:
            w[k] = v
    # render_scale used to live in legacy config; preserve if present.
    if "render_scale" in block and "render_scale" not in w:
        w["render_scale"] = block.get("render_scale")
    return p, w
