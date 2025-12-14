from __future__ import annotations

import ctypes
import json
from pathlib import Path


class WeaponLoadout(ctypes.Structure):
    _fields_ = [
        # Hardpoints / stores
        ("nose_gun", ctypes.c_int),
        ("left_hp1", ctypes.c_int),
        ("left_hp2", ctypes.c_int),
        ("right_hp1", ctypes.c_int),
        ("right_hp2", ctypes.c_int),
        ("bay_outer", ctypes.c_int),
        ("bay_inner", ctypes.c_int),

        # Guidance / aircraft systems
        ("targeting", ctypes.c_int),
        ("thrust", ctypes.c_int),

        # Virtual trigger mapping
        ("sourcing_mode", ctypes.c_int),
        ("weapon1_source", ctypes.c_int),
        ("weapon2_source", ctypes.c_int),
        ("weapon1_type", ctypes.c_int),
        ("weapon2_type", ctypes.c_int),
        ("type_fire_policy", ctypes.c_int),
    ]


NOSE_GUN_LOADOUTS: list[str] = [
    "machine gun",
    "laser",
    "camera",
]

# General ordnance list for wing hardpoints.
HARDPOINT_LOADOUTS: list[str] = [
    "none",
    "bomb",
    "laser",
    "rocket",
    "machine gun",
    "missile",
]

# Bay inner: droppable/aimable and accepts laser.
BAY_INNER_LOADOUTS: list[str] = [
    "none",
    "bomb",
    "laser",
    "camera",
    "glider",
    "missile",
]

# Bay outer: full list + "ultra heavy" category.
# If/when you decide specific ultra-heavy items, extend this list.
BAY_OUTER_LOADOUTS: list[str] = [
    "none",
    "bomb",
    "laser",
    "rocket",
    "machine gun",
    "missile",
    "camera",
    "glider",
    "ultra heavy",
]

TARGETING_MODES: list[str] = [
    "manual",
    "auto",
    "auto-fire",
]

THRUST_MODES: list[str] = [
    "main engine",
]

SOURCING_MODES: list[str] = [
    "pin to hardpoint",
    "by ordnance type",
]

WEAPON_SOURCES: list[str] = [
    "nose gun",
    "left wing hp1",
    "left wing hp2",
    "right wing hp1",
    "right wing hp2",
    "bay outer",
    "bay inner",
]

# For type-based firing, pick a type and choose how it fans out.
ORDNANCE_TYPES: list[str] = [
    "bomb",
    "laser",
    "rocket",
    "machine gun",
    "missile",
    "camera",
    "glider",
    "ultra heavy",
]

TYPE_FIRE_POLICIES: list[str] = [
    "any source",
    "all sources of type",
    "all weapons",
]


def weapon_loadout_default() -> WeaponLoadout:
    w = WeaponLoadout()
    # Defaults are intentionally conservative: a nose gun and one basic wing store.
    w.nose_gun = 1  # laser
    w.left_hp1 = 2  # laser
    w.left_hp2 = 0  # none
    w.right_hp1 = 5  # missile
    w.right_hp2 = 0  # none
    w.bay_outer = 0  # none
    w.bay_inner = 0  # none

    w.targeting = 0  # manual
    w.thrust = 0  # main engine

    # Weapon 1/2 default: pin to the primary wing hardpoints.
    w.sourcing_mode = 0  # pin
    w.weapon1_source = 1  # left wing hp1
    w.weapon2_source = 3  # right wing hp1
    w.weapon1_type = 0  # bomb
    w.weapon2_type = 0  # bomb
    w.type_fire_policy = 0  # any source
    return w


def weapon_loadout_field_specs() -> dict[str, dict]:
    # Use choices so the universal editor renders labels and cycles values.
    return {
        "nose_gun": {"label": "Nose Gun", "choices": NOSE_GUN_LOADOUTS, "wrap": True},
        "left_hp1": {"label": "Left Wing HP1", "choices": HARDPOINT_LOADOUTS, "wrap": True},
        "left_hp2": {"label": "Left Wing HP2", "choices": HARDPOINT_LOADOUTS, "wrap": True},
        "right_hp1": {"label": "Right Wing HP1", "choices": HARDPOINT_LOADOUTS, "wrap": True},
        "right_hp2": {"label": "Right Wing HP2", "choices": HARDPOINT_LOADOUTS, "wrap": True},
        "bay_outer": {"label": "Bay Outer", "choices": BAY_OUTER_LOADOUTS, "wrap": True},
        "bay_inner": {"label": "Bay Inner (requires outer deployed)", "choices": BAY_INNER_LOADOUTS, "wrap": True},
        "targeting": {"label": "Targeting", "choices": TARGETING_MODES, "wrap": True},
        "thrust": {"label": "Thrust", "choices": THRUST_MODES, "wrap": True},

        "sourcing_mode": {"label": "Ordnance Sourcing", "choices": SOURCING_MODES, "wrap": True},
        "weapon1_source": {"label": "Weapon 1 Source", "choices": WEAPON_SOURCES, "wrap": True},
        "weapon2_source": {"label": "Weapon 2 Source", "choices": WEAPON_SOURCES, "wrap": True},
        "weapon1_type": {"label": "Weapon 1 Type", "choices": ORDNANCE_TYPES, "wrap": True},
        "weapon2_type": {"label": "Weapon 2 Type", "choices": ORDNANCE_TYPES, "wrap": True},
        "type_fire_policy": {"label": "Type Fire Policy", "choices": TYPE_FIRE_POLICIES, "wrap": True},
    }


def load_weapon_loadout_json(path: str | Path) -> WeaponLoadout:
    p = Path(path)
    w = weapon_loadout_default()
    if not p.exists():
        return w
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return w
    if isinstance(data, dict):
        # Accept either a flat object {weapon_a, weapon_b} or the persisted-block
        # format {"weapon_loadout": {...}}.
        block = data
        if isinstance(data.get("weapon_loadout"), dict):
            block = data.get("weapon_loadout")  # type: ignore[assignment]
        # Backward compatibility:
        # - Older versions stored only weapon_a/weapon_b.
        #   We interpret those as Left HP1 / Right HP1 loadouts.
        old_wa = None
        old_wb = None
        try:
            if "weapon_a" in block:
                old_wa = int(block["weapon_a"])
        except Exception:
            old_wa = None
        try:
            if "weapon_b" in block:
                old_wb = int(block["weapon_b"])
        except Exception:
            old_wb = None

        if old_wa is not None:
            try:
                w.left_hp1 = int(old_wa)
            except Exception:
                pass
        if old_wb is not None:
            try:
                w.right_hp1 = int(old_wb)
            except Exception:
                pass

        # New fields
        for k in (
            "nose_gun",
            "left_hp1",
            "left_hp2",
            "right_hp1",
            "right_hp2",
            "bay_outer",
            "bay_inner",
            "targeting",
            "thrust",
            "sourcing_mode",
            "weapon1_source",
            "weapon2_source",
            "weapon1_type",
            "weapon2_type",
            "type_fire_policy",
        ):
            try:
                if k in block:
                    setattr(w, k, int(block[k]))
            except Exception:
                pass

        # If we loaded an old config, also default weapon triggers to those primary hardpoints.
        if (old_wa is not None) or (old_wb is not None):
            try:
                w.sourcing_mode = 0
                w.weapon1_source = 1  # left wing hp1
                w.weapon2_source = 3  # right wing hp1
            except Exception:
                pass
    return w


def save_weapon_loadout_json(path: str | Path, w: WeaponLoadout) -> None:
    p = Path(path)
    payload = {
        "nose_gun": int(getattr(w, "nose_gun", 0)),
        "left_hp1": int(getattr(w, "left_hp1", 0)),
        "left_hp2": int(getattr(w, "left_hp2", 0)),
        "right_hp1": int(getattr(w, "right_hp1", 0)),
        "right_hp2": int(getattr(w, "right_hp2", 0)),
        "bay_outer": int(getattr(w, "bay_outer", 0)),
        "bay_inner": int(getattr(w, "bay_inner", 0)),
        "targeting": int(getattr(w, "targeting", 0)),
        "thrust": int(getattr(w, "thrust", 0)),
        "sourcing_mode": int(getattr(w, "sourcing_mode", 0)),
        "weapon1_source": int(getattr(w, "weapon1_source", 0)),
        "weapon2_source": int(getattr(w, "weapon2_source", 0)),
        "weapon1_type": int(getattr(w, "weapon1_type", 0)),
        "weapon2_type": int(getattr(w, "weapon2_type", 0)),
        "type_fire_policy": int(getattr(w, "type_fire_policy", 0)),
    }
    try:
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        return


def _safe_choice(choices: list[str], idx: int) -> str | None:
    try:
        i = int(idx)
    except Exception:
        return None
    if i < 0 or i >= len(choices):
        return None
    return str(choices[i])


def hardpoint_weapon_type(loadout: WeaponLoadout, source_idx: int) -> str | None:
    """Return the configured weapon type string for a physical source.

    Returns None if out-of-range or unset.
    Returns "none" for hardpoints that explicitly support none.
    """
    try:
        si = int(source_idx)
    except Exception:
        return None

    if si == 0:
        return _safe_choice(NOSE_GUN_LOADOUTS, int(getattr(loadout, "nose_gun", 0)))
    if si == 1:
        return _safe_choice(HARDPOINT_LOADOUTS, int(getattr(loadout, "left_hp1", 0)))
    if si == 2:
        return _safe_choice(HARDPOINT_LOADOUTS, int(getattr(loadout, "left_hp2", 0)))
    if si == 3:
        return _safe_choice(HARDPOINT_LOADOUTS, int(getattr(loadout, "right_hp1", 0)))
    if si == 4:
        return _safe_choice(HARDPOINT_LOADOUTS, int(getattr(loadout, "right_hp2", 0)))
    if si == 5:
        return _safe_choice(BAY_OUTER_LOADOUTS, int(getattr(loadout, "bay_outer", 0)))
    if si == 6:
        return _safe_choice(BAY_INNER_LOADOUTS, int(getattr(loadout, "bay_inner", 0)))
    return None


def resolve_virtual_weapon(loadout: WeaponLoadout, weapon_slot: int) -> list[tuple[str, str]]:
    """Resolve Weapon 1/2 into concrete fire requests.

    Returns a list of (source_name, weapon_type).
    """
    slot = 1 if int(weapon_slot) == 1 else 2
    mode = int(getattr(loadout, "sourcing_mode", 0))

    # Pin-to-hardpoint.
    if mode == 0:
        src_idx = int(getattr(loadout, "weapon1_source" if slot == 1 else "weapon2_source", 0))
        if src_idx < 0 or src_idx >= len(WEAPON_SOURCES):
            return []
        wtype = hardpoint_weapon_type(loadout, int(src_idx))
        if not wtype or str(wtype) == "none":
            return []
        return [(str(WEAPON_SOURCES[int(src_idx)]), str(wtype))]

    # By ordnance type.
    desired_idx = int(getattr(loadout, "weapon1_type" if slot == 1 else "weapon2_type", 0))
    desired = _safe_choice(ORDNANCE_TYPES, int(desired_idx))
    policy = int(getattr(loadout, "type_fire_policy", 0))

    resolved: list[tuple[str, str]] = []
    if policy == 2:
        # All weapons: ignore desired, include all non-none sources.
        for i, src_name in enumerate(WEAPON_SOURCES):
            wtype = hardpoint_weapon_type(loadout, int(i))
            if not wtype or str(wtype) == "none":
                continue
            resolved.append((str(src_name), str(wtype)))
        return resolved

    if not desired:
        return []

    matches: list[tuple[str, str]] = []
    for i, src_name in enumerate(WEAPON_SOURCES):
        wtype = hardpoint_weapon_type(loadout, int(i))
        if not wtype or str(wtype) == "none":
            continue
        if str(wtype) == str(desired):
            matches.append((str(src_name), str(wtype)))

    if policy == 0:
        return matches[:1]
    return matches
