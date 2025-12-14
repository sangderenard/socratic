from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Dict


class AirplaneTuning(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # 0=manual, 1=auto
        ("control_mode", ctypes.c_uint32),
        ("flight_debug_print", ctypes.c_uint32),
        ("weapon_debug_print", ctypes.c_uint32),

        # 0=off, 1=on (legacy behavior), 2=lock_yellow, 3=lock_red
        ("los_probe_mode", ctypes.c_uint32),

        ("auto_kp", ctypes.c_float),
        ("auto_kd", ctypes.c_float),

        # Optional: apply throttle/strafe torque about a configured center.
        ("use_thrust_center_arm", ctypes.c_uint32),
        ("use_drag_center_arm", ctypes.c_uint32),

        # Body-frame offsets (ship local) for translation-force application.
        ("thrust_center_x", ctypes.c_float),
        ("thrust_center_y", ctypes.c_float),
        ("thrust_center_z", ctypes.c_float),

        ("drag_center_x", ctypes.c_float),
        ("drag_center_y", ctypes.c_float),
        ("drag_center_z", ctypes.c_float),

        # Drag-center coefficient (scaled by rho*v^2) for arm type=drag_center.
        ("drag_center_k", ctypes.c_float),

        # Generated attitude thruster arms (optional): matches the default arms layout.
        ("use_generated_attitude_arms", ctypes.c_uint32),
        ("att_arm_x", ctypes.c_float),
        ("att_arm_z", ctypes.c_float),
        ("att_max_force", ctypes.c_float),
    ]


def airplane_tuning_default() -> AirplaneTuning:
    t = AirplaneTuning()
    t.control_mode = 1
    t.flight_debug_print = 0
    t.weapon_debug_print = 0
    t.los_probe_mode = 2  # default: lock_yellow (reduce probe spam)
    t.auto_kp = 6.0
    t.auto_kd = 2.5

    t.use_thrust_center_arm = 0
    t.use_drag_center_arm = 0

    t.thrust_center_x = 0.0
    t.thrust_center_y = 0.0
    t.thrust_center_z = -2.0

    t.drag_center_x = 0.0
    t.drag_center_y = 0.0
    t.drag_center_z = 1.0
    t.drag_center_k = 0.0

    t.use_generated_attitude_arms = 0
    t.att_arm_x = 3.0
    t.att_arm_z = 3.0
    t.att_max_force = 0.02
    return t


def airplane_tuning_field_specs() -> dict[str, dict[str, Any]]:
    return {
        "control_mode": {"choices": [(0, "manual"), (1, "auto")]},
        "flight_debug_print": {"choices": [(0, "off"), (1, "on")]},
        "weapon_debug_print": {"choices": [(0, "off"), (1, "on")]},
        "los_probe_mode": {
            "choices": [
                (0, "off"),
                (1, "on"),
                (2, "lock_yellow"),
                (3, "lock_red"),
            ]
        },
        "auto_kp": {"step": 0.25, "min": 0.0},
        "auto_kd": {"step": 0.25, "min": 0.0},
        "use_thrust_center_arm": {"choices": [(0, "off"), (1, "on")]},
        "use_drag_center_arm": {"choices": [(0, "off"), (1, "on")]},
        "thrust_center_x": {"step": 0.25},
        "thrust_center_y": {"step": 0.25},
        "thrust_center_z": {"step": 0.25},
        "drag_center_x": {"step": 0.25},
        "drag_center_y": {"step": 0.25},
        "drag_center_z": {"step": 0.25},
        "drag_center_k": {"step": 0.01, "min": 0.0},
        "use_generated_attitude_arms": {"choices": [(0, "off"), (1, "on")]},
        "att_arm_x": {"step": 0.25, "min": 0.0},
        "att_arm_z": {"step": 0.25, "min": 0.0},
        "att_max_force": {"step": 0.002, "min": 0.0},
    }


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, root: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(root, f, indent=2)
            f.write("\n")
    except Exception:
        pass


def load_airplane_tuning_json(path: str = "airplane.json", *, key: str = "airplane_tuning") -> AirplaneTuning:
    t = airplane_tuning_default()
    root = _read_json(str(path))
    block = root.get(str(key))
    if isinstance(block, dict):
        # Best-effort mapping from dict into struct.
        for field_name, field_ctype in t._fields_:  # type: ignore[attr-defined]
            if str(field_name) not in block:
                continue
            val = block.get(str(field_name))
            try:
                if field_ctype in (ctypes.c_float, ctypes.c_double):
                    setattr(t, field_name, float(val))
                else:
                    setattr(t, field_name, int(val))
            except Exception:
                pass
    else:
        # Back-compat: populate from existing airplane.json top-level control_system if present.
        cs = root.get("control_system") if isinstance(root.get("control_system"), dict) else None
        if isinstance(cs, dict):
            mode = cs.get("mode")
            if isinstance(mode, str):
                t.control_mode = 1 if mode == "auto" else 0
            try:
                if isinstance(cs.get("debug_print"), (bool, int, float)):
                    t.flight_debug_print = 1 if bool(cs.get("debug_print")) else 0
            except Exception:
                pass
            try:
                if isinstance(cs.get("auto_kp"), (int, float)):
                    t.auto_kp = float(cs.get("auto_kp"))
                if isinstance(cs.get("auto_kd"), (int, float)):
                    t.auto_kd = float(cs.get("auto_kd"))
            except Exception:
                pass

    return t


def save_airplane_tuning_json(
    tuning: AirplaneTuning,
    path: str = "airplane.json",
    *,
    key: str = "airplane_tuning",
) -> None:
    root = _read_json(str(path))

    block: Dict[str, Any] = {}
    for field_name, field_ctype in tuning._fields_:  # type: ignore[attr-defined]
        try:
            v = getattr(tuning, field_name)
        except Exception:
            continue
        if field_ctype in (ctypes.c_float, ctypes.c_double):
            try:
                block[str(field_name)] = float(v)
            except Exception:
                pass
        else:
            try:
                block[str(field_name)] = int(v)
            except Exception:
                pass

    root[str(key)] = block

    # Also mirror into the existing control_system block so older code paths stay coherent.
    cs = root.get("control_system") if isinstance(root.get("control_system"), dict) else {}
    if not isinstance(cs, dict):
        cs = {}
    cs["mode"] = "auto" if int(block.get("control_mode", 1)) != 0 else "manual"
    cs["auto_kp"] = float(block.get("auto_kp", 0.0))
    cs["auto_kd"] = float(block.get("auto_kd", 0.0))
    cs["debug_print"] = bool(int(block.get("flight_debug_print", 0)) != 0)
    root["control_system"] = cs

    _write_json(str(path), root)


def build_extra_arms_from_tuning(tuning: AirplaneTuning) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []

    if int(getattr(tuning, "use_generated_attitude_arms", 0) or 0) != 0:
        ax = float(getattr(tuning, "att_arm_x", 0.0) or 0.0)
        az = float(getattr(tuning, "att_arm_z", 0.0) or 0.0)
        mf = float(getattr(tuning, "att_max_force", 0.0) or 0.0)

        # Pitch (2): z - / z +
        arms.append({"type": "thruster", "input_idx": 2, "pos_b": [0, 0, -az], "dir_b": [0, 1, 0], "max_force": mf})
        arms.append({"type": "thruster", "input_idx": 2, "pos_b": [0, 0, az], "dir_b": [0, -1, 0], "max_force": mf})
        # Yaw (3): z - / z +
        arms.append({"type": "thruster", "input_idx": 3, "pos_b": [0, 0, -az], "dir_b": [-1, 0, 0], "max_force": mf})
        arms.append({"type": "thruster", "input_idx": 3, "pos_b": [0, 0, az], "dir_b": [1, 0, 0], "max_force": mf})
        # Roll (4): x + / x -
        arms.append({"type": "thruster", "input_idx": 4, "pos_b": [ax, 0, 0], "dir_b": [0, 1, 0], "max_force": mf})
        arms.append({"type": "thruster", "input_idx": 4, "pos_b": [-ax, 0, 0], "dir_b": [0, -1, 0], "max_force": mf})

    if int(getattr(tuning, "use_thrust_center_arm", 0) or 0) != 0:
        arms.append(
            {
                "type": "thruster",
                "input_idx": 0,
                "flags": 1,  # interpreted by C: use throttle as input
                "pos_b": [
                    float(getattr(tuning, "thrust_center_x", 0.0) or 0.0),
                    float(getattr(tuning, "thrust_center_y", 0.0) or 0.0),
                    float(getattr(tuning, "thrust_center_z", 0.0) or 0.0),
                ],
                "dir_b": [0, 0, 1],
                "max_force": 1.0,
            }
        )

    if int(getattr(tuning, "use_drag_center_arm", 0) or 0) != 0:
        arms.append(
            {
                "type": "drag_center",
                "input_idx": 0,
                "flags": 0,
                "pos_b": [
                    float(getattr(tuning, "drag_center_x", 0.0) or 0.0),
                    float(getattr(tuning, "drag_center_y", 0.0) or 0.0),
                    float(getattr(tuning, "drag_center_z", 0.0) or 0.0),
                ],
                "k_drag": float(getattr(tuning, "drag_center_k", 0.0) or 0.0),
            }
        )

    return arms
