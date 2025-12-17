from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time
import math

import pygame

import input_graph
import controller_graph_compile
import signal_kernel_ids
import signal_status_flags
import scroll_model
import ui_tables

from c_physics import signal_kernel_api
from c_physics.signal_kernel_ctypes import GP_InputEvent, GP_SignalFrame


@dataclass
class WorkbenchState:
    # Runtime-only previous direction for signal-derived virtual hats (sid -> dir_idx or None)
    sig_hat_prev_dir: dict[str, int | None] | None = None

    # Keyboard naming mode for NEW
    naming: bool = False
    name_buf: str = ""
    name_error: str = ""

    # Combined graph compile status (signals + mixer).
    compile_ok: bool = False
    compile_msg: str = ""
    compile_last_ns: int = 0

    # UI collapse/expand state (runtime-only).
    collapsed_sigs: set[str] | None = None
    expanded_hats: set[int] | None = None
    collapsed_devices: set[str] | None = None

    # Virtual signal->button id mapping (runtime-only).
    virt_btn_ids: dict[str, int] | None = None
    virt_btn_next: int = 1

    # UI selection state.
    sel_left_idx: int = 0
    sel_signal_idx: int = 0
    sel_row_kind: str = ""
    sel_row_id: Any = 0
    sel_state: str = ""

    # If set, the next left-click binds into this arg slot.
    sel_arg_sid: str = ""
    sel_arg_idx: int = 0

    # Runtime-only per-axis calibration capture sessions.
    # key: "<device>:<axis_id>" -> {"mode":..., "start_ns":..., "red_ns":..., "total_ns":..., ...}
    calib_sessions: dict[str, dict[str, Any]] | None = None


@dataclass
class AxisStats:
    v_min: float = 0.0
    v_max: float = 0.0
    seen_min: bool = False
    seen_max: bool = False


@dataclass(frozen=True)
class HitBox:
    x0: int
    y0: int
    x1: int
    y1: int
    payload: dict[str, Any]

    def contains(self, x: int, y: int) -> bool:
        return int(self.x0) <= int(x) < int(self.x1) and int(self.y0) <= int(y) < int(self.y1)


def run_signal_workbench(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    menu_button: int | None,
    load_or_create_joystick_config,
    save_joystick_config,
    get_menu_nav,
    get_menu_scroll,
    nav_edge,
    poll_joystick_snapshot,
    draw_fullscreen_lines,
) -> None:
    """Signal Workbench (dense table).

    - Left: one row per joystick axis/button. Each row shows state via LEDs/ticks.
    - Right: signals list + NEW.
    - Click a state cell on the left, then click a signal on the right to bind.
    - NEW enters a keyboard naming prompt.

    Menu-nav cancel/menu-button/esc still exit.
    """

    try:
        import input_interest
    except Exception:
        input_interest = None

    # Workbench controls input_interest mode/allowlist; we do not poll-first
    # and then ask "should we push?". We derive a set of specs and only
    # poll/push from that set (or ALL during scan/listen).

    state = WorkbenchState()
    if state.collapsed_sigs is None:
        state.collapsed_sigs = set()
    if state.expanded_hats is None:
        state.expanded_hats = set()
    if state.virt_btn_ids is None:
        state.virt_btn_ids = {}
    if state.collapsed_devices is None:
        state.collapsed_devices = set()
    if state.calib_sessions is None:
        state.calib_sessions = {}
    clock = pygame.time.Clock()

    # Workbench defaults to scan-mode, but can live-toggle to announce.
    if not hasattr(state, "input_mode"):
        state.input_mode = "scan"  # type: ignore[attr-defined]
    if input_interest is not None:
        try:
            input_interest.set_mode(str(getattr(state, "input_mode", "scan")))
        except Exception:
            pass

    def _try_load_final_graph_kernel_inputs(path: str) -> set[tuple[int, int, int]] | None:
        try:
            import json
            import os

            if not os.path.exists(str(path)):
                return None
            with open(str(path), "r", encoding="utf-8") as f:
                doc = json.load(f)
            ins = doc.get("inputs") if isinstance(doc, dict) else None
            if not isinstance(ins, list):
                return None

            out: set[tuple[int, int, int]] = set()
            for it in ins:
                if not isinstance(it, dict):
                    continue
                dev = str(it.get("device", ""))
                kind = str(it.get("kind", ""))
                try:
                    item_id = int(it.get("id", 0))
                except Exception:
                    item_id = 0

                if dev in ("joystick", "joy"):
                    dev_code = int(signal_kernel_api.GP_DEV_JOYSTICK)
                elif dev in ("keyboard", "kbd"):
                    dev_code = int(signal_kernel_api.GP_DEV_KEYBOARD)
                elif dev in ("mouse",):
                    dev_code = int(signal_kernel_api.GP_DEV_MOUSE)
                else:
                    continue

                if kind in ("axis",):
                    kind_code = int(signal_kernel_api.GP_EV_AXIS)
                elif kind in ("button",):
                    kind_code = int(signal_kernel_api.GP_EV_BUTTON)
                elif kind in ("key",):
                    kind_code = int(signal_kernel_api.GP_EV_KEY)
                elif kind in ("mouse_motion", "motion"):
                    kind_code = int(signal_kernel_api.GP_EV_MOUSE_MOTION)
                elif kind in ("mouse_button", "mbutton"):
                    kind_code = int(signal_kernel_api.GP_EV_MOUSE_BUTTON)
                else:
                    continue

                out.add((int(dev_code), int(kind_code), int(item_id)))

            return out
        except Exception:
            return None

    # Derive announce allowlist from the compiled final graph (if present).
    # Missing/invalid graph => empty allowlist (quiet announce-mode).
    announce_interest = _try_load_final_graph_kernel_inputs("controller_graph_final.json")
    if not isinstance(announce_interest, set):
        announce_interest = set()
    if input_interest is not None:
        try:
            input_interest.set_announce_interest(announce_interest)
        except Exception:
            pass

    def _rebuild_graphs(now_ns: int) -> None:
        ok, msg, _compiled, _final = controller_graph_compile.try_build_final_graph(
            joystick_path="joystick.json",
            mixer_path="channel_mixer.json",
            compiled_out_path="controller_graph_compiled.json",
            final_out_path="controller_graph_final.json",
        )
        state.compile_ok = bool(ok)
        state.compile_msg = str(msg)
        state.compile_last_ns = int(now_ns)

        # Refresh announce-mode allowlist after a successful rebuild.
        nonlocal announce_interest
        announce_interest = _try_load_final_graph_kernel_inputs("controller_graph_final.json")
        if not isinstance(announce_interest, set):
            announce_interest = set()
        if input_interest is not None:
            try:
                input_interest.set_announce_interest(announce_interest)
            except Exception:
                pass

    left_scroll = scroll_model.ScrollModel(first_idx=0)
    right_scroll = scroll_model.ScrollModel(first_idx=0)

    axis_stats: dict[int, AxisStats] = {}

    # Disable Python signal evaluation fallback so we only see kernel-driven behavior.
    ENABLE_PY_SIGNAL_FALLBACK = False

    # Virtual-hardware threshold for turning a signal into a button.
    SIG_TO_BUTTON_EPS = 0.08

    # Trinary hat quantization threshold for converting analog 2D signals into a virtual hat.
    HAT_TRIT_EPS = 0.40

    # Reserved virtual joystick id ranges (u16) for hats + signal-derived hats.
    HARD_HAT_AXIS_BASE = 50000
    HARD_HAT_BTN_BASE = 51000
    SIG_HAT_AXIS_BASE = 52000
    SIG_HAT_BTN_BASE = 58000

    # Direction ordering (8-way): UL, U, UR, L, R, DL, D, DR
    DIRS_8: list[tuple[int, int]] = [
        (-1, 1),
        (0, 1),
        (1, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]

    def _signal_key_for_comp(sid: str, comp: str | None) -> str:
        return f"{sid}:{comp}" if comp else str(sid)

    def _split_signal_key(key: str) -> tuple[str, str | None]:
        s = str(key)
        if ":" in s:
            a, b = s.split(":", 1)
            return str(a), (str(b) if str(b) else None)
        return str(s), None

    def _virt_button_id_for_signal_key(key: str) -> int:
        # Use a small runtime-only mapping so the same signal always maps
        # to the same virtual button id during a session.
        if state.virt_btn_ids is None:
            state.virt_btn_ids = {}
        k = str(key)
        got = state.virt_btn_ids.get(k)
        if isinstance(got, int) and got > 0:
            return int(got)
        nxt = int(max(1, int(state.virt_btn_next)))
        # keep within 16-bit id space used by compose_signal_id
        if nxt >= 0xFFFE:
            nxt = 1
        state.virt_btn_ids[k] = int(nxt)
        state.virt_btn_next = int(nxt + 1)
        return int(nxt)

    def _virt_button_signal_id_for_signal_key(key: str) -> int:
        # Compose a signal_id for a virtual device button.
        # (device id is arbitrary; only needs to be stable)
        GP_DEV_VIRTUAL = 250
        return int(signal_kernel_api.compose_signal_id(int(GP_DEV_VIRTUAL), int(signal_kernel_api.GP_EV_BUTTON), int(_virt_button_id_for_signal_key(key))))

    def _dir_idx_8(x: int, y: int) -> int | None:
        try:
            xi = int(x)
            yi = int(y)
        except Exception:
            return None
        if xi == 0 and yi == 0:
            return None
        for i, (dx, dy) in enumerate(DIRS_8):
            if int(dx) == int(xi) and int(dy) == int(yi):
                return int(i)
        return None

    def _dir_label_8(i: int) -> str:
        # Must match DIRS_8 order
        labels = ["UL", "U", "UR", "L", "R", "DL", "D", "DR"]
        ii = int(max(0, min(len(labels) - 1, int(i))))
        return str(labels[ii])

    def _trit(v: float, eps: float) -> int:
        vv = float(v)
        ee = float(max(0.0, eps))
        if vv <= -ee:
            return -1
        if vv >= ee:
            return 1
        return 0

    def _sig_hat_axis_id(sid: str, comp: str) -> int:
        # Stable u16 id derived from sid.
        import zlib

        h = int(zlib.crc32(str(sid).encode("utf-8")) & 0x7FF)  # 0..2047
        base = int(SIG_HAT_AXIS_BASE + h * 2)
        return int(base + (1 if str(comp) == "y" else 0))

    def _sig_hat_btn_id(sid: str, dir_idx: int) -> int:
        import zlib

        h = int(zlib.crc32((str(sid) + ":hatdir").encode("utf-8")) & 0x2FF)  # 0..767
        return int(SIG_HAT_BTN_BASE + h * 8 + int(dir_idx))

    def _hard_hat_axis_id(hat_idx: int, comp: str) -> int:
        h = int(max(0, int(hat_idx)))
        return int(HARD_HAT_AXIS_BASE + h * 2 + (1 if str(comp) == "y" else 0))

    def _hard_hat_btn_id(hat_idx: int, dir_idx: int) -> int:
        h = int(max(0, int(hat_idx)))
        return int(HARD_HAT_BTN_BASE + h * 8 + int(dir_idx))

    # Per-signal value history (for the right-pane mini history strip)
    sig_history: dict[str, list[float]] = {}
    last_hist_ns: int = 0
    # History sampling period (ns). Use 0 to sample every paint so we don't
    # throttle waveform/flag updates and accidentally drop short pulses.
    hist_period_ns: int = 0
    hist_len: int = 40

    # Optional C signal-kernel integration (for live status flags / timers).
    _sigk_lib, _sigk = signal_kernel_api.try_load_signal_kernel()
    if _sigk is not None:
        try:
            _sigk.gp_sigk_reset()
        except Exception:
            _sigk = None

    # Gesture tracking for displayed button states (Python fallback only).
    down_since: dict[str, float] = {}
    last_down_t: dict[str, float] = {}
    dbl_pending: dict[str, bool] = {}
    toggle_state: dict[str, bool] = {}

    def now_s() -> float:
        try:
            return float(pygame.time.get_ticks()) * 0.001
        except Exception:
            return 0.0

    def update_button_machine(item_id: str, down: bool) -> tuple[float, int]:
        """Return (hold_s, flags).

        Flags are SIGF_* bits, intended to match the planned C kernel output.
        """
        t = now_s()
        if not down:
            was_down = item_id in down_since
            down_since.pop(item_id, None)
            dbl_pending[item_id] = False

            flags = 0
            if was_down:
                flags |= int(signal_status_flags.SIGF_UP_EDGE)
            if bool(toggle_state.get(item_id, False)):
                flags |= int(signal_status_flags.SIGF_TOGGLED)
            return 0.0, int(flags)

        double_edge = False
        down_edge = False
        if item_id not in down_since:
            down_edge = True
            last = float(last_down_t.get(item_id, -999.0))
            if (t - last) <= 0.30:
                double_edge = True
                dbl_pending[item_id] = True
            last_down_t[item_id] = float(t)
            down_since[item_id] = float(t)

        hold_s = float(t - float(down_since.get(item_id, t)))
        double_hold = bool(dbl_pending.get(item_id, False)) and hold_s >= 0.35
        flags = int(signal_status_flags.SIGF_DOWN)
        if bool(down_edge):
            flags |= int(signal_status_flags.SIGF_DOWN_EDGE)
        if hold_s >= 0.35:
            flags |= int(signal_status_flags.SIGF_HOLD)
        if bool(double_edge):
            flags |= int(signal_status_flags.SIGF_DOUBLE_EDGE)
        if bool(double_hold):
            flags |= int(signal_status_flags.SIGF_DOUBLE_HOLD)

        # Simple demo toggle logic for badges:
        # - Single press toggles, unless the press is a double (then double-toggle).
        if bool(down_edge):
            if bool(double_edge):
                toggle_state[item_id] = not bool(toggle_state.get(item_id, False))
                flags |= int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE)
                flags |= int(signal_status_flags.SIGF_TOGGLE_EDGE)
            else:
                toggle_state[item_id] = not bool(toggle_state.get(item_id, False))
                flags |= int(signal_status_flags.SIGF_TOGGLE_EDGE)
        if bool(toggle_state.get(item_id, False)):
            flags |= int(signal_status_flags.SIGF_TOGGLED)
        return hold_s, int(flags)

    def _mono_ns() -> int:
        try:
            return int(time.monotonic_ns())
        except Exception:
            return 0

    def _snap_to_axis_stats(axis: int, v: float) -> None:
        st = axis_stats.get(int(axis))
        if st is None:
            st = AxisStats(v_min=float(v), v_max=float(v))
            axis_stats[int(axis)] = st
        st.v_min = float(min(st.v_min, float(v)))
        st.v_max = float(max(st.v_max, float(v)))
        if float(v) <= -0.95:
            st.seen_min = True
        if float(v) >= +0.95:
            st.seen_max = True

    def controller_block(cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        return cfg["flight_controls"]["controller"]

    def _ensure_device_prefs(cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        if not isinstance(ctrl.get("device_prefs", None), dict):
            ctrl["device_prefs"] = {}
        dp = ctrl["device_prefs"]
        for dev in ("joystick", "keyboard", "mouse"):
            if not isinstance(dp.get(dev, None), dict):
                dp[dev] = {}
            dblk = dp[dev]
            if not isinstance(dblk.get("axis_invert", None), dict):
                dblk["axis_invert"] = {}
            if not isinstance(dblk.get("axis_calib", None), dict):
                dblk["axis_calib"] = {}
        return cfg

    def _axis_invert_pref(cfg: dict[str, Any], *, device: str, axis_id: int) -> bool:
        try:
            ctrl = controller_block(cfg)
        except Exception:
            return False
        dp = ctrl.get("device_prefs")
        if not isinstance(dp, dict):
            return False
        dblk = dp.get(str(device))
        if not isinstance(dblk, dict):
            return False
        amap = dblk.get("axis_invert")
        if not isinstance(amap, dict):
            return False
        return bool(amap.get(str(int(axis_id)), False))

    def _toggle_axis_invert(cfg: dict[str, Any], *, device: str, axis_id: int) -> dict[str, Any]:
        cfg = _ensure_device_prefs(cfg)
        ctrl = controller_block(cfg)
        dp = ctrl.get("device_prefs")
        dblk = dp.get(str(device)) if isinstance(dp, dict) else None
        if not isinstance(dblk, dict):
            return cfg
        amap = dblk.get("axis_invert")
        if not isinstance(amap, dict):
            amap = {}
            dblk["axis_invert"] = amap
        k = str(int(axis_id))
        cur = bool(amap.get(k, False))
        amap[k] = (not cur)
        return cfg

    def _axis_calib_pref(cfg: dict[str, Any], *, device: str, axis_id: int) -> dict[str, Any]:
        try:
            ctrl = controller_block(cfg)
        except Exception:
            return {}
        dp = ctrl.get("device_prefs")
        if not isinstance(dp, dict):
            return {}
        dblk = dp.get(str(device))
        if not isinstance(dblk, dict):
            return {}
        cmap = dblk.get("axis_calib")
        if not isinstance(cmap, dict):
            return {}
        node = cmap.get(str(int(axis_id)))
        return dict(node) if isinstance(node, dict) else {}

    def _write_axis_calib(cfg: dict[str, Any], *, device: str, axis_id: int, calib: dict[str, Any]) -> dict[str, Any]:
        cfg = _ensure_device_prefs(cfg)
        ctrl = controller_block(cfg)
        dp = ctrl.get("device_prefs")
        dblk = dp.get(str(device)) if isinstance(dp, dict) else None
        if not isinstance(dblk, dict):
            return cfg
        cmap = dblk.get("axis_calib")
        if not isinstance(cmap, dict):
            cmap = {}
            dblk["axis_calib"] = cmap
        out = {}
        for k in ("cap_min", "cap_max", "trim", "deadzone"):
            if k in calib and calib.get(k) is not None:
                try:
                    out[str(k)] = float(calib.get(k))
                except Exception:
                    pass
        cmap[str(int(axis_id))] = out
        return cfg

    def _reset_axis_calib(cfg: dict[str, Any], *, device: str, axis_id: int) -> dict[str, Any]:
        cfg = _ensure_device_prefs(cfg)
        ctrl = controller_block(cfg)
        dp = ctrl.get("device_prefs")
        dblk = dp.get(str(device)) if isinstance(dp, dict) else None
        if not isinstance(dblk, dict):
            return cfg
        try:
            amap = dblk.get("axis_invert")
            if isinstance(amap, dict):
                amap.pop(str(int(axis_id)), None)
        except Exception:
            pass
        try:
            cmap = dblk.get("axis_calib")
            if isinstance(cmap, dict):
                cmap.pop(str(int(axis_id)), None)
        except Exception:
            pass
        return cfg

    def _apply_axis_calib(
        cfg: dict[str, Any],
        *,
        device: str,
        kind: str,
        axis_id: int,
        v_raw: float,
    ) -> float:
        # Device-level preprocessing applied pre-signal:
        # invert -> cap (min/max) -> trim (zero shift + side scale) -> deadzone.
        v = float(v_raw)
        if str(kind) in ("axis", "mouse_motion", "motion"):
            if _axis_invert_pref(cfg, device=str(device), axis_id=int(axis_id)):
                v = -float(v)
            c = _axis_calib_pref(cfg, device=str(device), axis_id=int(axis_id))
            cap_min = c.get("cap_min")
            cap_max = c.get("cap_max")
            trim = c.get("trim")
            dz = c.get("deadzone")

            # cap normalization
            try:
                lo = float(cap_min)
                hi = float(cap_max)
            except Exception:
                lo, hi = None, None
            if lo is not None and hi is not None and float(hi) > float(lo) + 1e-6:
                mid = (float(lo) + float(hi)) * 0.5
                span = (float(hi) - float(lo)) * 0.5
                if span > 1e-6:
                    v = (float(v) - float(mid)) / float(span)
            v = float(max(-1.0, min(1.0, float(v))))

            # trim remap in capped space
            t = None
            try:
                t = float(trim)
            except Exception:
                t = None
            if t is not None:
                t = float(max(-0.95, min(0.95, float(t))))
                v0 = float(v) - float(t)
                if v0 >= 0.0:
                    denom = max(1e-6, 1.0 - float(t))
                else:
                    denom = max(1e-6, 1.0 + float(t))
                v = float(v0) / float(denom)
                v = float(max(-1.0, min(1.0, float(v))))

            # deadzone in final space
            try:
                dzf = float(dz)
            except Exception:
                dzf = 0.0
            if dzf > 0.0 and abs(float(v)) < float(dzf):
                v = 0.0
        return float(v)

    def signal_ids(cfg: dict[str, Any]) -> list[str]:
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return []
        return sorted([str(k) for k in sigs.keys()])

    def _safe_signal_id(name: str) -> str:
        s = str(name).strip()
        # Minimal sanitization: keep it a JSON key without spaces.
        s = "_".join([p for p in s.split() if p])
        return s

    def _ensure_signal(cfg: dict[str, Any], sid: str) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl["signals"]
        if str(sid) not in sigs:
            sigs[str(sid)] = {"op": "const", "value": 0.0, "kernel": "I"}
        return cfg

    def _ensure_args_len(node: dict[str, Any], n: int) -> dict[str, Any]:
        out = dict(node)
        args = out.get("args")
        if not isinstance(args, list):
            args = []
        args2 = list(args)
        while len(args2) < int(n):
            args2.append(None)
        out["args"] = args2
        return out

    def _arg_spec_from_row(*, row_kind: str, row_id: object, state_id: str) -> dict[str, Any] | None:
        rk = str(row_kind)
        if rk in ("axis", "button"):
            try:
                rid = int(row_id)
            except Exception:
                return None
            return {
                "source": "input",
                "device": "joystick",
                "kind": "button" if rk == "button" else "axis",
                "id": int(rid),
                "state": str(state_id),
            }
        if rk == "key":
            try:
                rid = int(row_id)
            except Exception:
                return None
            return {
                "source": "input",
                "device": "keyboard",
                "kind": "key",
                "id": int(rid),
                "state": str(state_id),
            }
        if rk == "mouse_motion":
            try:
                rid = int(row_id)
            except Exception:
                return None
            return {
                "source": "input",
                "device": "mouse",
                "kind": "mouse_motion",
                "id": int(rid),
                "state": str(state_id),
            }
        if rk == "kbd_axis":
            try:
                rid = int(row_id)
            except Exception:
                return None
            return {
                "source": "input",
                "device": "keyboard",
                "kind": "axis",
                "id": int(rid),
                "state": str(state_id),
            }
        if rk in ("sig_axis", "sig_button", "sig_axis2"):
            key = str(row_id)
            if not key:
                return None
            sid, comp = _split_signal_key(key)
            return {
                "source": "signal",
                "id": str(sid),
                "comp": (str(comp) if comp else ""),
                "kind": "axis" if rk in ("sig_axis", "sig_axis2") else "button",
                "state": str(state_id),
            }
        return None

    def _op_required_args(node: dict[str, Any]) -> int:
        op = str(node.get("op", ""))
        if op == "const":
            return 0
        if op in ("add", "sub"):
            return 2
        if op in ("2dseek", "2dflightstick"):
            return 2
        if op == "2dsumclamp":
            args = node.get("args")
            n = len(args) if isinstance(args, list) else 0
            n = max(2, int(n))
            if n % 2 == 1:
                n += 1
            return int(n)
        # default: one input for compiled graph nodes (op=passthrough)
        return 1

    def _bind_row_to_signal_arg(
        cfg: dict[str, Any],
        *,
        sid: str,
        arg_idx: int,
        row_kind: str,
        row_id: object,
        state_id: str,
    ) -> dict[str, Any]:
        """Bind a left-table row/state into a specific arg slot without changing op."""
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl["signals"]
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            node = {"op": "kernel", "kernel": "I"}

        spec = _arg_spec_from_row(row_kind=str(row_kind), row_id=row_id, state_id=str(state_id))
        if spec is None:
            return cfg

        req = int(_op_required_args(node))
        node2 = _ensure_args_len(node, max(int(req), int(arg_idx) + 1))
        args2 = list(node2.get("args", []))
        args2[int(arg_idx)] = dict(spec)
        node2["args"] = args2

        # Back-compat for kernel nodes: mirror arg0 input mapping into signal_id/state.
        if str(node2.get("op")) == "kernel" and isinstance(spec, dict) and spec.get("source") == "input":
            if str(spec.get("kind")) == "button":
                k_kind = int(signal_kernel_api.GP_EV_BUTTON)
            else:
                k_kind = int(signal_kernel_api.GP_EV_AXIS)
            k_dev = int(signal_kernel_api.GP_DEV_JOYSTICK)
            k_item = int(spec.get("id", 0))
            node2["signal_id"] = int(signal_kernel_api.compose_signal_id(k_dev, k_kind, k_item))
            node2["state"] = str(spec.get("state", ""))

        sigs[str(sid)] = node2
        return cfg

    def _bind_selected_input_to_signal(cfg: dict[str, Any], *, sid: str, row_kind: str, row_id: int, state_id: str) -> dict[str, Any]:
        """Persist a binding from a specific input+state to a signal.

        This stores a kernel-backed signal node. The runtime graph evaluator will
        treat unknown ops as 0.0 for now; the kernel pipeline will make it live.
        """
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl["signals"]

        if joystick is None:
            return cfg

        # Compose a stable kernel signal_id matching the C kernel scheme.
        if str(row_kind) == "button":
            k_kind = int(signal_kernel_api.GP_EV_BUTTON)
            k_dev = int(signal_kernel_api.GP_DEV_JOYSTICK)
            k_item = int(row_id)
        elif str(row_kind) == "axis":
            k_kind = int(signal_kernel_api.GP_EV_AXIS)
            k_dev = int(signal_kernel_api.GP_DEV_JOYSTICK)
            k_item = int(row_id)
        else:
            return cfg

        kernel_signal_id = int(signal_kernel_api.compose_signal_id(k_dev, k_kind, k_item))

        # Preserve kernel id if it already exists.
        prev = sigs.get(str(sid))
        prev_k = "I"
        if isinstance(prev, dict) and "kernel" in prev:
            prev_k = signal_kernel_ids.normalize_kernel_id(prev.get("kernel", "I"))

        sigs[str(sid)] = {
            "op": "kernel",
            "kernel": prev_k,
            "signal_id": int(kernel_signal_id),
            "state": str(state_id),
            "device": "joystick",
            "kind": "button" if str(row_kind) == "button" else "axis",
            "id": int(row_id),
        }
        return cfg

    def _get_signal_dim(cfg: dict[str, Any], sid: str) -> int:
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return 1
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return 1
        try:
            d = int(node.get("dim", 1))
        except Exception:
            d = 1
        return 2 if d == 2 else 1

    def _toggle_signal_dim2(cfg: dict[str, Any], sid: str) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict) or str(sid) not in sigs:
            return cfg
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return cfg
        node2 = dict(node)
        d = 1
        try:
            d = int(node2.get("dim", 1))
        except Exception:
            d = 1
        node2["dim"] = 1 if d == 2 else 2
        sigs[str(sid)] = node2
        return cfg

    def _get_signal_op(cfg: dict[str, Any], sid: str) -> str:
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return ""
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return ""
        return str(node.get("op", ""))

    def _set_signal_op(cfg: dict[str, Any], sid: str, op: str) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl["signals"]
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            node = {"op": "const", "value": 0.0, "kernel": "I"}
        node2 = dict(node)
        node2["op"] = str(op)

        # Op-driven dimensionality.
        if str(op) in ("2dseek", "2dflightstick", "2dsumclamp"):
            node2["dim"] = 2
            node2 = _ensure_args_len(node2, 2)
        else:
            # default to 1D for kernel/others
            node2["dim"] = 1

        sigs[str(sid)] = node2
        return cfg

    def _cycle_signal_op(cfg: dict[str, Any], sid: str) -> dict[str, Any]:
        ops = ["kernel", "2dseek", "2dflightstick", "2dsumclamp"]
        cur = _get_signal_op(cfg, sid)
        try:
            i = ops.index(cur)
        except Exception:
            i = 0
        nxt = ops[(i + 1) % len(ops)]
        return _set_signal_op(cfg, sid, nxt)

    def _cycle_signal_channel(cfg: dict[str, Any], sid: str) -> dict[str, Any]:
        # Channels are fixed numeric slots (0..7) plus None.
        opts: list[int | None] = [None] + [int(i) for i in range(8)]
        cur = mapped_channel_for_signal(cfg, sid)
        try:
            i = opts.index(int(cur) if cur is not None else None)
        except Exception:
            i = 0
        nxt = opts[(i + 1) % len(opts)]
        return assign_signal_to_channel(cfg, sid, nxt)

    def _signal_value_from_kernel_if_possible(cfg: dict[str, Any], sid: str, now_ns: int) -> tuple[float, int, float]:
        """Return (value, flags, hold_s) for kernel-backed signals if available."""
        if _sigk is None:
            return 0.0, 0, 0.0
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return 0.0, 0, 0.0
        node = sigs.get(str(sid))
        if not isinstance(node, dict) or str(node.get("op")) != "kernel":
            return 0.0, 0, 0.0
        try:
            ksid = int(node.get("signal_id"))
        except Exception:
            return 0.0, 0, 0.0
        fr = GP_SignalFrame()
        # Prefer peek-with-selector if present (newer DLL); otherwise peek and map in Python.
        ok = 0
        state = str(node.get("state", ""))
        sel_code = 0
        if state in ("axis", "raw", ""):
            sel_code = 0
        elif state == "D":
            sel_code = 2
        elif state == "+":
            sel_code = 3
        elif state == "-":
            sel_code = 4
        elif state == "H":
            sel_code = 5
        elif state == "hold_s":
            sel_code = 11
        elif state == "last_hold_s":
            sel_code = 12
        elif state == "last_hold_pulse":
            sel_code = 13
        elif state == "2":
            sel_code = 6
        elif state == "2H":
            sel_code = 7
        elif state == "T":
            sel_code = 8
        elif state == "t":
            sel_code = 9
        elif state == "2t":
            sel_code = 10
        try:
            if hasattr(_sigk, "gp_sigk_peek_sel"):
                ok = int(_sigk.gp_sigk_peek_sel(int(now_ns), int(ksid), int(sel_code), fr))
            else:
                ok = int(_sigk.gp_sigk_peek(int(now_ns), int(ksid), fr))
        except Exception:
            ok = 0
        if not ok:
            return 0.0, 0, 0.0

        v = float(fr.value)
        flags = int(fr.flags)
        # If we only have gp_sigk_peek, derive the selected value from flags.
        if not hasattr(_sigk, "gp_sigk_peek_sel"):
            if sel_code == 2:  # DOWN
                v = 1.0 if (flags & int(signal_status_flags.SIGF_DOWN)) else 0.0
            elif sel_code == 3:  # DOWN_EDGE
                v = 1.0 if (flags & int(signal_status_flags.SIGF_DOWN_EDGE)) else 0.0
            elif sel_code == 4:  # UP_EDGE
                v = 1.0 if (flags & int(signal_status_flags.SIGF_UP_EDGE)) else 0.0
            elif sel_code == 5:  # HOLD
                v = 1.0 if (flags & int(signal_status_flags.SIGF_HOLD)) else 0.0
            elif sel_code == 6:  # DOUBLE_EDGE
                v = 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_EDGE)) else 0.0
            elif sel_code == 7:  # DOUBLE_HOLD
                v = 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_HOLD)) else 0.0
            elif sel_code == 8:  # TOGGLED
                v = 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLED)) else 0.0
            elif sel_code == 9:  # TOGGLE_EDGE
                v = 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLE_EDGE)) else 0.0
            elif sel_code == 10:  # DOUBLE_TOGGLE_EDGE
                v = 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE)) else 0.0
            elif sel_code == 11:  # HOLD_S (normalized minutes)
                v = float(max(0.0, min(1.0, float(fr.hold_s) / 60.0)))
            elif sel_code == 12:  # LAST_HOLD_S (from aux ms if available)
                v = float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
            elif sel_code == 13:  # LAST_HOLD_PULSE (gate on UP_EDGE + HOLD)
                # Kernel emits UP_EDGE; last hold duration is carried in aux (ms).
                # Gate by a hold-threshold in ms to approximate "release had HOLD".
                if (flags & int(signal_status_flags.SIGF_UP_EDGE)) and (int(getattr(fr, "aux", 0)) >= 350):
                    v = float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
                else:
                    v = 0.0

        return float(v), int(flags), float(fr.hold_s)

    def _selector_code_from_state(state_id: str) -> int:
        st = str(state_id)
        if st in ("axis", "raw", ""):
            return 0
        if st == "D":
            return 2
        if st == "+":
            return 3
        if st == "-":
            return 4
        if st == "H":
            return 5
        if st == "2":
            return 6
        if st == "2H":
            return 7
        if st == "T":
            return 8
        if st == "t":
            return 9
        if st == "2t":
            return 10
        if st == "hold_s":
            return 11
        if st == "last_hold_s":
            return 12
        if st == "last_hold_pulse":
            return 13
        return 0

    def _sample_joystick_state_value(*, kind: str, item_id: int, state_id: str, axes_now: dict[int, float], now_ns: int) -> float:
        k = str(kind)
        st = str(state_id)
        if k == "axis":
            # Prefer local snapshot if available; otherwise query the kernel.
            if int(item_id) in axes_now:
                return float(axes_now.get(int(item_id), 0.0))
            if _sigk is not None:
                sid_k = int(signal_kernel_api.compose_signal_id(signal_kernel_api.GP_DEV_JOYSTICK, signal_kernel_api.GP_EV_AXIS, int(item_id)))
                fr = GP_SignalFrame()
                try:
                    ok = int(_sigk.gp_sigk_peek(int(now_ns), int(sid_k), fr))
                except Exception:
                    ok = 0
                if ok:
                    return float(fr.value)
            return 0.0
        if k != "button":
            return 0.0
        if _sigk is None:
            return 0.0

        sid_k = int(signal_kernel_api.compose_signal_id(signal_kernel_api.GP_DEV_JOYSTICK, signal_kernel_api.GP_EV_BUTTON, int(item_id)))
        fr = GP_SignalFrame()
        sel_code = int(_selector_code_from_state(st))
        try:
            if hasattr(_sigk, "gp_sigk_peek_sel"):
                ok = int(_sigk.gp_sigk_peek_sel(int(now_ns), int(sid_k), int(sel_code), fr))
            else:
                ok = int(_sigk.gp_sigk_peek(int(now_ns), int(sid_k), fr))
        except Exception:
            ok = 0
        if not ok:
            return 0.0

        if hasattr(_sigk, "gp_sigk_peek_sel"):
            return float(fr.value)

        # Fallback mapping when selector peek isn't available.
        flags = int(fr.flags)
        if sel_code == 2:
            return 1.0 if (flags & int(signal_status_flags.SIGF_DOWN)) else 0.0
        if sel_code == 3:
            return 1.0 if (flags & int(signal_status_flags.SIGF_DOWN_EDGE)) else 0.0
        if sel_code == 4:
            return 1.0 if (flags & int(signal_status_flags.SIGF_UP_EDGE)) else 0.0
        if sel_code == 5:
            return 1.0 if (flags & int(signal_status_flags.SIGF_HOLD)) else 0.0
        if sel_code == 6:
            return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_EDGE)) else 0.0
        if sel_code == 7:
            return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_HOLD)) else 0.0
        if sel_code == 8:
            return 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLED)) else 0.0
        if sel_code == 9:
            return 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLE_EDGE)) else 0.0
        if sel_code == 10:
            return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE)) else 0.0
        if sel_code == 11:
            return float(max(0.0, min(1.0, float(fr.hold_s) / 60.0)))
        if sel_code == 12:
            return float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
        if sel_code == 13:
            if (flags & int(signal_status_flags.SIGF_UP_EDGE)) and (int(getattr(fr, "aux", 0)) >= 350):
                return float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
            return 0.0
        return float(fr.value)

    def _value_from_arg_spec(cfg: dict[str, Any], spec: dict[str, Any], *, axes_now: dict[int, float], now_ns: int) -> float:
        if not isinstance(spec, dict):
            return 0.0
        src = str(spec.get("source", ""))

        def _peek_kernel_input_value(*, dev_code: int, kind_code: int, item_id: int, sel_code: int) -> float:
            if _sigk is None:
                return 0.0
            sid_k = int(signal_kernel_api.compose_signal_id(int(dev_code), int(kind_code), int(item_id)))
            fr = GP_SignalFrame()
            ok = 0
            try:
                if hasattr(_sigk, "gp_sigk_peek_sel"):
                    ok = int(_sigk.gp_sigk_peek_sel(int(now_ns), int(sid_k), int(sel_code), fr))
                else:
                    ok = int(_sigk.gp_sigk_peek(int(now_ns), int(sid_k), fr))
            except Exception:
                ok = 0
            if not ok:
                return 0.0
            # For axes (sel_code==0) the value is directly fr.value.
            if sel_code == 0 or hasattr(_sigk, "gp_sigk_peek_sel"):
                return float(fr.value)
            # Fallback selector mapping when peek_sel isn't available.
            flags = int(fr.flags)
            if sel_code == 2:
                return 1.0 if (flags & int(signal_status_flags.SIGF_DOWN)) else 0.0
            if sel_code == 3:
                return 1.0 if (flags & int(signal_status_flags.SIGF_DOWN_EDGE)) else 0.0
            if sel_code == 4:
                return 1.0 if (flags & int(signal_status_flags.SIGF_UP_EDGE)) else 0.0
            if sel_code == 5:
                return 1.0 if (flags & int(signal_status_flags.SIGF_HOLD)) else 0.0
            if sel_code == 6:
                return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_EDGE)) else 0.0
            if sel_code == 7:
                return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_HOLD)) else 0.0
            if sel_code == 8:
                return 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLED)) else 0.0
            if sel_code == 9:
                return 1.0 if (flags & int(signal_status_flags.SIGF_TOGGLE_EDGE)) else 0.0
            if sel_code == 10:
                return 1.0 if (flags & int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE)) else 0.0
            if sel_code == 11:
                return float(max(0.0, min(1.0, float(fr.hold_s) / 60.0)))
            if sel_code == 12:
                return float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
            if sel_code == 13:
                if (flags & int(signal_status_flags.SIGF_UP_EDGE)) and (int(getattr(fr, "aux", 0)) >= 350):
                    return float(max(0.0, min(1.0, (float(getattr(fr, "aux", 0)) * 0.001) / 60.0)))
                return 0.0
            return float(fr.value)

        def _sample_input_spec_value(*, dev: str, kind: str, item_id: int, state_id: str) -> float:
            if dev == "joystick":
                vv = float(_sample_joystick_state_value(kind=kind, item_id=int(item_id), state_id=state_id, axes_now=axes_now, now_ns=int(now_ns)))
                if str(kind) == "axis":
                    vv = float(_apply_axis_calib(cfg, device="joystick", kind="axis", axis_id=int(item_id), v_raw=float(vv)))
                return float(vv)

            sel_code = int(_selector_code_from_state(str(state_id)))
            if dev == "keyboard":
                dev_code = int(signal_kernel_api.GP_DEV_KEYBOARD)
                if kind == "axis":
                    kind_code = int(signal_kernel_api.GP_EV_AXIS)
                    vv = float(_peek_kernel_input_value(dev_code=dev_code, kind_code=kind_code, item_id=int(item_id), sel_code=0))
                    vv = float(_apply_axis_calib(cfg, device="keyboard", kind="axis", axis_id=int(item_id), v_raw=float(vv)))
                    return float(vv)
                if kind == "key":
                    kind_code = int(signal_kernel_api.GP_EV_KEY)
                    return float(_peek_kernel_input_value(dev_code=dev_code, kind_code=kind_code, item_id=int(item_id), sel_code=int(sel_code)))
                return 0.0

            if dev == "mouse":
                dev_code = int(signal_kernel_api.GP_DEV_MOUSE)
                if kind in ("mouse_motion", "motion"):
                    kind_code = int(signal_kernel_api.GP_EV_MOUSE_MOTION)
                    vv = float(_peek_kernel_input_value(dev_code=dev_code, kind_code=kind_code, item_id=int(item_id), sel_code=0))
                    vv = float(_apply_axis_calib(cfg, device="mouse", kind="mouse_motion", axis_id=int(item_id), v_raw=float(vv)))
                    return float(vv)
                return 0.0

            return 0.0

        if src == "input":
            dev = str(spec.get("device", ""))
            kind = str(spec.get("kind", ""))
            try:
                item_id = int(spec.get("id", 0))
            except Exception:
                item_id = 0
            state_id = str(spec.get("state", "axis"))
            return float(_sample_input_spec_value(dev=dev, kind=kind, item_id=int(item_id), state_id=state_id))

        if src == "signal":
            sid = str(spec.get("id", ""))
            comp = str(spec.get("comp", ""))
            comp2 = comp if comp else None
            key = _signal_key_for_comp(sid, comp2)
            return float(_signal_value_for_signal_key(cfg, str(key), axes_now=axes_now, now_ns=int(now_ns)))
        return 0.0

    def _signal_xy_from_op(cfg: dict[str, Any], sid: str, *, axes_now: dict[int, float], now_ns: int) -> tuple[float, float]:
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return 0.0, 0.0
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return 0.0, 0.0
        op = str(node.get("op", ""))
        if op not in ("2dseek", "2dflightstick", "2dsumclamp"):
            return 0.0, 0.0
        args = node.get("args")
        if not isinstance(args, list):
            args = []

        if op == "2dsumclamp":
            sx = 0.0
            sy = 0.0
            n = int(len(args))
            i = 0
            while i + 1 < n:
                a_spec = args[i]
                b_spec = args[i + 1]
                if isinstance(a_spec, dict):
                    sx += float(_value_from_arg_spec(cfg, a_spec, axes_now=axes_now, now_ns=int(now_ns)))
                if isinstance(b_spec, dict):
                    sy += float(_value_from_arg_spec(cfg, b_spec, axes_now=axes_now, now_ns=int(now_ns)))
                i += 2
            x = float(max(-1.0, min(1.0, float(sx))))
            y = float(max(-1.0, min(1.0, float(sy))))
            return float(x), float(y)

        a_spec = args[0] if len(args) > 0 else None
        b_spec = args[1] if len(args) > 1 else None
        a = float(_value_from_arg_spec(cfg, a_spec, axes_now=axes_now, now_ns=int(now_ns))) if isinstance(a_spec, dict) else 0.0
        b = float(_value_from_arg_spec(cfg, b_spec, axes_now=axes_now, now_ns=int(now_ns))) if isinstance(b_spec, dict) else 0.0

        x = float(a)
        y = float(b)
        if _sigk is not None:
            try:
                from ctypes import c_float, byref

                xo = c_float(0.0)
                yo = c_float(0.0)
                if op == "2dflightstick" and hasattr(_sigk, "gp_sigop_2dflightstick"):
                    _sigk.gp_sigop_2dflightstick(c_float(a), c_float(b), byref(xo), byref(yo))
                    x, y = float(xo.value), float(yo.value)
                elif op == "2dseek" and hasattr(_sigk, "gp_sigop_2dseek"):
                    _sigk.gp_sigop_2dseek(c_float(a), c_float(b), byref(xo), byref(yo))
                    x, y = float(xo.value), float(yo.value)
            except Exception:
                pass
        return float(x), float(y)

    def _signal_value_for_signal_key(cfg: dict[str, Any], key: str, *, axes_now: dict[int, float], now_ns: int) -> float:
        sid, comp = _split_signal_key(str(key))
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        node = sigs.get(str(sid)) if isinstance(sigs, dict) else None
        if isinstance(node, dict) and str(node.get("op", "")) in ("2dseek", "2dflightstick", "2dsumclamp"):
            x, y = _signal_xy_from_op(cfg, str(sid), axes_now=axes_now, now_ns=int(now_ns))
            if comp == "y":
                return float(y)
            return float(x)
        v, _f, _hs = _signal_value_from_kernel_if_possible(cfg, str(sid), int(now_ns))
        return float(v)

    def _update_virtual_buttons_from_signals(cfg: dict[str, Any], now_ns: int, *, axes_now: dict[int, float]) -> None:
        if _sigk is None or not hasattr(_sigk, "gp_sigk_sigtobutton"):
            return
        for sid in signal_ids(cfg):
            dim = _get_signal_dim(cfg, str(sid))
            comps: list[str | None] = [None] if dim != 2 else ["x", "y"]
            for comp in comps:
                key = _signal_key_for_comp(str(sid), comp)
                v_src = float(_signal_value_for_signal_key(cfg, str(key), axes_now=axes_now, now_ns=int(now_ns)))
                vb_sid = int(_virt_button_signal_id_for_signal_key(str(key)))
                try:
                    _sigk.gp_sigk_sigtobutton(int(now_ns), int(vb_sid), float(v_src), float(SIG_TO_BUTTON_EPS))
                except Exception:
                    pass

    def _update_virtual_hats_from_signals(cfg: dict[str, Any], now_ns: int, *, axes_now: dict[int, float]) -> list[GP_InputEvent]:
        # Convert each dim=2 signal into a *trinary* virtual hat:
        # - two virtual axes (-1/0/+1)
        # - eight virtual buttons (one active direction)
        if _sigk is None:
            return []
        if state.sig_hat_prev_dir is None:
            state.sig_hat_prev_dir = {}
        out: list[GP_InputEvent] = []
        for sid in signal_ids(cfg):
            if int(_get_signal_dim(cfg, str(sid))) != 2:
                continue
            # Sample analog x/y from the signal, then quantize to trits.
            x_src = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(sid), "x"), axes_now=axes_now, now_ns=int(now_ns)))
            y_src = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(sid), "y"), axes_now=axes_now, now_ns=int(now_ns)))
            tx = int(_trit(float(x_src), float(HAT_TRIT_EPS)))
            ty = int(_trit(float(y_src), float(HAT_TRIT_EPS)))

            # Push trinary axes each frame.
            out.append(
                GP_InputEvent(
                    t_mono_ns=int(now_ns),
                    device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                    kind=int(signal_kernel_api.GP_EV_AXIS),
                    id=int(_sig_hat_axis_id(str(sid), "x")),
                    v0=float(tx),
                    v1=0.0,
                    flags=0,
                )
            )
            out.append(
                GP_InputEvent(
                    t_mono_ns=int(now_ns),
                    device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                    kind=int(signal_kernel_api.GP_EV_AXIS),
                    id=int(_sig_hat_axis_id(str(sid), "y")),
                    v0=float(ty),
                    v1=0.0,
                    flags=0,
                )
            )

            cur_dir = _dir_idx_8(int(tx), int(ty))
            prev_dir = state.sig_hat_prev_dir.get(str(sid))
            if prev_dir != cur_dir:
                if prev_dir is not None:
                    out.append(
                        GP_InputEvent(
                            t_mono_ns=int(now_ns),
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_BUTTON),
                            id=int(_sig_hat_btn_id(str(sid), int(prev_dir))),
                            v0=0.0,
                            v1=0.0,
                            flags=0,
                        )
                    )
                if cur_dir is not None:
                    out.append(
                        GP_InputEvent(
                            t_mono_ns=int(now_ns),
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_BUTTON),
                            id=int(_sig_hat_btn_id(str(sid), int(cur_dir))),
                            v0=1.0,
                            v1=0.0,
                            flags=0,
                        )
                    )
                state.sig_hat_prev_dir[str(sid)] = cur_dir
        return out

    def _signal_value_fallback_python(cfg: dict[str, Any], sid: str, *, axes_now: dict[int, float], buttons_now: set[int], hats_now: dict[int, tuple[int, int]]) -> float:
        if not bool(ENABLE_PY_SIGNAL_FALLBACK):
            return 0.0
        try:
            ctx = input_graph.GraphEvalContext(axes=dict(axes_now), buttons=set(buttons_now), hats=dict(hats_now))
            ctrl = controller_block(cfg)
            return float(input_graph.eval_controller_signal(ctrl=ctrl, sid=str(sid), ctx=ctx))
        except Exception:
            return 0.0

    def _truncate_to_px(s: str, max_px: int) -> str:
        txt = str(s)
        if max_px <= 0:
            return ""
        try:
            w, _h = font.size(txt)
        except Exception:
            return txt
        if w <= max_px:
            return txt
        ell = "…"
        lo, hi = 0, len(txt)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = txt[:mid] + ell
            try:
                cw, _ = font.size(cand)
            except Exception:
                cw = len(cand) * 8
            if cw <= max_px:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best else ell

    def mapped_channel_for_signal(cfg: dict[str, Any], sid: str) -> int | None:
        ctrl = controller_block(cfg)
        chmap = ctrl.get("channels")
        if not isinstance(chmap, dict):
            return None
        for k, spec in chmap.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if not isinstance(spec, dict):
                continue
            if spec.get("source") == "signal" and str(spec.get("id")) == str(sid):
                return int(idx)
        return None

    def assign_signal_to_channel(cfg: dict[str, Any], sid: str, channel: int | None) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        chmap = ctrl["channels"]

        # Treat signal->channel as 1:1 for now: remove existing mappings for this sid.
        for k in list(chmap.keys()):
            spec = chmap.get(k)
            if isinstance(spec, dict) and spec.get("source") == "signal" and str(spec.get("id")) == str(sid):
                chmap.pop(k, None)

        if channel is None:
            return cfg
        chmap[str(int(channel))] = {"source": "signal", "id": str(sid)}
        return cfg

    def get_signal_kernel(cfg: dict[str, Any], sid: str) -> str:
        ctrl = controller_block(cfg)
        sigs = ctrl.get("signals")
        if not isinstance(sigs, dict):
            return "I"
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return "I"
        return signal_kernel_ids.normalize_kernel_id(node.get("kernel", "I"))

    def set_signal_kernel(cfg: dict[str, Any], sid: str, kid: str) -> dict[str, Any]:
        cfg = input_graph.ensure_controller_graph(cfg)
        ctrl = cfg["flight_controls"]["controller"]
        sigs = ctrl["signals"]
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            return cfg
        node2 = dict(node)
        node2["kernel"] = signal_kernel_ids.normalize_kernel_id(kid)
        sigs[str(sid)] = node2
        return cfg

    axes_prev: dict[int, float] = {}
    buttons_prev: set[int] = set()
    hats_prev: dict[int, tuple[int, int]] = {}
    keys_prev: set[int] = set()
    mouse_buttons_prev: set[int] = set()

    def _build_left_rows(cfg: dict[str, Any]) -> list[tuple[str, object]]:
        rows: list[tuple[str, object]] = []

        def _hdr(key: str, label: str) -> None:
            rows.append(("device_header", {"key": str(key), "label": str(label)}))

        def _is_collapsed(key: str) -> bool:
            return bool(state.collapsed_devices is not None and str(key) in state.collapsed_devices)

        # Keyboard
        _hdr("keyboard:0", "KEYBOARD")
        if not _is_collapsed("keyboard:0"):
            rows.append(("kbd_axis", {"id": 0, "label": "kbd.wasd.x"}))
            rows.append(("kbd_axis", {"id": 1, "label": "kbd.wasd.y"}))
            rows.append(("kbd_axis", {"id": 2, "label": "kbd.arrows.x"}))
            rows.append(("kbd_axis", {"id": 3, "label": "kbd.arrows.y"}))

        # Mouse
        _hdr("mouse:0", "MOUSE")
        if not _is_collapsed("mouse:0"):
            rows.append(("mouse_motion", {"id": 0, "label": "mouse.dx"}))
            rows.append(("mouse_motion", {"id": 1, "label": "mouse.dy"}))

        # Joystick
        jname = "JOYSTICK"
        if joystick is not None:
            try:
                jname = f"JOYSTICK: {str(joystick.get_name())}" if joystick.get_name() else "JOYSTICK"
            except Exception:
                jname = "JOYSTICK"
        _hdr("joystick:0", jname)
        if joystick is not None and not _is_collapsed("joystick:0"):
            try:
                na = int(joystick.get_numaxes())
            except Exception:
                na = 0
            try:
                nb = int(joystick.get_numbuttons())
            except Exception:
                nb = 0
            for a in range(max(0, na)):
                rows.append(("axis", int(a)))
            for b in range(max(0, nb)):
                rows.append(("button", int(b)))

            # Hardware hats (D-pad) shown as 2D trinary matrix rows.
            try:
                nh = int(joystick.get_numhats())
            except Exception:
                nh = 0
            for h in range(max(0, nh)):
                rows.append(("matrix2d_hat", int(h)))
                if state.expanded_hats is not None and int(h) in state.expanded_hats:
                    for di in range(8):
                        bid = int(_hard_hat_btn_id(int(h), int(di)))
                        rows.append(("vbutton", {"id": int(bid), "label": f"hat.{int(h)}.{_dir_label_8(int(di))}"}))

        # Synthetic signals (always visible; kept separate from device groups).
        for sid in signal_ids(cfg):
            dim = _get_signal_dim(cfg, str(sid))
            if int(dim) == 2:
                rows.append(("matrix2d_sig", str(sid)))
                for comp in ("x", "y"):
                    key = _signal_key_for_comp(str(sid), str(comp))
                    rows.append(("sig_axis2", str(key)))
            else:
                key = _signal_key_for_comp(str(sid), None)
                rows.append(("sig_axis", str(key)))
                if state.collapsed_sigs is None or str(key) not in state.collapsed_sigs:
                    rows.append(("sig_button", str(key)))

        return rows

    def _draw_dense_ui(
        *,
        cfg: dict[str, Any],
        axes_now: dict[int, float],
        buttons_now: set[int],
        hats_now: dict[int, tuple[int, int]],
        kbd_axes_now: dict[int, float],
        mouse_axes_now: dict[int, float],
        now_ns: int,
        hitboxes_out: list[HitBox],
    ) -> None:
        """Draw dense left/right tables and build hitboxes."""
        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_RGBA,
            GL_SRC_ALPHA,
            GL_TRIANGLE_FAN,
            GL_TRIANGLES,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_BYTE,
            glBegin,
            glBlendFunc,
            glClear,
            glClearColor,
            glColor4f,
            glDisable,
            glDrawPixels,
            glEnable,
            glEnd,
            glIsEnabled,
            glLoadIdentity,
            glMatrixMode,
            glOrtho,
            glPixelStorei,
            glPopMatrix,
            glPushMatrix,
            glRasterPos2f,
            glVertex2f,
        )

        def _px_to_nx(x: int) -> float:
            return float(x) / float(max(1, int(width)))

        def _px_to_ny(y: int) -> float:
            return 1.0 - (float(y) / float(max(1, int(height))))

        def _draw_rect_px(x: int, y: int, w: int, h: int, rgba: tuple[float, float, float, float]) -> None:
            x0 = _px_to_nx(int(x))
            x1 = _px_to_nx(int(x + w))
            y0 = _px_to_ny(int(y + h))
            y1 = _px_to_ny(int(y))
            r, g, b, a = rgba
            glColor4f(float(r), float(g), float(b), float(a))
            glBegin(GL_TRIANGLES)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

        def _draw_circle_px(cx: int, cy: int, r_px: int, rgba: tuple[float, float, float, float], segments: int = 10) -> None:
            cxn = _px_to_nx(int(cx))
            cyn = _px_to_ny(int(cy))
            # Radius in normalized units: use X scale.
            rn = float(r_px) / float(max(1, int(width)))
            r, g, b, a = rgba
            glColor4f(float(r), float(g), float(b), float(a))
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cxn, cyn)
            for i in range(int(segments) + 1):
                ang = (float(i) / float(max(1, int(segments)))) * 6.283185307179586
                glVertex2f(float(cxn + rn * float(__import__("math").cos(ang))), float(cyn + rn * float(__import__("math").sin(ang))))
            glEnd()

        def _draw_text_px(x: int, y: int, text: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
            s = str(text)
            if not s:
                return
            try:
                surf = font.render(s, True, color)
                surf = surf.convert_alpha()
            except Exception:
                return
            data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(_px_to_nx(int(x)), _px_to_ny(int(y)))
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        if depth_was_enabled:
            glDisable(GL_DEPTH_TEST)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Layout
        pad = 10
        header_h = max(18, int(font.get_linesize()) + 2)
        row_h = max(16, int(font.get_linesize()) + 2)
        # Keep signals pane narrow.
        left_w = int(width * 0.72)
        right_w = int(width - left_w)
        left_x0 = 0
        right_x0 = int(left_w)

        left_geom = ui_tables.TableGeom(x0=int(left_x0), y0=0, w=int(left_w), h=int(height), pad=int(pad), header_h=int(header_h), row_h=int(row_h))
        right_geom = ui_tables.TableGeom(x0=int(right_x0), y0=0, w=int(right_w), h=int(height), pad=int(pad), header_h=int(header_h), row_h=int(row_h))

        # Table headers
        _draw_text_px(pad, pad + header_h, "INPUTS (click LED/ticks)")
        _draw_text_px(right_x0 + pad, pad + header_h, "SIGNALS (click to bind)")

        # Left rows: inputs grouped by device headers.
        rows: list[tuple[str, object]] = _build_left_rows(cfg)

        max_left_rows = int(ui_tables.max_visible_rows(left_geom))
        state.sel_left_idx = int(max(0, min(int(state.sel_left_idx), max(0, len(rows) - 1)))) if rows else 0
        l_first, l_last, _ = ui_tables.visible_window(rows_total=len(rows), sel_idx=int(state.sel_left_idx), g=left_geom, scroll=left_scroll, center=False)

        # LED columns for buttons
        led_defs: list[tuple[str, int]] = [
            ("D", int(signal_status_flags.SIGF_DOWN)),
            ("+", int(signal_status_flags.SIGF_DOWN_EDGE)),
            ("-", int(signal_status_flags.SIGF_UP_EDGE)),
            ("H", int(signal_status_flags.SIGF_HOLD)),
            ("2", int(signal_status_flags.SIGF_DOUBLE_EDGE)),
            ("2H", int(signal_status_flags.SIGF_DOUBLE_HOLD)),
            ("T", int(signal_status_flags.SIGF_TOGGLED)),
            ("t", int(signal_status_flags.SIGF_TOGGLE_EDGE)),
            ("2t", int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE)),
        ]

        name_col_w = int(left_w * 0.22)
        led_col_w = 14
        led_r = 4
        bar_w = int(left_w * 0.28)

        # Header labels
        y_hdr = pad + header_h + 2
        _draw_text_px(pad, y_hdr, "id")
        x_led0 = pad + name_col_w
        for i, (lbl, _bit) in enumerate(led_defs):
            _draw_text_px(x_led0 + i * led_col_w, y_hdr, lbl)
        _draw_text_px(x_led0 + len(led_defs) * led_col_w + 12, y_hdr, "axis")

        # Draw rows
        for idx in range(int(l_first), int(l_last)):
            kind, rid = rows[int(idx)]
            y = int(ui_tables.row_baseline_y(g=left_geom, visible_idx=(int(idx) - int(l_first))))
            # Row background/border
            _draw_rect_px(left_x0 + 2, y - row_h + 2, left_w - 4, row_h, (0.06, 0.06, 0.06, 1.0))
            _draw_rect_px(left_x0 + 2, y - row_h + 2, left_w - 4, 1, (0.25, 0.25, 0.25, 1.0))
            if kind == "device_header":
                key = ""
                label = ""
                if isinstance(rid, dict):
                    key = str(rid.get("key", ""))
                    label = str(rid.get("label", ""))
                is_collapsed = bool(state.collapsed_devices is not None and str(key) in state.collapsed_devices)
                toggle_lbl = "+" if is_collapsed else "-"
                _draw_text_px(pad, y, f"[{toggle_lbl}] {label}")
                hitboxes_out.append(
                    HitBox(
                        x0=int(left_x0 + 2),
                        y0=int(y - row_h + 2),
                        x1=int(left_x0 + left_w - 4),
                        y1=int(y + 2),
                        payload={"kind": "dev_toggle", "dev": str(key), "row_idx": int(idx)},
                    )
                )
                continue
            if kind in ("sig_axis", "sig_button", "sig_axis2"):
                key = str(rid)
                sid0, comp0 = _split_signal_key(key)
                collapse_on = (state.collapsed_sigs is not None and key in state.collapsed_sigs)
                toggle_lbl = "+" if collapse_on else "-"
                # Collapse toggle (only on the signal axis row)
                if kind == "sig_axis":
                    _draw_text_px(pad, y, f"[{toggle_lbl}]")
                    hitboxes_out.append(
                        HitBox(
                            x0=int(left_x0 + 2),
                            y0=int(y - row_h + 2),
                            x1=int(left_x0 + 26),
                            y1=int(y + 2),
                            payload={"kind": "sig_collapse", "sid": str(key)},
                        )
                    )
                label = f"sig.{sid0}" if not comp0 else f"sig.{sid0}.{comp0}"
                _draw_text_px(pad + 28, y, _truncate_to_px(label, int(name_col_w - 28)))
            elif kind in ("matrix2d_hat", "matrix2d_sig"):
                # matrix rows draw their own label
                pass
            elif kind == "vbutton":
                lbl = ""
                if isinstance(rid, dict):
                    lbl = str(rid.get("label", ""))
                _draw_text_px(pad + 28, y, _truncate_to_px(lbl, int(name_col_w - 28)))
            elif kind in ("kbd_axis", "mouse_motion"):
                lbl = ""
                rid2 = 0
                if isinstance(rid, dict):
                    lbl = str(rid.get("label", ""))
                    try:
                        rid2 = int(rid.get("id", 0))
                    except Exception:
                        rid2 = 0
                else:
                    rid2 = int(rid)
                    lbl = f"{kind}.{rid2}"
                _draw_text_px(pad + 28, y, _truncate_to_px(lbl, int(name_col_w - 28)))
            else:
                _draw_text_px(pad, y, f"{kind}.{rid}")

            if kind in ("matrix2d_hat", "matrix2d_sig"):
                # Render a 2D trinary matrix row: two axes (x/y) + eight direction buttons.
                if kind == "matrix2d_hat":
                    hat_idx = int(rid)
                    hx, hy = hats_now.get(int(hat_idx), (0, 0))
                    tx, ty = int(hx), int(hy)
                    label = f"hat.{hat_idx}"
                    ax_x_id = int(_hard_hat_axis_id(int(hat_idx), "x"))
                    ax_y_id = int(_hard_hat_axis_id(int(hat_idx), "y"))

                    # Expand toggle for advanced binding (drops down per-direction button rows).
                    exp_on = bool(state.expanded_hats is not None and int(hat_idx) in state.expanded_hats)
                    toggle_lbl = "-" if exp_on else "+"
                    _draw_text_px(pad, y, f"[{toggle_lbl}]")
                    hitboxes_out.append(
                        HitBox(
                            x0=int(left_x0 + 2),
                            y0=int(y - row_h + 2),
                            x1=int(left_x0 + 26),
                            y1=int(y + 2),
                            payload={"kind": "hat_expand", "hat": int(hat_idx)},
                        )
                    )

                    def _btn_id_for_dir(di: int) -> int:
                        return int(_hard_hat_btn_id(int(hat_idx), int(di)))

                else:
                    sid = str(rid)
                    # Quantize the 2D signal into a virtual hat.
                    x_src = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(sid), "x"), axes_now=axes_now, now_ns=int(now_ns)))
                    y_src = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(sid), "y"), axes_now=axes_now, now_ns=int(now_ns)))
                    tx = int(_trit(float(x_src), float(HAT_TRIT_EPS)))
                    ty = int(_trit(float(y_src), float(HAT_TRIT_EPS)))
                    label = f"2d.{sid}"
                    ax_x_id = int(_sig_hat_axis_id(str(sid), "x"))
                    ax_y_id = int(_sig_hat_axis_id(str(sid), "y"))

                    def _btn_id_for_dir(di: int) -> int:
                        return int(_sig_hat_btn_id(str(sid), int(di)))

                _draw_text_px(pad + 28, y, _truncate_to_px(label, int(name_col_w - 28)))

                # Direction buttons (DOWN-only) in the LED column region.
                cur_dir = _dir_idx_8(int(tx), int(ty))
                cy = y - int(row_h // 2)
                for i, _d in enumerate(DIRS_8):
                    cx = x_led0 + i * led_col_w + 6
                    on = (cur_dir is not None) and (int(cur_dir) == int(i))
                    col_on = (0.25, 1.0, 0.25, 1.0)
                    col_off = (0.15, 0.15, 0.18, 1.0)
                    _draw_circle_px(cx, cy, led_r, col_on if on else col_off)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(cx - led_col_w // 2),
                            y0=int(cy - led_col_w // 2),
                            x1=int(cx + led_col_w // 2),
                            y1=int(cy + led_col_w // 2),
                            payload={
                                "kind": "left_state",
                                "row_kind": "button",
                                "row_id": int(_btn_id_for_dir(int(i))),
                                "row_idx": int(idx),
                                "state": "D",
                            },
                        )
                    )

                # Two small axis bars (x and y) stacked.
                x_bar = x_led0 + len(led_defs) * led_col_w + 12
                h_bar = 4
                gap = 2
                y_bar_x = int(cy + gap)
                y_bar_y = int(cy - h_bar - gap)

                def _x_at(val: float) -> int:
                    vv = max(-1.0, min(1.0, float(val)))
                    return int(x_bar + (vv + 1.0) * 0.5 * float(bar_w))

                for yb, vv in ((y_bar_x, float(tx)), (y_bar_y, float(ty))):
                    _draw_rect_px(x_bar, int(yb), bar_w, h_bar, (0.10, 0.10, 0.10, 1.0))
                    _draw_rect_px(x_bar, int(yb), bar_w, 1, (0.30, 0.30, 0.30, 1.0))
                    _draw_rect_px(x_bar, int(yb + h_bar - 1), bar_w, 1, (0.30, 0.30, 0.30, 1.0))
                    _draw_rect_px(_x_at(-1.0), int(yb - 1), 1, h_bar + 2, (1.0, 1.0, 1.0, 1.0))
                    _draw_rect_px(_x_at(0.0), int(yb - 1), 1, h_bar + 2, (1.0, 1.0, 1.0, 1.0))
                    _draw_rect_px(_x_at(+1.0), int(yb - 1), 1, h_bar + 2, (1.0, 1.0, 1.0, 1.0))
                    _draw_rect_px(_x_at(vv), int(yb - 2), 2, h_bar + 4, (1.0, 1.0, 0.25, 1.0))

                # Clickable hitboxes for the x/y axis bars.
                hitboxes_out.append(
                    HitBox(
                        x0=int(x_bar),
                        y0=int(y_bar_x - 4),
                        x1=int(x_bar + bar_w),
                        y1=int(y_bar_x + h_bar + 4),
                        payload={"kind": "left_axis", "row_kind": "axis", "row_id": int(ax_x_id), "row_idx": int(idx), "state": "axis"},
                    )
                )
                hitboxes_out.append(
                    HitBox(
                        x0=int(x_bar),
                        y0=int(y_bar_y - 4),
                        x1=int(x_bar + bar_w),
                        y1=int(y_bar_y + h_bar + 4),
                        payload={"kind": "left_axis", "row_kind": "axis", "row_id": int(ax_y_id), "row_idx": int(idx), "state": "axis"},
                    )
                )

                # Skip default per-kind rendering below.
                continue

            if kind in ("button", "sig_button", "vbutton"):
                # Query kernel flags if available; else fallback local machine.
                flags = 0
                hold_s = 0.0
                if kind in ("button", "vbutton"):
                    if kind == "vbutton" and isinstance(rid, dict):
                        try:
                            rid_btn = int(rid.get("id", 0))
                        except Exception:
                            rid_btn = 0
                    else:
                        rid_btn = int(rid)
                    if _sigk is not None:
                        sid_k = signal_kernel_api.compose_signal_id(signal_kernel_api.GP_DEV_JOYSTICK, signal_kernel_api.GP_EV_BUTTON, int(rid_btn))
                        fr = GP_SignalFrame()
                        try:
                            ok = int(_sigk.gp_sigk_peek(int(now_ns), int(sid_k), fr))
                        except Exception:
                            ok = 0
                        if ok:
                            flags = int(fr.flags)
                            hold_s = float(fr.hold_s)
                else:
                    # For synthetic signal-as-button rows, show the *virtual button* derived from
                    # the signal value (value > eps => down). This intentionally does NOT mirror
                    # the source signal's button flags.
                    if _sigk is not None:
                        vb_sid = int(_virt_button_signal_id_for_signal_key(str(rid)))
                        fr = GP_SignalFrame()
                        try:
                            ok = int(_sigk.gp_sigk_peek(int(now_ns), int(vb_sid), fr))
                        except Exception:
                            ok = 0
                        if ok:
                            flags = int(fr.flags)
                            hold_s = float(fr.hold_s)

                for i, (_lbl, bit) in enumerate(led_defs):
                    cx = x_led0 + i * led_col_w + 6
                    cy = y - int(row_h // 2)
                    on = (int(flags) & int(bit)) != 0
                    col_on = (0.25, 1.0, 0.25, 1.0)
                    col_off = (0.10, 0.20, 0.10, 1.0)
                    # Edges/toggles read better as white flashes.
                    if bit in (
                        int(signal_status_flags.SIGF_DOWN_EDGE),
                        int(signal_status_flags.SIGF_UP_EDGE),
                        int(signal_status_flags.SIGF_TOGGLE_EDGE),
                        int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE),
                        int(signal_status_flags.SIGF_DOUBLE_EDGE),
                    ):
                        col_on = (1.0, 1.0, 1.0, 1.0)
                        col_off = (0.20, 0.20, 0.20, 1.0)

                    _draw_circle_px(cx, cy, led_r, col_on if on else col_off)
                    # vbutton rows bind as a normal joystick button id.
                    bind_kind = "button" if kind == "vbutton" else kind
                    bind_id = int(rid.get("id")) if kind == "vbutton" and isinstance(rid, dict) else rid
                    hitboxes_out.append(
                        HitBox(
                            x0=int(cx - led_col_w // 2),
                            y0=int(cy - led_col_w // 2),
                            x1=int(cx + led_col_w // 2),
                            y1=int(cy + led_col_w // 2),
                            payload={
                                "kind": "left_state",
                                "row_kind": bind_kind,
                                "row_id": bind_id,
                                "row_idx": int(idx),
                                "state": str(_lbl),
                            },
                        )
                    )

                # Hold text (tiny)
                # Timer sources:
                # - hold_s: current time-since-on while down
                # - last_hold_pulse: value on UP_EDGE if the release qualified as HOLD
                last_ms = 0
                if kind in ("button", "vbutton"):
                    try:
                        last_ms = int(getattr(fr, "aux", 0))
                    except Exception:
                        last_ms = 0
                else:
                    # Synthetic signal-as-button row: use the virtual button's aux (last_hold_ms).
                    if _sigk is not None:
                        try:
                            vb_sid2 = int(_virt_button_signal_id_for_signal_key(str(rid)))
                            fr2 = GP_SignalFrame()
                            if int(_sigk.gp_sigk_peek(int(now_ns), int(vb_sid2), fr2)):
                                last_ms = int(getattr(fr2, "aux", 0))
                        except Exception:
                            last_ms = 0
                last_s = float(max(0.0, float(last_ms) * 0.001))

                tx = int(x_led0 + len(led_defs) * led_col_w + 6)
                _draw_text_px(tx, y, f"h:{hold_s:0.2f}")
                _draw_text_px(tx + 58, y, f"l:{last_s:0.2f}")

                # Clickable hitboxes on timers to bind them as signal sources.
                bind_kind2 = "button" if kind == "vbutton" else kind
                bind_id2 = int(rid.get("id")) if kind == "vbutton" and isinstance(rid, dict) else rid
                hitboxes_out.append(
                    HitBox(
                        x0=int(tx),
                        y0=int(y - row_h + 2),
                        x1=int(tx + 56),
                        y1=int(y + 2),
                        payload={"kind": "left_state", "row_kind": bind_kind2, "row_id": bind_id2, "row_idx": int(idx), "state": "hold_s"},
                    )
                )
                hitboxes_out.append(
                    HitBox(
                        x0=int(tx + 58),
                        y0=int(y - row_h + 2),
                        x1=int(tx + 58 + 56),
                        y1=int(y + 2),
                        payload={"kind": "left_state", "row_kind": bind_kind2, "row_id": bind_id2, "row_idx": int(idx), "state": "last_hold_pulse"},
                    )
                )

            if kind in ("axis", "sig_axis", "sig_axis2", "kbd_axis", "mouse_motion"):
                inv_dev: str | None = None
                inv_id: int | None = None
                if kind == "axis":
                    inv_dev = "joystick"
                    inv_id = int(rid)
                    v_raw = float(axes_now.get(int(rid), 0.0))
                    v = float(v_raw)
                    if _axis_invert_pref(cfg, device=str(inv_dev), axis_id=int(inv_id)):
                        v = -float(v)
                    _snap_to_axis_stats(int(rid), float(v))
                    st = axis_stats.get(int(rid), AxisStats())
                elif kind == "kbd_axis":
                    rid_k = int(rid.get("id", 0)) if isinstance(rid, dict) else int(rid)
                    inv_dev = "keyboard"
                    inv_id = int(rid_k)
                    v_raw = float(kbd_axes_now.get(int(rid_k), 0.0))
                    v = float(v_raw)
                    if _axis_invert_pref(cfg, device=str(inv_dev), axis_id=int(inv_id)):
                        v = -float(v)
                    st = AxisStats(v_min=float(v), v_max=float(v), seen_min=True, seen_max=True)
                elif kind == "mouse_motion":
                    rid_m = int(rid.get("id", 0)) if isinstance(rid, dict) else int(rid)
                    inv_dev = "mouse"
                    inv_id = int(rid_m)
                    v_raw = float(mouse_axes_now.get(int(rid_m), 0.0))
                    v = float(v_raw)
                    if _axis_invert_pref(cfg, device=str(inv_dev), axis_id=int(inv_id)):
                        v = -float(v)
                    st = AxisStats(v_min=float(v), v_max=float(v), seen_min=True, seen_max=True)
                else:
                    v = float(_signal_value_for_signal_key(cfg, str(rid), axes_now=axes_now, now_ns=int(now_ns)))
                    st = AxisStats(v_min=float(v), v_max=float(v), seen_min=True, seen_max=True)
                x_bar = x_led0 + len(led_defs) * led_col_w + 12
                y_bar = y - int(row_h // 2) - 1
                h_bar = 6

                # Bar outline
                _draw_rect_px(x_bar, y_bar, bar_w, h_bar, (0.10, 0.10, 0.10, 1.0))
                _draw_rect_px(x_bar, y_bar, bar_w, 1, (0.30, 0.30, 0.30, 1.0))
                _draw_rect_px(x_bar, y_bar + h_bar - 1, bar_w, 1, (0.30, 0.30, 0.30, 1.0))

                def _x_at(val: float) -> int:
                    vv = max(-1.0, min(1.0, float(val)))
                    return int(x_bar + (vv + 1.0) * 0.5 * float(bar_w))

                # End ticks (dead ends): red if never reached.
                col_end_ok = (1.0, 1.0, 1.0, 1.0)
                col_end_bad = (1.0, 0.3, 0.3, 1.0)
                _draw_rect_px(_x_at(-1.0), y_bar - 2, 1, h_bar + 4, col_end_ok if st.seen_min else col_end_bad)
                _draw_rect_px(_x_at(+1.0), y_bar - 2, 1, h_bar + 4, col_end_ok if st.seen_max else col_end_bad)

                # Center tick
                _draw_rect_px(_x_at(0.0), y_bar - 2, 1, h_bar + 4, (1.0, 1.0, 1.0, 1.0))

                # Deadzone region around center (light gray)
                dz = 0.10
                x_dz0 = _x_at(-dz)
                x_dz1 = _x_at(+dz)
                _draw_rect_px(min(x_dz0, x_dz1), y_bar, abs(x_dz1 - x_dz0), h_bar, (0.20, 0.20, 0.20, 1.0))

                # Calibration ticks for axis rows: trim center and deadzone boundaries.
                if inv_dev is not None and inv_id is not None and kind in ("axis", "kbd_axis", "mouse_motion"):
                    c = _axis_calib_pref(cfg, device=str(inv_dev), axis_id=int(inv_id))
                    cap_min = c.get("cap_min")
                    cap_max = c.get("cap_max")
                    trim = c.get("trim")
                    deadzone = c.get("deadzone")

                    lo = None
                    hi = None
                    try:
                        lo = float(cap_min)
                        hi = float(cap_max)
                    except Exception:
                        lo, hi = None, None
                    if lo is None or hi is None or float(hi) <= float(lo) + 1e-6:
                        lo, hi = -1.0, 1.0
                    mid = (float(lo) + float(hi)) * 0.5
                    span = max(1e-6, (float(hi) - float(lo)) * 0.5)

                    t = None
                    try:
                        t = float(trim)
                    except Exception:
                        t = None
                    if t is not None:
                        raw_trim = float(mid + float(t) * float(span))
                        _draw_rect_px(_x_at(raw_trim), y_bar - 2, 1, h_bar + 4, (1.0, 0.55, 0.25, 1.0))

                        try:
                            dz2 = float(deadzone)
                        except Exception:
                            dz2 = 0.0
                        if dz2 > 0.0:
                            d_raw = float(dz2) * float(span)
                            _draw_rect_px(_x_at(raw_trim - d_raw), y_bar - 2, 1, h_bar + 4, (0.25, 0.55, 1.0, 1.0))
                            _draw_rect_px(_x_at(raw_trim + d_raw), y_bar - 2, 1, h_bar + 4, (0.25, 0.55, 1.0, 1.0))

                # Observed min/max ticks
                _draw_rect_px(_x_at(st.v_min), y_bar - 1, 1, h_bar + 2, (0.25, 1.0, 0.25, 1.0))
                _draw_rect_px(_x_at(st.v_max), y_bar - 1, 1, h_bar + 2, (0.25, 1.0, 0.25, 1.0))

                # Current value tick
                _draw_rect_px(_x_at(v), y_bar - 3, 2, h_bar + 6, (1.0, 1.0, 0.25, 1.0))

                # Per-axis invert toggle (stored as device preference).
                if inv_dev is not None and inv_id is not None and kind in ("axis", "kbd_axis", "mouse_motion"):
                    inv_on = bool(_axis_invert_pref(cfg, device=str(inv_dev), axis_id=int(inv_id)))
                    # Controls to the right of the axis bar.
                    x_ctl0 = int(min(int(left_x0 + left_w - 156), int(x_bar + bar_w + 10)))
                    x_inv = int(x_ctl0)

                    # Session state (colors) for TRIM/CAP/DED.
                    sess = None
                    if state.calib_sessions is not None:
                        sess = state.calib_sessions.get(f"{str(inv_dev)}:{int(inv_id)}")
                    mode = str(sess.get("mode", "")) if isinstance(sess, dict) else ""
                    start_ns = int(sess.get("start_ns", 0)) if isinstance(sess, dict) else 0
                    now2 = int(now_ns)
                    el_s = float(max(0.0, (int(now2) - int(start_ns)) * 1e-9)) if start_ns else 0.0
                    red_s = float(sess.get("red_s", 5.0)) if isinstance(sess, dict) else 5.0
                    total_s = float(sess.get("total_s", 10.0)) if isinstance(sess, dict) else 10.0
                    in_red = bool(start_ns and el_s < red_s)
                    in_green = bool(start_ns and (el_s >= red_s) and (el_s < total_s))
                    col_red = (255, 80, 80)
                    col_green = (120, 255, 120)
                    col_dim = (160, 160, 160)

                    _draw_text_px(int(x_inv), int(y), "INV" if inv_on else "inv", (120, 255, 120) if inv_on else col_dim)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(x_inv - 2),
                            y0=int(y - row_h + 2),
                            x1=int(x_inv + 26),
                            y1=int(y + 2),
                            payload={"kind": "axis_invert", "device": str(inv_dev), "axis": int(inv_id)},
                        )
                    )

                    # TRIM
                    x_trim = int(x_inv + 30)
                    lbl_trim = "TRIM" if mode == "trim" else "trim"
                    col_trim = col_red if (mode == "trim" and in_red) else (col_green if (mode == "trim" and in_green) else col_dim)
                    _draw_text_px(int(x_trim), int(y), lbl_trim, col_trim)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(x_trim - 2),
                            y0=int(y - row_h + 2),
                            x1=int(x_trim + 30),
                            y1=int(y + 2),
                            payload={"kind": "axis_trim", "device": str(inv_dev), "axis": int(inv_id)},
                        )
                    )

                    # CAP
                    x_cap = int(x_trim + 34)
                    lbl_cap = "CAP" if mode == "cap" else "cap"
                    col_cap = col_red if (mode == "cap" and in_red) else (col_green if (mode == "cap" and in_green) else col_dim)
                    _draw_text_px(int(x_cap), int(y), lbl_cap, col_cap)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(x_cap - 2),
                            y0=int(y - row_h + 2),
                            x1=int(x_cap + 28),
                            y1=int(y + 2),
                            payload={"kind": "axis_cap", "device": str(inv_dev), "axis": int(inv_id)},
                        )
                    )

                    # DED
                    x_ded = int(x_cap + 32)
                    lbl_ded = "DED" if mode == "ded" else "ded"
                    col_ded = col_red if (mode == "ded" and start_ns and el_s < total_s) else col_dim
                    _draw_text_px(int(x_ded), int(y), lbl_ded, col_ded)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(x_ded - 2),
                            y0=int(y - row_h + 2),
                            x1=int(x_ded + 28),
                            y1=int(y + 2),
                            payload={"kind": "axis_ded", "device": str(inv_dev), "axis": int(inv_id)},
                        )
                    )

                    # RST
                    x_rst = int(x_ded + 32)
                    _draw_text_px(int(x_rst), int(y), "rst", col_dim)
                    hitboxes_out.append(
                        HitBox(
                            x0=int(x_rst - 2),
                            y0=int(y - row_h + 2),
                            x1=int(x_rst + 28),
                            y1=int(y + 2),
                            payload={"kind": "axis_rst", "device": str(inv_dev), "axis": int(inv_id)},
                        )
                    )

                # Clickable hitbox for the axis bar
                hitboxes_out.append(
                    HitBox(
                        x0=int(x_bar),
                        y0=int(y_bar - 4),
                        x1=int(x_bar + bar_w),
                        y1=int(y_bar + h_bar + 4),
                        payload={
                            "kind": "left_axis",
                            "row_kind": "kbd_axis" if kind == "kbd_axis" else ("mouse_motion" if kind == "mouse_motion" else kind),
                            "row_id": int(rid.get("id", 0)) if isinstance(rid, dict) else rid,
                            "row_idx": int(idx),
                            "state": "axis",
                        },
                    )
                )

        # Right signals list (+ NEW) one column + mini history + arg LEDs.
        sigs = signal_ids(cfg)
        sig_items = ["NEW"] + sigs
        if sig_items:
            state.sel_signal_idx = int(max(0, min(int(state.sel_signal_idx), len(sig_items) - 1)))

        max_right_rows = int(ui_tables.max_visible_rows(right_geom))
        r_first, r_last, _r_vis = ui_tables.visible_window(rows_total=len(sig_items), sel_idx=int(state.sel_signal_idx), g=right_geom, scroll=right_scroll, center=True)

        # Geometry inside right pane
        inner_w = max(1, int(right_w - pad * 2))
        hist_w = min(56, max(22, int(inner_w * 0.30)))
        hist_h = 10
        max_arg_leds = 12
        arg_led_gap = 8
        arg_w = int(max_arg_leds * arg_led_gap)
        op_w = 26
        ch_w = 34
        name_w = max(8, int(inner_w - hist_w - arg_w - op_w - ch_w - 18))

        # Update history at ~30Hz for visible signals.
        nonlocal last_hist_ns
        do_hist = (int(now_ns) - int(last_hist_ns)) >= int(hist_period_ns)
        if do_hist:
            last_hist_ns = int(now_ns)

        def _push_hist(sid: str, v: float) -> None:
            buf = sig_history.get(str(sid))
            if buf is None:
                buf = []
                sig_history[str(sid)] = buf
            buf.append(float(max(-1.0, min(1.0, float(v)))))
            if len(buf) > int(hist_len):
                del buf[: len(buf) - int(hist_len)]

        def _draw_hist(x: int, y: int, sid: str) -> None:
            # Background
            _draw_rect_px(int(x), int(y - hist_h + 2), int(hist_w), int(hist_h), (0.08, 0.08, 0.10, 1.0))
            _draw_rect_px(int(x), int(y - hist_h + 2), int(hist_w), 1, (0.25, 0.25, 0.25, 1.0))
            _draw_rect_px(int(x), int(y + 1), int(hist_w), 1, (0.25, 0.25, 0.25, 1.0))

            buf = sig_history.get(str(sid), [])
            if not buf:
                return
            n = min(int(hist_w), len(buf))
            # Draw right-aligned so newest is at the right edge.
            start = len(buf) - n
            for i in range(n):
                vv = float(buf[start + i])
                yy = int((hist_h - 3) * (1.0 - ((vv + 1.0) * 0.5)))
                px = int(x + i)
                py = int((y - hist_h + 3) + yy)
                # Value point
                _draw_rect_px(px, py, 1, 1, (0.25, 1.0, 0.25, 1.0))
            # Midline
            _draw_rect_px(int(x), int(y - hist_h + 3 + (hist_h // 2)), int(hist_w), 1, (0.18, 0.18, 0.18, 1.0))

        def _draw_hist_w(x: int, y: int, key: str, w: int) -> None:
            ww = int(max(6, w))
            _draw_rect_px(int(x), int(y - hist_h + 2), int(ww), int(hist_h), (0.08, 0.08, 0.10, 1.0))
            _draw_rect_px(int(x), int(y - hist_h + 2), int(ww), 1, (0.25, 0.25, 0.25, 1.0))
            _draw_rect_px(int(x), int(y + 1), int(ww), 1, (0.25, 0.25, 0.25, 1.0))
            buf = sig_history.get(str(key), [])
            if not buf:
                return
            n = min(int(ww), len(buf))
            start = len(buf) - n
            for i in range(n):
                vv = float(buf[start + i])
                yy = int((hist_h - 3) * (1.0 - ((vv + 1.0) * 0.5)))
                px = int(x + i)
                py = int((y - hist_h + 3) + yy)
                _draw_rect_px(px, py, 1, 1, (0.25, 1.0, 0.25, 1.0))
            _draw_rect_px(int(x), int(y - hist_h + 3 + (hist_h // 2)), int(ww), 1, (0.18, 0.18, 0.18, 1.0))

        def _signal_arg_counts(cfg: dict[str, Any], sid: str) -> tuple[int, int]:
            ctrl = controller_block(cfg)
            sigs_d = ctrl.get("signals")
            if not isinstance(sigs_d, dict):
                return 1, 0
            node = sigs_d.get(str(sid))
            if not isinstance(node, dict):
                return 1, 0
            req = int(max(0, _op_required_args(node)))

            # Prefer the new args[] representation.
            args = node.get("args")
            if isinstance(args, list):
                linked = 0
                for i in range(min(int(req), len(args))):
                    if args[i] is not None and args[i] != "":
                        linked += 1
                return int(req), int(linked)

            # Back-compat: infer from legacy fields.
            op = str(node.get("op", ""))
            if op == "const":
                return 0, 0
            if op in ("add", "sub"):
                linked = 0
                if isinstance(node.get("a"), str) and str(node.get("a")):
                    linked += 1
                if isinstance(node.get("b"), str) and str(node.get("b")):
                    linked += 1
                return 2, linked
            if op == "raw":
                return 1, 1 if str(node.get("feature", "")) else 0
            if op in ("scale", "clamp", "deadzone"):
                return 1, 1 if isinstance(node.get("in"), str) and str(node.get("in")) else 0
            if op == "kernel":
                return 1, 1 if ("signal_id" in node) else 0
            return max(1, int(req)), 0

        for idx in range(int(r_first), int(r_last)):
            y = int(ui_tables.row_baseline_y(g=right_geom, visible_idx=(int(idx) - int(r_first))))
            label = str(sig_items[int(idx)])
            x0 = int(right_x0 + pad)
            cell_w = int(right_w - pad * 2)
            _draw_rect_px(x0, y - row_h + 2, cell_w, row_h, (0.05, 0.05, 0.07, 1.0))
            if idx == int(state.sel_signal_idx):
                _draw_rect_px(x0, y - row_h + 2, cell_w, 1, (1.0, 1.0, 1.0, 1.0))

            # Sample kernel value for history only when op:'kernel'.
            if label != "NEW":
                dim = _get_signal_dim(cfg, str(label))
                v_k = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(label), "x") if dim == 2 else str(label), axes_now=axes_now, now_ns=int(now_ns)))
                if do_hist:
                    if dim == 2:
                        _push_hist(_signal_key_for_comp(str(label), "x"), float(v_k))
                        vy = float(_signal_value_for_signal_key(cfg, _signal_key_for_comp(str(label), "y"), axes_now=axes_now, now_ns=int(now_ns)))
                        _push_hist(_signal_key_for_comp(str(label), "y"), float(vy))
                    else:
                        _push_hist(label, float(v_k))

            prefix = ">" if idx == int(state.sel_signal_idx) else " "
            name = label
            if label != "NEW":
                name = _truncate_to_px(label, int(name_w))
            _draw_text_px(int(x0 + 4), int(y), f"{prefix} {name}")

            # Arg LEDs
            if label != "NEW":
                op_here = _get_signal_op(cfg, str(label))
                can_expand = str(op_here) == "2dsumclamp"
                req, linked = _signal_arg_counts(cfg, label)
                req = int(max(0, min(int(req), max_arg_leds)))
                linked = int(max(0, min(int(linked), max_arg_leds)))
                ax = int(x0 + 10 + name_w)
                ay = int(y - int(row_h // 2))
                for a_i in range(int(max_arg_leds)):
                    cx = int(ax + a_i * arg_led_gap + 4)
                    cy = int(ay)
                    active = (a_i < req)
                    on = active and (a_i < linked)
                    armed = (str(state.sel_arg_sid) == str(label)) and (int(state.sel_arg_idx) == int(a_i))
                    if on:
                        col = (0.25, 1.0, 0.25, 1.0)
                    elif active:
                        col = (0.20, 0.20, 0.20, 1.0)
                    else:
                        col = (0.10, 0.10, 0.12, 1.0)
                    if (not active) and can_expand:
                        col = (0.14, 0.14, 0.16, 1.0)
                    if armed:
                        _draw_circle_px(cx, cy, 5, (1.0, 1.0, 0.25, 1.0))
                    _draw_circle_px(cx, cy, 3, col)
                    if active or can_expand:
                        hitboxes_out.append(
                            HitBox(
                                x0=int(cx - 6),
                                y0=int(cy - 6),
                                x1=int(cx + 6),
                                y1=int(cy + 6),
                                payload={"kind": "sig_arg", "sid": str(label), "arg": int(a_i)},
                            )
                        )

            # History strip
            if label != "NEW":
                dim = _get_signal_dim(cfg, str(label))
                hx = int(x0 + 10 + name_w + arg_w)
                if dim == 2:
                    half = max(10, int((hist_w - 2) // 2))
                    _draw_hist_w(int(hx), int(y), _signal_key_for_comp(str(label), "x"), int(half))
                    _draw_hist_w(int(hx + half + 2), int(y), _signal_key_for_comp(str(label), "y"), int(half))
                else:
                    _draw_hist(int(hx), int(y), label)

            # Op + channel pickers
            if label != "NEW":
                op = _get_signal_op(cfg, str(label))
                op_lbl = "K" if op == "kernel" else ("2S" if op == "2dseek" else ("2F" if op == "2dflightstick" else ("2+" if op == "2dsumclamp" else "?")))
                x_op = int(x0 + 10 + name_w + arg_w + hist_w + 2)
                _draw_text_px(x_op, int(y), op_lbl)
                hitboxes_out.append(
                    HitBox(
                        x0=int(x_op - 4),
                        y0=int(y - row_h + 2),
                        x1=int(x_op + op_w),
                        y1=int(y + 2),
                        payload={"kind": "sig_op", "sid": str(label)},
                    )
                )

                ch = mapped_channel_for_signal(cfg, str(label))
                ch_lbl = "-" if ch is None else str(int(ch))
                x_ch = int(x_op + op_w)
                _draw_text_px(x_ch, int(y), f"ch:{ch_lbl}")
                hitboxes_out.append(
                    HitBox(
                        x0=int(x_ch - 2),
                        y0=int(y - row_h + 2),
                        x1=int(x_ch + ch_w),
                        y1=int(y + 2),
                        payload={"kind": "sig_chan", "sid": str(label)},
                    )
                )

            hitboxes_out.append(
                HitBox(
                    x0=int(x0),
                    y0=int(y - row_h + 2),
                    x1=int(x0 + cell_w),
                    y1=int(y + 2),
                    payload={"kind": "signal", "idx": int(idx), "sid": str(label)},
                )
            )

        # Footer/status
        _draw_text_px(pad, height - pad - 2, "ESC/Q/menu-btn: exit")

        mode_txt = str(getattr(state, "input_mode", "scan")).strip().lower() or "scan"
        mode_lbl = "SCAN" if mode_txt == "scan" else "ANNOUNCE"
        interest_lbl = "ALL" if announce_interest is None else str(len(announce_interest))
        mx0 = int(pad)
        my0 = int(height - pad - 56)
        _draw_text_px(mx0, my0, f"MODE: {mode_lbl}  (click to toggle)   announce-interest: {interest_lbl}")
        try:
            w_mode, _h_mode = font.size(f"MODE: {mode_lbl}  (click to toggle)   announce-interest: {interest_lbl}")
        except Exception:
            w_mode = 260
        hitboxes_out.append(
            HitBox(
                x0=int(mx0 - 4),
                y0=int(my0 - max(16, int(font.get_linesize()))),
                x1=int(mx0 + int(w_mode) + 6),
                y1=int(my0 + 6),
                payload={"kind": "mode_toggle"},
            )
        )

        if state.compile_ok:
            _draw_text_px(pad, height - pad - 38, "COMPILE: ACTIVE", (120, 255, 120))
        else:
            msg = state.compile_msg or "unknown error"
            _draw_text_px(pad, height - pad - 38, f"COMPILE: ERROR ({msg})", (255, 120, 120))
        if state.sel_row_kind and state.sel_state:
            _draw_text_px(pad, height - pad - 20, f"selected input: {state.sel_row_kind}.{state.sel_row_id}  state:{state.sel_state}")
        if state.naming:
            _draw_rect_px(right_x0 + 10, height - 70, right_w - 20, 56, (0.0, 0.0, 0.0, 0.85))
            _draw_rect_px(right_x0 + 10, height - 70, right_w - 20, 1, (1.0, 1.0, 1.0, 1.0))
            _draw_text_px(right_x0 + 16, height - 52, "NEW signal name:")
            _draw_text_px(right_x0 + 16, height - 32, state.name_buf + "_")
            if state.name_error:
                _draw_text_px(right_x0 + 16, height - 14, state.name_error, (255, 128, 128))

        glDisable(GL_BLEND)
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    while True:
        cfg = load_or_create_joystick_config("joystick.json")
        cfg = input_graph.ensure_controller_graph(cfg)

        # Calibration sessions update (may save+rebuild graphs when finishing).
        if state.calib_sessions is None:
            state.calib_sessions = {}

        def _raw_inv_for_axis(device: str, axis_id: int, *, axes_now: dict[int, float], kbd_axes_now: dict[int, float], mouse_axes_now: dict[int, float]) -> float:
            if device == "joystick":
                vv = float(axes_now.get(int(axis_id), 0.0))
                if _axis_invert_pref(cfg, device="joystick", axis_id=int(axis_id)):
                    vv = -float(vv)
                return float(vv)
            if device == "keyboard":
                vv = float(kbd_axes_now.get(int(axis_id), 0.0))
                if _axis_invert_pref(cfg, device="keyboard", axis_id=int(axis_id)):
                    vv = -float(vv)
                return float(vv)
            if device == "mouse":
                vv = float(mouse_axes_now.get(int(axis_id), 0.0))
                if _axis_invert_pref(cfg, device="mouse", axis_id=int(axis_id)):
                    vv = -float(vv)
                return float(vv)
            return 0.0

        nav = get_menu_nav(cfg)
        sc = get_menu_scroll(cfg)
        b_cancel = nav.get("cancel")

        # Collect events (mouse + keyboard naming + exits).
        pending_mouse_down: tuple[int, int] | None = None
        pending_wheel: int = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return
            if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                return

            if state.naming:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        cand = _safe_signal_id(state.name_buf)
                        if not cand:
                            state.name_error = "name required"
                        else:
                            sigs = signal_ids(cfg)
                            if cand in sigs:
                                state.name_error = "already exists"
                            else:
                                cfg2 = _ensure_signal(cfg, cand)
                                save_joystick_config(cfg2, "joystick.json")
                                _rebuild_graphs(int(_mono_ns()))
                                state.naming = False
                                state.name_buf = ""
                                state.name_error = ""
                    elif event.key == pygame.K_ESCAPE:
                        state.naming = False
                        state.name_buf = ""
                        state.name_error = ""
                    elif event.key == pygame.K_BACKSPACE:
                        state.name_buf = state.name_buf[:-1]
                    else:
                        ch = getattr(event, "unicode", "")
                        if isinstance(ch, str) and ch:
                            if ch.isprintable() and ch not in "\r\n\t":
                                state.name_buf += ch
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) in (1,):
                try:
                    pending_mouse_down = (int(event.pos[0]), int(event.pos[1]))
                except Exception:
                    pending_mouse_down = None

            if event.type == pygame.MOUSEWHEEL and not state.naming:
                try:
                    pending_wheel += int(getattr(event, "y", 0))
                except Exception:
                    pending_wheel += 0

        # Effective input spec set for this frame.
        try:
            if input_interest is not None:
                allow, deny = input_interest.get_effective_set()
            else:
                allow, deny = (None, set())
        except Exception:
            allow, deny = (None, set())

        deny = set(deny or set())
        if allow is not None:
            allow = set(allow) - deny

        # Joystick snapshot (poll only requested ids in announce-mode).
        if joystick is not None:
            try:
                if allow is None:
                    axes_now, buttons_now, hats_now = poll_joystick_snapshot(joystick)
                else:
                    HARD_HAT_AXIS_BASE = 50000
                    HARD_HAT_BTN_BASE = 51000
                    joy_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
                    joy_btns = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_BUTTON)}
                    phys_axes = sorted([int(a) for a in joy_axes if 0 <= int(a) < int(HARD_HAT_AXIS_BASE)])
                    phys_btns = sorted([int(b) for b in joy_btns if 0 <= int(b) < int(HARD_HAT_BTN_BASE)])
                    hat_ids: set[int] = set()
                    for ax_id in joy_axes:
                        if int(ax_id) >= int(HARD_HAT_AXIS_BASE) and int(ax_id) < int(HARD_HAT_BTN_BASE):
                            hat_ids.add(int((int(ax_id) - int(HARD_HAT_AXIS_BASE)) // 2))
                    for bid in joy_btns:
                        if int(bid) >= int(HARD_HAT_BTN_BASE):
                            hat_ids.add(int((int(bid) - int(HARD_HAT_BTN_BASE)) // 8))
                    axes_now, buttons_now, hats_now = poll_joystick_snapshot(
                        joystick,
                        axes_ids=phys_axes,
                        button_ids=phys_btns,
                        hat_ids=sorted(hat_ids) if hat_ids else [],
                    )
            except Exception:
                axes_now, buttons_now, hats_now = {}, set(), {}
        else:
            axes_now, buttons_now, hats_now = {}, set(), {}

        # Keyboard sources (poll only if allowed).
        pressed = None
        want_kbd_axes = None
        want_kbd_keys = None
        if allow is None:
            want_kbd_axes = None
            want_kbd_keys = None
        else:
            want_kbd_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
            want_kbd_keys = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_KEY)}
        if allow is None or (want_kbd_axes and len(want_kbd_axes)) or (want_kbd_keys and len(want_kbd_keys)):
            try:
                pressed = pygame.key.get_pressed()
            except Exception:
                pressed = None

        def _is_down(k: int) -> int:
            if pressed is None:
                return 0
            try:
                return 1 if bool(pressed[int(k)]) else 0
            except Exception:
                return 0

        # Build keyboard axes (four virtual axes: WASD and arrows as signed pairs).
        kbd_axes_now: dict[int, float] = {}
        if pressed is not None:
            a = _is_down(pygame.K_d) - _is_down(pygame.K_a)
            b = _is_down(pygame.K_w) - _is_down(pygame.K_s)
            c = _is_down(pygame.K_RIGHT) - _is_down(pygame.K_LEFT)
            d = _is_down(pygame.K_UP) - _is_down(pygame.K_DOWN)
            if allow is None or (want_kbd_axes and 0 in want_kbd_axes):
                kbd_axes_now[0] = float(a)
            if allow is None or (want_kbd_axes and 1 in want_kbd_axes):
                kbd_axes_now[1] = float(b)
            if allow is None or (want_kbd_axes and 2 in want_kbd_axes):
                kbd_axes_now[2] = float(c)
            if allow is None or (want_kbd_axes and 3 in want_kbd_axes):
                kbd_axes_now[3] = float(d)

        # Mouse motion axes (dx,dy) from kernel-peeked values if available.
        mouse_axes_now: dict[int, float] = {}
        if _sigk is not None:
            try:
                fr_m0 = GP_SignalFrame()
                fr_m1 = GP_SignalFrame()
                sid_m0 = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), 0))
                sid_m1 = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), 1))
                ok0 = int(_sigk.gp_sigk_peek(int(_mono_ns()), int(sid_m0), fr_m0))
                ok1 = int(_sigk.gp_sigk_peek(int(_mono_ns()), int(sid_m1), fr_m1))
                mouse_axes_now[0] = float(fr_m0.value) if ok0 else 0.0
                mouse_axes_now[1] = float(fr_m1.value) if ok1 else 0.0
            except Exception:
                mouse_axes_now[0] = 0.0
                mouse_axes_now[1] = 0.0

        # Update calibration sessions after inputs are available.
        if state.calib_sessions:
            done_keys: list[str] = []
            for sk, sess in list(state.calib_sessions.items()):
                if not isinstance(sess, dict):
                    done_keys.append(str(sk))
                    continue
                mode = str(sess.get("mode", ""))
                dev = str(sess.get("device", ""))
                try:
                    axis_id = int(sess.get("axis", 0))
                except Exception:
                    axis_id = 0
                try:
                    start_ns = int(sess.get("start_ns", 0))
                except Exception:
                    start_ns = 0
                now_ns2 = int(_mono_ns())
                el_s = float(max(0.0, (int(now_ns2) - int(start_ns)) * 1e-9))

                v_raw_inv = float(_raw_inv_for_axis(dev, int(axis_id), axes_now=axes_now, kbd_axes_now=kbd_axes_now, mouse_axes_now=mouse_axes_now))

                if mode in ("trim", "cap"):
                    red_s = float(sess.get("red_s", 5.0))
                    total_s = float(sess.get("total_s", 10.0))
                    # Record ranges
                    if el_s < red_s:
                        sess.setdefault("red_samples", []).append(float(v_raw_inv))
                        sess["red_min"] = float(min(float(sess.get("red_min", v_raw_inv)), float(v_raw_inv)))
                        sess["red_max"] = float(max(float(sess.get("red_max", v_raw_inv)), float(v_raw_inv)))
                    elif el_s < total_s:
                        sess["green_min"] = float(min(float(sess.get("green_min", v_raw_inv)), float(v_raw_inv)))
                        sess["green_max"] = float(max(float(sess.get("green_max", v_raw_inv)), float(v_raw_inv)))
                    else:
                        # Finalize
                        if mode == "trim":
                            red_samples = sess.get("red_samples")
                            if not isinstance(red_samples, list) or not red_samples:
                                done_keys.append(str(sk))
                                continue
                            avg_red = float(sum([float(x) for x in red_samples]) / max(1, len(red_samples)))
                            lo = float(min(float(sess.get("red_min", avg_red)), float(sess.get("green_min", avg_red))))
                            hi = float(max(float(sess.get("red_max", avg_red)), float(sess.get("green_max", avg_red))))
                            # Ensure we include the trim samples in the cap range.
                            if float(hi) <= float(lo) + 1e-6:
                                lo, hi = -1.0, 1.0
                            mid = (float(lo) + float(hi)) * 0.5
                            span = max(1e-6, (float(hi) - float(lo)) * 0.5)
                            trim = float((float(avg_red) - float(mid)) / float(span))
                            trim = float(max(-0.95, min(0.95, float(trim))))

                            # Deadzone must include the observed trim jitter during the red period.
                            jitter = 0.0
                            for s in red_samples:
                                jitter = float(max(float(jitter), abs((float(s) - float(avg_red)) / float(span))))
                            cur = _axis_calib_pref(cfg, device=str(dev), axis_id=int(axis_id))
                            cur_dz = 0.0
                            try:
                                cur_dz = float(cur.get("deadzone", 0.0))
                            except Exception:
                                cur_dz = 0.0
                            dz = float(max(float(cur_dz), float(min(0.95, float(jitter)))))

                            cfg2 = _write_axis_calib(
                                cfg,
                                device=str(dev),
                                axis_id=int(axis_id),
                                calib={"cap_min": float(lo), "cap_max": float(hi), "trim": float(trim), "deadzone": float(dz)},
                            )
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                            cfg = cfg2
                        else:
                            # CAP: set extents to observed range; preserve existing trim in raw space if present.
                            lo = float(min(float(sess.get("red_min", v_raw_inv)), float(sess.get("green_min", v_raw_inv))))
                            hi = float(max(float(sess.get("red_max", v_raw_inv)), float(sess.get("green_max", v_raw_inv))))
                            if float(hi) <= float(lo) + 1e-6:
                                lo, hi = -1.0, 1.0

                            cur = _axis_calib_pref(cfg, device=str(dev), axis_id=int(axis_id))
                            old_lo = cur.get("cap_min")
                            old_hi = cur.get("cap_max")
                            old_trim = cur.get("trim")
                            raw_trim = 0.0
                            try:
                                olo = float(old_lo)
                                ohi = float(old_hi)
                                ot = float(old_trim)
                                if float(ohi) > float(olo) + 1e-6:
                                    omid = (float(olo) + float(ohi)) * 0.5
                                    ospan = (float(ohi) - float(olo)) * 0.5
                                    raw_trim = float(omid + float(ot) * float(ospan))
                            except Exception:
                                raw_trim = 0.0

                            # Recompute trim in new cap space.
                            mid = (float(lo) + float(hi)) * 0.5
                            span = max(1e-6, (float(hi) - float(lo)) * 0.5)
                            new_trim = float((float(raw_trim) - float(mid)) / float(span))
                            new_trim = float(max(-0.95, min(0.95, float(new_trim))))

                            dz2 = None
                            try:
                                dz2 = float(cur.get("deadzone"))
                            except Exception:
                                dz2 = None

                            cfg2 = _write_axis_calib(
                                cfg,
                                device=str(dev),
                                axis_id=int(axis_id),
                                calib={"cap_min": float(lo), "cap_max": float(hi), "trim": float(new_trim), "deadzone": dz2},
                            )
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                            cfg = cfg2

                        done_keys.append(str(sk))
                elif mode == "ded":
                    total_s = float(sess.get("total_s", 5.0))
                    # Collect max deviation from trim point.
                    cur = _axis_calib_pref(cfg, device=str(dev), axis_id=int(axis_id))
                    lo = cur.get("cap_min")
                    hi = cur.get("cap_max")
                    span = 1.0
                    mid = 0.0
                    try:
                        lo_f = float(lo)
                        hi_f = float(hi)
                        if float(hi_f) > float(lo_f) + 1e-6:
                            mid = (float(lo_f) + float(hi_f)) * 0.5
                            span = max(1e-6, (float(hi_f) - float(lo_f)) * 0.5)
                    except Exception:
                        span = 1.0
                        mid = 0.0
                    trim = 0.0
                    try:
                        trim = float(cur.get("trim", 0.0))
                    except Exception:
                        trim = 0.0
                    raw_trim = float(mid + float(trim) * float(span))
                    devn = abs((float(v_raw_inv) - float(raw_trim)) / float(span))
                    sess["max_dev"] = float(max(float(sess.get("max_dev", 0.0)), float(devn)))

                    if el_s >= float(total_s):
                        dz = float(min(0.95, float(sess.get("max_dev", 0.0))))
                        cfg2 = _write_axis_calib(cfg, device=str(dev), axis_id=int(axis_id), calib={"deadzone": float(dz), "cap_min": lo, "cap_max": hi, "trim": cur.get("trim")})
                        save_joystick_config(cfg2, "joystick.json")
                        _rebuild_graphs(int(_mono_ns()))
                        cfg = cfg2
                        done_keys.append(str(sk))
                else:
                    done_keys.append(str(sk))

            for k in done_keys:
                try:
                    state.calib_sessions.pop(str(k), None)
                except Exception:
                    pass

        cancel_edge = nav_edge(b_cancel, axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev)
        if cancel_edge:
            return

        now_ns = _mono_ns()

        # Ensure we have a compile status and current final graph.
        if int(state.compile_last_ns) == 0:
            _rebuild_graphs(int(now_ns))

        # Apply menu scroll (line/page) and wheel scroll (pane-aware by mouse position)
        # NOTE: bindings for menu_scroll can be added later; we just support the frame.
        delta_lines = 0
        delta_pages = 0
        if sc and not state.naming:
            if nav_edge(sc.get("scroll_up"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                delta_lines -= 1
            if nav_edge(sc.get("scroll_down"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                delta_lines += 1
            if nav_edge(sc.get("page_up"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                delta_pages -= 1
            if nav_edge(sc.get("page_down"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                delta_pages += 1

        if pending_wheel:
            try:
                mx, my = pygame.mouse.get_pos()
            except Exception:
                mx, my = 0, 0
            left_w_guess = int(width * 0.72)
            if int(mx) < int(left_w_guess):
                # Wheel up should scroll up (decrease first_idx)
                # pygame wheel y>0 means up
                # ScrollModel delta>0 moves window down, so invert.
                total_rows = int(len(_build_left_rows(cfg)))
                left_geom = ui_tables.TableGeom(x0=0, y0=0, w=int(width * 0.72), h=int(height), pad=10, header_h=max(18, int(font.get_linesize()) + 2), row_h=max(16, int(font.get_linesize()) + 2))
                max_left_rows = int(ui_tables.max_visible_rows(left_geom))
                left_scroll.scroll_lines(delta=int(-pending_wheel), total=int(total_rows), max_visible=int(max_left_rows))
            else:
                sigs2_total = 1 + len(signal_ids(cfg))
                right_geom = ui_tables.TableGeom(x0=int(width * 0.72), y0=0, w=int(width - int(width * 0.72)), h=int(height), pad=10, header_h=max(18, int(font.get_linesize()) + 2), row_h=max(16, int(font.get_linesize()) + 2))
                max_right_rows = int(ui_tables.max_visible_rows(right_geom))
                right_scroll.scroll_lines(delta=int(-pending_wheel), total=int(sigs2_total), max_visible=int(max_right_rows))

        # Apply menu scroll after wheel so both can work.
        if int(delta_lines) or int(delta_pages):
            try:
                mx2, _my2 = pygame.mouse.get_pos()
            except Exception:
                mx2 = 0
            left_w_guess2 = int(width * 0.72)
            if int(mx2) < int(left_w_guess2):
                total_rows = int(len(_build_left_rows(cfg)))
                left_geom = ui_tables.TableGeom(x0=0, y0=0, w=int(width * 0.72), h=int(height), pad=10, header_h=max(18, int(font.get_linesize()) + 2), row_h=max(16, int(font.get_linesize()) + 2))
                max_left_rows = int(ui_tables.max_visible_rows(left_geom))
                if int(delta_lines):
                    left_scroll.scroll_lines(delta=int(delta_lines), total=int(total_rows), max_visible=int(max_left_rows))
                if int(delta_pages):
                    left_scroll.scroll_pages(delta_pages=int(delta_pages), total=int(total_rows), max_visible=int(max_left_rows))
            else:
                sigs2_total = 1 + len(signal_ids(cfg))
                right_geom = ui_tables.TableGeom(x0=int(width * 0.72), y0=0, w=int(width - int(width * 0.72)), h=int(height), pad=10, header_h=max(18, int(font.get_linesize()) + 2), row_h=max(16, int(font.get_linesize()) + 2))
                max_right_rows = int(ui_tables.max_visible_rows(right_geom))
                if int(delta_lines):
                    right_scroll.scroll_lines(delta=int(delta_lines), total=int(sigs2_total), max_visible=int(max_right_rows))
                if int(delta_pages):
                    right_scroll.scroll_pages(delta_pages=int(delta_pages), total=int(sigs2_total), max_visible=int(max_right_rows))

        # Feed C kernel with snapshot-derived edges/values.
        if _sigk is not None and joystick is not None:
            allow_all = allow is None
            want_joy_buttons: set[int] | None = None
            want_joy_axes: set[int] | None = None
            want_hat_btns: set[int] | None = None
            want_hat_axes: set[int] | None = None
            want_mouse_motion: set[int] | None = None
            want_mouse_buttons: set[int] | None = None
            want_kbd_keys_eff: set[int] | None = None

            if not allow_all:
                want_joy_buttons = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_BUTTON)}
                want_joy_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
                want_hat_btns = {int(iid) for iid in (want_joy_buttons or set()) if int(iid) >= int(HARD_HAT_BTN_BASE)}
                want_hat_axes = {int(iid) for iid in (want_joy_axes or set()) if int(iid) >= int(HARD_HAT_AXIS_BASE)}
                want_mouse_motion = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_MOUSE) and int(k) == int(signal_kernel_api.GP_EV_MOUSE_MOTION)}
                want_mouse_buttons = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_MOUSE) and int(k) == int(signal_kernel_api.GP_EV_MOUSE_BUTTON)}
                want_kbd_keys_eff = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_KEY)}

            # Joystick physical button edges.
            evs: list[GP_InputEvent] = []
            if allow_all:
                try:
                    nb = int(joystick.get_numbuttons())
                except Exception:
                    nb = 0
                button_iter = [int(b) for b in range(max(0, nb))]
            else:
                button_iter = sorted([int(b) for b in (want_joy_buttons or set()) if int(b) >= 0 and int(b) < int(HARD_HAT_BTN_BASE)])

            for b in button_iter:
                spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(b))
                if spec in deny:
                    continue
                was_down = int(b) in buttons_prev
                is_down = int(b) in buttons_now
                if was_down == is_down:
                    continue
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                        kind=int(signal_kernel_api.GP_EV_BUTTON),
                        id=int(b),
                        v0=1.0 if is_down else 0.0,
                        v1=0.0,
                        flags=0,
                    )
                )
            if evs:
                arr_t = GP_InputEvent * len(evs)
                _sigk.gp_sigk_push_events(arr_t(*evs), int(len(evs)))

            # Hat direction buttons (8-way) as virtual joystick buttons with proper edges.
            try:
                nh = int(joystick.get_numhats())
            except Exception:
                nh = 0
            hat_btn_evs: list[GP_InputEvent] = []
            for h in range(max(0, int(nh))):
                cur_xy = hats_now.get(int(h), (0, 0))
                prev_xy = hats_prev.get(int(h), (0, 0))
                cur_dir = _dir_idx_8(int(cur_xy[0]), int(cur_xy[1]))
                prev_dir = _dir_idx_8(int(prev_xy[0]), int(prev_xy[1]))
                if cur_dir == prev_dir:
                    continue
                if prev_dir is not None:
                    prev_id = int(_hard_hat_btn_id(int(h), int(prev_dir)))
                    spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(prev_id))
                    if spec not in deny and (allow_all or int(prev_id) in (want_hat_btns or set())):
                        hat_btn_evs.append(
                            GP_InputEvent(
                                t_mono_ns=now_ns,
                                device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                kind=int(signal_kernel_api.GP_EV_BUTTON),
                                id=int(prev_id),
                                v0=0.0,
                                v1=0.0,
                                flags=0,
                            )
                        )
                if cur_dir is not None:
                    cur_id = int(_hard_hat_btn_id(int(h), int(cur_dir)))
                    spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(cur_id))
                    if spec not in deny and (allow_all or int(cur_id) in (want_hat_btns or set())):
                        hat_btn_evs.append(
                            GP_InputEvent(
                                t_mono_ns=now_ns,
                                device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                kind=int(signal_kernel_api.GP_EV_BUTTON),
                                id=int(cur_id),
                                v0=1.0,
                                v1=0.0,
                                flags=0,
                            )
                        )
            if hat_btn_evs:
                arr3_t = GP_InputEvent * len(hat_btn_evs)
                _sigk.gp_sigk_push_events(arr3_t(*hat_btn_evs), int(len(hat_btn_evs)))

            # Joystick physical axes (per-frame).
            if allow_all:
                try:
                    na = int(joystick.get_numaxes())
                except Exception:
                    na = 0
                axis_iter = [int(a) for a in range(max(0, int(na)))]
            else:
                axis_iter = sorted([int(a) for a in (want_joy_axes or set()) if 0 <= int(a) < int(HARD_HAT_AXIS_BASE)])

            if axis_iter:
                a_evs: list[GP_InputEvent] = []
                for a in axis_iter:
                    spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(a))
                    if spec in deny:
                        continue
                    vv = float(axes_now.get(int(a), 0.0))
                    a_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(a),
                            v0=float(vv),
                            v1=0.0,
                            flags=0,
                        )
                    )
                if a_evs:
                    arr2_t = GP_InputEvent * len(a_evs)
                    _sigk.gp_sigk_push_events(arr2_t(*a_evs), int(len(a_evs)))

            # Keyboard axes (from kbd_axes_now, already restricted above).
            try:
                k_evs: list[GP_InputEvent] = []
                for kid, kval in (kbd_axes_now or {}).items():
                    spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(kid))
                    if spec in deny:
                        continue
                    k_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                            kind=int(signal_kernel_api.GP_EV_AXIS),
                            id=int(kid),
                            v0=float(kval),
                            v1=0.0,
                            flags=0,
                        )
                    )
                if k_evs:
                    arrk_t = GP_InputEvent * len(k_evs)
                    _sigk.gp_sigk_push_events(arrk_t(*k_evs), int(len(k_evs)))
            except Exception:
                pass

            # Keyboard key edges.
            try:
                if pressed is not None:
                    if allow_all:
                        keys_down: set[int] = set()
                        try:
                            for i, v in enumerate(pressed):
                                if v:
                                    keys_down.add(int(i))
                        except Exception:
                            keys_down = set()
                        changed = (keys_down - keys_prev) | (keys_prev - keys_down)
                        if changed:
                            kkey_evs: list[GP_InputEvent] = []
                            for keycode in sorted(changed):
                                spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_KEY), int(keycode))
                                if spec in deny:
                                    continue
                                is_down = int(keycode) in keys_down
                                kkey_evs.append(
                                    GP_InputEvent(
                                        t_mono_ns=now_ns,
                                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                                        kind=int(signal_kernel_api.GP_EV_KEY),
                                        id=int(keycode),
                                        v0=1.0 if is_down else 0.0,
                                        v1=0.0,
                                        flags=0,
                                    )
                                )
                            if kkey_evs:
                                arrkk_t = GP_InputEvent * len(kkey_evs)
                                _sigk.gp_sigk_push_events(arrkk_t(*kkey_evs), int(len(kkey_evs)))
                        keys_prev = set(keys_down)
                    else:
                        want_keys = sorted(set(want_kbd_keys_eff or set()))
                        if want_keys:
                            keys_down = set(keys_prev)
                            kkey_evs: list[GP_InputEvent] = []
                            for keycode in want_keys:
                                spec = (int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_KEY), int(keycode))
                                if spec in deny:
                                    continue
                                try:
                                    is_down = 1 if bool(pressed[int(keycode)]) else 0
                                except Exception:
                                    is_down = 0
                                was_down = int(keycode) in keys_prev
                                if was_down == bool(is_down):
                                    continue
                                if is_down:
                                    keys_down.add(int(keycode))
                                else:
                                    keys_down.discard(int(keycode))
                                kkey_evs.append(
                                    GP_InputEvent(
                                        t_mono_ns=now_ns,
                                        device=int(signal_kernel_api.GP_DEV_KEYBOARD),
                                        kind=int(signal_kernel_api.GP_EV_KEY),
                                        id=int(keycode),
                                        v0=1.0 if is_down else 0.0,
                                        v1=0.0,
                                        flags=0,
                                    )
                                )
                            if kkey_evs:
                                arrkk_t = GP_InputEvent * len(kkey_evs)
                                _sigk.gp_sigk_push_events(arrkk_t(*kkey_evs), int(len(kkey_evs)))
                            keys_prev = set(keys_down)
            except Exception:
                pass

            # Mouse motion axes (dx/dy).
            try:
                m_evs: list[GP_InputEvent] = []
                for mid, mval in (mouse_axes_now or {}).items():
                    spec = (int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), int(mid))
                    if spec in deny:
                        continue
                    if (not allow_all) and int(mid) not in (want_mouse_motion or set()):
                        continue
                    m_evs.append(
                        GP_InputEvent(
                            t_mono_ns=now_ns,
                            device=int(signal_kernel_api.GP_DEV_MOUSE),
                            kind=int(signal_kernel_api.GP_EV_MOUSE_MOTION),
                            id=int(mid),
                            v0=float(mval),
                            v1=0.0,
                            flags=0,
                        )
                    )
                if m_evs:
                    arrm_t = GP_InputEvent * len(m_evs)
                    _sigk.gp_sigk_push_events(arrm_t(*m_evs), int(len(m_evs)))
            except Exception:
                pass

            # Mouse button edges.
            try:
                mpressed = None
                if allow_all or (want_mouse_buttons and len(want_mouse_buttons)):
                    try:
                        mpressed = pygame.mouse.get_pressed()
                    except Exception:
                        mpressed = None
                if mpressed is not None:
                    cur_down: set[int] = set()
                    try:
                        for i, v in enumerate(mpressed):
                            if v:
                                cur_down.add(int(i))
                    except Exception:
                        cur_down = set()

                    if allow_all:
                        changed = (cur_down - mouse_buttons_prev) | (mouse_buttons_prev - cur_down)
                        want_bids = sorted(changed)
                    else:
                        # Only consider edges for declared ids.
                        want_bids = sorted(set(want_mouse_buttons or set()))

                    if want_bids:
                        mb_evs: list[GP_InputEvent] = []
                        for bid in want_bids:
                            spec = (int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_BUTTON), int(bid))
                            if spec in deny:
                                continue
                            is_down = int(bid) in cur_down
                            was_down = int(bid) in mouse_buttons_prev
                            if allow_all:
                                if int(bid) not in (cur_down ^ mouse_buttons_prev):
                                    # changed set already filtered, but keep safe.
                                    pass
                            else:
                                if was_down == bool(is_down):
                                    continue
                            mb_evs.append(
                                GP_InputEvent(
                                    t_mono_ns=now_ns,
                                    device=int(signal_kernel_api.GP_DEV_MOUSE),
                                    kind=int(signal_kernel_api.GP_EV_MOUSE_BUTTON),
                                    id=int(bid),
                                    v0=1.0 if is_down else 0.0,
                                    v1=0.0,
                                    flags=0,
                                )
                            )
                        if mb_evs:
                            arrmb_t = GP_InputEvent * len(mb_evs)
                            _sigk.gp_sigk_push_events(arrmb_t(*mb_evs), int(len(mb_evs)))
                    mouse_buttons_prev = set(cur_down)
            except Exception:
                pass

            # Hat axes each frame as virtual joystick axes.
            try:
                nh2 = int(joystick.get_numhats())
            except Exception:
                nh2 = 0
            if nh2 > 0:
                h_evs: list[GP_InputEvent] = []
                for h in range(max(0, int(nh2))):
                    hx, hy = hats_now.get(int(h), (0, 0))
                    ax_id = int(_hard_hat_axis_id(int(h), "x"))
                    ay_id = int(_hard_hat_axis_id(int(h), "y"))
                    specx = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(ax_id))
                    specy = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(ay_id))
                    if specx not in deny and (allow_all or int(ax_id) in (want_hat_axes or set())):
                        h_evs.append(
                            GP_InputEvent(
                                t_mono_ns=now_ns,
                                device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                kind=int(signal_kernel_api.GP_EV_AXIS),
                                id=int(ax_id),
                                v0=float(int(hx)),
                                v1=0.0,
                                flags=0,
                            )
                        )
                    if specy not in deny and (allow_all or int(ay_id) in (want_hat_axes or set())):
                        h_evs.append(
                            GP_InputEvent(
                                t_mono_ns=now_ns,
                                device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                                kind=int(signal_kernel_api.GP_EV_AXIS),
                                id=int(ay_id),
                                v0=float(int(hy)),
                                v1=0.0,
                                flags=0,
                            )
                        )
                if h_evs:
                    arrh_t = GP_InputEvent * len(h_evs)
                    _sigk.gp_sigk_push_events(arrh_t(*h_evs), int(len(h_evs)))

            # Update virtual hardware buttons derived from signals (signal->button thresholding).
            _update_virtual_buttons_from_signals(cfg, int(now_ns), axes_now=axes_now)

            # Update virtual hat inputs derived from 2D signals (trinary axes + direction buttons).
            sig_hat_evs = _update_virtual_hats_from_signals(cfg, int(now_ns), axes_now=axes_now)
            if sig_hat_evs and (not allow_all):
                sig_hat_evs = [
                    e
                    for e in sig_hat_evs
                    if (int(e.device), int(e.kind), int(e.id)) in (allow or set()) and (int(e.device), int(e.kind), int(e.id)) not in deny
                ]
            if sig_hat_evs:
                arrs_t = GP_InputEvent * len(sig_hat_evs)
                _sigk.gp_sigk_push_events(arrs_t(*sig_hat_evs), int(len(sig_hat_evs)))

        # Hitboxes rebuilt each frame.
        hitboxes: list[HitBox] = []
        _draw_dense_ui(
            cfg=cfg,
            axes_now=axes_now,
            buttons_now=buttons_now,
            hats_now=hats_now,
            kbd_axes_now=kbd_axes_now,
            mouse_axes_now=mouse_axes_now,
            now_ns=int(now_ns),
            hitboxes_out=hitboxes,
        )

        # Handle click after drawing (hitboxes are ready).
        if pending_mouse_down is not None and not state.naming:
            mx, my = pending_mouse_down
            for hb in hitboxes:
                if hb.contains(int(mx), int(my)):
                    p = hb.payload
                    if p.get("kind") == "mode_toggle":
                        cur = str(getattr(state, "input_mode", "scan")).strip().lower() or "scan"
                        nxt = "announce" if cur == "scan" else "scan"
                        state.input_mode = str(nxt)  # type: ignore[attr-defined]
                        if input_interest is not None:
                            try:
                                input_interest.set_mode(str(nxt))
                            except Exception:
                                pass

                        # Clear any stale kernel state/edges so ANNOUNCE reflects *current* interest gating.
                        try:
                            _sigk.gp_sigk_reset()
                        except Exception:
                            pass
                        try:
                            buttons_prev = set()
                            hats_prev = {}
                            keys_prev = set()
                            mouse_buttons_prev = set()
                        except Exception:
                            pass
                        pending_mouse_down = None
                        break
                    if p.get("kind") in ("left_state", "left_axis"):
                        state.sel_row_kind = str(p.get("row_kind", ""))
                        row_id = p.get("row_id", 0)
                        if str(state.sel_row_kind) in ("axis", "button"):
                            try:
                                state.sel_row_id = int(row_id)
                            except Exception:
                                state.sel_row_id = 0
                        else:
                            state.sel_row_id = str(row_id)
                        state.sel_state = str(p.get("state", ""))
                        try:
                            state.sel_left_idx = int(p.get("row_idx", state.sel_left_idx))
                        except Exception:
                            state.sel_left_idx = int(state.sel_left_idx)

                        # If an arg slot is armed, bind immediately into args[] without changing op.
                        if str(state.sel_arg_sid):
                            cfg2 = _bind_row_to_signal_arg(
                                cfg,
                                sid=str(state.sel_arg_sid),
                                arg_idx=int(state.sel_arg_idx),
                                row_kind=str(state.sel_row_kind),
                                row_id=state.sel_row_id,
                                state_id=str(state.sel_state),
                            )
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                            state.sel_arg_sid = ""
                            state.sel_arg_idx = 0
                    elif p.get("kind") == "signal":
                        try:
                            state.sel_signal_idx = int(p.get("idx", 0))
                        except Exception:
                            state.sel_signal_idx = 0
                        sid = str(p.get("sid", ""))
                        if sid == "NEW":
                            state.naming = True
                            state.name_buf = ""
                            state.name_error = ""
                        else:
                            # Bind if a left input-state is selected.
                            if state.sel_row_kind in ("axis", "button") and state.sel_state and sid:
                                cfg2 = _bind_selected_input_to_signal(
                                    cfg,
                                    sid=str(sid),
                                    row_kind=str(state.sel_row_kind),
                                    row_id=int(state.sel_row_id),
                                    state_id=str(state.sel_state),
                                )
                                save_joystick_config(cfg2, "joystick.json")
                                _rebuild_graphs(int(_mono_ns()))
                    elif p.get("kind") == "sig_arg":
                        a_sid = str(p.get("sid", ""))
                        a_idx = int(p.get("arg", 0))
                        if str(state.sel_arg_sid) == a_sid and int(state.sel_arg_idx) == int(a_idx):
                            state.sel_arg_sid = ""
                            state.sel_arg_idx = 0
                        else:
                            state.sel_arg_sid = a_sid
                            state.sel_arg_idx = int(a_idx)
                    elif p.get("kind") == "sig_collapse":
                        sid = str(p.get("sid", ""))
                        if sid and state.collapsed_sigs is not None:
                            if sid in state.collapsed_sigs:
                                state.collapsed_sigs.discard(sid)
                            else:
                                state.collapsed_sigs.add(sid)
                    elif p.get("kind") == "hat_expand":
                        try:
                            h = int(p.get("hat", 0))
                        except Exception:
                            h = 0
                        if state.expanded_hats is not None:
                            if int(h) in state.expanded_hats:
                                state.expanded_hats.discard(int(h))
                            else:
                                state.expanded_hats.add(int(h))
                    elif p.get("kind") == "sig_op":
                        sid = str(p.get("sid", ""))
                        if sid:
                            cfg2 = _cycle_signal_op(cfg, sid)
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                    elif p.get("kind") == "sig_chan":
                        sid = str(p.get("sid", ""))
                        if sid:
                            cfg2 = _cycle_signal_channel(cfg, sid)
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                    elif p.get("kind") == "dev_toggle":
                        devk = str(p.get("dev", ""))
                        if devk and state.collapsed_devices is not None:
                            if devk in state.collapsed_devices:
                                state.collapsed_devices.discard(devk)
                            else:
                                state.collapsed_devices.add(devk)
                    elif p.get("kind") == "axis_invert":
                        dev = str(p.get("device", ""))
                        try:
                            ax = int(p.get("axis", 0))
                        except Exception:
                            ax = 0
                        if dev:
                            cfg2 = _toggle_axis_invert(cfg, device=str(dev), axis_id=int(ax))
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                    elif p.get("kind") in ("axis_trim", "axis_cap", "axis_ded"):
                        dev = str(p.get("device", ""))
                        try:
                            ax = int(p.get("axis", 0))
                        except Exception:
                            ax = 0
                        if dev and state.calib_sessions is not None:
                            key = f"{dev}:{int(ax)}"
                            mode = "trim" if p.get("kind") == "axis_trim" else ("cap" if p.get("kind") == "axis_cap" else "ded")
                            # Toggle start/stop
                            if key in state.calib_sessions and str(state.calib_sessions[key].get("mode", "")) == mode:
                                state.calib_sessions.pop(key, None)
                            else:
                                if mode == "ded":
                                    state.calib_sessions[key] = {"mode": mode, "device": str(dev), "axis": int(ax), "start_ns": int(_mono_ns()), "total_s": 5.0}
                                else:
                                    state.calib_sessions[key] = {"mode": mode, "device": str(dev), "axis": int(ax), "start_ns": int(_mono_ns()), "red_s": 5.0, "total_s": 10.0}
                    elif p.get("kind") == "axis_rst":
                        dev = str(p.get("device", ""))
                        try:
                            ax = int(p.get("axis", 0))
                        except Exception:
                            ax = 0
                        if dev:
                            cfg2 = _reset_axis_calib(cfg, device=str(dev), axis_id=int(ax))
                            save_joystick_config(cfg2, "joystick.json")
                            _rebuild_graphs(int(_mono_ns()))
                    break

        # Do not clear pulses here; the C controller engine already owns pulse
        # lifetime. Clearing in the UI loop was dropping data for other
        # consumers and clobbering short-lived edges before they could be
        # visualized.

        pygame.display.flip()
        # No frame cap: let the workbench run as fast as the host can draw so
        # LED/hist strips track kernel updates without an artificial governor.
        clock.tick(0)

        axes_prev, buttons_prev, hats_prev = dict(axes_now), set(buttons_now), dict(hats_now)
