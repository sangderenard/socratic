from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os

from c_physics import signal_kernel_api
from c_physics.signal_kernel_ctypes import GP_SignalFrame


@dataclass
class ControllerBackendStatus:
    ok: bool
    msg: str


class ControllerBackend:
    """Evaluate the compiled/final controller graph using the C signal kernel as the source of truth.

    This is the "next step" bridge:
    - Inputs are read from the C kernel via gp_sigk_peek_sel.
    - Node ops call the C sigop helpers when available.

    Today this evaluator runs in Python but only consumes kernel-peeked values.
    It is intentionally shaped so we can move it into C later.
    """

    def __init__(self, sigk) -> None:
        self._sigk = sigk
        self._final_path = "controller_graph_final.json"
        self._mtime_ns = 0
        self._graph: dict[str, Any] | None = None
        self._last_status = ControllerBackendStatus(ok=False, msg="not loaded")
        self._prev_active: set[int] = set()

    @property
    def status(self) -> ControllerBackendStatus:
        return self._last_status

    def _load_if_changed(self) -> None:
        try:
            st = os.stat(self._final_path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        except Exception:
            self._graph = None
            self._mtime_ns = 0
            self._last_status = ControllerBackendStatus(ok=False, msg=f"missing {self._final_path}")
            return

        if self._graph is not None and int(mtime_ns) == int(self._mtime_ns):
            return

        try:
            with open(self._final_path, "r", encoding="utf-8") as f:
                g = json.load(f)
            if not isinstance(g, dict):
                raise ValueError("final graph is not a JSON object")
            self._graph = g
            self._mtime_ns = int(mtime_ns)
            self._last_status = ControllerBackendStatus(ok=True, msg="loaded")
        except Exception as e:
            self._graph = None
            self._mtime_ns = int(mtime_ns)
            self._last_status = ControllerBackendStatus(ok=False, msg=f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)

    def step(self, *, now_ns: int, active_eps: float = 0.20) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
        """Compute channel outputs.

        Returns:
        - overrides_1d: {channel_index -> value} for dim==1 channels only (safe for flight_camera override)
        - channels_2d: {channel_index -> (x,y)} for dim==2 channels
        """
        self._load_if_changed()
        if self._graph is None:
            return {}, {}
        if self._sigk is None:
            self._last_status = ControllerBackendStatus(ok=False, msg="signal kernel unavailable")
            return {}, {}

        g = self._graph
        inputs = g.get("inputs")
        nodes = g.get("nodes")
        final_channels = g.get("final_channels")
        signals = g.get("signals")

        if not isinstance(inputs, list) or not isinstance(nodes, list) or not isinstance(final_channels, dict) or not isinstance(signals, dict):
            self._last_status = ControllerBackendStatus(ok=False, msg="invalid final graph schema")
            return {}, {}

        # 1) Read scalar inputs from kernel.
        in_vals: dict[int, float] = {}
        fr = GP_SignalFrame()

        def _apply_input_calib(v_raw: float, inp: dict[str, Any], *, kind: str) -> float:
            v = float(v_raw)
            if str(kind) not in ("axis", "mouse_motion", "motion"):
                return float(v)

            cap_min = inp.get("cap_min")
            cap_max = inp.get("cap_max")
            trim = inp.get("trim")
            deadzone = inp.get("deadzone")

            # cap normalization
            lo = None
            hi = None
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

            # trim remap
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

            # deadzone
            try:
                dz = float(deadzone)
            except Exception:
                dz = 0.0
            if dz > 0.0 and abs(float(v)) < float(dz):
                v = 0.0

            return float(v)
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            try:
                iid = int(inp.get("iid"))
            except Exception:
                continue
            dev = str(inp.get("device", ""))
            kind = str(inp.get("kind", ""))
            try:
                item_id = int(inp.get("id", 0))
            except Exception:
                item_id = 0
            try:
                sigsel = int(inp.get("sigsel", 0))
            except Exception:
                sigsel = 0

            invert = bool(inp.get("invert", False))

            if dev == "joystick":
                dev_code = int(signal_kernel_api.GP_DEV_JOYSTICK)
            elif dev == "keyboard":
                dev_code = int(signal_kernel_api.GP_DEV_KEYBOARD)
            elif dev == "mouse":
                dev_code = int(signal_kernel_api.GP_DEV_MOUSE)
            else:
                dev_code = int(signal_kernel_api.GP_DEV_JOYSTICK)

            if kind == "axis":
                kind_code = int(signal_kernel_api.GP_EV_AXIS)
            elif kind == "button":
                kind_code = int(signal_kernel_api.GP_EV_BUTTON)
            elif kind == "hat":
                kind_code = int(signal_kernel_api.GP_EV_HAT)
            elif kind == "key":
                kind_code = int(signal_kernel_api.GP_EV_KEY)
            elif kind in ("mouse_motion", "motion"):
                kind_code = int(signal_kernel_api.GP_EV_MOUSE_MOTION)
            else:
                kind_code = int(signal_kernel_api.GP_EV_AXIS)

            sid = int(signal_kernel_api.compose_signal_id(int(dev_code), int(kind_code), int(item_id)))
            ok = 0
            try:
                if hasattr(self._sigk, "gp_sigk_peek_sel"):
                    ok = int(self._sigk.gp_sigk_peek_sel(int(now_ns), int(sid), int(sigsel), fr))
                else:
                    ok = int(self._sigk.gp_sigk_peek(int(now_ns), int(sid), fr))
            except Exception:
                ok = 0
            vv = float(fr.value) if ok else 0.0
            if invert and kind in ("axis", "mouse_motion", "motion"):
                vv = -float(vv)

            # Device-level calibration preprocessing.
            vv = float(_apply_input_calib(float(vv), inp, kind=str(kind)))
            in_vals[int(iid)] = float(vv)

        # 2) Evaluate nodes in topological order.
        node_out_1d: dict[int, float] = {}
        node_out_2d: dict[int, tuple[float, float]] = {}

        def _arg_scalar(a: dict[str, Any]) -> float:
            if not isinstance(a, dict):
                return 0.0
            ref = str(a.get("ref", ""))
            if ref == "imm":
                try:
                    return float(a.get("value", 0.0))
                except Exception:
                    return 0.0
            if ref == "input":
                try:
                    return float(in_vals.get(int(a.get("iid")), 0.0))
                except Exception:
                    return 0.0
            if ref == "node":
                try:
                    nid = int(a.get("nid"))
                except Exception:
                    nid = -1
                if nid in node_out_1d:
                    return float(node_out_1d.get(nid, 0.0))
                if nid in node_out_2d:
                    x, y = node_out_2d.get(nid, (0.0, 0.0))
                    comp = str(a.get("comp", "x")).lower().strip()
                    if comp == "y":
                        return float(y)
                    return float(x)
            return 0.0

        def _clamp11(v: float) -> float:
            return float(max(-1.0, min(1.0, float(v))))

        for n in nodes:
            if not isinstance(n, dict):
                continue
            try:
                nid = int(n.get("nid"))
            except Exception:
                continue
            op = str(n.get("op", ""))
            try:
                dim = int(n.get("dim", 1))
            except Exception:
                dim = 1
            args = n.get("args")
            if not isinstance(args, list):
                args = []

            if dim == 2:
                # Default 2D view: first (x,y) pair.
                a0 = _arg_scalar(args[0]) if len(args) > 0 else 0.0
                a1 = _arg_scalar(args[1]) if len(args) > 1 else 0.0
                x = float(a0)
                y = float(a1)
                if op == "2dflightstick" and hasattr(self._sigk, "gp_sigop_2dflightstick"):
                    try:
                        import ctypes

                        ox = ctypes.c_float(0.0)
                        oy = ctypes.c_float(0.0)
                        self._sigk.gp_sigop_2dflightstick(float(a0), float(a1), ctypes.byref(ox), ctypes.byref(oy))
                        x, y = float(ox.value), float(oy.value)
                    except Exception:
                        x, y = float(a0), float(-a1)
                elif op == "2dseek" and hasattr(self._sigk, "gp_sigop_2dseek"):
                    try:
                        import ctypes

                        ox = ctypes.c_float(0.0)
                        oy = ctypes.c_float(0.0)
                        self._sigk.gp_sigop_2dseek(float(a0), float(a1), ctypes.byref(ox), ctypes.byref(oy))
                        x, y = float(ox.value), float(oy.value)
                    except Exception:
                        x, y = float(a0), float(a1)
                elif op == "2dsumclamp":
                    sx = 0.0
                    sy = 0.0
                    # args are (x0,y0,x1,y1,...) pairs
                    n = int(len(args))
                    i = 0
                    while i + 1 < n:
                        sx += float(_arg_scalar(args[i]))
                        sy += float(_arg_scalar(args[i + 1]))
                        i += 2
                    x = _clamp11(float(sx))
                    y = _clamp11(float(sy))
                node_out_2d[int(nid)] = (float(x), float(y))
            else:
                if op == "const":
                    try:
                        v = float(n.get("value", 0.0))
                    except Exception:
                        v = 0.0
                else:
                    v = _arg_scalar(args[0]) if len(args) > 0 else 0.0
                node_out_1d[int(nid)] = float(v)

        # 3) Resolve final channels.
        overrides_1d: dict[int, float] = {}
        channels_2d: dict[int, tuple[float, float]] = {}

        for out_ch, spec in final_channels.items():
            try:
                out_i = int(out_ch)
            except Exception:
                continue
            if not isinstance(spec, dict):
                continue
            sid = str(spec.get("sid", ""))
            if not sid:
                continue
            s = signals.get(sid)
            if not isinstance(s, dict):
                continue
            try:
                nid = int(s.get("nid"))
            except Exception:
                continue
            try:
                dim = int(spec.get("dim", s.get("dim", 1)))
            except Exception:
                dim = 1

            if dim == 2 and nid in node_out_2d:
                channels_2d[int(out_i)] = tuple(node_out_2d.get(int(nid), (0.0, 0.0)))
            elif nid in node_out_1d:
                overrides_1d[int(out_i)] = float(node_out_1d.get(int(nid), 0.0))

        # 4) Simple activation reporting (edge-triggered prints).
        active_now: set[int] = set()
        eps = float(max(0.0, active_eps))
        for ch, v in overrides_1d.items():
            if abs(float(v)) > eps:
                active_now.add(int(ch))
        for ch, (x, y) in channels_2d.items():
            if max(abs(float(x)), abs(float(y))) > eps:
                active_now.add(int(ch))

        if active_now != self._prev_active:
            # Avoid spamming: only report on set change.
            try:
                print(f"[controller_backend] active channels: {sorted(active_now)}")
            except Exception:
                pass
            self._prev_active = set(active_now)

        self._last_status = ControllerBackendStatus(ok=True, msg="ok")
        return overrides_1d, channels_2d
