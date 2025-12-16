from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from c_physics.controller_engine_ctypes import (
    GP_CTL_HOOK_LEVEL,
    GP_CTL_HOOK_ON_FALL,
    GP_CTL_HOOK_ON_RISE,
    GP_CTL_HOOK_SRC_SIGNAL,
    GP_CtlHookEvent,
    GP_CtlHookWatch,
    GP_CtlMeta,
    GP_CtlOutputDesc,
    GP_WheelMeta,
    GP_WheelSample,
)

from c_physics import signal_kernel_api


@dataclass(frozen=True)
class HookBinding:
    action: str
    source: str = "signal"  # signal|channel
    # Channel source
    channel: int = 0
    comp: int = 0
    # Signal source
    device: int = int(signal_kernel_api.GP_DEV_JOYSTICK)
    kind: int = int(signal_kernel_api.GP_EV_BUTTON)
    item_id: int = 0
    sigsel: int = 2  # GP_SIGSEL_DOWN
    edge: str = "rise"  # rise|fall|level
    threshold: float = 0.5
    hysteresis: float = 0.05
    dispatch: str = "async"  # main|async (python-side); C-side reserved for future


def _edge_to_flags(edge: str) -> int:
    e = str(edge).strip().lower()
    if not e:
        return int(GP_CTL_HOOK_ON_RISE)
    # Support multi-flag strings like "rise+fall" or "rise|level".
    toks = [t.strip() for t in e.replace("|", "+").replace(",", "+").split("+") if t.strip()]
    if not toks:
        toks = [e]
    flags = 0
    for t in toks:
        if t in ("rise", "on_rise", "up"):
            flags |= int(GP_CTL_HOOK_ON_RISE)
        elif t in ("fall", "on_fall", "down"):
            flags |= int(GP_CTL_HOOK_ON_FALL)
        elif t in ("level", "held", "hold"):
            flags |= int(GP_CTL_HOOK_LEVEL)
    return int(flags) if flags else int(GP_CTL_HOOK_ON_RISE)


def _dir_to_hat_btn_idx(x: int, y: int) -> int | None:
    if int(x) == 0 and int(y) == 0:
        return None
    dirs = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    for i, (dx, dy) in enumerate(dirs):
        if int(dx) == int(x) and int(dy) == int(y):
            return int(i)
    return None


def _virtual_hat_btn_id(hat_idx: int, x: int, y: int) -> int | None:
    # Must match gl_animator_geodesic.py: _HARD_HAT_BTN_BASE = 51000
    base = 51000
    di = _dir_to_hat_btn_idx(int(x), int(y))
    if di is None:
        return None
    return int(base + int(max(0, int(hat_idx))) * 8 + int(di))


def _binding_to_signal_watch(binding: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Return (device, kind, item_id, sigsel) or None."""
    if not isinstance(binding, dict):
        return None
    t = str(binding.get("type") or "").strip().lower()
    if t == "button":
        try:
            b = int(binding.get("button"))
        except Exception:
            return None
        return (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(b), 2)
    if t == "hat":
        try:
            h = int(binding.get("hat"))
            x = int(binding.get("x", 0))
            y = int(binding.get("y", 0))
        except Exception:
            return None
        vid = _virtual_hat_btn_id(int(h), int(x), int(y))
        if vid is None:
            return None
        return (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(vid), 2)
    return None


def load_hook_bindings_from_cfg(cfg: dict[str, Any]) -> list[HookBinding]:
    """Load optional hook bindings from joystick.json.

    Schema (optional):

    cfg["flight_controls"]["controller"]["hook_bindings"] = [
      {"action":"flight_toggle", "channel":100, "edge":"rise", "threshold":0.5},
      ...
    ]
    """

    out: list[HookBinding] = []

    # If the user (or bootstrap) already provided explicit controller hook bindings
    # for menu actions, do NOT also synthesize legacy menu_button/menu_nav bindings,
    # or we'll emit duplicate semantic events.
    explicit_actions: set[str] = set()
    try:
        fc0 = cfg.get("flight_controls")
        ctrl0 = fc0.get("controller") if isinstance(fc0, dict) else None
        hb0 = ctrl0.get("hook_bindings") if isinstance(ctrl0, dict) else None
        if isinstance(hb0, list):
            for it0 in hb0:
                if isinstance(it0, dict) and isinstance(it0.get("action"), str):
                    explicit_actions.add(str(it0.get("action")))
    except Exception:
        explicit_actions = set()

    # Default dispatch (main vs async)
    dispatch_default = "async"
    dispatch_overrides: dict[str, str] = {}
    try:
        fc = cfg.get("flight_controls")
        ctrl = fc.get("controller") if isinstance(fc, dict) else None
        if isinstance(ctrl, dict):
            dispatch_default = str(ctrl.get("hook_dispatch_default", dispatch_default) or dispatch_default)
            ovs = ctrl.get("hook_dispatch_overrides")
            if isinstance(ovs, dict):
                for k, v in ovs.items():
                    kk = str(k).strip()
                    vv = str(v).strip().lower()
                    if kk and vv in ("main", "async"):
                        dispatch_overrides[kk] = vv
    except Exception:
        pass
    dispatch_default = str(dispatch_default).strip().lower() or "main"
    if dispatch_default not in ("main", "async"):
        dispatch_default = "main"

    # 1) Generate bindings from the existing joystick.json schema.
    try:
        fc = cfg.get("flight_controls") if isinstance(cfg, dict) else None
        fc = fc if isinstance(fc, dict) else {}

        # camera.zoom_in/out
        cam = fc.get("camera") if isinstance(fc.get("camera"), dict) else {}
        for key in ("zoom_in", "zoom_out"):
            b = cam.get(key)
            spec = _binding_to_signal_watch(b) if isinstance(b, dict) else None
            if spec is None:
                continue
            dev, kind, item_id, sigsel = spec
            out.append(
                HookBinding(
                    action=f"camera_{key}",
                    source="signal",
                    device=int(dev),
                    kind=int(kind),
                    item_id=int(item_id),
                    sigsel=int(sigsel),
                    edge="rise+fall",
                    threshold=0.5,
                    hysteresis=0.05,
                    dispatch=str(dispatch_default),
                )
            )

        # weapons.fire_1/2 (continuous)
        weap = fc.get("weapons") if isinstance(fc.get("weapons"), dict) else {}
        for key in ("fire_1", "fire_2"):
            b = weap.get(key)
            spec = _binding_to_signal_watch(b) if isinstance(b, dict) else None
            if spec is None:
                continue
            dev, kind, item_id, sigsel = spec
            out.append(
                HookBinding(
                    action=f"weapons_{key}",
                    source="signal",
                    device=int(dev),
                    kind=int(kind),
                    item_id=int(item_id),
                    sigsel=int(sigsel),
                    edge="rise+fall",
                    threshold=0.5,
                    hysteresis=0.05,
                    dispatch=str(dispatch_default),
                )
            )

        # menu button + nav
        try:
            menu_btn = int(cfg.get("menu_button", -1))
        except Exception:
            menu_btn = -1
        if menu_btn >= 0 and "menu_open" not in explicit_actions:
            out.append(
                HookBinding(
                    action="menu_open",
                    source="signal",
                    device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                    kind=int(signal_kernel_api.GP_EV_BUTTON),
                    item_id=int(menu_btn),
                    sigsel=2,
                    edge="rise",
                    threshold=0.5,
                    hysteresis=0.05,
                    dispatch=str(dispatch_default),
                )
            )
        nav = cfg.get("menu_nav") if isinstance(cfg.get("menu_nav"), dict) else {}
        nav_map = {
            "confirm": "menu_confirm",
            "cancel": "menu_cancel",
            "up": "menu_up",
            "down": "menu_down",
            "left": "menu_left",
            "right": "menu_right",
        }
        for k, action in nav_map.items():
            if str(action) in explicit_actions:
                continue
            b = nav.get(k)
            spec = _binding_to_signal_watch(b) if isinstance(b, dict) else None
            if spec is None:
                continue
            dev, kind, item_id, sigsel = spec
            out.append(
                HookBinding(
                    action=str(action),
                    source="signal",
                    device=int(dev),
                    kind=int(kind),
                    item_id=int(item_id),
                    sigsel=int(sigsel),
                    edge="rise",
                    threshold=0.5,
                    hysteresis=0.05,
                    dispatch=str(dispatch_default),
                )
            )
    except Exception:
        pass

    # 2) Also load any explicit overrides/additions.
    hb = None
    try:
        fc = cfg.get("flight_controls")
        ctrl = fc.get("controller") if isinstance(fc, dict) else None
        hb = ctrl.get("hook_bindings") if isinstance(ctrl, dict) else None
    except Exception:
        hb = None

    if isinstance(hb, list):
        for it in hb:
            if not isinstance(it, dict):
                continue
            action = str(it.get("action") or "").strip()
            if not action:
                continue
            source = str(it.get("source", "channel") or "channel").strip().lower()
            edge = str(it.get("edge", "rise") or "rise")
            dispatch = str(it.get("dispatch", dispatch_default) or dispatch_default).strip().lower()
            if dispatch not in ("main", "async"):
                dispatch = dispatch_default
            try:
                thr = float(it.get("threshold", 0.5))
            except Exception:
                thr = 0.5
            try:
                hys = float(it.get("hysteresis", 0.05))
            except Exception:
                hys = 0.05

            if source == "signal":
                try:
                    dev = int(it.get("device", signal_kernel_api.GP_DEV_JOYSTICK))
                    kind = int(it.get("kind", signal_kernel_api.GP_EV_BUTTON))
                    item_id = int(it.get("item_id", 0))
                    sigsel = int(it.get("sigsel", 2))
                except Exception:
                    continue
                out.append(
                    HookBinding(
                        action=action,
                        source="signal",
                        device=int(dev),
                        kind=int(kind),
                        item_id=int(item_id),
                        sigsel=int(sigsel),
                        edge=edge,
                        threshold=float(thr),
                        hysteresis=float(hys),
                        dispatch=dispatch,
                    )
                )
            else:
                try:
                    ch = int(it.get("channel"))
                except Exception:
                    continue
                try:
                    comp = int(it.get("comp", 0))
                except Exception:
                    comp = 0
                out.append(
                    HookBinding(
                        action=action,
                        source="channel",
                        channel=int(ch),
                        comp=int(comp),
                        edge=edge,
                        threshold=float(thr),
                        hysteresis=float(hys),
                        dispatch=dispatch,
                    )
                )

    # De-dupe by (action, source, channel, comp, device, kind, item_id, sigsel)
    dedup: dict[tuple[Any, ...], HookBinding] = {}
    for b0 in out:
        b = b0
        try:
            ov = dispatch_overrides.get(str(b.action))
            if ov in ("main", "async") and str(b.dispatch) != str(ov):
                b = HookBinding(**{**b.__dict__, "dispatch": str(ov)})
        except Exception:
            pass
        k = (
            b.action,
            b.source,
            int(b.channel),
            int(b.comp),
            int(b.device),
            int(b.kind),
            int(b.item_id),
            int(b.sigsel),
            str(b.edge),
            float(b.threshold),
            float(b.hysteresis),
            str(b.dispatch),
        )
        dedup[k] = b
    return list(dedup.values())


class HookEventPump:
    """Background waiter on the C hook queue, pushing events to a Python callback.

    This is intentionally minimal: one sleeping Python thread blocks in
    gp_ctl_hookq_wait and only wakes on hook events.
    """

    def __init__(self, ctl_lib) -> None:
        self._ctl = ctl_lib
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_seq: int = 0

    def start(self, *, on_event: Callable[[int, GP_CtlHookEvent], None], poll_timeout_ms: int = 50) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        def _run() -> None:
            last_seen = int(self._last_seq)
            ev = GP_CtlHookEvent()
            while not self._stop.is_set():
                out_seq = ctypes.c_uint64(0)
                rc = int(self._ctl.gp_ctl_hookq_wait(last_seen, int(poll_timeout_ms), ctypes.byref(out_seq)))
                new_seq = int(out_seq.value)
                if rc and new_seq > last_seen:
                    for s in range(last_seen + 1, new_seq + 1):
                        if int(self._ctl.gp_ctl_hookq_peek_seq(int(s), ctypes.byref(ev))) != 1:
                            break
                        on_event(int(s), ev)
                    last_seen = new_seq
                else:
                    # Stay responsive to stop requests.
                    continue

            self._last_seq = last_seen

        self._thread = threading.Thread(target=_run, name="HookEventPump", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=float(join_timeout_s))
        self._thread = None


class WheelHotPump:
    """Background waiter on the C signal wheel, sweeping HOT indices.

    This is the async-first path for channel outputs:
    - block in gp_ctl_wheel_wait() until a new tick
    - gp_ctl_wheel_drain_hot() to get indices that were HOT since last sweep
    """

    def __init__(self, ctl_lib) -> None:
        self._ctl = ctl_lib
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_tick_seq: int = 0

    def start(self, *, on_hot: Callable[[int], None], poll_timeout_ms: int = 50) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        def _run() -> None:
            last_seen = int(self._last_tick_seq)
            meta = GP_WheelMeta()
            if int(self._ctl.gp_ctl_wheel_get_meta(ctypes.byref(meta))) != 1:
                return
            cap = int(meta.signal_count)
            if cap <= 0:
                return
            arr_t = ctypes.c_uint32 * cap
            idx_buf = arr_t()
            out_n = ctypes.c_uint32(0)

            while not self._stop.is_set():
                out_tick = ctypes.c_uint64(0)
                rc = int(self._ctl.gp_ctl_wheel_wait(int(last_seen), int(poll_timeout_ms), ctypes.byref(out_tick)))
                if self._stop.is_set():
                    break
                if rc != 1:
                    continue
                last_seen = int(out_tick.value)

                out_n.value = 0
                if int(self._ctl.gp_ctl_wheel_drain_hot(idx_buf, int(cap), ctypes.byref(out_n))) != 1:
                    continue
                n = int(out_n.value)
                for i in range(n):
                    on_hot(int(idx_buf[i]))

            self._last_tick_seq = last_seen

        self._thread = threading.Thread(target=_run, name="WheelHotPump", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=float(join_timeout_s))
        self._thread = None


class HookBindingDispatcher:
    """Registers hook watches and dispatches named actions onto the main thread."""

    def __init__(self, ctl_lib) -> None:
        self._ctl = ctl_lib
        self._pump = HookEventPump(ctl_lib)
        self._wheel_pump = WheelHotPump(ctl_lib)
        self._lock = threading.Lock()
        self._pending: list[tuple[str, GP_CtlHookEvent]] = []
        self._hook_id_to_action: dict[int, str] = {}
        self._action_dispatch: dict[str, str] = {}

        # Wheel mapping: packed scalar index -> list[(action, dispatch, channel, comp)]
        self._wheel_idx_to_actions: dict[int, list[tuple[str, str, int, int]]] = {}
        self._wheel_sample = GP_WheelSample()
        self._wheel_ws = ctypes.c_uint64(0)

    def _wheel_available(self) -> bool:
        return all(hasattr(self._ctl, name) for name in ("gp_ctl_wheel_wait", "gp_ctl_wheel_drain_hot", "gp_ctl_wheel_peek_latest", "gp_ctl_wheel_get_meta"))

    def _build_channel_comp_to_signal_idx(self) -> dict[tuple[int, int], int]:
        meta = GP_CtlMeta()
        if int(self._ctl.gp_ctl_get_meta(ctypes.byref(meta))) != 1:
            return {}
        n_out = int(meta.output_count)
        if n_out <= 0:
            return {}
        arr_t = GP_CtlOutputDesc * n_out
        arr = arr_t()
        got = int(self._ctl.gp_ctl_get_outputs(arr, int(n_out)))
        out: dict[tuple[int, int], int] = {}
        for i in range(max(0, got)):
            d = arr[int(i)]
            ch = int(d.channel)
            dim = int(d.dim)
            stride = int(d.stride) if int(d.stride) > 0 else 1
            base = int(d.offset)
            for comp in range(max(0, dim)):
                out[(ch, int(comp))] = int(base + int(comp) * int(stride))
        return out

    def register(self, bindings: Iterable[HookBinding]) -> None:
        self._ctl.gp_ctl_hooks_clear()
        self._hook_id_to_action.clear()
        self._action_dispatch.clear()
        self._wheel_idx_to_actions.clear()

        wheel_ok = self._wheel_available()
        ch_map = self._build_channel_comp_to_signal_idx() if wheel_ok else {}

        next_hook_id = 1
        for b in bindings:
            disp = str(getattr(b, "dispatch", "main") or "main").strip().lower()
            if disp not in ("main", "async"):
                disp = "main"

            hid = int(next_hook_id)
            next_hook_id += 1

            flags = int(_edge_to_flags(b.edge))
            if str(getattr(b, "source", "channel") or "channel").strip().lower() == "signal":
                flags |= int(GP_CTL_HOOK_SRC_SIGNAL)
                sid = signal_kernel_api.compose_signal_id(int(b.device), int(b.kind), int(b.item_id))
                w = GP_CtlHookWatch(
                    hook_id=hid,
                    channel=0,
                    comp=0,
                    flags=int(flags),
                    threshold=float(b.threshold),
                    hysteresis=float(b.hysteresis),
                    signal_id=int(sid),
                    sigsel=int(b.sigsel),
                )
                self._ctl.gp_ctl_hooks_add(ctypes.byref(w))
                self._hook_id_to_action[int(hid)] = str(b.action)
                self._action_dispatch[str(b.action)] = str(disp)
            else:
                # Prefer the wheel path for channel bindings if available.
                sig_idx = ch_map.get((int(b.channel), int(b.comp))) if wheel_ok else None
                if sig_idx is not None:
                    self._wheel_idx_to_actions.setdefault(int(sig_idx), []).append((str(b.action), str(disp), int(b.channel), int(b.comp)))
                    self._action_dispatch[str(b.action)] = str(disp)
                else:
                    # Fallback: use hook watches on controller outputs.
                    w = GP_CtlHookWatch(
                        hook_id=hid,
                        channel=int(b.channel),
                        comp=int(b.comp),
                        flags=int(flags),
                        threshold=float(b.threshold),
                        hysteresis=float(b.hysteresis),
                        signal_id=0,
                        sigsel=0,
                    )
                    self._ctl.gp_ctl_hooks_add(ctypes.byref(w))
                    self._hook_id_to_action[int(hid)] = str(b.action)
                    self._action_dispatch[str(b.action)] = str(disp)

    def start(self, *, handlers: dict[str, Callable[[GP_CtlHookEvent], None]], poll_timeout_ms: int = 50) -> None:
        def _on_event(_seq: int, ev: GP_CtlHookEvent) -> None:
            action = self._hook_id_to_action.get(int(ev.hook_id))
            if not action:
                return
            disp = str(self._action_dispatch.get(str(action), "main") or "main").strip().lower()
            fn = handlers.get(str(action))

            # Copy struct value so subsequent reads don't overwrite.
            ev_copy = GP_CtlHookEvent()
            ctypes.pointer(ev_copy)[0] = ev

            if disp == "async" and fn is not None:
                try:
                    fn(ev_copy)
                except Exception:
                    pass
                return

            with self._lock:
                self._pending.append((str(action), ev_copy))

        # Start the hook-queue pump (for signal watches and any fallback channel watches).
        self._pump.start(on_event=_on_event, poll_timeout_ms=int(poll_timeout_ms))

        # Start the wheel pump (for channel bindings, if any).
        if self._wheel_idx_to_actions and self._wheel_available():

            def _on_hot(sig_idx: int) -> None:
                items = self._wheel_idx_to_actions.get(int(sig_idx))
                if not items:
                    return
                # Snapshot latest sample for this scalar.
                t_ns = 0
                v = 0.0
                try:
                    self._wheel_ws.value = 0
                    if int(self._ctl.gp_ctl_wheel_peek_latest(int(sig_idx), ctypes.byref(self._wheel_sample), ctypes.byref(self._wheel_ws))) == 1:
                        t_ns = int(getattr(self._wheel_sample, "t_ns", 0))
                        v = float(getattr(self._wheel_sample, "value", 0.0))
                except Exception:
                    pass

                for action, disp, ch, comp in list(items):
                    fn = handlers.get(str(action))
                    if fn is None:
                        continue
                    ev = GP_CtlHookEvent(
                        t_ns=int(t_ns),
                        hook_id=0,
                        channel=int(ch),
                        comp=int(comp),
                        flags=0,
                        value=float(v),
                        _pad0=0.0,
                    )
                    if disp == "async":
                        try:
                            fn(ev)
                        except Exception:
                            pass
                    else:
                        with self._lock:
                            self._pending.append((str(action), ev))

            self._wheel_pump.start(on_hot=_on_hot, poll_timeout_ms=int(poll_timeout_ms))

    def drain(self, *, handlers: dict[str, Callable[[GP_CtlHookEvent], None]]) -> int:
        """Run pending actions on the caller thread.

        Returns number of dispatched events.
        """

        with self._lock:
            items = list(self._pending)
            self._pending.clear()

        n = 0
        for action, ev in items:
            fn = handlers.get(str(action))
            if fn is None:
                continue
            try:
                fn(ev)
            except Exception:
                pass
            n += 1
        return n

    def stop(self) -> None:
        self._wheel_pump.stop()
        self._pump.stop()
