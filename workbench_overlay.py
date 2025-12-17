from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import workbench_elements as wbe


@dataclass(frozen=True)
class UiTextButton:
    """A small, deployable text-button model for UiDoc-driven overlays."""

    key: str
    text: str
    selected: bool = False

    def render_text(self) -> str:
        return str(self.text)


@dataclass(frozen=True)
class OverlaySnapshot:
    """MenuSnapshot-shaped overlay snapshot.

    We intentionally reuse the same shape as MenuSnapshot so the existing
    MenuBufferPresenter can render any overlay source (menu, workbench, etc.)
    through the same UiDoc pipeline.
    """

    active: bool
    gen: int
    node_id: str
    title: str
    item_labels: list[str]
    selected_idx: int
    selected_action: str

    # Extra fields are allowed; MenuBufferPresenter reads the common fields via getattr.
    signal_idx: int = 0


class WorkbenchOverlay:
    """Non-blocking Signal Workbench overlay (prototype).

    This is a separate overlay source (not a blocking mode). It is intended to
    run concurrently with gameplay and only reacts to semantic UI actions.

    Rendering is done via UiDoc (see menu_pages uidoc override for node_id
    "workbench").
    """

    def __init__(self) -> None:
        self._active = False
        self._gen = 0
        # Cursor spans: [0]=CAPTURE button, [1]=MODE button, [2..]=signals
        self._cursor = 0
        # Remember last selected signal even when cursor is on the buttons.
        self._sig_sel = 0
        self._sig_count = 0
        self._pushed_mode = False
        self._mode = "scan"  # scan|announce
        self._capture = False
        self._last_rebuild_ns = 0

    def _try_load_final_graph_kernel_inputs(self, path: str) -> set[tuple[int, int, int]] | None:
        try:
            if not os.path.exists(str(path)):
                return None
            with open(str(path), "r", encoding="utf-8") as f:
                doc = json.load(f)
            ins = doc.get("inputs") if isinstance(doc, dict) else None
            if not isinstance(ins, list):
                return None

            from c_physics import signal_kernel_api

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

    def _rebuild_graphs_and_refresh_allowlist(self, *, now_ns: int, force: bool = False) -> None:
        """Rebuild controller graphs and refresh announce allowlist.

        This mirrors the legacy workbench behavior: the allowlist for announce
        mode is derived from the compiled final graph inputs.
        """

        try:
            if (not force) and int(now_ns) - int(self._last_rebuild_ns or 0) < 250_000_000:
                return
            self._last_rebuild_ns = int(now_ns)

            # Rebuild final graph (safe, idempotent).
            try:
                import controller_graph_compile

                controller_graph_compile.try_build_final_graph(
                    joystick_path="joystick.json",
                    mixer_path="channel_mixer.json",
                    compiled_out_path="controller_graph_compiled.json",
                    final_out_path="controller_graph_final.json",
                )
            except Exception:
                pass

            interest = self._try_load_final_graph_kernel_inputs("controller_graph_final.json")
            if not isinstance(interest, set):
                interest = set()
            try:
                import input_interest

                input_interest.set_announce_interest(set(interest))
            except Exception:
                pass
        except Exception:
            return

    def is_active(self) -> bool:
        return bool(self._active)

    def open(self) -> None:
        if self._active:
            return
        self._active = True
        self._gen += 1
        # Ensure announce-mode ingestion whitelist is in sync with the compiled
        # controller graph when the overlay becomes active.
        try:
            self._rebuild_graphs_and_refresh_allowlist(now_ns=int(time.monotonic_ns()), force=False)
        except Exception:
            pass
        try:
            import input_interest

            input_interest.push_focus("workbench")
            # Default to scan-mode while the workbench is open; the MODE button
            # can toggle to announce without losing the underlying mode.
            input_interest.push_mode(str(self._mode))
            self._pushed_mode = True
        except Exception:
            pass

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._gen += 1
        try:
            # Drop any cached wave history so globals do not retain stale data.
            import workbench_elements as wbe

            wbe.clear_histories()
        except Exception:
            pass
        try:
            import input_interest

            input_interest.pop_focus("workbench")
            # Release exclusive capture if we held it.
            if bool(self._capture):
                try:
                    input_interest.pop_capture("workbench")
                except Exception:
                    pass
                self._capture = False

            if bool(self._pushed_mode):
                # Pop whatever the current overlay-top mode is.
                input_interest.pop_mode()
                self._pushed_mode = False
        except Exception:
            pass

    def toggle(self) -> None:
        if self._active:
            self.close()
        else:
            self.open()

    def _cursor_max(self) -> int:
        # 2 top-left buttons + N signals (or 0).
        return int(max(1, 2 + int(max(0, int(self._sig_count)))))

    def set_selected(self, idx: int) -> None:
        i = int(idx)
        if i < 0:
            i = 0
        i = int(min(i, self._cursor_max() - 1))

        if i != int(self._cursor):
            self._cursor = int(i)
            if int(self._cursor) >= 2:
                self._sig_sel = int(max(0, int(self._cursor) - 2))
            self._gen += 1

    def nav(self, delta: int) -> None:
        if not self._active:
            return
        self.set_selected(int(self._cursor) + int(delta))

    def cancel(self) -> None:
        # Cancel closes the workbench overlay (returns to gameplay).
        self.close()

    def confirm(self) -> None:
        if not self._active:
            return

        if int(self._cursor) == 0:
            self.toggle_capture()
            return
        if int(self._cursor) == 1:
            self.toggle_mode()
            return

        # Signal row confirm is a placeholder (future: binding, drilldown).

    def refresh_wheel_meta(self) -> None:
        """Refresh wheel signal_count if controller engine is loaded."""
        try:
            from c_physics.controller_engine_api import try_load_controller_engine
            from c_physics.controller_engine_ctypes import GP_WheelMeta

            lib, _ = try_load_controller_engine(search_dir="c_physics")
            if lib is None or not hasattr(lib, "gp_ctl_wheel_get_meta"):
                return
            meta = GP_WheelMeta()
            ok = int(lib.gp_ctl_wheel_get_meta(ctypes.byref(meta)))
            if not ok:
                return
            self._sig_count = int(meta.signal_count)
            if self._sig_count <= 0:
                self._sig_count = 0
            if self._sig_count and self._sig_sel >= self._sig_count:
                self._sig_sel = max(0, self._sig_count - 1)
            # Keep cursor in range.
            if self._cursor >= self._cursor_max():
                self._cursor = max(0, self._cursor_max() - 1)
        except Exception:
            return

    def toggle_capture(self) -> None:
        if not self._active:
            return
        self._capture = not bool(self._capture)
        self._gen += 1
        try:
            import input_interest

            if bool(self._capture):
                input_interest.push_capture("workbench")
                input_interest.push_focus("workbench")
            else:
                input_interest.pop_capture("workbench")
        except Exception:
            pass

    def toggle_mode(self) -> None:
        """Toggle scan/announce while the workbench is open."""

        if not self._active:
            return
        cur = str(self._mode or "scan").strip().lower() or "scan"
        nxt = "announce" if cur == "scan" else "scan"
        self._mode = str(nxt)
        self._gen += 1
        # When switching into announce-mode, rebuild + refresh allowlist so
        # ingestion filtering matches the latest compiled graph.
        try:
            if str(self._mode) == "announce":
                self._rebuild_graphs_and_refresh_allowlist(now_ns=int(time.monotonic_ns()), force=True)
        except Exception:
            pass
        try:
            import input_interest

            # Update the top-of-stack mode while keeping the underlying mode intact.
            input_interest.set_mode(str(self._mode))
        except Exception:
            pass

    def snapshot(self) -> OverlaySnapshot:
        self.refresh_wheel_meta()
        # Keep item_labels light; the UiDoc builder will query C for details.
        return OverlaySnapshot(
            active=bool(self._active),
            gen=int(self._gen),
            node_id="workbench",
            title="SIGNAL WORKBENCH (OVERLAY)",
            item_labels=[],
            selected_idx=int(self._cursor),
            selected_action="",
            signal_idx=int(self._sig_sel),
        )


_CTRL_CFG_CACHE: dict[str, Any] = {
    "mtime": None,
    "cfg": None,
    "signals": None,
    "channels": None,
}


_SIGK_CACHE: dict[str, Any] = {
    "loaded": False,
    "lib": None,
    "api": None,
}


# NOTE: In the legacy workbench, clearing pulses was fine because it ran as a
# blocking tool. In the overlay model, the workbench can run concurrently with
# other subsystems also peeking the signal kernel. Some C backends can block on
# clear/peek contention; disable pulse-clearing by default to avoid freezes.
_WORKBENCH_CLEAR_PULSES = False
_WORKBENCH_CLEAR_PULSES_INTERVAL_NS = int(250_000_000)  # 250ms
_workbench_last_pulse_clear_ns = 0


_JOY_CACHE: dict[str, Any] = {
    "last_probe_ns": 0,
    "name": "JOYSTICK",
    "axes": 0,
    "buttons": 0,
    "hats": 0,
}


_CTL_CACHE: dict[str, Any] = {
    "loaded": False,
    "lib": None,
    "meta": None,
    "outputs": None,
    "buf": None,
    "last_probe_ns": 0,
}


def _try_load_ctl_cached(*, now_ns: int, min_interval_ns: int = 1_000_000_000) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Return (lib, meta, outputs_by_channel, buf).

    This is intentionally opportunistic: the controller engine may not be loaded
    in the current session.
    """

    global _CTL_CACHE
    try:
        if int(now_ns) - int(_CTL_CACHE.get("last_probe_ns", 0) or 0) < int(min_interval_ns):
            return (
                _CTL_CACHE.get("lib"),
                _CTL_CACHE.get("meta"),
                _CTL_CACHE.get("outputs"),
                _CTL_CACHE.get("buf"),
            )
        _CTL_CACHE["last_probe_ns"] = int(now_ns)

        from c_physics.controller_engine_api import try_load_controller_engine
        from c_physics.controller_engine_ctypes import GP_CtlMeta, GP_CtlOutputDesc

        lib, _api = try_load_controller_engine(search_dir="c_physics")
        if lib is None or (not hasattr(lib, "gp_ctl_get_meta")) or (not hasattr(lib, "gp_ctl_get_outputs")) or (not hasattr(lib, "gp_ctl_peek_latest")):
            _CTL_CACHE["lib"] = None
            _CTL_CACHE["meta"] = None
            _CTL_CACHE["outputs"] = None
            _CTL_CACHE["buf"] = None
            return (None, None, None, None)

        meta = GP_CtlMeta()
        if not int(lib.gp_ctl_get_meta(ctypes.byref(meta))):
            _CTL_CACHE["lib"] = lib
            _CTL_CACHE["meta"] = None
            _CTL_CACHE["outputs"] = None
            _CTL_CACHE["buf"] = None
            return (lib, None, None, None)

        outputs_by_channel: dict[int, Any] = {}
        n_out = int(getattr(meta, "output_count", 0) or 0)
        if n_out > 0:
            arr_t = GP_CtlOutputDesc * n_out
            arr = arr_t()
            got = int(lib.gp_ctl_get_outputs(arr, int(n_out)))
            for i in range(int(max(0, got))):
                d = arr[int(i)]
                try:
                    outputs_by_channel[int(d.channel)] = d
                except Exception:
                    pass

        buf = None
        total_floats = int(getattr(meta, "total_floats", 0) or 0)
        if total_floats > 0:
            buf = (ctypes.c_float * int(total_floats))()

        _CTL_CACHE["lib"] = lib
        _CTL_CACHE["meta"] = meta
        _CTL_CACHE["outputs"] = outputs_by_channel
        _CTL_CACHE["buf"] = buf
        return (lib, meta, outputs_by_channel, buf)
    except Exception:
        return (None, None, None, None)


def _ctl_peek_channels(*, now_ns: int) -> list[dict[str, Any]]:
    """Return a list of {ch, dim, values} for available controller outputs."""

    lib, meta, outputs_by_channel, buf = _try_load_ctl_cached(now_ns=now_ns)
    if lib is None or meta is None or not isinstance(outputs_by_channel, dict) or buf is None:
        return []
    try:
        # Peek latest into the shared buffer.
        out_seq = ctypes.c_uint64(0)
        out_t = ctypes.c_uint64(0)
        ok = int(lib.gp_ctl_peek_latest(buf, int(getattr(meta, "total_floats", 0) or 0), ctypes.byref(out_seq), ctypes.byref(out_t)))
        if not ok:
            return []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for ch in sorted(outputs_by_channel.keys()):
        d = outputs_by_channel.get(int(ch))
        if d is None:
            continue
        try:
            dim = int(getattr(d, "dim", 1) or 1)
        except Exception:
            dim = 1
        try:
            stride = int(getattr(d, "stride", 1) or 1)
        except Exception:
            stride = 1
        try:
            offset = int(getattr(d, "offset", 0) or 0)
        except Exception:
            offset = 0

        vals: list[float] = []
        for i in range(max(1, dim)):
            idx = int(offset + i * stride)
            try:
                vals.append(float(buf[int(idx)]))
            except Exception:
                vals.append(0.0)
        out.append(
            {
                "ch": int(ch),
                "dim": int(2 if dim == 2 else 1),
                "values": vals,
                "sig_idx_base": int(offset),
                "stride": int(stride),
            }
        )

    return out


def _probe_joystick_shape_cached(*, now_ns: int, min_interval_ns: int = 1_000_000_000) -> tuple[str, int, int, int]:
    """Return (name, axes, buttons, hats) with a conservative throttle.

    Some pygame joystick queries can be slow/hang on flaky drivers; keep the
    UI build path from hammering OS calls.
    """

    try:
        last = int(_JOY_CACHE.get("last_probe_ns") or 0)
    except Exception:
        last = 0
    if last and int(now_ns) - int(last) < int(min_interval_ns):
        return (
            str(_JOY_CACHE.get("name") or "JOYSTICK"),
            int(_JOY_CACHE.get("axes") or 0),
            int(_JOY_CACHE.get("buttons") or 0),
            int(_JOY_CACHE.get("hats") or 0),
        )

    joy_name = "JOYSTICK"
    joy_axes = 0
    joy_buttons = 0
    joy_hats = 0
    try:
        import pygame

        try:
            if not pygame.joystick.get_init():
                pygame.joystick.init()
        except Exception:
            pass

        try:
            jcount = int(pygame.joystick.get_count())
        except Exception:
            jcount = 0

        if jcount > 0:
            j0 = pygame.joystick.Joystick(0)
            try:
                if not j0.get_init():
                    j0.init()
            except Exception:
                pass
            try:
                joy_name = str(j0.get_name() or "JOYSTICK")
            except Exception:
                joy_name = "JOYSTICK"
            try:
                joy_axes = int(j0.get_numaxes())
            except Exception:
                joy_axes = 0
            try:
                joy_buttons = int(j0.get_numbuttons())
            except Exception:
                joy_buttons = 0
            try:
                joy_hats = int(j0.get_numhats())
            except Exception:
                joy_hats = 0
    except Exception:
        pass

    _JOY_CACHE["last_probe_ns"] = int(now_ns)
    _JOY_CACHE["name"] = str(joy_name)
    _JOY_CACHE["axes"] = int(joy_axes)
    _JOY_CACHE["buttons"] = int(joy_buttons)
    _JOY_CACHE["hats"] = int(joy_hats)
    return str(joy_name), int(joy_axes), int(joy_buttons), int(joy_hats)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _load_controller_cfg_cached(*, path: str = "joystick.json") -> dict[str, Any] | None:
    """Load joystick/controller config with cheap mtime-based caching.

    This runs inside UiDoc building, which can be called frequently when live
    widgets (waveforms) are updating; avoid re-parsing JSON every time.
    """

    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return None

    if _CTRL_CFG_CACHE.get("mtime") == mtime and isinstance(_CTRL_CFG_CACHE.get("cfg"), dict):
        return _CTRL_CFG_CACHE.get("cfg")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return None
        try:
            import input_graph

            raw = input_graph.ensure_controller_graph(raw)
        except Exception:
            pass
        _CTRL_CFG_CACHE["mtime"] = mtime
        _CTRL_CFG_CACHE["cfg"] = raw
        _CTRL_CFG_CACHE["signals"] = None
        _CTRL_CFG_CACHE["channels"] = None
        return raw
    except Exception:
        return None


def _controller_block(cfg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        fc = cfg.get("flight_controls")
        if not isinstance(fc, dict):
            return None
        ctrl = fc.get("controller")
        if not isinstance(ctrl, dict):
            return None
        return ctrl
    except Exception:
        return None


def _get_saved_signal_ids(cfg: dict[str, Any]) -> list[str]:
    if not isinstance(_CTRL_CFG_CACHE.get("signals"), list):
        ctrl = _controller_block(cfg)
        sigs = ctrl.get("signals") if isinstance(ctrl, dict) else None
        if not isinstance(sigs, dict):
            _CTRL_CFG_CACHE["signals"] = []
        else:
            # Stable display ordering: numeric-ish first, then lexical.
            def _sig_sort_key(k: Any) -> tuple[int, str]:
                ks = str(k)
                try:
                    return (0, f"{int(ks):08d}")
                except Exception:
                    return (1, ks)

            _CTRL_CFG_CACHE["signals"] = [str(k) for k in sorted(sigs.keys(), key=_sig_sort_key)]
    out = _CTRL_CFG_CACHE.get("signals")
    return out if isinstance(out, list) else []


def _get_channel_map(cfg: dict[str, Any]) -> list[tuple[int, str]]:
    if not isinstance(_CTRL_CFG_CACHE.get("channels"), list):
        ctrl = _controller_block(cfg)
        chmap = ctrl.get("channels") if isinstance(ctrl, dict) else None
        out: list[tuple[int, str]] = []
        if isinstance(chmap, dict):
            for k, spec in chmap.items():
                try:
                    ch = int(k)
                except Exception:
                    continue
                if isinstance(spec, dict) and str(spec.get("source")) == "signal":
                    sid = str(spec.get("id", ""))
                    if sid:
                        out.append((int(ch), sid))
        out.sort(key=lambda t: int(t[0]))
        _CTRL_CFG_CACHE["channels"] = out
    out2 = _CTRL_CFG_CACHE.get("channels")
    return out2 if isinstance(out2, list) else []


def _mapped_channel_for_signal(cfg: dict[str, Any], sid: str) -> int | None:
    for ch, s in _get_channel_map(cfg):
        if str(s) == str(sid):
            return int(ch)
    return None


def _try_load_sigk_cached():
    if bool(_SIGK_CACHE.get("loaded")):
        return _SIGK_CACHE.get("lib"), _SIGK_CACHE.get("api")
    _SIGK_CACHE["loaded"] = True
    try:
        from c_physics import signal_kernel_api

        lib, api = signal_kernel_api.try_load_signal_kernel(search_dir="c_physics")
        _SIGK_CACHE["lib"] = lib
        _SIGK_CACHE["api"] = api
        return lib, api
    except Exception:
        _SIGK_CACHE["lib"] = None
        _SIGK_CACHE["api"] = None
        return None, None


def _sigk_peek(*, now_ns: int, signal_id: int) -> tuple[float, int, float, int]:
    """Return (value, flags, hold_s, aux) from the signal kernel frame."""

    try:
        _lib, api = _try_load_sigk_cached()
        if api is None or not hasattr(api, "gp_sigk_peek"):
            return 0.0, 0, 0.0, 0
        from c_physics.signal_kernel_ctypes import GP_SignalFrame

        fr = GP_SignalFrame()
        ok = 0
        try:
            ok = int(api.gp_sigk_peek(int(now_ns), int(signal_id), ctypes.byref(fr)))
        except Exception:
            ok = 0
        if not ok:
            return 0.0, 0, 0.0, 0
        try:
            aux = int(getattr(fr, "aux", 0))
        except Exception:
            aux = 0
        return float(fr.value), int(fr.flags), float(fr.hold_s), int(aux)
    except Exception:
        return 0.0, 0, 0.0, 0


def append_workbench_preview_panel(
    *,
    b: Any,
    x_px: int,
    y_px: int,
    w_px: int,
    h_px: int,
    signal_idx: int = 0,
) -> None:
    """Append a compact workbench-like preview panel into an existing UiDocBuilder.

    This is intended for menu preview panes (no full-screen clear, no focus).
    """

    x0 = int(x_px)
    y0 = int(y_px)
    w = int(max(120, w_px))
    h = int(max(90, h_px))

    pad = 10
    hdr_h = 22
    row_h = 16

    # Outer frame for the preview.
    b.rect_px(x_px=x0, y_px=y0, w_px=w, h_px=h, rgba_u8=(8, 8, 12, 235), outline_rgba_u8=(80, 80, 90, 255))
    b.text_px(x_px=x0 + 10, y_px=y0 + 6, text="SIGNAL WORKBENCH (PREVIEW)", kind="hdr")

    # Split inside the preview region.
    inner_x0 = x0 + pad
    inner_y0 = y0 + hdr_h + 6
    inner_w = max(1, w - pad * 2)
    inner_h = max(1, h - (hdr_h + 10) - pad)

    left_w = int(max(160, inner_w * 0.44))
    right_w = int(max(160, inner_w - left_w - pad))
    if left_w + right_w + pad > inner_w:
        right_w = max(1, inner_w - left_w - pad)

    left_x = int(inner_x0)
    right_x = int(inner_x0 + left_w + pad)
    panel_y = int(inner_y0)

    b.rect_px(x_px=left_x, y_px=panel_y, w_px=left_w, h_px=inner_h, rgba_u8=(6, 6, 8, 255), outline_rgba_u8=(60, 60, 70, 255))
    b.rect_px(x_px=right_x, y_px=panel_y, w_px=right_w, h_px=inner_h, rgba_u8=(6, 6, 8, 255), outline_rgba_u8=(60, 60, 70, 255))

    # Pull a small window of latest values.
    sig_count = 0
    vals: list[tuple[int, float]] = []
    try:
        from c_physics.controller_engine_api import try_load_controller_engine
        from c_physics.controller_engine_ctypes import GP_WheelMeta, GP_WheelSample

        lib, _ = try_load_controller_engine(search_dir="c_physics")
        if lib is not None and hasattr(lib, "gp_ctl_wheel_get_meta"):
            meta = GP_WheelMeta()
            if int(lib.gp_ctl_wheel_get_meta(ctypes.byref(meta))):
                sig_count = int(meta.signal_count)

        vis = max(6, int((inner_h - 10) / row_h))
        if sig_count > 0:
            signal_idx = max(0, min(int(signal_idx), sig_count - 1))
        first = 0
        if sig_count > vis:
            first = max(0, min(int(signal_idx) - vis // 2, sig_count - vis))

        if lib is not None and hasattr(lib, "gp_ctl_wheel_peek_latest"):
            for i in range(first, min(sig_count, first + vis)):
                smp = GP_WheelSample()
                wseq = ctypes.c_uint64(0)
                if int(lib.gp_ctl_wheel_peek_latest(int(i), ctypes.byref(smp), ctypes.byref(wseq))):
                    vals.append((int(i), float(smp.value)))
                else:
                    vals.append((int(i), 0.0))
    except Exception:
        vals = []

    # Left list.
    b.text_px(x_px=left_x + 8, y_px=panel_y + 6, text="SIG", kind="hdr")
    b.text_px(x_px=left_x + 58, y_px=panel_y + 6, text="VAL", kind="hdr")
    y_rows = int(panel_y + hdr_h)
    for row_i, (idx, v) in enumerate(vals):
        y = int(y_rows + row_i * row_h)
        is_sel = (idx == int(signal_idx))
        bg = (18, 18, 24, 255) if is_sel else (10, 10, 14, 255)
        b.rect_px(x_px=left_x + 4, y_px=y, w_px=left_w - 8, h_px=row_h - 2, rgba_u8=bg)
        b.text_px(x_px=left_x + 10, y_px=y + 2, text=f"{idx:4d}", kind="mono")
        b.text_px(x_px=left_x + 58, y_px=y + 2, text=f"{v:+0.3f}", kind="mono")

    # Right waveform.
    wf_x = int(right_x + 8)
    wf_y = int(panel_y + hdr_h)
    wf_w = int(max(1, right_w - 16))
    wf_h = int(max(1, inner_h - hdr_h - 10))
    b.text_px(x_px=right_x + 8, y_px=panel_y + 6, text=f"WAVE (sig {int(signal_idx)})", kind="hdr")
    b.wheel_waveform_px(x_px=wf_x, y_px=wf_y, w_px=wf_w, h_px=wf_h, signal_idx=int(signal_idx), span=1)


def build_workbench_uidoc(*, snap: Any, width_px: int, height_px: int):
    """Build a dense workbench-like UiDoc.

    This is intentionally a prototype: it establishes the overlay-source precedent
    and provides high information density without blocking.
    """

    from menu_full_overlay import UiDoc
    from menu_archetypes import UiDocBuilder

    b = UiDocBuilder(commands=[])
    gen = int(getattr(snap, "gen", 0))
    cursor = int(getattr(snap, "selected_idx", 0))
    sel = int(getattr(snap, "signal_idx", 0))

    b.clear((0, 0, 0, 220))

    # Layout
    pad = 14
    header_h = 26
    row_h = 18

    # Top-left text buttons.
    try:
        import input_interest

        cap = (str(input_interest.get_capture_origin() or "") == "workbench")
        mode = str(input_interest.get_mode() or "announce").strip().lower() or "announce"
        allow, deny = input_interest.get_effective_set()
        deny = set(deny or set())
        if allow is None:
            allow_n = -1
        else:
            allow_n = len(set(allow or set()) - deny)
    except Exception:
        cap = False
        mode = "announce"
        allow_n = -1

    btn_w = 240
    btn_h = 22
    btn_x = int(pad)
    btn_y0 = int(pad)
    btn_pad_y = 6

    cap_txt = f"CAPTURE: {'ON' if cap else 'OFF'}"
    if str(mode) == "scan":
        mode_txt = "INPUT: SCAN"
    else:
        mode_txt = f"INPUT: ANNOUNCE ({allow_n if allow_n >= 0 else 'ALL'})"

    cap_sel = bool(int(cursor) == 0)
    mode_sel = bool(int(cursor) == 1)

    # Button 1: CAPTURE
    b.rect_px(
        x_px=btn_x,
        y_px=btn_y0,
        w_px=btn_w,
        h_px=btn_h,
        rgba_u8=(18, 18, 24, 255) if cap_sel else (8, 8, 12, 255),
        outline_rgba_u8=(90, 90, 100, 255) if cap_sel else (60, 60, 70, 255),
    )
    b.text_px(x_px=btn_x + 8, y_px=btn_y0 + 4, text=cap_txt, kind="hud")

    # Button 2: MODE
    btn_y1 = int(btn_y0 + btn_h + btn_pad_y)
    b.rect_px(
        x_px=btn_x,
        y_px=btn_y1,
        w_px=btn_w,
        h_px=btn_h,
        rgba_u8=(18, 18, 24, 255) if mode_sel else (8, 8, 12, 255),
        outline_rgba_u8=(90, 90, 100, 255) if mode_sel else (60, 60, 70, 255),
    )
    b.text_px(x_px=btn_x + 8, y_px=btn_y1 + 4, text=mode_txt, kind="hud")

    # Layout: two top panels (inputs + signals) and a compact bottom strip for live channels.
    bottom_h = int(max(72, min(96, int(height_px * 0.14))))
    bottom_y0 = int(height_px - pad - bottom_h)

    top_y0 = int(pad)
    top_h = int(max(1, bottom_y0 - pad - top_y0))

    # Move divider farther right (wider left panel).
    left_w = int(max(520, width_px * 0.58))
    right_w = int(max(240, width_px - left_w - pad * 3))
    if left_w + right_w + pad * 3 > int(width_px):
        right_w = int(max(200, int(width_px) - int(left_w) - pad * 3))

    left_x0 = int(pad)
    left_y0 = int(top_y0)
    left_h = int(top_h)

    right_x0 = int(left_x0 + left_w + pad)
    right_y0 = int(top_y0)
    right_h = int(top_h)

    bottom_x0 = int(pad)
    bottom_w = int(max(1, int(width_px) - pad * 2))

    # Panels
    b.rect_px(x_px=left_x0, y_px=left_y0, w_px=left_w, h_px=left_h, rgba_u8=(8, 8, 12, 255), outline_rgba_u8=(70, 70, 80, 255))
    b.rect_px(x_px=right_x0, y_px=right_y0, w_px=right_w, h_px=right_h, rgba_u8=(8, 8, 12, 255), outline_rgba_u8=(70, 70, 80, 255))
    b.rect_px(x_px=bottom_x0, y_px=bottom_y0, w_px=bottom_w, h_px=bottom_h, rgba_u8=(8, 8, 12, 255), outline_rgba_u8=(70, 70, 80, 255))

    # Headers
    b.text_px(x_px=left_x0 + 8, y_px=left_y0 + 6, text="INPUTS", kind="hdr")
    b.text_px(x_px=right_x0 + 8, y_px=right_y0 + 6, text="SIGNALS (saved)", kind="hdr")
    b.text_px(x_px=bottom_x0 + 8, y_px=bottom_y0 + 6, text="LIVE CHANNEL", kind="hdr")

    # LEFT: legacy-style dense INPUT table (device headers, 9 LED flags for buttons, axis bars).
    from c_physics import signal_kernel_api
    import signal_status_flags

    # IMPORTANT:
    # The input table is meant to reflect *post-ingestion* inputs.
    # In announce-mode we only ingest (poll/push) the allowlist specs, so the
    # workbench should show only that allowlist by default.
    try:
        import input_interest

        allow, deny = input_interest.get_effective_set()
    except Exception:
        allow, deny = (None, set())
    deny = set(deny or set())
    allow_set: set[tuple[int, int, int]] | None
    if allow is None:
        allow_set = None
    else:
        allow_set = set(allow or set()) - deny

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

    now_ns = int(time.monotonic_ns())

    # Discover joystick shape (if present), but avoid hammering pygame/OS.
    joy_name, joy_axes, joy_buttons, joy_hats = _probe_joystick_shape_cached(now_ns=now_ns)

    # Build row list like legacy.
    # - scan-mode (allow_set is None): show the standard device inventory.
    # - announce-mode (allow_set is set): show only allowed specs (what is ingested).
    rows: list[tuple[str, Any]] = []

    if allow_set is None:
        rows.append(("device_header", {"label": "KEYBOARD"}))
        rows.append(("kbd_axis", {"id": 0, "label": "kbd.wasd.x"}))
        rows.append(("kbd_axis", {"id": 1, "label": "kbd.wasd.y"}))
        rows.append(("kbd_axis", {"id": 2, "label": "kbd.arrows.x"}))
        rows.append(("kbd_axis", {"id": 3, "label": "kbd.arrows.y"}))
        rows.append(("device_header", {"label": "MOUSE"}))
        rows.append(("mouse_motion", {"id": 0, "label": "mouse.dx"}))
        rows.append(("mouse_motion", {"id": 1, "label": "mouse.dy"}))
        rows.append(("device_header", {"label": f"JOYSTICK: {joy_name}" if joy_name else "JOYSTICK"}))
        for a in range(max(0, int(joy_axes))):
            rows.append(("joy_axis", {"id": int(a), "label": f"joy.axis.{int(a)}"}))
        for b_id in range(max(0, int(joy_buttons))):
            rows.append(("joy_button", {"id": int(b_id), "label": f"joy.btn.{int(b_id)}"}))
        if int(joy_hats) > 0:
            rows.append(("device_note", {"label": f"hats: {int(joy_hats)} (not shown)"}))
    else:
        # Announce-mode: only show what is allowed to be ingested.
        rows.append(("device_header", {"label": f"INGEST (announce): {len(allow_set)} specs"}))

        def _add_dev_header(lbl: str) -> None:
            if not rows or rows[-1][0] != "device_header" or str((rows[-1][1] or {}).get("label", "")) != str(lbl):
                rows.append(("device_header", {"label": str(lbl)}))

        # Keyboard axes
        kbd_axes = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_AXIS)})
        if kbd_axes:
            _add_dev_header("KEYBOARD")
            for kid in kbd_axes:
                nm = {0: "kbd.wasd.x", 1: "kbd.wasd.y", 2: "kbd.arrows.x", 3: "kbd.arrows.y"}.get(int(kid), f"kbd.axis.{int(kid)}")
                rows.append(("kbd_axis", {"id": int(kid), "label": str(nm)}))

        # Keyboard keys (edges)
        kbd_keys = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_KEYBOARD) and int(k) == int(signal_kernel_api.GP_EV_KEY)})
        if kbd_keys:
            _add_dev_header("KEYBOARD KEYS")
            for kid in kbd_keys[:32]:
                rows.append(("kbd_key", {"id": int(kid), "label": f"kbd.key.{int(kid)}"}))
            if len(kbd_keys) > 32:
                rows.append(("device_note", {"label": f"(+{len(kbd_keys) - 32} more keys)"}))

        # Mouse motion
        mm = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_MOUSE) and int(k) == int(signal_kernel_api.GP_EV_MOUSE_MOTION)})
        if mm:
            _add_dev_header("MOUSE")
            for mid in mm:
                nm = {0: "mouse.dx", 1: "mouse.dy"}.get(int(mid), f"mouse.motion.{int(mid)}")
                rows.append(("mouse_motion", {"id": int(mid), "label": str(nm)}))

        # Mouse buttons
        mb = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_MOUSE) and int(k) == int(signal_kernel_api.GP_EV_MOUSE_BUTTON)})
        if mb:
            _add_dev_header("MOUSE BUTTONS")
            for bid in mb:
                rows.append(("mouse_button", {"id": int(bid), "label": f"mouse.btn.{int(bid)}"}))

        # Joystick axes/buttons
        ja = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS) and int(iid) < 50000})
        jb = sorted({iid for (d, k, iid) in allow_set if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_BUTTON) and int(iid) < 51000})
        if ja or jb:
            _add_dev_header(f"JOYSTICK: {joy_name}" if joy_name else "JOYSTICK")
            for a in ja:
                rows.append(("joy_axis", {"id": int(a), "label": f"joy.axis.{int(a)}"}))
            for b_id in jb:
                rows.append(("joy_button", {"id": int(b_id), "label": f"joy.btn.{int(b_id)}"}))

    # Geometry
    table_x0 = int(left_x0 + 8)
    table_y0 = int(btn_y1 + btn_h + 12)
    table_w = int(left_w - 16)
    table_h = int(left_h - (table_y0 - left_y0) - 10)

    row_h2 = 18
    hdr_h2 = 18
    name_col_w = int(max(140, table_w * 0.34))
    led_col_w = 14
    led_r = 4
    bar_w = int(max(120, table_w - name_col_w - len(led_defs) * led_col_w - 120))

    # Table header row
    b.text_px(x_px=table_x0, y_px=table_y0, text="INPUTS", kind="hud")
    b.text_px(x_px=table_x0 + 4, y_px=table_y0 + hdr_h2, text="id", kind="mono")
    x_led0 = int(table_x0 + name_col_w)
    for i, (lbl, _bit) in enumerate(led_defs):
        b.text_px(x_px=x_led0 + i * led_col_w, y_px=table_y0 + hdr_h2, text=str(lbl), kind="mono")
    b.text_px(x_px=x_led0 + len(led_defs) * led_col_w + 10, y_px=table_y0 + hdr_h2, text="axis", kind="mono")
    b.text_px(x_px=x_led0 + len(led_defs) * led_col_w + 10 + bar_w + 10, y_px=table_y0 + hdr_h2, text="timers", kind="mono")

    # Visible window (no scrolling yet).
    first_row_y = int(table_y0 + hdr_h2 + 8)
    max_rows = max(1, int((table_h - hdr_h2 - 10) / row_h2))
    vis_rows = rows[: max_rows]

    # Optional: clear pulses, but throttle heavily (and disabled by default).
    global _workbench_last_pulse_clear_ns
    if bool(_WORKBENCH_CLEAR_PULSES):
        try:
            if int(now_ns) - int(_workbench_last_pulse_clear_ns or 0) >= int(_WORKBENCH_CLEAR_PULSES_INTERVAL_NS):
                _lib, api = _try_load_sigk_cached()
                if api is not None and hasattr(api, "gp_sigk_clear_pulses"):
                    api.gp_sigk_clear_pulses()
                    _workbench_last_pulse_clear_ns = int(now_ns)
        except Exception:
            pass

    def _axis_x_at(*, x0: int, w: int, v: float) -> int:
        vv = max(-1.0, min(1.0, float(v)))
        return int(x0 + (vv + 1.0) * 0.5 * float(w))

    for r_i, (kind, spec) in enumerate(vis_rows):
        y = int(first_row_y + r_i * row_h2)
        b.rect_px(x_px=table_x0, y_px=y, w_px=table_w, h_px=row_h2 - 2, rgba_u8=(10, 10, 14, 255))

        if kind == "device_header":
            lbl = str(spec.get("label", "")) if isinstance(spec, dict) else str(spec)
            b.rect_px(x_px=table_x0, y_px=y, w_px=table_w, h_px=row_h2 - 2, rgba_u8=(14, 14, 18, 255))
            b.text_px(x_px=table_x0 + 4, y_px=y + 2, text=str(lbl), kind="hdr")
            continue
        if kind == "device_note":
            lbl = str(spec.get("label", "")) if isinstance(spec, dict) else str(spec)
            b.text_px(x_px=table_x0 + 4, y_px=y + 2, text=str(lbl), kind="mono")
            continue

        # Resolve signal_id for kernel peek.
        label = ""
        signal_id = None
        is_button = False
        is_axis = False

        if isinstance(spec, dict):
            label = str(spec.get("label", ""))

        if kind == "kbd_axis":
            is_axis = True
            axis_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_AXIS), int(axis_id)))
        elif kind == "mouse_motion":
            is_axis = True
            axis_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_MOTION), int(axis_id)))
        elif kind == "joy_axis":
            is_axis = True
            axis_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(axis_id)))
        elif kind == "joy_button":
            is_button = True
            btn_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(btn_id)))
        elif kind == "kbd_key":
            is_button = True
            key_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_KEYBOARD), int(signal_kernel_api.GP_EV_KEY), int(key_id)))
        elif kind == "mouse_button":
            is_button = True
            btn_id = int(spec.get("id", 0)) if isinstance(spec, dict) else int(spec)
            signal_id = int(signal_kernel_api.compose_signal_id(int(signal_kernel_api.GP_DEV_MOUSE), int(signal_kernel_api.GP_EV_MOUSE_BUTTON), int(btn_id)))

        # Label column
        b.text_px(x_px=table_x0 + 4, y_px=y + 2, text=str(label) if label else str(kind), kind="mono")

        v, flags, hold_s, aux = (0.0, 0, 0.0, 0)
        if signal_id is not None:
            v, flags, hold_s, aux = _sigk_peek(now_ns=now_ns, signal_id=int(signal_id))

        # Button LED columns.
        if is_button:
            cy = int(y + (row_h2 // 2))
            for i, (_lbl, bit) in enumerate(led_defs):
                cx = int(x_led0 + i * led_col_w + 6)
                on = (int(flags) & int(bit)) != 0
                is_edge = bit in (
                    int(signal_status_flags.SIGF_DOWN_EDGE),
                    int(signal_status_flags.SIGF_UP_EDGE),
                    int(signal_status_flags.SIGF_TOGGLE_EDGE),
                    int(signal_status_flags.SIGF_DOUBLE_TOGGLE_EDGE),
                    int(signal_status_flags.SIGF_DOUBLE_EDGE),
                )
                wbe.draw_led_flag(b, cx_px=cx, cy_px=cy, r_px=led_r, on=bool(on), is_edge=bool(is_edge))

            # Timers: hold_s + last_hold_ms (aux)
            last_s = float(max(0.0, float(int(aux)) * 0.001))
            tx = int(x_led0 + len(led_defs) * led_col_w + 10 + bar_w + 10)
            b.text_px(x_px=tx, y_px=y + 2, text=f"h:{hold_s:0.2f} l:{last_s:0.2f}", kind="mono")

        # Axis bar column.
        if is_axis:
            bar_x0 = int(x_led0 + len(led_defs) * led_col_w + 10)
            bar_y0 = int(y + 6)
            bar_h = 6
            b.rect_px(x_px=bar_x0, y_px=bar_y0, w_px=bar_w, h_px=bar_h, rgba_u8=(12, 12, 16, 255), outline_rgba_u8=(60, 60, 70, 255))
            # ticks at -1,0,+1
            for t in (-1.0, 0.0, 1.0):
                xx = _axis_x_at(x0=bar_x0, w=bar_w, v=float(t))
                b.line_px(x0_px=xx, y0_px=bar_y0 - 2, x1_px=xx, y1_px=bar_y0 + bar_h + 2, rgba_u8=(200, 200, 200, 255), w_px=1)
            # current value
            xxv = _axis_x_at(x0=bar_x0, w=bar_w, v=float(v))
            b.line_px(x0_px=xxv, y0_px=bar_y0 - 3, x1_px=xxv, y1_px=bar_y0 + bar_h + 3, rgba_u8=(255, 255, 120, 255), w_px=2)
            tx = int(x_led0 + len(led_defs) * led_col_w + 10 + bar_w + 10)
            b.text_px(x_px=tx, y_px=y + 2, text=f"v:{float(v):+0.3f}", kind="mono")

    # RIGHT: saved signals table (legacy-style columns).
    cfg = _load_controller_cfg_cached(path="joystick.json")
    chans = _ctl_peek_channels(now_ns=now_ns)
    ch_sig_map: dict[int, dict[str, Any]] = {}
    for it in chans:
        try:
            ch_sig_map[int(it.get("ch", -1))] = it
        except Exception:
            continue

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
        return 1

    def _op_label(node: dict[str, Any]) -> str:
        op = str(node.get("op", ""))
        if op == "kernel":
            return "K"
        if op == "2dseek":
            return "2S"
        if op == "2dflightstick":
            return "2F"
        if op == "2dsumclamp":
            return "2+"
        if op == "const":
            return "C"
        if op == "add":
            return "+"
        if op == "sub":
            return "-"
        if op:
            return "?"
        return "1"

    # Table region.
    t_x0 = int(right_x0 + 8)
    t_y0 = int(right_y0 + header_h + 10)
    t_w = int(max(1, right_w - 16))
    t_h = int(max(1, right_y0 + right_h - t_y0 - 10))

    max_arg_leds = 12
    arg_gap = 10
    arg_r = 4
    op_w = 28
    ch_w = 46
    args_w = int(max_arg_leds * arg_gap + 10)
    name_w = int(max(80, t_w - args_w - op_w - ch_w - 24))

    b.text_px(x_px=t_x0 + 2, y_px=t_y0, text="NAME", kind="mono")
    b.text_px(x_px=t_x0 + name_w + 6, y_px=t_y0, text="ARGS", kind="mono")
    b.text_px(x_px=t_x0 + name_w + 6 + args_w + 2, y_px=t_y0, text="OP", kind="mono")
    b.text_px(x_px=t_x0 + name_w + 6 + args_w + op_w + 2, y_px=t_y0, text="CH", kind="mono")

    def _truncate(s: str, max_chars: int) -> str:
        ss = str(s)
        if max_chars <= 0:
            return ""
        if len(ss) <= max_chars:
            return ss
        return ss[: max(0, max_chars - 1)] + "…"

    if not isinstance(cfg, dict):
        b.text_px(x_px=t_x0 + 2, y_px=t_y0 + row_h, text="(joystick.json not loaded)", kind="mono")
    else:
        ctrl = _controller_block(cfg)
        sigs = ctrl.get("signals") if isinstance(ctrl, dict) else None
        if not isinstance(sigs, dict):
            sig_items: list[str] = ["NEW"]
        else:
            sig_items = ["NEW"] + _get_saved_signal_ids(cfg)

        # No scroll yet: show top window.
        max_rows_right = int(max(1, (t_h - row_h - 6) // row_h))
        vis_items = sig_items[: max_rows_right]

        for i, sid in enumerate(vis_items):
            y = int(t_y0 + row_h + 6 + i * row_h)
            b.rect_px(x_px=t_x0, y_px=y, w_px=t_w, h_px=row_h - 2, rgba_u8=(10, 10, 14, 255))

            if str(sid) == "NEW":
                b.text_px(x_px=t_x0 + 2, y_px=y + 2, text="NEW", kind="mono")
                continue

            node = sigs.get(str(sid)) if isinstance(sigs, dict) else None
            if not isinstance(node, dict):
                node = {"op": "kernel", "kernel": "I"}

            op_lbl = _op_label(node)
            req = int(_op_required_args(node))
            args = node.get("args")
            args_list = list(args) if isinstance(args, list) else []
            ch = _mapped_channel_for_signal(cfg, str(sid))
            ch_lbl = "-" if ch is None else str(int(ch))

            b.text_px(x_px=t_x0 + 2, y_px=y + 2, text=_truncate(str(sid), max(8, name_w // 9)), kind="mono")

            # Arg LEDs (required slots are outlined by brighter fill when bound).
            x_args0 = int(t_x0 + name_w + 10)
            cy = int(y + (row_h // 2))
            for a_i in range(max_arg_leds):
                cx = int(x_args0 + a_i * arg_gap)
                bound = bool(a_i < len(args_list) and isinstance(args_list[a_i], dict) and bool(args_list[a_i]))
                required = bool(a_i < int(req))
                wbe.draw_led_bind(b, cx_px=cx, cy_px=cy, r_px=arg_r, bound=bool(bound), required=bool(required))

            x_op = int(t_x0 + name_w + 10 + args_w)
            b.text_px(x_px=x_op, y_px=y + 2, text=str(op_lbl), kind="mono")
            x_ch = int(x_op + op_w)
            b.text_px(x_px=x_ch, y_px=y + 2, text=f"{ch_lbl}", kind="mono")

            # Waveform widget for signals: if this signal is mapped to an output channel,
            # reuse the live-channel history to draw a tiny sparkline.
            if ch is not None:
                try:
                    ch_i = int(ch)
                    sig_ent = ch_sig_map.get(int(ch_i))
                    sig_idx = int(sig_ent.get("sig_idx_base", 0)) if isinstance(sig_ent, dict) else None
                    if sig_idx is not None:
                        sp_x = int(x_ch + 18)
                        sp_y = int(y + 4)
                        sp_w = int(max(10, ch_w - 22))
                        sp_h = int(max(8, row_h - 8))
                        b.wheel_waveform_px(x_px=sp_x, y_px=sp_y, w_px=sp_w, h_px=sp_h, signal_idx=int(sig_idx), span=1)
                except Exception:
                    pass

    # BOTTOM: live channel selector (single channel + large sparkline).
    inner_x = int(bottom_x0 + 8)
    inner_y = int(bottom_y0 + header_h + 6)
    inner_w = int(max(1, bottom_w - 16))
    inner_h = int(max(1, bottom_h - header_h - 12))

    if not chans:
        b.text_px(x_px=inner_x + 2, y_px=inner_y + 2, text="(no controller outputs)", kind="mono")
    else:
        # Choose which channel to display.
        # For now this is implicit (future: make selectable):
        # - prefer the currently-selected signal's mapped channel
        # - else fallback to first output channel
        preferred_ch: int | None = None
        try:
            cfg2 = _load_controller_cfg_cached(path="joystick.json")
            if isinstance(cfg2, dict):
                sigs2 = _get_saved_signal_ids(cfg2)
                sig_sel2 = int(getattr(snap, "signal_idx", 0))
                if sigs2 and 0 <= sig_sel2 < len(sigs2):
                    preferred_ch = _mapped_channel_for_signal(cfg2, str(sigs2[sig_sel2]))
        except Exception:
            preferred_ch = None

        ch_sel = int(preferred_ch) if preferred_ch is not None else int(chans[0].get("ch", 0))
        sel_it = None
        for it in chans:
            try:
                if int(it.get("ch", -9999)) == int(ch_sel):
                    sel_it = it
                    break
            except Exception:
                continue
        if sel_it is None:
            sel_it = chans[0]
            ch_sel = int(sel_it.get("ch", 0))

        dim = int(sel_it.get("dim", 1))
        vals = list(sel_it.get("values", []) or [])
        v0 = float(vals[0]) if vals else 0.0
        v1 = float(vals[1]) if (dim == 2 and len(vals) > 1) else 0.0
        eps = 0.08
        act = max(abs(v0), abs(v1)) if dim == 2 else abs(v0)

        # Single-line label + activity dot.
        b.text_px(x_px=inner_x + 2, y_px=inner_y + 2, text=f"ch {int(ch_sel):d}", kind="mono")
        if dim == 2:
            b.text_px(x_px=inner_x + 70, y_px=inner_y + 2, text=f"x:{v0:+0.3f}  y:{v1:+0.3f}", kind="mono")
        else:
            b.text_px(x_px=inner_x + 70, y_px=inner_y + 2, text=f"v:{v0:+0.3f}", kind="mono")
        wbe.draw_led_activity(b, cx_px=int(inner_x + inner_w - 10), cy_px=int(inner_y + 10), r_px=5, active=bool(act > eps))

        # Large waveform via controller wheel raster; no Python-side history kept.
        wf_sig_idx = int(sel_it.get("sig_idx_base", 0))
        sp_pad_top = 18
        sp_x = int(inner_x + 2)
        sp_y = int(inner_y + sp_pad_top)
        sp_w = int(max(1, inner_w - 4))
        sp_h = int(max(1, inner_h - sp_pad_top - 2))
        b.wheel_waveform_px(x_px=sp_x, y_px=sp_y, w_px=sp_w, h_px=sp_h, signal_idx=int(wf_sig_idx), span=1)

    return UiDoc(gen=gen, commands=b.commands)
