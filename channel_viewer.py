from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os
import time
import ctypes

import pygame

import controller_backend
import controller_graph_compile

from c_physics import signal_kernel_api
from c_physics import controller_engine_api
from c_physics.controller_engine_ctypes import GP_CtlMeta, GP_CtlOutputDesc
from c_physics.signal_kernel_ctypes import GP_InputEvent


@dataclass
class WatchedChannel:
    channel: int
    dim: int
    sid: str
    color: tuple[int, int, int]


def _hsv_to_rgb255(h: float, s: float, v: float) -> tuple[int, int, int]:
    # Minimal HSV->RGB (0..1 floats -> 0..255 ints)
    h = float(h) % 1.0
    s = float(max(0.0, min(1.0, float(s))))
    v = float(max(0.0, min(1.0, float(v))))

    i = int(h * 6.0)
    f = h * 6.0 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return (int(r * 255), int(g * 255), int(b * 255))


def _auto_color(i: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping for distinct colors.
    hue = (0.61803398875 * float(max(0, int(i)))) % 1.0
    return _hsv_to_rgb255(hue, 0.65, 0.95)


def run_channel_viewer(
    *,
    font: pygame.font.Font,
    width: int,
    height: int,
    joystick: pygame.joystick.Joystick | None,
    menu_button: int | None,
    load_or_create_joystick_config,
    get_menu_nav,
    get_menu_scroll,
    nav_edge,
    poll_joystick_snapshot,
) -> None:
    """Live channel viewer.

    - Press 'A' to add a watched channel from the compiled output list. Each gets an auto color.
    - ESC / menu button exits.

    Displays a scrolling history of sampled controller-backend channel outputs.
    """

    if joystick is None:
        return

    # Ensure controller graph is compiled so ControllerBackend can load final graph.
    try:
        controller_graph_compile.try_build_final_graph(
            joystick_path="joystick.json",
            mixer_path="channel_mixer.json",
            compiled_out_path="controller_graph_compiled.json",
            final_out_path="controller_graph_final.json",
        )
    except Exception:
        pass

    _sigk_lib, sigk = signal_kernel_api.try_load_signal_kernel()
    if sigk is None:
        return
    try:
        sigk.gp_sigk_reset()
    except Exception:
        pass

    backend: controller_backend.ControllerBackend | None = controller_backend.ControllerBackend(sigk)

    # Input policy: (allow, deny) set model.
    # allow is None => allow all (scan/listen). allow is set() => allow none (quiet announce).
    try:
        import input_interest

        get_effective_set = input_interest.get_effective_set
    except Exception:

        def get_effective_set():  # type: ignore[no-redef]
            return (None, set())

    # Prefer the C controller engine if present.
    ctl = None
    try:
        controller_engine_api.bind_controller_engine(_sigk_lib)
        ctl = _sigk_lib
    except Exception:
        ctl = None

    clock = pygame.time.Clock()

    watched: list[WatchedChannel] = []

    # Picker state
    picker_active: bool = False
    picker_sel: int = 0
    picker_items: list[dict[str, Any]] = []

    # Backend sampling rate (C engine thread rate or Python fallback rate)
    backend_hz: float = 240.0
    sample_period_ns: int = int(1e9 / max(1.0, float(backend_hz)))
    next_sample_ns: int = 0

    ctl_meta = GP_CtlMeta()
    ctl_outputs: dict[int, GP_CtlOutputDesc] = {}
    ctl_buf = None
    ctl_last_seq: int = 0
    ctl_ready: bool = False

    if ctl is not None:
        try:
            # Load graph blob and start fixed-rate evaluator.
            ok_load = int(ctl.gp_ctl_load_graph_file(b"controller_graph_final.bin"))
            if ok_load:
                # Start thread at backend_hz (defaults to 240 if 0).
                ctl.gp_ctl_start(int(backend_hz), 4096)
                ctl.gp_ctl_get_meta(ctypes.byref(ctl_meta))
                n_out = int(ctl_meta.output_count)
                if n_out > 0:
                    arr_t = GP_CtlOutputDesc * n_out
                    arr = arr_t()
                    got = int(ctl.gp_ctl_get_outputs(arr, int(n_out)))
                    for i in range(int(got)):
                        d = arr[int(i)]
                        ctl_outputs[int(d.channel)] = d
                if int(ctl_meta.total_floats) > 0:
                    ctl_buf = (ctypes.c_float * int(ctl_meta.total_floats))()
                    ctl_ready = True
        except Exception:
            ctl_ready = False

    # History as parallel arrays (for speed/simplicity)
    hist_t_ns: list[int] = []
    hist_vals: list[list[Any]] = []  # each row: values per watched[] (float or (x,y))

    def _load_compiled_channels() -> list[dict[str, Any]]:
        # Prefer final graph (has mixer routing applied).
        path = "controller_graph_final.json"
        try:
            st = os.stat(path)
            _ = st.st_mtime
        except Exception:
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
        except Exception:
            return []
        fc = g.get("final_channels")
        if not isinstance(fc, dict):
            return []
        out: list[dict[str, Any]] = []
        for k, spec in fc.items():
            if not isinstance(spec, dict):
                continue
            try:
                ch = int(k)
            except Exception:
                continue
            try:
                dim = int(spec.get("dim", 1))
            except Exception:
                dim = 1
            sid = str(spec.get("sid", ""))
            src = str(spec.get("from", ""))
            out.append({"ch": int(ch), "dim": int(2 if dim == 2 else 1), "sid": sid, "from": src})
        out.sort(key=lambda it: int(it.get("ch", 0)))
        return out

    def _max_rows() -> int:
        # leave some room for header/stats
        row_h = max(14, int(font.get_linesize()) + 2)
        usable = int(height - row_h * 6)
        return max(10, int(usable / row_h))

    def _append_sample(t_ns: int, values: list[Any]) -> None:
        hist_t_ns.append(int(t_ns))
        hist_vals.append(list(values))
        # Keep a bounded history: 4x visible rows for simple stats.
        max_keep = int(_max_rows() * 4)
        if len(hist_t_ns) > max_keep:
            drop = len(hist_t_ns) - max_keep
            del hist_t_ns[:drop]
            del hist_vals[:drop]

    def _scalar_from_value(v: Any) -> float:
        # Normalize values for stats: for 2D use max-norm.
        if isinstance(v, tuple) and len(v) == 2:
            try:
                x = float(v[0])
                y = float(v[1])
                return float(max(abs(x), abs(y)))
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    def _series_stats(series_idx: int, *, window_s: float = 3.0) -> tuple[float, float, float, float, float]:
        """Return (last, mean, vmin, vmax, act_hz) over recent window.

        For 2D channels, the scalar used is max(|x|,|y|).
        """
        if not hist_t_ns or not hist_vals:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        last = _scalar_from_value(hist_vals[-1][series_idx]) if series_idx < len(hist_vals[-1]) else 0.0

        t0 = int(hist_t_ns[-1] - int(window_s * 1e9))
        xs: list[float] = []
        ts: list[int] = []
        for t, row in zip(hist_t_ns, hist_vals):
            if int(t) < int(t0):
                continue
            if series_idx >= len(row):
                continue
            ts.append(int(t))
            xs.append(_scalar_from_value(row[series_idx]))

        if not xs:
            return last, last, last, last, 0.0

        vmin = min(xs)
        vmax = max(xs)
        mean = sum(xs) / float(len(xs))

        # Simple "human actuation" estimate: count significant deltas.
        # This is intentionally low-frequency-ish: thresholded, rate over window.
        thr = 0.15
        acts = 0
        for a, b in zip(xs, xs[1:]):
            if abs(float(b) - float(a)) >= float(thr):
                acts += 1
        dur_s = float(max(1e-6, (ts[-1] - ts[0]) / 1e9)) if len(ts) >= 2 else float(window_s)
        act_hz = float(acts) / float(max(1e-6, dur_s))
        return last, float(mean), float(vmin), float(vmax), float(act_hz)

    # Minimal OpenGL drawing (same approach as signal_workbench).
    try:
        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_RGBA,
            GL_SRC_ALPHA,
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
    except Exception:
        return

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

    # Track previous snapshot for edge detection.
    axes_prev: dict[int, float] = {}
    buttons_prev: set[int] = set()
    hats_prev: dict[int, tuple[int, int]] = {}

    # Start baseline snapshot.
    try:
        axes_prev, buttons_now0, hats_prev = poll_joystick_snapshot(joystick)
        buttons_prev = set(buttons_now0)
    except Exception:
        axes_prev, buttons_prev, hats_prev = {}, set(), {}

    # Picker initial list.
    picker_items = _load_compiled_channels()

    # UI helpers using existing menu nav binding (scroll not used today).
    nav = None
    sc = None

    while True:
        now_ns = int(time.perf_counter_ns())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if picker_active:
                        picker_active = False
                    else:
                        return
                if event.key == pygame.K_a:
                    # Open picker.
                    picker_items = _load_compiled_channels()
                    picker_sel = int(max(0, min(int(picker_sel), max(0, len(picker_items) - 1))))
                    picker_active = True
                if picker_active and event.key in (pygame.K_UP, pygame.K_w):
                    picker_sel = int(max(0, int(picker_sel) - 1))
                if picker_active and event.key in (pygame.K_DOWN, pygame.K_s):
                    picker_sel = int(min(max(0, len(picker_items) - 1), int(picker_sel) + 1))
                if picker_active and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if 0 <= int(picker_sel) < len(picker_items):
                        it = picker_items[int(picker_sel)]
                        watched.append(
                            WatchedChannel(
                                channel=int(it.get("ch", 0)),
                                dim=int(it.get("dim", 1)),
                                sid=str(it.get("sid", "")),
                                color=_auto_color(len(watched)),
                            )
                        )
                    picker_active = False
                if event.key == pygame.K_c:
                    watched.clear()
                    hist_t_ns.clear()
                    hist_vals.clear()
                    picker_active = False

            if menu_button is not None and event.type == pygame.JOYBUTTONDOWN and int(event.button) == int(menu_button):
                return

        # Reload nav bindings (for consistent exit behavior if user rebinds).
        try:
            cfg = load_or_create_joystick_config("joystick.json")
            nav = get_menu_nav(cfg)
            sc = get_menu_scroll(cfg)
        except Exception:
            nav = None
            sc = None

        # Poll current snapshot.
        allow, deny = get_effective_set()
        allow_all = allow is None
        if allow_all:
            try:
                axes_now, buttons_now, hats_now = poll_joystick_snapshot(joystick)
            except Exception:
                axes_now, buttons_now, hats_now = {}, set(), {}
        else:
            want_buttons = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_BUTTON)}
            want_axes = {int(iid) for (d, k, iid) in allow if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS)}
            try:
                axes_now, buttons_now, hats_now = poll_joystick_snapshot(
                    joystick,
                    axes_ids=want_axes,
                    button_ids=want_buttons,
                    hat_ids=None,  # keep hats available for menu-nav bindings
                )
            except Exception:
                axes_now, buttons_now, hats_now = {}, set(), {}

        # Menu-nav based picker controls (joystick-driven).
        if picker_active and nav:
            try:
                if nav_edge(nav.get("up"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                    picker_sel = int(max(0, int(picker_sel) - 1))
                if nav_edge(nav.get("down"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                    picker_sel = int(min(max(0, len(picker_items) - 1), int(picker_sel) + 1))
                if nav_edge(nav.get("confirm"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                    if 0 <= int(picker_sel) < len(picker_items):
                        it = picker_items[int(picker_sel)]
                        watched.append(
                            WatchedChannel(
                                channel=int(it.get("ch", 0)),
                                dim=int(it.get("dim", 1)),
                                sid=str(it.get("sid", "")),
                                color=_auto_color(len(watched)),
                            )
                        )
                    picker_active = False
                if nav_edge(nav.get("cancel"), axes_now=axes_now, axes_prev=axes_prev, buttons_now=buttons_now, buttons_prev=buttons_prev, hats_now=hats_now, hats_prev=hats_prev):
                    picker_active = False
            except Exception:
                pass

        # Feed kernel with per-frame axis values and button edges.
        try:
            evs: list[GP_InputEvent] = []

            if allow_all:
                try:
                    nb = int(joystick.get_numbuttons())
                except Exception:
                    nb = 0
                button_iter = [int(b) for b in range(max(0, nb))]

                try:
                    na = int(joystick.get_numaxes())
                except Exception:
                    na = 0
                axis_iter = [int(a) for a in range(max(0, na))]
            else:
                button_iter = sorted({int(iid) for (d, k, iid) in (allow or set()) if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_BUTTON)})
                axis_iter = sorted({int(iid) for (d, k, iid) in (allow or set()) if int(d) == int(signal_kernel_api.GP_DEV_JOYSTICK) and int(k) == int(signal_kernel_api.GP_EV_AXIS)})

            # Button edges
            for b in button_iter:
                spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_BUTTON), int(b))
                if spec in deny:
                    continue
                was = int(b) in buttons_prev
                isd = int(b) in buttons_now
                if was == isd:
                    continue
                evs.append(
                    GP_InputEvent(
                        t_mono_ns=now_ns,
                        device=int(signal_kernel_api.GP_DEV_JOYSTICK),
                        kind=int(signal_kernel_api.GP_EV_BUTTON),
                        id=int(b),
                        v0=1.0 if isd else 0.0,
                        v1=0.0,
                        flags=0,
                    )
                )

            # Axis values
            for a in axis_iter:
                spec = (int(signal_kernel_api.GP_DEV_JOYSTICK), int(signal_kernel_api.GP_EV_AXIS), int(a))
                if spec in deny:
                    continue
                vv = float(axes_now.get(int(a), 0.0))
                evs.append(
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

            if evs:
                arr_t = GP_InputEvent * len(evs)
                sigk.gp_sigk_push_events(arr_t(*evs), int(len(evs)))
        except Exception:
            pass

        # Evaluate backend channels.
        if ctl_ready and ctl is not None and ctl_buf is not None:
            # Catch up on all backend ticks since last UI frame.
            try:
                write_seq = int(ctl.gp_ctl_get_write_seq())
            except Exception:
                write_seq = 0

            if ctl_last_seq == 0:
                ctl_last_seq = int(write_seq)

            # Bound catch-up to avoid spiraling.
            max_catch = 512
            start_seq = int(ctl_last_seq + 1)
            if write_seq - start_seq > max_catch:
                start_seq = int(write_seq - max_catch)

            def _read_channel_from_ctl(ch: int, dim: int) -> Any:
                d = ctl_outputs.get(int(ch))
                if d is None:
                    return (0.0, 0.0) if int(dim) == 2 else 0.0
                off = int(getattr(d, "offset", 0))
                stride = int(getattr(d, "stride", 1))
                ddim = int(getattr(d, "dim", dim))
                if ddim == 2:
                    try:
                        x = float(ctl_buf[off + 0 * stride])
                        y = float(ctl_buf[off + 1 * stride])
                        return (x, y)
                    except Exception:
                        return (0.0, 0.0)
                try:
                    return float(ctl_buf[off])
                except Exception:
                    return 0.0

            for seq in range(int(start_seq), int(write_seq + 1)):
                t_ns = ctypes.c_uint64(0)
                ok = 0
                try:
                    ok = int(ctl.gp_ctl_peek_seq(int(seq), ctl_buf, int(ctl_meta.total_floats), ctypes.byref(t_ns)))
                except Exception:
                    ok = 0
                if not ok:
                    continue

                if watched:
                    row: list[Any] = []
                    for w in watched:
                        row.append(_read_channel_from_ctl(int(w.channel), int(w.dim)))
                    _append_sample(int(t_ns.value), row)

            ctl_last_seq = int(write_seq)
        else:
            # Python fallback evaluator (still kernel-sourced).
            if next_sample_ns == 0:
                next_sample_ns = int(now_ns)
            while int(now_ns) >= int(next_sample_ns):
                overrides_1d, channels_2d = backend.step(now_ns=int(next_sample_ns)) if backend is not None else ({}, {})
                if watched:
                    row: list[Any] = []
                    for w in watched:
                        if int(w.dim) == 2:
                            row.append(tuple(channels_2d.get(int(w.channel), (0.0, 0.0))))
                        else:
                            row.append(float(overrides_1d.get(int(w.channel), 0.0)))
                    _append_sample(int(next_sample_ns), row)
                next_sample_ns = int(next_sample_ns + sample_period_ns)

        # Draw.
        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        if depth_was_enabled:
            glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Header
        y = 10
        _draw_text_px(10, y, "VIEW CHANNEL (live)", (255, 255, 255))
        y += int(font.get_linesize()) + 2
        if picker_active:
            _draw_text_px(10, y, "PICK CHANNEL: up/down + confirm (Enter)  cancel (Esc/menu)", (220, 220, 220))
        else:
            if ctl_ready and ctl is not None:
                try:
                    ctl.gp_ctl_get_meta(ctypes.byref(ctl_meta))
                    hz_s = int(getattr(ctl_meta, "tick_hz", int(backend_hz)))
                except Exception:
                    hz_s = int(backend_hz)
                _draw_text_px(10, y, f"A: add channel   C: clear   backend C@{hz_s}Hz   Esc/menu: back", (180, 180, 180))
            else:
                _draw_text_px(10, y, f"A: add channel   C: clear   backend PY@{backend_hz:.0f}Hz   Esc/menu: back", (180, 180, 180))
        y += int(font.get_linesize()) + 6

        if picker_active:
            # Simple picker overlay.
            if not picker_items:
                _draw_text_px(10, y, "(no compiled channels found)", (200, 200, 200))
                y += int(font.get_linesize()) + 4
            else:
                # Show up to ~12 items.
                max_vis = 12
                start = int(max(0, min(int(picker_sel) - 5, max(0, len(picker_items) - max_vis))))
                end = int(min(len(picker_items), start + max_vis))
                for i in range(start, end):
                    it = picker_items[int(i)]
                    ch = int(it.get("ch", 0))
                    dim = int(it.get("dim", 1))
                    sid = str(it.get("sid", ""))
                    src = str(it.get("from", ""))
                    prefix = ">" if int(i) == int(picker_sel) else " "
                    col = (255, 255, 255) if int(i) == int(picker_sel) else (180, 180, 180)
                    _draw_text_px(10, y, f"{prefix} ch{ch}  dim{dim}  sid={sid}  from={src}", col)
                    y += int(font.get_linesize()) + 2
                y += 6
        elif not watched:
            _draw_text_px(10, y, "(no watched channels) press A to add", (200, 200, 200))
            y += int(font.get_linesize()) + 4
        else:
            # Legend + low-frequency stats line(s)
            for i, w in enumerate(watched):
                last, mean, vmin, vmax, act_hz = _series_stats(i, window_s=3.0)
                label = f"ch{int(w.channel)} dim{int(w.dim)} sid={w.sid}  last {last:+.2f}  mean {mean:+.2f}  [{vmin:+.2f},{vmax:+.2f}]  act {act_hz:.2f}Hz"
                _draw_text_px(10, y, label, w.color)
                y += int(font.get_linesize()) + 2
            y += 6

        # Table area background
        table_y0 = int(y)
        _draw_rect_px(0, table_y0, int(width), int(height - table_y0), (0.0, 0.0, 0.0, 1.0))

        # Column headers
        col_x0 = 10
        col_w = max(90, int(width / max(1, (len(watched) + 1))))
        _draw_text_px(col_x0, table_y0, "t(s)", (160, 160, 160))
        for i, w in enumerate(watched):
            _draw_text_px(col_x0 + col_w * (i + 1), table_y0, f"ch{int(w.channel)}", w.color)

        # Rows: show newest at bottom (scrolling up)
        row_h = max(14, int(font.get_linesize()) + 2)
        max_rows = _max_rows()
        rows = list(zip(hist_t_ns, hist_vals))
        rows = rows[-max_rows:] if len(rows) > max_rows else rows

        if rows:
            t_ref = int(rows[-1][0])
            # Start drawing so the last row ends at bottom-ish.
            y0 = int(height - row_h * (len(rows) + 1))
            y0 = max(table_y0 + row_h + 4, y0)

            for j, (t_ns, vals) in enumerate(rows):
                yy = int(y0 + row_h * (j + 1))
                dt_s = float((int(t_ns) - int(t_ref)) / 1e9)
                _draw_text_px(col_x0, yy, f"{dt_s:+.2f}", (120, 120, 120))
                for i, w in enumerate(watched):
                    vv = vals[i] if i < len(vals) else 0.0
                    if int(w.dim) == 2 and isinstance(vv, tuple) and len(vv) == 2:
                        try:
                            x = float(vv[0])
                            yv = float(vv[1])
                        except Exception:
                            x, yv = 0.0, 0.0
                        _draw_text_px(col_x0 + col_w * (i + 1), yy, f"({x:+.2f},{yv:+.2f})", w.color)
                    else:
                        try:
                            v = float(vv)
                        except Exception:
                            v = 0.0
                        _draw_text_px(col_x0 + col_w * (i + 1), yy, f"{v:+.3f}", w.color)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()
        clock.tick(60)

        axes_prev, buttons_prev, hats_prev = axes_now, set(buttons_now), dict(hats_now)
