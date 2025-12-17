from __future__ import annotations

from typing import Any

from c_physics.controller_engine_ctypes import GP_CtlTimerDesc


# Keep in sync with c_physics/controller_engine_abi.h
GP_CTL_TIMER_REPEAT = 1


def _ticks_from_spec(*, it: dict[str, Any], key_ticks: str, key_s: str, key_ms: str, tick_hz: int, default_ticks: int) -> int:
    if key_ticks in it:
        try:
            return max(1, int(it.get(key_ticks)))
        except Exception:
            return max(1, int(default_ticks))

    if key_ms in it:
        try:
            ms = float(it.get(key_ms))
        except Exception:
            ms = 0.0
        ticks = int(round((ms * 0.001) * float(tick_hz)))
        return max(1, ticks)

    if key_s in it:
        try:
            s = float(it.get(key_s))
        except Exception:
            s = 0.0
        ticks = int(round(float(s) * float(tick_hz)))
        return max(1, ticks)

    return max(1, int(default_ticks))


def _ticks_from_phase(*, it: dict[str, Any], key_ticks: str, key_s: str, key_ms: str, tick_hz: int) -> int:
    if key_ticks in it:
        try:
            return max(0, int(it.get(key_ticks)))
        except Exception:
            return 0

    if key_ms in it:
        try:
            ms = float(it.get(key_ms))
        except Exception:
            ms = 0.0
        ticks = int(round((ms * 0.001) * float(tick_hz)))
        return max(0, ticks)

    if key_s in it:
        try:
            s = float(it.get(key_s))
        except Exception:
            s = 0.0
        ticks = int(round(float(s) * float(tick_hz)))
        return max(0, ticks)

    return 0


def load_timer_descs_from_cfg(*, cfg: dict[str, Any], tick_hz: int = 240) -> list[GP_CtlTimerDesc]:
    """Parse controller timers from joystick.json-style cfg.

    Expected schema (optional):

    cfg["flight_controls"]["controller"]["timers"] = [
      {
        "timer_id": 0,
        "period_ticks": 240,   # OR period_s / period_ms
        "duty_ticks": 1,       # OR duty_s / duty_ms
        "phase_ticks": 0,      # OR phase_s / phase_ms
        "repeat": true
      }
    ]

    Notes:
    - Timers are registered into the C controller engine (gp_ctl_*), so they
      only function when the C controller engine is in use.
    """

    out: list[GP_CtlTimerDesc] = []

    fc = cfg.get("flight_controls") if isinstance(cfg, dict) else None
    ctrl = fc.get("controller") if isinstance(fc, dict) else None
    timers = ctrl.get("timers") if isinstance(ctrl, dict) else None

    if not isinstance(timers, list):
        return out

    for it0 in timers:
        if not isinstance(it0, dict):
            continue
        it = dict(it0)

        try:
            timer_id = int(it.get("timer_id"))
        except Exception:
            continue
        if timer_id < 0:
            continue

        period_ticks = _ticks_from_spec(
            it=it,
            key_ticks="period_ticks",
            key_s="period_s",
            key_ms="period_ms",
            tick_hz=int(tick_hz),
            default_ticks=int(tick_hz),
        )

        duty_ticks = _ticks_from_spec(
            it=it,
            key_ticks="duty_ticks",
            key_s="duty_s",
            key_ms="duty_ms",
            tick_hz=int(tick_hz),
            default_ticks=1,
        )

        phase_ticks = _ticks_from_phase(
            it=it,
            key_ticks="phase_ticks",
            key_s="phase_s",
            key_ms="phase_ms",
            tick_hz=int(tick_hz),
        )

        repeat = bool(it.get("repeat", True))
        flags = int(GP_CTL_TIMER_REPEAT) if repeat else 0

        td = GP_CtlTimerDesc()
        td.timer_id = int(timer_id)
        td.period_ticks = int(period_ticks)
        td.duty_ticks = int(duty_ticks)
        td.phase_ticks = int(phase_ticks)
        td.flags = int(flags)
        td._reserved0 = 0
        out.append(td)

    return out
