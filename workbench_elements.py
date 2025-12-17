from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


RGBA = tuple[int, int, int, int]


# Shared rolling histories used by workbench waveform elements.
# Keyed by ("ch", ch, comp) where comp is 0 or 1.
WB_CH_HISTORY: dict[tuple[str, int, int], list[float]] = {}
WB_CH_DIM: dict[int, int] = {}


def clear_histories() -> None:
    """Reset cached channel histories and dimensions."""

    WB_CH_HISTORY.clear()
    WB_CH_DIM.clear()


# Palette: match legacy workbench semantics.
# - Level/state LEDs: green on, dark-green off.
# - Edge/toggle LEDs: white on, dark-gray off.
LED_LEVEL_ON: RGBA = (64, 255, 64, 255)
LED_LEVEL_OFF: RGBA = (26, 52, 26, 255)
LED_EDGE_ON: RGBA = (255, 255, 255, 255)
LED_EDGE_OFF: RGBA = (48, 48, 52, 255)

# Binding dots (signal args): still use the same primitive, but a different intensity ramp.
LED_BIND_REQUIRED_ON: RGBA = (180, 255, 180, 255)
LED_BIND_OPTIONAL_ON: RGBA = (140, 200, 140, 255)
LED_BIND_REQUIRED_OFF: RGBA = (70, 70, 80, 255)
LED_BIND_OPTIONAL_OFF: RGBA = (40, 40, 48, 255)

# Activity dot (channels): reuse the same green/gray language.
LED_ACTIVITY_ON: RGBA = LED_BIND_REQUIRED_ON
LED_ACTIVITY_OFF: RGBA = LED_BIND_REQUIRED_OFF


def push_history(history: dict[Any, list[float]], *, key: Any, value: float, max_len: int = 64) -> None:
    """Append a sample into a bounded list history."""

    if max_len <= 0:
        return
    buf = history.get(key)
    if not isinstance(buf, list):
        buf = []
        history[key] = buf
    buf.append(float(value))
    extra = len(buf) - int(max_len)
    if extra > 0:
        del buf[0:extra]


def draw_led_flag(
    b: Any,
    *,
    cx_px: int,
    cy_px: int,
    r_px: int,
    on: bool,
    is_edge: bool,
) -> None:
    col = (LED_EDGE_ON if on else LED_EDGE_OFF) if bool(is_edge) else (LED_LEVEL_ON if on else LED_LEVEL_OFF)
    b.circle_px(cx_px=int(cx_px), cy_px=int(cy_px), r_px=int(r_px), rgba_u8=col)


def draw_led_bind(
    b: Any,
    *,
    cx_px: int,
    cy_px: int,
    r_px: int,
    bound: bool,
    required: bool,
) -> None:
    if bound:
        col = LED_BIND_REQUIRED_ON if bool(required) else LED_BIND_OPTIONAL_ON
    else:
        col = LED_BIND_REQUIRED_OFF if bool(required) else LED_BIND_OPTIONAL_OFF
    b.circle_px(cx_px=int(cx_px), cy_px=int(cy_px), r_px=int(r_px), rgba_u8=col)


def draw_led_activity(b: Any, *, cx_px: int, cy_px: int, r_px: int, active: bool) -> None:
    col = LED_ACTIVITY_ON if bool(active) else LED_ACTIVITY_OFF
    b.circle_px(cx_px=int(cx_px), cy_px=int(cy_px), r_px=int(r_px), rgba_u8=col)


def draw_sparkline_px(
    b: Any,
    *,
    x_px: int,
    y_px: int,
    w_px: int,
    h_px: int,
    samples_a: Iterable[float] | None,
    samples_b: Iterable[float] | None = None,
    bg_rgba_u8: RGBA = (12, 12, 16, 255),
    outline_rgba_u8: RGBA = (60, 60, 70, 255),
    line_a_rgba_u8: RGBA = (64, 255, 64, 255),
    line_b_rgba_u8: RGBA = (255, 255, 255, 255),
) -> None:
    """Draw a tiny waveform widget using existing UiDoc primitives.

    We assume values are roughly in [-1, +1]; we clamp to that range.
    """

    x_px = int(x_px)
    y_px = int(y_px)
    w_px = int(w_px)
    h_px = int(h_px)
    if w_px <= 2 or h_px <= 2:
        return

    b.rect_px(x_px=x_px, y_px=y_px, w_px=w_px, h_px=h_px, rgba_u8=bg_rgba_u8, outline_rgba_u8=outline_rgba_u8)

    def _as_list(it: Iterable[float] | None) -> list[float]:
        if it is None:
            return []
        if isinstance(it, list):
            return [float(v) for v in it]
        return [float(v) for v in list(it)]

    a = _as_list(samples_a)
    bb = _as_list(samples_b)

    # Midline
    mid_y = int(y_px + h_px // 2)
    b.line_px(x0_px=x_px + 1, y0_px=mid_y, x1_px=x_px + w_px - 2, y1_px=mid_y, rgba_u8=(40, 40, 44, 255), w_px=1)

    def _plot_line(samples: list[float], rgba: RGBA) -> None:
        if len(samples) < 2:
            return
        n = len(samples)
        # Use full width; sample-to-pixel mapping is linear.
        for i in range(1, n):
            t0 = (i - 1) / float(max(1, n - 1))
            t1 = i / float(max(1, n - 1))
            x0 = int(x_px + 1 + t0 * float(w_px - 3))
            x1 = int(x_px + 1 + t1 * float(w_px - 3))

            v0 = max(-1.0, min(1.0, float(samples[i - 1])))
            v1 = max(-1.0, min(1.0, float(samples[i])))
            y0 = int(y_px + 1 + (1.0 - (v0 + 1.0) * 0.5) * float(h_px - 3))
            y1 = int(y_px + 1 + (1.0 - (v1 + 1.0) * 0.5) * float(h_px - 3))
            b.line_px(x0_px=x0, y0_px=y0, x1_px=x1, y1_px=y1, rgba_u8=rgba, w_px=1)

    _plot_line(a, line_a_rgba_u8)
    if bb:
        _plot_line(bb, line_b_rgba_u8)
