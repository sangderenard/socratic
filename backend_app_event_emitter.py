from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os

from app_events import AppEvent, AppEventMeta, AppEventName


@dataclass(slots=True)
class _Bind1D:
    name: AppEventName
    ch: int
    edge: str = "rising"  # rising|falling|level
    threshold: float = 0.5


@dataclass(slots=True)
class _Bind2D:
    name: AppEventName
    ch2d: int
    deadzone: float = 0.0
    emit: str = "change"  # change|always


def _as_event_name(s: str) -> AppEventName | None:
    try:
        return AppEventName(str(s))
    except Exception:
        return None


class BackendAppEventEmitter:
    """Emit AppEvents from controller *channels* (not from hardware).

    Intended usage:
    - Main thread pushes GP_InputEvent to C signal kernel.
    - Controller graph is evaluated from kernel peek values.
    - This class converts channel outputs into intent-level AppEvents.

    Bindings live in joystick.json under root key `app_event_bindings`:

    {
      "app_event_bindings": {
        "MENU_CONFIRM": {"ch": 10, "edge": "rising", "threshold": 0.5},
        "MENU_CANCEL":  {"ch": 11, "edge": "rising", "threshold": 0.5},
        "MENU_TOGGLE":  {"ch": 12, "edge": "rising", "threshold": 0.5},
        "MENU_POINTER_MOVE": {"ch2d": 3, "deadzone": 0.0, "emit": "change"},
        "MENU_POINTER_AFFIRM": {"ch": 13, "edge": "rising", "threshold": 0.5}
      }
    }

    Notes:
    - 1D channels are interpreted as -1..1 (or 0/1) with threshold.
    - 2D channels are interpreted as (x,y) in -1..1.
    """

    def __init__(self, *, joystick_cfg_path: str = "joystick.json") -> None:
        self._cfg_path = str(joystick_cfg_path)
        self._cached_mtime_ns: int | None = None
        self._bind_1d: list[_Bind1D] = []
        self._bind_2d: list[_Bind2D] = []

        self._prev_1d: dict[int, float] = {}
        self._prev_2d: dict[int, tuple[float, float]] = {}

        self._refresh_if_needed(force=True)

    def _refresh_if_needed(self, *, force: bool = False) -> None:
        try:
            st = os.stat(self._cfg_path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        except Exception:
            mtime_ns = None

        if not force:
            if mtime_ns is None:
                return
            if self._cached_mtime_ns is not None and int(mtime_ns) == int(self._cached_mtime_ns):
                return

        self._cached_mtime_ns = int(mtime_ns) if mtime_ns is not None else None

        cfg: dict[str, Any] = {}
        try:
            if os.path.exists(self._cfg_path):
                with open(self._cfg_path, "r", encoding="utf-8") as f:
                    v = json.load(f)
                cfg = v if isinstance(v, dict) else {}
        except Exception:
            cfg = {}

        binds = cfg.get("app_event_bindings")
        if not isinstance(binds, dict):
            binds = {}

        b1: list[_Bind1D] = []
        b2: list[_Bind2D] = []

        for k, spec in binds.items():
            ev_name = _as_event_name(str(k))
            if ev_name is None:
                continue
            if not isinstance(spec, dict):
                continue

            if "ch2d" in spec:
                try:
                    ch2d = int(spec.get("ch2d"))
                except Exception:
                    continue
                try:
                    deadzone = float(spec.get("deadzone", 0.0))
                except Exception:
                    deadzone = 0.0
                emit = str(spec.get("emit", "change")).strip().lower() or "change"
                if emit not in ("change", "always"):
                    emit = "change"
                b2.append(_Bind2D(name=ev_name, ch2d=int(ch2d), deadzone=float(max(0.0, deadzone)), emit=str(emit)))
                continue

            if "ch" in spec:
                try:
                    ch = int(spec.get("ch"))
                except Exception:
                    continue
                edge = str(spec.get("edge", "rising")).strip().lower() or "rising"
                if edge not in ("rising", "falling", "level"):
                    edge = "rising"
                try:
                    th = float(spec.get("threshold", 0.5))
                except Exception:
                    th = 0.5
                b1.append(_Bind1D(name=ev_name, ch=int(ch), edge=str(edge), threshold=float(th)))

        self._bind_1d = b1
        self._bind_2d = b2

    def emit_from_channels(
        self,
        *,
        overrides_1d: dict[int, float],
        channels_2d: dict[int, tuple[float, float]],
        t_s: float,
    ) -> list[AppEvent]:
        """Return a list of AppEvents derived from channel outputs."""

        self._refresh_if_needed(force=False)

        out: list[AppEvent] = []

        # 1D edges
        for b in list(self._bind_1d):
            ch = int(b.ch)
            v = float(overrides_1d.get(ch, 0.0))
            prev = float(self._prev_1d.get(ch, 0.0))
            self._prev_1d[ch] = float(v)

            th = float(b.threshold)
            fired = False
            if b.edge == "level":
                fired = float(v) > float(th)
            elif b.edge == "falling":
                fired = (float(prev) >= float(th)) and (float(v) < float(th))
            else:  # rising
                fired = (float(prev) <= float(th)) and (float(v) > float(th))

            if fired:
                out.append(
                    AppEvent(
                        name=b.name,
                        t_s=float(t_s),
                        payload={"ch": int(ch), "v": float(v), "threshold": float(th), "edge": str(b.edge)},
                        meta=AppEventMeta(source="channels", channel_1d=int(ch), binding=str(b.name)),
                    )
                )

        # 2D vectors
        for b in list(self._bind_2d):
            ch2d = int(b.ch2d)
            x, y = channels_2d.get(ch2d, (0.0, 0.0))
            x = float(x)
            y = float(y)

            # deadzone
            dz = float(b.deadzone)
            if dz > 0.0:
                if (x * x + y * y) ** 0.5 < float(dz):
                    x, y = 0.0, 0.0

            prev = self._prev_2d.get(ch2d, (0.0, 0.0))
            self._prev_2d[ch2d] = (float(x), float(y))

            if b.emit == "always":
                changed = True
            else:
                # change: emit when vector changes meaningfully
                changed = (abs(float(x) - float(prev[0])) > 1e-6) or (abs(float(y) - float(prev[1])) > 1e-6)

            if changed:
                out.append(
                    AppEvent(
                        name=b.name,
                        t_s=float(t_s),
                        payload={"ch2d": int(ch2d), "x": float(x), "y": float(y)},
                        meta=AppEventMeta(source="channels", channel_2d=int(ch2d), binding=str(b.name)),
                    )
                )

        return out
