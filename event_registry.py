from __future__ import annotations

import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List

from app_events import AppEvent, AppEventName


Callback = Callable[[AppEvent], None]


@dataclass(slots=True)
class _Subscription:
    token: int
    cb: Callback


class EventRegistry:
    """Simple event registry.

    - Menus register interest in high-level AppEventName values.
    - Events are emitted by a dispatcher (typically fed by the main loop).

    You can run callbacks synchronously on the emitter thread (default) or
    attach an EventRunner to execute callbacks on a dedicated thread.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: DefaultDict[AppEventName, List[_Subscription]] = defaultdict(list)
        self._next_token = 1
        self._runner: EventRunner | None = None

    def attach_runner(self, runner: "EventRunner | None") -> None:
        with self._lock:
            self._runner = runner

    def subscribe(self, name: AppEventName, cb: Callback) -> int:
        with self._lock:
            token = int(self._next_token)
            self._next_token += 1
            self._subs[name].append(_Subscription(token=token, cb=cb))
            return token

    def unsubscribe(self, token: int) -> bool:
        with self._lock:
            for name, lst in list(self._subs.items()):
                for i, sub in enumerate(list(lst)):
                    if int(sub.token) == int(token):
                        del lst[i]
                        self._subs[name] = lst
                        return True
        return False

    def emit(self, ev: AppEvent) -> None:
        # Copy callbacks under lock; run outside lock.
        with self._lock:
            targets = [s.cb for s in list(self._subs.get(ev.name, []))]
            runner = self._runner

        if runner is not None:
            for cb in targets:
                runner.enqueue(cb, ev)
            return

        for cb in targets:
            try:
                cb(ev)
            except Exception:
                # Keep registry robust; per-callback failures should not
                # take down the event pipeline.
                pass


class EventRunner:
    """Dedicated callback runner thread.

    This is meant for *logic* callbacks (state mutation, file IO, etc.).
    Rendering (OpenGL) should remain on the main thread.
    """

    def __init__(self, *, name: str = "EventRunner") -> None:
        self._q: "queue.Queue[tuple[Callback, AppEvent]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=str(name), daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self, *, drain: bool = False, timeout_s: float = 1.0) -> None:
        if not self._started:
            return
        if drain:
            # Best-effort drain with a sentinel approach.
            pass
        self._stop.set()
        try:
            self._q.put_nowait((lambda _ev: None, AppEvent(name=AppEventName.MENU_POINTER_MOVE, t_s=0.0, payload={})))
        except Exception:
            pass
        self._thread.join(timeout=float(timeout_s))

    def enqueue(self, cb: Callback, ev: AppEvent) -> None:
        if not self._started:
            # Allow a lazy start pattern.
            self.start()
        try:
            self._q.put_nowait((cb, ev))
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cb, ev = self._q.get(timeout=0.1)
            except Exception:
                continue
            if self._stop.is_set():
                break
            try:
                cb(ev)
            except Exception:
                pass
