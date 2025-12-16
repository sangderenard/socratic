from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable, Optional

from c_physics.controller_engine_ctypes import GP_CtlHookEvent, GP_CtlHookWatch


class ParkedHookRunner:
    """Block on the C hook queue and dispatch events on a Python thread.

    Intended usage:
    - Configure watches via `add_watch()`.
    - Start the controller engine tick thread.
    - Start this runner with a callback.

    The callback is invoked on the runner thread.
    """

    def __init__(self, ctl_lib) -> None:
        self._ctl = ctl_lib
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_seq: int = 0

    def add_watch(self, watch: GP_CtlHookWatch) -> bool:
        return bool(self._ctl.gp_ctl_hooks_add(ctypes.byref(watch)))

    def clear_watches(self) -> None:
        self._ctl.gp_ctl_hooks_clear()

    def start(self, *, on_event: Callable[[int, GP_CtlHookEvent], None], poll_timeout_ms: int = 50) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        def _run() -> None:
            last_seen = int(self._last_seq)
            ev = GP_CtlHookEvent()
            while not self._stop.is_set():
                out_seq_u64 = ctypes.c_uint64(0)
                rc = int(self._ctl.gp_ctl_hookq_wait(last_seen, int(poll_timeout_ms), ctypes.byref(out_seq_u64)))
                new_seq = int(out_seq_u64.value)

                if rc and new_seq > last_seen:
                    # Drain sequentially.
                    for s in range(last_seen + 1, new_seq + 1):
                        if int(self._ctl.gp_ctl_hookq_peek_seq(s, ctypes.byref(ev))) != 1:
                            # Overrun or race; resync to latest.
                            break
                        on_event(s, ev)
                    last_seen = new_seq
                else:
                    # Keep last_seen as-is.
                    time.sleep(0)

            self._last_seq = last_seen

        self._thread = threading.Thread(target=_run, name="ParkedHookRunner", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=join_timeout_s)
        self._thread = None
