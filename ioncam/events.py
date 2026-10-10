"""Thread-safe event log with blocking waits and optional JSON-lines persistence."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Iterable, List, Optional

logger = logging.getLogger("ioncam.events")

_LEVEL = {"info": logging.INFO, "warning": logging.WARNING, "alarm": logging.ERROR}


class EventLog:
    def __init__(self, maxlen: int = 10000, path: Optional[str] = None):
        self._cond = threading.Condition()
        self._events: Deque[dict] = deque(maxlen=maxlen)
        self._next_id = 1
        self._path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @property
    def last_id(self) -> int:
        with self._cond:
            return self._next_id - 1

    def add(self, event: dict) -> dict:
        ev = dict(event)
        ev.setdefault("t", time.time())
        ev.setdefault("severity", "info")
        ev.setdefault("details", {})
        ev["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["t"]))
        with self._cond:
            ev["id"] = self._next_id
            self._next_id += 1
            self._events.append(ev)
            self._cond.notify_all()
        logger.log(_LEVEL.get(ev["severity"], logging.INFO), "%s", ev.get("message", ev["kind"]))
        if self._path:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, default=_jsonable) + "\n")
            except OSError as e:  # pragma: no cover - disk problems must not stop the camera
                logger.debug("cannot write event log: %s", e)
        return ev

    def emit(self, kind: str, message: str, severity: str = "info", **details) -> dict:
        return self.add({"kind": kind, "message": message, "severity": severity,
                         "details": details})

    def since(self, last_id: int = 0, kinds: Optional[Iterable[str]] = None,
              limit: int = 200) -> List[dict]:
        kinds = set(kinds) if kinds else None
        with self._cond:
            out = [e for e in self._events
                   if e["id"] > last_id and (kinds is None or e["kind"] in kinds)]
        return out[-limit:] if limit else out

    def wait(self, last_id: int, kinds: Optional[Iterable[str]] = None,
             timeout: Optional[float] = None) -> Optional[dict]:
        """Block until an event newer than ``last_id`` (of one of ``kinds``) arrives."""
        kinds = set(kinds) if kinds else None
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                for e in self._events:
                    if e["id"] > last_id and (kinds is None or e["kind"] in kinds):
                        return e
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining if remaining is not None else 1.0)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
