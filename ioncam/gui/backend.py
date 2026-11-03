"""How the GUI talks to the brain: in-process (embedded) or over sipyco RPC.

Calls never run in the GUI thread.  Each *channel* ("poll" for the periodic
frame/status refresh, "cmd" for user actions) has one worker thread - for
the remote backend each worker owns its own sipyco ``Client`` (clients are not
thread-safe).  Results are delivered back to the GUI thread through a Qt
signal.

In the embedded mode the GUI does not depend on the RPC server it runs for
ARTIQ: coroutine API methods run on an event loop of the worker's own, so the
GUI works whether or not the server could start (``rpc_state`` says).
"""

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import traceback

from pyqtgraph.Qt import QtCore

logger = logging.getLogger(__name__)


class _Bridge(QtCore.QObject):
    deliver = QtCore.Signal(object)


class Backend:
    remote = False

    def __init__(self):
        self._bridge = _Bridge()
        self._bridge.deliver.connect(lambda fn: fn())
        self._pools = {name: concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=name)
                       for name in ("poll", "cmd", "wait")}
        self.on_error = lambda name, exc: logger.error("%s failed: %s", name, exc)

    def describe(self):
        return ""

    def rpc_state(self):
        """The RPC server ARTIQ talks to, as seen from here: None, or a dict
        with "state" (running / starting / failed / stopped / off / remote),
        "text", a "hint" and, while running, "lan" (False: it answers only on
        this PC)."""
        return None

    def call(self, name, *args, callback=None, error=None, channel="cmd", **kwargs):
        """Call API method ``name`` in a worker; ``callback(result)`` runs in the GUI thread."""
        fut = self._pools[channel].submit(self._invoke, channel, name, args, kwargs)

        def finished(f):
            try:
                res, ok = f.result(), True
            except Exception as e:  # delivered to the GUI
                res, ok = e, False
                logger.debug("%s: %s", name, "".join(traceback.format_exception(e)))

            def deliver():
                if ok:
                    if callback is not None:
                        callback(res)
                else:
                    (error or self.on_error)(name, res)
            self._bridge.deliver.emit(deliver)
        fut.add_done_callback(finished)
        return fut

    def run_in_poll(self, fn, callback=None):
        """Run ``fn(invoke)`` in the poll worker (several calls in one round trip)."""
        def job():
            return fn(lambda name, *a, **k: self._invoke("poll", name, a, k))
        fut = self._pools["poll"].submit(job)

        def finished(f):
            try:
                res, ok = f.result(), True
            except Exception as e:
                res, ok = e, False

            def deliver():
                if ok and callback is not None:
                    callback(res)
                elif not ok:
                    self.on_error("refresh", res)
            self._bridge.deliver.emit(deliver)
        fut.add_done_callback(finished)
        return fut

    def _invoke(self, channel, name, args, kwargs):
        raise NotImplementedError

    def close(self):
        for p in self._pools.values():
            p.shutdown(wait=False, cancel_futures=True)


class LocalBackend(Backend):
    """The API object lives in this process (embedded mode).

    ``server_thread``: the RPC server for ARTIQ (a ``ServerThread``, which
    may have failed to start) or None.  ``on_rpc_stop(state, reason)`` is
    called in the GUI thread when it stops on its own."""

    def __init__(self, api, server_thread=None):
        super().__init__()
        self.api = api
        self.server_thread = server_thread
        self.on_rpc_stop = None
        self._lock = threading.Lock()
        self._tasks = set()             # (loop, task) of the coroutines running
        self._closed = False
        if server_thread is not None:
            server_thread.on_stop = self._rpc_stopped

    def _rpc_stopped(self, state, reason):      # in the server thread
        def deliver():
            if self.on_rpc_stop is not None:
                self.on_rpc_stop(state, reason)
        self._bridge.deliver.emit(deliver)

    def rpc_state(self):
        from ioncam.server import failure_hint
        st = self.server_thread
        if st is None:
            return {"state": "off", "port": None, "reason": "started with --no-server",
                    "hint": "", "text": "no RPC server (--no-server): ARTIQ cannot connect"}
        out = {"state": st.state, "port": st.port, "reason": st.reason, "hint": ""}
        if st.state == "running":
            # where it answers: only this PC unless started with --bind
            where = st.listening()
            out["lan"] = where["lan"]
            out["text"] = "RPC server on port {}, {}: ARTIQ connects to target 'ioncam'".format(
                st.port, where["where"])
            out["hint"] = where["hint"]
        elif st.state == "starting":
            out["text"] = "RPC server on port {} starting".format(st.port)
        else:
            out["text"] = "RPC server on port {} {}, ARTIQ cannot connect: {}".format(
                st.port, st.state, st.reason)
            if st.error is not None:
                out["hint"] = failure_hint(st.error, st.port)
        return out

    def describe(self):
        st = self.server_thread
        if st is not None and st.state in ("running", "starting"):
            return "local camera, RPC on port {}{}".format(
                st.port, "" if st.listening()["lan"] else " for this PC only")
        return "local camera, no RPC server"

    def _invoke(self, channel, name, args, kwargs):
        res = getattr(self.api, name)(*args, **kwargs)
        if inspect.iscoroutine(res):
            try:
                res = asyncio.run(self._tracked(res))
            except asyncio.CancelledError:
                raise RuntimeError("{} cancelled: the GUI is closing".format(name)) from None
        return res

    async def _tracked(self, coro):
        """Run ``coro`` on this worker's own event loop, cancelled by close()."""
        me = (asyncio.get_running_loop(), asyncio.current_task())
        with self._lock:
            if self._closed:
                coro.close()
                raise RuntimeError("the GUI is closing")
            self._tasks.add(me)
        try:
            return await coro
        finally:
            with self._lock:
                self._tasks.discard(me)

    def close(self):
        super().close()
        with self._lock:
            self._closed = True
            tasks = list(self._tasks)
        # e.g. a wait for ions: the process would otherwise not exit until it ends
        for loop, task in tasks:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:        # finished meanwhile
                pass


class RemoteBackend(Backend):
    """Talk to a running ``aqctl_ioncam`` over the network."""
    remote = True

    def __init__(self, host, port, target="ioncam"):
        super().__init__()
        self.host, self.port, self.target = host, port, target
        self._clients = {}

    def describe(self):
        return "remote {}:{}".format(self.host, self.port)

    def rpc_state(self):
        return {"state": "remote", "text": "connected to {}:{}".format(self.host, self.port)}

    def _client(self, channel):
        from sipyco.pc_rpc import Client
        c = self._clients.get(channel)
        if c is None:
            c = Client(self.host, self.port, self.target,
                       timeout=None if channel in ("cmd", "wait") else 20.0)
            self._clients[channel] = c
        return c

    def _invoke(self, channel, name, args, kwargs):
        c = self._client(channel)
        try:
            return getattr(c, name)(*args, **kwargs)
        except (OSError, EOFError):
            self._clients.pop(channel, None)
            try:
                c.close_rpc()
            except Exception:
                pass
            raise

    def close(self):
        super().close()
        for c in self._clients.values():
            try:
                c.close_rpc()
            except Exception:
                pass
