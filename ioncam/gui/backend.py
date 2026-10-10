"""How the GUI talks to the brain: in-process (embedded) or over sipyco RPC.

Calls never run in the GUI thread.  Each *channel* ("poll" for the periodic
frame/status refresh, "cmd" for user actions) has one worker thread - for
the remote backend each worker owns its own sipyco ``Client`` (clients are not
thread-safe).  Results are delivered back to the GUI thread through a Qt
signal.
"""

import asyncio
import concurrent.futures
import inspect
import logging
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
    """The API object lives in this process (embedded mode)."""

    def __init__(self, api, server_thread=None):
        super().__init__()
        self.api = api
        self.server_thread = server_thread

    def describe(self):
        if self.server_thread is not None:
            return "local camera, RPC on port {}".format(self.server_thread.port)
        return "local camera"

    def _invoke(self, channel, name, args, kwargs):
        res = getattr(self.api, name)(*args, **kwargs)
        if inspect.iscoroutine(res):
            if self.server_thread is not None and self.server_thread.loop is not None:
                res = self.server_thread.run_coroutine(res)
            else:
                res = asyncio.run(res)
        return res


class RemoteBackend(Backend):
    """Talk to a running ``aqctl_ioncam`` over the network."""
    remote = True

    def __init__(self, host, port, target="ioncam"):
        super().__init__()
        self.host, self.port, self.target = host, port, target
        self._clients = {}

    def describe(self):
        return "remote {}:{}".format(self.host, self.port)

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
