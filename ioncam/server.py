"""``aqctl_ioncam`` - the ioncam controller (sipyco RPC server, ARTIQ conventions).

Examples::

    aqctl_ioncam --simulate                       # try everything without hardware
    aqctl_ioncam -c ioncam.toml --bind '*'        # real camera, reachable from the LAN
    aqctl_ioncam -c ioncam.toml --gui             # with the GUI in the same process
    aqctl_ioncam --write-config ioncam.toml       # commented default configuration

ARTIQ ``device_db.py``::

    device_db["ioncam"] = {
        "type": "controller",
        "host": "192.168.1.50",            # the camera PC
        "port": 3270,
        "target_name": "ioncam",
        "command": "aqctl_ioncam -p {port} --bind {bind} -c C:/ioncam/ioncam.toml",
    }
"""

import argparse
import asyncio
import errno
import logging
import sys
import threading

from ioncam import __version__
from ioncam.config import Config

logger = logging.getLogger("ioncam")

DEFAULT_PORT = 3270
TARGET = "ioncam"


def _sipyco():
    try:
        from sipyco.pc_rpc import Server
        from sipyco import common_args
    except ImportError as e:
        raise SystemExit("sipyco is required for the RPC server: "
                         "pip install git+https://github.com/m-labs/sipyco ({})".format(e))
    return Server, common_args


def get_argparser():
    _, common_args = _sipyco()
    parser = argparse.ArgumentParser(
        description="ioncam {} - ion-trap camera brain (sipyco RPC target '{}')".format(
            __version__, TARGET))
    common_args.simple_network_args(parser, DEFAULT_PORT)
    common_args.verbosity_args(parser)
    parser.add_argument("-c", "--config", default=None, help="TOML configuration file")
    parser.add_argument("--simulate", action="store_true",
                        help="use the simulated camera + ion trap (no hardware needed)")
    parser.add_argument("--sim-camera", choices=("897", "888"), default=None,
                        help="simulated camera model: iXon Ultra 897 (512 x 512) or 888 "
                             "(1024 x 1024); default: simulator.camera_model")
    parser.add_argument("--no-live", action="store_true",
                        help="do not start live video at start-up")
    parser.add_argument("--gui", action="store_true",
                        help="also open the GUI in this process")
    parser.add_argument("--write-config", metavar="PATH",
                        help="write a commented configuration file with all defaults and exit")
    return parser


def make_server(api):
    Server, _ = _sipyco()
    return Server({TARGET: api}, description="ioncam {}".format(__version__),
                  builtin_terminate=True, allow_parallel=True)


async def serve(api, host, port, stop_event=None):
    """Run the RPC server until terminated (Ctrl-C, ``terminate`` RPC, or stop_event)."""
    server = make_server(api)
    await server.start(host, port)
    logger.info("listening on %s port %d (target '%s')", host, port, TARGET)
    waits = [asyncio.ensure_future(server.wait_terminate())]
    if stop_event is not None:
        waits.append(asyncio.ensure_future(stop_event.wait()))
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for w in waits:
            w.cancel()
        await server.stop()


def failure_hint(exc, port):
    """What to do about an RPC server that cannot start with ``exc``."""
    if "sipyco" in str(exc):
        return ("Install sipyco into this Python environment "
                "(pip install git+https://github.com/m-labs/sipyco) and start ioncam again.")
    if isinstance(exc, OSError) and exc.errno in (errno.EADDRINUSE, 10048):
        return ("Another program listens on port {} - another ioncam_gui or aqctl_ioncam "
                "(perhaps started by artiq_ctlmgr)?  Close it, or start ioncam with "
                "another port (-p).".format(port))
    return ""


class ServerThread:
    """RPC server in a background thread (the GUI's embedded mode, tests).

    ``start`` waits until the server listens and raises when it cannot (sipyco
    missing, the port in use, ...).  ``state`` is "starting", "running",
    "failed" or "stopped" and ``reason`` says why it is not running.  A server
    that stops on its own afterwards (a ``terminate`` request, an error) is
    logged and reported to ``on_stop(state, reason)``, called in the server
    thread.  Nothing else depends on the server's event loop: whoever embeds
    it keeps working when it is gone.
    """

    def __init__(self, api, host, port, on_stop=None):
        self.api, self.host, self.port = api, host, port
        self.on_stop = on_stop
        self.loop = None
        self.error = None
        self.state, self.reason = "starting", ""
        self._stop = None
        self._stopping = False
        self._late = False          # start() gave up waiting: the thread reports
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ioncam-rpc", daemon=True)

    def start(self, timeout=30.0):
        """Start the server and wait until it listens; raises if it cannot start.
        After ``timeout`` (a very slow computer) it carries on starting in the
        background and a failure is logged and reported to ``on_stop``."""
        self.thread.start()
        if not self._ready.wait(timeout):
            with self._lock:
                self._late = not self._ready.is_set()
            if self._late:
                logger.warning("the RPC server on port %d takes more than %.0f s to start; "
                               "carrying on without waiting for it", self.port, timeout)
                return self
        if self.error is not None:
            raise self.error
        return self

    def _run(self):
        how, exc = "error", None
        try:
            how = asyncio.run(self._main())
        except BaseException as e:      # SystemExit too: _sipyco() when sipyco is missing
            exc = RuntimeError(str(e.code)) if isinstance(e, SystemExit) else e
        self._ended(how, exc)

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = make_server(self.api)
        await server.start(self.host, self.port)
        with self._lock:
            self.state = "running"
            self._ready.set()
        if self._stopping:
            await server.stop()
            return "stop"
        waits = [asyncio.ensure_future(server.wait_terminate()),
                 asyncio.ensure_future(self._stop.wait())]
        try:
            done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waits:
                w.cancel()
            await server.stop()
        return "terminate" if waits[0] in done and not self._stopping else "stop"

    def _ended(self, how, exc):
        with self._lock:
            starting = not self._ready.is_set()
            if exc is not None:
                self.state, self.error = "failed", exc
                self.reason = str(exc) or type(exc).__name__
            elif how == "terminate":
                self.state, self.reason = "stopped", "terminated by an RPC client"
            else:
                self.state, self.reason = "stopped", "stopped"
            # start() raises a failure to start, unless it gave up waiting
            report = (not self._stopping and self.reason != "stopped"
                      and (self._late or not starting))
            self._ready.set()
        if not report:
            return
        if self.state == "failed":
            logger.error("the RPC server on port %d failed, ARTIQ cannot connect: %s",
                         self.port, self.reason, exc_info=None if starting else exc)
        else:
            logger.warning("the RPC server on port %d stopped: %s", self.port, self.reason)
        if self.on_stop is not None:
            try:
                self.on_stop(self.state, self.reason)
            except Exception:
                logger.exception("RPC server on_stop callback failed")

    def stop(self):
        self._stopping = True
        loop, ev = self.loop, self._stop
        if loop is not None and ev is not None:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:        # the loop is already closed
                pass
        if self.thread.ident is not None:
            self.thread.join(timeout=5.0)


def build(config, simulate=False):
    """Create camera + engine + API from a configuration."""
    from ioncam.api import IonCamAPI
    from ioncam.camera import make_camera
    from ioncam.engine import IonCamEngine
    camera = make_camera(config, simulate=simulate)
    engine = IonCamEngine(config, camera)
    return engine, IonCamAPI(engine)


def main(argv=None):
    args = get_argparser().parse_args(argv)
    _, common_args = _sipyco()
    common_args.init_logger_from_args(args)
    # ioncam's own messages (events, procedure steps) at INFO by default; -q/-v adjust
    level = logging.INFO + 10 * args.quiet - 10 * args.verbose
    logging.getLogger("ioncam").setLevel(max(logging.DEBUG, min(logging.CRITICAL, level)))
    if args.write_config:
        Config().save(args.write_config)
        print("wrote", args.write_config)
        return
    config = Config.load(args.config)
    if args.sim_camera:
        config.simulator.camera_model = args.sim_camera
    engine, api = build(config, simulate=args.simulate)
    engine.start(auto_live=not args.no_live)
    bind = common_args.bind_address_from_args(args)
    try:
        if args.gui:
            from ioncam.gui.app import run_embedded
            run_embedded(engine, api, bind, args.port)
        else:
            try:
                asyncio.run(_serve_with_signals(api, bind, args.port))
            except KeyboardInterrupt:
                pass
    finally:
        engine.shutdown()


async def _serve_with_signals(api, host, port):
    stop = asyncio.Event()
    try:
        from sipyco.tools import SignalHandler
    except ImportError:  # older sipyco
        try:
            from sipyco.asyncio_tools import SignalHandler
        except ImportError:
            SignalHandler = None
    if SignalHandler is not None:
        handler = SignalHandler()
        handler.setup()

        async def watch():
            await handler.wait_terminate()
            stop.set()
        task = asyncio.ensure_future(watch())
        try:
            await serve(api, host, port, stop)
        finally:
            task.cancel()
            handler.teardown()
    else:
        await serve(api, host, port, stop)


if __name__ == "__main__":
    main(sys.argv[1:])
