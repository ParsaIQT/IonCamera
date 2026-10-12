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


class ServerThread:
    """RPC server in a background thread (used by the GUI's embedded mode/tests)."""

    def __init__(self, api, host, port):
        self.api, self.host, self.port = api, host, port
        self.loop = None
        self._stop = None
        self._ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="ioncam-rpc", daemon=True)

    def start(self, timeout=10.0):
        self.thread.start()
        self._ready.wait(timeout)
        if self.error is not None:
            raise self.error
        return self

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._stop = asyncio.Event()

        async def main():
            server = make_server(self.api)
            try:
                await server.start(self.host, self.port)
            except Exception as e:
                self.error = e
                self._ready.set()
                return
            self._ready.set()
            waits = [asyncio.ensure_future(server.wait_terminate()),
                     asyncio.ensure_future(self._stop.wait())]
            try:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waits:
                    w.cancel()
                await server.stop()
        try:
            self.loop.run_until_complete(main())
        finally:
            self.loop.close()

    def run_coroutine(self, coro, timeout=None):
        """Run a coroutine (e.g. an async API method) on the server loop."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        if self.loop is not None and self._stop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._stop.set)
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
