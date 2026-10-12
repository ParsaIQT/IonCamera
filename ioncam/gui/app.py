"""``ioncam_gui`` - the ioncam GUI.

* ``ioncam_gui --simulate``            try it without hardware
* ``ioncam_gui -c ioncam.toml``        run the camera in this process (and serve RPC
                                       for ARTIQ on port 3270, like aqctl_ioncam)
* ``ioncam_gui --host camera-pc``      view/control a running aqctl_ioncam remotely
"""

import argparse
import logging
import sys

DEFAULT_PORT = 3270


def get_argparser():
    p = argparse.ArgumentParser(description="ioncam GUI")
    p.add_argument("--host", default=None,
                   help="connect to a running aqctl_ioncam on this host (remote mode)")
    p.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="RPC port")
    p.add_argument("-c", "--config", default=None, help="TOML configuration (local mode)")
    p.add_argument("--simulate", action="store_true", help="simulated camera + ion trap")
    p.add_argument("--sim-camera", choices=("897", "888"), default=None,
                   help="simulated camera model: iXon Ultra 897 (512 x 512) or 888 "
                        "(1024 x 1024); default: simulator.camera_model")
    p.add_argument("--bind", default=[], action="append",
                   help="additional address for the RPC server to bind to ('*' = all)")
    p.add_argument("--no-localhost-bind", action="store_true")
    p.add_argument("--no-server", action="store_true",
                   help="local mode without the RPC server (ARTIQ cannot connect)")
    p.add_argument("--no-live", action="store_true", help="do not start live video")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _bind(args):
    if "*" in args.bind:
        return None
    return list(args.bind) if args.no_localhost_bind else ["localhost"] + list(args.bind)


def show(backend):
    """Create the window for ``backend`` and run the Qt event loop."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from ioncam.gui.main_window import MainWindow
    app = pg.mkQApp("ioncam")
    config = backend.call("get_config").result(30)
    info = backend.call("get_info").result(30)
    # the theme and the window layout are kept between sessions
    win = MainWindow(backend, config, info, settings=QtCore.QSettings("ioncam", "ioncam_gui"))
    win.show()
    pg.exec()
    return app


def run_embedded(engine, api, bind, port, server=True):
    """GUI + RPC server in this process for an already started engine."""
    from ioncam.gui.backend import LocalBackend
    from ioncam.server import ServerThread
    st = ServerThread(api, bind, port).start() if server else None
    backend = LocalBackend(api, st)
    try:
        show(backend)
    finally:
        backend.close()
        if st is not None:
            st.stop()


def main(argv=None):
    args = get_argparser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("ioncam").setLevel(logging.INFO - 10 * max(0, args.verbose - 1))
    if args.host:
        from ioncam.gui.backend import RemoteBackend
        backend = RemoteBackend(args.host, args.port)
        try:
            show(backend)
        finally:
            backend.close()
        return
    from ioncam.config import Config
    from ioncam.server import build
    config = Config.load(args.config)
    if args.sim_camera:
        config.simulator.camera_model = args.sim_camera
    engine, api = build(config, simulate=args.simulate)
    engine.start(auto_live=not args.no_live)
    try:
        run_embedded(engine, api, _bind(args), args.port, server=not args.no_server)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
