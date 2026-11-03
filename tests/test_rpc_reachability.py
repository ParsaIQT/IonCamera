"""Can ARTIQ reach ioncam?  An experiment that could not used to die after
15 s in build() with "Timeout receiving data from worker" and no word of why:
ioncam_gui answers on its own PC only unless started with --bind '*', and ARTIQ
connects a controller requested in build() with no time limit of its own.
Now ioncam says where it answers, and the ARTIQ examples check that it answers
(in prepare(), within seconds) and say what to do if it does not."""

import ast
import os
import socket
import time

import pytest

from ioncam.gui.app import get_argparser as gui_argparser
from ioncam.server import bind_from_args, listening

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "examples", "artiq", "repository")


def test_bind_addresses_from_the_command_line():
    """localhost unless --bind; '*' is every address of the PC - also with the
    quotes Windows' cmd.exe passes on (PowerShell and bash remove them)."""
    def bind(argv):
        return bind_from_args(gui_argparser().parse_args(argv))
    assert bind([]) == ["localhost"]
    for star in ("*", "'*'", '"*"'):
        assert bind(["--bind", star]) is None
    assert bind(["--bind", "192.168.1.50"]) == ["localhost", "192.168.1.50"]
    assert bind(["--no-localhost-bind", "--bind", "'10.0.0.2'"]) == ["10.0.0.2"]
    pytest.importorskip("sipyco")
    from ioncam.server import get_argparser
    assert bind_from_args(get_argparser().parse_args(["--bind", "'*'"])) is None


def test_listening_says_who_can_connect():
    """Only this PC (localhost): what ARTIQ on another PC sees and what to do;
    every address: the device_db host to use; and the firewall."""
    here = listening(["localhost"], 3270, ["192.168.1.50"])
    assert not here["lan"] and "this PC only" in here["where"]
    assert "--bind '*'" in here["hint"] and "(192.168.1.50)" in here["hint"]
    assert "Timeout receiving data from worker" in here["hint"]
    assert not listening("::1", 3270)["lan"] and not listening(["127.0.0.1"], 3270)["lan"]
    every = listening(None, 3270, ["192.168.1.50", "10.0.0.2"])
    assert every["lan"] and "(192.168.1.50, 10.0.0.2)" in every["where"]
    assert '"host": "192.168.1.50", "port": 3270' in every["hint"]
    assert "firewall must let TCP port 3270 in" in every["hint"]
    assert "<this PC's IP address>" in listening(None, 3270)["hint"]
    some = listening(["localhost", "10.0.0.2"], 3270)
    assert some["lan"] and '"host": "10.0.0.2"' in some["hint"]


def _examples():
    names = sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py"))
    assert len(names) >= 4, names
    out = {}
    for name in names:
        with open(os.path.join(EXAMPLES, name), encoding="utf-8") as f:
            out[name] = ast.parse(f.read())
    return out


def _method(tree, name):
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for f in cls.body:
            if isinstance(f, ast.FunctionDef) and f.name == name:
                return f
    return None


def _helper(tree):
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "get_ioncam")


def test_artiq_examples_get_ioncam_in_prepare():
    """No example connects to ioncam in build() (ARTIQ gives build() 15 s);
    each gets it first thing in prepare() through the same get_ioncam."""
    helpers = set()
    for name, tree in _examples().items():
        asks = [c for c in ast.walk(_method(tree, "build")) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", "") in ("setattr_device", "get_device")
                and any(isinstance(a, ast.Constant) and a.value == "ioncam" for a in c.args)]
        assert not asks, name
        prepare = _method(tree, "prepare")
        assert prepare is not None, name
        assert ast.unparse(prepare.body[0]) == "self.ioncam = get_ioncam(self)", name
        helpers.add(ast.unparse(_helper(tree)))
    assert len(helpers) == 1            # the same helper in every example


class _Experiment:
    """The part of ARTIQ's HasEnvironment that get_ioncam uses."""

    def __init__(self, db):
        self.db, self.got = db, []

    def get_device_db(self):
        return self.db

    def get_device(self, name):
        self.got.append(name)
        return "the {} controller".format(name)


def test_get_ioncam_says_what_to_do_when_ioncam_does_not_answer():
    """Nothing answers: a ConnectionError at once that names the address and
    what to check, no controller requested; it answers (through a device_db
    alias too): the controller."""
    ns = {"socket": socket}
    exec(compile(ast.Module(body=[_helper(_examples()["ioncam_load_ions.py"])],
                            type_ignores=[]), "get_ioncam", "exec"), ns)
    get_ioncam = ns["get_ioncam"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]           # closed again: nothing listens
    exp = _Experiment({"ioncam": {"type": "controller", "host": "127.0.0.1", "port": port}})
    t = time.time()
    with pytest.raises(ConnectionError) as e:
        get_ioncam(exp)
    assert time.time() - t < 5.0 and not exp.got
    msg = str(e.value)
    assert "ioncam does not answer at 127.0.0.1:{}".format(port) in msg
    assert "--bind '*'" in msg and "firewall let TCP port {} in".format(port) in msg
    assert "sipyco_rpctool 127.0.0.1 {} list-targets".format(port) in msg
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        exp = _Experiment({"ioncam": "camera",
                           "camera": {"type": "controller", "host": "127.0.0.1",
                                      "port": srv.getsockname()[1]}})
        assert get_ioncam(exp) == "the ioncam controller" and exp.got == ["ioncam"]
