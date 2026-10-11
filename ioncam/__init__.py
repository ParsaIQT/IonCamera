"""ioncam - the camera "brain" of an ion trap experiment.

Layers (bottom to top):

* :mod:`ioncam.camera` - camera drivers (Andor iXon via pyAndorSDK2, and a
  physically realistic simulator) behind one small interface.
* :mod:`ioncam.analysis` - spot detection, sub-pixel fitting, the linear
  chain model, the chain tracker that classifies trapped / lost / dark ions,
  and the Grabber readout planner.
* :mod:`ioncam.engine` - owns the camera thread and the analysis thread and
  implements the operating modes and procedures (live, locate, calibrate,
  readout).
* :mod:`ioncam.api` / :mod:`ioncam.server` - the sipyco RPC interface used by
  the ARTIQ host (``aqctl_ioncam``).
* :mod:`ioncam.gui` - optional Qt GUI (embedded or remote).
"""

import os as _os

__version__ = "1.0.0"


def _git_commit() -> str:
    """Short commit of the git checkout this code runs from - with an editable
    install (``pip install -e .``) that is the checkout itself, so a ``git
    pull`` takes effect at the next start; "" when not in a checkout (a copy
    installed into site-packages, which needs re-installing after a pull)."""
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    git = _os.path.join(root, ".git")
    try:
        if _os.path.isfile(git):                     # worktree: "gitdir: <path>"
            with open(git) as f:
                git = _os.path.join(root, f.read().split(":", 1)[1].strip())
        common = git
        if _os.path.isfile(_os.path.join(git, "commondir")):
            with open(_os.path.join(git, "commondir")) as f:
                common = _os.path.join(git, f.read().strip())
        with open(_os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(":", 1)[1].strip()
        for base in (git, common):
            path = _os.path.join(base, ref)
            if _os.path.isfile(path):
                with open(path) as f:
                    return f.read().strip()[:7]
        with open(_os.path.join(common, "packed-refs")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:7]
    except (OSError, IndexError):
        pass
    return ""


# read once: the code running is the code imported, even if the checkout is
# pulled while it runs
_COMMIT = _git_commit()


def version_string() -> str:
    """``__version__`` plus where the code comes from (git commit / installed copy)."""
    return "{} (git {})".format(__version__, _COMMIT) if _COMMIT else \
        "{} (installed copy)".format(__version__)
