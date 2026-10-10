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

__version__ = "1.0.0"
