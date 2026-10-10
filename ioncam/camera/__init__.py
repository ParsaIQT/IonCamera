"""Camera drivers: Andor iXon (pyAndorSDK2) and a realistic simulator."""

from ioncam.camera.base import (AcquisitionSettings, AppliedSettings, CameraBase,
                                CameraError, Frame, normalize_roi, roi_from_sdk,
                                roi_to_sdk)

__all__ = ["AcquisitionSettings", "AppliedSettings", "CameraBase", "CameraError",
           "Frame", "normalize_roi", "roi_from_sdk", "roi_to_sdk", "make_camera"]


def make_camera(config, simulate=False):
    """Create the camera selected by ``config.camera.backend`` (or the simulator)."""
    backend = "simulator" if simulate else config.camera.backend
    if backend == "simulator":
        from ioncam.camera.simulator import SimulatedCamera
        return SimulatedCamera(config.simulator)
    if backend == "andor":
        from ioncam.camera.andor import AndorCamera
        return AndorCamera(config.camera)
    if backend == "fake_sdk":
        from ioncam.camera.andor import AndorCamera
        from ioncam.camera.fake_sdk import FakeAtmcd
        return AndorCamera(config.camera, sdk=FakeAtmcd(config.simulator))
    raise ValueError("unknown camera backend '{}'".format(backend))
