# ioncam — the camera brain of an ion trap

`ioncam` runs an Andor **iXon Ultra 897** for a trapped-ion experiment:

* **live view** for beginners, with a large colour-coded "N ions" status and alarms;
* **intelligent crystal tracking**: it tells an ion that was **trapped**, **lost** (the
  crystal re-arranges) or went **dark** (everything else stays exactly where it was).
  It also recognises hot **clouds**, **melting**, **lasers off** and **rigid moves**
  (shuttling, a bumped camera), and it infers where dark ions sit;
* **careful Grabber readout preparation**. A long exposure at low gain locates every
  ion to about 0.01 px. The camera **AQROI** is then centred on the crystal and aligned
  to the binning grid, each ion gets **its own Grabber ROI** sized for maximum light
  containment and minimum crosstalk, and everything is **verified with the real
  readout settings** before the camera is **armed** (external trigger + Camera Link);
* **monitoring during experiments**. The iXon sends every Grabber frame over USB as
  well, so ioncam keeps watching the ions, fits bright/dark thresholds and warns when
  the crystal changes under a running experiment;
* a **sipyco RPC server** (`aqctl_ioncam`, ARTIQ controller conventions) so ARTIQ
  experiments can wait for ions, calibrate and arm the readout, get the ROIs, hand off
  control while shuttling, and so on;
* a **Qt GUI** (`ioncam_gui`) that runs with the camera in-process or as a remote viewer;
* a physically realistic **simulator** (ion chain physics plus an EMCCD noise model),
  so everything can be tried and tested without hardware.

```
                 ┌───────────────────── camera PC ─────────────────────┐
 iXon Ultra 897 ─┤USB  aqctl_ioncam / ioncam_gui                        │
        │        │  camera thread ─ frames ─▶ analysis thread           │
        │        │  (all SDK calls)            detection → tracker →    │
        │        │                             events, readout monitor  │
        │        │  sipyco RPC "ioncam" :3270 ◀───────────────┐         │
        │        └────────────────────────────────────────────┼─────────┘
        │ Camera Link                                         │ sipyco
        ▼                                                     │
   ARTIQ Grabber ── per-ion ROI sums ──▶ kernel        ARTIQ host (experiments)
        ▲ ext. trigger TTL ───────────────────────────────────┘
```

## Quick start

```bash
pip install -e ".[gui,test]"
pip install git+https://github.com/m-labs/sipyco      # not on PyPI (ARTIQ envs have it)

ioncam_gui --simulate          # GUI + simulated trap; use the "Simulation" tab to play
aqctl_ioncam --simulate        # headless RPC server with the simulator
pytest                         # 78 tests (~45 s), all on the simulator / emulated SDK
```

With the real camera on the camera PC (Windows):

1. Install the Andor SDK2 driver and Andor's `pyAndorSDK2` package (`pip install .` in
   the folder from Andor's zip). **Close Andor Solis**, since only one program can own
   the camera.
2. `aqctl_ioncam --write-config ioncam.toml`, then edit it. Every option is commented.
3. `ioncam_gui -c ioncam.toml` opens the camera, starts live video and serves RPC on port
   3270. Alternatively run `aqctl_ioncam -c ioncam.toml --bind '*'` headless, and from
   any PC `ioncam_gui --host <camera-pc>`.
4. Add the controller to ARTIQ's `device_db.py` (see `examples/artiq/device_db_ioncam.py`):

```python
device_db["ioncam"] = {
    "type": "controller", "host": "192.168.1.50", "port": 3270, "target_name": "ioncam",
    "command": "aqctl_ioncam -p {port} --bind {bind} -c C:/ioncam/ioncam.toml",
}
```

## Using it from ARTIQ

```python
self.setattr_device("ioncam")              # a sipyco Client

# loading: switch oven/PI on, then
r = self.ioncam.wait_for_ions(3, timeout=300)        # stable crystal of 3 bright ions
# Grabber readout: locate -> plan -> verify -> arm, all in one call
plan = self.ioncam.calibrate_readout(n_ions=3)
x0, y0, x1, y1 = (plan["grabber_x0"], plan["grabber_y0"],
                  plan["grabber_x1"], plan["grabber_y1"])   # plain ints, kernel-ready
# in the kernel:
#   grabber0.setup_roi(i, x0[i], y0[i], x1[i], y1[i]) for every ROI
#   with parallel: grabber0.gate_roi_pulse(plan_mask, gate); ttl_camera_trigger.pulse(10*us)
#   grabber0.input_mu(sums)
```

Complete experiments are in `examples/artiq/repository/`:

| file | what it shows |
|---|---|
| `ioncam_load_ions.py` | loading until ioncam confirms N ions, with a control lease |
| `ioncam_grabber_readout.py` | calibrate + arm, Grabber ROI setup, triggered shots, thresholds |
| `ioncam_grabber_mapping_check.py` | rebuilds the Grabber frame from single-pixel ROIs and checks its orientation against ioncam |
| `ioncam_shuttle.py` | `hold_tracking()` around transport, survival check, position-vs-voltage scan with `locate_ions()` |

The most used RPC methods (`sipyco_rpctool <host> 3270 list-methods` lists all of them
with their docstrings):

| method | purpose |
|---|---|
| `get_status()` | everything at a glance (mode, temperature, ions, readout, control, last event) |
| `get_ion_count()` / `get_ion_counts()` / `get_ions()` | bright ions / counts / per-ion sites (position, state, brightness) |
| `wait_for_ions(n, timeout, at_least)` | block until a *stable* crystal of n bright ions |
| `get_events(since_id)` / `wait_for_event(kinds, since_id, timeout)` | the event stream |
| `locate_ions(expected)` | long exposure + joint fit: positions to ~0.01 px, PSF, chain fit |
| `calibrate_readout(n_ions, hbin, vbin, auto_binning, ...)` | locate → plan AQROI/ROIs → verify → arm |
| `get_readout_plan()` / `get_grabber_rois()` / `get_grabber_roi_lists()` / `is_readout_valid()` | the plan |
| `arm_readout()` / `disarm_readout(resume_live)` | external trigger + Camera Link on/off |
| `get_readout_counts(since_shot)` / `get_readout_stats()` | per-ROI sums of every shot (USB copy), threshold fits |
| `hold_tracking(reason, timeout)` / `resume_tracking()` | "ions will move on purpose" (shuttling, camera) |
| `take_control(owner, reason, timeout)` / `release_control(owner)` | advisory lease shown in all GUIs |
| `start_live(**settings)` / `set_live(...)` / `set_exposure` / `set_em_gain` / `set_roi` / `stop()` | live video |
| `get_frame(since_index, bin, roi)` | newest image (uint16, raw orientation) |
| `learn_background(n)` | learn hot pixels / scatter structure (ions are masked) |
| `sim_*` | drive the simulated trap: load/lose/darken ions, melt, shift, lasers, triggers |

Long calls run as coroutines on the server (`allow_parallel`), so one client waiting for
ions never blocks another client or the GUI.

## How the brain decides

Positions of a linear chain depend only on the ions' charge, not on their mass. That
gives clean fingerprints:

| what happens | what the camera sees | event |
|---|---|---|
| ion trapped | a new, persistent configuration with more ions (often after a hot cloud) | `cloud_detected`, `ion_trapped` |
| ion goes **dark** (shelved, dark species) | one site stops fluorescing, **all other ions stay put** | `ion_dark` (→ `ion_bright`) |
| ion **lost** | the remaining ions **re-arrange** (re-centre, re-space) | `ion_lost` |
| a dark ion is lost | bright ions re-arrange, bright count unchanged | `ion_lost` ("dark ion lost") |
| dark and bright ion swap | a dark site lights up and a bright one goes dark, rest unmoved | `dark_ion_moved` |
| collision / heating | light turns diffuse and dim, then re-crystallises | `crystal_melted` → `crystal_recovered` or `ion_lost` |
| lasers blocked / off | all light gone, including the diffuse light | `fluorescence_lost` → `fluorescence_restored`, or `ions_presumed_lost` after `forget_time` |
| shuttling, camera bumped | rigid shift with every ion matched | `chain_moved` |

* A missing ion **together with** a common shift means the chain re-centred: that is a
  loss, never "an end ion went dark".
* After a re-arrangement the new configuration is fitted with the linear-chain
  equilibrium model (James 1998), using the previous length scale and centre as priors,
  so hidden **dark ions are located** and still counted.
* Nothing is reported until it has persisted for `tracker.confirm_time` (default 0.5 s)
  and at least `confirm_frames` frames. Noise peaks and cosmic rays never become events.
* Brightness is normalised by exposure × EM gain, so changing those settings does not
  look like an ion going dark.
* With a single ion, "dark" and "lost" can't be told apart from the image, and ioncam
  says so.

The detector uses a matched filter with a robust spatial noise map; EMCCD noise is skewed
at low light, so the threshold comes from upper quantiles, not from a Gaussian assumption.
It measures each spot's width from the log-curvature of the filtered peak, which
separates ions from clouds (wider) and from single-pixel spikes (narrower). Measured on
the simulator: about 0.02 px position error for separated ions, about 1 false spot per
15 empty full frames at 6σ (removed by the temporal confirmation), and around 15 ms per
512×512 frame (about 1 ms with an analysis ROI).

## Grabber readout, step by step

`calibrate_readout()` (the GUI's **Calibrate + arm**) does:

1. **Locate.** It takes `calibration.n_frames` exposures with the `[locate]` settings
   (0.5 s, minimum EM gain, 1×1 binning), averages them, and jointly fits a
   pixel-integrated Gaussian to *all* ions at once. Overlapping PSFs in dense chains are
   resolved this way; ions the detector merged are recovered from the chain model when
   `n_ions` is given. Dark ions known to the tracker are carried along.
2. **Plan.** For every binning-grid phase it gives each ion its own ROI. An ion centred
   in a super-pixel needs 1, an ion on a boundary needs 2. The layout maximises the worst
   ion's light **containment** (exact Gaussian integrals with the fitted PSF) while
   penalising **crosstalk** and area. The AQROI is the bounding box plus `frame_margin`
   super-pixels, so it is centred on the crystal, and it includes a background ROI.
   With one ion at 4×4 binning this gives an 8×8 px AQROI with the ion at the centre of
   2×2 super-pixels and over 99% containment, the same layout as an AQROI of
   `257,264,161,168` in Solis.
3. **Verify.** It takes `verify_frames` frames with the *final* readout settings
   (internal trigger) and checks that every ion stands out in its own ROI.
4. **Arm.** External trigger, Camera Link on, run-till-abort, so every TTL gives one
   frame and nothing has to be re-armed per shot.

The returned plan includes `binning_advice`, which shows how other binnings would score.
For example, a 3-ion chain with 6 px spacing gets 85% / 12% (containment / crosstalk) at
4×4, but 96% / 3% at 2×8 with an even smaller frame. `auto_binning=True` picks the
cheapest binning that meets the targets.

Your Solis readout settings map onto `[readout]` (the defaults):

| Solis | ioncam `[readout]` |
|---|---|
| Acquisition mode: single | `acquisition = "run_till_abort"`: every trigger gives a frame with no re-arming, so no missed shots |
| Trigger: external | `trigger = "external"` (`fast_ext_trigger` optional) |
| Readout mode: image | always image |
| Exposure 0.0015 | `exposure = 0.0015` |
| Vertical pixel shift speed 0.5 | `vs_speed_us = 0.5` (matched by value) |
| Vertical clock voltage amplitude Normal | `vs_amplitude = 0` |
| Horizontal readout rate 5 MHz | `hs_speed_mhz = 5.0` (matched by value) |
| Pre-amplifier gain 3 | `preamp_gain_index = 2` |
| EM gain on, level 200 | `amplifier = "em"`, `em_gain = 200` with `camera.em_gain_mode = 3` (real gain) |
| Baseline clamp | `baseline_clamp = true` |
| Binning 4×4, software binning off | `hbin = vbin = 4` (always on-chip) |
| AQROI 257,264,161,168 | computed per calibration, reported as `aqroi_solis` |
| Image orientation: clockwise | **display only** (`[display] rotation = 90`) |

Some points that matter when using the Grabber:

* **Orientation.** Andor applies rotate/flip in host software, never in the camera, so
  Camera Link carries the raw sensor order. ioncam never uses the SDK's rotate/flip: all
  geometry is in raw sensor coordinates, and rotation exists only in the GUI display. If
  you want to confirm this on your hardware, run `ioncam_grabber_mapping_check.py` once
  (`grabber.flip_x/flip_y` exist just in case).
* **Frame size check.** After arming, the core device log should print
  `grabber0 frame size: WxH` with the W×H of the plan.
* **ROI semantics.** Grabber ROIs are half-open, `x0 ≤ x < x1`. The gateware only clears
  its "in ROI" x flag when the pixel counter *equals* `x1`, so an ROI with `x1` beyond the
  frame width silently sums whole rows. ioncam never produces that, and
  `analysis/readout.py` contains a cycle-faithful model of the ROI engine that the tests
  check against.
* **Timing.** The plan reports `min_trigger_period` (exposure + readout + keep-clean).
  Keep the ROI gate open from the trigger until the frame has arrived
  (exposure + readout + margin), as in the example experiment.
* **During experiments.** USB copies of every frame feed `get_readout_counts()` and
  `get_readout_stats()`. If more light than at arming falls outside the ROIs, you get
  `readout_geometry_changed`; any crystal change in live mode marks the plan
  **outdated** (`is_readout_valid()` returns False).

## Taking over

* `hold_tracking("shuttle", 30)` … `resume_tracking()`: no loss alarms while ions move
  deliberately. Afterwards the new crystal is compared with the old one and `hold_done`
  reports the counts before and after (and `ion_lost` if any were lost).
* `take_control("artiq:xyz", reason, timeout)`: an advisory lease. GUIs show who is
  driving the camera, and it expires automatically.
* `locate_ions()` doubles as a precision position sensor, e.g. for calibrating shuttling
  voltages (see `ioncam_shuttle.py`).
* Overexposure protection: with EM gain above the safe level and saturated pixels, live
  gain is dropped to the minimum (`overexposure` alarm). During readout it only warns.

## Layout

```
ioncam/
  camera/base.py        coordinate conventions, AcquisitionSettings, CameraBase
  camera/andor.py       iXon via pyAndorSDK2 (single SDK thread, uint16 zero-copy hot path)
  camera/simulator.py   ion chain physics + EMCCD model + timing/trigger model
  camera/fake_sdk.py    emulated pyAndorSDK2 (runs the real driver without hardware)
  analysis/detection.py matched-filter detection, noise maps, cloud/spike rejection
  analysis/fitting.py   joint pixel-integrated Gaussian fit (analytic Jacobian)
  analysis/chain.py     James equilibrium positions, chain fit with hidden ions
  analysis/tracker.py   crystal tracking and event classification
  analysis/readout.py   AQROI/ROI planner, Grabber gateware model, verification, thresholds
  engine.py             camera thread, analysis thread, procedures, safety
  api.py                the RPC target (sipyco "ioncam")
  server.py             aqctl_ioncam
  gui/                  ioncam_gui (pyqtgraph; PyQt5/PyQt6/PySide6)
examples/artiq/         device_db entry and experiments
tests/                  pytest suite (simulator, emulated SDK, real sipyco RPC, GUI smoke)
```

## What was wrong with the previous script (`beta5.py`)

These are worth knowing, because several would silently corrupt data:

1. **Negative pixel values.** `pyAndorSDK2`'s `GetMostRecentImage16`/`GetImages16` use
   *signed* `c_short` buffers, so every pixel above 32767 (EM-gain saturation, bright
   ions, cosmic rays) wrapped to a negative number. ioncam reads the DLL directly into a
   `uint16` array.
2. **The vertical shift speed was never set.** `SetFKVShiftSpeed` is the *fast-kinetics*
   speed and does nothing in image mode. ioncam uses `SetVSSpeed` and `SetVSAmplitude`.
3. **The EM gain was never applied.** `SetEMCCDGain` was only called when the exposure
   changed, after `StartAcquisition` (the SDK refuses with DRV_ACQUIRING), and before
   `SetEMGainMode`. ioncam sets the gain mode once, then the gain while idle, and checks
   every return code.
4. **Thread safety.** SDK calls came from both the Tk thread and the worker thread (SDK2
   is not thread-safe), and Tk/matplotlib widgets were redrawn and read from the worker
   thread (Tk is not thread-safe), which causes random freezes and crashes. ioncam has one
   camera thread for every SDK call, and the GUI is updated only in its own thread.
5. **Speed.** The figure was cleared and redrawn with matplotlib every frame,
   `SimpleBlobDetector` was rebuilt every frame, the temperature was read every frame, and
   images were reduced to 8 bits before detection.
6. **Ion detection** was one count threshold (55000) in a fixed box, with no notion of
   individual ions, chains, dark states, losses or clouds.
7. **Geometry.** `SetImageRotate(1)` (host-side rotation) plus a manual reshape and flip
   made displayed coordinates differ from the Camera Link stream, so ROIs taken from the
   display would not match the Grabber.
8. Smaller issues: ROI width and height were swapped (`y_end = iy + ROIW`), "learn
   background" also learned (and subtracted) the ions, the safety rule used fixed counts
   regardless of bias and gain, and the program exited via `os._exit(0)`.

## Tests

`pytest` runs 78 tests in under a minute:
- chain physics against James' published values;
- detector accuracy and false-alarm rates;
- the joint fit and its Jacobian;
- every tracker scenario (load via cloud, dark vs lost at every chain position, loss with
  a dark ion present, melting, lasers off, rigid moves, shuttling with hold, a quiet
  stable crystal);
- planner invariants, and agreement with the Grabber gateware model, including its `x1`
  quirk;
- the Andor driver's exact SDK call sequence on the emulated SDK;
- a full stack over real sipyco RPC with concurrent clients;
- a headless GUI smoke test.
