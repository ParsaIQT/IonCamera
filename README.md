# ioncam — the camera brain of an ion trap

`ioncam` runs an Andor **iXon Ultra 897** for a trapped-ion experiment:

* **live view** for beginners, with a large colour-coded "N ions" status and alarms;
* **intelligent crystal tracking**: it tells an ion that was **trapped**, **lost** (the
  crystal re-arranges) or went **dark** (everything else stays exactly where it was).
  It also recognises hot **clouds**, **melting**, **lasers off** and **rigid moves**
  (shuttling, a bumped camera), and it infers where dark ions sit;
* **careful Grabber readout preparation**. A long exposure at low gain locates every
  ion to a few hundredths of a pixel and measures the PSF, including its broad halo
  (which is what makes neighbouring ROIs see each other). The camera **AQROI** is then
  centred on the crystal and aligned to the binning grid, each ion gets **its own
  Grabber ROI** sized for maximum light containment and minimum crosstalk, and
  everything is **verified with the real readout settings** before the camera is
  **armed** (external trigger + Camera Link);
* **learned background**. The static light of the trap surface - laser scatter,
  defects and dust that look like ions or clouds, electrode edges, hot pixels - is
  learned once and subtracted from then on, at any ROI and binning;
* **monitoring during experiments**. The iXon sends every Grabber frame over USB as
  well, so ioncam keeps watching the ions, fits bright/dark thresholds and warns when
  the crystal changes under a running experiment;
* a **sipyco RPC server** (`aqctl_ioncam`, ARTIQ controller conventions) so ARTIQ
  experiments can wait for ions, calibrate and arm the readout, get the ROIs, hand off
  control while shuttling, and so on;
* a **Qt GUI** (`ioncam_gui`) that runs with the camera in-process or as a remote viewer;
* a physically realistic **simulator**: ion chain physics, ~13k photons/s per ion into
  the camera, a blurred PSF with a halo, laser scatter from the trap surface and surface
  defects out of focus (set by NA and ion height), from a clean to a horrendous trap,
  and an EMCCD noise model - so everything can be tried and tested without hardware.

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
pytest                         # 93 tests (~2 min), all on the simulator / emulated SDK
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
| `locate_ions(expected)` | long exposure + joint fit: positions (with errors), PSF core + halo, chain fit |
| `calibrate_readout(n_ions, hbin, vbin, auto_binning, ...)` | locate → plan AQROI/ROIs → verify → arm |
| `get_readout_plan()` / `get_grabber_rois()` / `get_grabber_roi_lists()` / `is_readout_valid()` | the plan |
| `arm_readout()` / `disarm_readout(resume_live)` | external trigger + Camera Link on/off |
| `get_readout_counts(since_shot)` / `get_readout_stats()` | per-ROI sums of every shot (USB copy), threshold fits |
| `hold_tracking(reason, timeout)` / `resume_tracking()` | "ions will move on purpose" (shuttling, camera) |
| `take_control(owner, reason, timeout)` / `release_control(owner)` | advisory lease shown in all GUIs |
| `start_live(**settings)` / `set_live(...)` / `set_exposure` / `set_em_gain` / `set_roi` / `stop()` | live video |
| `get_frame(since_index, bin, roi, view)` | newest image (raw orientation); `view="corrected"`: background removed, as the detector sees it |
| `learn_background(n, mask_ions)` / `get_background()` / `clear_background()` | learn / inspect / forget the static surface light (see below) |
| `sim_*` | drive the simulated trap: ions, lasers, repumper, oven, camera moves, triggers; `sim_set_params()` for surface, optics and light |

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
The ions are masked out of the noise and background statistics, so a camera ROI that
hugs the crystal works as well as the full frame - without mistaking the photon-counting
noise of EM gain 300 (just as skewed) for a frame full of ions. (`locate_ions` still pads
its ROI to at least 64 × 64 px, so the fit always has background around the ions.)
Ions whose images overlap (blurred PSF, dense chains) are separated by joint fits with
the PSF model (core + halo) and model selection, started from the tracked crystal when
there is one. A hot, uncrystallised ion is recognised as one broad blob rather than a
cluster of faint "ions", and faint clouds are found by integrating the light no ion
explains over ~0.3 s against the usual background.

Measured on the simulator (13k photons/s per ion, PSF σ 1.8 px + 15 % halo, typical
surface): every ion found and ~0.2 px rms per 50 ms frame at EM gain 50-100 (the photon
limit); single frames resolve chains down to ~2.7 σ spacing, denser ones are followed
through the tracker's prior; no false clouds and < 1 false spot per 7 empty frames at EM
gain 300; ~26 ms per 512×512 frame, ~12-14 ms with a smaller ROI (scipy; installing
`opencv-python-headless`, the `fast` extra, speeds up the filtering).

## Grabber readout, step by step

`calibrate_readout()` (the GUI's **Calibrate + arm**) does:

1. **Locate.** It takes `calibration.n_frames` exposures with the `[locate]` settings
   (0.5 s, EM gain 10, 1×1 binning - a little EM gain keeps the read noise below the
   shot noise of ~10⁴ photons/s ions), averages them, and jointly fits a
   pixel-integrated PSF - a Gaussian core plus a broad Gaussian halo (aberration wings),
   kept when significant - to *all* ions at once. Overlapping PSFs in dense chains are
   resolved this way; ions the detector merged are recovered from the chain model when
   `n_ions` is given. Position errors are honest (EMCCD noise law, sandwich estimator).
   Dark ions known to the tracker are carried along. With `detection.auto_psf` the
   measured core and halo are used by the detector and the planner from then on.
2. **Plan.** For every binning-grid phase it gives each ion its own ROI. An ion centred
   in a super-pixel needs 1, an ion on a boundary needs 2. The layout maximises the worst
   ion's light **containment** (exact integrals of the fitted PSF, halo included) while
   penalising **crosstalk** and area. The AQROI is the bounding box plus `frame_margin`
   super-pixels, so it is centred on the crystal, and it includes a background ROI.
   ROI edges can only lie on the binning grid, so an ion sits as centrally in its ROI as
   the grid allows (the GUI draws the grid as dotted lines, and the plan lists each
   ion's offset from its ROI centre) - offsets of up to half a super-pixel are expected,
   not a drawing error. With one ion at 4×4 binning this gives an 8×8 px AQROI with the
   ion at the centre of 2×2 super-pixels, the same layout as an AQROI of
   `257,264,161,168` in Solis.
3. **Verify.** It takes `verify_frames` frames with the *final* readout settings
   (internal trigger) and checks that every ion stands out in its own ROI.
4. **Arm.** External trigger, Camera Link on, run-till-abort, so every TTL gives one
   frame and nothing has to be re-armed per shot.

The returned plan includes `binning_advice`, which shows how other binnings would score.
Real ion images are blurred and have wings, so neighbouring ROIs do see each other: a
3-ion chain with 6 px spacing (PSF σ 1.8 px + 15 % halo) gets 58 % / 11 % (worst
containment / crosstalk) at 4×4, 80 % / 10 % at 3×3 and 71 % / 4 % at 1×1.
`auto_binning=True` picks the cheapest binning that meets the targets
(`calibration.min_containment`, `max_crosstalk`).

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

## Learning the background (dirty traps)

Surface traps scatter the cooling laser: a broad, Gaussian-ish glow along the beam
(obvious in long exposures), plus bright points from defects, dust, scratches and
electrode edges. Out of focus these are blurred disks (radius h·tan(asin NA)·M/pixel for
a point at depth h below the ions - ~19 px at NA 0.25, 60 µm); dust standing up from the
surface is closer to focus and looks like an ion. **Learn background** (GUI, or
`learn_background()` over RPC) takes `n_frames` (default 50) of the full sensor without
binning at the live exposure and gain and keeps:

* the static light: a smooth part everywhere plus the fine structure where it is
  significant (defects, edges, hot pixels) - not the noise of the learning frames, and
  not single amplified photons (EMCCD bursts are told from hot pixels by the per-frame
  median);
* the camera's noise law (photon transfer: variance per count of signal), so the
  detector expects more noise where it subtracted bright static light.

From then on it is subtracted from every analysed frame - any ROI, binning, exposure and
EM gain (it is stored per sensor pixel and per unit exposure × gain) - and its brightness
follows the laser power frame by frame. Ions may stay in the trap (they are modelled and
masked); better, switch the **repumper off** (ions shelved: dark, but the laser still
scatters) or empty the trap, and tick *ions dark / trap empty* (`mask_ions=False`).
The display can show the raw image or the image with the background removed (what the
detector sees). Learn again after moving the camera or realigning the beam (ioncam hints
at it when the whole picture shifts).

On the simulator, a dirty surface with dust near focus gives ~20-50 false "ions" per
frame without it and none with it; the ions are still found.

## The simulator

`ioncam_gui --simulate` / `aqctl_ioncam --simulate` run on a simulated trap. The
*Simulation* tab (or `sim_set_params()` over RPC) sets: surface preset (`clean`,
`typical`, `dirty`, `horrendous`) and a new random surface, ion fluorescence (photons/s
reaching the camera, default 13 000), laser power, PSF core and halo, chain length scale,
uniform stray light, laser scatter (rate at the beam centre, widths along / across the
beam, angle, offset), defects (number, brightness, height above the surface), scratches,
electrode-gap scatter, collection NA and ion height; plus ions (load, lose, dark,
melt), lasers, repumper, oven, camera moves and external triggers. The EMCCD model has
Poisson photo-electrons, clock-induced charge, the gamma-distributed EM register, read
noise, bias and 16-bit saturation.

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

`pytest` runs 93 tests in about 2 minutes, in the realistic regime (13k photons/s
per ion, blurred PSF with halo, laser scatter, surface defects):
- chain physics against James' published values;
- detector accuracy (photon-limited) and false-alarm rates at EM gain 1, 50 and 300,
  also with a camera ROI hugging the crystal, clouds vs ions, the learned background on a
  dirty surface with ion-like dust (any ROI and binning, laser power changes);
- the joint core + halo fit of a dense chain (unbiased, honest errors) and its Jacobian;
- every tracker scenario (load via cloud, dark vs lost at every chain position, loss with
  a dark ion present, melting, lasers off, rigid moves, shuttling with hold, a quiet
  stable crystal);
- planner invariants, its containment / crosstalk against the simulated image (halo
  included), and agreement with the Grabber gateware model, including its `x1` quirk;
- the Andor driver's exact SDK call sequence on the emulated SDK;
- a full stack over real sipyco RPC with concurrent clients;
- a headless GUI smoke test.
