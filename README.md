# ioncam — the camera brain of an ion trap

`ioncam` runs an Andor **iXon Ultra** - the 897 (512 × 512) or the 888 (1024 × 1024), or
any iXon on Andor SDK2: it reads the sensor, speeds and gains from the camera - for a
trapped-ion experiment:

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
  **armed** (external trigger + Camera Link). Ions too close for the binning still get
  ROIs (overlapping, with the crosstalk reported), and everything can be adjusted by
  hand - a D-pad moves the AQROI in 1 px steps;
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
 iXon Ultra     ─┤USB  aqctl_ioncam / ioncam_gui                        │
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
ioncam_gui --simulate --sim-camera 888    # ... as an iXon Ultra 888 (1024 x 1024)
aqctl_ioncam --simulate        # headless RPC server with the simulator
pytest                         # 177 tests (~6 min), all on the simulator / emulated SDK
```

### The GUI

On top, the crystal status in big colour-coded letters (green stable, orange changing,
red melted or no light, blue tracking on hold), with chips for the camera mode (or the
step of a running procedure), the sensor temperature, the frame rate and who has
control. On the left the image over each ion's brightness trace; on the right the tabs,
in the order of the work:

| tab | what is there |
|---|---|
| **Live** | start / stop, exposure, EM gain, binning, **Snap** (average N frames and save them), the camera ROI (crop to the view, around the ions with a margin, full sensor), display (colormap, levels, orientation) and the live readout electronics (amplifier, horizontal and vertical speed, pre-amp, frame transfer) |
| **Ions** | the crystal at a glance (bright, dark, state, chain scale and angle, PSF σ, noise, the brightness in photoelectrons/s), every site (with its photoelectrons/s; every row in view - a short window scrolls the page), tracking (accept as reference, forget, hold, locate), ion detection (where to search, threshold), the learned background, a loading alert |
| **Readout** | state chips (plan valid or outdated, armed, shots, verified); calibrate and run (Arm, Disarm, inside the AQROI set by hand); the camera settings of the long exposure and of the readout side by side (see below); the AQROI by hand (New AQROI, Place by click, the size pad, the D-pad); the results: shot histograms with the fidelity estimate, the plan (and a typed AQROI), the ARTIQ snippet |
| **Events** | the event log, filtered by severity or text; save it to a file |
| **Camera** | cooling, the shutter (internal and external output, polarity, opening and closing times; Open / Close), the cooling fan (full, low, off), the control lease, configuration (**Settings...**: every option with its help, changed at run time; **Save configuration...**), camera information |
| **Simulation** | (simulated camera) ions, the lab (lasers, repumper, oven, camera moves, triggers) and the realism knobs |

The toolbar holds what is needed from everywhere - Live, Stop, Snap, Learn background,
Hold, Locate, Calibrate + arm, Verify, Arm, Disarm, Abort, and save image, settings and
the dark / light theme - with what cannot run now greyed out (Abort only while a
procedure runs). The bar above the image: fit, zoom to the ions, what to show (raw,
background removed, the learned background), which overlays (markers, numbers, ROIs,
detected spots), where to search for ions, and save the image (the raw data as `.npy` / `.csv` with a `.json` of
its geometry and camera settings, or the view as `.png`). Keyboard shortcuts (**F1**
lists them): F5 live, F6 stop, Ctrl+K calibrate + arm, Ctrl+L locate, Ctrl+Shift+V
verify, Ctrl+R arm, Ctrl+D disarm, Esc abort, Ctrl+B learn background, Ctrl+H hold,
Ctrl+S save image, Ctrl+0 fit, Ctrl+I zoom to the ions, Ctrl+T theme, Ctrl+, settings.
The theme and the window layout are kept between sessions.

The image brightness follows every frame (**Auto** under the histogram). Drag the
histogram's level handles - or untick Auto, or right-click the histogram → *Auto
levels* - and the levels stay exactly where you put them; **Fit** sets them to the
current frame once.

With the real camera on the camera PC (Windows):

1. Install the Andor SDK2 driver and Andor's `pyAndorSDK2` package (`pip install .` in
   the folder from Andor's zip). **Close Andor Solis**, since only one program can own
   the camera. Install ioncam itself from your git checkout **in editable mode**, once:
   `pip install -e ".[gui]"` (if you installed it with `pip install .` before, first
   `pip uninstall ioncam`). Then a `git pull` is all an update takes - just restart
   `ioncam_gui` / `aqctl_ioncam`; re-run the pip command only if the dependencies
   changed. Install sipyco into the same Python too
   (`pip install git+https://github.com/m-labs/sipyco`, which needs git; ARTIQ
   environments already have it): without it ARTIQ cannot connect, and the GUI says so. The window title (and `get_version()`) shows the commit that is running,
   e.g. `1.0.0 (git dd10636)`; `installed copy` means a copy in site-packages that does
   not follow the checkout. (Without installing at all, `python -m ioncam.gui.app` and
   `python -m ioncam.server` run straight from the checkout.)
2. `aqctl_ioncam --write-config ioncam.toml`, then edit it. Every option is commented.
3. `ioncam_gui -c ioncam.toml` opens the camera, starts live video and serves RPC on port
   3270: the chip at the right end of the status bar reads **RPC :3270** when ARTIQ can
   connect (see [Troubleshooting](#troubleshooting) otherwise). Alternatively run `aqctl_ioncam -c ioncam.toml --bind '*'` headless, and from
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
| `locate_ions(expected, region)` | long exposure + joint fit: positions (with errors), PSF core + halo, chain fit |
| `calibrate_readout(n_ions, hbin, vbin, auto_binning, aqroi, ...)` | locate → plan AQROI/ROIs → verify → arm; `aqroi`: keep an AQROI set by hand |
| `set_search_region(x0, y0, x1, y1)` / `clear_search_region()` | look for ions only there (live, locate, calibrate) - stray light elsewhere is never an ion |
| `manual_aqroi(cols, rows, x, y)` / `resize_readout(dcols, drows)` | an AQROI by hand: N × M super-pixels centred on (x, y); more / fewer columns and rows |
| `nudge_readout(dx, dy)` | move the AQROI by whole sensor pixels (the ROIs move with it); re-arms |
| `replan_readout(hbin, vbin, aqroi)` | plan again from the located ions: another binning and/or a fixed AQROI (ions outside it are left out) |
| `verify_readout()` | test frames with the current plan (e.g. after adjusting it by hand) |
| `get_readout_plan()` / `get_grabber_rois()` / `get_grabber_roi_lists()` / `is_readout_valid()` | the plan |
| `arm_readout()` / `disarm_readout(resume_live)` | external trigger + Camera Link on/off; arming follows the crystal on screen first if it changed since the plan, and returns the plan armed |
| `get_readout_counts(since_shot)` / `get_readout_stats()` | per-ROI sums of every shot (USB copy), threshold fits |
| `get_readout_histograms(source)` | histograms of every ROI (verification or triggered shots) and, with every shot prepared bright, a threshold and quick fidelity estimate per ion |
| `get_readout_check(last)` | while armed: is the crystal still there? `verdict` ok / dark / lost / no_light from the averaged readout frames, every ion's light (see below) |
| `hold_tracking(reason, timeout)` / `resume_tracking()` | "ions will move on purpose" (shuttling, camera) |
| `take_control(owner, reason, timeout)` / `release_control(owner)` | advisory lease shown in all GUIs |
| `start_live(**settings)` / `set_live(...)` / `set_exposure` / `set_em_gain` / `set_roi` / `stop()` | live video |
| `get_ion_brightness()` | every ion's brightness in photoelectrons/s, the mean of the bright ones, and the conversion's inputs (see below) |
| `start_characterization(**options)` / `characterization_status()` / `characterization_results(directory, **spam)` / `list_characterizations()` / `characterization_point(index)` | the camera characterization for SPAM (see below): start it (`dry_run=True`: only the plan), follow it, its results and SPAM ranking, one setting's histogram; `abort()` stops it |
| `set_shutter(mode, external, ttl_high, open_ms, close_ms)` / `get_shutter()` / `set_fan(mode)` | the shutter (`"open"`, `"closed"`, `"auto"`, `"series"`: open while acquiring) and the cooling fan (`"full"`, `"low"`, `"off"`); live video or the armed readout carries on |
| `get_frame(since_index, bin, roi, view)` | newest image (raw orientation); `view="corrected"`: background removed, as the detector sees it; `view="background"`: the learned background |
| `learn_background(n, mask_ions)` / `get_background()` / `clear_background()` | learn / inspect / forget the static surface light (see below) |
| `set_acquisition(section, **settings)` | camera settings of the long exposure (`"locate"`) or the readout (`"readout"`), several at once, e.g. `vs_speed_us=1.13, hs_speed_mhz=10`; an armed readout is re-armed with them at once |
| `get_config()` / `get_config_help()` / `set_config(section, key, value)` / `save_config(path)` | every option, what it means, change it at run time, write a commented file |
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
| refocusing | the same ions, imaged wider (or sharper) | `focus_changed` (wider); the PSF is re-measured |

* A missing ion **together with** a common shift means the chain re-centred: that is a
  loss, never "an end ion went dark".
* In a tight crystal (two ions ~4 px apart with a 1.2 px PSF) single frames often show
  two ions as one spot, and after a loss the remaining ions re-centre towards the gap -
  which looks much the same. So a missing ion is decided on the **average of the frames
  that show the change**: dark if the other ions are exactly where they were, lost if
  they re-arranged and the old crystal no longer explains the average, nothing if the
  average still shows the old crystal (dim chains produce odd single frames all the
  time). The crystal's light budget helps: two ions seen as one keep their light, a lost
  ion takes one ion's worth with it.
* A dark ion's place is known only relative to the bright ions, so it keeps that place
  exactly. When it goes dark it is put where the averaged frames last measured it. From
  then on it moves only as the centroid of the bright ions moves, in every frame and at
  every re-measurement. The chain's length and angle are fitted to the positions measured
  on averaged frames (every `tracker.verify_period`, 2 s), so they do not jitter with
  single frames.
* The PSF (core + halo) is measured on averaged frames of the live crystal and handed to
  the detector (`detection.auto_psf`), so a wrong `detection.psf_sigma` corrects itself
  without a calibration. One measurement alone jitters by a few per cent, so successive
  ones are averaged, and a clearly different one counts only when the next one agrees:
  the detector, the brightness references and the markers do not change every few
  seconds. After refocusing, a detector still assuming the sharper image
  would split each ion in two: if the same ions with a re-fitted, wider PSF explain the
  frames, it is reported as `focus_changed`, not as trapped ions.
* The green circles around the ions hold `tracker.marker_light_fraction` (80 %) of an
  ion's light, from the PSF measured in the image (core + halo); after refocusing they
  settle to the new size within `tracker.marker_settle_time` (5 s), whatever the frame
  rate. The trace under the image is each ion's brightness relative to its reference,
  measured in every frame (a dark ion drops to ~0).
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
gain 300; ~19 ms per 512×512 frame, ~10 ms with a 200×140 ROI (scipy; installing
`opencv-python-headless`, the `fast` extra, speeds up the filtering). ioncam runs numpy's
BLAS with one thread (`OPENBLAS_NUM_THREADS=1` unless you set it yourself): its matrix
products are small, and BLAS worker threads competing with the camera, analysis and GUI
threads made the live analysis about three times slower.

## Grabber readout, step by step

`calibrate_readout()` (the GUI's **Calibrate + arm**) does:

1. **Locate.** It takes `calibration.n_frames` exposures with the `[locate]` settings
   (0.5 s, EM gain 10, 1×1 binning - a little EM gain keeps the read noise below the
   shot noise of ~10⁴ photons/s ions), averages them, and jointly fits a
   pixel-integrated PSF - a Gaussian core plus a broad Gaussian halo (aberration wings),
   kept when significant - to *all* ions at once. Overlapping PSFs in dense chains are
   resolved this way; ions the detector merged are recovered from the chain model when
   `n_ions` is given - even a pair seen as a single spot, split along its elongation
   (or the tracked chain axis) - and, with `n_ions = 0`, when the tracker sees more ions
   than the long exposure resolved. Position errors are honest (EMCCD noise law,
   sandwich estimator).
   Dark ions known to the tracker are carried along. With `detection.auto_psf` the
   measured core and halo are used by the detector and the planner from then on.
   When the tracker follows a stable crystal, only that crystal is located: spots of
   the long exposure away from it - surface light such as defects and dust, which the
   exposure shows when no background is learned (e.g. after turning the simulated
   trap) - are ignored, and the plan's warnings say how many. Spots far apart are
   fitted in windows of their own, so the fit never spans the sensor; a calibration
   takes about 2 s (two 0.5 s exposures, the plan, 200 verification frames).
2. **Plan.** By default every ion gets **one super-pixel** (`calibration.roi_size =
   [1, 1]`, "ROI per ion" in the GUI), and of all binning-grid phases the planner picks
   the one that puts the ions closest to the centres of their super-pixels. Every
   super-pixel adds its own dark counts (clock-induced charge, scatter, read-noise
   spread) to an ROI sum, so the fewest pixels around the brightest part of the ion -
   rather than the most light - usually give the best bright/dark discrimination (SPAM
   fidelity). With the lab trap (PSF 1.2 px + halo, two ions 4.3 px apart) at 4×4
   binning that is 66-68 % of each ion's light in its super-pixel, with 2-8 %
   crosstalk. `roi_size = [0, 0]` ("auto") instead sizes each ROI for the most light
   (exact integrals of the fitted PSF, halo included), penalising crosstalk and area -
   bigger ROIs, more dark counts. The AQROI is the bounding box plus `frame_margin`
   super-pixels, so it is centred on the crystal; its background ROI is a strip of
   super-pixels placed far enough out to hold < 2 % of an ion's light. ROI edges can
   only lie on the binning grid, so an ion sits as centrally in its ROI as the grid
   allows (the GUI draws the grid as dotted lines, and the plan lists each ion's offset
   from its ROI centre); with several ions their spacing decides how well all can be
   centred at once.
   Ions closer than the binning can separate are never refused: each still gets its own
   best ROI, the ROIs overlap (the Grabber sums each one independently) and a warning
   names them, so the crosstalk can be measured - e.g. in a SPAM experiment - before
   thresholding. Two ions inside one super-pixel get the same ROI; only less binning
   tells them apart.
3. **Verify.** It takes `verify_frames` (200) frames with the *final* readout settings
   (internal trigger) and checks that every ion stands out in its own ROI. The ions shine
   in every frame, so these frames also give the first shot histograms and a quick
   fidelity estimate with a threshold per ion (see below).
4. **Arm.** External trigger, Camera Link on, run-till-abort, so every TTL gives one
   frame and nothing has to be re-armed per shot.

The returned plan includes `binning_advice`, which shows how other binnings would score.
Real ion images are blurred and have wings, so neighbouring ROIs do see each other: a
3-ion chain with 6 px spacing (PSF σ 1.8 px + 15 % halo) gets 37 % / 6 % (worst
containment / crosstalk) with one 4×4 super-pixel per ion - a super-pixel smaller than
the ion image holds little of its light, and the plan says so - while auto-sized ROIs
get 58 % / 11 % at 4×4, 80 % / 10 % at 3×3 and 71 % / 4 % at 1×1.
`auto_binning=True` picks the cheapest binning that meets the targets
(`calibration.min_containment`, `max_crosstalk`).

### The AQROI by hand

On the image every Grabber ROI has its own colour and a label beside it - **ion 0 87 %**
is Grabber ROI 0 with the share of ion 0's light it contains (chain order, as in the plan
table), **bg** the background ROI - plus the AQROI (dashed) and the binning grid
(dotted). The Readout tab's **AQROI by hand** box plans for the located positions (before
any locate, the tracked ones; once the crystal changed, the ions on screen - see *When the
crystal changes* below) and re-computes containment and crosstalk at once:

* **New AQROI**: a square AQROI of N × N super-pixels (5 by default) on the ions in view -
  zoom to the ion first - or the middle of the view, on the binning-grid phase that
  centres the ions in their super-pixels (as an automatic plan would). **Place by
  click**: then click on the image, and the square is centred there.
* **Size pad** (next to the D-pad): **wider / narrower / taller / shorter** by one
  super-pixel as seen on screen, through the display rotation. It adds a column or row on
  the side with the narrower margin around the ions' ROIs (and takes one from the wider
  side), so the ions stay centred and the grid stays; it never cuts through an ion's
  ROI. The ROIs are placed again inside.
* **D-pad** (▲ ▼ ◀ ▶, step 1 px by default; hold to repeat): moves the AQROI - and every
  ROI with it - on screen. The Grabber ROI numbers stay the same (they are frame
  coordinates); the ions shift the other way inside them, at 4×4 binning by a quarter
  super-pixel per step. **Undo moves**: back to where the D-pad found it.
* **Re-plan**: plan again with the binning set above, without a new locate. **Verify**:
  test frames with the plan as it is now; they also refresh the shot histograms, so you
  can step with the D-pad and compare the fidelity.
* **Use this AQROI** (under the *Plan* tab): place the ROIs inside an AQROI you type in
  (sensor px, `x1`/`y1` exclusive, its size a multiple of the binning).

An AQROI set by hand reads out the ions **inside** it: located spots outside it - stray
light taken for ions, or ions you leave out on purpose - are listed in the plan's
warnings, not an error. With **inside the AQROI set by hand** ticked (it is, once you
make one), **Calibrate + arm** keeps that AQROI: it searches for the ions only inside it,
locates them precisely, places their ROIs in it, verifies and arms. An armed readout is
re-armed after every change by hand.

**When the crystal changes.** A plan reads out the ions it was made for. Once the live
tracking sees the crystal change after they were located - an ion trapped or lost, the
crystal re-arranged or moved - and shows other ions in the AQROI, the plan is outdated
(the *plan outdated* chip), and the next step follows the crystal on screen:

* **Arm**: if the crystal only moved and took the planned ions along, the AQROI moves
  with it by whole pixels and the Grabber ROIs stay as they were. If ions were trapped,
  lost or re-arranged, an AQROI of the same size is centred on the ions near the old one
  and their ROIs are placed again. The event log says which (`readout_adjusted`);
  `arm_readout()` returns the plan it armed - use its ROIs.
* **D-pad**: the AQROI moves as asked, and the ROIs are placed again for the ions on
  screen inside it (none inside yet: the ROIs stay as they were, so the AQROI can be
  walked over to the ions).
* **New AQROI**, the size pad and **Re-plan** plan for the ions on screen.

Ions that moved by less than 1 px keep their planned positions, so plans by hand in a row
stay put. Without a change seen since the locate, the locate wins over the tracking (it
is the more precise). Before, these steps planned for the last locate's ions however
long ago it ran: with a locate of one ion and a second ion loaded since, a 2 × 2 AQROI
by hand read out one ion, and the armed readout's banner said **1 ion**.

From ARTIQ the same is `manual_aqroi(cols, rows, x, y)`, `resize_readout(dcols, drows)`,
`nudge_readout(dx, dy)`, `replan_readout(hbin, vbin, aqroi)`, `verify_readout()` and
`calibrate_readout(..., aqroi=...)`.

### Shot histograms and a quick fidelity estimate

The bottom of the Readout tab shows histograms of the shots. Each ion ROI is drawn in its
colour on the image and the background in grey, with each ion's threshold as a dashed
line. The y axis is logarithmic by default, because the tails, where bright and dark
overlap, set the fidelity. **shots: latest** shows the newer of two sources: the last
verification (**Verify** or **Calibrate + arm**) or the externally triggered shots since
arming. The other two choices pick one source.

The estimate assumes **every shot is prepared bright** - do that on purpose:

* each ion ROI's sums are its bright distribution;
* the dark reference is the background ROI's super-pixels from the same shots, summed to
  the ion ROI's size. With one super-pixel per ion, every background super-pixel is one
  sample of what that ROI reads with the ion dark;
* **bright if sum >** is the threshold that misclassifies the fewest, counted directly
  on the shots with no model of the distributions. **bright→dark** is the fraction of
  bright shots at or below it, **bg→bright** the fraction of dark-reference samples
  above it, and **fidelity** is 1 minus their mean;
* **neighbours bright** adds the other ions' light to the dark reference: the dark error
  of an ion whose neighbours are bright. The crosstalk comes from the plan's crosstalk
  matrix and the measured signals, added as EM-amplified photo-electrons (the counts per
  photo-electron come from the bright shots' excess noise).

On the simulated version of a tight trap (PSF 1.2 px, two ions 4.3 px apart, one 4×4
super-pixel each, 2000 shots) the estimate predicted that ions 0 and 1, when dark, read
bright 5.3 % and 4.1 % of the time; with the ions really dark it measured 4.7 % and 3.8 %.
With the other ion bright it predicted 8.5 % and 16.9 %, and measured 7.5 % and 16.0 %.
The estimate has these limits:

* the bright error counts every shot in which an ion did not shine (failed preparation,
  decay, loss), which is what a SPAM experiment counts too;
* a dark neighbour also lowers the bright counts a little (its crosstalk is missing),
  which is not modelled;
* the background strip lies a few pixels from the ions, so a scatter gradient between the
  strip and the ions' row is not captured;
* 200 shots resolve errors only down to 0.5 %, so trigger more shots for finer numbers.

From ARTIQ: `get_readout_histograms(source="auto")`. The verification in a plan also
carries `threshold`, `fidelity` and `dark_error_neighbours_bright` per ion, and the
example `ioncam_grabber_readout.py` has an `all_bright` option that takes its thresholds
from this estimate.

Your Solis readout settings map onto `[readout]` (the defaults):

| Solis | ioncam `[readout]` |
|---|---|
| Acquisition mode: single | `acquisition = "run_till_abort"`: every trigger gives a frame with no re-arming, so no missed shots |
| Trigger: external | `trigger = "external"` (`fast_ext_trigger` optional) |
| Readout mode: image | always image |
| Exposure 0.0015 | `exposure = 0.0015` |
| Vertical pixel shift speed 0.5 | `vs_speed_us = 0.5`, or `0` for the fastest speed the camera recommends. A speed the camera has is used exactly; one it does not have (0.5 µs on an 888) is never guessed: the recommended one is used, and ioncam says so |
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
* **During experiments.** USB copies of every frame feed `get_readout_counts()`,
  `get_readout_stats()` and `get_readout_histograms()`. If more light than at arming
  falls outside the ROIs, you get `readout_geometry_changed` (in frames with pixels away
  from the ions to measure it against; from the light of the last 1000 shots added up,
  as one shot's few photo-electrons drown in its EM noise and clock-induced charge, and
  only once there is light enough to tell); any crystal change in live mode marks the plan
  **outdated** (`is_readout_valid()` returns False). The crystal check below watches the
  ions themselves.

### Is the crystal still there? (while armed)

While the readout is armed, the live tracking waits, but the averaged readout frames still
show where the light comes from. One shot holds a few photo-electrons per ion. Over many
shots, whatever states the experiment prepares, each super-pixel's mean signal is

> mean signal = Σ<sub>ions</sub> (fraction of the ion's light in this super-pixel) × (its
> mean light per shot) + a uniform background (laser scatter).

The fractions come from the located positions and the fitted PSF, core and halo, as in
the plan. Every 100 shots, ioncam fits the last 1000 shots this way. It tries the crystal
as calibrated, and also crystals of one, two or more ions fewer: an ion that is lost
leaves the others to re-arrange around the trap centre, so two ions become one in the
middle.

Means alone cannot tell a dim ion from a few counts of static pattern: the bias of binned
super-pixels differs by about 2 counts from column to column (17 Oct characterization),
and over 1000 shots a few counts at the ion's super-pixel are significant. So each ion's
ROI is also asked for the shots an ion gives and nothing else does: shots with more than
5 photo-electrons. Pixels without ion light in the same shots (the background ROI, or
super-pixels the ions miss) show how often the camera and the laser scatter alone get
there; a frame without such pixels uses the camera alone, from its read-noise peak, as
an upper bound. These bright shots are what makes the ROI's histogram differ from the
background's. The verdict comes from both:

| verdict | what the light says | event |
|---|---|---|
| `ok` | every ion at its place, shining | `readout_crystal_ok` (when back after a problem) |
| `dark` | an ion's light is gone, the others are in place: a dark state that does not end (a molecular ion, a long-lived level), or the experiment keeps that ion dark in every shot | `readout_ion_dark` (warning) |
| `lost` | the light fits fewer ions at their new places: an ion was lost and the crystal re-arranged (the plan is marked outdated) | `readout_ion_lost` (alarm) |
| `no_light` | no ion light: no ion's ROI has the bright shots a shining ion gives, whatever the means say. It also says whether the laser scatter is still there (ions dark or lost) or gone (lasers off?) | `readout_no_light` (warning) |
| `uncertain` | another crystal fits about as well, or an ion's ROI has too few bright shots to say it is there: too little light to tell yet | none |
| `unclear` | the light fits no crystal expected here (ions moved, wrong calibration): recalibrate | `readout_pattern_unclear` (warning) |

For your crystal of two ions at 45°, each in a diagonal super-pixel of a 2 × 2 AQROI, the
patterns are:
- **both ions:** the diagonal super-pixels lit, with the leakage into the other two
  (on the simulated trap 310 counts per shot in each diagonal super-pixel, 50-170 in the
  other two);
- **one dark:** one diagonal super-pixel lit, its neighbours with its leakage only;
- **one lost:** the other ion moves to the middle, so all four super-pixels are lit about
  the same.

**Where to see it.** While armed, the banner on top shows it, for example "2 ions ·
readout: in place, shining", "1 ion + 1 dark", or a red "ION LOST", and the Readout tab
has a crystal chip whose tooltip gives the full message. From ARTIQ:

```python
r = self.ioncam.get_readout_check()        # the latest check
if r["verdict"] in ("dark", "lost", "no_light"):
    print(r["message"])                    # e.g. "ion 1 dark: no light at its place ..."
r = self.ioncam.get_readout_check(300)     # or check the last 300 shots now
```

The result also holds each ion's mean light per shot (`ions`: `light` in counts,
`relative` to the others, `state`, and `bright_fraction`: the share of shots in which it
shone), each ROI's bright-shot test (`light`: `state` yes / no / unsure, `bright_shots`,
`threshold`, `camera_fraction`) and the mean frame above bias against the best crystal's
model (`signal`, `model`). Events need two checks in a row that agree. The banner shows
a verdict the light does not decide yet as "2 ions?" in grey, never in a confident colour.

**How it decides.**
- **Bright shots.** An ROI has ion light ("yes") when it has clearly more shots above 5
  photo-electrons than the camera and the scatter give (p below 10⁻⁵, and at least 0.5 %
  of the shots); none ("no") when the ion shone in fewer than 2 % of the shots (95 %
  upper limit). No ROI with ion light: `no_light`, decided. An ion whose ROI has none,
  beside an ROI that clearly has (at least 5 % and three times as many): dark, whatever
  its mean says. An ion whose ROI has about as many bright shots as the brightest is not
  called dark by its mean. `ok` needs every ion's ROI to say "yes".
- **Bias.** A 2 × 2 frame has no dark pixels, so the bias comes from the read-noise peak
  of the counts: shots in which a super-pixel caught no photo-electron are common in the
  darkest super-pixels, even with the ions bright in every shot.
- **Reference.** The first check that sees every ion shine, decided, is the reference
  (from 250 shots). It records each ion's light, so every ion is compared with its own
  brightness and the others, and the background, so the background cannot pass for a
  centred ion. Armed without an ion, no reference is taken.
- **Recent shots.** The last quarter of the window (250 shots) is checked on its own as
  well. When it decides something different, the crystal changed within the window, and
  the recent part is reported.
- **Giving up the planned crystal.** It takes a clearly better fit (χ² lower by 16) before
  an ion is called lost.

On frames made like the simulated lab trap's (PSF 1.2 px, ions 4.3 px apart, 3 × 3
binning, 2 × 2 AQROI, about 12 photo-electrons per ion per bright shot), over 40 random
runs per case, with clock-induced charge and a random 2-count bias pattern per super-pixel:
- **1000 shots:** no wrong verdict for ok, either ion dark, lost, lasers off or ions gone,
  with the ions bright in every shot, in half of them or in 20 % of them (2 of 720 runs
  undecided). The same holds with the PSF model 15 % too narrow or the halo
  under-estimated.
- **250 shots:** no wrong verdict either. A lost ion among ions bright in only 20 % of the
  shots is mostly undecided, and so is "ions gone" in about a third of the runs (two or
  three camera shots above 5 photo-electrons cannot prove the ROI empty); the full window
  then decides.
- **No ion, a few counts of pattern at the ion's super-pixel** (+5 or +6 counts, on the
  1 ion + background ROI frame and the two-ion frame): never anything but `no_light`. The
  means alone said `ok` or `uncertain` in up to 23 of 40 runs (one ion, +6 counts; two
  ions, +6 counts: 11 of 40).
- **Larger crystals:** a three-ion chain is handled the same way, with any one ion dark
  or one lost.

**Limits.**
- An ion the experiment keeps dark in every shot looks dark; one bright in fewer than 2 %
  of the shots counts as no light.
- A single ion that is dark looks like one that is lost: there is no light either way.
- Two ions lost from three (one left in the middle) look like the outer two dark; the
  message says so.
- The bias needs EM gain: without it, give the frame a background row.
- The reference waits until every ion shines. Until then, ions are judged against each
  other and by their bright shots only.
- Options are in `[readout_check]`: the window, how often to check, the dark level
  (15 % of the others' light) and the margin.

## The ions' brightness in photoelectrons per second

The Ions tab shows every ion's brightness in photoelectrons per second (and the mean of the
bright ones), from Andor's conversion (technical note FAQ040, and Count Convert in the iXon
Ultra hardware guides):

    photoelectrons = (counts - offset) x sensitivity / EM gain        (per exposure)
    photoelectrons/s = that / exposure

* **counts - offset**: each ion's light in counts, fitted with the PSF (core and halo) above
  its local background. The background takes the offset (about 200 counts on an iXon Ultra,
  with the baseline clamp), the scattered light and the dark current. A neighbour's light is
  attributed to the neighbour.
* **sensitivity**: electrons per count. It depends on the readout speed, the amplifier and the
  pre-amp gain, and ioncam asks the camera for it at the settings in use (the SDK's
  `GetSensitivity`). If your camera does not answer, or you want the number of its System
  Performance Sheet, set `camera.sensitivity`. The Ions tab says which one is used.
* **EM gain**: the set gain is the multiplication only in RealGain mode (`camera.em_gain_mode
  = 3`, the default). In the DAC modes the brightness is not given, and the Ions tab says why.
  RealGain is calibrated, but an EM register ages with heavy use; Andor's EMCAL recalibrates
  it.
* **exposure**: the camera's actual exposure (`GetAcquisitionTimings`).

The value is smoothed over about 10 frames, and stays the same when you change the exposure,
EM gain, pre-amp gain or readout speed. For photons arriving at the camera, divide by the
sensor's quantum efficiency at the ion's wavelength. A saturated ion (EM register or 16-bit
ADC) reads too low; the overexposure protection warns of that. In the simulator the
measurement matches the truth (photons reaching the camera × quantum efficiency) to a few
per cent at any EM gain, exposure, pre-amp gain and binning.

The tracker normalises the ions' brightness the same way: it is in photoelectrons/s, so a
change of pre-amp gain or readout speed no longer looks like a fluorescence drop.

## The shutter and the fan

The iXon Ultra has its own shutter, and an external shutter output (Shutter TTL on the I/O
connector) that it controls separately (`SetShutterEx`). Andor's advice (iXon Ultra 897
hardware guide): in normal use permanently open, with 0 ms opening and closing times. If it
must open and close automatically, use 27 ms each. It is for background frames and protection
from light and dust, at most about 2 Hz, so it is not for video. **Close** in the Camera tab
gives dark frames while live video carries on, and **Open** brings the light back. "Open while
acquiring" keeps the sensor dark between acquisitions.

While the internal shutter is closed, ion tracking waits. Dark frames say nothing about the
ions, so there is no "fluorescence lost" and no "ions presumed lost" 30 s later. When the
shutter opens, tracking carries on with the crystal as it was, dark ions included, and any
change shows up as usual. A camera without an internal shutter (the Camera tab says
"internal shutter no") only switches its shutter output, so tracking carries on regardless.

How a change reaches the camera: ioncam first sends it while the camera runs (other Andor
drivers, such as pylablib, do too). Only if the camera refuses that ("acquisition in
progress") does ioncam stop the acquisition, wait until the camera reports it is idle, apply
the setting, give it a moment, and restart live video or re-arm the readout. If the restart
fails twice, ioncam says so and leaves the camera stopped instead of seemingly live. For the
shutter, ioncam uses `SetShutterEx` (internal and external shutter) and falls back to
`SetShutter`. If the camera refuses the opening or closing times (a 2016 report of an iXon
Ultra 888 refusing 0 ms), it retries with the camera's minimum (`GetShutterMinTimes`), and
it always uses at least that minimum in the automatic modes. The Camera tab shows what the
camera reports about its shutter (internal shutter fitted, external output set separately,
minimum times) and what it answered last. Every shutter and fan call and its answer is in the
console.

The cooling fan runs full, low or off. Lower it, or switch it off, only against vibration,
and off only for short periods unless the camera is water-cooled: the body warms up and the
camera beeps when it is too hot. The sensor may need a warmer set point, and its temperature
should settle before you measure (transient dark current otherwise).

## Camera characterization for SPAM

**Characterize camera...** (Camera tab, or `start_characterization()` over RPC) measures
how the camera behaves when no light reaches it, at many readout settings. The results
serve two purposes: a model of the camera for simulations of state detection (SPAM), and
the readout settings that reach a target SPAM error in the shortest detection time. It runs
by itself (about half an hour with the defaults), keeps everything on disk as it goes, and
an abort keeps what was measured.

The 17 Oct 2026 run on the lab's iXon Ultra 897, and what it changes in the 29 Sep readout
report (the background traced to clock-induced charge at a 3.3 µs vertical shift; camera
hyper-parameters for the SPAM pmfs), is written up in `docs/yb171_readout_addendum.html`,
with the camera values as JSON in `docs/camera_hyperparameters_20261017.json`.

What it does:

1. It **closes the shutter**. A camera without an internal shutter only switches its
   shutter output, so cover it (lens cap, black cloth) first.
2. It waits for the sensor temperature to settle (`temperature_timeout_s`).
3. It checks that **the sensor is dark**: 20 ms frames against 1 ms ones, pixel by pixel.
   Light across the region (a leak), a spot of light (ions or laser scatter: the shutter
   did not close), or saturated pixels stop the run before anything is measured. A lone
   hot pixel does not. During the run, frames that saturate at EM gain also stop it, to
   protect the EM register. `stop_on_light=False` only reports the light.
4. It **sweeps the readout settings** around the Readout tab's, stage by stage. Most
   exposures are below 10 ms. Every setting gets about 3 million pixel samples on a
   144 × 144 px region centred on the readout AQROI (or on the ions):

   | stage | settings | what they show |
   |---|---|---|
   | `baseline` | the readout's, at the start, every 10 min and at the end | drift of bias, CIC, temperature |
   | `sensor` | the whole sensor | hot pixels, CIC against the row, bias structure |
   | `electronics` | every EM readout speed × pre-amp gain | read noise, bias, serial CIC |
   | `vertical` | every vertical shift speed × clock voltage | parallel CIC (faster clocks and lower voltage: less) |
   | `gain` | EM gains 1 … the camera's maximum | the real EM gain against the set one |
   | `exposure` | 0.1 … 10 ms, and 30 ms … 1 s | dark current (CIC is per frame, dark current grows with time), light leaks |
   | `binning` | 1 × 1 … 8 × 8 and the readout's | super-pixel noise: CIC adds up, read noise does not |
   | `triggered` | software-triggered single shots 0 … 1000 ms apart | Grabber-like shots: first-frame effects, bias after idling |
   | `conventional` | the conventional amplifier's speeds × pre-amp gains | read noise without EM gain |
   | `grid` | every speed × pre-amp × vertical speed × voltage at the readout's and the maximum EM gain | the full map for the ranking |
   | `temperature` | other set points (off by default; e.g. −60, −70, −80 °C) | dark current against temperature |

   With the defaults that is 642 settings, about 28 min, on an iXon Ultra 897, and 378
   settings, about 18 min, on an 888. Each setting is analysed while the next is
   measured.
5. It **restores** the shutter and the set point, and live video or the armed readout
   resumes.

**What every setting gives.** A frame minus a bias map taken from the *other* frames
(not its own, which would make its noise look smaller) gives:

- the bias, its spatial pattern and frame-to-frame common mode, and its drift;
- the read noise in counts and electrons;
- hot pixels (a Poisson test), charge trails, events repeating in the next frame, and
  the first frames after a start;
- the whole **histogram of counts above bias**. The dark model of an EMCCD (Lantz et al.
  2008, appendix A) is fitted to it by maximum likelihood:

  > counts = read noise + EM register output of Poisson(`lam_p`) electrons (parallel CIC
  > and dark current, each Gamma-distributed with mean `g` counts) + Poisson(`lam_s`)
  > serial-CIC electrons born inside the register (a log-uniform share of the gain).

  The fit gives `g` (counts per electron, so the **real EM gain** = `g` × electrons per
  count), `lam_p`, `lam_s` with 1σ errors, and a χ² and tail fractions that check the
  model where SPAM is decided.

At moderate EM gain (`g` only a few times the read noise), serial CIC and a lower gain
explain a dark histogram almost equally well. Such settings (gain error > 5 %) are fitted
again with the gain fixed at the set gain × the real/set ratio that the high-gain settings
measured. The "fit" column says which settings this applies to ("gain calibrated").

**Together**, the settings give: dark current (from the exposures) and its temperature
dependence, a light leak if there is one, the real/set EM gain ratio (an aged EM register
has less gain), how binning adds noise up, stability over the run, first-frame and
triggered-shot effects, and hot pixels.

**SPAM ranking.** For an ion giving `ion_rate` photoelectrons/s, every setting is ranked.
By default `ion_rate` is the brightness measured live, with `roi_fraction` of it in its ROI
of `roi_pixels` super-pixels at the readout binning (from the readout plan). Each setting's
dark model is moved to that binning and detection time; the ion's counts go through the EM
register on top. The ranking gives the best threshold, the error at each detection time
(optionally with stray light and the bright/dark lifetimes under the detection light), and
the **shortest detection time reaching `target_error`** (default 10⁻³). It agrees with
Monte Carlo shots to within their statistics. **Use for readout** sets the selected setting
and that detection time (as exposure) in the Readout tab.
`characterization_results(ion_rate=..., roi_pixels=..., target_error=...)` ranks again for
another ion. Differences smaller than the settings' errors (settings with the same
electronics, for instance) are not significant.

**Files** (`<storage.data_dir>/characterization/<time>_<camera>/`; `README.txt` there
describes every field):

| file | contents |
|---|---|
| `plan.json` | the camera, its capabilities, the options and every setting |
| `points.jsonl` | one line per setting as it is measured: settings asked and applied, temperatures, all numbers, the fit |
| `hist/NNNN.npz` | the histogram of counts above bias (and bias / event maps for some) |
| `summary.json`, `summary.csv` | everything, the global results and the SPAM ranking |
| `camera_profile.json` | the dark model at every setting, for simulations |

**In simulations.** `simulator.camera_profile = ".../camera_profile.json"` gives the
simulated camera the measured bias, read noise, electrons per count, real EM gain, parallel
and serial CIC and dark current (against temperature) of the nearest measured setting, so
everything in ioncam can be tried with your camera's noise. Your own simulations can read
the profile directly: per setting it holds `bias`, `read_noise` (counts), `sensitivity`
(electrons per count), `g`, `lam_p` and `lam_s` (per pixel and frame), and the model above.
`ioncam.analysis.darkframes` has it as code: `dark_model_pmf` (the distribution of a dark
pixel), `predict_spam` (the threshold and errors for a signal) and `scaled_model` (another
binning or exposure).

Options are in `[characterize]` (stages, samples per setting, region, gains, exposures,
binnings, trigger spacings, grid gains, clock voltages, temperatures, and the SPAM
parameters). Fewer stages, fewer samples or `grid_gains = [0]` make it quicker.

Background reading: Andor, *Minimizing Clock Induced Charge*, and the iXon Ultra user
guide (faster vertical shifts and lower clock voltages give less CIC; CIC comes once per
frame, dark current with the exposure); Lantz et al. 2008
(arXiv:0803.1420, the EMCCD dark model with serial CIC); Harpsøe et al. 2012
(arXiv:1111.2066, fitting EMCCD bias, read noise, gain and CIC from dark frames); Burrell
et al. 2010 (arXiv:0906.3304, EMCCD qubit readout: CIC sets the dark-state background).

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

### Stray light taken for ions

Light that is not an ion but looks like one - a reflection off an electrode, dust near
focus, a bright defect - is found as an "ion" on the full sensor. When the "crystal" is
spread far apart (an "ion" more than 30 px, or 15 PSF σ, from the next) ioncam says so
once (`stray_light_hint`). What helps, best first:

1. **Learn the background** (Ions tab), best with the repumper off: the ions go dark,
   the laser still scatters, so everything static that remains is learned and
   subtracted.
2. **Search only where the ions are**: *search* in the bar above the image (or the Ions
   tab, or the Ions menu): **only in the visible area** (zoom to the ions first), **only
   around the ions**, or **everywhere**. The region is drawn on the image (amber); live
   tracking, Locate and Calibrate + arm look for ions only inside it, and the crystal is
   tracked afresh when it changes. (`set_search_region()` over RPC.)
3. **Raise the detection threshold** (Ions tab, `detection.threshold_sigma`, default
   6 σ): a bright ion is far above it (SNR of a few hundred), faint stray light is not.

**Next to the crystal** stray light is told from ions by how it shines: every ion of a
crystal is lit alike, so a source at less than `tracker.min_ion_fraction` (0.6) of the
crystal's other ions - judged on the frames where it is lit, so a blinking ion stays an
ion - is not one. A spot found together with the crystal is dropped within a few seconds
(`crystal_refined`: "2 bright ions (not 3): the spot at (x, y) shines far less than the
crystal's ions ..."), and a spot that faint never becomes a new ion. An extra source that
only the averaged frames show is no hidden ion either when it sits closer to another ion
than this trap lets ions come (0.6 × the closest spacing of a chain that many ions long,
at the tracked length scale): that is light the PSF model leaves next to an ion. Neither
comes back at the same place until the background is learned or cleared. Before, a
spot of half an ion's light beside two ions made "3 bright ions", or flipped the count
between 2 and 3 every few seconds. For long chains in a narrow beam, whose end ions are
dimmer, lower `min_ion_fraction`.

Cropping the camera works too: Locate and Calibrate + arm search for ions only inside the
live camera ROI. The long exposure still takes at least 64 × 64 px for its background
statistics, but nothing found in that padding counts. On the 888's 1024 × 1024 sensor a
search region or a crop also makes the live analysis faster (a full frame takes
~100 ms).

## The simulator

`ioncam_gui --simulate` / `aqctl_ioncam --simulate` run on a simulated trap. The
*Simulation* tab (or `sim_set_params()` over RPC) sets: surface preset (`clean`,
`typical`, `dirty`, `horrendous`) and a new random surface, ion fluorescence (photons/s
reaching the camera, default 13 000), laser power, PSF core and halo, chain length scale,
uniform stray light, laser scatter (rate at the beam centre, widths along / across the
beam, angle, offset), defects (number, brightness, height above the surface), scratches,
electrode-gap scatter, collection NA and ion height, and the **trap rotation** in the
image (0 / 45 / 90° buttons, or any angle: chain, electrodes, surface and laser beam
turn together about the trap centre - like mounting a different trap, so a learned
background is cleared, the crystal is tracked afresh and a readout plan is outdated);
plus ions (load, lose, dark, melt), lasers,
repumper, oven, camera moves and external triggers. The EMCCD model has
Poisson photo-electrons, clock-induced charge, the gamma-distributed EM register, read
noise, bias and 16-bit saturation. `simulator.camera_model` (or `--sim-camera`) picks the
camera: `897` (512 × 512 px of 16 µm; 17, 10, 5, 1 MHz) or `888` (1024 × 1024 px of
13 µm; 30, 20, 10, 1 MHz) - sensor, readout speeds and gains; the emulated SDK
(`backend = "fake_sdk"`) follows it.

Readout speeds a camera does not have are taken by ratio: the 5 MHz of the default locate
and readout settings is 10 MHz on an 888 (not 1 MHz, ten times slower). The pre-amp gain
is by default the highest the camera offers at that speed (`preamp_gain_index = -1`:
Gain 3 on an 897, Gain 2 on an 888). The plan's `applied` settings show what the camera
uses.

## Troubleshooting

**The status bar says *RPC failed* or *RPC stopped*.** The GUI and the camera work, but
ARTIQ cannot connect: the RPC server on port 3270 is not running. Its tooltip, a dialog at
start-up and the console say why. Usually either:
- sipyco is not installed in the Python that runs ioncam. Check with
  `python -c "import sipyco.pc_rpc"`, and install it as in step 1 above.
- Another program listens on the port: a second `ioncam_gui`, or an `aqctl_ioncam`
  started by `artiq_ctlmgr` from the `command` of the device_db entry. Either let
  `artiq_ctlmgr` run `aqctl_ioncam` and watch it with `ioncam_gui --host localhost`, or
  drop the `command` and run `ioncam_gui` yourself. `-p` picks another port.

A `terminate` request (from `artiq_ctlmgr`) closes `ioncam_gui` like any ARTIQ controller.

**iXon Ultra 888: the long exposure, Verify and the armed readout look like nonsense,
while live video is fine.** At 0.6 µs the 888 does not shift the charge unless the
vertical clock voltage is raised (Andor's iXon Ultra 888 hardware guide). Older versions
matched the 0.5 µs of an 897 configuration to the nearest 888 speed, 0.6 µs, so those
images were garbled and the plan built from the long exposure put the AQROI anywhere.
Now:

- The camera settings of the long exposure and of the readout are in the Readout tab,
  side by side: exposure, EM gain, horizontal readout, vertical shift, vertical clock
  voltage and pre-amp. Every menu lists only this camera's own values.
- What you choose is what the camera uses. **Apply** stores it and re-arms an armed
  readout at once. Locate, Calibrate + arm, Verify and Arm apply pending changes first.
  The line under Apply says what the camera is running with now, and the plan says what
  it was armed with.
- A speed faster than the camera recommends at normal voltage is shown in brackets, as
  Solis does. It is used as chosen, with a note in the Events tab. If the image looks
  smeared or garbled, raise the vertical clock voltage or choose a slower speed.
- A value the camera does not have (an 897's 0.5 µs or 5 MHz on an 888) is never
  guessed: the tab says which value is used instead, and **Apply** stores it.

**The shutter has no visible effect.** Look at the Shutter group in the Camera tab:
- **"internal shutter no":** only the external shutter output (a TTL on the I/O connector)
  switches, which does something only if an external shutter is wired to it.
- **The "camera:" line:** it says which call the camera accepted (`SetShutterEx` or
  `SetShutter`), whether the times were raised to its minimum, or which errors it gave.

The console has every call with its answer. A camera that controls its internal and external
shutters separately needs `SetShutterEx`; ioncam uses it whenever the camera has it or does
not say.

**A dark ion drifts away, and the chain's length and angle change on their own**
(versions before October 2026). A dark ion's position can only be carried along with the
bright ions. Every 2 s the tracker re-measures the bright ions on averaged frames. It
moved them to the new measurement but left the dark ion where it was.
- When these corrections lean the same way every time, the dark ion drifts steadily in
  one direction. That happens when the whole image drifts slowly (the camera or
  objective warming up, the trap charging), because a 2 s average lags behind the latest
  frame. It also happens when single frames place the ions a little off the averaged
  measurement. In the simulator, a drift of 3 px a minute moved the dark ion 1-2 px a
  minute relative to the bright one.
- Otherwise the corrections are random and the dark ion wanders back and forth.

With three or more bright ions, the dark ion also followed the median motion of the
bright ions from frame to frame, which adds a little more. The chain is fitted to all
the ions, so its length and angle followed the dark ion.

Now a dark ion keeps its place among the bright ions exactly, and the chain is fitted
only to positions measured on averaged frames. With two ions, one of them dark, the
length and angle stay constant, and with every ion bright they vary about three times
less than before.

The same corrections also produced a false "chain moved" event, which marks the readout
plan outdated. The tracker adds up the crystal's moves from frame to frame and reports
one at 3 px, but the corrections were not counted. Now they are, so "chain moved" means
the crystal really moved.

The PSF measured on the same averaged frames also jumped by a few per cent each time
and was handed to the detector every few seconds. It is now averaged.

**Every button fails with "Event loop is closed"** (versions before October 2026). The
RPC server had failed to start, usually because sipyco was missing, and the GUI ran its
actions through the server's event loop. The GUI hid the failure: start-up hung for 10 s,
then the image kept updating but every button failed. The GUI now runs its actions
itself and says why the server is not running, as above.

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
  camera/profile.py     camera_profile.json: a characterized camera's dark model
  analysis/detection.py matched-filter detection, noise maps, cloud/spike rejection
  analysis/fitting.py   joint pixel-integrated Gaussian fit (analytic Jacobian)
  analysis/chain.py     James equilibrium positions, chain fit with hidden ions
  analysis/tracker.py   crystal tracking and event classification
  analysis/readout.py   AQROI/ROI planner, Grabber gateware model, verification, thresholds
  analysis/darkframes.py  EMCCD dark model, its fit, SPAM errors and detection times
  analysis/readout_check.py  the crystal from averaged readout frames (dark, lost, no light)
  characterize.py       camera characterization: plan, sweep, results, files
  engine.py             camera thread, analysis thread, procedures, safety
  api.py                the RPC target (sipyco "ioncam")
  server.py             aqctl_ioncam
  gui/                  ioncam_gui (pyqtgraph; PyQt5/PyQt6/PySide6)
examples/artiq/         device_db entry and experiments
docs/                   the readout addendum (17 Oct 2026) and its camera values
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

`pytest` runs 177 tests in about 6 minutes, in the realistic regime (13k photons/s
per ion, blurred PSF with halo, laser scatter, surface defects):
- chain physics against James' published values;
- detector accuracy (photon-limited) and false-alarm rates at EM gain 1, 50 and 300,
  also with a camera ROI hugging the crystal, clouds vs ions, the learned background on a
  dirty surface with ion-like dust (any ROI and binning, laser power changes);
- the joint core + halo fit of a dense chain (unbiased, honest errors) and its Jacobian;
- every tracker scenario (load via cloud, dark vs lost at every chain position, a dark
  ion that keeps its place for long in a chain of 2 and of 5 (with the chain's length
  and angle), also while the crystal drifts slowly and when single frames place the
  ions off the averaged measurement (no false "chain moved"), loss with
  a dark ion present, melting, lasers off, rigid moves, shuttling with hold, a quiet
  stable crystal), and a tight trap (PSF 1.2 px, two or three ions ~4 px apart): dark
  then bright then lost at every site, the PSF learned from the live crystal, a
  full-sensor start, refocusing, a trap turned by 45° and 90°, the per-ion brightness
  trace and the 80 %-light marker size, the learned PSF averaged (a stray fit ignored,
  a refocus taken); the lab's imaging (PSF 0.8 px + 26 % halo, l = 3.4 px, 45°): a spot
  of stray light beside or between two ions dropped as no ion and never taken for one
  again, blinking ions kept, no hidden ion closer than the trap allows;
- planner invariants, its containment / crosstalk against the simulated image (halo
  included), and agreement with the Grabber gateware model, including its `x1` quirk;
  close ions get overlapping ROIs instead of an error, one centred super-pixel per ion
  with a clean background strip, a fixed AQROI, moving the AQROI by hand, every
  candidate ROI's score against the PSF integrals, an AQROI set by hand (square on the
  automatic plan's grid, the size pad, spots outside left out); locate splitting a merged pair,
  merging an ion the detector split in two, keeping to the tracked crystal and fitting
  scattered spots in windows of their own; the quick fidelity estimate from bright
  shots against EMCCD shots with the ions really dark (alone and next to a bright ion);
- the Andor driver's exact SDK call sequence on the emulated SDK, and an emulated
  iXon Ultra 888 (1024 × 1024, 10 MHz for a 5 MHz setting, a missing pre-amp gain, a
  vertical shift speed used exactly as set and an 897 one never guessed, said once), and
  the readout camera settings enforced (set while armed, calibrate's overrides, the
  settings dialog: one `[readout]`); shutter (SetShutterEx's argument order), fan, and the
  electrons per count asked of the camera in the header's argument order (or configured);
  the shutter and fan as a real camera may answer (refused while acquiring, busy just after
  the abort, 0 ms refused, capabilities unreadable, no SetShutterEx), also during live video
  on an emulated 888, including a restart that fails;
- a full stack over real sipyco RPC with concurrent clients; the brightness in
  photoelectrons/s against the simulated truth at different gain settings, the shutter
  darkening live video and opening again (the tracker waiting meanwhile), the fan; stray
  light taken for ions
  (the hint, a tight crop, a search region, a typed AQROI, the AQROI by hand and
  Calibrate + arm inside it);
- the crystal check while armed: the bias from the read-noise peak, the re-arranged
  crystals, two ions at 45° in a 2 × 2 AQROI (ok, either ion dark, lost, lasers off, ions
  gone; in 1000 and 250 shots, ions bright in every shot or half of them), a three-ion chain
  (any ion dark, one lost, two lost), a PSF 15 % off, a wrong calibration ("unclear"), and
  on the simulator while armed with random qubit states: the events for dark, back, lost
  (the plan outdated), over RPC and in the banner;
- the plan following the crystal on screen: a locate of one ion and a second loaded
  since (a 2 × 2 AQROI by hand holds both, the banner says 2 ions), an ion lost (Arm
  re-centres the AQROI), the crystal moved (Arm moves the AQROI, the Grabber ROIs
  unchanged), nothing changed (the plan armed as it is), the D-pad on an outdated plan,
  and a locate trusted over a tracking that saw no change;
- the camera characterization: fits of synthetic EMCCD dark frames with known read
  noise, gain, CIC and hot pixels (unbiased, honest errors; the gain fixed at moderate
  gain), SPAM predictions against Monte Carlo shots (ROI sums, depumping), detection
  times; the sweep on the simulator (the real/set EM gain, CIC against the clock voltage
  and binning, dark current, the ranking, the files, everything restored, other calls
  refused meanwhile), an abort keeping what was measured, light on the sensor stopping
  it (a hot pixel does not), the emulated Andor SDK (shutter, software triggers), the
  profile driving the simulator, the default plans of both cameras, over RPC and in the
  GUI dialog (Use for readout);
- a headless GUI test (levels set by hand stay put, the D-pad direction through the
  display rotation, ROI labels, the shot histograms, the trap-rotation buttons, the
  overlays one kind at a time, the actions that can run, the event filter, saving the
  image, both themes, an option changed in the settings dialog, New AQROI, the size pad,
  Place by click, the search region, and the camera settings of the long exposure and
  readout: the camera's own choices, Apply, a change made elsewhere shown, Locate applying
  pending changes first; the photoelectrons/s tile and column; the shutter's Close and
  Open; the fan, asked before it goes off), and the GUI without its RPC server (sipyco
  missing, the port in use, a server failing after a slow start, a terminate request)
  and closing it during a long wait.
