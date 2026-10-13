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
pytest                         # 131 tests (~3 min), all on the simulator / emulated SDK
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
| **Ions** | the crystal at a glance (bright, dark, state, chain scale and angle, PSF σ, noise), every site, tracking (accept as reference, forget, hold, locate), ion detection (where to search, threshold), the learned background, a loading alert |
| **Readout** | state chips (plan valid or outdated, armed, shots, verified); calibrate and run (Arm, Disarm, inside the AQROI set by hand); the camera settings of the long exposure and of the readout side by side (see below); the AQROI by hand (New AQROI, Place by click, the size pad, the D-pad); the results: shot histograms with the fidelity estimate, the plan (and a typed AQROI), the ARTIQ snippet |
| **Events** | the event log, filtered by severity or text; save it to a file |
| **Camera** | cooling, the control lease, configuration (**Settings...**: every option with its help, changed at run time; **Save configuration...**), camera information |
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
| `arm_readout()` / `disarm_readout(resume_live)` | external trigger + Camera Link on/off |
| `get_readout_counts(since_shot)` / `get_readout_stats()` | per-ROI sums of every shot (USB copy), threshold fits |
| `get_readout_histograms(source)` | histograms of every ROI (verification or triggered shots) and, with every shot prepared bright, a threshold and quick fidelity estimate per ion |
| `hold_tracking(reason, timeout)` / `resume_tracking()` | "ions will move on purpose" (shuttling, camera) |
| `take_control(owner, reason, timeout)` / `release_control(owner)` | advisory lease shown in all GUIs |
| `start_live(**settings)` / `set_live(...)` / `set_exposure` / `set_em_gain` / `set_roi` / `stop()` | live video |
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
* The PSF (core + halo) is measured on averaged frames of the live crystal and handed to
  the detector (`detection.auto_psf`), so a wrong `detection.psf_sigma` corrects itself
  without a calibration. After refocusing, a detector still assuming the sharper image
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
(dotted). The Readout tab's **AQROI by hand** box keeps the located positions (or, before
any locate, the tracked ones) and re-computes containment and crosstalk at once:

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
  falls outside the ROIs, you get `readout_geometry_changed`; any crystal change in live
  mode marks the plan **outdated** (`is_readout_valid()` returns False).

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

`pytest` runs 131 tests in about 3 minutes, in the realistic regime (13k photons/s
per ion, blurred PSF with halo, laser scatter, surface defects):
- chain physics against James' published values;
- detector accuracy (photon-limited) and false-alarm rates at EM gain 1, 50 and 300,
  also with a camera ROI hugging the crystal, clouds vs ions, the learned background on a
  dirty surface with ion-like dust (any ROI and binning, laser power changes);
- the joint core + halo fit of a dense chain (unbiased, honest errors) and its Jacobian;
- every tracker scenario (load via cloud, dark vs lost at every chain position, loss with
  a dark ion present, melting, lasers off, rigid moves, shuttling with hold, a quiet
  stable crystal), and a tight trap (PSF 1.2 px, two or three ions ~4 px apart): dark
  then bright then lost at every site, the PSF learned from the live crystal, a
  full-sensor start, refocusing, a trap turned by 45° and 90°, the per-ion brightness
  trace and the 80 %-light marker size;
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
  settings dialog: one `[readout]`);
- a full stack over real sipyco RPC with concurrent clients; stray light taken for ions
  (the hint, a tight crop, a search region, a typed AQROI, the AQROI by hand and
  Calibrate + arm inside it);
- a headless GUI test (levels set by hand stay put, the D-pad direction through the
  display rotation, ROI labels, the shot histograms, the trap-rotation buttons, the
  overlays one kind at a time, the actions that can run, the event filter, saving the
  image, both themes, an option changed in the settings dialog, New AQROI, the size pad,
  Place by click, the search region, and the camera settings of the long exposure and
  readout: the camera's own choices, Apply, a change made elsewhere shown, Locate applying
  pending changes first), and the GUI without its RPC server (sipyco
  missing, the port in use, a server failing after a slow start, a terminate request)
  and closing it during a long wait.
