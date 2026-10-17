# Readout documents

| File | What it is |
|---|---|
| `yb171_readout_addendum.html` | Addendum to the 29 Sep 2026 report "Hyperfine-qubit readout of ¹⁷¹Yb⁺ with an EMCCD": what the 17 Oct dark-frame characterization of the iXon Ultra 897 (s/n 12729) confirms and changes, camera hyper-parameters for the SPAM pmfs, the free-parameter budget, a validation plan and reference tables. Opens in any browser. Also published at https://claude.ai/artifact/YZhnH691KTST3h1SXuRRM9 |
| `camera_hyperparameters_20261017.json` | The addendum's machine-readable camera block (section 5): per vertical-shift speed at 4×4 binning, 5 MHz, pre-amp 3, EM 200, the full-gain CIC, serial CIC, σ, g, B and their provenance |

The numbers come from the characterization run `20261017-113225_DU897-BB` (dark frames,
shutter closed, −70 °C, 677 settings), fitted with ioncam's dark model
(`ioncam/analysis/darkframes.py`). The run's own files (`points.jsonl`,
`camera_profile.json`, `summary.json`, `hist/`) stay with the lab's data; `camera_profile.json`
loads into the simulator with `[simulator] camera_profile`.
