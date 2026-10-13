# IEEE broken rotor bar dataset

The data comes from [*Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor*](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) (Treml et al., 2020, DOI [10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95)).

The dataset has 5 rotors: `rs` (healthy) and `r1b`–`r4b` (1–4 broken bars). Each rotor has 8 load levels (`torque05`–`torque40`) and 10 recordings at each load level, for 400 recordings in total. The pipeline uses only the phase currents `Ia`, `Ib` and `Ic`, sampled at 50 kHz for 20 s.

## How to run it

**1. Download.** Log in with a free IEEE account and download the zip file. Put the five `struct_*.mat` files in `data/`.

**2. Convert.** From the repo root, run:

```bash
uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat
```

This writes one CSV per recording to `data/IEEE/signals/` and the class labels to `data/IEEE/labels.csv`. Class 1 is healthy, class 2 is one broken bar, and so on.

**3. Run the notebook.** Run all cells of `notebooks/IEEE_data_preparation.ipynb`. It produces one DataFrame of windows per phase: `df_a`, `df_b` and `df_c`. `MAX_WINDOWS` (default 200) sets how many windows it keeps from each recording.

## Known problems in the data

- **The motor is off for the first 2 s**, then starts up. The notebook skips everything before sample 160,000 (3.2 s), set by `STEADY_STATE_START` in `IEEE/IEEE_constants.py`.
- **`rs_torque05_09` has no motor current**, only sensor noise. Remove it.

## Small test set

To make small CSVs that you can commit, for checking that the code runs:

```bash
uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat --start 160000 --n-samples 3000
git add -f data/IEEE
```

Each test file covers only 60 ms, which is too short to detect broken bars.
