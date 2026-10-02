# IEEE broken rotor bar dataset

## About the data

The data comes from [*Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor*](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) on IEEE DataPort (Treml et al., 2020, DOI [10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95)).

The motor is a 1 hp, 4-pole, 60 Hz induction motor. Its rotor has 34 bars. The dataset tests five rotors:

- one healthy rotor
- four rotors with 1, 2, 3, or 4 adjacent broken bars

Each rotor runs at 8 load levels, from 12.5% to 100% of full load. Each load level is recorded 10 times.

The dataset contains one `.mat` file per rotor, for example `struct_r1b_R1.mat` for the rotor with one broken bar. Each file holds:

| Signal | Sampled at |
|---|---|
| Phase currents `Ia`, `Ib`, `Ic` | 50 kHz |
| Phase voltages `Va`, `Vb`, `Vc` | 50 kHz |
| 5 vibration channels and a trigger | 7.6 kHz |

This pipeline uses only the three currents.

## How to run it

**1. Download the data.** Downloading needs a free IEEE account. Put the `.mat` files in `data/IEEE/raw/`. Git ignores everything under `data/`.

**2. Convert to CSV.** Run this from the repo root:

```bash
python convert_IEEE_mat.py data/IEEE/raw/*.mat
```

This writes:

- `data/IEEE/signals/<rotor>_<load>_<rep>.csv`: one file per recording
- `data/IEEE/labels.csv`: the label for each recording. Class 1 is healthy, class 2 is one broken bar, and so on.

Each `.mat` file becomes about 2.5 GB of CSVs and takes about 1.5 minutes to convert. You can convert files one at a time; earlier labels are kept.

**3. Run the notebook.** Open `notebooks/IEEE_TSFRESH.ipynb` and run all cells.

`MAX_WINDOWS` sets how many windows the notebook keeps from each recording. The default is 200. Set it to `None` to use every window, but feature extraction will be much slower.

You need at least two rotors converted. With only one, every recording has the same class and the classifier fails.

## Files

| File | Purpose |
|---|---|
| `convert_IEEE_mat.py` | Converts `.mat` files to CSVs and `labels.csv` |
| `prepare_IEEE.py` | Splits recordings into windows and downsamples them |
| `tsfresh_features_IEEE.py` | Extracts tsfresh features |
| `notebooks/IEEE_TSFRESH.ipynb` | Trains and evaluates the classifiers |
