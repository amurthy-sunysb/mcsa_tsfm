# IEEE broken rotor bar dataset

## About the data

The data comes from [*Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor*](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) on IEEE DataPort (Treml et al., 2020, DOI [10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95)).

The motor is a 1 hp, 4-pole, 60 Hz induction motor. Its rotor has 34 bars. The dataset has data for five rotors:

- one healthy rotor
- four rotors with 1, 2, 3, or 4 adjacent broken bars

Each rotor runs at 8 load levels, from 0.5 N·m to 4.0 N·m in steps of 0.5 N·m. The rated torque is 4.1 N·m. Each load level has 10 recordings.

The total is 5 rotors × 8 load levels × 10 recordings = **400 recordings**.

### Terms

| Term | Meaning |
|---|---|
| Rotor | One of the five test rotors: `rs` (healthy), `r1b`, `r2b`, `r3b`, `r4b` (1–4 broken bars) |
| Load level | The torque on the motor shaft: `torque05` (0.5 N·m) to `torque40` (4.0 N·m) |
| Recording | One 20-second test run of one rotor at one load level |
| Class | The label the classifier predicts: 1 = healthy, 2 = one broken bar, …, 5 = four broken bars |
| Window | A short part of a recording that the notebook gives to the feature extractor |

## What is in the .mat files

The dataset has one `.mat` file for each rotor, for example `struct_r1b_R1.mat`. Each file is about 1.4 GB. The files are MATLAB v7.3 files, which are HDF5 files. Use `h5py` to read them. `scipy.io.loadmat` cannot read them.

Each file has this structure:

```
struct_r1b_R1.mat
└── r1b                          the rotor
    └── torque05 … torque40      the 8 load levels
        └── Ia, Ib, Ic, ...      one entry for each of the 10 recordings
```

| Signal | Sample rate | Samples per recording |
|---|---|---|
| Phase currents `Ia`, `Ib`, `Ic` | 50 kHz | 1,001,000 (20.02 s) |
| Phase voltages `Va`, `Vb`, `Vc` | 50 kHz | 1,001,000 |
| 5 vibration signals (`Vib_*`) and `Trigger` | about 7.7 kHz | about 153,500 (changes from file to file) |

This pipeline uses only the three phase currents.

## Known problems in the data

### The motor is off at the start of each recording

Do not use the first 2 seconds of a recording. Each current recording has three parts:

| Samples | Time | What the motor does |
|---|---|---|
| 0 – 100,000 | 0 – 2.0 s | The motor is off. The signal is sensor noise of about 0.007 A. |
| 100,000 – 115,000 | 2.0 – 2.3 s | The motor starts. The current is much higher than normal. |
| 115,000 – 1,001,000 | 2.3 – 20 s | The motor runs at a constant speed. The current is 1.0 – 1.8 A, by load. |

In some recordings the motor starts at sample 95,000.

The data that is useful is the constant-speed part. If you take a short part of a recording, start at sample 150,000 or later.

### One recording has no motor current

The recording `rs_torque05_09` (healthy rotor, 0.5 N·m, recording 9) has only sensor noise for all 20 seconds. Its current is 0.008 A. It is in the class 1 data. Remove it if you do not want this noise in the classifier.

## How to run it

**1. Download the data.** You must log in with a free IEEE account. Download the zip file (6.73 GB) from the dataset page. Extract it. Put the five `struct_*.mat` files in `data/`. Git ignores everything in `data/`.

**2. Convert the currents to CSV.** Run this command from the repo root:

```bash
uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat
```

The command writes these files:

- `data/IEEE/signals/<rotor>_<load level>_<recording>.csv`: one file for each recording. The file has no header. The columns are sample index, `Ia`, `Ib`, `Ic`.
- `data/IEEE/labels.csv`: one row for each recording, with its class.

The folder name must be `data/IEEE`, with capital letters. `IEEE/prepare_IEEE.py` reads from this folder. The path is in `IEEE/IEEE_constants.py`.

Each `.mat` file gives about 2.5 GB of CSV files. You can convert one `.mat` file at a time. The command keeps the labels from earlier conversions.

**3. Run the notebook.** Open `notebooks/IEEE_TSFRESH.ipynb` and run all cells. The notebook divides each recording into windows. Then it makes one DataFrame for each phase current: `df_a`, `df_b` and `df_c`. The notebook does not extract features or train a classifier yet.

`MAX_WINDOWS` sets the maximum number of windows from each recording. The default is 200. If you set it to `None`, the notebook uses all windows (1,954 for each recording), but the DataFrames are much larger.

Before you train a classifier, convert at least two rotors. With only one rotor, all recordings have the same class.

## Make a small test set

The full CSV files are too large for git. To make a small set of test files, convert only part of each recording:

```bash
uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat --start 150000 --n-samples 3000
git add -f data/IEEE
```

- `--start` is the first sample to export. Use 150,000 or more. Earlier samples are from before the motor starts.
- `--n-samples` is the number of samples to export from each recording.
- `git add -f` is necessary because `.gitignore` ignores `data/*` and `*.csv`.

With 3000 samples, the command writes 400 files (40 MB) in about 15 seconds.

Use the small test set only to make sure that the code runs. 3000 samples is only 60 ms of data. Broken bars cause small changes in the current spectrum near 60 Hz. You must have some seconds of data to see these changes. With the small test set, an earlier version of the notebook (with tsfresh features and an SVM) got a classifier accuracy of 0.33. A random guess gives 0.20.

## Data shapes in the notebook

The notebook uses windows of 1024 samples. Each window overlaps the next window by 512 samples. The notebook then averages each pair of samples, so each window has 512 values.

Windows for each recording = (samples − 1024) ÷ 512 + 1, rounded down. The notebook keeps no more than `MAX_WINDOWS`.

| Step | Small test set (3000 samples) | Full recordings |
|---|---|---|
| Windows for each recording | 4 | 200 (`MAX_WINDOWS`) |
| Windows for each phase current | 1,600 | 80,000 |
| `df_a`, `df_b`, `df_c` | 1,600 × 517 | 80,000 × 517 |

Each DataFrame has 5 metadata columns (`file_name`, `window_num`, `window_size`, `overlap`, `signal_type`) and 512 sample columns (`sample_0` … `sample_511`).

## Files

| File | Purpose |
|---|---|
| `IEEE/convert_IEEE_mat.py` | Converts `.mat` files to CSV files and `labels.csv` |
| `IEEE/IEEE_constants.py` | Data paths and CSV column numbers |
| `IEEE/prepare_IEEE.py` | Reads the CSV files and `labels.csv`, and makes the windows with `common/windowing.py` |
| `common/windowing.py` | Divides a signal into windows and averages the samples in each window. LIAS uses the same functions. |
| `common/tsfresh_features.py` | Extracts tsfresh features. LIAS uses the same function. |
| `notebooks/IEEE_TSFRESH.ipynb` | Makes the windows and the DataFrames |
| `.claude/skills/working-with-ieee-rotor-dataset/` | A Claude Code skill with the same facts, for later sessions |
