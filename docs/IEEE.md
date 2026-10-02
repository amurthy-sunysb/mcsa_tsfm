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
uv run python convert_IEEE_mat.py data/struct_*.mat
```

The command writes these files:

- `data/IEEE/signals/<rotor>_<load level>_<recording>.csv`: one file for each recording. The file has no header. The columns are sample index, `Ia`, `Ib`, `Ic`.
- `data/IEEE/labels.csv`: one row for each recording, with its class.

The folder name must be `data/IEEE`, with capital letters. `prepare_IEEE.py` reads from this folder.

Each `.mat` file gives about 2.5 GB of CSV files. You can convert one `.mat` file at a time. The command keeps the labels from earlier conversions.

**3. Run the notebook.** Open `notebooks/IEEE_TSFRESH.ipynb` and run all cells.

`MAX_WINDOWS` sets the maximum number of windows from each recording. The default is 200. If you set it to `None`, the notebook uses all windows (1,953 for each recording), but feature extraction is much slower.

You must convert at least two rotors. With only one rotor, all recordings have the same class and the classifier fails.

## Make a small test set

The full CSV files are too large for git. To make a small set of test files, convert only part of each recording:

```bash
uv run python convert_IEEE_mat.py data/struct_*.mat --start 150000 --n-samples 3000
git add -f data/IEEE
```

- `--start` is the first sample to export. Use 150,000 or more. Earlier samples are from before the motor starts.
- `--n-samples` is the number of samples to export from each recording.
- `git add -f` is necessary because `.gitignore` ignores `data/*` and `*.csv`.

With 3000 samples, the command writes 400 files (40 MB) in about 15 seconds.

Use the small test set only to make sure that the code runs. 3000 samples is only 60 ms of data. Broken bars cause small changes in the current spectrum near 60 Hz. You must have some seconds of data to see these changes. With the small test set, the classifier accuracy is 0.33. A random guess gives 0.20.

## Data shapes in the notebook

The notebook uses windows of 1024 samples. Each window overlaps the next window by 512 samples. The notebook then averages each pair of samples, so each window has 512 values.

Windows for each recording = (samples − 1024) ÷ 512 + 1, rounded down. The notebook keeps no more than `MAX_WINDOWS`.

| Step | Small test set (3000 samples, measured) | Full recordings (calculated, not run) |
|---|---|---|
| Windows for each recording | 4 | 200 (`MAX_WINDOWS`) |
| Windows for each phase current | 1,600 | 80,000 |
| tsfresh features for each phase current | 1,600 × 46 | 80,000 × 46 |
| All three phase currents together | 1,600 × 127 | 80,000 × 127 |
| Test set (20% of recordings) | 320 windows | 16,000 windows |

The notebook puts all windows from one recording in the training set or all of them in the test set, never in both. This stops overlapping windows from leaking between the two sets.

## Files

| File | Purpose |
|---|---|
| `convert_IEEE_mat.py` | Converts `.mat` files to CSV files and `labels.csv` |
| `prepare_IEEE.py` | Divides recordings into windows and averages the samples in each window |
| `tsfresh_features_IEEE.py` | Extracts tsfresh features |
| `notebooks/IEEE_TSFRESH.ipynb` | Trains and evaluates the classifiers |
| `.claude/skills/working-with-ieee-rotor-dataset/` | A Claude Code skill with the same facts, for later sessions |
