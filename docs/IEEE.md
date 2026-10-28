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

**3. Run the notebook.** Open `notebooks/IEEE_data_preparation.ipynb` and run all cells. The notebook divides each recording into windows. Then it makes one DataFrame for each phase current: `df_a`, `df_b` and `df_c`. The notebook does not extract features or train a classifier yet.

`MAX_WINDOWS` sets the maximum number of windows from each recording. The default is 200. If you set it to `None`, the notebook uses all windows (1,954 for each recording), but the DataFrames are much larger. With full-length CSV files, the first 200 windows are from before the motor starts. See [Why the pipeline uses less data](#why-the-pipeline-uses-less-data-than-the-dataset-has).

Before you train a classifier, convert at least two rotors. With only one rotor, all recordings have the same class.

## Small test set

The full conversion writes about 2.5 GB of CSV files for each rotor. To check the code quickly on your computer, convert only part of each recording:

```bash
uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat --start 150000 --n-samples 3000
```

- `--start` is the first sample to export. Use 150,000 or more. Earlier samples are from before the motor starts.
- `--n-samples` is the number of samples to export from each recording.

The data stays on your computer. Do not commit it. Git ignores `data/*` and `*.csv`.

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

## Why the pipeline uses less data than the dataset has

The pipeline makes the data smaller in three places. Each one has a different reason.

| Where | What it keeps | Why |
|---|---|---|
| `--n-samples 3000` (small test set) | 3,000 of 1,001,000 samples for each recording | A fast check that the code runs |
| `MAX_WINDOWS = 200` | 200 of 1,954 windows for each recording | The DataFrames must fit in memory |
| Averaging pairs of samples | 512 of 1,024 values for each window | Each window has 512 values, the same as LIAS |

### The small test set: a fast check

The full CSV files are about 2.5 GB for each rotor, or about 12.5 GB for all five rotors. The 3000-sample test set is 40 MB, and it converts in about 15 seconds. You still must download the `.mat` files first. The test set only makes the conversion and the notebook faster.

Use the test set only to make sure that the code runs. It has too little data to detect broken bars. See [Make a small test set](#make-a-small-test-set).

### MAX_WINDOWS: memory

With all 1,954 windows, each DataFrame has 781,600 rows (1,954 × 400 recordings). The sample values alone need about 3.2 GB for each phase current, or 9.6 GB for all three (512 float64 values in each row, 8 bytes each). `windows_to_dataframe` first makes a Python dict for each row, so the notebook needs more memory than this. With 200 windows, each DataFrame has 80,000 rows, and the sample values need about 330 MB.

**Problem: with full-length CSV files, the 200 windows come from before the motor starts.** `max_windows` keeps the first windows of the CSV file. 200 windows use samples 0 – 102,912 (0 – 2.06 s). The motor is off until about sample 100,000. Thus 194 of the 200 windows contain only sensor noise, and the other 6 contain the start of the motor. `prepare_IEEE.py` does not have a start sample.

The small test set does not have this problem, because the convert script exports it from sample 150,000.

If you use full-length files, convert them with `--start 150000` and without `--n-samples`. The CSV files then start in the constant-speed part and continue to the end of the recording.

### Averaging pairs: 512 values for each window

`DOWNSAMPLE_FACTORS` in `IEEE/prepare_IEEE.py` gives 512 values for each window size. It keeps a 512-sample window as it is, averages a 1,024-sample window in pairs, and averages a 2,048-sample window in groups of 4. LIAS uses the same functions, so the windows from both datasets have the same shape. The code does not record why the length is 512.

After the averaging, the sample rate is 25,000 Hz.

### Short windows and broken bars

One window is 1,024 samples. At 50 kHz this is 20.5 ms, about 1.2 supply cycles. A LIAS window has the same number of samples, but it covers 0.72 s, because the LIAS sample rate is 1,429 Hz. Thus the same tsfresh function measures different things on the two datasets. For example, the FFT of an IEEE window gives one value each 48.8 Hz. The FFT of a LIAS window gives one value each 1.4 Hz.

Broken bars cause small peaks at (1 ± 2s) × 60 Hz, where s is the slip. These peaks are a few Hz from the 60 Hz peak. The FFT of one 20.5 ms window cannot separate them from the 60 Hz peak. To separate them, a window must cover some seconds of data.

## Files

| File | Purpose |
|---|---|
| `IEEE/convert_IEEE_mat.py` | Converts `.mat` files to CSV files and `labels.csv` |
| `IEEE/IEEE_constants.py` | Data paths and CSV column numbers |
| `IEEE/prepare_IEEE.py` | Reads the CSV files and `labels.csv`, and makes the windows with `common/windowing.py` |
| `common/windowing.py` | Divides a signal into windows and averages the samples in each window. LIAS uses the same functions. |
| `common/tsfresh_features.py` | Extracts tsfresh features. LIAS uses the same function. |
| `notebooks/IEEE_data_preparation.ipynb` | Makes the windows and the DataFrames |
| `.claude/skills/working-with-ieee-rotor-dataset/` | A Claude Code skill with the same facts, for later sessions |
