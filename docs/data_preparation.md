# Data preparation: IEEE and LIAS

This document compares how the pipeline prepares the IEEE data and the LIAS data before feature extraction.

Both datasets go through the same three steps:

1. Read the CSV files.
2. Divide each phase current into windows of 1,024 samples that overlap by 512 samples.
3. Average each pair of samples, so each window has 512 values.

Both datasets use the same functions from `common/windowing.py`, and both give the same output. The differences are in which files they read, how much of each file they use, and how much time one window covers.

For the IEEE dataset, see [IEEE.md](IEEE.md) and [IEEE_dataset.md](IEEE_dataset.md).

## The same in both

| Step | IEEE and LIAS |
|---|---|
| Columns used | Sample index or time, then the three phase currents (columns 0 – 3) |
| Windowing | `create_overlapping_windows`, 1,024 samples, 512 overlap |
| Averaging | `downsample_data`, factor 2 for a 1,024-sample window |
| Output | Three lists (`isa`, `isb`, `isc`). Each item is `{"data": 512 values, "metadata": {file_name, window_num, window_size, overlap, signal_type}}`. |
| DataFrame in the notebook | 5 metadata columns and `sample_0` … `sample_511` |

## Different

| Item | LIAS ([LIAS/prepare_LIAS.py](../LIAS/prepare_LIAS.py)) | IEEE ([IEEE/prepare_IEEE.py](../IEEE/prepare_IEEE.py)) |
|---|---|---|
| How it finds the files | Lists all `ccs*.csv` files in `data/LIAS` (162 files) | Reads the file names from `data/IEEE/labels.csv` (400 recordings) |
| Labels | Not in the prepare step. The notebook adds the labels later. | `load_IEEE_labels()` gives the class and a `file_group` (rotor + load level) for the split |
| Columns in the CSV | 9 columns. It reads all of them, then keeps 4. | 4 columns. It reads only the 3 current columns. |
| Sample rate | 1,429 Hz (one sample each 0.7 ms) | 50,000 Hz |
| Length of one file | 13,347 samples, 9.3 s | 1,001,000 samples, 20 s |
| Windows for each file | 25, all of them | 1,954 available. `max_windows` keeps the first 200 (default). |
| Part of the file it uses | All of it | From sample 160,000 (3.2 s). With 200 windows, samples 160,000 to 262,912. |
| Time in one window | 0.72 s, about 43 supply cycles | 20.5 ms, about 1.2 supply cycles |
| Sample rate after averaging | 714 Hz | 25,000 Hz |

## What this means

### The same window size covers very different lengths of time

The windows have the same number of samples, but a LIAS window is 35 times longer in time than an IEEE window. A LIAS window shows many supply cycles. An IEEE window shows about one supply cycle with high time detail.

Thus the features from the two datasets measure different things, even with the same tsfresh function. For example, the FFT of a LIAS window gives one value each 1.4 Hz. The FFT of an IEEE window gives one value each 48.8 Hz.

### IEEE skips the motor start

The motor is off for the first 2 s of each IEEE recording, then starts up. In one recording (`rs_torque15_10`), the startup lasts until about sample 151,800. Thus the IEEE prepare step skips all samples before sample 160,000 (3.2 s). The value is `STEADY_STATE_START` in `IEEE/IEEE_constants.py`.

The CSV files keep the original sample number in column 0, so the skip also works for the small test set. Export the test set with `--start 160000`. If a CSV has no full window after sample 160,000, the prepare step stops with an error.

LIAS uses each full file and does not skip any samples.

### The labels come from different places

For IEEE, the labels come from `labels.csv`. The convert script writes this file. For LIAS, the notebook makes the labels from the file names after the prepare step.
