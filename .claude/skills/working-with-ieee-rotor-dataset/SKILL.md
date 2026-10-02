---
name: working-with-ieee-rotor-dataset
description: Use when loading, converting, subsetting, or exploring the IEEE DataPort broken-rotor-bar induction motor dataset (struct_rs_R1.mat, struct_r1b_R1.mat … struct_r4b_R1.mat) in mcsa_tsfm, or when running convert_IEEE_mat.py, prepare_IEEE.py or notebooks/IEEE_TSFRESH.ipynb, or when IEEE classifier results look like chance.
---

# Working with the IEEE broken-rotor-bar dataset

## Overview
Five MATLAB v7.3 (HDF5) files, one per rotor: healthy plus 1–4 adjacent broken bars. Each has 8 load levels × 10 repetitions = 400 recordings in total. Background and pipeline: `docs/IEEE.md`.

**Core trap: the motor is OFF for the first 2 s of every recording.** "Take the first N samples" yields only sensor noise.

## File structure (verified with h5py)
```
data/struct_r1b_R1.mat            1.4 GB each; open with h5py, not scipy.io
└── 'r1b'                         struct key: rs | r1b | r2b | r3b | r4b
    └── 'torque05' … 'torque40'   0.5 … 4.0 Nm, 8 loads
        └── 'Ia','Ib','Ic','Va','Vb','Vc'                      refs (10,1) → f[ref] = (1, 1001000) float64, 50 kHz, 20.02 s
            'Vib_acpe','Vib_acpi','Vib_axial','Vib_base','Vib_carc','Trigger'
                                                               refs (10,1) → (1, ~153.5k) float64, ~7.7 kHz; length varies per file
```
Read one signal: `f[f['r1b']['torque05']['Ia'][rep, 0]][()].ravel()`

## Timeline of every current recording
| Samples | Time | Content |
|---|---|---|
| 0 – ~100,000 | 0 – 2.0 s | Motor off, noise ~0.007 A RMS |
| ~100,000 – ~115,000 | 2.0 – 2.3 s | Direct-on-line start, inrush current |
| ~115,000 – 1,001,000 | 2.3 – 20 s | Steady state, ~1.0–1.8 A RMS depending on load |

Onset is 95,000–100,000 in all reps checked. **Slice from 150,000 or later for steady-state data.**

## Known bad recording
`rs_torque05_09` (healthy, torque05, repetition 9) contains noise only, at 0.008 A RMS for the whole recording. It is labelled class 1 and adds noise to that class. Check for it with: Ia RMS < 0.1 A.

## Quick reference
| Task | Command / fact |
|---|---|
| Full conversion | `uv run python convert_IEEE_mat.py data/struct_*.mat` (writes ~2.5 GB of CSV per .mat file) |
| Small committable subset | `uv run python convert_IEEE_mat.py data/struct_*.mat --start 150000 --n-samples 3000` (400 files, 40 MB, ~15 s) |
| Output dir | `data/IEEE/` (uppercase; Linux is case-sensitive, and `prepare_IEEE.py` reads `../data/IEEE`) |
| CSV columns | no header: original sample index, Ia, Ib, Ic |
| Committing data | `.gitignore` ignores `data/*` and `*.csv`, so use `git add -f data/IEEE` |
| Running prepare_IEEE outside the notebook | `cd notebooks && PYTHONPATH=.. uv run python ...` (its paths are relative to `notebooks/`) |
| Classes | 1 = healthy (rs), 2–5 = 1–4 broken bars; 80 recordings per class |

## Notebook shapes (WINDOW_SIZE=1024, OVERLAP=512, MAX_WINDOWS=200)
Windows per recording = `(n_samples - 1024) // 512 + 1`, capped at MAX_WINDOWS. Each window is averaged down to 512 samples.

| Stage | 3000-sample subset (measured) | Full recordings (computed, not run) |
|---|---|---|
| Windows per recording | 4 | 1953, capped to 200 |
| Windows per phase (isa/isb/isc) | 1,600 | 80,000 |
| tsfresh features per phase | (1600, 49), then (1600, 46) after drop | (80000, 46) |
| Merged a+b+c | 1600 × 127 | 80000 × 127 |
| Test split (20%, file-level) | 320 windows | 16,000 |

The 3000-sample subset reached 0.33 accuracy with the scaled RBF SVM (chance is 0.20). 60 ms is only ~3.6 supply cycles, far too short to resolve broken-bar sidebands at (1±2s)·60 Hz, which need seconds of data. **The subset is for checking that the pipeline runs, not for judging the model.**

## Common mistakes
- Slicing from sample 0: every class becomes identical noise, and accuracy sits at chance.
- Writing to `data/ieee`: the notebook finds no files.
- Using `scipy.io.loadmat`: it fails on v7.3 files. Use `h5py` and dereference the object refs.
- Assuming the vibration channels align sample-for-sample with the currents: they use a different rate and length.
