# mcsa_tsfm
Applying Time Series Foundation Models for Motor Current Signature Analysis

## Datasets

- **IEEE broken rotor bar:** current signals from an induction motor with 0–4 broken rotor bars. See [docs/IEEE.md](docs/IEEE.md) for how to download, convert, and run it.
- **LIAS:** three-phase current signals, in `data/LIAS/ccs*.csv`.

## Data preparation

Both datasets use the same windowing functions in `common/windowing.py`. The windows have the same number of samples, but a LIAS window covers 0.72 s and an IEEE window covers 20.5 ms. See [docs/data_preparation.md](docs/data_preparation.md) for the full comparison.
