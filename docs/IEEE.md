# IEEE broken rotor bar dataset

Source: [IEEE DataPort](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) (DOI [10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95)).

## How to run

1. Download the zip (free IEEE login) and put the `struct_*.mat` files in `data/`.
2. Convert to CSV:

   ```bash
   uv run python IEEE/convert_IEEE_mat.py data/struct_*.mat --start 150000
   ```

   Add `--n-samples 3000` for a quick test set (code check only, too short to detect broken bars).
3. Run all cells in `notebooks/IEEE_data_preparation.ipynb`.

Convert at least two rotors before training. Don't commit `data/`.

## Files

| File | Purpose |
|---|---|
| `IEEE/convert_IEEE_mat.py` | `.mat` → CSV + `labels.csv` |
| `IEEE/IEEE_constants.py` | Paths and column numbers |
| `IEEE/prepare_IEEE.py` | Loads CSVs and windows them |
| `common/windowing.py` | Windowing and downsampling (shared with LIAS) |
| `common/tsfresh_features.py` | tsfresh features (shared with LIAS) |
| `notebooks/IEEE_data_preparation.ipynb` | Builds windows and DataFrames |
