"""
Converts the IEEE broken-rotor-bar .mat files (e.g. struct_r1b_R1.mat) into
per-recording CSV files in the same layout as the LIAS ccs*.csv files, so they
can be windowed with prepare_LIAS.process_all_LIAS_files.

Each .mat file holds one struct (e.g. 'r1b') with one field per load level
('torque05' ... 'torque40'), and each load level holds 10 recordings of every
signal. Only the phase currents Ia, Ib, Ic are exported.

Output:
    <out_dir>/signals/<condition>_<load>_<rep>.csv   no header; columns: sample index, Ia, Ib, Ic
    <out_dir>/labels.csv                             one row per recording, with its class label

Usage:
    python convert_IEEE_mat.py struct_r1b_R1.mat [more .mat files ...] [--out-dir data/IEEE]
                               [--start 150000 --n-samples 3000]
"""

import argparse
import os
import re

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

CURRENT_SIGNALS = ["Ia", "Ib", "Ic"]


def broken_bars_from_condition(condition: str) -> int:
    """'rs' (healthy) -> 0, 'r1b' -> 1, 'r2b' -> 2, ..."""
    match = re.fullmatch(r"r(\d+)b", condition)
    if match:
        return int(match.group(1))
    if condition == "rs":
        return 0
    raise ValueError(f"Unrecognised condition name '{condition}'.")


def convert_mat_file(mat_path: str, signals_dir: str, start: int = 0,
                     n_samples: int | None = None) -> list[dict]:
    """
    Writes one CSV per recording in a single .mat file.

    Args:
        mat_path (str): Path to the v7.3 .mat file.
        signals_dir (str): Directory to write the CSV files into.
        start (int): First sample to export. The motor is off for the first ~2 s (100,000 samples).
        n_samples (int | None): Number of samples to export from 'start' (None exports to the end).

    Returns:
        list[dict]: One label row per recording written.
    """
    label_rows = []

    with h5py.File(mat_path, "r") as f:
        condition = [k for k in f.keys() if not k.startswith("#")][0]
        broken_bars = broken_bars_from_condition(condition)
        root = f[condition]

        for load in tqdm(list(root.keys()), desc=f"Converting {os.path.basename(mat_path)}"):
            group = root[load]
            n_reps = group[CURRENT_SIGNALS[0]].shape[0]

            for rep in range(n_reps):
                stop = None if n_samples is None else start + n_samples
                currents = [f[group[s][rep, 0]][()].ravel()[start:stop] for s in CURRENT_SIGNALS]
                n = min(len(c) for c in currents)
                data = np.column_stack([np.arange(start, start + n)] + [c[:n] for c in currents])

                file_name = f"{condition}_{load}_{rep + 1:02d}.csv"
                np.savetxt(os.path.join(signals_dir, file_name), data,
                           delimiter=",", fmt=["%d", "%.7g", "%.7g", "%.7g"])

                label_rows.append({
                    "file_name": file_name,
                    "condition": condition,
                    "broken_bars": broken_bars,
                    "load": load,
                    "repetition": rep + 1,
                    "class": broken_bars + 1,  # 1 = healthy, matching the LIAS notebook
                    "source_file": os.path.basename(mat_path),
                })

    return label_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mat_files", nargs="+", help="One or more struct_*.mat files.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "IEEE"))
    parser.add_argument("--start", type=int, default=0, help="First sample to export (default 0).")
    parser.add_argument("--n-samples", type=int, default=None, help="Samples to export per recording (default all).")
    args = parser.parse_args()

    signals_dir = os.path.join(args.out_dir, "signals")
    os.makedirs(signals_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, "labels.csv")

    new_rows = []
    for mat_path in args.mat_files:
        new_rows.extend(convert_mat_file(mat_path, signals_dir, args.start, args.n_samples))
    new_labels = pd.DataFrame(new_rows)

    # Keep labels from previously converted files, replacing any that were re-converted
    if os.path.exists(labels_path):
        old_labels = pd.read_csv(labels_path)
        old_labels = old_labels[~old_labels["file_name"].isin(new_labels["file_name"])]
        new_labels = pd.concat([old_labels, new_labels], ignore_index=True)

    new_labels.sort_values("file_name").to_csv(labels_path, index=False)
    print(f"Wrote {len(new_rows)} recordings to {signals_dir}")
    print(f"Labels for {len(new_labels)} recordings in {labels_path}")


if __name__ == "__main__":
    main()
