"""
We have all the functions needed to load the IEEE broken-rotor-bar CSV files
(produced by convert_IEEE_mat.py) and prepare them for feature extraction.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

IEEE_DATA_PATH = "../data/IEEE"
IEEE_SIGNALS_PATH = os.path.join(IEEE_DATA_PATH, "signals")
IEEE_LABELS_PATH = os.path.join(IEEE_DATA_PATH, "labels.csv")
SAMPLE_COL, ISA_COL, ISB_COL, ISC_COL = 0, 1, 2, 3

DOWNSAMPLE_FACTORS = {512: 1, 1024: 2, 2048: 4}


def create_overlapping_windows(x: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
    """
    Creates overlapping windows from a 1-D array.

    Args:
        x (np.ndarray): The input time series.
        window_size (int): The size of each window.
        overlap (int): The number of samples by which consecutive windows overlap.

    Returns:
        np.ndarray: Array of shape (n_windows, window_size).
    """
    step = window_size - overlap
    if step <= 0:
        raise ValueError("Window size must be greater than overlap for a positive step.")
    starts = range(0, len(x) - window_size + 1, step)
    return np.stack([x[i : i + window_size] for i in starts])


def downsample_windows(windows: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsamples each window by averaging groups of 'factor' consecutive samples.

    Args:
        windows (np.ndarray): Array of shape (n_windows, window_size).
        factor (int): The downsampling factor.

    Returns:
        np.ndarray: Array of shape (n_windows, window_size // factor).
    """
    if factor <= 0:
        raise ValueError("Downsampling factor must be a positive integer.")
    n_windows, window_size = windows.shape
    return windows.reshape(n_windows, window_size // factor, factor).mean(axis=2)


def process_single_file(filename: str, window_size: int, overlap: int,
                        max_windows: int | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes a single IEEE CSV file, extracts 'isa', 'isb', and 'isc' signals,
    creates overlapping windows, and downsamples them.

    Args:
        filename (str): The name of the CSV file to process.
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.
        max_windows (int | None): Keep only the first this-many windows per signal (None keeps all).

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Three lists for 'isa', 'isb', and 'isc'.
        Each list contains dictionaries with 'data' (downsampled window) and 'metadata'.
    """
    if window_size not in DOWNSAMPLE_FACTORS:
        raise ValueError("Invalid window size. Please use one of the supported "
                         "window sizes: 512, 1024, or 2048.")

    df = pd.read_csv(os.path.join(IEEE_SIGNALS_PATH, filename), header=None,
                     usecols=[ISA_COL, ISB_COL, ISC_COL])
    df.columns = ["isa", "isb", "isc"]

    outputs = {}
    for signal_type in df.columns:
        windows = create_overlapping_windows(df[signal_type].to_numpy(), window_size, overlap)
        if max_windows is not None:
            windows = windows[:max_windows]
        windows = downsample_windows(windows, DOWNSAMPLE_FACTORS[window_size])

        outputs[signal_type] = [
            {"data": window.tolist(),
             "metadata": {"file_name": filename,
                          "window_num": i,
                          "window_size": window_size,
                          "overlap": overlap,
                          "signal_type": signal_type}}
            for i, window in enumerate(windows)
        ]

    return outputs["isa"], outputs["isb"], outputs["isc"]


def process_all_IEEE_files(window_size: int, overlap: int,
                           max_windows: int | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes every recording listed in labels.csv.

    Args:
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.
        max_windows (int | None): Keep only the first this-many windows per signal per file.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Three lists containing processed
        windows and metadata for 'isa', 'isb', and 'isc' signals respectively.
    """
    file_list = sorted(load_IEEE_labels()["file_name"])
    print(f"Found {len(file_list)} files")

    all_a, all_b, all_c = [], [], []
    for filename in tqdm(file_list, desc="Processing files"):
        a, b, c = process_single_file(filename, window_size, overlap, max_windows)
        all_a.extend(a)
        all_b.extend(b)
        all_c.extend(c)

    return all_a, all_b, all_c


def load_IEEE_labels() -> pd.DataFrame:
    """
    Loads labels.csv and adds a 'file_group' column (condition + load) for stratified splitting.

    Returns:
        pd.DataFrame: One row per recording with file_name, condition, broken_bars,
        load, repetition, class and file_group.
    """
    labels = pd.read_csv(IEEE_LABELS_PATH)
    labels["file_group"] = labels["condition"] + "_" + labels["load"]
    return labels


def main():
    # A sample run
    WINDOW_SIZE = 1024
    OVERLAP = int(WINDOW_SIZE / 2)

    all_a, all_b, all_c = process_all_IEEE_files(WINDOW_SIZE, OVERLAP, max_windows=100)

    print(f"Total processed windows for 'isa': {len(all_a)}")
    print(f"Total processed windows for 'isb': {len(all_b)}")
    print(f"Total processed windows for 'isc': {len(all_c)}")


if __name__ == "__main__":
    main()
