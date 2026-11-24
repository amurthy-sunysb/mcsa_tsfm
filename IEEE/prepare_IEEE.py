"""
We have all the functions needed to load the IEEE broken-rotor-bar CSV files
(produced by convert_IEEE_mat.py) and prepare them for feature extraction.
"""

import os

import pandas as pd
from tqdm import tqdm

from IEEE.IEEE_constants import (IEEE_SIGNALS_PATH, IEEE_LABELS_PATH, SAMPLE_COL, ISA_COL, ISB_COL, ISC_COL,
                                 STEADY_STATE_START)
from common.windowing import create_overlapping_windows, downsample_data

DOWNSAMPLE_FACTORS = {512: 1, 1024: 2, 2048: 4}


def process_single_file(filename: str, window_size: int, overlap: int,
                        max_windows: int | None = None,
                        start_sample: int = STEADY_STATE_START) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes a single IEEE CSV file, extracts 'isa', 'isb', and 'isc' signals,
    creates overlapping windows, and downsamples them.

    Args:
        filename (str): The name of the CSV file to process.
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.
        max_windows (int | None): Keep only the first this-many windows per signal (None keeps all).
        start_sample (int): Skip samples before this one, i.e. the motor-off and startup part.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Three lists for 'isa', 'isb', and 'isc'.
        Each list contains dictionaries with 'data' (downsampled window) and 'metadata'.
    """
    if window_size not in DOWNSAMPLE_FACTORS:
        raise ValueError("Invalid window size. Please use one of the supported "
                         "window sizes: 512, 1024, or 2048.")

    df = pd.read_csv(os.path.join(IEEE_SIGNALS_PATH, filename), header=None,
                     usecols=[SAMPLE_COL, ISA_COL, ISB_COL, ISC_COL])
    df.columns = ["sample", "isa", "isb", "isc"]

    # Column 0 holds the original sample number, so this also works for CSVs exported with --start
    df = df[df["sample"] >= start_sample].drop(
        columns="sample").reset_index(drop=True)
    if len(df) < window_size:
        raise ValueError(f"{filename} has {len(df)} samples from sample {start_sample} on, "
                         f"fewer than one window ({window_size}). Re-export it with --start {start_sample}.")

    # A full steady-state recording (841,000 samples from sample 160,000) has 1,641 windows at 1024/512;
    # only read as many samples as the kept windows need
    if max_windows is not None:
        df = df.iloc[: window_size +
                     (max_windows - 1) * (window_size - overlap)]

    outputs = {}
    for signal_type in df.columns:
        raw_windows = create_overlapping_windows(
            df[signal_type], window_size=window_size, overlap=overlap)
        outputs[signal_type] = [
            {"data": downsample_data(window, DOWNSAMPLE_FACTORS[window_size]),
             # {"data": window,
             "metadata": {"file_name": filename,
                          "window_num": i,
                          "window_size": window_size,
                          "overlap": overlap,
                          "signal_type": signal_type}}
            for i, window in enumerate(raw_windows)
        ]

    return outputs["isa"], outputs["isb"], outputs["isc"]


def process_all_IEEE_files(window_size: int, overlap: int,
                           max_windows: int | None = None,
                           start_sample: int = STEADY_STATE_START) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes every recording listed in labels.csv.

    Args:
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.
        max_windows (int | None): Keep only the first this-many windows per signal per file.
        start_sample (int): Skip samples before this one, i.e. the motor-off and startup part.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Three lists containing processed
        windows and metadata for 'isa', 'isb', and 'isc' signals respectively.
    """
    file_list = sorted(load_IEEE_labels()["file_name"])
    print(f"Found {len(file_list)} files")

    all_a, all_b, all_c = [], [], []
    for filename in tqdm(file_list, desc="Processing files"):
        a, b, c = process_single_file(
            filename, window_size, overlap, max_windows, start_sample)
        all_a.extend(a)
        all_b.extend(b)
        all_c.extend(c)

    return all_a, all_b, all_c


def load_IEEE_labels() -> pd.DataFrame:
    """
    Loads labels.csv.

    Returns:
        pd.DataFrame: One row per recording with file_name, condition, broken_bars,
        load, repetition and class.
    """
    return pd.read_csv(IEEE_LABELS_PATH)


def add_labels_IEEE(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the windows of the IEEE files listed in labels.csv, and labels them.

    The IEEE analogue of add_labels_LIAS(). Two columns are added: 'class'
    (1 = healthy, 2-5 = 1-4 broken bars) and 'file_group' (condition and load,
    e.g. 'r1b_torque05'), which train_test_split_LIAS() stratifies on.

    Args:
        df (pd.DataFrame): One row per window, with a 'file_name' column, e.g. the table
            returned by rename_and_merge().

    Returns:
        pd.DataFrame: A copy of the rows of df that belong to the labelled files, with the
        'class' and 'file_group' columns added.

    Raises:
        ValueError: If df has no 'file_name' column.
    """
    if "file_name" not in df.columns:
        raise ValueError("df has no 'file_name' column.")

    labels = load_IEEE_labels()
    file_to_class_map = dict(zip(labels["file_name"], labels["class"]))
    file_to_group_map = dict(zip(labels["file_name"], labels["condition"] + "_" + labels["load"]))

    # Keep only the files whose condition we know, and label their windows
    df_labelled = df[df["file_name"].isin(file_to_class_map)].copy()
    df_labelled["class"] = df_labelled["file_name"].map(file_to_class_map)
    df_labelled["file_group"] = df_labelled["file_name"].map(file_to_group_map)

    print(f"Kept {df_labelled['file_name'].nunique()} labelled files ({len(df_labelled)} windows).")
    return df_labelled


def main():
    # A sample run
    WINDOW_SIZE = 1024
    OVERLAP = int(WINDOW_SIZE / 2)

    all_a, all_b, all_c = process_all_IEEE_files(
        WINDOW_SIZE, OVERLAP, max_windows=100)

    print(f"Total processed windows for 'isa': {len(all_a)}")
    print(f"Total processed windows for 'isb': {len(all_b)}")
    print(f"Total processed windows for 'isc': {len(all_c)}")


if __name__ == "__main__":
    main()
