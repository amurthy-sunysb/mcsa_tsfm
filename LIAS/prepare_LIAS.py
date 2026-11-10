"""
We have all the functions needed to load the CSV files
and prepare them for feature extraction, either via TSFRESH
or via LIAS, and to label the windows with their motor condition.
"""

import pandas as pd
import os
from tqdm import tqdm

from LIAS.LIAS_constants import TIME_COL, ISA_COL, ISB_COL, ISC_COL, LIAS_DATA_PATH, MOTOR_CONDITIONS_RANGES
from common.windowing import create_overlapping_windows, downsample_data


def process_single_file(filename: str, window_size: int, overlap: int) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes a single CSV file, extracts 'isa', 'isb', and 'isc' signals,
    creates overlapping windows, and downsamples them.

    Args:
        filename (str): The name of the CSV file to process.
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: A tuple containing three lists,
        each corresponding to 'isa', 'isb', and 'isc' signals. Each list
        contains dictionaries with 'data' (downsampled window) and 'metadata'.
    """
    downsample_factors = {512: 1, 1024:2, 2048:4}

    if window_size not in downsample_factors.keys():
        raise ValueError("Invalid window size. Please use one of the supported" \
                         "window sizes: 512, 1024, or 2048.")
    
    df = pd.read_csv(os.path.join(LIAS_DATA_PATH, filename), header=None)
    df = df[[TIME_COL, ISA_COL, ISB_COL, ISC_COL]]

    df.columns = ["t", "isa", "isb", "isc"]
    signal_types = ["isa", "isb", "isc"]
    processed_data_outputs = {signal: [] for signal in signal_types}


    

    for signal_type in signal_types:
        raw_windows = create_overlapping_windows(df[signal_type], window_size=window_size, overlap=overlap)
        for i, window in enumerate(raw_windows):
            metadata = {
                "file_name": filename,
                "window_num": i,
                "window_size": window_size,
                "overlap": overlap,
                "signal_type": signal_type
            }
            processed_data_outputs[signal_type].append({"data": downsample_data(window, downsample_factors[window_size] ), 
                                                        "metadata": metadata})

    return (processed_data_outputs["isa"],
            processed_data_outputs["isb"],
            processed_data_outputs["isc"])


def process_all_LIAS_files(window_size: int, overlap: int) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Processes all 'ccs*.csv' files in a given data path to create overlapping windows and downsample signals.

    Args:
        window_size (int): The size of the overlapping windows.
        overlap (int): The number of samples by which windows overlap.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Three lists containing processed
        windows and metadata for 'isa', 'isb', and 'isc' signals respectively.
    """
    # Get a list of all 'ccs*.csv' files in the data_path, excluding those with '(1)' or '(2)'
    file_list = [f for f in os.listdir(LIAS_DATA_PATH) if f.startswith('ccs') and f.endswith('.csv')]
    file_list.sort() # Sort the list for consistent processing order

    print(f"Found {len(file_list)} files: {file_list}")

    all_processed_windows_a = []
    all_processed_windows_b = []
    all_processed_windows_c = []

    for filename in tqdm(file_list, desc="Processing files"):
        downs_a, downs_b, downs_c = process_single_file(filename, window_size, overlap)
        all_processed_windows_a.extend(downs_a)
        all_processed_windows_b.extend(downs_b)
        all_processed_windows_c.extend(downs_c)

    return all_processed_windows_a, all_processed_windows_b, all_processed_windows_c


def generate_valid_files_list(motor_conditions_ranges: dict) -> list:
    """
    Generates a list of valid file names based on motor condition ranges.

    Args:
        motor_conditions_ranges (dict): A dictionary where keys are condition names
            and values are range objects representing file indices.

    Returns:
        list: A list of file names (e.g., ['ccs0.csv', 'ccs1.csv', ...]).
    """
    valid_files = []
    for r in motor_conditions_ranges.values():
        for i in r:
            valid_files.append(f"ccs{i}.csv")
    return valid_files


def generate_file_to_class_map(motor_conditions_ranges: dict) -> dict:
    """
    Generates a mapping from file names to their corresponding class labels.

    Args:
        motor_conditions_ranges (dict): A dictionary where keys are condition names
            and values are range objects representing file indices.

    Returns:
        dict: A dictionary where keys are file names (e.g., 'ccs0.csv') and values
            are class labels (1 for healthy, 2 for one broken, 3 for two broken).
    """
    file_to_class_map = {}

    # Class 1: Healthy conditions
    for key in ["healthy_low", "healthy_med", "healthy_high"]:
        for i in motor_conditions_ranges[key]:
            file_to_class_map[f"ccs{i}.csv"] = 1

    # Class 2: One broken condition
    for key in ["one_broken_low", "one_broken_med", "one_broken_high"]:
        for i in motor_conditions_ranges[key]:
            file_to_class_map[f"ccs{i}.csv"] = 2

    # Class 3: Two broken conditions
    for key in ["two_broken_low", "two_broken_med", "two_broken_high"]:
        for i in motor_conditions_ranges[key]:
            file_to_class_map[f"ccs{i}.csv"] = 3

    return file_to_class_map


def generate_file_to_group_map(motor_conditions_ranges: dict) -> dict:
    """
    Generates a mapping from file names to their corresponding group names.

    Args:
        motor_conditions_ranges (dict): A dictionary where keys are group names
            and values are range objects representing file indices.

    Returns:
        dict: A dictionary where keys are file names (e.g., 'ccs0.csv') and values
            are group names (e.g., 'healthy_low').
    """
    file_to_group_map = {}
    for group_name, file_range in motor_conditions_ranges.items():
        for i in file_range:
            file_to_group_map[f"ccs{i}.csv"] = group_name

    return file_to_group_map


def add_labels_LIAS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the windows of the LIAS files whose motor condition is known, and labels them.

    The 132 files listed in MOTOR_CONDITIONS_RANGES (LIAS/LIAS_constants.py) are kept; the
    other 30 are not described in the dataset's README. Two columns are added: 'class'
    (1 = healthy, 2 = one broken bar, 3 = two broken bars) and 'file_group' (condition and
    speed, e.g. 'healthy_low'), which train_test_split_LIAS() stratifies on.

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

    valid_files = generate_valid_files_list(MOTOR_CONDITIONS_RANGES)
    file_to_class_map = generate_file_to_class_map(MOTOR_CONDITIONS_RANGES)
    file_to_group_map = generate_file_to_group_map(MOTOR_CONDITIONS_RANGES)

    # Keep only the files whose condition we know, and label their windows
    df_labelled = df[df["file_name"].isin(valid_files)].copy()
    df_labelled["class"] = df_labelled["file_name"].map(file_to_class_map)
    df_labelled["file_group"] = df_labelled["file_name"].map(file_to_group_map)

    print(f"Kept {df_labelled['file_name'].nunique()} labelled files ({len(df_labelled)} windows).")
    return df_labelled


def main():
    # A sample run
    WINDOW_SIZE = 1024
    OVERLAP = int(WINDOW_SIZE/2)

    # Call the new function to process all files
    all_processed_windows_a, all_processed_windows_b, all_processed_windows_c = process_all_LIAS_files(WINDOW_SIZE, OVERLAP)

    print(f"Total processed windows for 'isa': {len(all_processed_windows_a)}")
    print(f"Total processed windows for 'isb': {len(all_processed_windows_b)}")
    print(f"Total processed windows for 'isc': {len(all_processed_windows_c)}")

if __name__ == "__main__":
    main()
