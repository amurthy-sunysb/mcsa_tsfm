"""
We have all the functions needed to load the CSV files
and prepare them for feature extraction, either via TSFRESH
or via LIAS.
"""

import pandas as pd
import os
import numpy as np
from typing import List
from tqdm import tqdm

from LIAS_constants import TIME_COL, ISA_COL, ISB_COL, ISC_COL, LIAS_DATA_PATH



def create_overlapping_windows(s: pd.Series, window_size: int, overlap: int) -> List[List[float]]:
    """
    Creates overlapping windows from a pandas Series.

    Args:
        s (pd.Series): The input time series as a pandas Series.
        window_size (int): The size of each window.
        overlap (int): The number of samples by which consecutive windows overlap.

    Returns:
        List[List[float]]: A list of lists, where each inner list represents a window.
    """
    step = window_size - overlap
    if step <= 0:
        raise ValueError("Window size must be greater than overlap for a positive step.")
    return [s.iloc[i : i + window_size].tolist()
            for i in range(0, len(s) - window_size + 1, step)]


def downsample_data(data: List[float], factor: int) -> List[float]:
    """
    Downsamples a list of float values by averaging groups of 'factor' elements.

    Args:
        data (List[float]): The input list of numerical data.
        factor (int): The downsampling factor (e.g., 2 for downsampling by 2, 4 for downsampling by 4).

    Returns:
        List[float]: The downsampled data as a list of floats.
    """
    if factor <= 0:
        raise ValueError("Downsampling factor must be a positive integer.")
    data_series = pd.Series(data)
    return data_series.groupby(np.arange(len(data_series)) // factor).mean().tolist()


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
