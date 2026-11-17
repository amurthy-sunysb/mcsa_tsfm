"""
Dataset-agnostic functions for windowing and downsampling time series
before feature extraction, either via TSFRESH or via MOMENT, and for
turning the windows into a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List


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


def windows_to_dataframe(processed_windows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts the windows of one phase into a DataFrame with one row per window.

    Args:
        processed_windows (List[Dict[str, Any]]): The windows of one phase (isa, isb or isc),
            as returned by process_all_LIAS_files(). Each window is a dict with 'data'
            (the downsampled samples) and 'metadata' (file_name, window_num, window_size,
            overlap and signal_type).

    Returns:
        pd.DataFrame: One row per window. The metadata columns come first, followed by one
        column per sample: sample_0, sample_1, ... (sample_511 for 512-sample windows).
    """
    rows = []

    for window in processed_windows:
        # Start the row with the window's metadata (a copy, so the original dict is not changed)
        row = window["metadata"].copy()

        # Add one column per sample
        for i, value in enumerate(window["data"]):
            row[f"sample_{i}"] = value

        rows.append(row)

    return pd.DataFrame(rows)
