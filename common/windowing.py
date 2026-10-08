"""
Dataset-agnostic functions for windowing and downsampling time series
before feature extraction, either via TSFRESH or via MOMENT.
"""

import pandas as pd
import numpy as np
from typing import List


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
