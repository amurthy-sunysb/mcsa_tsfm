"""
The MOMENT feature pipeline in one call, for either dataset: from the CSV files to the
labelled table that goes into the train-test split.

    windows -> MOMENT embeddings (per phase) -> per-phase tables -> rename_and_merge -> add_labels_<dataset>
"""

import pandas as pd
import torch
from momentfm import MOMENTPipeline

from common.feature_merging import rename_and_merge
from common.moment_embeddings import get_embeddings_in_batches, embeddings_to_dataframe
from IEEE.prepare_IEEE import process_all_IEEE_files, add_labels_IEEE
from LIAS.prepare_LIAS import process_all_LIAS_files, add_labels_LIAS

DATASETS = ("LIAS", "IEEE")


def build_moment_table(dataset: str,
                       model: MOMENTPipeline,
                       device: torch.device | str,
                       window_size: int = 1024,
                       overlap: int = 512,
                       batch_size: int = 128,
                       max_windows: int | None = None) -> pd.DataFrame:
    """
    Builds the labelled MOMENT feature table of a dataset, ready for the train-test split.

    Args:
        dataset (str): "LIAS" or "IEEE".
        model (MOMENTPipeline): MOMENT loaded in 'embedding' mode, already on `device`.
        device (torch.device | str): Device the model is on, e.g. "cuda", "mps" or "cpu".
        window_size (int): The size of the overlapping windows: 512, 1024 or 2048.
        overlap (int): The number of samples by which windows overlap.
        batch_size (int): Number of windows sent to the model at once. Lower it if the
            GPU runs out of memory.
        max_windows (int | None): IEEE only. Keep only the first this-many windows per
            signal per recording (None keeps all).

    Returns:
        pd.DataFrame: One row per labelled window: file_name, window_num, the features
        moment_a_0 ... moment_c_1023, then 'class' and 'file_group'.

    Raises:
        ValueError: If dataset is not supported, or max_windows is given for LIAS.
    """
    if dataset == "LIAS":
        if max_windows is not None:
            raise ValueError("max_windows is only supported for IEEE.")
        windows = process_all_LIAS_files(window_size, overlap)
        add_labels = add_labels_LIAS
    elif dataset == "IEEE":
        windows = process_all_IEEE_files(window_size, overlap, max_windows)
        add_labels = add_labels_IEEE
    else:
        raise ValueError(f"dataset must be one of {DATASETS}, not '{dataset}'.")

    # One table per phase (isa, isb, isc)
    phase_tables = []
    for phase_windows in windows:
        embeddings, metadata = get_embeddings_in_batches(phase_windows, model, device=device,
                                                         batch_size=batch_size)
        phase_tables.append(embeddings_to_dataframe(embeddings, metadata))

    df_merged = rename_and_merge(*phase_tables, feature_type="moment")
    return add_labels(df_merged)
