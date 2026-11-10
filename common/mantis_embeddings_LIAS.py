"""
In this file, we have functions that are needed to create Mantis-based embeddings
from time series windows of LIAS data.

It mirrors `moment_embeddings_LIAS.py`: same input (one phase's list of windows),
same output (embeddings plus the matching metadata), so the rest of the pipeline
can treat MOMENT and Mantis features the same way.

The windows would usually be produced by the functions in `prepare_LIAS.py`.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from mantis.trainer import MantisTrainer

# Columns that identify a window; used to line the three phases up
KEY_COLUMNS = ["file_name", "window_num"]


def get_embeddings_in_batches(data: List[Dict[str, Any]],
                              model: MantisTrainer,
                              batch_size: int = 256) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Generates Mantis embeddings for one phase's windows.

    Args:
        data: List of windows for a single phase (isa, isb or isc), each a dict with
            "data" (the samples) and "metadata", as returned by `process_single_file`
            or `process_all_LIAS_files`.
        model: A MantisTrainer wrapping a pretrained Mantis network. The device
            (CPU or GPU) is the one the trainer was created with.
        batch_size: Number of windows sent to the model at once. Lower it if the
            GPU runs out of memory.

    Returns:
        A tensor of shape (n_windows, 256) on the CPU, and the list of metadata
        dicts in the same order (row i of the tensor belongs to metadata[i]).
    """
    time_series_data_only = [item["data"] for item in data]
    metadata_only = [item["metadata"] for item in data]

    # (n_windows, 512) -> (n_windows, 1, 512): Mantis wants (samples, channels, length)
    x = np.asarray(time_series_data_only, dtype=np.float32)[:, np.newaxis, :]

    # Mantis splits each window into 32 patches, so the length must be a multiple of 32
    if x.shape[2] % 32 != 0:
        raise ValueError(f"Window length {x.shape[2]} is not a multiple of 32.")

    # transform() batches and turns off gradients itself, so no manual loop as in MOMENT
    embeddings = model.transform(x, batch_size=batch_size, to_numpy=False)

    return embeddings.cpu(), metadata_only


def embeddings_to_dataframe(embeddings: torch.Tensor,
                            metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Puts one phase's embeddings and metadata into a single table.

    Args:
        embeddings: Tensor of shape (n_windows, 256) from `get_embeddings_in_batches`.
        metadata: The matching list of metadata dicts.

    Returns:
        A DataFrame with columns file_name, window_num and the features 0 to 255,
        one row per window. The other metadata columns (window_size, overlap,
        signal_type) are dropped, as in the MOMENT notebook.
    """
    embeddings_df = pd.DataFrame(embeddings.numpy())
    metadata_df = pd.DataFrame(metadata)[KEY_COLUMNS]

    return pd.concat([metadata_df, embeddings_df], axis=1)


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device:", device)

    # A sample run
    WINDOW_SIZE = 1024
    OVERLAP = int(WINDOW_SIZE/2)

    from LIAS.prepare_LIAS import process_all_LIAS_files
    from mantis.architecture import MantisV1

    network = MantisV1(device=device).from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device=device, network=network)

    all_processed_windows_a, all_processed_windows_b, all_processed_windows_c = \
        process_all_LIAS_files(WINDOW_SIZE, OVERLAP)

    print(f"Total processed windows for 'isa': {len(all_processed_windows_a)}")
    print(f"Total processed windows for 'isb': {len(all_processed_windows_b)}")
    print(f"Total processed windows for 'isc': {len(all_processed_windows_c)}")

    batch_size = 256 # You can adjust this based on your GPU memory
    embeddings_a, metadata_a = get_embeddings_in_batches(all_processed_windows_a, model, batch_size=batch_size)
    embeddings_b, metadata_b = get_embeddings_in_batches(all_processed_windows_b, model, batch_size=batch_size)
    embeddings_c, metadata_c = get_embeddings_in_batches(all_processed_windows_c, model, batch_size=batch_size)

    df_a = embeddings_to_dataframe(embeddings_a, metadata_a)
    df_b = embeddings_to_dataframe(embeddings_b, metadata_b)
    df_c = embeddings_to_dataframe(embeddings_c, metadata_c)

    # 2 key columns + 256 feature columns per phase; the phases are merged with rename_and_merge
    print(f"Shape of the per-phase DataFrames: {df_a.shape}, {df_b.shape}, {df_c.shape}")
    print(df_a.head())


if __name__ == "__main__":
    main()
