"""
In this file, we have functions that are needed to create MOMENT-based embeddings
from time series windows of LIAS data.

The windows would usually be produced by the functions in `prepare_LIAS.py`.
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
import torch
from tqdm import tqdm
from momentfm import MOMENTPipeline

def get_embeddings_in_batches(data: List[Dict[str, Any]],
                              model: MOMENTPipeline,
                              device: torch.device,   
                              batch_size: int = 128) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    
    all_embeddings = []
    all_metadata = [] # To store metadata

    # Extract just the time series data for batch processing
    time_series_data_only = [item["data"] for item in data]
    metadata_only = [item["metadata"] for item in data]

    for i in tqdm(range(0, len(time_series_data_only), batch_size), desc="Generating embeddings in batches"):
        batch_ts_data = time_series_data_only[i : i + batch_size]
        batch_meta = metadata_only[i : i + batch_size]

        batch_x = (
            torch.tensor(batch_ts_data)
            .unsqueeze(1).float().to(device))

        batch_mask = torch.ones(
            batch_x.shape[0],
            batch_x.shape[2],
            dtype=torch.long,
            device=device
        )

        # Produce embeddings
        with torch.no_grad():
            output = model(x_enc=batch_x, input_mask=batch_mask)
            all_embeddings.append(output.embeddings.cpu())
            all_metadata.extend(batch_meta) # Collect corresponding metadata

    return torch.cat(all_embeddings, dim=0), all_metadata


def embeddings_to_dataframe(embeddings: torch.Tensor,
                            metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Puts one phase's embeddings and metadata into a single table, one row per window.

    Args:
        embeddings: Tensor of shape (n_windows, 1024) from `get_embeddings_in_batches`.
        metadata: The matching list of metadata dicts.

    Returns:
        A DataFrame with the metadata columns followed by the features 0 to 1023.
    """
    embeddings_df = pd.DataFrame(embeddings.numpy())
    metadata_df = pd.DataFrame(metadata)

    return pd.concat([metadata_df, embeddings_df], axis=1)


def get_moment_model_and_device() -> Tuple[MOMENTPipeline, torch.device]:
    """
    Returns the MOMENT model and the appropriate device (CUDA, MPS, or CPU). This ensures that the model is ready for inference on the available hardware.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode
    )
    model.init()
    model = model.to(device).float()
    return model, device