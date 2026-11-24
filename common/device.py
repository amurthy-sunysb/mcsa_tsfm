"""
One place to pick the device the models run on. Not tied to a dataset or a model.
"""

import torch


def get_device() -> str:
    """
    Picks the device to run the models on: an NVIDIA GPU if there is one, otherwise an
    Apple GPU, otherwise the CPU.

    Returns:
        str: "cuda", "mps" or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
