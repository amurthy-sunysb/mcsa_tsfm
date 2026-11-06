# mcsa_tsfm
Applying Time Series Foundation Models for Motor Current Signature Analysis

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

This creates `.venv` with the exact versions in `uv.lock`. Select that environment as the Jupyter kernel to run the notebooks in `notebooks/`.

If you set up the environment before the pandas change below, run `uv sync` again so pandas is downgraded to 2.3.x.

### Dependency notes

- **pandas is pinned to `>=2.3,<3`.** `mantis-tsfm` requires `pandas<3.0`, so pandas 3 cannot be installed alongside it. Do not upgrade pandas past 2.x until Mantis supports it.
- **mantis-tsfm** (`>=1.1.0`) provides the Mantis models used in `notebooks/lias_demo_MANTIS.ipynb`. Pretrained weights download from Hugging Face on first use.
- **momentfm** installs from the MOMENT GitHub repo (see `[tool.uv.sources]` in `pyproject.toml`).
- **torch** comes from PyPI. On Windows that build is CPU-only; on Linux it includes CUDA. To use an NVIDIA GPU on Windows, install a CUDA build of torch into `.venv` (see [pytorch.org](https://pytorch.org/get-started/locally/)). The notebooks run on CPU, just more slowly.
