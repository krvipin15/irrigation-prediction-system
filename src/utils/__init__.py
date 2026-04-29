"""
Utilities package for ML pipeline support.

This package contains helper modules for:

- Artifact management (models, encoders, pipelines)
- Data availability checks and retrieval (DVC integration)
- External service integrations (MLflow, DagsHub)

Modules
-------
artifact_utils
    Utilities for ensuring availability of required artifacts such as models,
    preprocessing pipelines, and encoders. Includes logic for validation and
    optional downloading (e.g., via DVC or remote storage).

artifact_loader
    Cached loaders for ML artifacts including trained models, preprocessing
    pipelines, and encoders.

Examples
--------
>>> from src.utils import (
...     ensure_model_exists,
...     ensure_file_exists,
...     get_model,
...     get_preprocessor,
...     get_encoder,
... )

>>> ensure_model_exists()
>>> model = get_model()
"""

from src.utils.artifact_loader import get_encoder, get_model, get_preprocessor
from src.utils.artifact_utils import ensure_file_exists, ensure_model_exists

__all__: list[str] = [
    "ensure_model_exists",
    "ensure_file_exists",
    "get_model",
    "get_preprocessor",
    "get_encoder",
]
