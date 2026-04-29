"""
Utilities package for ML pipeline support.

This package contains helper modules for:

- Artifact management (models, encoders, pipelines)
- Data availability checks and retrieval (DVC integration)
- External service integrations (MLflow, DagsHub)

Modules
-------
- artifact_utils : Handles model downloading and file validation.

Design Goals
------------
- Keep pipeline code clean by isolating infrastructure logic.
- Provide reusable, testable utility functions.
- Centralize interactions with external systems.
"""

from src.utils.artifact_utils import ensure_file_exists, ensure_model_exists

__all__: list[str] = [
    "ensure_model_exists",
    "ensure_file_exists",
]
