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

Usage
-----
Import required utilities directly:

    from src.utils.artifact_utils import download_model, ensure_file_exists
"""
