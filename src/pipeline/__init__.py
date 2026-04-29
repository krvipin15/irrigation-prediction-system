"""
Pipeline Package.

This package provides a modular, end-to-end machine learning pipeline
built around Prefect for orchestration. It includes components for data
ingestion, validation, preprocessing, inference, and monitoring, enabling
a clean and production-ready workflow.

Modules
-------
ingestion
    Load datasets from various file formats into pandas DataFrames.

validation
    Enforce schema constraints and validate dataset integrity using Pandera.

preprocess
    Apply pre-trained preprocessing transformations to prepare data for modeling.

inference
    Run batch predictions using trained machine learning models and encoders.

monitoring
    Detect data drift and generate analytical reports using Evidently.
"""

from src.pipeline.inference import run_inference
from src.pipeline.ingestion import load_dataset
from src.pipeline.monitoring import generate_evidently_report
from src.pipeline.preprocess import preprocess_dataset
from src.pipeline.validation import validate_dataset

__all__: list[str] = [
    "load_dataset",
    "validate_dataset",
    "preprocess_dataset",
    "run_inference",
    "generate_evidently_report",
]
