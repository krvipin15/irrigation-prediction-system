"""
Model Inference Pipeline.

This module performs batch inference using a trained machine learning model
on preprocessed input data. It is designed as the final stage of the ML
pipeline and integrates with Prefect for orchestration, logging, and
execution tracking.

The inference pipeline loads preprocessed data, applies the trained model
to generate predictions, optionally decodes them into human-readable labels,
and returns the results for downstream consumption.

Features
--------
- Seamless integration with upstream preprocessing outputs.
- Batch prediction using a preloaded trained model.
- Support for decoding predictions via a target encoder.
- Automatic loading of preprocessed datasets from configured storage.
- Structured output generation with unique identifiers.
- Prefect task integration with logging and execution control.

Notes
-----
- Input data is expected to be preprocessed and aligned with model features.
- Predictions are reshaped to meet encoder input requirements before decoding.
- Output IDs are generated sequentially for traceability.
- File paths and naming conventions are controlled via application settings.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from prefect import get_run_logger, task

from src.configs import Settings, get_settings
from src.pipeline.ingestion import load_dataset

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------
# Settings Initialization
# ---------------------------------------------------------------------

settings: Settings = get_settings()

# ---------------------------------------------------------------------------------
# Prefect Task: Model Inference
# ---------------------------------------------------------------------------------


@task(
    name="Model Inference",
    description="Loads a saved model and performs inference on the provided dataset.",
    tags=["inference", "pipeline"],
    timeout_seconds=300,
    log_prints=True,
)
def run_inference(model, encoder, proc_data_path: Path) -> pd.DataFrame:
    """
    Run batch inference using a trained ML model.

    This task:
    1. Loads a trained model
    2. Loads preprocessed input data
    3. Performs predictions
    4. Optionally decodes predictions using a target encoder

    Parameters
    ----------
    model
        Loaded ml model for inference.
    encoder
        Loaded encoder to decode prediction values e.g., 0 -> Low.
    proc_data_path: Path
        Path of preprocessed data file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing original data along with predicted values.

    Raises
    ------
    Exception
        If model loading, inference, or file operations fail.
    """
    logger: Logger | LoggerAdapter = get_run_logger()

    # --- 1. Load preprocessed dataset ---
    logger.info(f"Loading data from {proc_data_path}")
    df = load_dataset(proc_data_path)

    # Extract features
    X = df.copy()

    # --- 2. Perform inference ---
    logger.info(f"Running inference on {len(X)} samples...")
    raw_predictions = model.predict(X)

    # --- 3. Decode predictions ---
    logger.info("Decoding predictions using target_encoder.joblib")

    # Reshape from (N,) to (N, 1) to satisfy 2D requirement
    predictions_2d = raw_predictions.reshape(-1, 1)

    # Inverse transform and then flatten back to 1D so it fits in a DataFrame column
    predictions = encoder.inverse_transform(predictions_2d).flatten()

    # --- 4. Attach predictions to dataset ---
    results_df = df.copy()
    results_df["Irrigation_Need"] = predictions

    logger.info("Inference completed successfully.")

    return results_df
