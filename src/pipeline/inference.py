"""
Model Inference Pipeline.

This module performs batch inference using a trained machine learning model
on preprocessed input data. It is designed as the final stage of the ML
pipeline and integrates with Prefect for orchestration, logging, and
execution tracking.

The inference pipeline loads preprocessed data, applies the trained model
to generate predictions, optionally decodes them into human-readable labels,
and persists the results for downstream consumption.

Features
--------
- Seamless integration with upstream preprocessing outputs.
- Batch prediction using a preloaded trained model.
- Support for decoding predictions via a target encoder.
- Automatic loading of preprocessed datasets from configured storage.
- Structured output generation with unique identifiers.
- Persistent storage of predictions in CSV format.
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
preprocessed_data_filepath: Path = settings.PREPROCESSED_DATA_DIR / settings.PREPROCESSED_DATA_FILENAME
output_path: Path = settings.PREDICTIONS_DIR / settings.PREDICTION_FILENAME

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
def run_inference(loaded_model, loaded_encoder) -> pd.DataFrame:
    """
    Run batch inference using a trained ML model.

    This task:
    1. Loads a trained model (downloads if missing)
    2. Loads preprocessed input data
    3. Performs predictions
    4. Optionally decodes predictions using a target encoder
    5. Saves prediction results to disk

    Parameters
    ----------
    loaded_model
        Loaded ml model for inference
    loaded_encoder
        Loaded encoder to decode prediction values e.g., 0 -> Low

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
    logger.info(f"Loading data from {preprocessed_data_filepath}")
    df = load_dataset(preprocessed_data_filepath)

    # Extract features
    X = df.copy()

    # --- 2. Perform inference ---
    logger.info(f"Running inference on {len(X)} samples...")
    raw_predictions = loaded_model.predict(X)

    # --- 3. Decode predictions ---
    logger.info("Decoding predictions using target_encoder.joblib")

    # Reshape from (N,) to (N, 1) to satisfy 2D requirement
    predictions_2d = raw_predictions.reshape(-1, 1)

    # Inverse transform and then flatten back to 1D so it fits in a DataFrame column
    predictions = loaded_encoder.inverse_transform(predictions_2d).flatten()

    # --- 4. Attach predictions to dataset ---
    results_df = df.copy()
    results_df["Irrigation_Need"] = predictions

    logger.info("Inference completed successfully.")

    # --- 5. Save predictions ---
    results_df.index = range(630000, 630000 + len(results_df))

    # Give the index a name directly, then reset it
    results_df.index.name = "id"
    output_df = results_df.reset_index()[["id", "Irrigation_Need"]]

    # Save predictions to CSV
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    return results_df
