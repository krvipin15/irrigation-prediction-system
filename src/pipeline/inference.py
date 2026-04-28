"""
Model Inference Module with MLflow & DagsHub Integration.

This module handles:
- Loading preprocessing artifacts (encoder, dataset)
- Running batch inference
- Decoding predictions (if encoder is available)
- Saving predictions to disk
"""

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import skops.io as sio
from prefect import get_run_logger, task

from src.configs.settings import Settings, get_settings
from src.pipeline.ingestion import load_dataset

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# Setup configs
settings: Settings = get_settings()

# ------------------------------------------------------------------------------
# File Paths & Constants
# ------------------------------------------------------------------------------

preprocessed_data_filepath: Path = settings.PREPROCESSED_DATA_DIR / settings.PREPROCESSED_DATA_FILENAME
output_path: Path = settings.PREDICTIONS_DIR / settings.PREDICTION_FILENAME
model_path: Path = settings.MODELS_DIR / settings.MODEL_FILENAME

# ------------------------------------------------------------------------------
# Internal Utility: Download Model from MLflow Registry
# ------------------------------------------------------------------------------


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
def run_inference(encoder_path: Path) -> pd.DataFrame:
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
    encoder_path: Path
        Target encoder joblib file to decode prediction values e.g., 0 -> Low

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

    # --- 1. Ensure model is available locally ---
    logger.info("Loading model from %s", model_path)

    # Load the model using skops
    model = sio.load(model_path)

    # --- 2. Load preprocessed dataset ---
    logger.info(f"Loading data from {preprocessed_data_filepath}")
    df = load_dataset(preprocessed_data_filepath)

    # Extract features
    X = df.copy()

    # --- 3. Perform inference ---
    logger.info(f"Running inference on {len(X)} samples...")
    raw_predictions = model.predict(X)

    # --- 4. Decode predictions (if encoder exists) ---
    if encoder_path.exists():
        logger.info("Decoding predictions using target_encoder.joblib")

        encoder = joblib.load(encoder_path)

        # Reshape from (N,) to (N, 1) to satisfy 2D requirement
        predictions_2d = raw_predictions.reshape(-1, 1)

        # Inverse transform and then flatten back to 1D so it fits in a DataFrame column
        predictions = encoder.inverse_transform(predictions_2d).flatten()
    else:
        logger.warning("Encoder not found! Using raw numeric predictions.")
        predictions = raw_predictions

    # --- 5. Attach predictions to dataset ---
    results_df = df.copy()
    results_df["Irrigation_Need"] = predictions

    logger.info("Inference completed successfully.")

    # --- 6. Format output with custom ID index ---
    results_df.index = range(630000, 630000 + len(results_df))

    # Give the index a name directly, then reset it
    results_df.index.name = "id"
    output_df = results_df.reset_index()[["id", "Irrigation_Need"]]

    # Save predictions to CSV
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    return results_df
