"""
Model Inference Module with MLflow & DagsHub Integration.

This module handles:
- Fetching the latest registered model from MLflow (via DagsHub)
- Loading preprocessing artifacts (encoder, dataset)
- Running batch inference
- Decoding predictions (if encoder is available)
- Saving predictions to disk
"""

from pathlib import Path
from typing import TYPE_CHECKING

import dagshub
import joblib
import mlflow
import pandas as pd
import skops.io as sio
from mlflow.tracking import MlflowClient
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

encoder_path: Path = settings.EXP_ARTIFACT_DIR / settings.TARGET_ENCODER_FILENAME
preprocessed_data_filepath: Path = settings.PREPROCESSED_DATA_DIR / settings.PREPROCESSED_DATA_FILENAME
output_path: Path = settings.PREDICTIONS_DIR / settings.PREDICTION_FILENAME
model_path = settings.MODELS_DIR / settings.MODEL_FILENAME

# MLflow model registry name
model_name = "best_tree_model"

# ------------------------------------------------------------------------------
# Internal Utility: Download Model from MLflow Registry
# ------------------------------------------------------------------------------


def _download_model() -> Path | None:
    """
    Download the latest version of a registered MLflow model from DagsHub.

    This function connects to the MLflow tracking server hosted via DagsHub,
    retrieves the latest version of the specified model, downloads it, and
    saves it locally using `skops`.

    Returns
    -------
    pathlib.Path or None
        Path to the saved model if successful, otherwise None.

    Raises
    ------
    Exception
        If connection to DagsHub or model retrieval fails.
    """
    logger: Logger | LoggerAdapter = get_run_logger()

    logger.info("Searching for the latest version of model in MLFlow: '%s'...", model_name)

    # --- 1. Initialize DagsHub MLflow tracking ---
    try:
        dagshub.init(
            repo_name=settings.DAGSHUB_REPO_NAME,
            repo_owner=settings.DAGSHUB_REPO_OWNER,
            mlflow=True,
        )
        client = MlflowClient()

    except Exception as e:
        logger.error("Network error: Could not connect to DagsHub. Details: %s", e)
        raise

    #  --- 2. Fetch latest model version from registry ---
    try:
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            logger.warning("No registered versions found for model '%s'.", model_name)
            return None

        # Select highest version number
        latest_version = max(versions, key=lambda x: int(x.version)).version
        model_uri = f"models:/{model_name}/{latest_version}"

        logger.info(f"Found version {latest_version}. Downloading from {model_uri}...")

        # --- 3. Load model via MLflow ---
        model = mlflow.sklearn.load_model(model_uri)

        # Save locally using skops for security & portability
        sio.dump(model, model_path)

        logger.info("Success! Best model saved locally at: %s", model_path)

        return model_path

    except Exception as e:
        logger.error("Failed to retrieve model: %s", e)
        raise


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
def run_inference() -> pd.DataFrame:
    """
    Run batch inference using a trained ML model.

    This task:
    1. Loads a trained model (downloads if missing)
    2. Loads preprocessed input data
    3. Performs predictions
    4. Optionally decodes predictions using a target encoder
    5. Saves prediction results to disk

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

    if not model_path.exists():
        logger.warning("Model not found at %s.", model_path)
        _download_model()

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
