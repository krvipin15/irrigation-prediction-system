"""
Utility module for managing ML artifacts and dependencies in the pipeline.

This module provides helper functions to:

- Download the latest registered model from a DagsHub MLflow registry.
- Ensure required data and artifact files are available locally.
- Automatically fetch missing files using DVC.

Key Features
------------
- Integrates with DagsHub + MLflow for model versioning.
- Uses `skops` for secure model serialization.
- Supports Prefect logging for workflow observability.
- Centralized configuration via `Settings`.
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import dagshub
import mlflow
import skops.io as sio
from mlflow.tracking import MlflowClient
from prefect import get_run_logger

from src.configs import Settings, get_settings

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------
# Settings Initialization
# ---------------------------------------------------------------------

settings: Settings = get_settings()

# ------------------------------------------------------------------------------
# File Paths & Constants
# ------------------------------------------------------------------------------

raw_filepath: Path = settings.RAW_DATA_DIR / settings.RAW_DATA_FILENAME
processed_raw_filepath: Path = settings.EXPERIMENTS_DATA_DIR / settings.REF_DATA_FILENAME
preprocess_filepath: Path = settings.EXP_ARTIFACT_DIR / settings.PREPROCESS_PIPELINE_FILENAME
encoder_path: Path = settings.EXP_ARTIFACT_DIR / settings.TARGET_ENCODER_FILENAME

MODEL_NAME: str = "best_tree_model"
MODEL_PATH: Path = settings.MODELS_DIR / settings.MODEL_FILENAME

PATHS_TO_CHECK: list[Path] = [
    raw_filepath,
    processed_raw_filepath,
    preprocess_filepath,
    encoder_path,
]

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def ensure_model_exists(model_name: str = MODEL_NAME, target_path: Path = MODEL_PATH) -> Path | None:
    """
    Download the latest version of a model from DagsHub MLflow registry if not present.

    This function:
    1. Initializes DagsHub MLflow tracking.
    2. Fetches all versions of the given model.
    3. Selects the latest version.
    4. Downloads and saves it locally using `skops`.

    Parameters
    ----------
    model_name : str, optional
        Name of the registered MLflow model, by default ``MODEL_NAME``.
    target_path : Path, optional
        Local path where the model will be saved, by default ``MODEL_PATH``.

    Returns
    -------
    Path | None
        Path to the downloaded model, or ``None`` if no versions exist.

    Raises
    ------
    Exception
        If connection or model retrieval fails.
    """
    logger: Logger | LoggerAdapter = get_run_logger()

    # Return early if model already exists
    if target_path.exists():
        logger.info("Model already present: %s", target_path)
        return target_path

    # --- 1. Initialize DagsHub MLflow tracking ---
    logger.warning("Model missing. Fetching from DagsHub...")
    try:
        dagshub.init(
            repo_name=settings.DAGSHUB_REPO_NAME,
            repo_owner=settings.DAGSHUB_REPO_OWNER,
            mlflow=True,
        )
        client = MlflowClient()

    except Exception as e:
        logger.error("Could not connect to DagsHub. Details: %s", e)
        raise

    #  --- 2. Fetch latest model version from registry ---
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning("No versions found for '%s'", model_name)
            return None

        # Select highest version number
        latest_version = max(versions, key=lambda x: int(x.version)).version
        model_uri = f"models:/{model_name}/{latest_version}"
        logger.info("Loading %s v%s", model_name, latest_version)

        # --- 3. Load model via MLflow ---
        model = mlflow.sklearn.load_model(model_uri)
        sio.dump(model, target_path)
        logger.info("Model saved: %s", target_path)

        return target_path

    except Exception as e:
        logger.error("Failed to retrieve model: %s", e)
        raise


def ensure_file_exists(paths: list[Path] = PATHS_TO_CHECK) -> None:
    """
    Ensure required files exist locally, pulling missing ones via DVC.

    This function:
    - Checks for missing files from a predefined list.
    - Attempts to fetch them using ``dvc pull`` if absent.

    Parameters
    ----------
    paths : list[Path], optional
        List of file paths to verify, by default ``PATHS_TO_CHECK``.

    Raises
    ------
    subprocess.CalledProcessError
        If DVC pull fails.
    FileNotFoundError
        If DVC is not installed or not available in PATH.
    """
    logger: Logger | LoggerAdapter = get_run_logger()

    # --- 1. Check for missing files ---
    missing_files: list[str] = [str(p) for p in paths if not p.exists()]
    if not missing_files:
        logger.info("All artifacts present...")
        return

    # --- 2. Pull files from DVC if missing ---
    logger.warning("Missing %d files. Pulling via DVC...", len(missing_files))

    try:
        subprocess.run(
            ["dvc", "pull"] + missing_files,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully pulled missing files from remote.")

    except subprocess.CalledProcessError as e:
        logger.error("Failed to pull files from DVC: %s", e.stderr or e)
        raise

    except FileNotFoundError:
        logger.error("DVC command not found. Ensure DVC is installed and in your PATH.")
        raise
