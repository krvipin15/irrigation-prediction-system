"""
Model & Artifact Loading Utilities.

This module provides cached access to trained machine learning artifacts,
including the model, preprocessing pipeline, and target encoder. It ensures
efficient reuse of these components across the application without repeated
disk I/O.

Artifacts are loaded from configured paths defined in the application
settings and are cached in memory using `functools.lru_cache` for improved
performance during repeated inference calls.

Features
--------
- Centralized loading of ML artifacts (model, preprocessor, encoder).
- Integration with application settings for dynamic path resolution.
- In-memory caching using `lru_cache` to avoid redundant loads.
- Support for:
  - `skops` for secure model deserialization
  - `joblib` for preprocessing and encoder artifacts

Notes
-----
- Cached instances persist for the lifetime of the process.
- Changes to artifact files require process restart to refresh cache.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import skops.io as sio

from src.configs import Settings, get_settings

# ---------------------------------------------------------------------
# Settings initialization
# ---------------------------------------------------------------------

settings: Settings = get_settings()

# ------------------------------------------------------------------------------
# File paths & constants
# ------------------------------------------------------------------------------

preprocess_filepath: Path = settings.EXP_ARTIFACT_DIR / settings.PREPROCESS_PIPELINE_FILENAME
encoder_path: Path = settings.EXP_ARTIFACT_DIR / settings.TARGET_ENCODER_FILENAME
model_path: Path = settings.MODELS_DIR / settings.MODEL_FILENAME

# ------------------------------------------------------------------------------
# Load models and joblib files in cache
# ------------------------------------------------------------------------------


@lru_cache
def get_model():
    """
    Load and return the trained machine learning model.

    The model is loaded from disk using `skops.io.load`, which provides
    a secure alternative to pickle-based deserialization for sklearn models.

    Returns
    -------
    object
        The loaded machine learning model ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the configured path.
    Exception
        If loading fails due to corruption or incompatible format.

    Notes
    -----
    - The loaded model instance is cached to avoid repeated disk reads.
    - Cache persists for the lifetime of the application process.

    Examples
    --------
    >>> model = get_model()
    >>> predictions = model.predict(X)
    """
    return sio.load(model_path)


@lru_cache
def get_preprocessor():
    """
    Load and return the preprocessing pipeline.

    The preprocessing artifact is typically a serialized scikit-learn
    pipeline (e.g., ColumnTransformer, Pipeline) used to transform raw
    input features into a model-compatible format.

    Returns
    -------
    object
        The loaded preprocessing pipeline.

    Raises
    ------
    FileNotFoundError
        If the preprocessing file is missing.
    Exception
        If loading fails due to incompatible or corrupted artifact.

    Notes
    -----
    - Uses `joblib.load` for efficient deserialization.
    - Cached in memory for reuse across multiple calls.

    Examples
    --------
    >>> preprocessor = get_preprocessor()
    >>> X_transformed = preprocessor.transform(df)
    """
    return joblib.load(preprocess_filepath)


@lru_cache
def get_encoder():
    """
    Load and return the target encoder.

    The encoder is used to convert model output labels (e.g., numeric
    predictions) into human-readable categories via inverse transformation.

    Returns
    -------
    object
        The loaded encoder object.

    Raises
    ------
    FileNotFoundError
        If the encoder file is not found.
    Exception
        If loading fails due to incompatible or corrupted artifact.

    Notes
    -----
    - Typically a scikit-learn encoder (e.g., LabelEncoder, OrdinalEncoder).
    - Cached for efficient reuse during inference.

    Examples
    --------
    >>> encoder = get_encoder()
    >>> labels = encoder.inverse_transform(predictions.reshape(-1, 1))
    """
    return joblib.load(encoder_path)
