"""
Data Preprocessing Pipeline.

This module applies a pre-trained preprocessing artifact to transform
validated datasets into a model-ready format. It is designed as part of a
machine learning pipeline and integrates with Prefect for orchestration,
logging, and execution tracking.

The preprocessing step ensures that input data is converted into the same
feature space used during model training. It supports transformations such
as encoding categorical variables, scaling numerical features, and applying
feature engineering logic encapsulated in a serialized pipeline (e.g.,
scikit-learn pipeline stored via joblib).

Features
--------
- Applies a pre-fitted preprocessing pipeline to incoming datasets.
- Ensures consistency between training and inference feature spaces.
- Automatically handles transformation outputs:
  - Converts NumPy arrays back to pandas DataFrames.
  - Assigns feature names using the preprocessor metadata.
- Saves transformed datasets to disk in efficient Parquet format.
- Prefect task integration with logging and execution control.

Notes
-----
- The preprocessing artifact must be compatible with the input schema.
- Feature names are derived from `get_feature_names_out()` when available.
- Output location is defined via application settings configuration.
- This step assumes prior validation has already been completed.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from prefect import get_run_logger, task

from src.configs import Settings, get_settings

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------
# Settings Initialization
# ---------------------------------------------------------------------

settings: Settings = get_settings()
preprocessed_data_filepath: Path = settings.PREPROCESSED_DATA_DIR / settings.PREPROCESSED_DATA_FILENAME

# ---------------------------------------------------------------------------------
# Prefect Task: Encoding the dataset
# ---------------------------------------------------------------------------------


@task(
    name="Preprocess the Dataset",
    description="Preprocess the dataset before inference the model.",
    tags=["preprocess", "pipeline"],
    timeout_seconds=300,
    log_prints=True,
)
def preprocess_dataset(df: pd.DataFrame, loaded_preprocessor) -> pd.DataFrame:
    """
    Apply a pre-trained preprocessing pipeline to a dataset.

    This function applies serialized preprocessing artifact (e.g.,
    a scikit-learn pipeline) to the input DataFrame. The transformation
    ensures compatibility with the trained model's expected feature space.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to preprocess. Typically validated prior to this step.

    loaded_preprocessor
        Loaded serialized preprocessing artifact ready to apply on dataset.

    Returns
    -------
    pandas.DataFrame
        Transformed dataset. If the preprocessor returns a NumPy array, it is
        converted back into a DataFrame with feature names.

    Raises
    ------
    FileNotFoundError
        If the preprocessing artifact file does not exist.
    Exception
        Propagates errors raised during artifact loading or transformation.

    Examples
    --------
    >>> transformed_df = preprocess_dataset(df, preprocessed_joblib)

    >>> # Output ready for model inference
    >>> transformed_df.shape
    """
    logger: Logger | LoggerAdapter[Logger] = get_run_logger()
    logger.info("Starting data transformation...")

    # --- 1. Apply transformation ---
    transformed_df = loaded_preprocessor.transform(df)

    # If the preprocessor returns a numpy array, convert it back to DataFrame
    if isinstance(transformed_df, np.ndarray):
        transformed_df = pd.DataFrame(transformed_df, columns=loaded_preprocessor.get_feature_names_out())

    # --- 2. Save processed data ---
    transformed_df.to_parquet(preprocessed_data_filepath)
    logger.info("Preprocessing is complete and data is saved at %s", preprocessed_data_filepath)

    return transformed_df
