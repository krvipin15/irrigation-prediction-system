"""
Dataset preprocessing module.

This module loads a pre-trained preprocessing artifact (e.g., sklearn pipeline)
and applies it to incoming datasets. It integrates with Prefect for orchestration,
logging, and observability.

The preprocessing step ensures that raw validated data is transformed into the
feature space expected by the trained model.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from prefect import get_run_logger, task

from src.configs.settings import Settings, get_settings

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# Setup configs
settings: Settings = get_settings()

# Setup data paths
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
def preprocess_dataset(df: pd.DataFrame, preprocess_filepath: Path) -> pd.DataFrame:
    """
    Apply a pre-trained preprocessing pipeline to a dataset.

    This function loads a serialized preprocessing artifact (e.g., a scikit-learn
    pipeline) and applies it to the input DataFrame. The transformation ensures
    compatibility with the trained model's expected feature space.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to preprocess. Typically validated prior to this step.

    preprocess_filepath: Path
        Contains preprocessing as joblib. Ready to load and apply on datatset.

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

    Notes
    -----
    - The preprocessor is expected to implement `transform` and
      `get_feature_names_out`.
    - Commonly used with scikit-learn pipelines or column transformers.
    - Logging is handled via Prefect's `get_run_logger`.

    Examples
    --------
    >>> transformed_df = preprocess_dataset(df)

    >>> # Output ready for model inference
    >>> transformed_df.shape
    """
    logger: Logger | LoggerAdapter[Logger] = get_run_logger()
    logger.info("Loading preprocessor from %s", preprocess_filepath)

    # Load the preprocessor
    preprocessor = joblib.load(preprocess_filepath)
    logger.info("Starting data transformation...")

    # Perform transformation
    transformed_df = preprocessor.transform(df)

    # If the preprocessor returns a numpy array, convert it back to DataFrame
    if isinstance(transformed_df, np.ndarray):
        transformed_df = pd.DataFrame(transformed_df, columns=preprocessor.get_feature_names_out())

    # Save processed data
    transformed_df.to_parquet(preprocessed_data_filepath)
    logger.info("Preprocessing is complete and data is saved at %s", preprocessed_data_filepath)

    return transformed_df
