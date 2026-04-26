"""
End-to-End ML Pipeline Orchestration using Prefect.

This module defines the main pipeline flow that orchestrates:
- Data ingestion (raw and unseen datasets)
- Data validation using schema checks
- Data drift detection (raw and preprocessed stages)
- Data preprocessing
- Model inference

The pipeline ensures data quality and monitors distribution shifts
before performing predictions, enabling robust production workflows.
"""

import sys
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaErrors
from prefect import flow

from src.configs.data_cfg import FALSE_VALUES, NA_VALUES, OPTIMIZED_DTYPES, TRUE_VALUES
from src.configs.settings import Settings, get_settings
from src.pipeline.inference import run_inference
from src.pipeline.ingestion import load_dataset
from src.pipeline.monitoring import generate_evidently_report
from src.pipeline.preprocess import preprocess_dataset
from src.pipeline.validation import validate_dataset

# Setup settings
settings: Settings = get_settings()

# ------------------------------------------------------------------------------
# File Paths
# ------------------------------------------------------------------------------

df: pd.DataFrame | None = None
raw_filepath: Path = settings.RAW_DATA_DIR / settings.RAW_DATA_FILENAME
unseen_filepath: Path = settings.UNSEEN_DATA_DIR / settings.UNSEEN_DATA_FILENAME
processed_raw_filepath: Path = settings.EXPERIMENTS_DATA_DIR / settings.TREE_FILENAME

# ------------------------------------------------------------------------------
# Prefect Flow Definition
# ------------------------------------------------------------------------------


@flow(name="ML Pipeline")
def pipeline() -> None:
    """
    Execute the end-to-end ML pipeline.

    This flow performs the following steps:
    1. Load raw and unseen datasets
    2. Validate unseen data against schema constraints
    3. Detect data drift on raw data
    4. Preprocess unseen data
    5. Detect data drift on processed data
    6. Run model inference and save predictions

    Raises
    ------
    SystemExit
        If validation fails or any pipeline step encounters a critical error.
    """
    try:
        # --- 1. Load the datasets ---
        # Load historical (reference) dataset
        raw_df = load_dataset(
            raw_filepath,
            sep=",",
            header=0,
            index_col="id",
            true_values=TRUE_VALUES,
            false_values=FALSE_VALUES,
            dtype=OPTIMIZED_DTYPES,
            na_values=NA_VALUES,
            keep_default_na=True,
            on_bad_lines="warn",
            float_precision="round_trip",
            skipinitialspace=True,
            encoding="utf-8",
            encoding_errors="replace",
            memory_map=True,
            low_memory=False,
        )

        # Load unseen (incoming/production) dataset
        unseen_df = load_dataset(
            unseen_filepath,
            sep=",",
            header=0,
            index_col="id",
            true_values=TRUE_VALUES,
            false_values=FALSE_VALUES,
            dtype=OPTIMIZED_DTYPES,
            na_values=NA_VALUES,
            keep_default_na=True,
            on_bad_lines="warn",
            float_precision="round_trip",
            skipinitialspace=True,
            encoding="utf-8",
            encoding_errors="replace",
            memory_map=True,
            low_memory=False,
        )

        # Load preprocessed reference dataset (used for drift comparison)
        preprocessed_raw_df = load_dataset(processed_raw_filepath)

        # --- 2. Validate the dataset ---
        unseen_df = validate_dataset(unseen_df)

        # --- 3. Data Drift (Raw/Validated Data) ---
        generate_evidently_report(
            current_data=unseen_df,
            reference_data=raw_df,
            report_suffix="raw",
            fail_on_drift=False,
        )

        # --- 4. Apply preprocessing ---
        preprocessed_unseen_df = preprocess_dataset(unseen_df)

        # --- 5. Data Drift (Processed Data) ---
        generate_evidently_report(
            current_data=preprocessed_unseen_df,
            reference_data=preprocessed_raw_df,
            report_suffix="preprocessed",
            fail_on_drift=True,
        )

        # --- 6. Model Inference ---
        run_inference()

    except SchemaErrors:
        sys.stderr.write("Data validation failed. Check Prefect logs for details.\n")
        sys.exit(1)

    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Pipeline failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    pipeline()
