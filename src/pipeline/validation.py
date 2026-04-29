"""
Data Validation Pipeline.

This module defines schema-based validation logic for datasets using
Pandera, ensuring data quality and structural consistency before
downstream processing. It is designed to operate within a Prefect-based
pipeline, providing robust validation, detailed error reporting, and
observability through artifacts.

The validation pipeline distinguishes between training and inference
datasets by detecting the presence of the target column
(`Irrigation_Need`). It applies strict schema checks including data types,
value ranges, categorical constraints, and required columns.

Features
--------
- Schema validation using Pandera with strict and coercive rules.
- Separate schemas for training (with target) and inference (without target).
- Comprehensive validation checks including:
  - Numeric ranges (e.g., pH, temperature, humidity)
  - Non-negative and positive constraints
  - Categorical enforcement
- Lazy validation to aggregate all errors in a single run.
- Automatic generation of detailed failure reports in JSON format.
- Prefect artifact integration for visual inspection of validation issues.
- Logging support for traceability and debugging.

Notes
-----
- Validation reports are saved to a configured directory defined in settings.
- The schema is shared across training and inference to ensure feature consistency.
- Missing optional dependencies or incorrect dtypes may trigger validation errors.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from prefect import get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from src.configs import Settings, get_settings

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------
# Settings Initialization
# ---------------------------------------------------------------------

settings: Settings = get_settings()

# ---------------------------------------------------------------------------------
# Schema to verify the columns and index of Pandas DataFrame
# ---------------------------------------------------------------------------------

target_col: str = "Irrigation_Need"

# Define shared features to ensure consistency between Train and Inference
shared_columns: dict[str, object] = {
    "Soil_Type": pa.Column("category"),
    "Soil_pH": pa.Column("float32", pa.Check.in_range(0, 14)),
    "Soil_Moisture": pa.Column("float32", pa.Check.in_range(0, 100)),
    "Organic_Carbon": pa.Column("float32", pa.Check.ge(0)),
    "Electrical_Conductivity": pa.Column("float32", pa.Check.ge(0)),
    "Temperature_C": pa.Column("float32", pa.Check.in_range(-50, 60)),
    "Humidity": pa.Column("float32", pa.Check.in_range(0, 100)),
    "Rainfall_mm": pa.Column("float32", pa.Check.ge(0)),
    "Sunlight_Hours": pa.Column("float32", pa.Check.in_range(0, 24)),
    "Wind_Speed_kmh": pa.Column("float32", pa.Check.ge(0)),
    "Crop_Type": pa.Column("category"),
    "Crop_Growth_Stage": pa.Column("category"),
    "Season": pa.Column("category"),
    "Irrigation_Type": pa.Column("category"),
    "Water_Source": pa.Column("category"),
    "Field_Area_hectare": pa.Column("float32", pa.Check.gt(0)),
    "Mulching_Used": pa.Column("boolean"),
    "Previous_Irrigation_mm": pa.Column("float32", pa.Check.ge(0)),
    "Region": pa.Column("category"),
}

# Training Schema (includes target)
raw_data_schema = pa.DataFrameSchema(
    columns=shared_columns | {target_col: pa.Column("category")},
    index=pa.Index(int),
    strict=True,
    coerce=True,
)

# Inference Schema (no target)
unseen_data_schema = pa.DataFrameSchema(
    columns=shared_columns,
    index=pa.Index(int),
    strict=True,
    coerce=True,
)

# ---------------------------------------------------------------------------------
# Prefect Task: Validating dataset using Pandera
# ---------------------------------------------------------------------------------


@task(
    name="Validate Dataset",
    description="Validate dataset before applying preprocessing.",
    tags=["validate", "pipeline"],
    timeout_seconds=300,
    log_prints=True,
)
def validate_dataset(df: pd.DataFrame, report_filepath: Path) -> pd.DataFrame:
    """
    Validate a dataset against predefined Pandera schemas.

    This function checks whether the input DataFrame conforms to the expected
    schema. If the target column (`Irrigation_Need`) is present, the dataset is
    treated as training data and validated against the full schema. Otherwise,
    it is treated as unseen/inference data and validated against a reduced schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to validate.
    report_filepath: Path
        Path to save the pandera report.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame with enforced schema types.

    Raises
    ------
    pandera.errors.SchemaErrors
        If one or more schema validation checks fail. Includes detailed failure
        cases and aggregated summaries.
    Exception
        Propagates unexpected errors encountered during validation.

    Examples
    --------
    >>> df_valid = validate_dataset(df)

    >>> # For inference dataset (no target column)
    >>> df_valid = validate_dataset(df.drop(columns=["Irrigation_Need"]))
    """
    logger: Logger | LoggerAdapter[Logger] = get_run_logger()
    logger.info("Starting validation for dataframe with %s rows.", df.shape[0])

    try:
        if target_col in df.columns:
            validated_df = raw_data_schema.validate(df, lazy=True)
        else:
            validated_df = unseen_data_schema.validate(df, lazy=True)

    except SchemaErrors as err:
        # --- 1. Generate failure report data ---
        summary: dict = err.failure_cases.groupby(["column", "check"]).size().to_dict()
        summary_serializable: dict = {str(k): v for k, v in summary.items()}

        # Convert failure cases to a list of dicts, ensuring types are JSON compatible
        details: dict = err.failure_cases.assign(index=err.failure_cases["index"].astype(str)).to_dict(
            orient="records"
        )

        failure_report: dict[str, dict | int] = {
            "summary": summary_serializable,
            "details": details,
            "schema_errors_count": len(err.failure_cases),
        }

        try:
            with Path.open(report_filepath, "w") as f:
                json.dump(failure_report, f, indent=4, default=str)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Could not write JSON report to disk: %s", e)

        # --- 2. Prefect Integration: Create a Markdown Artifact ---
        markdown_table = err.failure_cases.head(10).to_markdown()
        create_markdown_artifact(
            key="validation-report",
            markdown=(
                f"## Validation Failed\n\n"
                f"Found **{len(err.failure_cases)}** violations.\n\n"
                f"### Top Failure Cases (Preview)\n\n{markdown_table}\n\n"
                f"Check the logs or `{report_filepath}` for the full report."
            ),
        )

        logger.exception("Validation failed! Report saved to %s", report_filepath)
        raise

    except Exception:
        logger.exception("An unexpected error occurred during validation")
        raise

    else:
        logger.info("Validation successful: Data matches the schema.")
        return validated_df
