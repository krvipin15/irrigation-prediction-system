"""
Monitoring Module for Data Drift Detection using Evidently.

This module defines a Prefect task that evaluates data drift between
reference and current datasets using Evidently's built-in presets.
It generates an HTML report and optionally fails the pipeline if
significant drift is detected.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset, DataSummaryPreset
from prefect import get_run_logger, task

from src.configs.settings import Settings, get_settings

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# Application configs
settings: Settings = get_settings()

# ------------------------------------------------------------------------------
# Schema Definition
# ------------------------------------------------------------------------------

schema = DataDefinition(
    numerical_columns=[
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Temperature_C",
        "Humidity",
        "Rainfall_mm",
        "Sunlight_Hours",
        "Wind_Speed_kmh",
        "Field_Area_hectare",
        "Previous_Irrigation_mm",
    ],
    categorical_columns=[
        "Mulching_Used",
        "Crop_Growth_Stage",
        "Soil_Type",
        "Crop_Type",
        "Season",
        "Irrigation_Type",
        "Water_Source",
        "Region",
    ],
)

# ---------------------------------------------------------------------------------
# Prefect Task: Data drift detection
# ---------------------------------------------------------------------------------


@task(
    name="Evaluating Data Drift",
    description="Evaluating distribution shifts (data drift) in ML inputs and predictions.",
    tags=["monitor", "pipeline"],
    timeout_seconds=300,
    log_prints=True,
)
def generate_evidently_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    report_suffix: str = "preprocessed",
    fail_on_drift: bool = True,
) -> dict[str, object]:
    """
    Generate an Evidently report to detect data drift between datasets.

    This task compares a reference dataset with a current dataset to identify
    distribution shifts (data drift) using Evidently's built-in metrics.
    It produces an HTML report and optionally raises an exception if drift
    exceeds acceptable thresholds.

    Parameters
    ----------
    current_data : pandas.DataFrame
        The dataset representing the latest (production or incoming) data.
    reference_data : pandas.DataFrame
        The baseline dataset used as a reference for comparison.
    report_suffix : str, optional
        Suffix for naming the generated report file, by default "preprocessed".
    fail_on_drift : bool, optional
        If True, raises an exception when drift is detected, by default True.

    Returns
    -------
    dict[str, object]
        A dictionary containing:
        - status : str
            Execution status ("success").
        - drift_detected : bool
            Whether dataset drift was detected.
        - drift_share : float
            Fraction of features showing drift.
        - report_paths : dict
            Paths to generated report artifacts.

    Raises
    ------
    ValueError
        If drift is detected and `fail_on_drift` is True.
    """
    logger: Logger | LoggerAdapter[Logger] = get_run_logger()

    # Create defensive copies to avoid mutating input data
    current_data = current_data.copy()
    reference_data = reference_data.copy()

    # 0: Normalize dtypes for Evidently compatibility
    for df in [current_data, reference_data]:
        int_cols = df.select_dtypes(include=["int8"]).columns
        float_cols = df.select_dtypes(include=["Float32"]).columns
        bool_cols = df.select_dtypes(include=["boolean"]).columns

        df[int_cols] = df[int_cols].astype("int")
        df[float_cols] = df[float_cols].astype("float32")
        df[bool_cols] = df[bool_cols].astype("bool")

    # 1. Convert pandas DataFrames to Evidently Datasets
    current_dataset: Dataset = Dataset.from_pandas(current_data, data_definition=schema)
    reference_dataset: Dataset = Dataset.from_pandas(reference_data, data_definition=schema)

    # 2. Generate Evidently Report
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    eval_report: Snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    # 3. Save HTML Reports
    html_report_filename: Path = (
        settings.MONITORING_REPORT_DIR / f"{report_suffix}_{settings.EVIDENTLY_HTML_FILENAME}"
    )

    eval_report.save_html(str(html_report_filename))
    logger.info("Report saved locally to %s", settings.MONITORING_REPORT_DIR)

    # 4. Analyze Results
    drift_detected = False
    drift_share = 0.0

    try:
        metrics_dict = eval_report.dict()

        # Extract dataset-level drift results
        drift_result = metrics_dict.get("metrics", [{}])[0].get("result", {})
        drift_detected = drift_result.get("dataset_drift", False)
        drift_share = drift_result.get("drift_share", 0)

        logger.info(f"Drift Detected: {drift_detected} ({drift_share * 100:.2f}% of features)")

        # 5. Optionally fail pipeline if drift is detected
        if drift_detected and fail_on_drift:
            error_msg = f"CRITICAL: Data drift detected in {report_suffix} data!"
            logger.error(error_msg)
            raise ValueError(f"Data drift detected above threshold in {report_suffix}!")

    except KeyError as e:
        logger.warning(f"Could not extract drift metrics: {e}. Check HTML report manually.")

    # 6. Return structured result
    return {
        "status": "success",
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report_paths": {"html": str(html_report_filename)},
    }
