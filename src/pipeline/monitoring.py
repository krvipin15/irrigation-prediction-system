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

# Map the column types
schema = DataDefinition(
    # Feature Types
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
# Prefect Task: Monitoring & data drift detection
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
    logger: Logger | LoggerAdapter[Logger] = get_run_logger()
    current_data = current_data.copy()
    reference_data = reference_data.copy()

    # 0. Convert dtypes
    for df in [current_data, reference_data]:
        # Target only the problematic extension dtypes
        int_cols = df.select_dtypes(include=["int8"]).columns
        float_cols = df.select_dtypes(include=["Float32"]).columns
        bool_cols = df.select_dtypes(include=["boolean"]).columns

        df[int_cols] = df[int_cols].astype("int")
        df[float_cols] = df[float_cols].astype("float32")
        df[bool_cols] = df[bool_cols].astype("bool")

    # 1. Create Evidently Datasets to work with
    current_dataset: Dataset = Dataset.from_pandas(current_data, data_definition=schema)
    reference_dataset: Dataset = Dataset.from_pandas(reference_data, data_definition=schema)

    # 2. Build & execute report
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    eval_report: Snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    # 3. Save Artifacts
    html_report_filename: Path = (
        settings.MONITORING_REPORT_DIR / f"{report_suffix}_{settings.EVIDENTLY_HTML_FILENAME}"
    )

    eval_report.save_html(str(html_report_filename))

    logger.info("Report saved locally to %s", settings.MONITORING_REPORT_DIR)

    try:
        # 4. Analyze Results
        metrics_dict = eval_report.dict()

        # Extract the drift flag
        drift_result = metrics_dict.get("metrics", [{}])[0].get("result", {})
        drift_detected = drift_result.get("dataset_drift", False)
        drift_share = drift_result.get("drift_share", 0)

        logger.info(f"Drift Detected: {drift_detected} ({drift_share * 100:.2f}% of features)")

        # 5. Stop if drift detected
        if drift_detected and fail_on_drift:
            error_msg = f"CRITICAL: Data drift detected in {report_suffix} data!"
            logger.error(error_msg)
            raise ValueError(f"Data drift detected above threshold in {report_suffix}!")

    except KeyError as e:
        logger.warning(f"Could not extract drift metrics: {e}. Check HTML report manually.")

    return {
        "status": "success",
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report_paths": {"html": str(html_report_filename)},
    }
