"""
Application configuration module.

This module defines a centralized configuration system for the irrigation
prediction pipeline. It manages environment variables, file paths, artifact
locations, and integrations with external services such as MLflow, DagsHub,
and Supabase.

The configuration is environment-aware and enforces required fields depending
on the deployment stage (development, staging, production).
"""

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Centralized application configuration.

    This class manages all configuration required by the irrigation prediction
    system, including environment variables, filesystem paths, artifact
    locations, and external service integrations. Values are loaded from
    environment variables and optional `.env` files using Pydantic.

    The configuration is environment-aware and performs validation to ensure
    that all required settings are present for non-test environments.

    Attributes
    ----------
    Core Settings
        PROJECT_NAME : str
            Human-readable project name.
        ENVIRONMENT : str
            Active runtime environment (development, staging, production, test).

    Artifacts & Reporting
        ARTIFACTS_DIR : Path
            Run-specific root directory for current execution outputs.

    Data Directories & Filenames
        RAW_DATA_DIR / RAW_DATA_FILENAME : Path, str
            Directory and filename for the raw input datasets.
        UNSEEN_DATA_DIR / UNSEEN_DATA_FILENAME : Path, str
            Directory and filename for inference-time datasets.
        PREPROCESSED_DATA_DIR / PREPROCESSED_DATA_FILENAME : Path, str
            Directory and filename for the main pipeline's processed data.
        EXPERIMENTS_DATA_DIR : Path
            Directory for specialized experiment datasets (Linear/Tree variants).
        PREDICTIONS_DIR / PREDICTION_FILENAME : Path, str
            Directory and filename for local prediction outputs.

    Models & Transformers
        MODELS_DIR / MODEL_FILENAME : Path, str
            Location and name of the production-ready model artifact.
        EXP_ARTIFACT_DIR : Path
            Directory for experiment-generated artifacts (transformers).
        PREPROCESS_PIPELINE_FILENAME / TARGET_ENCODER_FILENAME : str
            Filenames for specific preprocessing joblib objects.

    Reporting
        VALIDATION_REPORT_DIR / PANDERA_REPORT_FILENAME : Path, str
            Directory and filename for Pandera data validation results.
        MONITORING_REPORT_DIR / EVIDENTLY_HTML_FILENAME : Path, str
            Directory and filename for Evidently monitoring/drift reports.

    Experiment Tracking (MLflow & DagsHub)
        MLFLOW_TRACKING_URI : AnyHttpUrl | None
            MLflow tracking server endpoint.
        MLFLOW_REGISTERED_MODEL_NAME : str
            Name used to identify the model in the MLflow registry.
        DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME : str | None
            DagsHub repository ownership and project identifiers.
        DAGSHUB_TOKEN : SecretStr | None
            Sensitive token for DagsHub authentication.

    Storage & Monitoring (Supabase)
        SUPABASE_URL : AnyHttpUrl | None
            Supabase project endpoint.
        SUPABASE_KEY : SecretStr | None
            Secret key for Supabase database access.
        SUPABASE_PREDICTIONS_TABLE / SUPABASE_REPORTS_TABLE : str
            Database table names for storing inference results and monitoring logs.

    Security & Serving (FastAPI)
        JWT_SECRET_KEY : SecretStr | None
            Secret key used for authentication and token signing.
        JWT_ALGORITHM : str
            Algorithm used for JWT encoding (default: HS256).
        ACCESS_TOKEN_EXPIRE_MINUTES : int
            Duration in minutes before a JWT token expires.
    """

    # Configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    timestamp: str = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------
    # Project and Environment
    # ---------------------------------------------------------

    PROJECT_NAME: str = "Irrigation Prediction System"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production|test)$")

    # ---------------------------------------------------------
    # Centralized Directories & Filenames
    # ---------------------------------------------------------

    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

    # Contains raw dataset with features and target used for training
    RAW_DATA_DIR: Path = ARTIFACTS_DIR / "data" / "raw"
    RAW_DATA_FILENAME: str = "raw_v1.csv"

    # Contains dataset with features only used for testing
    UNSEEN_DATA_DIR: Path = ARTIFACTS_DIR / "data" / "unseen"
    UNSEEN_DATA_FILENAME: str = "unseen_v1.csv"

    # Preprocessed dataset for main pipeline
    PREPROCESSED_DATA_DIR: Path = ARTIFACTS_DIR / "data" / "processed"
    PREPROCESSED_DATA_FILENAME: str = f"processed_{timestamp}.parquet"

    # Preprocessed dataset for experiment notebooks
    EXPERIMENTS_DATA_DIR: Path = ARTIFACTS_DIR / "data" / "experiments"
    LINEAR_FILENAME: str = "linear_data_v1.parquet"
    TREE_FILENAME: str = "tree_data_v1.parquet"

    # Reference data for evidently
    REF_DATA_FILENAME: str = "tree_data_v1.parquet"

    # Predictions
    PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "data" / "predictions"
    PREDICTION_FILENAME: str = f"prediction_{timestamp}.csv"

    # Best model from MLFlow registry
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    MODEL_FILENAME: str = "best_model_v1.skops"

    # Joblib files
    EXP_ARTIFACT_DIR: Path = ARTIFACTS_DIR / "transformers"
    PREPROCESS_PIPELINE_FILENAME: str = "preprocessing_v1.joblib"
    TARGET_ENCODER_FILENAME: str = "target_encoder_v1.joblib"

    # Reports directory
    VALIDATION_REPORT_DIR: Path = ARTIFACTS_DIR / "reports" / "validation"
    PANDERA_REPORT_FILENAME: str = f"report_{timestamp}.json"
    MONITORING_REPORT_DIR: Path = ARTIFACTS_DIR / "reports" / "monitoring"
    EVIDENTLY_HTML_FILENAME: str = f"report_{timestamp}.html"

    @field_validator(
        "ARTIFACTS_DIR",
        "RAW_DATA_DIR",
        "UNSEEN_DATA_DIR",
        "PREPROCESSED_DATA_DIR",
        "EXPERIMENTS_DATA_DIR",
        "PREDICTIONS_DIR",
        "EXP_ARTIFACT_DIR",
        "MODELS_DIR",
        "VALIDATION_REPORT_DIR",
        "MONITORING_REPORT_DIR",
        mode="after",
    )
    @classmethod
    def ensure_dir(cls, v: Path) -> Path:
        """
        Ensure that directory paths exist.

        Parameters
        ----------
        v : pathlib.Path
            Directory path to validate.

        Returns
        -------
        pathlib.Path
            The same path after ensuring it exists.

        Notes
        -----
        - Creates directories recursively if they do not exist.
        - Used automatically by Pydantic during model initialization.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    # ---------------------------------------------------------
    # MLflow and DagsHub (Experimentation & Remote Storage)
    # ---------------------------------------------------------

    MLFLOW_TRACKING_URI: AnyHttpUrl | None = None
    MLFLOW_REGISTERED_MODEL_NAME: str = "model_v1"

    DAGSHUB_REPO_OWNER: str | None = None
    DAGSHUB_REPO_NAME: str | None = None
    DAGSHUB_TOKEN: SecretStr | None = None

    # ---------------------------------------------------------
    # Supabase (Metadata, Predictions, Evidently JSON Reports)
    # ---------------------------------------------------------

    SUPABASE_URL: AnyHttpUrl | None = None
    SUPABASE_KEY: SecretStr | None = None
    SUPABASE_PREDICTIONS_TABLE: str = "preds"
    SUPABASE_REPORTS_TABLE: str = "monitoring_reports"

    # ---------------------------------------------------------
    # FastAPI Security Config (Serving)
    # ---------------------------------------------------------

    JWT_SECRET_KEY: SecretStr | None = None
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        """
        Validate required settings based on environment.

        Returns
        -------
        Self
            The validated settings instance.

        Raises
        ------
        ValueError
            If required configuration fields are missing in non-test environments.

        Notes
        -----
        - Enforces presence of external service credentials in development,
          staging, and production environments.
        - Skips strict validation in test environments.
        """
        if self.ENVIRONMENT in ("development", "staging", "production"):
            required_fields: dict = {
                "DAGSHUB_REPO_OWNER": self.DAGSHUB_REPO_OWNER,
                "DAGSHUB_REPO_NAME": self.DAGSHUB_REPO_NAME,
                "DAGSHUB_TOKEN": self.DAGSHUB_TOKEN,
                "MLFLOW_TRACKING_URI": self.MLFLOW_TRACKING_URI,
                "SUPABASE_URL": self.SUPABASE_URL,
                "SUPABASE_KEY": self.SUPABASE_KEY,
                "JWT_SECRET_KEY": self.JWT_SECRET_KEY,
            }

            missing: list[str] = [k for k, v in required_fields.items() if v is None]

            if missing:
                error_msg = f"Missing required settings: {missing}"
                raise ValueError(error_msg)

        return self


@lru_cache
def get_settings() -> Settings:
    """
    Retrieve a cached instance of application settings.

    Returns
    -------
    Settings
        Singleton instance of the application configuration.

    Notes
    -----
    - Uses `functools.lru_cache` to avoid reloading settings multiple times.
    - Ensures consistent configuration across the application lifecycle.
    """
    return Settings()
