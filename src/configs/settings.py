"""
Application Settings Module.

This module defines the centralized configuration system for the entire
machine learning pipeline using Pydantic Settings. It manages environment
variables, file paths, artifact locations, and external service credentials,
ensuring consistency and reliability across all components of the system.

The configuration is environment-aware and supports automatic validation,
directory creation, and secure handling of sensitive information.

Features
--------
- Centralized configuration using Pydantic `BaseSettings`.
- Automatic loading from environment variables and `.env` files.
- Dynamic timestamp-based file naming for reproducibility.
- Automatic directory creation for all required paths.
- Environment-aware validation (development, staging, production, test).
- Secure handling of secrets (e.g., tokens) using `SecretStr`.
- Cached singleton access for performance and consistency.

Notes
-----
- All paths are resolved relative to the project base directory.
- Timestamp-based filenames prevent overwriting previous artifacts.
- Missing required environment variables will raise a ValueError in
  non-test environments.
- This module should be imported wherever configuration is needed.
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

    PROJ_NAME: str = "Irrigation Prediction System"
    ENV: str = Field(default="development", pattern="^(development|staging|production|test)$")

    # ---------------------------------------------------------
    # Centralized Directories & Filenames
    # ---------------------------------------------------------

    # Root
    ART_DIR: Path = BASE_DIR / "artifacts"
    DATA_DIR: Path = ART_DIR / "data"

    # Data Sources
    RAW_DIR: Path = DATA_DIR / "raw"
    RAW_FILE: str = "raw_v1.csv"

    UNSEEN_DIR: Path = DATA_DIR / "unseen"
    UNSEEN_FILE: str = "unseen_v1.csv"

    PROC_DIR: Path = DATA_DIR / "proc"
    PROC_FILE: str = f"proc_{timestamp}.parquet"

    # Experimentation
    EXP_DIR: Path = DATA_DIR / "exp"
    LIN_FILE: str = "lin_v1.parquet"
    TREE_FILE: str = "tree_v1.parquet"
    REF_FILE: str = TREE_FILE

    # Output
    PRED_DIR: Path = DATA_DIR / "pred"
    PRED_FILE: str = f"pred_{timestamp}.csv"

    # Models & Objects
    MODEL_DIR: Path = ART_DIR / "models"
    MODEL_FILE: str = "best_v1.skops"

    OBJ_DIR: Path = ART_DIR / "objects"
    PIPE_FILE: str = "pipe_v1.joblib"
    ENC_FILE: str = "enc_v1.joblib"

    # Reports
    VAL_DIR: Path = ART_DIR / "reports/val"
    VAL_FILE: str = f"val_{timestamp}.json"

    MON_DIR: Path = ART_DIR / "reports/mon"
    MON_FILE: str = f"mon_{timestamp}.html"

    @field_validator(
        "ART_DIR",
        "RAW_DIR",
        "UNSEEN_DIR",
        "PROC_DIR",
        "EXP_DIR",
        "PRED_DIR",
        "OBJ_DIR",
        "MODEL_DIR",
        "VAL_DIR",
        "MON_DIR",
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
        if self.ENV in ("development", "staging", "production"):
            required_fields: dict = {
                "DAGSHUB_REPO_OWNER": self.DAGSHUB_REPO_OWNER,
                "DAGSHUB_REPO_NAME": self.DAGSHUB_REPO_NAME,
                "DAGSHUB_TOKEN": self.DAGSHUB_TOKEN,
                "MLFLOW_TRACKING_URI": self.MLFLOW_TRACKING_URI,
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
