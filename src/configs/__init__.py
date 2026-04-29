"""
Configuration Package.

This package provides centralized configuration utilities for the
machine learning pipeline. It includes modules for data-related constants
and application-wide settings, ensuring consistent behavior across all
pipeline components.

Modules
-------
data_cfg
    Defines standardized data handling configurations such as null values,
    boolean mappings, and optimized data types.

settings
    Provides environment-aware application settings using Pydantic,
    including paths, filenames, and external service configurations.

Design Principles
-----------------
- Centralized: All configuration is defined in one place.
- Consistent: Shared across ingestion, validation, preprocessing, and inference.
- Environment-aware: Supports development, staging, production, and test modes.
- Safe: Includes validation and secure handling of sensitive values.
"""

from src.configs.data_cfg import (
    FALSE_VALUES,
    NA_VALUES,
    OPTIMIZED_DTYPES,
    TRUE_VALUES,
)
from src.configs.settings import Settings, get_settings

__all__: list[str] = [
    "NA_VALUES",
    "TRUE_VALUES",
    "FALSE_VALUES",
    "OPTIMIZED_DTYPES",
    "Settings",
    "get_settings",
]
