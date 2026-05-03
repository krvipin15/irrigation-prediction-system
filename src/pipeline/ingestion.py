"""
Data Ingestion Pipeline.

This module provides functionality for loading datasets from disk into
pandas DataFrame objects in a flexible and format-agnostic manner. It is
designed as part of a data pipeline and integrates with Prefect for task
orchestration, logging, and execution control.

The ingestion pipeline supports multiple file formats by mapping file
extensions to their corresponding pandas reader functions. It ensures
robust handling of common issues such as missing files, unsupported
formats, and missing dependencies.

Features
--------
- Supports a wide range of file formats including CSV, Parquet, Excel,
  JSON, Pickle, XML, Feather, HDF5, Stata, and ORC.
- Automatic detection of file type based on extension.
- Prefect task integration with logging, retries, and timeout handling.
- Graceful error handling with informative log messages.
- Extensible design for adding new file loaders.

Notes
-----
- Additional keyword arguments are forwarded to the underlying pandas
  reader functions, allowing full customization of loading behavior.
- Some formats may require optional dependencies (e.g., `pyarrow`,
  `openpyxl`, `fastparquet`).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from prefect import get_run_logger, task

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------------------
# Map file extensions to pandas data loading functions
# ---------------------------------------------------------------------------------

LOADERS: dict = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".xml": pd.read_xml,
    ".feather": pd.read_feather,
    ".hdf": pd.read_hdf,
    ".h5": pd.read_hdf,
    ".dta": pd.read_stata,
    ".orc": pd.read_orc,
}

# ---------------------------------------------------------------------------------
# Prefect Task: Load dataset into Pandas DataFrame
# ---------------------------------------------------------------------------------


@task(
    name="Load Dataset",
    description="Load various file formats into a pandas DataFrame.",
    tags=["ingestion", "pipeline"],
    timeout_seconds=300,
)
def load_dataset(input_file: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load a dataset from disk into a pandas DataFrame.

    The function determines the appropriate pandas reader based on the file
    extension and supports multiple formats such as CSV, Parquet, Excel, JSON,
    and more. It is implemented as a Prefect task with built-in logging,
    retries, and timeout handling.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Path to the input dataset file.
    **kwargs : Any
        Additional keyword arguments passed to the corresponding pandas
        reader function (e.g., `pd.read_csv`, `pd.read_parquet`, etc.).

    Returns
    -------
    pandas.DataFrame
        The loaded dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is not supported.
    ImportError
        If a required dependency for the file format is missing.
    Exception
        Propagates any exception raised during file loading.

    Examples
    --------
    >>> df = load_dataset("data.csv")
    >>> df.head()

    >>> df = load_dataset("data.parquet", columns=["col1", "col2"])
    """
    file_path: Path = Path(input_file).resolve()
    log: Logger | LoggerAdapter[Logger] = get_run_logger()

    # --- 1. Validation ---
    if not file_path.is_file():
        msg = f"File not found at path: {file_path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    # Check extension is supported or not
    ext: str = file_path.suffix.lower()
    if ext not in LOADERS:
        supported: str = ", ".join(LOADERS.keys())
        msg = f"Unsupported extension '{ext}'. Supported: {supported}"
        log.error(msg)
        raise ValueError(msg)

    # --- 2. Execution ---
    try:
        log.info("Loading %s...", file_path.name)
        result: pd.DataFrame = LOADERS[ext](file_path, **kwargs)

        # Handle cases where pandas returns an iterator/reader instead of a DataFrame
        if not isinstance(result, pd.DataFrame):
            log.warning("Loader returned %s instead of DataFrame. Coercing...", type(result))
            df: pd.DataFrame = pd.concat(list(result)) if hasattr(result, "__iter__") else result

        else:
            df: pd.DataFrame = result

    except ImportError as e:
        missing_lib: str = str(e).split("'")[-2] if "'" in str(e) else "required library"
        log.error("Missing dependency for %s: Try 'pip install %s'", ext, missing_lib)
        raise

    except Exception as e:
        log.error("An error occurred while loading %s: %s", file_path, str(e))
        raise

    # --- 3. Success Telemetry ---
    log.info("Successfully loaded %d rows and %d columns.", len(df), len(df.columns))
    return df
