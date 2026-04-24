"""
Data loading module.

This module provides a flexible dataset loader that supports multiple file
formats using pandas. It automatically selects the appropriate pandas reader
based on file extension and integrates with Prefect's task system for logging,
retries, and observability.

Supported formats include CSV, Parquet, Excel, JSON, Pickle, XML, Feather,
HDF5, Stata, and ORC.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from prefect import get_run_logger, task

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

# ---------------------------------------------------------------------------------
# Map file extensions to pandas data loading functions
# ---------------------------------------------------------------------------------

loaders: dict = {
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
    log_prints=True,
)
def load_dataset(input_file: str | Path, **kwargs: object) -> pd.DataFrame:
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
    **kwargs : object
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

    Notes
    -----
    - If the loader does not return a DataFrame, an attempt is made to coerce
      the result into one.
    - Logging is handled via Prefect's `get_run_logger`.
    - Supported file extensions are defined in the `loaders` mapping.

    Examples
    --------
    >>> df = load_dataset("data.csv")
    >>> df.head()

    >>> df = load_dataset("data.parquet", columns=["col1", "col2"])
    """
    file_path: Path = Path(input_file).resolve()
    log: Logger | LoggerAdapter[Logger] = get_run_logger()

    # Check file exists
    if not file_path.is_file():
        msg = f"File not found at path: {file_path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    # Check extension is supported or not
    ext: str = file_path.suffix.lower()
    if ext not in loaders:
        supported: str = ", ".join(loaders.keys())
        msg = f"Unsupported extension '{ext}'. Supported: {supported}"
        log.error(msg)
        raise ValueError(msg)

    try:
        log.info("Loading %s...", file_path.name)
        result: pd.DataFrame = loaders[ext](file_path, **kwargs)

        if not isinstance(result, pd.DataFrame):
            log.warning("Loader returned %s instead of DataFrame. Coercing...", type(result))
            df: pd.DataFrame = pd.concat(list(result)) if hasattr(result, "__iter__") else result

        else:
            df: pd.DataFrame = result

    except ImportError as e:
        missing_lib: str = str(e).split("'")[-2] if "'" in str(e) else "required library"
        log.exception("Missing dependency for %s: Try 'pip install %s'", ext, missing_lib)
        raise

    except Exception:
        log.exception("An error occurred while loading %s", file_path)
        raise

    else:
        log.info("Successfully loaded %d rows and %d columns.", len(df), len(df.columns))
        return df
