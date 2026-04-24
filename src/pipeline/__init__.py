"""
Data pipeline package.

This package contains modular pipeline components responsible for data
ingestion, transformation, and orchestration within the Irrigation
Prediction System. It is designed to integrate with workflow orchestration
tools such as Prefect and supports scalable, fault-tolerant data processing.

Modules
-------
ingest : Data ingestion tasks for loading raw datasets

Notes
-----
- Each module represents a logical stage in the data pipeline.
- Tasks are designed to be composable and reusable across workflows.
- Logging and configuration are centralized via `src.configs`.
- Pipelines are orchestrated externally (e.g., `orchestrator.py`).

Examples
--------
>>> from src.pipeline.ingest import load_dataset
>>> df = load_dataset("data/raw/raw_data.csv")
"""
