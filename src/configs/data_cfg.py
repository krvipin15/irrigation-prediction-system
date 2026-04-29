"""
Data Configuration Module.

This module defines standardized data handling configurations used across
the machine learning pipeline. It centralizes definitions for missing values,
boolean parsing, and optimized data types to ensure consistency, memory
efficiency, and robust data ingestion.

The configurations in this module are primarily used during data loading,
cleaning, and preprocessing stages to normalize raw input data into a
structured and model-compatible format.

Features
--------
- Comprehensive definitions of null value representations commonly found
  in real-world datasets.
- Standardized mappings for boolean values (true/false variants).
- Optimized pandas dtypes for improved memory usage and performance.
- Centralized configuration to ensure consistency across pipeline stages.

Notes
-----
- Using "Float32" instead of "float64" significantly reduces memory footprint.
- "category" dtype improves performance for encoding and modeling steps.
- Consistent NA handling prevents silent data quality issues.
- This module should remain lightweight and free of runtime logic.
"""

# Types of null values
NA_VALUES: list[str] = [
    "NA",
    "N/A",
    "n/a",
    "NaN",
    "nan",
    "Null",
    "NULL",
    "null",
    "None",
    "none",
    "",
    "<NA>",
    "Unknown",
    "unknown",
    "UNKNOWN",
    "Missing",
    "missing",
    "MISSING",
    "Not Applicable",
    "not applicable",
    "Not Available",
    "not available",
    "TBD",
    "tbd",
    "Pending",
    "pending",
    "Nil",
    "nil",
    "NIL",
    "-",
    "--",
    "---",
    ".",
    "..",
    "...",
    "?",
    "??",
    " ",
    "  ",
    "\t",
    "\n",
    "\\N",
    "NULL_VALUE",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "#VALUE!",
    "#DIV/0!",
    "#REF!",
    "#NAME?",
]

# Types of true values
TRUE_VALUES: list[str] = [
    "True",
    "true",
    "TRUE",
    "T",
    "t",
    "Yes",
    "yes",
    "YES",
    "Y",
    "y",
]

# Types of false values
FALSE_VALUES: list[str] = [
    "False",
    "false",
    "FALSE",
    "F",
    "f",
    "No",
    "no",
    "NO",
    "N",
    "n",
]

# Map better data types to optimize memory size
OPTIMIZED_DTYPES: dict[str, str] = {
    "Soil_Type": "category",
    "Soil_pH": "Float32",
    "Soil_Moisture": "Float32",
    "Organic_Carbon": "Float32",
    "Electrical_Conductivity": "Float32",
    "Temperature_C": "Float32",
    "Humidity": "Float32",
    "Rainfall_mm": "Float32",
    "Sunlight_Hours": "Float32",
    "Wind_Speed_kmh": "Float32",
    "Crop_Type": "category",
    "Crop_Growth_Stage": "category",
    "Season": "category",
    "Irrigation_Type": "category",
    "Water_Source": "category",
    "Field_Area_hectare": "Float32",
    "Mulching_Used": "boolean",
    "Previous_Irrigation_mm": "Float32",
    "Region": "category",
    "Irrigation_Need": "category",
}
