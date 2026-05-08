"""
src.data_processor – Data loading, validation, and preprocessing utilities.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> Optional[pd.DataFrame]:
    """Load a CSV or Excel file from raw bytes."""
    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(pd.io.common.BytesIO(file_bytes))
        if file_name.endswith((".xls", ".xlsx")):
            return pd.read_excel(pd.io.common.BytesIO(file_bytes))
        raise ValueError(f"Unsupported file type: {file_name}")
    except Exception as exc:
        logger.error("Failed to load file %s: %s", file_name, exc)
        st.error(f"❌ Could not load file: {exc}")
        return None


def validate_dataset(data: pd.DataFrame) -> Tuple[bool, str]:
    """Run basic sanity checks on the uploaded dataset."""
    if data is None or data.empty:
        return False, "Dataset is empty."
    if data.shape[0] < 20:
        return False, "Dataset has fewer than 20 rows — too small for reliable training."
    if data.shape[1] < 2:
        return False, "Dataset must have at least 2 columns (features + target)."
    return True, "Dataset looks good."


def get_dataset_summary(data: pd.DataFrame) -> Dict[str, object]:
    """Return a lightweight summary dict for the overview cards."""
    return {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_values": int(data.isnull().sum().sum()),
        "missing_pct": round(data.isnull().sum().sum() / data.size * 100, 2),
        "numeric_cols": data.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": data.select_dtypes(include="object").columns.tolist(),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def auto_process_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Automatically impute missing values and label-encode categoricals."""
    processed = data.copy()
    label_encoders: Dict[str, LabelEncoder] = {}

    n_dupes = processed.duplicated().sum()
    if n_dupes:
        processed = processed.drop_duplicates()
        logger.info("Dropped %d duplicate rows.", n_dupes)

    num_cols = processed.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    if num_cols:
        imputer = SimpleImputer(strategy="median")
        processed[num_cols] = imputer.fit_transform(processed[num_cols])

    cat_cols = processed.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if processed[col].isnull().any():
            fill_val = processed[col].mode().iloc[0]
            processed[col] = processed[col].fillna(fill_val)

    for col in cat_cols:
        le = LabelEncoder()
        processed[col] = le.fit_transform(processed[col].astype(str))
        label_encoders[col] = le

    return processed, label_encoders
