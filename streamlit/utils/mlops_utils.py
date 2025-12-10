"""
MLOps Utility Functions
Shared functions for ML lifecycle and model registry pages
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
from google.cloud import bigquery
import os

# Standard metric keys used across MLOps pages
METRIC_KEYS = ["MAE", "R2", "SMAPE", "RMSE_RECENT6", "PEARSON_R", "MASE"]

# Metric mapping from lowercase JSON keys to normalized uppercase keys
METRIC_KEY_MAPPING = {
    "mae": "MAE",
    "r2": "R2",
    "smape": "SMAPE",
    "rmse_recent6": "RMSE_RECENT6",
    "pearson_r": "PEARSON_R",
    "mase": "MASE"
}


def parse_json(value):
    """
    Safely parse JSON string or return dict.
    
    Args:
        value: JSON string, dict, or None
        
    Returns:
        dict: Parsed JSON or empty dict if parsing fails
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def safe_display(x):
    """
    Safely display a value, returning '–' for None or NaN.
    
    Args:
        x: Value to display
        
    Returns:
        str: Display string or '–' for missing values
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return str(x)


@st.cache_resource(show_spinner=False)
def get_bq_client(project_id=None):
    """
    Get cached BigQuery client.
    
    Args:
        project_id: GCP project ID (defaults to env var or default)
        
    Returns:
        bigquery.Client or None if connection fails
    """
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT") or "pipeline-882-team-project"
    
    try:
        return bigquery.Client(project=project_id)
    except Exception:
        return None


def extract_metrics(df, col_name, metric_map=None):
    """
    Extract metrics from JSON column and create separate columns.
    
    Args:
        df: DataFrame with JSON metrics column
        col_name: Name of the JSON column (e.g., "metrics", "metrics_json")
        metric_map: Optional custom metric mapping (defaults to METRIC_KEY_MAPPING)
        
    Returns:
        DataFrame: DataFrame with extracted metric columns
    """
    if metric_map is None:
        metric_map = METRIC_KEY_MAPPING
    
    # Create empty columns for all metrics
    for col in metric_map.values():
        if col not in df.columns:
            df[col] = np.nan
    
    # Fill values from JSON
    for idx, row in df.iterrows():
        data = parse_json(row.get(col_name))
        for key, val in data.items():
            key_lower = key.lower()
            if key_lower in metric_map:
                df.at[idx, metric_map[key_lower]] = val
    
    return df


def extract_model_name(df, col_name="params"):
    """
    Extract model name (algorithm) from params JSON column.
    
    Args:
        df: DataFrame with params JSON column
        col_name: Name of the params column (default: "params")
        
    Returns:
        DataFrame: DataFrame with model_name column added
    """
    df = df.copy()
    if "model_name" not in df.columns:
        df["model_name"] = None
    
    for idx, row in df.iterrows():
        params = parse_json(row.get(col_name))
        if isinstance(params, dict) and "algorithm" in params:
            df.at[idx, "model_name"] = params["algorithm"]
    
    return df


def format_datetime_column(df, col_name, format_str="%Y-%m-%d %H:%M:%S"):
    """
    Format datetime column for display.
    
    Args:
        df: DataFrame
        col_name: Name of datetime column
        format_str: Format string (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        DataFrame: DataFrame with formatted datetime column
    """
    df = df.copy()
    if col_name in df.columns:
        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        df[col_name] = df[col_name].apply(
            lambda x: x.strftime(format_str) if pd.notna(x) else "–"
        )
    return df


def get_all_metrics_from_row(row, metrics_col="metrics", metric_map=None):
    """
    Extract all metrics from a DataFrame row (from both extracted columns and JSON).
    
    Args:
        row: pandas Series (row from DataFrame)
        metrics_col: Name of metrics JSON column
        metric_map: Optional custom metric mapping
        
    Returns:
        dict: Dictionary of all available metrics
    """
    if metric_map is None:
        metric_map = METRIC_KEY_MAPPING
    
    available_metrics = {}
    
    # First, get metrics from JSON column and normalize keys
    if metrics_col in row.index:
        metrics_json_raw = row[metrics_col]
        if metrics_json_raw:
            metrics_json = parse_json(metrics_json_raw)
            if metrics_json:
                for key, val in metrics_json.items():
                    if isinstance(val, (int, float)) and not pd.isna(val) and key.lower() != "test_count":
                        key_lower = key.lower()
                        normalized_key = metric_map.get(key_lower, key.upper())
                        available_metrics[normalized_key] = val
    
    # Then add extracted metric columns (avoid duplicates)
    for metric_key in METRIC_KEYS:
        if metric_key in row.index and metric_key not in available_metrics:
            val = row[metric_key]
            if val is not None and pd.notna(val):
                available_metrics[metric_key] = val
    
    return available_metrics

