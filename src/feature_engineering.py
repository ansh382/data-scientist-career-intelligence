"""Reusable text and range-parsing helpers for the cleaning pipeline."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

_RANGE_PATTERN = re.compile(
    r"^\s*[^\d]*?([+-]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:to|[-\u2013\u2014])\s*"
    r"([+-]?\d[\d,]*(?:\.\d+)?)\s*[^\d]*$",
    flags=re.IGNORECASE,
)
_SEPARATOR_PATTERN = re.compile(r"(?:\s*[,;|]\s*)+")


def _parse_numeric_range(value: object) -> tuple[float, float]:
    """Parse a two-ended numeric range; return NaNs for unsupported values."""
    if not isinstance(value, str) or not value.strip():
        return np.nan, np.nan

    match = _RANGE_PATTERN.fullmatch(value.strip())
    if match is None:
        return np.nan, np.nan

    try:
        lower = float(match.group(1).replace(",", ""))
        upper = float(match.group(2).replace(",", ""))
    except ValueError:
        return np.nan, np.nan

    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return np.nan, np.nan
    return lower, upper


def parse_salary_range(value: object) -> tuple[float, float]:
    """Parse values such as ``10to15`` into a (minimum, maximum) pair."""
    return _parse_numeric_range(value)


def parse_experience_range(value: object) -> tuple[float, float]:
    """Parse experience ranges such as ``0-3 yrs`` into numeric bounds."""
    return _parse_numeric_range(value)


def clean_text_columns(
    dataframe: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Trim and collapse whitespace in selected text columns without altering nulls."""
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Text columns not found: {missing_columns}")

    cleaned = dataframe.copy()
    for column in columns:
        def clean_value(value: object) -> object:
            if not isinstance(value, str):
                return value
            normalized = " ".join(value.split())
            return normalized if normalized else pd.NA

        cleaned[column] = cleaned[column].map(clean_value)
    return cleaned


def create_salary_features(
    dataframe: pd.DataFrame, source_column: str = "salary"
) -> pd.DataFrame:
    """Add numeric minimum, maximum, and midpoint salary range features."""
    if source_column not in dataframe.columns:
        raise KeyError(f"Salary source column not found: {source_column}")

    engineered = dataframe.copy()
    bounds = [parse_salary_range(value) for value in engineered[source_column]]
    engineered["salary_min"] = [lower for lower, _ in bounds]
    engineered["salary_max"] = [upper for _, upper in bounds]
    engineered["salary_midpoint"] = [
        (lower + upper) / 2 if np.isfinite(lower) and np.isfinite(upper) else np.nan
        for lower, upper in bounds
    ]
    return engineered


def create_experience_features(
    dataframe: pd.DataFrame, source_column: str = "experience"
) -> pd.DataFrame:
    """Add numeric minimum, maximum, and midpoint experience features."""
    if source_column not in dataframe.columns:
        raise KeyError(f"Experience source column not found: {source_column}")

    engineered = dataframe.copy()
    bounds = [parse_experience_range(value) for value in engineered[source_column]]
    engineered["experience_min"] = [lower for lower, _ in bounds]
    engineered["experience_max"] = [upper for _, upper in bounds]
    engineered["experience_midpoint"] = [
        (lower + upper) / 2 if np.isfinite(lower) and np.isfinite(upper) else np.nan
        for lower, upper in bounds
    ]
    return engineered


def normalize_skill_text(value: object) -> object:
    """Normalize casing, whitespace, and common list separators in skill text."""
    if not isinstance(value, str):
        return value

    normalized = " ".join(value.split()).casefold()
    normalized = _SEPARATOR_PATTERN.sub(", ", normalized)
    return normalized
