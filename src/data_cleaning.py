"""Clean the Phase 1 datasets and write derived files without changing raw data."""
from __future__ import annotations

import json
import logging
import re
from numbers import Number
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import CLEANED_DIR, OUTPUT_DIR, configure_logging
from src.data_loader import load_all_datasets
from src.feature_engineering import (
    clean_text_columns,
    create_experience_features,
    create_salary_features,
    normalize_skill_text,
    parse_experience_range,
    parse_salary_range,
)

logger = logging.getLogger(__name__)

_SALARY_VALUE_PATTERN = re.compile(
    r"^\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*[lL]?\s*$"
)
_EXPERIENCE_VALUE_PATTERN = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*$",
    flags=re.IGNORECASE,
)

_JDS_NUMERIC_COLUMNS = (
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
)
_SDS_NUMERIC_COLUMNS = (
    "neuroticism",
    "extraversion",
    "openness_to_experience",
    "agreeableness",
    "conscientiousness",
)


def _json_value(value: object) -> object:
    """Convert common pandas/numpy scalars to JSON-safe values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _missing_values(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in frame.isna().sum().items()}


def _normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace around/within underscores in imported column labels."""
    cleaned = frame.copy()
    renamed = [
        re.sub(r"\s*_\s*", "_", str(column).strip())
        for column in cleaned.columns
    ]
    if len(set(renamed)) != len(renamed):
        raise ValueError("Column-name whitespace normalization would create duplicates.")
    cleaned.columns = renamed
    return cleaned


def _failure_records(
    values: pd.Series,
    parser: Callable[[object], object],
    field: str,
    invalid: Callable[[object], bool],
    problem: str,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row_index, value in values.items():
        if pd.isna(value):
            continue
        try:
            parsed = parser(value)
            failed = invalid(parsed)
        except (TypeError, ValueError, OverflowError):
            failed = True
        if not failed:
            continue

        key = repr(value)
        if key not in grouped:
            grouped[key] = {
                "field": field,
                "original_value": _json_value(value),
                "row_count": 0,
                "example_row_indices": [],
                "problem": problem,
            }
        record = grouped[key]
        record["row_count"] = int(record["row_count"]) + 1
        examples = record["example_row_indices"]
        if len(examples) < 5:
            examples.append(_json_value(row_index))
    return list(grouped.values())


def _invalid_numeric(parsed: object) -> bool:
    try:
        return not np.isfinite(float(parsed))
    except (TypeError, ValueError, OverflowError):
        return True


def _parse_salary_value(value: object) -> float:
    if isinstance(value, Number) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else np.nan
    if not isinstance(value, str):
        return np.nan
    match = _SALARY_VALUE_PATTERN.fullmatch(value)
    if match is None:
        return np.nan
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def _parse_experience_value(value: object) -> float:
    if isinstance(value, Number) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else np.nan
    if not isinstance(value, str):
        return np.nan
    match = _EXPERIENCE_VALUE_PATTERN.fullmatch(value)
    if match is None:
        return np.nan
    try:
        return float(match.group(1))
    except ValueError:
        return np.nan


def _convert_numeric_column(
    frame: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if column not in frame.columns:
        raise KeyError(f"Numeric column not found: {column}")
    original = frame[column]
    failures = _failure_records(
        original,
        lambda value: pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
        column,
        _invalid_numeric,
        "Value could not be converted to a number.",
    )
    converted = frame.copy()
    converted[column] = pd.to_numeric(original, errors="coerce")
    return converted, failures


def _range_failures(
    values: pd.Series,
    field: str,
    parser: Callable[[object], tuple[float, float]],
) -> list[dict[str, object]]:
    return _failure_records(
        values,
        parser,
        field,
        lambda bounds: (
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or not all(np.isfinite(bound) for bound in bounds)
        ),
        "Value is not a supported, ordered numeric range.",
    )


def _clean_ds_jobs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    cleaned = clean_text_columns(frame, ("company_name", "job_title"))
    transformations = [
        "Trimmed and collapsed whitespace in company_name and job_title; "
        "whitespace-only values were marked missing."
    ]
    failures: list[dict[str, object]] = []

    for source, destination in (
        ("min_salary", "min_salary_num"),
        ("avg_salary", "avg_salary_num"),
        ("max_salary", "max_salary_num"),
    ):
        failures.extend(
            _failure_records(
                cleaned[source],
                _parse_salary_value,
                source,
                _invalid_numeric,
                "Value is not a supported numeric salary (optional L suffix).",
            )
        )
        cleaned[destination] = cleaned[source].map(_parse_salary_value).astype(float)
    transformations.append(
        "Added min_salary_num, avg_salary_num, and max_salary_num; original "
        "salary columns were retained and L suffixes were removed without scaling."
    )

    if is_numeric_dtype(cleaned["min_experience"]):
        cleaned["min_experience_num"] = cleaned["min_experience"].copy()
    else:
        failures.extend(
            _failure_records(
                cleaned["min_experience"],
                _parse_experience_value,
                "min_experience",
                _invalid_numeric,
                "Value is not a standalone numeric experience value.",
            )
        )
        cleaned["min_experience_num"] = (
            cleaned["min_experience"].map(_parse_experience_value).astype(float)
        )
    transformations.append(
        "Added min_experience_num; existing numeric values were preserved and "
        "text values were parsed only when unambiguous."
    )
    return cleaned, transformations, failures


def _clean_analytics_jobs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    text_columns = (
        "job_description",
        "job_desig",
        "job_type",
        "key_skills",
        "location",
        "salary",
        "experience",
    )
    cleaned = clean_text_columns(frame, text_columns)
    transformations = [
        "Trimmed and collapsed whitespace in job description, designation, job "
        "type, skills, location, salary, and experience text."
    ]

    missing_job_type = cleaned["job_type"].isna()
    cleaned.loc[~missing_job_type, "job_type"] = cleaned.loc[
        ~missing_job_type, "job_type"
    ].map(lambda value: value.casefold() if isinstance(value, str) else value)
    cleaned.loc[missing_job_type, "job_type"] = "Unknown"
    transformations.append(
        "Normalized job_type capitalization and replaced missing job_type "
        "values with the requested Unknown label; other category spellings "
        "were not merged."
    )

    cleaned["key_skills_clean"] = cleaned["key_skills"].map(normalize_skill_text)
    transformations.append(
        "Added key_skills_clean with normalized case, whitespace, and common "
        "comma/semicolon/pipe separators; key_skills was retained unchanged."
    )

    failures = _range_failures(
        cleaned["experience"], "experience", parse_experience_range
    )
    cleaned = create_experience_features(cleaned)
    failures.extend(_range_failures(cleaned["salary"], "salary", parse_salary_range))
    cleaned = create_salary_features(cleaned)
    transformations.extend(
        (
            "Added experience_min, experience_max, and experience_midpoint "
            "from supported numeric experience ranges.",
            "Added salary_min, salary_max, and salary_midpoint from supported "
            "numeric salary ranges.",
        )
    )
    return cleaned, transformations, failures


def _unexpected_target_values(
    values: pd.Series, target_column: str
) -> list[dict[str, object]]:
    numeric = pd.to_numeric(values, errors="coerce")
    grouped: dict[str, dict[str, object]] = {}
    for row_index, (original, numeric_value) in enumerate(zip(values, numeric)):
        if pd.isna(original):
            continue
        if pd.isna(numeric_value) or numeric_value not in (0, 1):
            key = repr(original)
            if key not in grouped:
                grouped[key] = {
                    "field": target_column,
                    "value": _json_value(original),
                    "row_count": 0,
                    "example_row_indices": [],
                }
            record = grouped[key]
            record["row_count"] = int(record["row_count"]) + 1
            examples = record["example_row_indices"]
            if len(examples) < 5:
                examples.append(row_index)
    return list(grouped.values())


def _clean_traits(
    frame: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]], list[dict[str, object]]]:
    cleaned = _normalize_column_names(frame)
    transformations: list[str] = []
    if list(cleaned.columns) != list(frame.columns):
        transformations.append(
            "Removed accidental whitespace in column names; no columns were dropped."
        )

    failures: list[dict[str, object]] = []
    for column in numeric_columns:
        cleaned, column_failures = _convert_numeric_column(cleaned, column)
        failures.extend(column_failures)
    transformations.append(
        "Validated and converted trait score columns to numeric; unparseable "
        "values were set to NaN and recorded."
    )

    if target_column not in cleaned.columns:
        raise KeyError(f"Target column not found: {target_column}")
    unexpected = _unexpected_target_values(cleaned[target_column], target_column)
    transformations.append(
        "Validated target values as binary 0/1 without replacing or deleting "
        "unexpected target values."
    )
    return cleaned, transformations, failures, unexpected


def _dataset_report(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    transformations: list[str],
    parsing_failures: list[dict[str, object]],
    unexpected_targets: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "original_row_count": int(len(original)),
        "cleaned_row_count": int(len(cleaned)),
        "original_column_count": int(original.shape[1]),
        "cleaned_column_count": int(cleaned.shape[1]),
        "missing_values_before_cleaning": _missing_values(original),
        "missing_values_after_cleaning": _missing_values(cleaned),
        "duplicate_rows": int(original.duplicated().sum()),
        "duplicate_rows_after_cleaning": int(cleaned.duplicated().sum()),
        "rows_removed_as_duplicates": 0,
        "transformations_performed": transformations,
        "parsing_failures": parsing_failures,
        "unexpected_target_values": unexpected_targets or [],
    }


def _save_cleaned(
    name: str,
    frame: pd.DataFrame,
    output_directory: Path = CLEANED_DIR,
) -> None:
    destination = output_directory / name
    frame.to_csv(destination, index=False)
    logger.info("Saved %s (%d rows x %d columns)", destination, *frame.shape)


def main() -> None:
    """Load, clean, report, and save all four datasets."""
    configure_logging()
    frames = load_all_datasets()
    names = ("ds_jobs", "analytics_jobs", "jds_skills", "sds_personality")
    originals = dict(zip(names, frames))

    ds_jobs, ds_transforms, ds_failures = _clean_ds_jobs(originals["ds_jobs"])
    analytics, analytics_transforms, analytics_failures = _clean_analytics_jobs(
        originals["analytics_jobs"]
    )
    jds, jds_transforms, jds_failures, jds_unexpected = _clean_traits(
        originals["jds_skills"],
        _JDS_NUMERIC_COLUMNS,
        "salary_hike_high_or_low",
    )
    sds, sds_transforms, sds_failures, sds_unexpected = _clean_traits(
        originals["sds_personality"],
        _SDS_NUMERIC_COLUMNS,
        "success_classification_high_low",
    )
    cleaned_frames = {
        "ds_jobs": ds_jobs,
        "analytics_jobs": analytics,
        "jds_skills": jds,
        "sds_personality": sds,
    }

    report = {
        name: _dataset_report(
            originals[name],
            cleaned_frames[name],
            transforms,
            failures,
            unexpected,
        )
        for name, transforms, failures, unexpected in (
            ("ds_jobs", ds_transforms, ds_failures, []),
            ("analytics_jobs", analytics_transforms, analytics_failures, []),
            ("jds_skills", jds_transforms, jds_failures, jds_unexpected),
            ("sds_personality", sds_transforms, sds_failures, sds_unexpected),
        )
    }

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_names = {
        "ds_jobs": "ds_jobs_clean.csv",
        "analytics_jobs": "analytics_jobs_clean.csv",
        "jds_skills": "jds_skills_clean.csv",
        "sds_personality": "sds_personality_clean.csv",
    }
    for dataset, filename in output_names.items():
        _save_cleaned(filename, cleaned_frames[dataset])

    report_path = OUTPUT_DIR / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, allow_nan=False)
        report_file.write("\n")

    failure_count = sum(
        len(dataset_report["parsing_failures"]) for dataset_report in report.values()
    )
    unexpected_count = sum(
        len(dataset_report["unexpected_target_values"])
        for dataset_report in report.values()
    )
    print(
        "Phase 2 cleaning complete: "
        f"{len(cleaned_frames)} datasets saved to {CLEANED_DIR}; "
        f"{failure_count} distinct parsing failure(s); "
        f"{unexpected_count} distinct unexpected target value(s). "
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
