"""Load the four RAW datasets and summarise them.

Raw files are only ever READ here. Nothing is renamed, dropped or imputed;
cleaning lives in data_cleaning.py and writes to outputs/cleaned_data/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_DIR, DATASET_FILES, EXPECTED_COLUMNS

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when a dataset is missing, unreadable or empty."""


def _check_file(path: Path) -> None:
    """Fail early with an actionable message if a file is missing."""
    if not path.exists():
        raise DataLoadError(
            f"Required file not found: {path}\n"
            f"Place '{path.name}' inside the folder: {path.parent}"
        )
    if not path.is_file():
        raise DataLoadError(f"Expected a file but found a directory: {path}")


def check_required_files(data_dir: Path = DATA_DIR) -> None:
    """Verify ALL four files exist and report every missing one at once."""
    missing = [n for n in DATASET_FILES.values() if not (data_dir / n).is_file()]
    if missing:
        raise DataLoadError(
            "Missing required data file(s) in "
            f"{data_dir}:\n  - " + "\n  - ".join(missing)
        )


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, trying common encodings (public job data is often cp1252)."""
    _check_file(path)
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Loaded %s (encoding=%s): %d rows x %d cols",
                        path.name, enc, *df.shape)
            return df
        except UnicodeDecodeError as exc:
            last_err = exc  # try the next encoding
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataLoadError(f"Could not parse CSV {path.name}: {exc}") from exc
    raise DataLoadError(f"Could not decode {path.name} with any encoding: {last_err}")


def load_excel(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read an Excel sheet with openpyxl."""
    _check_file(path)
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ImportError as exc:
        raise DataLoadError("openpyxl is required: pip install openpyxl") from exc
    except Exception as exc:  # corrupt workbook, bad sheet name, etc.
        raise DataLoadError(f"Could not read Excel file {path.name}: {exc}") from exc
    logger.info("Loaded %s: %d rows x %d cols", path.name, *df.shape)
    return df


def _warn_on_column_differences(key: str, df: pd.DataFrame) -> None:
    """Log (not raise) when raw columns differ from the data dictionary."""
    expected, actual = EXPECTED_COLUMNS[key], list(df.columns)
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing:
        logger.warning("[%s] columns in data dictionary but NOT in file: %s", key, missing)
    if extra:
        logger.warning("[%s] columns in file but NOT in data dictionary: %s", key, extra)


def load_all_datasets(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ds_jobs, analytics_jobs, jds_skills, sds_personality) - raw."""
    check_required_files(data_dir)
    ds_jobs = load_csv(data_dir / DATASET_FILES["ds_jobs"])
    analytics_jobs = load_csv(data_dir / DATASET_FILES["analytics_jobs"])
    jds_skills = load_excel(data_dir / DATASET_FILES["jds_skills"])
    sds_personality = load_excel(data_dir / DATASET_FILES["sds_personality"])

    frames = {"ds_jobs": ds_jobs, "analytics_jobs": analytics_jobs,
              "jds_skills": jds_skills, "sds_personality": sds_personality}
    for key, df in frames.items():
        if df.empty:
            raise DataLoadError(f"Dataset '{key}' is empty.")
        _warn_on_column_differences(key, df)
    return ds_jobs, analytics_jobs, jds_skills, sds_personality


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Row/column counts, names, dtypes, missing values and duplicate rows."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("get_dataset_summary expects a pandas DataFrame.")
    n_rows = len(df)
    missing = df.isna().sum()
    return {
        "row_count": n_rows,
        "column_count": df.shape[1],
        "column_names": list(df.columns),
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in missing.items()},
        "missing_pct": {c: round(100 * v / n_rows, 2) if n_rows else 0.0
                        for c, v in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    from src.config import configure_logging
    configure_logging()
    names = ("ds_jobs", "analytics_jobs", "jds_skills", "sds_personality")
    for name, frame in zip(names, load_all_datasets()):
        print(f"\n=== {name} ===")
        for k, v in get_dataset_summary(frame).items():
            print(f"{k}: {v}")
