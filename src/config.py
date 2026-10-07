"""Central configuration: paths, expected file names and expected columns.

All paths are built with pathlib relative to the project root, so the project
runs from any machine without editing hard-coded absolute paths.
"""
from __future__ import annotations

import logging
from pathlib import Path

# Project root = parent of the `src` package folder.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"          # RAW data - never modified
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
CLEANED_DIR: Path = OUTPUT_DIR / "cleaned_data"
MODEL_DIR: Path = OUTPUT_DIR / "model_results"

# Dataset key -> raw file name (exactly as provided by the hackathon).
DATASET_FILES: dict[str, str] = {
    "ds_jobs": "DataScience Jobs.csv",
    "analytics_jobs": "Analytics Jobs.csv",
    "jds_skills": "JDS Skill Traits.xlsx",
    "sds_personality": "SDS Personality Traits.xlsx",
}

# Columns listed in the official Data Description document. Used only to WARN
# when something differs; the loader never silently drops or renames raw data.
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "ds_jobs": [
        "reference_no", "company_name", "job_title", "min_experience",
        "avg_salary", "min_salary", "max_salary", "num_of_jobs",
    ],
    "analytics_jobs": [
        "s_no", "experience", "job_description", "job_desig",
        "job_type", "key_skills", "location", "salary",
    ],
    "jds_skills": [
        "id", "big_data_skills", "maths-stats_skills", "coding_skills",
        "ai_and_ml_skills", "dashboard_and_storytelling_skills",
        "salary_hike_high_or_low",
    ],
    "sds_personality": [
        "id", "neuroticism", " extraversion", "openness_to_experience",
        "agreeableness", "conscientiousness", "success_ classification_ high_low",
    ],
}


def configure_logging(level: int = logging.INFO) -> None:
    """Set up a simple console logger once (safe to call repeatedly)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
