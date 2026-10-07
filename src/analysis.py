"""Exploratory analysis for the cleaned Phase 2 datasets."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CLEANED_DIR, OUTPUT_DIR, configure_logging

logger = logging.getLogger(__name__)

FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / "analysis_results"

CLEANED_DATASETS = {
    "ds_jobs": "ds_jobs_clean.csv",
    "analytics_jobs": "analytics_jobs_clean.csv",
    "jds_skills": "jds_skills_clean.csv",
    "sds_personality": "sds_personality_clean.csv",
}

DS_SALARY_COLUMNS = ("min_salary_num", "avg_salary_num", "max_salary_num")
DS_SKILL_COLUMNS = (
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
)
SDS_TRAIT_COLUMNS = (
    "neuroticism",
    "extraversion",
    "openness_to_experience",
    "agreeableness",
    "conscientiousness",
)

SKILL_CATEGORY_MAPPING: dict[str, set[str]] = {
    "Programming": {
        "c", "c#", "c++", "java", "javascript", "typescript", "python",
        "r", "php", "ruby", "scala", "kotlin", "golang", "go", "perl",
        "matlab", "shell scripting", "programming", "coding",
    },
    "Database": {
        "sql", "mysql", "postgresql", "postgres", "oracle", "mongodb",
        "database", "plsql", "pl/sql", "sql server", "nosql", "redis",
    },
    "AI/ML": {
        "machine learning", "deep learning", "artificial intelligence",
        "ai", "ml", "tensorflow", "pytorch", "nlp", "natural language processing",
        "computer vision", "scikit-learn", "neural networks",
    },
    "Statistics": {
        "statistics", "statistical analysis", "probability", "hypothesis testing",
        "regression", "mathematics", "maths", "biostatistics",
    },
    "Big Data": {
        "big data", "hadoop", "spark", "pyspark", "hive", "kafka",
        "mapreduce", "databricks",
    },
    "Visualization": {
        "tableau", "power bi", "powerbi", "qlikview", "qlik sense",
        "matplotlib", "seaborn", "plotly", "visualization", "data visualization",
        "dashboard", "excel",
    },
    "Cloud": {
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "cloud", "cloud computing", "s3", "ec2",
    },
    "Analytics": {
        "data analysis", "data analytics", "business analytics",
        "business intelligence", "analytics", "data mining", "etl",
    },
}

_SKILL_SPLIT_PATTERN = re.compile(r"[,;|]+")
_PNG_EXPORT_AVAILABLE: bool | None = None


def load_cleaned_datasets(
    cleaned_dir: Path = CLEANED_DIR,
) -> dict[str, pd.DataFrame]:
    """Load all four cleaned datasets after checking that every file exists."""
    paths = {name: cleaned_dir / filename for name, filename in CLEANED_DATASETS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required cleaned dataset file(s):\n  - " + "\n  - ".join(missing)
        )

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"Could not load cleaned dataset '{path}': {exc}") from exc
        if frame.empty:
            raise ValueError(f"Cleaned dataset '{path}' contains no records.")
        datasets[name] = frame
        logger.info("Loaded %s: %d rows x %d columns", path.name, *frame.shape)
    return datasets


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required column(s): {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _numeric_summary(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    records = []
    for column in columns:
        values = _numeric(frame, column).dropna()
        records.append({
            "variable": column,
            "count": int(values.count()),
            "mean": float(values.mean()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "standard_deviation": float(values.std()) if len(values) > 1 else None,
            "minimum": float(values.min()) if not values.empty else None,
            "maximum": float(values.max()) if not values.empty else None,
        })
    return pd.DataFrame(records)


def _categorical_frequency(
    frame: pd.DataFrame,
    column: str,
    count_name: str = "posting_count",
) -> pd.DataFrame:
    values = frame[column].astype("string").fillna("<missing>")
    counts = values.value_counts(dropna=False)
    return pd.DataFrame({
        column: counts.index.astype(str),
        count_name: counts.to_numpy(dtype=np.int64),
        "percentage": counts.to_numpy(dtype=float) * 100 / len(frame),
    })


def analyze_ds_job_roles(ds_jobs: pd.DataFrame) -> pd.DataFrame:
    """Rank Data Science job titles by total listed job volume."""
    _require_columns(ds_jobs, ["job_title", "num_of_jobs"], "ds_jobs")
    frame = ds_jobs.assign(
        job_title=ds_jobs["job_title"].astype("string").fillna("<missing>"),
        num_of_jobs=_numeric(ds_jobs, "num_of_jobs"),
    )
    result = (
        frame.groupby("job_title", as_index=False, dropna=False)["num_of_jobs"]
        .sum(min_count=1)
        .rename(columns={"num_of_jobs": "total_jobs"})
        .sort_values(["total_jobs", "job_title"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    return result


def analyze_ds_companies(ds_jobs: pd.DataFrame) -> pd.DataFrame:
    """Rank Data Science companies by total listed job volume."""
    _require_columns(ds_jobs, ["company_name", "num_of_jobs"], "ds_jobs")
    frame = ds_jobs.assign(
        company_name=ds_jobs["company_name"].astype("string").fillna("<missing>"),
        num_of_jobs=_numeric(ds_jobs, "num_of_jobs"),
    )
    return (
        frame.groupby("company_name", as_index=False, dropna=False)["num_of_jobs"]
        .sum(min_count=1)
        .rename(columns={"num_of_jobs": "total_jobs"})
        .sort_values(["total_jobs", "company_name"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )


def analyze_ds_salary(ds_jobs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize salary fields overall and by job title."""
    _require_columns(ds_jobs, ["job_title", *DS_SALARY_COLUMNS], "ds_jobs")
    summary = _numeric_summary(ds_jobs, DS_SALARY_COLUMNS)
    title_frame = ds_jobs.assign(
        job_title=ds_jobs["job_title"].astype("string").fillna("<missing>")
    )
    for column in DS_SALARY_COLUMNS:
        title_frame[column] = _numeric(ds_jobs, column)
    by_title = (
        title_frame.groupby("job_title", dropna=False)[list(DS_SALARY_COLUMNS)]
        .agg(["count", "mean", "median", "min", "max"])
    )
    by_title.columns = [f"{column}_{stat}" for column, stat in by_title.columns]
    by_title = by_title.reset_index().sort_values("job_title").reset_index(drop=True)
    return summary, by_title


def analyze_ds_experience(ds_jobs: pd.DataFrame) -> dict[str, Any]:
    """Describe experience levels and their observed association with salary."""
    _require_columns(ds_jobs, ["min_experience_num", "avg_salary_num"], "ds_jobs")
    experiences = _numeric(ds_jobs, "min_experience_num")
    distribution = (
        experiences.value_counts(dropna=False)
        .rename_axis("min_experience_num")
        .reset_index(name="posting_count")
        .sort_values("min_experience_num", na_position="last")
        .reset_index(drop=True)
    )
    distribution["percentage"] = distribution["posting_count"] * 100 / len(ds_jobs)
    valid_experience = experiences.dropna()
    salaries = _numeric(ds_jobs, "avg_salary_num")
    paired = pd.DataFrame({"experience": experiences, "salary": salaries}).dropna()
    association = (
        paired.groupby("experience", as_index=False)["salary"]
        .agg(posting_count="count", average_salary="mean", median_salary="median")
        .rename(columns={"experience": "min_experience_num"})
        .sort_values("min_experience_num")
        .reset_index(drop=True)
    )
    return {
        "distribution": distribution,
        "summary": {
            "count": int(valid_experience.count()),
            "mean": float(valid_experience.mean()) if not valid_experience.empty else None,
            "median": float(valid_experience.median()) if not valid_experience.empty else None,
            "standard_deviation": float(valid_experience.std()) if len(valid_experience) > 1 else None,
            "minimum": float(valid_experience.min()) if not valid_experience.empty else None,
            "maximum": float(valid_experience.max()) if not valid_experience.empty else None,
            "salary_pair_count": int(len(paired)),
            "pearson_correlation_min_experience_avg_salary": (
                float(paired["experience"].corr(paired["salary"]))
                if len(paired) > 1 and paired["experience"].nunique() > 1
                and paired["salary"].nunique() > 1 else None
            ),
        },
        "salary_by_experience": association,
    }


def analyze_analytics_roles(analytics_jobs: pd.DataFrame) -> pd.DataFrame:
    """Return the 15 most frequent analytics job designations."""
    _require_columns(analytics_jobs, ["job_desig"], "analytics_jobs")
    return _categorical_frequency(analytics_jobs, "job_desig").head(15).reset_index(drop=True)


def analyze_locations(analytics_jobs: pd.DataFrame) -> pd.DataFrame:
    """Return the 15 most frequent analytics job locations."""
    _require_columns(analytics_jobs, ["location"], "analytics_jobs")
    return _categorical_frequency(analytics_jobs, "location").head(15).reset_index(drop=True)


def analyze_job_types(analytics_jobs: pd.DataFrame) -> pd.DataFrame:
    """Calculate posting counts and percentages by analytics job type."""
    _require_columns(analytics_jobs, ["job_type"], "analytics_jobs")
    return _categorical_frequency(analytics_jobs, "job_type")


def analyze_salary_ranges(analytics_jobs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize numeric salary fields, observed ranges and midpoint values."""
    columns = ["salary_min", "salary_max", "salary_midpoint"]
    _require_columns(analytics_jobs, columns, "analytics_jobs")
    frame = analytics_jobs.copy()
    for column in columns:
        frame[column] = _numeric(frame, column)
    range_frame = frame.dropna(subset=["salary_min", "salary_max"])
    range_frame = range_frame.assign(
        salary_range=range_frame.apply(
            lambda row: f"{row['salary_min']:g}-{row['salary_max']:g}", axis=1
        )
    )
    ranges = (
        range_frame.groupby("salary_range", as_index=False)
        .size()
        .rename(columns={"size": "posting_count"})
        .sort_values(["posting_count", "salary_range"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranges["percentage"] = ranges["posting_count"] * 100 / len(analytics_jobs)
    midpoint = frame["salary_midpoint"].dropna()
    midpoint_counts = midpoint.value_counts()
    midpoint_distribution = pd.DataFrame({
        "salary_midpoint": midpoint_counts.index,
        "posting_count": midpoint_counts.to_numpy(dtype=np.int64),
    }).sort_values("salary_midpoint").reset_index(drop=True)
    midpoint_distribution["percentage"] = (
        midpoint_distribution["posting_count"] * 100 / len(analytics_jobs)
    )
    return {
        "summary": _numeric_summary(frame, columns),
        "ranges": ranges,
        "midpoint_distribution": midpoint_distribution,
    }


def analyze_experience(analytics_jobs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Describe numeric analytics experience and midpoint salary by experience."""
    columns = ["experience_min", "experience_max", "experience_midpoint", "salary_midpoint"]
    _require_columns(analytics_jobs, columns, "analytics_jobs")
    frame = analytics_jobs.copy()
    for column in columns:
        frame[column] = _numeric(frame, column)
    midpoint = frame["experience_midpoint"].dropna()
    distribution = (
        midpoint.value_counts()
        .rename_axis("experience_midpoint")
        .reset_index(name="posting_count")
        .sort_values("experience_midpoint")
        .reset_index(drop=True)
    )
    distribution["percentage"] = distribution["posting_count"] * 100 / len(frame)
    by_experience = (
        frame.dropna(subset=["experience_midpoint", "salary_midpoint"])
        .groupby("experience_midpoint", as_index=False)["salary_midpoint"]
        .agg(posting_count="count", average_salary_midpoint="mean",
             median_salary_midpoint="median")
        .sort_values("experience_midpoint")
        .reset_index(drop=True)
    )
    return {
        "summary": _numeric_summary(frame, columns[:3]),
        "distribution": distribution,
        "salary_by_experience": by_experience,
    }


def extract_skill_frequency(key_skills: pd.Series) -> pd.DataFrame:
    """Count each normalized skill once per posting."""
    mentions: dict[str, int] = {}
    for value in key_skills.dropna():
        if not isinstance(value, str):
            continue
        posting_skills = set()
        for part in _SKILL_SPLIT_PATTERN.split(value):
            skill = " ".join(part.split()).casefold().strip(" .…")
            if skill and not re.fullmatch(r"[.…]+", skill):
                posting_skills.add(skill)
        for skill in posting_skills:
            mentions[skill] = mentions.get(skill, 0) + 1

    total_postings = len(key_skills)
    return pd.DataFrame(
        [
            {"skill": skill, "count": count,
             "percentage": count * 100 / total_postings if total_postings else 0.0}
            for skill, count in mentions.items()
        ],
        columns=["skill", "count", "percentage"],
    ).sort_values(["count", "skill"], ascending=[False, True]).reset_index(drop=True)


def analyze_skill_frequency(analytics_jobs: pd.DataFrame) -> pd.DataFrame:
    """Return all observed skill posting counts and posting percentages."""
    _require_columns(analytics_jobs, ["key_skills_clean"], "analytics_jobs")
    return extract_skill_frequency(analytics_jobs["key_skills_clean"])


def _normalize_skill_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def analyze_skill_categories(
    skill_frequency: pd.DataFrame,
    category_mapping: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Assign each skill to at most one configurable taxonomy category."""
    _require_columns(skill_frequency, ["skill", "count", "percentage"], "skill_frequency")
    mapping = category_mapping if category_mapping is not None else SKILL_CATEGORY_MAPPING
    normalized_mapping = {
        _normalize_skill_label(skill): category
        for category, skills in mapping.items()
        for skill in skills
    }
    result = skill_frequency[["skill", "count", "percentage"]].copy()
    result.insert(
        1,
        "category",
        result["skill"].map(
            lambda skill: normalized_mapping.get(_normalize_skill_label(str(skill)), "Other")
        ),
    )
    return result[["skill", "category", "count", "percentage"]]


def _group_labels(target: pd.Series, dataset_name: str) -> pd.Series:
    """Map the cleaned binary high/low target values to display labels."""
    normalized = target.astype("string").str.strip().str.casefold()
    groups = normalized.map({"1": "High", "0": "Low", "high": "High", "low": "Low"})
    invalid = groups.isna()
    if invalid.any():
        values = target[invalid].drop_duplicates().tolist()
        raise ValueError(
            f"{dataset_name} target contains values outside High/Low or 1/0: {values}"
        )
    return groups


def _analyze_binary_groups(
    frame: pd.DataFrame,
    target_column: str,
    score_columns: tuple[str, ...],
    dataset_name: str,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    _require_columns(frame, [target_column, *score_columns], dataset_name)
    groups = _group_labels(frame[target_column], dataset_name)
    group_counts = groups.value_counts().reindex(["High", "Low"], fill_value=0)
    working = frame.loc[:, list(score_columns)].apply(pd.to_numeric, errors="coerce")
    working["group"] = groups
    grouped = working.groupby("group")[list(score_columns)]
    mean = grouped.mean().reindex(["High", "Low"])
    median = grouped.median().reindex(["High", "Low"])
    standard_deviation = grouped.std().reindex(["High", "Low"])
    minimum = grouped.min().reindex(["High", "Low"])
    maximum = grouped.max().reindex(["High", "Low"])
    score_counts = grouped.count().reindex(["High", "Low"])
    mean_comparison = pd.DataFrame({
        "skill": list(score_columns),
        "high_group_mean": mean.loc["High"].to_numpy(),
        "low_group_mean": mean.loc["Low"].to_numpy(),
    })
    mean_comparison["difference"] = (
        mean_comparison["high_group_mean"] - mean_comparison["low_group_mean"]
    )
    detailed = pd.DataFrame({
        "variable": list(score_columns),
        "high_group_count": score_counts.loc["High"].to_numpy(),
        "low_group_count": score_counts.loc["Low"].to_numpy(),
        "high_group_mean": mean.loc["High"].to_numpy(),
        "low_group_mean": mean.loc["Low"].to_numpy(),
        "high_group_median": median.loc["High"].to_numpy(),
        "low_group_median": median.loc["Low"].to_numpy(),
        "high_group_standard_deviation": standard_deviation.loc["High"].to_numpy(),
        "low_group_standard_deviation": standard_deviation.loc["Low"].to_numpy(),
        "high_group_minimum": minimum.loc["High"].to_numpy(),
        "low_group_minimum": minimum.loc["Low"].to_numpy(),
        "high_group_maximum": maximum.loc["High"].to_numpy(),
        "low_group_maximum": maximum.loc["Low"].to_numpy(),
        "difference_in_means": mean.loc["High"].to_numpy() - mean.loc["Low"].to_numpy(),
    })
    return (
        {"High": int(group_counts["High"]), "Low": int(group_counts["Low"])},
        mean_comparison,
        detailed,
    )


def analyze_jds_skills(
    jds_skills: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    """Compare JDS skill score summaries across observed salary-hike groups."""
    counts, comparison, detailed = _analyze_binary_groups(
        jds_skills, "salary_hike_high_or_low", DS_SKILL_COLUMNS, "jds_skills"
    )
    comparison = comparison.rename(columns={"skill": "skill"})
    return counts, comparison, detailed


def analyze_sds_personality(
    sds_personality: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    """Compare personality trait summaries across observed success groups."""
    counts, comparison, detailed = _analyze_binary_groups(
        sds_personality,
        "success_classification_high_low",
        SDS_TRAIT_COLUMNS,
        "sds_personality",
    )
    return counts, comparison.rename(
        columns={
            "skill": "trait",
            "high_group_mean": "high_success_mean",
            "low_group_mean": "low_success_mean",
        }
    ), detailed


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _save_table(frame: pd.DataFrame, filename: str, table_names: list[str]) -> None:
    path = ANALYSIS_RESULTS_DIR / filename
    frame.to_csv(path, index=False)
    table_names.append(filename)


def _save_figure(figure: go.Figure, filename: str, figure_names: list[str]) -> None:
    global _PNG_EXPORT_AVAILABLE
    html_path = FIGURES_DIR / f"{filename}.html"
    figure.write_html(html_path, include_plotlyjs="cdn")
    figure_names.append(html_path.name)

    if _PNG_EXPORT_AVAILABLE is False:
        return
    try:
        figure.write_image(FIGURES_DIR / f"{filename}.png")
        _PNG_EXPORT_AVAILABLE = True
        figure_names.append(f"{filename}.png")
    except Exception as exc:
        if _PNG_EXPORT_AVAILABLE is not False:
            logger.warning(
                "Static PNG export is unavailable; interactive HTML figures were saved: %s",
                exc,
            )
        _PNG_EXPORT_AVAILABLE = False


def _build_figures(
    ds_roles: pd.DataFrame,
    ds_companies: pd.DataFrame,
    ds_jobs: pd.DataFrame,
    analytics_jobs: pd.DataFrame,
    analytics_roles: pd.DataFrame,
    locations: pd.DataFrame,
    salary_ranges: pd.DataFrame,
    skill_frequency: pd.DataFrame,
    skill_categories: pd.DataFrame,
    jds_comparison: pd.DataFrame,
    sds_comparison: pd.DataFrame,
    figures: list[str],
) -> None:
    if not ds_roles.empty:
        _save_figure(
            px.bar(ds_roles.head(10).sort_values("total_jobs"),
                   x="total_jobs", y="job_title", orientation="h",
                   title="Top 10 Data Science Job Titles by Total Jobs"),
            "ds_top_10_job_titles", figures,
        )
    if not ds_companies.empty:
        _save_figure(
            px.bar(ds_companies.head(10).sort_values("total_jobs"),
                   x="total_jobs", y="company_name", orientation="h",
                   title="Top 10 Companies by Data Science Job Volume"),
            "ds_top_10_companies", figures,
        )

    salary_columns = [column for column in DS_SALARY_COLUMNS if column in ds_jobs]
    salary_data = ds_jobs[salary_columns].apply(pd.to_numeric, errors="coerce").melt(
        var_name="salary_metric", value_name="salary"
    )
    _save_figure(
        px.histogram(salary_data.dropna(), x="salary", color="salary_metric",
                     barmode="overlay", opacity=0.65, title="Data Science Salary Distribution"),
        "ds_salary_distribution", figures,
    )
    experience = _numeric(ds_jobs, "min_experience_num")
    _save_figure(
        px.histogram(pd.DataFrame({"min_experience_num": experience.dropna()}),
                     x="min_experience_num", title="Data Science Minimum Experience Distribution"),
        "ds_experience_distribution", figures,
    )
    if not analytics_roles.empty:
        _save_figure(
            px.bar(analytics_roles.sort_values("posting_count"),
                   x="posting_count", y="job_desig", orientation="h",
                   title="Top 15 Analytics Job Roles"),
            "analytics_top_15_roles", figures,
        )
    if not locations.empty:
        _save_figure(
            px.bar(locations.sort_values("posting_count"),
                   x="posting_count", y="location", orientation="h",
                   title="Top 15 Analytics Job Locations"),
            "analytics_top_15_locations", figures,
        )
    if not salary_ranges.empty:
        _save_figure(
            px.bar(salary_ranges.head(15).sort_values("posting_count"),
                   x="posting_count", y="salary_range", orientation="h",
                   title="Analytics Job Postings by Salary Range"),
            "analytics_salary_range_distribution", figures,
        )
    if not skill_frequency.empty:
        _save_figure(
            px.bar(skill_frequency.head(15).sort_values("count"),
                   x="count", y="skill", orientation="h",
                   title="Top 15 Skills Mentioned in Analytics Job Postings"),
            "analytics_top_15_skills", figures,
        )
    category_demand = (
        skill_categories.groupby("category", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=True)
    )
    if not category_demand.empty:
        _save_figure(
            px.bar(category_demand, x="count", y="category", orientation="h",
                   title="Analytics Skill Mentions by Category"),
            "analytics_skill_category_demand", figures,
        )
    if not jds_comparison.empty:
        comparison_long = jds_comparison.melt(
            id_vars="skill",
            value_vars=["high_group_mean", "low_group_mean"],
            var_name="group",
            value_name="mean_score",
        )
        _save_figure(
            px.bar(comparison_long, x="skill", y="mean_score", color="group",
                   barmode="group", title="JDS Mean Skill Scores by Salary-Hike Group"),
            "jds_high_low_skill_comparison", figures,
        )
    if not sds_comparison.empty:
        comparison_long = sds_comparison.melt(
            id_vars="trait",
            value_vars=["high_success_mean", "low_success_mean"],
            var_name="group",
            value_name="mean_score",
        )
        _save_figure(
            px.bar(comparison_long, x="trait", y="mean_score", color="group",
                   barmode="group", title="SDS Mean Personality Scores by Success Group"),
            "sds_high_low_personality_comparison", figures,
        )


def main() -> None:
    """Run all Phase 3 analyses and save tables, figures, and a JSON summary."""
    configure_logging()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_cleaned_datasets()
    ds_jobs = datasets["ds_jobs"]
    analytics_jobs = datasets["analytics_jobs"]
    jds_skills = datasets["jds_skills"]
    sds_personality = datasets["sds_personality"]

    table_names: list[str] = []
    figure_names: list[str] = []
    ds_roles = analyze_ds_job_roles(ds_jobs)
    ds_companies = analyze_ds_companies(ds_jobs)
    ds_salary, ds_salary_by_title = analyze_ds_salary(ds_jobs)
    ds_experience = analyze_ds_experience(ds_jobs)

    analytics_roles = analyze_analytics_roles(analytics_jobs)
    locations = analyze_locations(analytics_jobs)
    job_types = analyze_job_types(analytics_jobs)
    salary_analysis = analyze_salary_ranges(analytics_jobs)
    analytics_experience = analyze_experience(analytics_jobs)
    skill_frequency = analyze_skill_frequency(analytics_jobs)
    skill_categories = analyze_skill_categories(skill_frequency)
    jds_counts, jds_comparison, jds_detailed = analyze_jds_skills(jds_skills)
    sds_counts, sds_comparison, sds_detailed = analyze_sds_personality(sds_personality)

    tables = {
        "ds_top_roles.csv": ds_roles,
        "ds_top_companies.csv": ds_companies,
        "ds_salary_summary.csv": ds_salary,
        "ds_salary_by_job_title.csv": ds_salary_by_title,
        "ds_experience_summary.csv": pd.DataFrame([ds_experience["summary"]]),
        "ds_experience_distribution.csv": ds_experience["distribution"],
        "ds_salary_by_experience.csv": ds_experience["salary_by_experience"],
        "analytics_top_roles.csv": analytics_roles,
        "analytics_top_locations.csv": locations,
        "analytics_job_types.csv": job_types,
        "analytics_salary_summary.csv": salary_analysis["summary"],
        "analytics_salary_ranges.csv": salary_analysis["ranges"],
        "analytics_salary_midpoint_distribution.csv": salary_analysis["midpoint_distribution"],
        "analytics_experience_summary.csv": analytics_experience["summary"],
        "analytics_experience_distribution.csv": analytics_experience["distribution"],
        "analytics_experience_salary.csv": analytics_experience["salary_by_experience"],
        "skill_frequency.csv": skill_frequency,
        "skill_category_frequency.csv": skill_categories,
        "jds_group_comparison.csv": jds_comparison,
        "jds_group_statistics.csv": jds_detailed,
        "sds_group_comparison.csv": sds_comparison,
        "sds_group_statistics.csv": sds_detailed,
    }
    for filename, frame in tables.items():
        _save_table(frame, filename, table_names)

    category_totals = (
        skill_categories.groupby("category", as_index=False)["count"]
        .sum()
        .sort_values(["count", "category"], ascending=[False, True])
    )
    _build_figures(
        ds_roles, ds_companies, ds_jobs, analytics_jobs, analytics_roles, locations,
        salary_analysis["ranges"], skill_frequency,
        skill_categories, jds_comparison, sds_comparison, figure_names,
    )

    summary = {
        "datasets": {
            name: {"file": CLEANED_DATASETS[name], "row_count": int(len(frame))}
            for name, frame in datasets.items()
        },
        "total_records_analyzed": int(sum(len(frame) for frame in datasets.values())),
        "major_analysis_outputs": {
            "ds_top_roles": ds_roles.head(10),
            "ds_top_companies": ds_companies.head(10),
            "ds_salary_summary": ds_salary,
            "ds_experience_summary": ds_experience["summary"],
            "analytics_top_roles": analytics_roles,
            "analytics_top_locations": locations,
            "analytics_job_types": job_types,
            "analytics_salary_summary": salary_analysis["summary"],
            "analytics_experience_summary": analytics_experience["summary"],
            "skill_count": int(len(skill_frequency)),
            "top_skills": skill_frequency.head(15),
            "skill_category_demand": category_totals,
            "jds_group_counts": jds_counts,
            "jds_group_comparison": jds_comparison,
            "sds_group_counts": sds_counts,
            "sds_group_comparison": sds_comparison,
        },
        "generated_figures": figure_names,
        "generated_result_tables": table_names,
    }
    summary_path = ANALYSIS_RESULTS_DIR / "eda_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    chart_count = sum(name.endswith(".html") for name in figure_names)
    logger.info("EDA complete: %d datasets, %d records, %d skills, %d charts, %d result tables.",
                len(datasets), summary["total_records_analyzed"], len(skill_frequency),
                chart_count, len(table_names))
    print("\n=== Phase 3 EDA summary ===")
    print(f"Datasets analyzed: {len(datasets)}")
    print(f"Records analyzed: {summary['total_records_analyzed']}")
    print(f"Skills identified: {len(skill_frequency)}")
    print(f"Charts generated: {chart_count}")
    print(f"Result tables generated: {len(table_names)}")
    if not ds_roles.empty:
        top = ds_roles.iloc[0]
        print(f"Top Data Science job title by total jobs: {top['job_title']} ({top['total_jobs']:g})")
    if not ds_companies.empty:
        top = ds_companies.iloc[0]
        print(f"Top Data Science company by total jobs: {top['company_name']} ({top['total_jobs']:g})")
    if not analytics_roles.empty:
        top = analytics_roles.iloc[0]
        print(f"Most frequent analytics designation: {top['job_desig']} ({top['posting_count']} postings)")
    if not skill_frequency.empty:
        top = skill_frequency.iloc[0]
        print(f"Most-mentioned extracted skill: {top['skill']} ({top['count']} postings, {top['percentage']:.2f}%)")
    print(f"Results saved to: {ANALYSIS_RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
