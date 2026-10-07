"""Streamlit dashboard for the Data Scientist Career Intelligence project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS / "analysis_results"
MODEL_DIR = OUTPUTS / "model_results"
CLEANED_DIR = OUTPUTS / "cleaned_data"

st.set_page_config(
    page_title="Data Scientist Career Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = (
    "Executive Overview",
    "Job Market",
    "Skill Demand",
    "Technical Skills & Salary Hike",
    "Personality & Senior Success",
    "Model Performance",
    "Career Recommendations",
    "Data Quality",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #f7fafc, #ffffff);
        border: 1px solid #e5ebf1;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(21, 41, 58, 0.04);
    }
    [data-testid="stMetricLabel"] {color: #526579;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .page-kicker {color: #50718f; font-size: .8rem; font-weight: 700;
                  letter-spacing: .12em; text-transform: uppercase;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _read_csv(path_string: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path_string)


@st.cache_data(show_spinner=False)
def _read_json(path_string: str, modified_time: float) -> dict[str, Any]:
    del modified_time
    with open(path_string, "r", encoding="utf-8") as source:
        return json.load(source)


def load_csv(path: Path, label: str | None = None) -> pd.DataFrame | None:
    """Load a CSV with caching and show a user-friendly message on failure."""
    display_name = label or path.name
    if not path.is_file():
        st.warning(f"Expected data file is missing: **{display_name}** (`{path}`).")
        return None
    try:
        return _read_csv(str(path), path.stat().st_mtime)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.warning(f"Could not load **{display_name}**: {exc}")
        return None


def load_json(path: Path, label: str | None = None) -> dict[str, Any] | None:
    """Load a JSON file with caching and show a user-friendly message on failure."""
    display_name = label or path.name
    if not path.is_file():
        st.warning(f"Expected data file is missing: **{display_name}** (`{path}`).")
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        st.warning(f"Could not load **{display_name}**: {exc}")
        return None


def _title(page: str, description: str) -> None:
    st.markdown(f'<div class="page-kicker">{page}</div>', unsafe_allow_html=True)
    st.title("Data Scientist Career Intelligence")
    st.caption(description)


def _heading(text: str, help_text: str | None = None) -> None:
    st.subheader(text, help=help_text)


def _records_from_summary(summary: dict[str, Any] | None, name: str) -> int | None:
    if not summary:
        return None
    value = summary.get("datasets", {}).get(name, {}).get("row_count")
    return int(value) if value is not None else None


def _model_selected(summary: dict[str, Any] | None, group: str) -> str:
    if not summary:
        return "Unavailable"
    return str(summary.get(group, {}).get("selected_model", "Unavailable"))


def _metric_value(frame: pd.DataFrame | None, column: str, variable: str | None = None) -> float | None:
    if frame is None or column not in frame.columns:
        return None
    selected = frame
    if variable is not None and "variable" in frame.columns:
        selected = frame.loc[frame["variable"] == variable]
    if selected.empty:
        return None
    value = pd.to_numeric(pd.Series([selected.iloc[0][column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _format_number(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:,.{decimals}f}"


def _safe_bar(
    frame: pd.DataFrame | None,
    x: str,
    y: str,
    title: str,
    *,
    orientation: str = "v",
    color: str | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    if frame is None or frame.empty or x not in frame.columns or y not in frame.columns:
        st.info(f"{title} is unavailable because the required analysis data is missing.")
        return
    figure = px.bar(
        frame,
        x=x,
        y=y,
        color=color if color in frame.columns else None,
        orientation=orientation,
        labels=labels,
        title=title,
        template="plotly_white",
    )
    figure.update_layout(margin=dict(l=8, r=8, t=56, b=8), legend_title_text="")
    st.plotly_chart(figure, width="stretch")


def _load_overview_data() -> dict[str, Any]:
    summary = load_json(ANALYSIS_DIR / "eda_summary.json", "EDA summary")
    model_summary = load_json(MODEL_DIR / "model_summary.json", "Model summary")
    return {"eda": summary, "models": model_summary}


def render_executive_overview() -> None:
    _title(
        "Executive Overview",
        "Data-driven insights into job-market demand, technical skills, and career success",
    )
    data = _load_overview_data()
    eda = data["eda"]
    models = data["models"]

    role = load_csv(ANALYSIS_DIR / "ds_top_roles.csv", "Data Science top roles")
    locations = load_csv(ANALYSIS_DIR / "analytics_top_locations.csv", "Analytics top locations")
    skills = load_csv(ANALYSIS_DIR / "skill_frequency.csv", "Skill frequency")
    jds_count = _records_from_summary(eda, "jds_skills")
    sds_count = _records_from_summary(eda, "sds_personality")
    ds_count = _records_from_summary(eda, "ds_jobs")
    analytics_count = _records_from_summary(eda, "analytics_jobs")
    skill_count = len(skills) if skills is not None else None

    metrics = st.columns(5)
    for column, label, value in zip(
        metrics,
        ("Data Science job records", "Analytics postings", "JDS records",
         "SDS records", "Normalized skills"),
        (ds_count, analytics_count, jds_count, sds_count, skill_count),
    ):
        column.metric(label, _format_number(value))

    st.divider()
    findings = st.columns(5)
    top_role = (
        f"{role.iloc[0]['job_title']} ({_format_number(role.iloc[0]['total_jobs'])})"
        if role is not None and not role.empty else "Unavailable"
    )
    top_location = (
        f"{locations.iloc[0]['location']} ({_format_number(locations.iloc[0]['posting_count'])})"
        if locations is not None and not locations.empty else "Unavailable"
    )
    top_skill = (
        f"{skills.iloc[0]['skill']} ({_format_number(skills.iloc[0]['count'])})"
        if skills is not None and not skills.empty else "Unavailable"
    )
    for column, label, value in zip(
        findings,
        ("Top Data Science role", "Top analytics location", "Top demanded skill",
         "Selected JDS model", "Selected SDS model"),
        (top_role, top_location, top_skill,
         _model_selected(models, "JDS"), _model_selected(models, "SDS")),
    ):
        column.metric(label, value)

    st.divider()
    _heading("At-a-glance evidence")
    tabs = st.tabs(["Job market", "Skills", "Outcome models"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        if role is not None and not role.empty:
            col1.write(
                f"**{role.iloc[0]['job_title']}** is the leading Data Science job title "
                "by total listed jobs in the cleaned job dataset."
            )
        if locations is not None and not locations.empty:
            col2.write(
                f"**{locations.iloc[0]['location']}** is the leading analytics location "
                "by posting count in the analyzed location table."
            )
    with tabs[1]:
        if skills is not None and not skills.empty:
            st.write(
                f"The highest-frequency extracted skill is **{skills.iloc[0]['skill']}**, "
                f"mentioned in {_format_number(skills.iloc[0]['count'])} postings "
                f"({_format_number(skills.iloc[0]['percentage'], 2)}% of postings)."
            )
        st.caption("Skill mentions are derived from extracted job-posting text.")
    with tabs[2]:
        st.write(
            f"Selected cross-validated models: JDS — **{_model_selected(models, 'JDS')}**; "
            f"SDS — **{_model_selected(models, 'SDS')}**."
        )
        st.caption("Model estimates describe the supplied samples and are not causal findings.")


def _filter_analytics(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    filtered = frame.copy()
    controls = st.columns(4)
    filters_active = False
    if "location" in filtered:
        options = sorted(filtered["location"].dropna().astype(str).unique())
        selected = controls[0].multiselect("Location", options, key="market_location")
        if selected:
            filters_active = True
            filtered = filtered[filtered["location"].astype(str).isin(selected)]
    if "job_desig" in filtered:
        options = sorted(filtered["job_desig"].dropna().astype(str).unique())
        selected = controls[1].multiselect("Role", options, key="market_role")
        if selected:
            filters_active = True
            filtered = filtered[filtered["job_desig"].astype(str).isin(selected)]
    if "job_type" in filtered:
        options = sorted(filtered["job_type"].dropna().astype(str).unique())
        selected = controls[2].multiselect("Job type", options, key="market_job_type")
        if selected:
            filters_active = True
            filtered = filtered[filtered["job_type"].astype(str).isin(selected)]
    if "experience_midpoint" in filtered:
        experience = pd.to_numeric(filtered["experience_midpoint"], errors="coerce")
        finite = experience.dropna()
        if not finite.empty:
            low, high = float(finite.min()), float(finite.max())
            selected_range = controls[3].slider(
                "Experience midpoint",
                min_value=low,
                max_value=high,
                value=(low, high),
                key="market_experience",
            )
            filters_active = filters_active or selected_range != (low, high)
            filtered = filtered.loc[
                experience.between(selected_range[0], selected_range[1]) | experience.isna()
            ]
    return filtered, filters_active


def _filtered_frequency(
    frame: pd.DataFrame | None,
    column: str,
    count_name: str,
    limit: int = 15,
) -> pd.DataFrame | None:
    if frame is None or column not in frame.columns:
        return None
    return (
        frame.groupby(column, dropna=False).size()
        .reset_index(name=count_name)
        .sort_values(count_name, ascending=False)
        .head(limit)
    )


def render_job_market() -> None:
    _title("Job Market", "Explore job volume, hiring locations, salaries, and experience ranges.")
    ds_roles = load_csv(ANALYSIS_DIR / "ds_top_roles.csv")
    ds_companies = load_csv(ANALYSIS_DIR / "ds_top_companies.csv")
    analytics_roles = load_csv(ANALYSIS_DIR / "analytics_top_roles.csv")
    locations = load_csv(ANALYSIS_DIR / "analytics_top_locations.csv")
    salary_summary = load_csv(ANALYSIS_DIR / "analytics_salary_summary.csv")
    ds_experience = load_csv(ANALYSIS_DIR / "ds_experience_distribution.csv")
    analytics_experience = load_csv(ANALYSIS_DIR / "analytics_experience_distribution.csv")
    salary_experience = load_csv(ANALYSIS_DIR / "analytics_experience_salary.csv")
    ds_salary_experience = load_csv(ANALYSIS_DIR / "ds_salary_by_experience.csv")
    cleaned_ds = load_csv(CLEANED_DIR / "ds_jobs_clean.csv", "Cleaned Data Science jobs")
    cleaned_analytics = load_csv(CLEANED_DIR / "analytics_jobs_clean.csv", "Cleaned analytics jobs")

    filtered = cleaned_analytics
    filters_active = False
    if cleaned_analytics is not None:
        st.markdown("#### Filtered analytics postings")
        filtered, filters_active = _filter_analytics(cleaned_analytics)
        st.caption(f"Showing {len(filtered):,} of {len(cleaned_analytics):,} postings.")
        if not filtered.empty:
            result = (
                filtered.groupby("job_desig", dropna=False)
                .size().reset_index(name="posting_count")
                .sort_values("posting_count", ascending=False).head(15)
            )
            with st.expander("Top roles for current filters", expanded=False):
                st.dataframe(result, width="stretch", hide_index=True)

    if filters_active:
        analytics_roles = _filtered_frequency(filtered, "job_desig", "posting_count")
        locations = _filtered_frequency(filtered, "location", "posting_count")
        salary_midpoints = pd.to_numeric(filtered["salary_midpoint"], errors="coerce")
        salary_midpoints = salary_midpoints.dropna().value_counts().rename_axis(
            "salary_midpoint"
        ).reset_index(name="posting_count")
        experience_values = pd.to_numeric(filtered["experience_midpoint"], errors="coerce")
        analytics_experience = experience_values.dropna().value_counts().rename_axis(
            "experience_midpoint"
        ).reset_index(name="posting_count")
        salary_experience = (
            filtered.assign(
                experience_midpoint=pd.to_numeric(filtered["experience_midpoint"], errors="coerce"),
                salary_midpoint=pd.to_numeric(filtered["salary_midpoint"], errors="coerce"),
            )
            .dropna(subset=["experience_midpoint", "salary_midpoint"])
            .groupby("experience_midpoint", as_index=False)["salary_midpoint"]
            .mean()
            .rename(columns={"salary_midpoint": "average_salary_midpoint"})
        )
    else:
        salary_midpoints = load_csv(ANALYSIS_DIR / "analytics_salary_midpoint_distribution.csv")

    first_row = st.columns(2)
    with first_row[0]:
        _safe_bar(ds_roles, "total_jobs", "job_title", "Top Data Science roles",
                  orientation="h", labels={"total_jobs": "Total jobs", "job_title": "Role"})
    with first_row[1]:
        _safe_bar(analytics_roles, "posting_count", "job_desig", "Top Analytics roles",
                  orientation="h", labels={"posting_count": "Postings", "job_desig": "Role"})
    second_row = st.columns(2)
    with second_row[0]:
        _safe_bar(ds_companies.head(15) if ds_companies is not None else None,
                  "total_jobs", "company_name", "Top companies by job volume",
                  orientation="h", labels={"total_jobs": "Total jobs", "company_name": "Company"})
    with second_row[1]:
        _safe_bar(locations, "posting_count", "location", "Top analytics locations",
                  orientation="h", labels={"posting_count": "Postings", "location": "Location"})

    st.divider()
    tabs = st.tabs(["Salary distribution", "Experience distribution", "Salary vs experience"])
    with tabs[0]:
        if salary_summary is not None:
            salary_columns = st.columns(2)
            if cleaned_ds is not None and "avg_salary_num" in cleaned_ds.columns:
                values = pd.to_numeric(cleaned_ds["avg_salary_num"], errors="coerce").dropna()
                figure = px.histogram(
                    pd.DataFrame({"Average salary": values}),
                    x="Average salary",
                    title="Data Science average salary distribution",
                    template="plotly_white",
                    labels={"Average salary": "Average salary"},
                )
                figure.update_layout(margin=dict(l=8, r=8, t=56, b=8))
                salary_columns[0].plotly_chart(figure, width="stretch")
            else:
                salary_columns[0].info("Data Science salary distribution is unavailable.")
            with salary_columns[1]:
                _safe_bar(salary_midpoints, "salary_midpoint", "posting_count",
                          "Analytics postings by salary midpoint" +
                          (" (filtered)" if filters_active else ""),
                          labels={"salary_midpoint": "Salary midpoint", "posting_count": "Postings"})
            st.dataframe(salary_summary, width="stretch", hide_index=True)
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            _safe_bar(ds_experience, "min_experience_num", "posting_count",
                      "Data Science minimum experience distribution",
                      labels={"min_experience_num": "Minimum experience", "posting_count": "Records"})
        with col2:
            _safe_bar(analytics_experience, "experience_midpoint", "posting_count",
                      "Analytics experience midpoint distribution" +
                      (" (filtered)" if filters_active else ""),
                      labels={"experience_midpoint": "Experience midpoint", "posting_count": "Postings"})
    with tabs[2]:
        salary_columns = st.columns(2)
        with salary_columns[0]:
            _safe_bar(ds_salary_experience, "min_experience_num", "average_salary",
                      "Data Science average salary by minimum experience",
                      labels={"min_experience_num": "Minimum experience",
                              "average_salary": "Average salary"})
        with salary_columns[1]:
            _safe_bar(salary_experience, "experience_midpoint", "average_salary_midpoint",
                      "Average analytics salary midpoint by experience" +
                      (" (filtered)" if filters_active else ""),
                      labels={"experience_midpoint": "Experience midpoint",
                              "average_salary_midpoint": "Average salary midpoint"})
        st.caption("These grouped summaries show association in the observed postings, not causation.")


def render_skill_demand() -> None:
    _title("Skill Demand", "Skills are counted from the normalized skill text in analytics postings.")
    frequency = load_csv(ANALYSIS_DIR / "skill_frequency.csv", "Complete skill frequency")
    category = load_csv(ANALYSIS_DIR / "skill_category_frequency.csv", "Skill category frequency")
    if frequency is None:
        return
    frequency["count"] = pd.to_numeric(frequency["count"], errors="coerce")
    frequency["percentage"] = pd.to_numeric(frequency["percentage"], errors="coerce")
    st.caption("Skill mentions are based on extracted job-posting text. Each skill count reflects postings mentioning that normalized skill.")
    top = frequency.sort_values(["count", "skill"], ascending=[False, True]).head(15)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        _safe_bar(top.sort_values("count"), "count", "skill", "Top 15 demanded skills",
                  orientation="h", labels={"count": "Postings mentioning skill", "skill": "Skill"})
    with col2:
        if category is not None:
            demand = (
                category.groupby("category", as_index=False)["count"].sum()
                .sort_values("count", ascending=True)
            )
            _safe_bar(demand, "count", "category", "Skill category demand",
                      orientation="h", labels={"count": "Skill mentions", "category": "Category"})
    st.metric("Normalized skills identified", f"{len(frequency):,}")
    with st.expander("Full skill frequency table", expanded=True):
        st.dataframe(
            frequency.sort_values(["count", "skill"], ascending=[False, True]),
            width="stretch",
            hide_index=True,
            height=480,
        )
    if category is not None:
        with st.expander("Skill categorization table"):
            st.dataframe(category, width="stretch", hide_index=True, height=360)


def _group_sizes_from_summary(summary: dict[str, Any] | None, group: str) -> dict[str, int]:
    if not summary:
        return {}
    counts = summary.get(group, {}).get("class_counts", {})
    return {str(name): int(count) for name, count in counts.items()}


def _jds_group_sizes() -> dict[str, int]:
    source = load_csv(CLEANED_DIR / "jds_skills_clean.csv", "Cleaned JDS data")
    if source is None or "salary_hike_high_or_low" not in source:
        return {}
    target = pd.to_numeric(source["salary_hike_high_or_low"], errors="coerce")
    return {"High": int((target == 1).sum()), "Low": int((target == 0).sum())}


def _sds_group_sizes() -> dict[str, int]:
    source = load_csv(CLEANED_DIR / "sds_personality_clean.csv", "Cleaned SDS data")
    if source is None or "success_classification_high_low" not in source:
        return {}
    target = pd.to_numeric(source["success_classification_high_low"], errors="coerce")
    return {"High": int((target == 1).sum()), "Low": int((target == 0).sum())}


def _group_comparison_chart(
    frame: pd.DataFrame | None,
    name_column: str,
    high_column: str,
    low_column: str,
    title: str,
) -> None:
    if frame is None or frame.empty:
        st.info(f"{title} is unavailable because the comparison table is missing.")
        return
    long = frame.melt(
        id_vars=name_column,
        value_vars=[high_column, low_column],
        var_name="group",
        value_name="mean",
    )
    long["group"] = long["group"].map({high_column: "High", low_column: "Low"})
    figure = px.bar(
        long,
        x=name_column,
        y="mean",
        color="group",
        barmode="group",
        title=title,
        template="plotly_white",
        labels={name_column: "Feature", "mean": "Mean score", "group": "Outcome group"},
    )
    figure.update_layout(margin=dict(l=8, r=8, t=56, b=8))
    st.plotly_chart(figure, width="stretch")


def render_technical_skills() -> None:
    _title(
        "Technical Skills & Salary Hike",
        "Descriptive group comparisons and model-associated feature influence for the JDS sample.",
    )
    comparison = load_csv(ANALYSIS_DIR / "jds_group_comparison.csv")
    statistics = load_csv(ANALYSIS_DIR / "jds_group_statistics.csv")
    importance = load_csv(MODEL_DIR / "jds_feature_importance.csv")
    model_summary = load_json(MODEL_DIR / "model_summary.json", "Model summary")
    sizes = _jds_group_sizes()
    high, low = st.columns(2)
    high.metric("High salary-hike records", _format_number(sizes.get("High")))
    low.metric("Low salary-hike records", _format_number(sizes.get("Low")))
    _group_comparison_chart(
        comparison, "skill", "high_group_mean", "low_group_mean",
        "Mean technical skill scores by salary-hike group",
    )
    if statistics is not None:
        st.markdown("#### Means, medians, and mean differences")
        st.dataframe(statistics, width="stretch", hide_index=True)
    if importance is not None:
        selected = _model_selected(model_summary, "JDS")
        selected_importance = importance.loc[importance["model"] == selected].sort_values(
            "importance", ascending=False
        )
        if selected_importance.empty:
            selected_importance = importance.sort_values("importance", ascending=False)
        _safe_bar(
            selected_importance, "importance", "feature",
            f"Model-associated feature importance — {selected}",
            orientation="h",
            labels={"importance": "Importance / absolute standardized coefficient",
                    "feature": "Feature"},
        )
    st.caption("Group differences and feature importance are associations within this sample, not causal effects.")
    st.metric("Selected JDS model", _model_selected(model_summary, "JDS"))


def render_personality_success() -> None:
    _title(
        "Personality & Senior Success",
        "Descriptive trait comparisons and model-associated influence in the SDS sample.",
    )
    comparison = load_csv(ANALYSIS_DIR / "sds_group_comparison.csv")
    statistics = load_csv(ANALYSIS_DIR / "sds_group_statistics.csv")
    importance = load_csv(MODEL_DIR / "sds_feature_importance.csv")
    model_summary = load_json(MODEL_DIR / "model_summary.json", "Model summary")
    sizes = _sds_group_sizes()
    high, low = st.columns(2)
    high.metric("High success records", _format_number(sizes.get("High")))
    low.metric("Low success records", _format_number(sizes.get("Low")))
    _group_comparison_chart(
        comparison, "trait", "high_success_mean", "low_success_mean",
        "Mean personality scores by success group",
    )
    if comparison is not None:
        _safe_bar(
            comparison.sort_values("difference", ascending=False),
            "difference", "trait", "Difference in group means (High − Low)",
            orientation="h",
            labels={"difference": "Mean difference", "trait": "Personality trait"},
        )
    if statistics is not None:
        with st.expander("Mean and median group statistics"):
            st.dataframe(statistics, width="stretch", hide_index=True)
    if importance is not None:
        selected = _model_selected(model_summary, "SDS")
        selected_importance = importance.loc[importance["model"] == selected].sort_values(
            "importance", ascending=False
        )
        if selected_importance.empty:
            selected_importance = importance.sort_values("importance", ascending=False)
        _safe_bar(
            selected_importance, "importance", "feature",
            f"Features associated with model predictions — {selected}",
            orientation="h",
            labels={"importance": "Feature importance", "feature": "Trait"},
        )
    st.caption("Observed group differences and model importance do not establish causation.")
    st.metric("Selected SDS model", _model_selected(model_summary, "SDS"))


def _performance_long(comparison: pd.DataFrame, selected: str) -> pd.DataFrame:
    records = []
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    for _, row in comparison.iterrows():
        for metric, label in metric_labels.items():
            mean = row.get(f"mean_{metric}")
            deviation = row.get(f"std_{metric}")
            if pd.isna(mean):
                continue
            records.append({
                "model": row["model"],
                "metric": label,
                "mean": mean,
                "standard_deviation": deviation,
                "selected": "Selected model" if row["model"] == selected else "Other models",
            })
    return pd.DataFrame(records)


def _render_model_comparison(
    title: str,
    comparison: pd.DataFrame | None,
    selected: str,
) -> None:
    st.markdown(f"#### {title}")
    if comparison is None:
        return
    display = comparison.copy()
    display["selected"] = display["model"].eq(selected).map(
        {True: "Selected", False: ""}
    )
    display = display.rename(columns={
        "model": "Model",
        "mean_accuracy": "Accuracy — mean",
        "std_accuracy": "Accuracy — SD",
        "mean_precision": "Precision — mean",
        "std_precision": "Precision — SD",
        "mean_recall": "Recall — mean",
        "std_recall": "Recall — SD",
        "mean_f1": "F1 — mean",
        "std_f1": "F1 — SD",
        "mean_roc_auc": "ROC-AUC — mean",
        "std_roc_auc": "ROC-AUC — SD",
        "selected": "Selected model",
    })
    st.dataframe(display, width="stretch", hide_index=True)
    long = _performance_long(comparison, selected)
    if long.empty:
        st.info(f"No valid metric values are available for {title}.")
        return
    figure = px.bar(
        long,
        x="metric",
        y="mean",
        color="model",
        barmode="group",
        error_y="standard_deviation",
        title=f"{title}: cross-validated mean metrics",
        template="plotly_white",
        labels={"metric": "Metric", "mean": "Mean score", "model": "Model"},
    )
    figure.update_yaxes(range=[0, 1])
    figure.update_layout(margin=dict(l=8, r=8, t=56, b=8))
    st.plotly_chart(figure, width="stretch")
    st.caption(f"Selected model from model_summary.json: **{selected}**.")


def render_model_performance() -> None:
    _title("Model Performance", "Cross-validated baseline estimates for both binary outcome datasets.")
    models = load_json(MODEL_DIR / "model_summary.json", "Model summary")
    jds = load_csv(MODEL_DIR / "jds_model_comparison.csv")
    sds = load_csv(MODEL_DIR / "sds_model_comparison.csv")
    if models:
        tabs = st.tabs(["JDS salary hike", "SDS success"])
        with tabs[0]:
            _render_model_comparison("JDS models", jds, _model_selected(models, "JDS"))
        with tabs[1]:
            _render_model_comparison("SDS models", sds, _model_selected(models, "SDS"))
    st.info("Cross-validation metrics are estimates and may vary because the datasets are small.")


def _recommendations(
    frequency: pd.DataFrame | None,
    category: pd.DataFrame | None,
    jds_stats: pd.DataFrame | None,
    sds_stats: pd.DataFrame | None,
    jds_importance: pd.DataFrame | None,
    sds_importance: pd.DataFrame | None,
    model_summary: dict[str, Any] | None,
) -> pd.DataFrame:
    """Derive concise recommendations from observed demand, group deltas, and importance."""
    rows: list[dict[str, str]] = []
    if frequency is not None and not frequency.empty:
        top_skills = frequency.sort_values("count", ascending=False).head(15)
        for _, skill_row in top_skills.iterrows():
            skill = str(skill_row["skill"])
            count = int(skill_row["count"])
            percentage = float(skill_row["percentage"])
            rows.append({
                "factor": skill,
                "market_demand": (
                    f"Top-15 skill by frequency: {count:,} postings ({percentage:.2f}%)."
                ),
                "outcome_association": "No direct skill-level outcome comparison is available.",
                "evidence_source": "skill_frequency.csv",
                "recommendation": (
                    f"Consider reviewing {skill} in light of its observed posting mentions; "
                    "validate relevance against the roles you are targeting."
                ),
            })

    for label, stats, importance, group_key, feature_name in (
        ("JDS", jds_stats, jds_importance, "JDS", "feature"),
        ("SDS", sds_stats, sds_importance, "SDS", "feature"),
    ):
        if stats is None:
            continue
        selected_model = _model_selected(model_summary, group_key)
        selected_importance = (
            importance.loc[importance["model"] == selected_model].copy()
            if importance is not None and "model" in importance.columns else pd.DataFrame()
        )
        if not selected_importance.empty:
            selected_importance = selected_importance.sort_values("importance", ascending=False)
        importance_rank = {
            str(row[feature_name]): rank
            for rank, (_, row) in enumerate(selected_importance.iterrows(), start=1)
        }
        for _, row in stats.iterrows():
            feature = str(row.get("feature", row.get("variable", "")))
            if not feature:
                continue
            high_mean = pd.to_numeric(pd.Series([row.get("high_group_mean")]), errors="coerce").iloc[0]
            low_mean = pd.to_numeric(pd.Series([row.get("low_group_mean")]), errors="coerce").iloc[0]
            difference = high_mean - low_mean if pd.notna(high_mean) and pd.notna(low_mean) else None
            rank = importance_rank.get(feature)
            demand_label = "Not assessed as job-posting demand in this table."
            if label == "JDS":
                category_map = {
                    "big_data_skills": "Big Data",
                    "maths-stats_skills": "Statistics",
                    "coding_skills": "Programming",
                    "ai_and_ml_skills": "AI/ML",
                    "dashboard_and_storytelling_skills": "Visualization",
                }
                category_name = category_map.get(feature)
                if category is not None and category_name and "category" in category.columns:
                    mention_total = pd.to_numeric(
                        category.loc[category["category"] == category_name, "count"],
                        errors="coerce",
                    ).sum()
                    demand_label = (
                        f"{category_name} category has {mention_total:,.0f} skill mentions "
                        "in the extracted-posting taxonomy."
                    )
            association = (
                f"High-group mean exceeds Low by {difference:.3f} in the supplied sample."
                if difference is not None and difference > 0
                else (
                    f"High-group mean is below Low by {abs(difference):.3f} in the supplied sample."
                    if difference is not None and difference < 0
                    else "No positive High-versus-Low mean difference is observed."
                )
            )
            importance_evidence = (
                f"Ranked #{rank} by importance for selected {selected_model}."
                if rank is not None else "Selected-model feature importance is unavailable."
            )
            if difference is not None and difference > 0 and rank is not None and rank <= 2:
                recommendation = (
                    f"Consider prioritizing further review of {feature}: the group mean "
                    "difference and selected-model importance both support investigating it; "
                    "this is not evidence of a causal effect."
                )
            elif difference is not None and difference > 0:
                recommendation = (
                    f"Consider developing or evaluating {feature}; its score is associated "
                    "with the higher outcome group in this sample, without establishing causality."
                )
            else:
                recommendation = (
                    f"Treat {feature} as a descriptive consideration; this sample does not "
                    "show a positive High-group mean difference."
                )
            rows.append({
                "factor": feature,
                "market_demand": demand_label,
                "outcome_association": f"{association} {importance_evidence}",
                "evidence_source": (
                    f"{label.lower()}_group_statistics.csv; "
                    f"{label.lower()}_feature_importance.csv; model_summary.json"
                ),
                "recommendation": recommendation,
            })
    return pd.DataFrame(
        rows,
        columns=[
            "factor", "market_demand", "outcome_association",
            "evidence_source", "recommendation",
        ],
    )


def render_recommendations() -> None:
    _title(
        "Career Recommendations",
        "Evidence-linked considerations based on the supplied postings and outcome samples.",
    )
    frequency = load_csv(ANALYSIS_DIR / "skill_frequency.csv")
    category = load_csv(ANALYSIS_DIR / "skill_category_frequency.csv")
    jds_stats = load_csv(ANALYSIS_DIR / "jds_group_statistics.csv")
    sds_stats = load_csv(ANALYSIS_DIR / "sds_group_statistics.csv")
    jds_importance = load_csv(MODEL_DIR / "jds_feature_importance.csv")
    sds_importance = load_csv(MODEL_DIR / "sds_feature_importance.csv")
    summary = load_json(MODEL_DIR / "model_summary.json", "Model summary")
    recommendations = _recommendations(
        frequency, category, jds_stats, sds_stats,
        jds_importance, sds_importance, summary,
    )
    st.caption(
        "Recommendations are descriptive and evidence-linked. Group differences and model "
        "feature importance indicate associations, not causes or guaranteed outcomes."
    )
    market, technical, professional = st.tabs([
        "A. Market priorities",
        "B. Technical development priorities",
        "C. Professional development considerations",
    ])
    if recommendations.empty:
        st.warning("Recommendation evidence is unavailable because required result files are missing.")
        return
    market_rows = recommendations[
        recommendations["evidence_source"].eq("skill_frequency.csv")
    ]
    technical_rows = recommendations[
        recommendations["evidence_source"].str.startswith("jds_")
    ]
    professional_rows = recommendations[
        recommendations["evidence_source"].str.startswith("sds_")
    ]
    with market:
        st.write("Highest-frequency extracted skills; rankings come directly from the saved frequency table.")
        st.dataframe(market_rows, width="stretch", hide_index=True)
    with technical:
        st.write("Technical scores compared across salary-hike groups alongside selected-model importance.")
        st.dataframe(technical_rows, width="stretch", hide_index=True)
    with professional:
        st.write("Personality trait comparisons are sample associations, not prescriptions or causal explanations.")
        st.dataframe(professional_rows, width="stretch", hide_index=True)


def _cleaning_quality_table(report: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    if not report:
        return pd.DataFrame()
    target_specs = {
        "ds_jobs": None,
        "analytics_jobs": None,
        "jds_skills": ("salary_hike_high_or_low", [0, 1]),
        "sds_personality": ("success_classification_high_low", [0, 1]),
    }
    for dataset, record in report.items():
        before_missing = record.get("missing_values_before_cleaning", {})
        after_missing = record.get("missing_values_after_cleaning", {})
        failures = record.get("parsing_failures", [])
        unexpected = record.get("unexpected_target_values", [])
        target_status = "Not applicable"
        spec = target_specs.get(dataset)
        if spec:
            target_name, valid = spec
            unexpected_values = []
            if target_name in after_missing:
                unexpected_values = [
                    value for value in unexpected if value not in valid
                ]
            target_status = (
                "Valid 0/1" if not unexpected_values
                else f"Unexpected values: {unexpected_values}"
            )
        rows.append({
            "dataset": dataset,
            "original_rows": record.get("original_row_count"),
            "cleaned_rows": record.get("cleaned_row_count"),
            "missing_before": int(sum(before_missing.values())),
            "missing_after": int(sum(after_missing.values())),
            "duplicates_before": record.get("duplicate_rows"),
            "duplicates_after": record.get("duplicate_rows_after_cleaning"),
            "parsing_failures": sum(
                int(item.get("row_count", 1)) if isinstance(item, dict) else 1
                for item in failures
            ),
            "unexpected_target_values": len(unexpected),
            "target_validation": target_status,
        })
    return pd.DataFrame(rows)


def render_data_quality() -> None:
    _title("Data Quality", "Cleaning checks and known limitations from the Phase 2 cleaning report.")
    report = load_json(OUTPUTS / "cleaning_report.json", "Phase 2 cleaning report")
    if report is None:
        st.info("Data-quality details are unavailable until the cleaning report is present.")
        return
    table = _cleaning_quality_table(report)
    if table.empty:
        st.warning("The cleaning report did not contain dataset-level records.")
        return
    st.dataframe(table, width="stretch", hide_index=True)
    total_failures = int(table["parsing_failures"].sum())
    total_unexpected = int(table["unexpected_target_values"].sum())
    metrics = st.columns(3)
    metrics[0].metric("Parsing failures", f"{total_failures:,}")
    metrics[1].metric("Unexpected target values", f"{total_unexpected:,}")
    metrics[2].metric("Records retained", f"{int(table['cleaned_rows'].fillna(0).sum()):,}")

    targets = st.tabs(["JDS target validation", "SDS target validation", "Limitations"])
    with targets[0]:
        _show_target_validation(
            CLEANED_DIR / "jds_skills_clean.csv",
            "salary_hike_high_or_low",
            "JDS",
        )
    with targets[1]:
        _show_target_validation(
            CLEANED_DIR / "sds_personality_clean.csv",
            "success_classification_high_low",
            "SDS",
        )
    with targets[2]:
        st.markdown(
            "- JDS and SDS samples are small; model estimates may be unstable.\n"
            "- Model results have not been externally validated.\n"
            "- Skill extraction depends on the text available in job postings.\n"
            "- Differences and model features indicate association, not causation.\n"
            "- Generalization beyond these supplied datasets is not established."
        )
    with st.expander("Cleaning transformations"):
        for dataset, record in report.items():
            st.markdown(f"**{dataset}**")
            transformations = record.get("transformations_performed", [])
            if transformations:
                for description in transformations:
                    st.write(f"- {description}")
            else:
                st.write("No transformations recorded.")


def _show_target_validation(path: Path, target_column: str, dataset_name: str) -> None:
    frame = load_csv(path, f"{dataset_name} cleaned target data")
    if frame is None:
        return
    if target_column not in frame.columns:
        st.warning(f"{dataset_name} target column `{target_column}` is missing.")
        return
    target = pd.to_numeric(frame[target_column], errors="coerce")
    invalid = frame.loc[target.isna() | ~target.isin([0, 1]), target_column]
    counts = target.value_counts(dropna=False).rename_axis("target_value").reset_index(name="records")
    valid = len(invalid) == 0 and target.notna().all()
    st.metric("Target check", "Valid binary 0/1" if valid else "Invalid target values found")
    st.dataframe(counts, width="stretch", hide_index=True)
    if not valid:
        st.warning(f"Values outside 0/1 or missing: {invalid.astype(str).unique().tolist()}")


def main() -> None:
    """Render the selected one of eight dashboard pages."""
    with st.sidebar:
        st.markdown("### Data Scientist\nCareer Intelligence")
        st.caption("Evidence-led market and career analytics")
        selected_page = st.radio(
            "Navigate",
            PAGES,
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Analysis and model outputs are loaded from the project’s generated files.")

    pages = {
        "Executive Overview": render_executive_overview,
        "Job Market": render_job_market,
        "Skill Demand": render_skill_demand,
        "Technical Skills & Salary Hike": render_technical_skills,
        "Personality & Senior Success": render_personality_success,
        "Model Performance": render_model_performance,
        "Career Recommendations": render_recommendations,
        "Data Quality": render_data_quality,
    }
    pages[selected_page]()


if __name__ == "__main__":
    main()
