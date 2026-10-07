"""Cross-validated baseline models for the JDS and SDS cleaned datasets."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CLEANED_DIR, OUTPUT_DIR, configure_logging

logger = logging.getLogger(__name__)

MODEL_RESULTS_DIR = OUTPUT_DIR / "model_results"
JDS_FILENAME = "jds_skills_clean.csv"
SDS_FILENAME = "sds_personality_clean.csv"

JDS_TARGET = "salary_hike_high_or_low"
JDS_FEATURES = (
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
)
SDS_TARGET = "success_classification_high_low"
SDS_FEATURES = (
    "neuroticism",
    "extraversion",
    "openness_to_experience",
    "agreeableness",
    "conscientiousness",
)
MODEL_NAMES = ("Logistic Regression", "Decision Tree", "Random Forest")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_model_data(
    cleaned_dir: Path = CLEANED_DIR,
) -> dict[str, pd.DataFrame]:
    """Load the cleaned JDS and SDS datasets, raising a clear missing-file error."""
    paths = {
        "jds": cleaned_dir / JDS_FILENAME,
        "sds": cleaned_dir / SDS_FILENAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required cleaned model dataset file(s):\n  - "
            + "\n  - ".join(missing)
        )

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"Could not load model dataset '{path}': {exc}") from exc
        if frame.empty:
            raise ValueError(f"Model dataset '{path}' contains no records.")
        datasets[name] = frame
        logger.info("Loaded %s: %d rows x %d columns", path.name, *frame.shape)
    return datasets


def validate_target(target: pd.Series, target_name: str = "target") -> tuple[Any, Any]:
    """Validate a non-missing binary target and return (negative, positive) labels."""
    if target.isna().any():
        raise ValueError(f"Target '{target_name}' still contains missing values.")
    classes = list(pd.unique(target))
    if len(classes) != 2:
        raise ValueError(
            f"Target '{target_name}' must be binary; found {len(classes)} classes: {classes}"
        )

    one_label = next(
        (label for label in classes if str(label).strip().casefold() in {"1", "true", "high", "yes"}),
        None,
    )
    if one_label is not None:
        positive = one_label
        negative = next(label for label in classes if label != positive)
    else:
        try:
            ordered = sorted(classes)
        except TypeError as exc:
            raise ValueError(
                f"Target '{target_name}' has incomparable class labels: {classes}"
            ) from exc
        negative, positive = ordered
    return negative, positive


def _prepare_xy(
    frame: pd.DataFrame,
    target_name: str,
    features: tuple[str, ...],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series, Any, Any]:
    required = [target_name, *features]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"{dataset_name} is missing required column(s): {missing_columns}")

    target_missing = frame[target_name].isna()
    if target_missing.any():
        logger.warning(
            "%s: excluding %d record(s) with a missing target.",
            dataset_name, int(target_missing.sum()),
        )
        frame = frame.loc[~target_missing].copy()

    negative, positive = validate_target(frame[target_name], target_name)
    feature_frame = frame.loc[:, list(features)].copy()
    for column in features:
        if not is_numeric_dtype(feature_frame[column].dtype):
            raise TypeError(
                f"{dataset_name} feature '{column}' must be numeric; "
                f"found dtype {feature_frame[column].dtype}."
            )
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="raise")
    infinite = np.isinf(feature_frame.to_numpy(dtype=float))
    if infinite.any():
        logger.warning(
            "%s: replacing %d infinite feature value(s) with missing values for imputation.",
            dataset_name, int(infinite.sum()),
        )
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    all_missing = feature_frame.columns[feature_frame.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            f"{dataset_name} has feature(s) with no observed values: {all_missing}"
        )
    missing_values = int(feature_frame.isna().sum().sum())
    if missing_values:
        logger.info(
            "%s: %d missing feature value(s) will be median-imputed within each CV fold.",
            dataset_name, missing_values,
        )
    return feature_frame, frame[target_name].copy(), negative, positive


def _model_pipeline(estimator: BaseEstimator, scale: bool) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def build_jds_models() -> dict[str, BaseEstimator]:
    """Build scaled logistic regression and unscaled tree classifiers for JDS."""
    return {
        "Logistic Regression": _model_pipeline(
            LogisticRegression(max_iter=2000, random_state=42), scale=True
        ),
        "Decision Tree": _model_pipeline(
            DecisionTreeClassifier(random_state=42), scale=False
        ),
        "Random Forest": _model_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
            scale=False,
        ),
    }


def build_sds_models() -> dict[str, BaseEstimator]:
    """Build scaled logistic regression and unscaled tree classifiers for SDS."""
    return {
        "Logistic Regression": _model_pipeline(
            LogisticRegression(max_iter=2000, random_state=42), scale=True
        ),
        "Decision Tree": _model_pipeline(
            DecisionTreeClassifier(random_state=42), scale=False
        ),
        "Random Forest": _model_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
            scale=False,
        ),
    }


def get_logistic_coefficients(
    model: BaseEstimator,
    feature_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Return absolute coefficients for the fitted standardized logistic model."""
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise TypeError("Expected a fitted model pipeline with a 'classifier' step.")
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("The fitted pipeline does not contain LogisticRegression.")
    if classifier.coef_.shape[0] != 1:
        raise ValueError("Coefficient extraction expects a binary logistic model.")
    return pd.DataFrame({
        "feature": list(feature_names),
        "importance": np.abs(classifier.coef_[0]),
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def get_tree_feature_importance(
    model: BaseEstimator,
    feature_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Return fitted decision-tree or random-forest feature importances."""
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise TypeError("Expected a fitted model pipeline with a 'classifier' step.")
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier)):
        raise TypeError("The fitted pipeline does not contain a supported tree model.")
    return pd.DataFrame({
        "feature": list(feature_names),
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _positive_probability(model: BaseEstimator, X: pd.DataFrame, positive: Any) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if positive not in classes:
        return None
    return probabilities[:, classes.index(positive)]


def cross_validate_model(
    model: BaseEstimator,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    negative_label: Any,
    positive_label: Any,
) -> dict[str, Any]:
    """Fit fold-local models and collect metrics, out-of-fold predictions, and influence."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_values = y.to_numpy()
    oof_predictions = np.empty(len(y), dtype=y_values.dtype)
    oof_probabilities = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict[str, float | None]] = []
    importance_frames: list[pd.DataFrame] = []
    has_probabilities = True

    for train_indices, test_indices in splitter.split(X, y_values):
        fitted = clone(model)
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y_values[train_indices]
        y_test = y_values[test_indices]
        fitted.fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        oof_predictions[test_indices] = predictions

        scores: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            )),
            "roc_auc": None,
        }
        probabilities = _positive_probability(fitted, X_test, positive_label)
        if probabilities is None:
            has_probabilities = False
        else:
            oof_probabilities[test_indices] = probabilities
            if len(pd.unique(y_test)) == 2:
                binary_y = (y_test == positive_label).astype(int)
                scores["roc_auc"] = float(roc_auc_score(binary_y, probabilities))
        fold_metrics.append(scores)

        if model_name == "Logistic Regression":
            importance = get_logistic_coefficients(fitted, list(X.columns))
        else:
            importance = get_tree_feature_importance(fitted, list(X.columns))
        importance["model"] = model_name
        importance_frames.append(importance)

    metric_summary: dict[str, float | None] = {}
    for metric in METRICS:
        values = np.array(
            [fold[metric] for fold in fold_metrics if fold[metric] is not None],
            dtype=float,
        )
        metric_summary[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        metric_summary[f"std_{metric}"] = (
            float(values.std(ddof=1)) if len(values) > 1 else None
        )

    importances = pd.concat(importance_frames, ignore_index=True)
    mean_importances = (
        importances.groupby(["feature", "model"], as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    probabilities_available = bool(
        has_probabilities and np.isfinite(oof_probabilities).all()
    )
    return {
        "model": model_name,
        "metrics": metric_summary,
        "fold_metrics": fold_metrics,
        "predictions": oof_predictions,
        "probabilities": oof_probabilities if probabilities_available else None,
        "feature_importance": mean_importances,
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator],
    negative_label: Any,
    positive_label: Any,
) -> dict[str, Any]:
    """Cross-validate every model and select by ROC-AUC, falling back to F1."""
    class_counts = y.value_counts()
    smallest_class = int(class_counts.min())
    if smallest_class < 2:
        raise ValueError(
            "Stratified cross-validation requires at least two records in each class; "
            f"class counts are {class_counts.to_dict()}."
        )
    n_splits = min(5, smallest_class)
    if n_splits < 5:
        logger.warning(
            "Using %d stratified folds because the smallest class has only %d records.",
            n_splits, smallest_class,
        )

    results = {
        name: cross_validate_model(
            model, name, X, y, n_splits, negative_label, positive_label
        )
        for name, model in models.items()
    }
    rows = []
    for name, result in results.items():
        row: dict[str, Any] = {"model": name}
        row.update(result["metrics"])
        rows.append(row)
    comparison = pd.DataFrame(rows, columns=[
        "model",
        "mean_accuracy", "std_accuracy",
        "mean_precision", "std_precision",
        "mean_recall", "std_recall",
        "mean_f1", "std_f1",
        "mean_roc_auc", "std_roc_auc",
    ])
    roc_available = comparison["mean_roc_auc"].notna().any()
    criterion = "mean_roc_auc" if roc_available else "mean_f1"
    if not roc_available:
        logger.warning(
            "Fold-level ROC-AUC could not be calculated; selecting by cross-validated F1."
        )
    ranked = comparison.sort_values(
        [criterion, "mean_f1"], ascending=[False, False], na_position="last"
    )
    selected_model = str(ranked.iloc[0]["model"])
    logger.info(
        "Selected %s using cross-validated %s.",
        selected_model, "ROC-AUC" if roc_available else "F1",
    )
    return {
        "comparison": comparison,
        "results": results,
        "selected_model": selected_model,
        "n_splits": n_splits,
        "selection_metric": criterion,
    }


def _group_statistics(
    frame: pd.DataFrame,
    target_name: str,
    features: tuple[str, ...],
    negative_label: Any,
    positive_label: Any,
) -> pd.DataFrame:
    """Return observed count, mean, and median for each feature by target group."""
    rows = []
    high_group = frame.loc[frame[target_name] == positive_label]
    low_group = frame.loc[frame[target_name] == negative_label]
    for feature in features:
        high = pd.to_numeric(high_group[feature], errors="coerce").dropna()
        low = pd.to_numeric(low_group[feature], errors="coerce").dropna()
        high_mean = float(high.mean()) if not high.empty else None
        low_mean = float(low.mean()) if not low.empty else None
        high_median = float(high.median()) if not high.empty else None
        low_median = float(low.median()) if not low.empty else None
        rows.append({
            "feature": feature,
            "high_group_count": int(high.count()),
            "low_group_count": int(low.count()),
            "high_group_mean": high_mean,
            "low_group_mean": low_mean,
            "mean_difference": (
                high_mean - low_mean
                if high_mean is not None and low_mean is not None else None
            ),
            "high_group_median": high_median,
            "low_group_median": low_median,
            "median_difference": (
                high_median - low_median
                if high_median is not None and low_median is not None else None
            ),
        })
    return pd.DataFrame(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _json_value(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_value(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def save_model_results(results: dict[str, Any]) -> list[str]:
    """Save each provided model result independently, logging any write failures."""
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for filename, output in results.items():
        path = MODEL_RESULTS_DIR / filename
        try:
            if filename.endswith(".json"):
                path.write_text(
                    json.dumps(_json_value(output), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            elif isinstance(output, pd.DataFrame):
                output.to_csv(path, index=False)
            else:
                raise TypeError(f"Unsupported output type for {filename}: {type(output).__name__}")
            saved.append(filename)
            logger.info("Saved %s", path)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Could not save model result '%s': %s", path, exc)
    return saved


def _serialize_model_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": result["model"],
        "metrics": result["metrics"],
        "fold_metrics": result["fold_metrics"],
    }


def _assemble_outputs(
    prefix: str,
    frame: pd.DataFrame,
    target_name: str,
    features: tuple[str, ...],
    negative_label: Any,
    positive_label: Any,
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_name = evaluation["selected_model"]
    selected = evaluation["results"][selected_name]
    labels = [negative_label, positive_label]
    matrix = confusion_matrix(
        frame[target_name].to_numpy(),
        selected["predictions"],
        labels=labels,
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ).rename_axis("actual")
    roc_frame: pd.DataFrame | None = None
    probabilities = selected["probabilities"]
    if probabilities is not None and len(pd.unique(frame[target_name])) == 2:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            frame[target_name].to_numpy(),
            probabilities,
            pos_label=positive_label,
        )
        roc_frame = pd.DataFrame({
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        })
    else:
        logger.warning(
            "%s ROC data unavailable: the selected model did not produce complete "
            "out-of-fold probabilities.",
            prefix.upper(),
        )

    importance_frames = [
        result["feature_importance"] for result in evaluation["results"].values()
    ]
    importances = pd.concat(importance_frames, ignore_index=True)[
        ["feature", "importance", "model"]
    ].sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)
    group_stats = _group_statistics(
        frame, target_name, features, negative_label, positive_label
    )
    selected_importance = (
        selected["feature_importance"]
        .sort_values("importance", ascending=False)
        .to_dict(orient="records")
    )
    model_summary = {
        "selected_model": selected_name,
        "selection_metric": evaluation["selection_metric"],
        "cross_validated_metrics": selected["metrics"],
        "feature_importance_summary": selected_importance,
        "class_counts": {
            "High": int((frame[target_name] == positive_label).sum()),
            "Low": int((frame[target_name] == negative_label).sum()),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": int(evaluation["n_splits"]),
            "shuffle": True,
            "random_state": 42,
        },
        "all_model_metrics": {
            name: _serialize_model_result(result)
            for name, result in evaluation["results"].items()
        },
        "group_statistics": group_stats,
    }
    output_files: dict[str, Any] = {
        f"{prefix}_model_comparison.csv": evaluation["comparison"],
        f"{prefix}_feature_importance.csv": importances,
        f"{prefix}_confusion_matrix.csv": matrix_frame.reset_index(),
        f"{prefix}_group_statistics.csv": group_stats,
    }
    if roc_frame is not None:
        output_files[f"{prefix}_roc_curve.csv"] = roc_frame
    return output_files, model_summary


def main() -> None:
    """Run stratified cross-validation and save Phase 4 metrics and diagnostics."""
    configure_logging()
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_model_data()
    jds_frame = datasets["jds"]
    sds_frame = datasets["sds"]

    jds_X, jds_y, jds_negative, jds_positive = _prepare_xy(
        jds_frame, JDS_TARGET, JDS_FEATURES, "JDS"
    )
    sds_X, sds_y, sds_negative, sds_positive = _prepare_xy(
        sds_frame, SDS_TARGET, SDS_FEATURES, "SDS"
    )
    jds_models = build_jds_models()
    sds_models = build_sds_models()

    logger.info("Evaluating JDS models on %d records.", len(jds_X))
    jds_evaluation = evaluate_models(
        jds_X, jds_y, jds_models, jds_negative, jds_positive
    )
    logger.info("Evaluating SDS models on %d records.", len(sds_X))
    sds_evaluation = evaluate_models(
        sds_X, sds_y, sds_models, sds_negative, sds_positive
    )

    jds_outputs, jds_summary = _assemble_outputs(
        "jds", jds_frame.loc[jds_y.index], JDS_TARGET, JDS_FEATURES,
        jds_negative, jds_positive, jds_evaluation,
    )
    sds_outputs, sds_summary = _assemble_outputs(
        "sds", sds_frame.loc[sds_y.index], SDS_TARGET, SDS_FEATURES,
        sds_negative, sds_positive, sds_evaluation,
    )
    limitations = [
        "Both datasets are small; model estimates may be unstable.",
        "Cross-validation metrics vary across folds and should be interpreted with their reported fold standard deviations.",
        "Results describe associations within the supplied samples and do not establish causation.",
        "Results should not be generalized to all data scientists without external validation.",
        "Feature influence is model-associated, not causal.",
    ]
    summary = {
        "JDS": jds_summary,
        "SDS": sds_summary,
        "limitations": limitations,
        "interpretation": (
            "Reported feature importance describes features associated with each "
            "model's predictions; it does not establish causal effects."
        ),
    }
    files: dict[str, Any] = {
        **jds_outputs,
        **sds_outputs,
        "model_summary.json": summary,
    }
    saved = save_model_results(files)

    print("\n=== Phase 4 model evaluation summary ===")
    print(f"Models evaluated per target: {len(MODEL_NAMES)}")
    print(f"JDS cross-validation: StratifiedKFold ({jds_evaluation['n_splits']} folds)")
    print(f"SDS cross-validation: StratifiedKFold ({sds_evaluation['n_splits']} folds)")
    print(f"Best JDS model: {jds_evaluation['selected_model']}")
    print(f"Best SDS model: {sds_evaluation['selected_model']}")
    print(f"Output files saved: {len(saved)}")
    print(f"Results directory: {MODEL_RESULTS_DIR}")
    print("Limitations: small samples, fold variability, and no external validation; "
          "feature influence is not causal.")


if __name__ == "__main__":
    main()
