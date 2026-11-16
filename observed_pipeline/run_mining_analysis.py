#!/usr/bin/env python3
"""Run reproducible EDA, mining, and baseline evaluation on local prepared tables."""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = os.environ.get("OBSERVED_PIPELINE_OUTPUT_DIR")
if not OUTPUT_PATH:
    raise RuntimeError("Set OBSERVED_PIPELINE_OUTPUT_DIR or run through run_pipeline.py; outputs must stay outside this source package.")
DATA = Path(OUTPUT_PATH).resolve()
if DATA.is_relative_to(ROOT):
    raise RuntimeError("OBSERVED_PIPELINE_OUTPUT_DIR must be outside observed_pipeline.")
OUT = DATA / "mining"
LABEL_COLUMN = "goal_met"
SEED = 20261031

CATEGORICAL = ["typeExam", "mode", "difficulty_level"]
NUMERIC = [
    "assessment_duration_minutes",
    "prior_same_skill_attempt_count",
    "prior_other_skill_attempt_count",
    "prior_same_skill_score_last",
    "prior_same_skill_score_mean_last3",
    "prior_same_skill_score_std_last5",
    "prior_same_skill_score_trend_last3_vs_prev3",
    "days_since_prior_same_skill_attempt",
    "prior_same_skill_attempts_30d",
    "prior_same_skill_attempts_31_90d",
    "prior_same_assessment_count",
    "prior_ap_events_30d",
    "prior_ap_events_31_90d",
    "prior_ap_amount_per_event_90d",
    "prior_ap_improvement_rate_90d",
    "days_since_prior_ap_event",
]
FEATURES = CATEGORICAL + NUMERIC
HISTORY = ["typeExam", "mode", "assessment_duration_minutes"] + [
    "prior_same_skill_attempt_count",
    "prior_other_skill_attempt_count",
    "prior_same_skill_score_last",
    "prior_same_skill_score_mean_last3",
    "prior_same_skill_score_std_last5",
    "prior_same_skill_score_trend_last3_vs_prev3",
    "days_since_prior_same_skill_attempt",
    "prior_same_skill_attempts_30d",
    "prior_same_skill_attempts_31_90d",
    "prior_same_assessment_count",
]
AP_FEATURES = [
    "prior_ap_events_30d",
    "prior_ap_events_31_90d",
    "prior_ap_amount_per_event_90d",
    "prior_ap_improvement_rate_90d",
    "days_since_prior_ap_event",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA / "attempt_features.csv")
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True)
    for col in NUMERIC + ["real_score", "ielts_band_score", "target_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    observed_labels = df["real_score"].notna() & df["target_score"].notna()
    if not observed_labels.any():
        raise ValueError("No rows contain both observed real_score and target_score.")
    df = df.loc[observed_labels & df["started_at"].notna()].copy()
    if df.empty:
        raise ValueError("No labeled rows have a source started_at timestamp for time-based evaluation.")
    df[LABEL_COLUMN] = (df["real_score"] >= df["target_score"]).astype(int)
    return df


def time_split(df: pd.DataFrame):
    ordered = df.sort_values(["started_at", "attempt_key"], kind="mergesort")
    t1 = ordered.iloc[round(0.70 * len(ordered))]["started_at"]
    t2 = ordered.iloc[round(0.85 * len(ordered))]["started_at"]
    train = df[df.started_at < t1].copy()
    valid = df[(df.started_at >= t1) & (df.started_at < t2)].copy()
    test = df[df.started_at >= t2].copy()
    return train, valid, test, t1, t2


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_eda(df, train, valid, test, t1, t2):
    OUT.mkdir(parents=True, exist_ok=True)
    skill = (
        df.groupby("typeExam").agg(
            attempts=("attempt_key", "count"),
            accounts=("student_key", "nunique"),
            mean_real_score=("real_score", "mean"),
            median_real_score=("real_score", "median"),
            goal_met_count=(LABEL_COLUMN, "sum"),
            goal_met_rate=(LABEL_COLUMN, "mean"),
            difficulty_levels=("difficulty_level", "nunique"),
        ).reset_index()
    )
    skill.to_csv(OUT / "profile_by_skill.csv", index=False)
    difficulty = (
        df.groupby("difficulty_level").agg(
            attempts=("attempt_key", "count"),
            accounts=("student_key", "nunique"),
            mean_real_score=("real_score", "mean"),
            goal_met_count=(LABEL_COLUMN, "sum"),
            goal_met_rate=(LABEL_COLUMN, "mean"),
        ).reset_index()
    )
    difficulty.to_csv(OUT / "profile_by_difficulty.csv", index=False)

    prior_bins = pd.cut(
        df["prior_same_skill_attempt_count"],
        bins=[-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"],
    )
    ap_90 = df["prior_ap_events_30d"].fillna(0) + df["prior_ap_events_31_90d"].fillna(0)
    trend = df["prior_same_skill_score_trend_last3_vs_prev3"]
    trend_group = pd.Series("no_trend", index=df.index)
    trend_group.loc[trend < -0.01] = "decreasing"
    trend_group.loc[trend > 0.01] = "increasing"
    trend_group.loc[trend.between(-0.01, 0.01)] = "stable"
    conditional_specs = [
        ("skill", df["typeExam"]),
        ("difficulty", df["difficulty_level"]),
        ("prior_same_skill_attempts", prior_bins),
        ("prior_ap_events_90d", pd.cut(ap_90, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])),
        ("prior_score_trend", trend_group),
    ]
    base_rate = df[LABEL_COLUMN].mean()
    rows = []
    for feature_name, series in conditional_specs:
        for category, indexes in series.groupby(series, dropna=False, observed=False).groups.items():
            subset = df.loc[indexes]
            if not len(subset):
                continue
            rate = subset[LABEL_COLUMN].mean()
            rows.append({
                "feature": feature_name,
                "category": str(category),
                "attempts": len(subset),
                "accounts": subset["student_key"].nunique(),
                "goal_met_count": int(subset[LABEL_COLUMN].sum()),
                "p_goal_met": rate,
                "overall_p_goal_met": base_rate,
                "lift_vs_overall": rate / base_rate if base_rate else np.nan,
            })
    pd.DataFrame(rows).sort_values(["feature", "category"]).to_csv(
        OUT / "conditional_probabilities.csv", index=False
    )

    correlations = []
    corr_features = [c for c in NUMERIC if df[c].nunique(dropna=True) > 1]
    for scope, part in [("all", df)] + [(skill_name, group) for skill_name, group in df.groupby("typeExam")]:
        for feature in corr_features:
            pair = part[[feature, "real_score"]].dropna()
            if len(pair) < 3 or pair[feature].nunique() < 2 or pair.real_score.nunique() < 2:
                continue
            pr = pearsonr(pair[feature], pair.real_score)
            sr = spearmanr(pair[feature], pair.real_score)
            correlations.append({
                "scope": scope,
                "feature": feature,
                "n": len(pair),
                "pearson_r": pr.statistic,
                "pearson_p_unadjusted": pr.pvalue,
                "spearman_rho": sr.statistic,
                "spearman_p_unadjusted": sr.pvalue,
                "pearson_r_squared_univariate": pr.statistic ** 2,
            })
    pd.DataFrame(correlations).to_csv(OUT / "feature_target_correlations.csv", index=False)

    split_rows = []
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        warm = part[part["prior_same_skill_score_last"].notna()]
        split_rows.append({
            "partition": name,
            "start_min": part["started_at"].min(),
            "start_max": part["started_at"].max(),
            "attempts": len(part),
            "accounts": part["student_key"].nunique(),
            "warm_start_attempts": len(warm),
            "goal_met_count": int(part[LABEL_COLUMN].sum()),
            "goal_not_met_count": int((1 - part[LABEL_COLUMN]).sum()),
        })
    pd.DataFrame(split_rows).to_csv(OUT / "temporal_split_profile.csv", index=False)
    metadata = {
        "classification_label": "goal_met = 1 when real_score >= target_score; otherwise goal_not_met",
        "overall_goal_met_count": int(df[LABEL_COLUMN].sum()),
        "overall_goal_not_met_count": int((1 - df[LABEL_COLUMN]).sum()),
        "overall_goal_met_rate": float(df[LABEL_COLUMN].mean()),
        "split_method": "global chronological 70/15/15 by started_at; equal timestamps remain in the same partition",
        "train_validation_cutoff_utc": t1,
        "validation_test_cutoff_utc": t2,
        "partitions": {name: len(part) for name, part in [("train", train), ("validation", valid), ("test", test)]},
        "warm_start_rows": int(df["prior_same_skill_score_last"].notna().sum()),
        "rows_with_unknown_mode": int(df["mode"].isna().sum()),
        "features": FEATURES,
        "target": "real_score",
        "pipeline_seed": SEED,
        "account_cluster_bootstrap_iterations": 500,
        "frequent_itemset_min_support_count": max(30, math.ceil(0.02 * len(train))),
        "frequent_itemset_min_distinct_accounts": 10,
        "sequential_min_support_count": 10,
        "sequential_min_distinct_accounts": 5,
        "interpretation": "Metrics and mined associations describe only the externally supplied prepared dataset and its represented cohort.",
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_value) + "\n", encoding="utf-8"
    )
    return correlations, rows


def make_preprocessor(features):
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                           ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def regression_metrics(y, pred):
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "r2": r2_score(y, pred),
    }


def classification_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expanding_time_folds(development: pd.DataFrame, initial_fraction=0.60, fold_fraction=0.10):
    """Four expanding-window folds; tied start times are never split."""
    ordered = development.sort_values(["started_at", "attempt_key"], kind="mergesort")
    boundaries = []
    for fraction in [initial_fraction + i * fold_fraction for i in range(5)]:
        position = min(round(fraction * len(ordered)), len(ordered) - 1)
        boundaries.append(ordered.iloc[position].started_at)
    folds = []
    for i in range(4):
        train = ordered[ordered.started_at < boundaries[i]].copy()
        valid = ordered[(ordered.started_at >= boundaries[i]) & (ordered.started_at < boundaries[i + 1])].copy()
        if i == 3:
            valid = ordered[ordered.started_at >= boundaries[i]].copy()
        train = train[train.prior_same_skill_score_last.notna()]
        valid = valid[valid.prior_same_skill_score_last.notna()]
        if train.empty or valid.empty:
            raise ValueError(f"Empty warm-start rows in expanding fold {i + 1}.")
        folds.append((i + 1, train, valid))
    return folds


def candidate_estimators():
    return {
        "regression": {
            "ridge": Ridge(alpha=10.0),
            "tree_depth4": DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED),
        },
        "classification": {
            "logistic_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            "tree_depth4_balanced": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                             class_weight="balanced", random_state=SEED),
        },
    }


def feature_configurations():
    return {
        "history_only": HISTORY,
        "history_plus_ap": HISTORY + AP_FEATURES,
        "history_plus_difficulty": HISTORY + ["difficulty_level"],
        "all_features": FEATURES,
    }


def run_time_cross_validation(development):
    folds = expanding_time_folds(development)
    configurations = feature_configurations()
    estimators = candidate_estimators()
    target_columns = {"regression": "real_score", "classification": LABEL_COLUMN}
    fold_rows = []
    oof_predictions = {}
    for fold, train, valid in folds:
        for task, candidates in estimators.items():
            target = target_columns[task]
            for model_name, estimator in candidates.items():
                for config_name, features in configurations.items():
                    task_features = features if task == "regression" else features + ["target_score"]
                    pipe = Pipeline([("prep", make_preprocessor(task_features)), ("model", estimator)])
                    pipe.fit(train[task_features], train[target])
                    pred = pipe.predict(valid[task_features])
                    if task == "classification":
                        oof_predictions.setdefault((model_name, config_name), []).append(
                            (valid[target].to_numpy(), pipe.predict_proba(valid[task_features])[:, 1])
                        )
                    metrics = regression_metrics(valid[target], pred) if task == "regression" else classification_metrics(valid[target], pred)
                    fold_rows.append({
                        "task": task, "model": model_name, "configuration": config_name,
                        "feature_count": len(task_features),
                        "fold": fold, "train_rows": len(train), "validation_rows": len(valid),
                        "train_end": train.started_at.max(), "validation_start": valid.started_at.min(),
                        "validation_end": valid.started_at.max(), **metrics,
                    })
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "cv_fold_metrics.csv", index=False)
    metric_names = {"regression": ["mae", "rmse", "r2"],
                    "classification": ["precision", "recall", "f1", "accuracy"]}
    grouped = []
    for (task, model, config), part in folds_df.groupby(["task", "model", "configuration"], sort=False):
        row = {"task": task, "model": model, "configuration": config,
               "folds": int(part.fold.nunique()), "mean_train_rows": part.train_rows.mean(),
               "mean_validation_rows": part.validation_rows.mean()}
        for metric in metric_names[task]:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        grouped.append(row)
    summary_df = pd.DataFrame(grouped)
    summary_df["selected"] = False
    chosen = {}
    for task, criterion, ascending, tie_break in [
        ("regression", "mae_mean", True, "rmse_mean"),
        ("classification", "f1_mean", False, "precision_mean"),
    ]:
        candidates = summary_df[summary_df.task == task].sort_values(
            [criterion, tie_break], ascending=[ascending, False], kind="mergesort")
        winner = candidates.iloc[0]
        mask = ((summary_df.task == task) & (summary_df.model == winner.model)
                & (summary_df.configuration == winner.configuration))
        summary_df.loc[mask, "selected"] = True
        chosen[task] = {"model": winner.model, "configuration": winner.configuration,
                        "selection_metric": criterion, "selection_value": float(winner[criterion]),
                        "tie_break_metric": tie_break}
    # Select the operating probability threshold from development-only out-of-fold predictions.
    selection = chosen["classification"]
    y_oof = np.concatenate([pair[0] for pair in oof_predictions[(selection["model"], selection["configuration"])]])
    p_oof = np.concatenate([pair[1] for pair in oof_predictions[(selection["model"], selection["configuration"])]])
    thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    threshold_scores = [(float(f1_score(y_oof, p_oof >= t, zero_division=0)), float(t)) for t in thresholds]
    best_f1 = max(score for score, _ in threshold_scores)
    chosen_threshold = min(t for score, t in threshold_scores if np.isclose(score, best_f1))
    chosen["classification"]["probability_threshold"] = chosen_threshold
    chosen["classification"]["threshold_selection_metric"] = "pooled out-of-fold F1 on development period"
    chosen["classification"]["oof_f1_at_threshold"] = best_f1
    chosen["classification"]["additional_known_input"] = "target_score"
    summary_df.to_csv(OUT / "cv_metrics.csv", index=False)
    details = {
        "method": "four-fold expanding-window validation over the first 85% chronological development period",
        "folds": [{"fold": fold, "train_rows": len(train), "validation_rows": len(valid),
                   "train_start": train.started_at.min(), "train_end": train.started_at.max(),
                   "validation_start": valid.started_at.min(), "validation_end": valid.started_at.max()}
                  for fold, train, valid in folds],
        "candidate_models": {task: list(models) for task, models in estimators.items()},
        "feature_configurations": {key: value for key, value in configurations.items()},
        "classification_additional_known_input": "target_score; available before the attempt and excluded from regression",
        "selection": chosen,
        "preprocessing": "imputation, scaling, and categorical encoding are fitted within each fold's training rows",
        "warm_start_only": True,
        "final_test_excluded_from_selection": True,
    }
    (OUT / "cv_summary.json").write_text(json.dumps(details, indent=2, default=json_value) + "\n", encoding="utf-8")
    return summary_df, details


def account_cluster_bootstrap(y, pred, groups, task, iterations=500, seed=SEED):
    frame = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "group": np.asarray(groups)})
    indexes = {key: value.index.to_numpy() for key, value in frame.groupby("group", sort=False)}
    keys = np.asarray(list(indexes), dtype=object)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        positions = np.concatenate([indexes[key] for key in chosen])
        actual = frame.loc[positions, "y"]
        guessed = frame.loc[positions, "pred"]
        scores = regression_metrics(actual, guessed) if task == "regression" else classification_metrics(actual, guessed)
        samples.append(scores)
    metrics = list(samples[0])
    out = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in samples], dtype=float)
        out[f"{metric}_cluster_ci_low"] = float(np.nanpercentile(values, 2.5))
        out[f"{metric}_cluster_ci_high"] = float(np.nanpercentile(values, 97.5))
    return out


def run_models(train, valid, test, cv_summary):
    # The primary score-history comparison is warm-start only.
    train_w = train[train.prior_same_skill_score_last.notna()].copy()
    valid_w = valid[valid.prior_same_skill_score_last.notna()].copy()
    test_w = test[test.prior_same_skill_score_last.notna()].copy()
    if min(len(train_w), len(valid_w), len(test_w)) == 0:
        raise ValueError("A temporal partition has no warm-start rows.")
    rows = []
    # Baselines can be evaluated without fitting any target-dependent preprocessing.
    for name, part in [("test", test_w)]:
        last_metrics = regression_metrics(part.real_score, part.prior_same_skill_score_last)
        if name == "test":
            last_metrics.update(account_cluster_bootstrap(
                part.real_score, part.prior_same_skill_score_last, part.student_key, "regression", seed=SEED + 1
            ))
        rows.append({
            "task": "regression", "model": "last_same_skill_score", "partition": name,
            "features": "prior_same_skill_score_last", **last_metrics,
        })
        mean_metrics = regression_metrics(part.real_score, part.prior_same_skill_score_mean_last3)
        if name == "test":
            mean_metrics.update(account_cluster_bootstrap(
                part.real_score, part.prior_same_skill_score_mean_last3, part.student_key, "regression", seed=SEED + 2
            ))
        rows.append({
            "task": "regression", "model": "mean_last3", "partition": name,
            "features": "prior_same_skill_score_mean_last3", **mean_metrics,
        })
        majority_class = int(train_w[LABEL_COLUMN].mean() >= 0.5)
        majority_pred = np.full(len(part), majority_class)
        majority_metrics = classification_metrics(part[LABEL_COLUMN], majority_pred)
        if name == "test":
            majority_metrics.update(account_cluster_bootstrap(
                part[LABEL_COLUMN], majority_pred, part.student_key, "classification", seed=SEED + 3
            ))
        rows.append({
            "task": "classification", "model": "majority_class_baseline", "partition": name,
            "features": "training_prevalence_only",
            **majority_metrics,
        })

    development_w = pd.concat([train_w, valid_w])
    estimator_map = candidate_estimators()
    configurations = feature_configurations()
    selected_test_results = {}
    for task in ["regression", "classification"]:
        chosen = cv_summary["selection"][task]
        model_name, config_name = chosen["model"], chosen["configuration"]
        estimator = estimator_map[task][model_name]
        features = configurations[config_name]
        task_features = features if task == "regression" else features + ["target_score"]
        target_col = "real_score" if task == "regression" else LABEL_COLUMN
        pipe = Pipeline([("prep", make_preprocessor(task_features)), ("model", estimator)])
        pipe.fit(development_w[task_features], development_w[target_col])
        if task == "classification":
            probabilities = pipe.predict_proba(test_w[task_features])[:, 1]
            threshold = chosen.get("probability_threshold", 0.5)
            pred = (probabilities >= threshold).astype(int)
        else:
            pred = pipe.predict(test_w[task_features])
            probabilities = None
        scores = regression_metrics(test_w[target_col], pred) if task == "regression" else classification_metrics(test_w[target_col], pred)
        scores.update(account_cluster_bootstrap(test_w[target_col], pred, test_w.student_key, task,
                                               seed=SEED + (4 if task == "regression" else 5)))
        rows.append({"task": task, "model": model_name, "partition": "test", "features": config_name,
                     **({"classification_threshold": threshold, "label": LABEL_COLUMN,
                        "brier_score": float(np.mean((probabilities - test_w[target_col].to_numpy()) ** 2))}
                       if task == "classification" else {}),
                     "cv_selected": True, **scores})
        selected_test_results[task] = {"model": model_name, "configuration": config_name,
                                       "test_rows": len(test_w), **scores}

    # Feature ablation is evaluated on validation folds; the final test is not used for selection.
    ablation_rows = []
    cv_agg = pd.read_csv(OUT / "cv_metrics.csv")
    for _, result in cv_agg.iterrows():
        features = configurations[result.configuration]
        item = {"task": result.task, "model": result.model, "configuration": result.configuration,
                "feature_count": len(features) + (1 if result.task == "classification" else 0),
                "folds": int(result.folds)}
        names = ["mae", "rmse", "r2"] if result.task == "regression" else ["precision", "recall", "f1", "accuracy"]
        for name in names:
            item[f"{name}_mean"] = result[f"{name}_mean"]
            item[f"{name}_std"] = result[f"{name}_std"]
        ablation_rows.append(item)
    pd.DataFrame(rows).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT / "feature_ablation.csv", index=False)
    return rows, ablation_rows, selected_test_results


def itemize_for_rules(df):
    result = pd.DataFrame(index=df.index)
    result["skill"] = "skill=" + df.typeExam.fillna("Unknown").astype(str)
    result["difficulty"] = "difficulty=" + df.difficulty_level.fillna("Unknown").astype(str)
    result["mode"] = "mode=" + df["mode"].fillna("Unknown").astype(str)
    count = df.prior_same_skill_attempt_count.fillna(0)
    result["history_count"] = "prior_skill_attempts=" + pd.cut(
        count, [-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"]
    ).astype(str)
    last = df.prior_same_skill_score_last
    result["last_score"] = "prior_last_score=" + pd.cut(
        last, [-np.inf, 4.5, 6.0, np.inf], labels=["low", "mid", "high"]
    ).astype(object).where(last.notna(), "unavailable").astype(str)
    ap_count = df.prior_ap_events_30d.fillna(0) + df.prior_ap_events_31_90d.fillna(0)
    result["ap_activity"] = "prior_ap_events_90d=" + pd.cut(
        ap_count, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"]
    ).astype(str)
    trend = df.prior_same_skill_score_trend_last3_vs_prev3
    trend_cat = pd.Series("unavailable", index=df.index)
    trend_cat.loc[trend < -0.01] = "decreasing"
    trend_cat.loc[trend > 0.01] = "increasing"
    trend_cat.loc[trend.between(-0.01, 0.01)] = "stable"
    result["score_trend"] = "prior_trend=" + trend_cat
    # target_score defines goal_met, so it is intentionally excluded from
    # pattern antecedents to avoid tautological target-to-label rules.
    result["outcome"] = "goal_attainment=" + np.where(df[LABEL_COLUMN].to_numpy() == 1, "met", "not_met")
    return result


def mine_frequent_itemsets(train, test):
    train_items = itemize_for_rules(train)
    test_items = itemize_for_rules(test)
    columns = [c for c in train_items if c != "outcome"]
    min_support_count = max(30, math.ceil(0.02 * len(train_items)))
    min_accounts = 10
    patterns = []
    for k in [1, 2, 3]:
        for cols in itertools.combinations(columns, k):
            for combo in itertools.product(*[train_items[c].unique() for c in cols]):
                mask = np.ones(len(train_items), dtype=bool)
                test_mask = np.ones(len(test_items), dtype=bool)
                for col, item in zip(cols, combo):
                    mask &= train_items[col].to_numpy() == item
                    test_mask &= test_items[col].to_numpy() == item
                support_count = int(mask.sum())
                account_support = int(train.loc[train_items.index[mask], "student_key"].nunique())
                if support_count < min_support_count or account_support < min_accounts:
                    continue
                test_support_count = int(test_mask.sum())
                patterns.append({
                    "itemset": " AND ".join(str(x) for x in combo),
                    "items": k,
                    "train_support_count": support_count,
                    "train_support": support_count / len(train_items),
                    "train_account_support": account_support,
                    "test_support_count": test_support_count,
                    "test_support": test_support_count / len(test_items) if len(test_items) else np.nan,
                    "test_account_support": int(test.loc[test_items.index[test_mask], "student_key"].nunique()),
                })
    result = pd.DataFrame(patterns)
    if not result.empty:
        result = result.sort_values(["items", "train_support_count"], ascending=[True, False])
    result.to_csv(OUT / "frequent_itemsets.csv", index=False)
    return result, min_support_count, min_accounts


def mine_association_rules(train, test):
    train_items = itemize_for_rules(train)
    test_items = itemize_for_rules(test)
    feature_item_cols = [c for c in train_items if c != "outcome"]
    labels = ["goal_attainment=met", "goal_attainment=not_met"]
    total = len(train_items)
    min_support_count = max(30, math.ceil(0.02 * total))
    min_accounts = 10
    base = train_items.outcome.value_counts(normalize=True).to_dict()
    candidates = []
    for k in [1, 2]:
        for cols in itertools.combinations(feature_item_cols, k):
            # Each field contributes exactly one item per row; combinations are unambiguous.
            for combo in itertools.product(*[train_items[c].unique() for c in cols]):
                mask = np.ones(len(train_items), dtype=bool)
                test_mask = np.ones(len(test_items), dtype=bool)
                for col, item in zip(cols, combo):
                    mask &= (train_items[col].to_numpy() == item)
                    test_mask &= (test_items[col].to_numpy() == item)
                n = int(mask.sum())
                accounts = int(train.loc[train_items.index[mask], "student_key"].nunique())
                if n < min_support_count or accounts < min_accounts:
                    continue
                for label in labels:
                    label_name = label.split("=", 1)[1]
                    label_mask = (train_items.outcome.to_numpy() == label)
                    joint = int((mask & label_mask).sum())
                    if joint < min_support_count:
                        continue
                    conf = joint / n
                    p_label = base.get(label, 0.0)
                    lift = conf / p_label if p_label else np.nan
                    test_label_mask = (test_items.outcome.to_numpy() == label)
                    test_n = int(test_mask.sum())
                    test_joint = int((test_mask & test_label_mask).sum())
                    test_conf = test_joint / test_n if test_n else np.nan
                    test_label_rate = float((test_items.outcome == label).mean()) if len(test_items) else np.nan
                    test_lift = test_conf / test_label_rate if test_label_rate and not np.isnan(test_conf) else np.nan
                    candidates.append({
                        "antecedent": " AND ".join(str(x) for x in combo),
                        "consequent": label,
                        "train_support_count": joint,
                        "train_support": joint / total,
                        "train_antecedent_count": n,
                        "train_account_support": accounts,
                        "train_confidence": conf,
                        "train_lift": lift,
                        "test_support_count": test_joint,
                        "test_antecedent_count": test_n,
                        "test_antecedent_account_support": int(test.loc[test_items.index[test_mask], "student_key"].nunique()),
                        "test_rule_account_support": int(test.loc[test_items.index[test_mask & test_label_mask], "student_key"].nunique()),
                        "test_confidence": test_conf,
                        "test_lift": test_lift,
                    })
    rules = pd.DataFrame(candidates)
    if not rules.empty:
        rules = rules.sort_values(["train_lift", "train_support_count"], ascending=[False, False])
    rules.to_csv(OUT / "association_rules.csv", index=False)
    return rules, min_support_count, min_accounts


def mine_sequential_patterns(df, train, test):
    ordered = df.sort_values(["student_key", "started_at", "attempt_key"], kind="mergesort")
    prior_sequences = {}
    by_student = {key: group.sort_values(["submitted_at", "started_at", "attempt_key"], kind="mergesort")
                  for key, group in ordered.groupby("student_key", sort=False)}
    for row_index, row in ordered.iterrows():
        past = by_student[row.student_key]
        past = past[(past.submitted_at < row.started_at) & (past.index != row_index)]
        skills = past.typeExam.tail(3).tolist()
        prior_sequences[row_index] = {
            size: tuple(skills[-size:]) for size in (2, 3) if len(skills) >= size
        }

    split_by_name = {"train": train, "test": test}
    seq_rows = {name: [] for name in split_by_name}
    rule_rows = {name: [] for name in split_by_name}
    for name, part in split_by_name.items():
        for row_index, row in part.iterrows():
            for size, pattern in prior_sequences.get(row_index, {}).items():
                label = "met" if row[LABEL_COLUMN] == 1 else "not_met"
                seq_rows[name].append({
                    "pattern": pattern,
                    "size": size,
                    "student_key": row.student_key,
                })
                rule_rows[name].append({
                    "pattern": pattern,
                    "size": size,
                    "label": label,
                    "student_key": row.student_key,
                })
    seq_tables = {name: pd.DataFrame(rows) for name, rows in seq_rows.items()}
    rule_tables = {name: pd.DataFrame(rows) for name, rows in rule_rows.items()}
    min_support = 10
    min_accounts = 5

    frequent = []
    if not seq_tables["train"].empty:
        for (size, pattern), group in seq_tables["train"].groupby(["size", "pattern"], sort=False):
            accounts = group.student_key.nunique()
            if len(group) < min_support or accounts < min_accounts:
                continue
            test_group = seq_tables["test"]
            test_match = test_group[(test_group["size"] == size) & (test_group.pattern == pattern)] if not test_group.empty else test_group
            frequent.append({
                "sequence": " → ".join(pattern),
                "length": size,
                "train_occurrences": len(group),
                "train_accounts": accounts,
                "test_occurrences": len(test_match),
                "test_accounts": test_match.student_key.nunique() if not test_match.empty else 0,
            })
    frequent_df = pd.DataFrame(frequent)
    if not frequent_df.empty:
        frequent_df = frequent_df.sort_values(["length", "train_occurrences"], ascending=[True, False])
    frequent_df.to_csv(OUT / "frequent_sequences.csv", index=False)

    train_rules = rule_tables["train"]
    test_rules = rule_tables["test"]
    rules = []
    if not train_rules.empty:
        for (size, pattern), group in train_rules.groupby(["size", "pattern"], sort=False):
            train_base = train_rules[train_rules["size"] == size].label.value_counts(normalize=True).to_dict()
            antecedent_count = len(group)
            antecedent_accounts = group.student_key.nunique()
            if antecedent_count < min_support or antecedent_accounts < min_accounts:
                continue
            test_size_rules = test_rules[test_rules["size"] == size] if not test_rules.empty else test_rules
            test_base = test_size_rules.label.value_counts(normalize=True).to_dict() if not test_size_rules.empty else {}
            tgroup = test_size_rules[test_size_rules.pattern == pattern] if not test_size_rules.empty else test_size_rules
            for label in ("met", "not_met"):
                joint = group[group.label == label]
                joint_count = len(joint)
                if joint_count < min_support or joint.student_key.nunique() < min_accounts:
                    continue
                test_joint = tgroup[tgroup.label == label] if not tgroup.empty else tgroup
                confidence = joint_count / antecedent_count
                test_confidence = len(test_joint) / len(tgroup) if len(tgroup) else np.nan
                rules.append({
                    "prior_skill_sequence": " → ".join(pattern),
                    "sequence_length": size,
                    "consequent": "goal_attainment=" + label,
                    "train_support_count": joint_count,
                    "train_account_support": joint.student_key.nunique(),
                    "train_confidence": confidence,
                    "train_lift": confidence / train_base[label] if train_base.get(label) else np.nan,
                    "test_support_count": len(test_joint),
                    "test_account_support": test_joint.student_key.nunique() if not test_joint.empty else 0,
                    "test_confidence": test_confidence,
                    "test_lift": test_confidence / test_base[label] if test_base.get(label) and not np.isnan(test_confidence) else np.nan,
                })
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(["test_lift", "test_support_count"], ascending=[False, False])
    rules_df.to_csv(OUT / "sequential_outcome_rules.csv", index=False)
    return frequent_df, rules_df, min_support, min_accounts


def main():
    df = load_data()
    train, valid, test, t1, t2 = time_split(df)
    save_eda(df, train, valid, test, t1, t2)
    development = pd.concat([train, valid])
    cv_rows, cv_summary = run_time_cross_validation(development)
    model_rows, ablation_rows, selected_test_results = run_models(train, valid, test, cv_summary)
    itemsets, itemset_support_count, itemset_accounts = mine_frequent_itemsets(train, test)
    rules, support_count, min_accounts = mine_association_rules(train, test)
    sequences, sequence_rules, sequence_support, sequence_accounts = mine_sequential_patterns(df, train, test)
    summary = {
        "rows": len(df),
        "accounts": int(df.student_key.nunique()),
        "train_validation_test": [len(train), len(valid), len(test)],
        "temporal_cutoffs": [t1.isoformat(), t2.isoformat()],
        "classification_label": "real_score >= target_score",
        "goal_met_not_met": [int(df[LABEL_COLUMN].sum()), int((1-df[LABEL_COLUMN]).sum())],
        "models_reported": len(model_rows),
        "ablation_rows": len(ablation_rows),
        "cross_validation": cv_summary,
        "final_test_selected_models": selected_test_results,
        "frequent_itemsets": len(itemsets),
        "frequent_itemset_min_support_count": itemset_support_count,
        "frequent_itemset_min_distinct_accounts": itemset_accounts,
        "association_rules": len(rules),
        "association_min_joint_support_count": support_count,
        "association_min_distinct_accounts": min_accounts,
        "frequent_sequences": len(sequences),
        "sequential_outcome_rules": len(sequence_rules),
        "sequential_min_support_count": sequence_support,
        "sequential_min_distinct_accounts": sequence_accounts,
        "warning": "Interpret metrics only for the observed cohort and time period represented by the input exports.",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=json_value) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=json_value))


if __name__ == "__main__":
    main()
