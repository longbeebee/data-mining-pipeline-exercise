#!/usr/bin/env python3
"""Phase 3: leakage audit, account holdout, and error analysis."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = os.environ.get("OBSERVED_PIPELINE_OUTPUT_DIR")
if not OUTPUT_PATH:
    raise RuntimeError("Set OBSERVED_PIPELINE_OUTPUT_DIR or run through run_pipeline.py; outputs must stay outside this source package.")
DATA = Path(OUTPUT_PATH).resolve()
if DATA.is_relative_to(ROOT):
    raise RuntimeError("OBSERVED_PIPELINE_OUTPUT_DIR must be outside observed_pipeline.")
OUT = DATA / "mining"
sys.path.insert(0, str(ROOT))
import run_mining_analysis as analysis  # noqa: E402

TARGET = "real_score"
POINT_IN_TIME_COLUMNS = [
    "prior_same_skill_attempt_count", "prior_other_skill_attempt_count",
    "prior_same_skill_score_last", "prior_same_skill_score_mean_last3",
    "prior_same_skill_score_std_last5", "prior_same_skill_score_trend_last3_vs_prev3",
    "days_since_prior_same_skill_attempt", "prior_same_skill_attempts_30d",
    "prior_same_skill_attempts_31_90d", "prior_same_assessment_count",
    "prior_ap_events_30d", "prior_ap_events_31_90d",
    "prior_ap_amount_per_event_90d", "prior_ap_improvement_rate_90d",
    "days_since_prior_ap_event",
]
CURRENT_OUTCOME_OR_ID = {
    "attempt_key", "student_key", "assessment_key", "started_at", "submitted_at",
    "duration_seconds", "total_score", "max_score", "ielts_band_score",
    "target_score", "real_score",
}


def close(actual, expected, tol=1e-8):
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


def record_leakage_audit(df: pd.DataFrame) -> dict:
    attempts = df.copy()
    ap = pd.read_csv(DATA / "ap_events.csv")
    ap["event_at"] = pd.to_datetime(ap["event_at"], utc=True)
    ap["amount"] = pd.to_numeric(ap["amount"], errors="coerce")
    observed = attempts.set_index("attempt_key")
    mismatches = {column: [] for column in POINT_IN_TIME_COLUMNS}
    eligible_rows = 0
    for _, target in attempts.iterrows():
        if pd.isna(target.started_at):
            continue
        eligible_rows += 1
        start = target.started_at
        history = attempts[
            (attempts.student_key == target.student_key)
            & (attempts.submitted_at < start)
        ].sort_values(["submitted_at", "started_at", "attempt_key"], kind="mergesort")
        same = history[history.typeExam == target.typeExam]
        other = history[history.typeExam != target.typeExam]
        scored = same[same.real_score.notna()]
        scores = scored.real_score.astype(float).tolist()
        latest5, latest3, previous3 = scores[-5:], scores[-3:], scores[-6:-3]
        same_ages = (start - same.submitted_at).dt.total_seconds() / 86400
        expected = {
            "prior_same_skill_attempt_count": len(same),
            "prior_other_skill_attempt_count": len(other),
            "prior_same_skill_score_last": scores[-1] if scores else np.nan,
            "prior_same_skill_score_mean_last3": float(np.mean(latest3)) if latest3 else np.nan,
            "prior_same_skill_score_std_last5": float(np.std(latest5, ddof=0)) if latest5 else np.nan,
            "prior_same_skill_score_trend_last3_vs_prev3": float(np.mean(latest3) - np.mean(previous3)) if len(scores) >= 6 else np.nan,
            "days_since_prior_same_skill_attempt": (start - scored.iloc[-1].submitted_at).total_seconds() / 86400 if scores else np.nan,
            "prior_same_skill_attempts_30d": int((same_ages <= 30).sum()),
            "prior_same_skill_attempts_31_90d": int(((same_ages > 30) & (same_ages <= 90)).sum()),
            "prior_same_assessment_count": int((history.assessment_key == target.assessment_key).sum()),
        }
        user_ap = ap[(ap.student_key == target.student_key) & (ap.event_at < start)]
        canonical = user_ap[user_ap.scope == "OVERALL"]
        age = (start - canonical.event_at).dt.total_seconds() / 86400
        recent90 = canonical[age <= 90]
        recent30 = canonical[age <= 30]
        all_prior_user_ap = user_ap
        expected.update({
            "prior_ap_events_30d": len(recent30),
            "prior_ap_events_31_90d": len(recent90) - len(recent30),
            # The builder divides the available amount sum by event count.
            "prior_ap_amount_per_event_90d": recent90.amount.sum() / len(recent90) if len(recent90) else np.nan,
            "prior_ap_improvement_rate_90d": (recent90.event_type.eq("IMPROVEMENT").sum() / len(recent90)) if len(recent90) else np.nan,
            # The builder uses all prior scopes for recency, but canonical OVERALL for aggregates.
            "days_since_prior_ap_event": (start - all_prior_user_ap.event_at.max()).total_seconds() / 86400 if len(all_prior_user_ap) else np.nan,
        })
        for column, value in expected.items():
            if not close(target[column], value):
                mismatches[column].append({"attempt_key": target.attempt_key,
                                           "stored": None if pd.isna(target[column]) else float(target[column]),
                                           "recomputed": None if pd.isna(value) else float(value)})

    feature_input_set = set(analysis.FEATURES + ["target_score"])
    forbidden_inputs = sorted(feature_input_set & (CURRENT_OUTCOME_OR_ID - {"target_score"}))
    order_violations = int((attempts.submitted_at <= attempts.started_at).sum())
    duplicate_attempt_keys = int(attempts.attempt_key.duplicated().sum())
    missing_target = int(attempts[TARGET].isna().sum())
    per_assessment_difficulty = attempts.groupby("assessment_key").difficulty_level.nunique()
    difficulty_inconsistency = int((per_assessment_difficulty > 1).sum())
    mismatch_counts = {key: len(value) for key, value in mismatches.items()}
    mismatch_row_keys = {item["attempt_key"] for values in mismatches.values() for item in values}
    return {
        "rows": len(attempts), "eligible_rows_with_start": eligible_rows,
        "attempt_key_duplicates": duplicate_attempt_keys,
        "submitted_not_after_started": order_violations,
        "rows_missing_target_real_score": missing_target,
        "difficulty_labels_inconsistent_within_assessment": difficulty_inconsistency,
        "regression_model_features": analysis.FEATURES,
        "classification_model_features": analysis.FEATURES + ["target_score"],
        "forbidden_current_outcome_or_identifier_features": forbidden_inputs,
        "point_in_time_rows_with_mismatch": len(mismatch_row_keys),
        "point_in_time_features_with_any_mismatch": int(sum(bool(n) for n in mismatch_counts.values())),
        "point_in_time_mismatch_counts_by_feature": mismatch_counts,
        "mismatch_examples": {k: v[:3] for k, v in mismatches.items() if v},
        "pass": not any([duplicate_attempt_keys, order_violations, missing_target,
                         difficulty_inconsistency, forbidden_inputs, sum(mismatch_counts.values())]),
        "scope_note": "Reconstructs listed historical aggregate features from the prepared attempt/AP tables; does not verify raw-source correctness or independent score validity.",
    }


def metric_block(y, pred):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(analysis.r2_score(y, pred)),
    }


def unseen_account_holdout(df):
    # Hold out whole accounts, deterministically. This is a cold-account stress test,
    # not a chronological future-period estimate.
    accounts = np.array(sorted(df.student_key.unique()))
    rng = np.random.default_rng(analysis.SEED + 101)
    rng.shuffle(accounts)
    n_test_accounts = max(1, round(0.20 * len(accounts)))
    heldout = set(accounts[:n_test_accounts])
    train = df[~df.student_key.isin(heldout)].copy()
    test = df[df.student_key.isin(heldout)].copy()
    cv_summary = json.loads((OUT / "cv_summary.json").read_text(encoding="utf-8"))
    outputs = []

    regression_selection = cv_summary["selection"]["regression"]
    regression_features = analysis.feature_configurations()[regression_selection["configuration"]]
    regression_fit = train[train.prior_same_skill_score_last.notna()]
    regression_test = test[test.prior_same_skill_score_last.notna()]
    reg = Pipeline([("prep", analysis.make_preprocessor(regression_features)),
                    ("model", analysis.candidate_estimators()["regression"][regression_selection["model"]])])
    reg.fit(regression_fit[regression_features], regression_fit.real_score)
    pred = reg.predict(regression_test[regression_features])
    outputs.append({"task": "regression", "model": regression_selection["model"],
                    "configuration": regression_selection["configuration"],
                    "selection_source": "expanding-time cross-validation on development data",
                    "population": "warm_start",
                    "train_accounts": int(train.student_key.nunique()),
                    "test_accounts": int(test.student_key.nunique()),
                    "train_rows": len(regression_fit), **metric_block(regression_test.real_score, pred)})

    classification_selection = cv_summary["selection"]["classification"]
    classification_features = analysis.feature_configurations()[classification_selection["configuration"]] + ["target_score"]
    cls_fit = train[train.prior_same_skill_score_last.notna()]
    cls_test = test[test.prior_same_skill_score_last.notna()]
    clf = Pipeline([("prep", analysis.make_preprocessor(classification_features)),
                    ("model", analysis.candidate_estimators()["classification"][classification_selection["model"]])])
    clf.fit(cls_fit[classification_features], cls_fit.goal_met)
    cls_prob = clf.predict_proba(cls_test[classification_features])[:, 1]
    probability_threshold = classification_selection["probability_threshold"]
    cls_pred = (cls_prob >= probability_threshold).astype(int)
    outputs.append({"task": "classification", "model": classification_selection["model"],
                    "configuration": classification_selection["configuration"],
                    "selection_source": "expanding-time cross-validation on development data",
                    "probability_threshold": probability_threshold,
                    "threshold_source": classification_selection["threshold_selection_metric"],
                    "population": "warm_start",
                    "train_accounts": int(train.student_key.nunique()),
                    "test_accounts": int(test.student_key.nunique()),
                    "train_rows": len(cls_fit), "n": len(cls_test),
                    "precision": float(precision_score(cls_test.goal_met, cls_pred, zero_division=0)),
                    "recall": float(recall_score(cls_test.goal_met, cls_pred, zero_division=0)),
                    "f1": float(f1_score(cls_test.goal_met, cls_pred, zero_division=0)),
                    "accuracy": float(accuracy_score(cls_test.goal_met, cls_pred))})
    summary = {"method": "deterministic 80/20 account-group holdout",
               "seed": analysis.SEED + 101,
               "train_accounts": int(train.student_key.nunique()),
               "heldout_accounts": int(test.student_key.nunique()),
               "train_rows": len(train), "heldout_rows": len(test),
               "heldout_warm_start_rows": int(test.prior_same_skill_score_last.notna().sum()),
               "heldout_cold_start_rows": int(test.prior_same_skill_score_last.isna().sum()),
               "metrics": outputs,
               "model_selection_source": "Models and classification probability threshold were selected using development-period expanding-time cross-validation; this account holdout was not used for selection.",
               "scope_note": "Accounts are disjoint between fit and evaluation; calendar periods overlap, so this isolates unseen-account generalization and is not a future-time test."}
    return summary


def temporal_error_and_sensitivity(df):
    train, valid, test, t1, t2 = analysis.time_split(df)
    fit = pd.concat([train, valid])
    fit = fit[fit.prior_same_skill_score_last.notna()]
    test = test[test.prior_same_skill_score_last.notna()].copy()
    cv_summary = json.loads((OUT / "cv_summary.json").read_text(encoding="utf-8"))
    selected = cv_summary["selection"]["regression"]
    features = analysis.feature_configurations()[selected["configuration"]]
    estimator = analysis.candidate_estimators()["regression"][selected["model"]]
    model = Pipeline([("prep", analysis.make_preprocessor(features)), ("model", estimator)])
    model.fit(fit[features], fit.real_score)
    test["prediction"] = model.predict(test[features])
    test["absolute_error"] = (test.real_score - test.prediction).abs()
    test["signed_error"] = test.prediction - test.real_score
    error_rows = []
    for feature, groups in [
        ("skill", test.groupby("typeExam", dropna=False)),
        ("history_length", test.assign(history_bin=pd.cut(
            test.prior_same_skill_attempt_count, [-1, 0, 2, 5, np.inf],
            labels=["0", "1-2", "3-5", "6+"], ordered=True)).groupby("history_bin", observed=False)),
        ("difficulty", test.groupby("difficulty_level", dropna=False)),
    ]:
        for category, part in groups:
            if not len(part):
                continue
            error_rows.append({"dimension": feature, "group": str(category),
                               "attempts": len(part), "accounts": part.student_key.nunique(),
                               "mae": part.absolute_error.mean(),
                               "rmse": math.sqrt(mean_squared_error(part.real_score, part.prediction)),
                               "mean_signed_error_prediction_minus_actual": part.signed_error.mean(),
                               "real_score_mean": part.real_score.mean(),
                               "prediction_mean": part.prediction.mean()})
    pd.DataFrame(error_rows).to_csv(OUT / "phase3_error_breakdown.csv", index=False)

    selection = cv_summary["selection"]["classification"]
    classification_features = analysis.feature_configurations()[selection["configuration"]] + ["target_score"]
    estimator = analysis.candidate_estimators()["classification"][selection["model"]]
    classifier = Pipeline([("prep", analysis.make_preprocessor(classification_features)),
                           ("model", estimator)])
    classifier.fit(fit[classification_features], fit.goal_met)
    prob = classifier.predict_proba(test[classification_features])[:, 1]
    sensitivity_rows = []
    for threshold in (0.50, selection["probability_threshold"]):
        y_test = test.goal_met.to_numpy()
        pred = (prob >= threshold).astype(int)
        sensitivity_rows.append({"probability_threshold": threshold,
                                 "threshold_source": "fixed reference" if threshold == 0.50 else "selected from development cross-validation",
                                 "test_positive_rate": float(y_test.mean()), "n_test": len(test),
                                 "precision": float(precision_score(y_test, pred, zero_division=0)),
                                 "recall": float(recall_score(y_test, pred, zero_division=0)),
                                 "f1": float(f1_score(y_test, pred, zero_division=0)),
                                 "accuracy": float(accuracy_score(y_test, pred))})
    reg_test = test.copy()
    reg_test["regression_goal_prediction"] = model.predict(reg_test[features])
    reg_pred_label = (reg_test.regression_goal_prediction >= reg_test.target_score).astype(int)
    sensitivity_rows.append({"probability_threshold": np.nan,
                             "threshold_source": "regression score compared with individual target",
                             "test_positive_rate": float(reg_test.goal_met.mean()), "n_test": len(reg_test),
                             "precision": float(precision_score(reg_test.goal_met, reg_pred_label, zero_division=0)),
                             "recall": float(recall_score(reg_test.goal_met, reg_pred_label, zero_division=0)),
                             "f1": float(f1_score(reg_test.goal_met, reg_pred_label, zero_division=0)),
                             "accuracy": float(accuracy_score(reg_test.goal_met, reg_pred_label))})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "phase3_threshold_sensitivity.csv", index=False)
    return {"temporal_cutoffs": [t1.isoformat(), t2.isoformat()],
            "regression_model": selected["model"],
            "regression_configuration": selected["configuration"],
            "test_warm_start_rows": len(test),
            "error_breakdown": "phase3_error_breakdown.csv",
            "goal_attainment_decision_sensitivity": sensitivity_rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = analysis.load_data()
    audit = record_leakage_audit(df)
    account = unseen_account_holdout(df)
    temporal = temporal_error_and_sensitivity(df)
    (OUT / "phase3_leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "phase3_unseen_account_summary.json").write_text(json.dumps(account, indent=2) + "\n", encoding="utf-8")
    summary = {"leakage_audit": audit, "unseen_account": account, "temporal_error_and_goal_attainment_sensitivity": temporal}
    (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
