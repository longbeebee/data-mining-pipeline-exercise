#!/usr/bin/env python3
"""Validate external prepared tables and recompute point-in-time features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ATTEMPT_REQUIRED = {
    "attempt_key", "student_key", "assessment_key", "typeExam", "mode",
    "difficulty_level", "started_at", "submitted_at", "assessment_duration_minutes",
    "real_score", "target_score",
}
AP_REQUIRED = {"student_key", "event_at", "event_type", "scope", "amount", "reference_type", "reference_key"}
FEATURES = [
    "prior_same_skill_attempt_count", "prior_other_skill_attempt_count",
    "prior_same_skill_score_last", "prior_same_skill_score_mean_last3",
    "prior_same_skill_score_std_last5", "prior_same_skill_score_trend_last3_vs_prev3",
    "days_since_prior_same_skill_attempt", "prior_same_skill_attempts_30d",
    "prior_same_skill_attempts_31_90d", "prior_same_assessment_count",
    "prior_ap_events_30d", "prior_ap_events_31_90d",
    "prior_ap_amount_per_event_90d", "prior_ap_improvement_rate_90d",
    "days_since_prior_ap_event",
]
ATTEMPT_OUTPUT_COLUMNS = [
    "attempt_key", "student_key", "assessment_key", "typeExam",
    "assessment_duration_minutes", "mode", "difficulty_level", "started_at",
    "submitted_at", "duration_seconds", "total_score", "max_score",
    "ielts_band_score", "target_score", "real_score", *FEATURES,
]
AP_OUTPUT_COLUMNS = [
    "student_key", "event_at", "event_type", "scope", "amount",
    "reference_type", "reference_key",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def parse_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def calculate_attempt_history(attempts: pd.DataFrame) -> pd.DataFrame:
    result = attempts.copy()
    for column in FEATURES:
        result[column] = np.nan

    for _, group in result.groupby("student_key", sort=False, dropna=False):
        indices = list(group.sort_values(["started_at", "attempt_key"], kind="mergesort").index)
        history = group
        for idx in indices:
            start = result.at[idx, "started_at"]
            if pd.isna(start):
                continue
            past = history[(history["attempt_key"] != result.at[idx, "attempt_key"])
                           & (history["submitted_at"] < start) & history["submitted_at"].notna()]
            skill = result.at[idx, "typeExam"]
            same = past[past["typeExam"] == skill]
            other = past[past["typeExam"] != skill]
            scored = same[same["real_score"].notna()].sort_values(
                ["submitted_at", "started_at", "attempt_key"], kind="mergesort")
            scores = scored["real_score"].astype(float).tolist()
            last3, last5, previous3 = scores[-3:], scores[-5:], scores[-6:-3]
            ages = (start - same["submitted_at"]).dt.total_seconds() / 86400
            last_scored_at = scored["submitted_at"].iloc[-1] if len(scored) else pd.NaT
            assessment_key = result.at[idx, "assessment_key"]
            result.at[idx, "prior_same_skill_attempt_count"] = len(same)
            result.at[idx, "prior_other_skill_attempt_count"] = len(other)
            result.at[idx, "prior_same_skill_score_last"] = scores[-1] if scores else np.nan
            result.at[idx, "prior_same_skill_score_mean_last3"] = float(np.mean(last3)) if last3 else np.nan
            result.at[idx, "prior_same_skill_score_std_last5"] = float(np.std(last5)) if last5 else np.nan
            result.at[idx, "prior_same_skill_score_trend_last3_vs_prev3"] = (
                float(np.mean(last3) - np.mean(previous3)) if len(scores) >= 6 else np.nan)
            result.at[idx, "days_since_prior_same_skill_attempt"] = (
                (start - last_scored_at).total_seconds() / 86400 if pd.notna(last_scored_at) else np.nan)
            result.at[idx, "prior_same_skill_attempts_30d"] = int((ages <= 30).sum())
            result.at[idx, "prior_same_skill_attempts_31_90d"] = int(((ages > 30) & (ages <= 90)).sum())
            result.at[idx, "prior_same_assessment_count"] = int((past["assessment_key"] == assessment_key).sum())
    return result


def calculate_ap_history(attempts: pd.DataFrame, ap_events: pd.DataFrame,
                         sat_attempt_keys: set[str]) -> pd.DataFrame:
    result = attempts.copy()
    for column in FEATURES[-5:]:
        result[column] = np.nan
    ap = ap_events.copy()
    ap["event_at"] = pd.to_datetime(ap["event_at"], utc=True, errors="coerce")
    ap["amount"] = pd.to_numeric(ap["amount"], errors="coerce")
    ap = ap[ap["event_at"].notna()].copy()
    ap["reference_type"] = ap["reference_type"].fillna("").astype(str)
    ap = ap[~ap["reference_type"].isin({"ShortVideoView", "ShortVideoCompletion"})]
    sat_refs = ap["reference_type"].isin({"Attempt", "AttemptImprovement"}) & ap["reference_key"].astype(str).isin(sat_attempt_keys)
    ap = ap[~sat_refs]

    events_by_student = {key: part for key, part in ap.groupby("student_key", sort=False)}
    for idx, row in result.iterrows():
        start = row["started_at"]
        if pd.isna(start):
            continue
        prior = events_by_student.get(row["student_key"])
        if prior is None:
            prior = ap.iloc[0:0]
        prior = prior[prior["event_at"] < start]
        canonical = prior[prior["scope"].astype(str) == "OVERALL"]
        age = (start - canonical["event_at"]).dt.total_seconds() / 86400
        recent30 = canonical[age <= 30]
        recent90 = canonical[age <= 90]
        recent31_90 = canonical[(age > 30) & (age <= 90)]
        amounts = recent90["amount"].dropna()
        result.at[idx, "prior_ap_events_30d"] = len(recent30)
        result.at[idx, "prior_ap_events_31_90d"] = len(recent31_90)
        result.at[idx, "prior_ap_amount_per_event_90d"] = (
            float(amounts.sum() / len(recent90)) if len(recent90) and len(amounts) == len(recent90) else np.nan)
        result.at[idx, "prior_ap_improvement_rate_90d"] = (
            float((recent90["event_type"] == "IMPROVEMENT").mean()) if len(recent90) else np.nan)
        result.at[idx, "days_since_prior_ap_event"] = (
            (start - prior["event_at"].max()).total_seconds() / 86400 if len(prior) else np.nan)
    return result, ap


def prepare(attempts_path: Path, ap_path: Path, output_dir: Path) -> dict:
    if output_dir.resolve().is_relative_to(HERE.resolve()):
        raise ValueError("Choose an output directory outside observed_pipeline so no dataset/results are stored with the source package.")
    attempts = pd.read_csv(attempts_path)
    ap_events = pd.read_csv(ap_path)
    require_columns(attempts, ATTEMPT_REQUIRED, "attempts table")
    require_columns(ap_events, AP_REQUIRED, "AP table")
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = attempts.copy()
    attempts["typeExam"] = attempts["typeExam"].astype("string").str.upper()
    attempts["started_at"] = pd.to_datetime(attempts["started_at"], utc=True, errors="coerce")
    attempts["submitted_at"] = pd.to_datetime(attempts["submitted_at"], utc=True, errors="coerce")
    parse_numeric(attempts, ["assessment_duration_minutes", "duration_seconds", "total_score",
                             "max_score", "ielts_band_score", "target_score", "real_score"])
    sat_attempt_keys = set(attempts.loc[attempts["typeExam"] == "SAT", "attempt_key"].astype(str))
    attempts = attempts[attempts["typeExam"] != "SAT"].copy()
    if attempts["attempt_key"].duplicated().any():
        n = int(attempts["attempt_key"].duplicated().sum())
        raise ValueError(f"Found {n} duplicate attempt keys in the prepared dataset; resolve them before experiments.")
    if attempts["student_key"].isna().any() or attempts["attempt_key"].isna().any():
        raise ValueError("Prepared attempts must have non-empty attempt_key and student_key values.")

    attempts = calculate_attempt_history(attempts)
    attempts, ap = calculate_ap_history(attempts, ap_events, sat_attempt_keys)
    # Retain only the report's modeling columns; every outcome and assessment
    # attribute comes directly from the external prepared dataset.
    for column in ATTEMPT_OUTPUT_COLUMNS:
        if column not in attempts:
            attempts[column] = np.nan
    attempts[ATTEMPT_OUTPUT_COLUMNS].to_csv(output_dir / "attempt_features.csv", index=False)
    ap[AP_OUTPUT_COLUMNS].to_csv(output_dir / "ap_events.csv", index=False)

    manifest = {
        "pipeline": "external-prepared-dataset",
        "input_files": {
            attempts_path.name: {"sha256": sha256(attempts_path), "rows": int(len(pd.read_csv(attempts_path)))},
            ap_path.name: {"sha256": sha256(ap_path), "rows": int(len(pd.read_csv(ap_path)))},
        },
        "prepared_rows": int(len(attempts)),
        "accounts": int(attempts["student_key"].nunique()),
        "observed_outcomes": {
            "real_score": int(attempts["real_score"].notna().sum()),
            "target_score": int(attempts["target_score"].notna().sum()),
            "difficulty_level": int(attempts["difficulty_level"].notna().sum()),
        },
        "quality_checks": {
            "duplicate_attempt_keys": 0,
            "missing_started_at": int(attempts["started_at"].isna().sum()),
            "missing_submitted_at": int(attempts["submitted_at"].isna().sum()),
            "submitted_at_not_after_started_at": int((
                attempts["submitted_at"].notna() & attempts["started_at"].notna()
                & (attempts["submitted_at"] <= attempts["started_at"])).sum()),
        },
        "point_in_time_rule": "A history or AP event is eligible only when its submitted/event timestamp is strictly before the target attempt started_at.",
        "outputs": ["attempt_features.csv", "ap_events.csv"],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-csv", type=Path, required=True, help="External prepared attempt-level table.")
    parser.add_argument("--ap-events-csv", type=Path, required=True, help="External prepared AP transaction table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output path outside this source package.")
    args = parser.parse_args()
    manifest = prepare(args.attempts_csv, args.ap_events_csv, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
