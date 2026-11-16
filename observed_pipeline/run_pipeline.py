#!/usr/bin/env python3
"""Run source-only aggregation, feature engineering, mining, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from prepare_external_dataset import prepare


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-csv", type=Path, required=True, help="External prepared attempt-level dataset.")
    parser.add_argument("--ap-events-csv", type=Path, required=True, help="External prepared AP transaction table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Analysis output directory outside this source folder.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest = prepare(args.attempts_csv.resolve(), args.ap_events_csv.resolve(), output_dir)
    attempts = pd.read_csv(output_dir / "attempt_features.csv")
    if "real_score" in attempts and "target_score" in attempts:
        real_score = pd.to_numeric(attempts["real_score"], errors="coerce")
        target_score = pd.to_numeric(attempts["target_score"], errors="coerce")
        eligible = int((real_score.notna() & target_score.notna()
                        & pd.to_datetime(attempts["started_at"], utc=True, errors="coerce").notna()).sum())
    else:
        eligible = 0

    status = {
        "dataset_rows": len(attempts),
        "rows_with_observed_real_target_and_start": eligible,
        "analysis_status": "ready" if eligible else "prepared_only_missing_observed_outcomes",
        "outcome_counts": manifest["observed_outcomes"],
    }
    (output_dir / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if not eligible:
        print(json.dumps(status, indent=2))
        print(
            "Analysis stopped after cleaning and feature engineering because the external dataset contains "
            "no rows with both real_score and target_score plus started_at. The pipeline does not invent outcomes."
        )
        return 0

    env = os.environ.copy()
    env["OBSERVED_PIPELINE_OUTPUT_DIR"] = str(output_dir)
    subprocess.run([sys.executable, str(HERE / "run_mining_analysis.py")], check=True, env=env)
    subprocess.run([sys.executable, str(HERE / "run_phase3_validation.py")], check=True, env=env)
    status["analysis_status"] = "completed"
    (output_dir / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
