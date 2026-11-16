# Report experiment pipeline — source only

This package contains the code to run the report's preparation and experiment stages. It reads the prepared dataset supplied externally at runtime.

## Pipeline

1. Read the external attempt-level table and AP transaction table.
2. Check required columns, duplicate keys, missing timestamps, and invalid start/submit order. Duplicate attempt keys stop the run; missing values remain missing. Invalid chronology is reported, not repaired.
3. Exclude SAT attempts and their linked AP records, and omit video-view/completion AP events, matching the report scope.
4. Recalculate attempt history and AP features from events strictly before each attempt's `started_at`.
5. Run the report analyses: descriptive summaries, correlations, conditional probabilities, frequent itemsets, association and skill-sequence rules, temporal train/validation/test evaluation, cross-validation, regression/classification comparisons, and Phase 3 validation.

## Run end to end

The external data directory should provide the prepared tables, including the report's per-attempt fields and AP transaction records:

- `attempt_features.csv` (or another attempt-level CSV with the required columns below)
- `ap_events.csv`

Install `pandas`, `numpy`, `scipy`, and `scikit-learn`, then run:

```bash
python3 research/data_mining/observed_pipeline/run_pipeline.py \
  --attempts-csv /path/to/prepared-dataset/attempt_features.csv \
  --ap-events-csv /path/to/prepared-dataset/ap_events.csv \
  --output-dir /path/outside/observed_pipeline
```

`output-dir` must be outside this source package. The runner re-creates the analysis-ready tables there, then executes both experiment scripts against that directory. It never modifies the external input files.

## Input contract

The attempt table must include `attempt_key`, `student_key`, `assessment_key`, `typeExam`, `mode`, `difficulty_level`, `started_at`, `submitted_at`, `assessment_duration_minutes`, `real_score`, and `target_score`. Existing `total_score`, `max_score`, `ielts_band_score`, and `duration_seconds` are also retained when present. `real_score` and `target_score` remain the observed business outcomes used by the report; the class label is calculated as `real_score >= target_score`.

The AP table must include `student_key`, `event_at`, `event_type`, `scope`, `amount`, `reference_type`, and `reference_key`. AP award counts and amounts use `OVERALL` rows as the canonical one-row-per-award view; recency may use any eligible scope, consistent with the report's feature definitions.

If either outcome column is absent or no complete labeled rows have `started_at`, the pipeline writes `pipeline_status.json` and stops after preparation. `dataset_manifest.json` records input hashes, row counts, quality checks, and output field counts. Experiment tables and Phase 3 outputs are written under `output-dir/mining/`.

## Files

- [prepare_external_dataset.py](prepare_external_dataset.py): external table validation, cleaning, and point-in-time feature engineering.
- [run_mining_analysis.py](run_mining_analysis.py): report's data mining, CV, and modeling calculations.
- [run_phase3_validation.py](run_phase3_validation.py): leakage audit, unseen-account evaluation, and error/threshold analysis.
- [run_pipeline.py](run_pipeline.py): end-to-end orchestration.

