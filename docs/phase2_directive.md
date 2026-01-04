# Phase 2 Master Directive

Interpretability-first Tokoname system. Prediction is subordinate to human judgment.

## Core Philosophy
- Goal: improve human decision quality; accuracy is a byproduct.
- Explainability beats optimization when in conflict.
- No black-box-only ML; no automated betting.

## Scope (Phase 2)
- Build factor mapping (raw columns -> 9 factors) with coverage reporting.
- No model training, probabilities, or betting logic.
- Venue: Tokoname only until structure is stable.

## Factors (fixed set)
1. Race Environment
2. Frame & Course
3. Racer Skill
4. Motor Performance
5. Boat
6. Start Timing
7. Race Scenario / Development
8. Risk & Reliability
9. Odds & Value (may be empty until odds are available)

## Deliverables
- `factors_config.yaml`: canonical factor definitions with raw columns, aliases, and derived features.
- `src/factors/factor_binder.py`: loads config, resolves aliases, computes derived features, and returns factor coverage and a factor-organized dataframe.
- `scripts/phase2_factor_report.py`: generates markdown coverage report (rows, date range, venue counts, factor coverage, sample derived features) to `reports/phase2_factor_binding_report.md`.
- `run_phase2.py`: CLI entrypoint that loads Tokoname data, binds factors, emits the report, and stops (no training).

## Rules
- Do not remove columns “to simplify.”
- If a factor has zero columns, mark it BLOCKED but continue; if all core factors (1–8) are BLOCKED, stop with a clear error.
- Derived features: deterministic only; skip with a warning if required columns are missing.
- Normalize venue (常　滑 → 常滑) and column aliases before binding.
- Prefer explicit logging and fail-fast validation.

## Exit Criteria (Phase 2)
- Factors documented and reproducible.
- Coverage report shows no BLOCKED core factors (1–8) and at least 1,000 Tokoname rows.
- Report is beginner-friendly: shows what was found, what is missing, and why it matters.

## How to Run Phase 2
```
python run_phase2.py --venue 常滑
```
(Optional) add `--save-csv out/factor_bound.csv` to inspect outputs.

## Troubleshooting (Top 5)
1) No data loaded: ensure boatrace-master/data/results_race/*.TXT exist; run downloader.
2) Zero Tokoname rows: check venue normalization; inspect `df['venue_clean'].unique()`.
3) Core factors BLOCKED: confirm column names/aliases in `factors_config.yaml` match the dataframe.
4) Derived features missing: logs will show skipped features; add required columns or adjust aliases.
5) Report not written: confirm `reports/` exists or is creatable; check script permissions.
