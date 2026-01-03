# Copilot / Agent Instructions — Boatrace-Tokoname-ai

Purpose: Help AI coding agents be immediately productive in this repository.

- **Preferred model:** Claude Haiku 4.5 (use for coding, refactoring, and PR suggestions). If unavailable, fall back to GPT-5 mini.

Big picture
- Repo split: top-level utilities (e.g., `predict_tokoname.py`) and the main project in `boatrace-master/`.
- `boatrace-master/` contains crawlers (`src/crawl`), data preparation (`src/data_preparing`), analysis (`src/analyze`), and raw/processed data under `data/` (e.g., `data/beforeinfo`, `data/results_race`). Data flows: crawlers -> `data/` -> data_preparing -> analysis / model training -> `predict_tokoname.py`.

Key files and examples
- `predict_tokoname.py`: loads `tokoname_p1_model.pkl` and `tokoname_p1_features.pkl` via `joblib.load` and expects a features dict matching the saved feature column order. When editing model code, keep the `feature_cols` consistent with saved artifacts.
- `boatrace-master/src/data_preparing/downloader_race_results.py`: runner for downloading race CSVs into `data/beforeinfo/` (CSV filenames are YYYYMMDD-ish patterns). Use it to refresh raw data.
- `boatrace-master/README.md`: project-level notes (read first for context).

Developer workflows (commands observed in the repo)
- Activate venv (Windows):

  ```powershell
  venv\Scripts\activate
  ```

- Run data downloader (from `boatrace-master` root):

  ```powershell
  python src\data_preparing\downloader_race_results.py
  ```

- Run prediction example (top-level):

  ```powershell
  python predict_tokoname.py
  ```

Project-specific conventions and patterns
- Models and feature lists are stored as `joblib` pickles at repo root (example names above). When changing feature engineering, update both the saved `*_features.pkl` and any callers that construct DataFrames by column order.
- CSV naming: many inputs follow a date-like filename (e.g., `190101.csv`) under `data/beforeinfo/` — code assumes these patterns when aggregating.
- Scripts use relative paths and expect to be run from the repository root or `boatrace-master` depending on the script; prefer running from `boatrace-master` for scripts under that folder.

Integration points / external dependencies
- Crawlers fetch live race and odds data and write into `data/` folders. Treat `data/` as the canonical source for downstream steps.
- ML pipelines rely on `joblib`, `pandas`, `numpy` (see `predict_tokoname.py`). Ensure these libs are available in the active venv.

What to watch for when editing
- When changing feature names or order: update `tokoname_p1_features.pkl` and re-export the model or update callers that build DataFrames with `columns=feature_cols` (see `predict_tokoname.py`).
- Avoid changing CSV filename conventions without updating the downloader and any aggregators under `src/data_preparing/`.

Short checklist for PRs from agents
- Read `boatrace-master/README.md` for context.
- Run the relevant scripts locally (activate venv) and verify no path-related failures.
- If updating model inputs, provide a regeneration step for `*_features.pkl` and a short test run of `predict_tokoname.py`.

If anything in these instructions is unclear or you want more detail about CI, env setup, or specific scripts, ask and I will iterate.
