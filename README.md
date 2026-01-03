# Boatrace-Tokoname-ai

Quick notes:

- Loader logging & diagnostics: run `python scripts/diagnose_tokoname.py` to execute the loader and emit diagnostics; a saved snapshot lives in `diagnostics/tokoname_diagnostics.json`.
- Weather preservation: `src/data_preparing/loader.py` now carries TXT-sourced weather columns through merges, with block-level error capture and diagnostics so the pipeline won’t drop weather data silently.
- Feature branch: `feature/add-loader-logging` is published; main already contains the current changes.
