# Boatrace-Tokoname-ai

Quick notes:

- Loader logging & diagnostics: run `python scripts/diagnose_tokoname.py` to execute the loader and emit diagnostics; a saved snapshot lives in `diagnostics/tokoname_diagnostics.json`.
- Weather preservation: `src/data_preparing/loader.py` now carries TXT-sourced weather columns through merges, with block-level error capture and diagnostics so the pipeline won’t drop weather data silently.
- Feature branch: `feature/add-loader-logging` is published; main already contains the current changes.

## Phase 2 (Factor × Feature Binding)
- Master directive (interpretability-first, no training): see [docs/phase2_directive.md](docs/phase2_directive.md).
- Run Phase 2 only (loads Tokoname, binds factors, writes report, no modeling):
	- `python run_phase2.py --venue 常滑`
- Report output: [reports/phase2_factor_binding_report.md](reports/phase2_factor_binding_report.md) (created by the run command).
