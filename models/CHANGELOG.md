# Models Changelog

Track every model artefact written to `models/*.pkl`. One row per (re-)train.

Format: `YYYY-MM-DD  <model_name>  <git_sha>  Brier=X  ECE=X  Notes`

| Date | Model | Git SHA | Brier | ECE | Notes |
|------|-------|---------|-------|-----|-------|
| 2026-05-22 | LogisticRegression | 965746a | 0.0636 | 0.0281 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
| 2026-05-22 | XGBoost | 965746a | 0.0639 | 0.0194 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
| 2026-05-22 | LightGBM | 965746a | 0.0645 | 0.0160 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
