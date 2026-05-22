# Models Changelog

Track every model artefact written to `models/*.pkl`. One row per (re-)train.

Format: `YYYY-MM-DD  <model_name>  <git_sha>  Brier=X  ECE=X  Notes`

| Date | Model | Git SHA | Brier | ECE | Notes |
|------|-------|---------|-------|-----|-------|
| 2026-05-22 | LogisticRegression | 965746a | 0.0636 | 0.0281 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
| 2026-05-22 | XGBoost | 965746a | 0.0639 | 0.0194 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
| 2026-05-22 | LightGBM | 965746a | 0.0645 | 0.0160 | Phase 1.4 skeleton — baseline-9 feat, walk-forward+Optuna (raw probs) |
| 2026-05-22 | LogisticRegression | e9ea24b | 0.0649 | 0.0196 | Phase 1.4 — rich feature table (31 num + track_id), walk-forward+Optuna (raw probs) |
| 2026-05-22 | XGBoost | e9ea24b | 0.0629 | 0.0247 | Phase 1.4 — rich feature table (31 num + track_id), walk-forward+Optuna (raw probs) |
| 2026-05-22 | LightGBM | e9ea24b | 0.0631 | 0.0228 | Phase 1.4 — rich feature table (31 num + track_id), walk-forward+Optuna (raw probs) |
| 2026-05-23 | LogisticRegression | 3b682ee | 0.1952 | 0.0337 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-23 | XGBoost | 3b682ee | 0.2009 | 0.0366 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-23 | LightGBM | 3b682ee | 0.2016 | 0.0409 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
