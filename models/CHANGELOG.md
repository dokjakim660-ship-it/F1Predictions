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
| 2026-05-23 | LogisticRegression | f068b15 | 0.0649 | 0.0196 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-23 | XGBoost | f068b15 | 0.0629 | 0.0247 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-23 | LightGBM | f068b15 | 0.0631 | 0.0228 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-23 | Ensemble | f068b15 | 0.0629 | 0.0220 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-23 | Ensemble | f068b15 | 0.2010 | 0.0349 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-24 | LogisticRegression | de449df | 0.1957 | 0.0343 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-24 | XGBoost | de449df | 0.2010 | 0.0313 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-24 | LightGBM | de449df | 0.2009 | 0.0391 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-24 | Ensemble | de449df | 0.2006 | 0.0320 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-05-24 | LogisticRegression | 1cb39ed | 0.0642 | 0.0224 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-24 | XGBoost | 1cb39ed | 0.0631 | 0.0264 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-24 | LightGBM | 1cb39ed | 0.0624 | 0.0240 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-05-24 | Ensemble | 1cb39ed | 0.0625 | 0.0279 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-06-01 | TeamReliability (DNF) | 44c940d | 0.1155 | 0.0057 | Phase 5.2 — DNF; nothing beats it (best LogReg 0.1149, CI straddles 0); ConstantRate 0.1158 level; default deployed |
| 2026-06-02 | LogisticRegression | f743c19 | 0.0658 | 0.0267 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-06-02 | XGBoost | f743c19 | 0.0634 | 0.0244 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-06-02 | LightGBM | f743c19 | 0.0624 | 0.0236 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-06-02 | Ensemble | f743c19 | 0.0626 | 0.0259 | Phase 1.4 — rich features (podium), walk-forward+Optuna (raw probs) |
| 2026-06-02 | LogisticRegression | f743c19 | 0.1956 | 0.0305 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-06-02 | XGBoost | f743c19 | 0.2004 | 0.0280 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-06-02 | LightGBM | f743c19 | 0.2002 | 0.0296 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |
| 2026-06-02 | Ensemble | f743c19 | 0.2000 | 0.0255 | Phase 1.4 — rich features (teammate), walk-forward+Optuna (raw probs) |

### Phase 5.2 DNF sub-model — 2026-06-01 (negative result)

New binary market P(DNF), three timing modes (pre_weekend / post_fp2 / race).
Target `target_dnf`; stack LogReg/XGB/LGBM/Ensemble (default params, no Optuna);
walk-forward OOF → isotonic → holdout Brier. Baselines: ConstantRate +
TeamReliability (1-feature logit on `team_form_dnf_rate_l10`).

- Holdout (827 rows / 41 races, base rate 0.133): no model beats TeamReliability
  (brier_raw 0.1155, brier_cal 0.1149, ECE 0.0057). Best per mode = LogReg
  (~0.1149–0.1152 raw) but every CI straddles 0 (race diff −0.0003,
  CI [−0.0047, +0.0040]); ConstantRate 0.1158 level. Dev-selection picks
  TeamReliability in all three modes. DNF near-unpredictable beyond team
  reliability.
- C1 ablation: a leak-free `pred_dnf` feature does not help the podium ensemble
  (base 0.0638 vs aug 0.0649, diff +0.0010, CI [−0.0009, +0.0029]).
- C2 (DNF betting market) deliberately not built (user can't bet DNFs).
- Deployed default = TeamReliability (best calibration; varies per team).
- New: src/models/{dnf,predict_dnf,compose_dnf}.py, app/app_pages/dnf.py, just
  eval-dnf/select-dnf/predict-dnf/compose-dnf. See PLANNING.md §16.
