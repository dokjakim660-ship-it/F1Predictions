# F1 Prediction — Planungsdokument

> Konsolidierte Master-Referenz. Erstellt 2026-05-20.
> Stand: Planungsphase abgeschlossen, vor Implementation.

---

## 1. Executive Summary

| | |
|---|---|
| **Projekt** | F1-Prediction-Modell von Grund auf |
| **Use Case** | Lernprojekt + Portfolio + persönlicher Tipp/Wett-Einsatz |
| **Erfolgs-Nordstern** | Positives EV langfristig — den Buchmacher-Markt (Tipico) schlagen |
| **MVP-Ziel** | Pre-Race-Modell für Podium (P1–P3) + Head-to-Head Teammate, probabilistisch |
| **Stack** | Python 3.11+, FastF1, XGBoost+LightGBM, Optuna, MLflow, HuggingFace Spaces |
| **Wochenbudget** | 8–12h, "solide statt schnell" |
| **MVP-Ende-Schätzung** | Mitte Juli 2026 |
| **Phase-4-Ende (ROI-Backtest)** | Q3/Q4 2026 |

---

## 2. MVP-Scope

**Was wird vorhergesagt:**
- **Primär:** Podium-Wahrscheinlichkeit pro Fahrer (P(Top 3))
- **Sekundär:** Head-to-Head Teammate (P(Driver A > Driver B), pro Team)

**Wann:** Pre-Race (nach Qualifying), Samstag-Abend.

**Output-Form:** Probabilistisch (0.0–1.0 pro Fahrer), nicht "wer gewinnt".

**Bewusst NICHT im MVP:** Race-Winner (zu unbalanciert), Finishing-Position (Ranking, später), Pre-Quali-Modell (Phase 3), Live-Updates (Phase 5+).

---

## 3. Datenquellen — Stack

| Quelle | Rolle | Begründung |
|---|---|---|
| **FastF1** | Primär: Quali/Race/Sektoren/Telemetrie | Beste Tiefe, ab 2018 zuverlässig |
| **Jolpica-F1** (Ergast-Fork) | Sekundär: Standings, Schedule, DNF-Status | Lange Historie, einfache API |
| **Open-Meteo** | Wetter (Historical Forecast + Live Forecast) | Forecast ist Pflicht für Pre-Race ohne Leakage |
| Bookmaker-Quoten | Phase 4 (Backtest) | Verschoben — The Odds API + manuell |

**Backtest-Range:** 2018–heute (FastF1-Telemetrie zuverlässig ab 2018).
**Kritischer Bruch im Range:** 2022 Ground-Effect-Reglement → Era-Feature + Sample-Weight-Decay.

---

## 4. Feature Engineering

**Granularität:** Eine Zeile = ein Fahrer × ein Rennen (long format).

**Endmenge:** ~55–70 Features (final), **~30–35 im MVP**.

**Feature-Kategorien:**

| Kategorie | Beispiele | MVP |
|---|---|---|
| A) Form Driver | quali_pos_last, race_pos_roll5, podium_count_last10, dnf_rate_last10 | ✅ |
| B) Team | team_standing, team_pace_gap_pole_roll5, team_dnf_rate_last10 | ✅ |
| C) Race-Specific (post-quali) | quali_pos_thisrace, quali_time_gap_pole, **long_run_pace_fp2** ⭐, tyre_choice_q3 | ✅ |
| D) Track (manuelle Attribute) | track_length, track_type, track_character, track_id | ✅ |
| E) Driver × Track | driver_track_avg_pos_last3 | ✅ |
| F) Wetter | weather_rain_prob_forecast, wet_race_flag | ✅ |
| G) Saison/Zeit | race_number_in_season, is_sprint_weekend | ✅ |
| H) Era | era_2022plus | ✅ |
| Sub-Modelle DNF (Phase 2+) | P(DNF) als Feature | ⏸ |

**Der heimliche Star:** Long-Run-Pace aus FP2 (aggregierte Lap-Times pro Stint und Compound). Stärker als Quali allein.

**Pflicht-Regeln:**
- Alle Form-Features mit `shift(1)` lagged
- DNF-Status: "letzte gefahrene Position" + `is_dnf` Boolean (nicht naiv P20)
- Team-Features an Team-Historie, Driver-Features an Driver-Historie (für Fahrer-Wechsel)
- Pre-Race-Modell: Quali OK, Race-Daten NICHT. Pre-Quali-Modell: Quali NICHT.

**User-Hypothese (zu validieren):** Auto/Team-Pace dominant über Driver-Track-Historie. Soll von Modell empirisch geprüft werden.

---

## 5. Architektur (4-Layer Medallion-light)

```
SOURCES → Layer 1 RAW → Layer 2 PROCESSED → Layer 3 FEATURES → Layer 4 MODELS

FastF1 ─┐
Jolpica ─┼──► data/raw/* ──► data/processed/*.parquet ──► data/features/*.parquet ──► models/*.pkl
Meteo  ─┤      (fastf1                                         (eine Zeile pro
Tracks ─┘       cache lokal                                     Driver × Race)
                gitignored)
```

**Repo-Struktur (Blueprint):**

```
F1Predictions/
├── pyproject.toml (uv), justfile, .gitignore, README.md
├── .github/workflows/refresh.yml  ← Phase 3
├── data/{raw,processed,features,reference}/
├── notebooks/                     ← EDA
├── src/
│   ├── ingest/   {fastf1,jolpica,openmeteo}_ingest.py
│   ├── process/  clean_results.py, join_layer2.py
│   ├── features/ form, team, race_specific, track, weather, build
│   ├── models/   train_podium, train_teammate, predict, baseline
│   ├── eval/     metrics, walk_forward, calibration
│   └── utils/
├── models/  *.pkl + CHANGELOG.md + CURRENT.json
├── predictions/  YYYY-MM-circuit.csv
└── tests/  test_features.py, test_no_leakage.py
```

**Tooling:** `uv` (env), `just` (Pipeline-Runner, Windows-tauglich), DuckDB (SQL/Joins), pandas+pyarrow, pytest.

**Versionierung:** Git only (kein DVC im MVP). Parquet im Repo bis <1GB. Modelle mit CHANGELOG.

---

## 6. Modell-Stack

| Modell | Rolle | Bibliothek |
|---|---|---|
| DummyClassifier (stratified) | Trivial-Baseline | sklearn |
| F1-Domain-Baseline "Top-3-Quali = Podium" | F1-Sanity-Check ⭐ wichtigste Hürde | eigene Implementierung |
| LogisticRegression | Lineare Baseline | sklearn |
| **XGBoost** | Hauptmodell | xgboost |
| **LightGBM** | Zweites Tree-Ensemble | lightgbm |

**Bewusst NICHT:** Random Forest, CatBoost (MVP), Neural Nets (zu wenig Daten), SVM, KNN, AutoML.

**Klassen-Imbalance Podium:** `scale_pos_weight ≈ 5.7` (XGB) / `is_unbalance=True` (LGBM).
**Wichtig:** danach **Calibration-Layer (Isotonic) Pflicht** — sonst sind Wahrscheinlichkeiten unbrauchbar für EV.

**Hyperparameter-Tuning:** Optuna (TPE), 50–100 Trials, mit SQLite-Persistence. Auf Walk-Forward (kein Random K-Fold!) optimiert.

**Ensemble:** Erst Phase 2+, nach Korrelations-Check der Single-Model-Predictions.

---

## 7. Evaluation & Validierung

**Optuna-Objective:** `mean(Brier) − 0.2·std(Brier)` über Folds (Stabilität gewünscht).

**Validation-Strategie:** Hybrid Walk-Forward — Expanding Window + Time-Decay (`sample_weight = 0.97^months_old`).

```
TRAIN/VAL (Optuna sieht):
Fold 1: 2018-01→2022-06  val→2022-12
Fold 2: 2018-01→2022-12  val→2023-06
Fold 3: 2018-01→2023-06  val→2023-12
Fold 4: 2018-01→2023-12  val→2024-06

TEST (Optuna sieht NIE):
Train [2018→2024-06]  Test [2024-07→2024-12]
Train [2018→2024-12]  Test [2025-01→2025-12]
```

**Pflicht-Metriken:** Brier (Optuna), LogLoss, AUC-ROC, AP/PR-AUC, **ECE**, Top-3 Accuracy, Brier Skill Score.

**Phase-4-Metriken (mit echten Quoten):** ROI, Edge, Hit-Rate vs. Implied, Sharpe, Max Drawdown, Kelly-Stake-Distribution.

**Kalibrierungs-Workflow:**
1. Modell mit `scale_pos_weight` trainieren
2. Isotonic Regression auf Val-Slice fitten
3. Bei Inferenz: raw probs → calibrator → calibrated probs
4. Reliability-Plot als Pflicht-Visualisierung

**Race-Group-Integrität:** NIEMALS Race-IDs in Train und Val gleichzeitig.

**Statistische Signifikanz beim Modell-Vergleich:** Paired Bootstrap (1000 Resamples) für 95 % CI auf Brier.

**EV-Backtest-Stub im MVP:** Backtest-Code mit synthetischen fairen Quoten bauen (Pipeline-Test + Upper-Bound). Nicht als Optuna-Objective.

---

## 8. MLOps & Deployment

| Komponente | Tool | Begründung |
|---|---|---|
| Experiment Tracking | **MLflow lokal** | Industry-Standard, lokal, gratis |
| HP-Storage | **Optuna SQLite** | Persistent, parallel zu MLflow |
| Modell-Versionierung | Git Tags + `models/CHANGELOG.md` + `CURRENT.json` | Pragmatisch |
| Reproduzierbarkeit | `uv.lock` | Pinned deps |
| UI / Deployment | **HuggingFace Spaces + Streamlit (Multi-Page)** | Portfolio + 16GB RAM gratis |
| CI/Cron | GitHub Actions Workflow `refresh.yml` (Phase 3) | Sonntag 22:00 UTC |

**Re-Training-Cadence:** Hybrid.
- Feature-Build: wöchentlich (Sonntag)
- Inferenz/Predict: wöchentlich (Samstag nach Quali)
- Modell-Retraining: **monatlich** (Stabilität > Aktualität)
- Optuna Hyperparameter-Suche: quartalsweise oder bei Performance-Drop

**HF Spaces ist Inferenz-only.** Kein FastF1 dort. Lokal trainieren, Modell + Predictions als CSV nach HF pushen.

**App-Tabs (Streamlit Multi-Page):**
1. Next Race — aktuelle Predictions
2. Backtest History — Predictions vs. Actual + Brier-Trend
3. Feature Importance — SHAP / XGBoost-Importance
4. Methodology — Portfolio-Text

---

## 9. Roadmap

| Phase | Wochen | Inhalt | Geschätztes Ende |
|---|---|---|---|
| 0 — Setup | 1 | Repo, uv, just, FastF1-Test | Ende Mai 2026 |
| 1 — MVP | 5–6 | Ingest, Features, Baselines, XGB+LGBM, MLflow, Calibration | Mitte Juli 2026 |
| 2 — HF Spaces App | 3–4 | Streamlit Multi-Page, HF Deployment | Anfang August 2026 |
| 3 — Pre-Quali-Modell | 3 | Zweites Modell ohne Quali-Features, A/B vs. Pre-Race | Ende August 2026 |
| 4 — Quoten + ROI | 4–6 | The Odds API + manuell, ROI-Backtest, Kelly | Sep/Okt 2026 |
| 5+ | open-ended | Ranking-Modell, DNF-Sub-Modell, Ensemble, Pole, Top-6/10, Live | ab Q4 2026 |

**Was schon nach Phase 1 nutzbar ist:**
- ✅ Eigene Tipps fürs F1-Tippspiel
- ✅ Intuition-vs-Modell-Vergleich
- ✅ Brier-vs-Baselines-Check
- ⏸ UI fehlt (P2), Pre-Quali (P3), ROI-Aussage (P4)

---

## 10. Kritische Erfolgskriterien

1. Modell schlägt **"Top-3-Quali = Podium"-Baseline** auf Test-Set (Phase 1)
2. **Calibration-Plot** zeigt saubere Linie nahe der Diagonalen (ECE < 0.05)
3. App auf HF Spaces zeigt aktuelle Predictions reproduzierbar (Phase 2)
4. ROI-Backtest mit echten Quoten ergibt klare Antwort auf "schlagen wir Tipico?" (Phase 4)
5. Code-Reproduzierbarkeit: `just refresh && just train && just predict` läuft frisch durch

---

## 11. Top-Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|---|---|---|---|
| Modell schlägt Top-3-Quali nicht | Mittel | Re-Design nötig | Long-Run-Pace ist die Hoffnung; ggf. mehr Feature-Iteration |
| FastF1-Cache-Setup-Hölle | Mittel | 2–3 Tage Verzug | Inkrementell ziehen, Cache-Pfad früh fixieren |
| Quoten-Sourcing rechtlich grau | Hoch | Phase 4 fragwürdig | The Odds API Free + manuelles Sammeln |
| Concept Drift 2026-Reglement | Hoch | Modell driftet | Era-Feature + Sample-Weight-Decay; ab P4 evaluieren |
| Data Leakage (Lookahead in Roll-Features) | Mittel | Test-Performance illusorisch | `tests/test_no_leakage.py` in CI |
| `scale_pos_weight` zerstört Kalibrierung | Hoch (Default) | Kelly-Sizing schief | Calibration-Layer als Pflicht-Step |

---

## 12. Nächste konkrete Schritte (Phase 0, jetzt)

1. **Python 3.11+** installieren (falls noch nicht): https://python.org
2. **uv** installieren: `winget install astral-sh.uv` (oder `pipx install uv`)
3. **just** installieren: `winget install Casey.Just` (Windows-Pipeline-Runner)
4. **VS Code** mit Python-Extension einrichten
5. Repo initialisieren: `git init`, `uv init`, Grund-Verzeichnisse (`src/`, `data/`, etc.)
6. `pyproject.toml` mit Basis-Deps: fastf1, pandas, pyarrow, duckdb, xgboost, lightgbm, scikit-learn, optuna, mlflow, streamlit, pytest
7. `justfile` mit ersten Targets: `setup`, `lint`, `test`
8. `.gitignore` (`data/raw/fastf1_cache/`, `mlruns/`, `.venv/`, `*.pyc`)
9. FastF1 erstes Sample testen: einen Race (z. B. Monaco 2024) ziehen und Lap-Times inspizieren

→ Wenn Phase 0 steht: nächste Session über Phase-1-Implementierung starten.

---

## Memory-Referenzen (für Folge-Sessions)

Die detaillierten Entscheidungen liegen in `C:\Users\morit\.claude\projects\C--Projekte-F1Predictions\memory\`:
- `user_profile.md` — User-Profil
- `project_f1_predictions.md` — Scope, MVP, "Tipico schlagen"
- `project_feature_decisions.md` — Feature-Engineering-Details
- `project_architecture.md` — Pipeline & Tooling
- `project_models.md` — Modell-Auswahl & Tuning
- `project_evaluation.md` — Validation & Metriken
- `project_mlops.md` — Deployment & MLOps
- `project_roadmap.md` — Phasenplan
