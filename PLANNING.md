# F1 Prediction — Planungsdokument

> Konsolidierte Master-Referenz. Erstellt 2026-05-20.
> Stand: Phasen 0–2 ✅ DONE (2026-05-23). Phase 3 startet mit Plan-Abweichung gegenüber Original-Roadmap — Next-Race-Predict-Pipeline statt Pre-Quali-Modell (Pre-Quali rutscht auf Phase 4). Begründung + Sub-Plan in §13.

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

| Phase | Wochen | Inhalt | Status / Geschätztes Ende |
|---|---|---|---|
| 0 — Setup | 1 | Repo, uv, just, FastF1-Test | ✅ DONE Mai 2026 |
| 1 — MVP | 5–6 | Ingest, Features, Baselines, XGB+LGBM, MLflow, Calibration | ✅ DONE 2026-05-23 |
| 2 — HF Spaces App | 3–4 | Streamlit Multi-Page (Methodology, Backtest, Importance), HF Deployment | ✅ DONE 2026-05-23 |
| **3 — Next-Race-Predict-Pipeline** | 3–4 | Ingest "next-only", Feature-Row für zukünftiges Rennen, `just predict-next`, Streamlit "Next Race"-Page, Sprint-WE-Verifikation. Details §13. | ✅ DONE 2026-05-24 |
| **4 — Quoten/ROI + Pre-Quali-Modell** ⏳ LAUFEND | 6–8 | 4.1 ROI-Loop (manuelle Quoten, Kelly, Post-Race-Eval) ✅; 4.2 Pre-Quali-Modell-Linie (4 Quali-Märkte) ✅ inkl. Composition-A/B (Negativ). Details §14. | Q3 2026 |
| 5+ | open-ended | Ranking-Modell, DNF-Sub-Modell, Pole, Top-6/10, Live | ab Q4 2026 |

**Plan-Abweichung Phase 3** (2026-05-23): Ursprünglich war Phase 3 das Pre-Quali-Modell. Beim Push-Review vor Phase 3 fiel auf, dass die Pipeline zwar sprint-robust ist (Ingest/Process/Features tolerieren fehlendes FP2), aber kein End-to-End-Predict-Pfad für ein zukünftiges Rennen existiert — die App zeigt nur historische Holdout-Predictions. Das war der eigentliche Nutzwert des Projekts ("Sa-Abend nach Quali Tipico-Quoten vergleichen"). Pre-Quali-Modell ist obendrein nur sinnvoll, wenn die Predict-Strecke einmal sauber steht, also gehört es architektonisch hinter Phase 3.

**Was schon nach Phase 2 (jetzt) nutzbar ist:**
- ✅ Brier-/Calibration-Validierung auf Holdout 2024.5–2025
- ✅ Intuition-vs-Modell-Vergleich auf historischen Rennen (Backtest-Page)
- ✅ Portfolio-fähige App auf HF Spaces
- ⏸ "Quali laden → Predictions für *das nächste* Rennen" (P3)
- ⏸ Pre-Quali-Tipps (P4)
- ⏸ ROI-Aussage "schlagen wir Tipico?" (P4)

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

## 13. Phase 3 — Next-Race-Predict-Pipeline (Sub-Plan)

**Ziel:** Sa-Abend nach Quali einen Befehl ausführen und die Podium-/Teammate-Wahrscheinlichkeiten für das *nächste* Rennen in der App sehen. Sprint-Wochenenden inklusive (FP2 fehlt → `has_fp2=0`, NaN-tolerant durch XGB/LGBM).

**Aktuelle Lücke:** Ingest/Process/Features sind bereits robust gebaut. Was fehlt: ein Inferenz-Pfad, der eine Feature-Row für ein noch-nicht-stattgefundenes Rennen produziert und das calibrated Modell aus Phase 1.5 darauf anwendet. `justfile predict RACE` ist noch ein Stub.

**Sub-Phasen:**

| Sub | Inhalt | Artefakte |
|---|---|---|
| **3.1 Ingest "next-only"** | Leichtgewichtiger Modus: für ein einzelnes zukünftiges Rennen nur `Q` + (`FP2` wenn vorhanden) + Wetter-Forecast ziehen. Keine `R`-Session, da noch nicht stattgefunden. | `just ingest-next YEAR ROUND` |
| **3.2 Feature-Row für zukünftiges Rennen** | Build muss eine Race-Zeile bauen, in der Race-Outcome-Spalten (Podium, DNF, Teammate-Beat) bewusst NaN sind, aber alle Pre-Race-Features sauber gefüllt. Rolling-Form-Features ziehen aus der letzten kompletten Race im Inventory. Strikte Leakage-Garantie: keine Daten der Ziel-Race-ID in irgendeinem Feature. | `data/features/next_race.parquet`, neuer Test `test_no_leakage_next.py` |
| **3.3 Predict-Script + just-Target** | Calibrated Ensemble-Estimator von Phase 1.5 laden, `predict_proba` auf die Next-Race-Row, Output als Parquet + CLI-Tabelle. | `just predict-next YEAR ROUND`, `predictions/next_race.parquet` |
| **3.4 Streamlit "Next Race"-Page** | Default-Tab der App. Tabelle: Fahrer · Team · Grid · P(Podium) · P(beat Teammate). Hinweis "Sprint-WE, keine FP2-Pace" wenn `has_fp2=0`. Auf HF Spaces deploybar via existierender `sync_to_hf.py`. | `app/app_pages/next_race.py` |
| **3.5 Sprint-WE-Verifikation** | Echter Trockenlauf auf historischem Sprint-WE (z. B. 2024 Brasilien oder Austin): ingest-next → feature-row → predict läuft fehlerfrei, NaN-FP2 wird sauber als 0 propagiert, Predictions plausibel. | dokumentierter Lauf im `models/CHANGELOG.md` |

**Geschätzter Aufwand:** 3–4 Wochen bei 8–12h/Wo. Größter Brocken ist 3.2 — Feature-Row für noch-nicht-existierendes-Rennen ohne Leakage; bestehender Feature-Build geht vom Long-Format "Driver × *vergangenes* Race" aus.

**Erfolgskriterien Phase 3:**
1. `just predict-next 2026 <round>` läuft frisch durch (ohne Cache, ohne manuelle Edits) und schreibt eine plausible Predictions-Tabelle.
2. Sprint-WE-Verifikationslauf (3.5) zeigt: keine FP2 → keine Pipeline-Errors → Predictions weichen vom Normal-WE in vorhersehbarer Richtung ab (FP2-Features NaN, alle anderen identisch).
3. "Next Race"-Page auf HF Spaces live, Default-Tab.
4. `tests/test_no_leakage_next.py` grün — keine Ziel-Race-Daten in den Features.

---

## 14. Phase 4 — Quoten/ROI + Pre-Quali-Modell (Sub-Plan)

**Ziel:** Den eigentlichen Projekt-Treiber bedienen — "schlagen wir Tipico?" — über einen geschlossenen Wett-Loop (Quoten eintragen → Kelly-Sizing → Post-Race-P&L) und eine Pre-Quali-Vorhersage, damit man schon vor dem Qualifying tippen kann.

**Plan-Abweichung:** Die Reihenfolge gegenüber der Roadmap-Tabelle wurde getauscht — der ROI-Loop (4.1) kam vor dem Pre-Quali-Modell (4.2), weil der Wett-Loop unabhängig vom Pre-Quali-Modell nutzbar ist (er sizet die bestehenden Pre-Race-Märkte Podium/Teammate) und damit sofort Mehrwert liefert.

### 4.1 ROI-Loop ✅ DONE 2026-05-25

Manueller Quoten-Workflow statt API: The Odds API führt kein F1, Tipico-Scraping bräuchte Playwright gegen Akamai. Bei ~20 Rennen/Jahr × ~2 Min Eingabe ist manuell vertretbar.

| Baustein | Inhalt |
|---|---|
| Quoten-Persistenz | Stakes-Page "Save odds" schreibt `data/odds/{race_id}_{target}.json` und archiviert die Prediction nach `predictions/archive/`. |
| Post-Race-Eval | `just post-race YEAR ROUND` zieht Ergebnisse, settled die Wetten, schreibt `data/roi/log.parquet`. |
| ROI-Tracker-Page | Laufende P&L-Anzeige aus dem Log. |

**Workflow:** Sa-Abend `just predict-next` → Stakes-Page Quoten eintragen + Save → So-Abend `just post-race`. Erstes Live-Rennen: Monaco 2026-06-07.

### 4.2 Pre-Quali-Modell-Linie ✅ DONE 2026-05-30

Vorhersage der vier **Qualifying-Märkte** (`pole`, `top3_quali`, `top10_quali`, `teammate_quali`) VOR dem Qualifying, in zwei Timing-Modi (`pre_weekend` / `post_fp2`). Default-Modell LogReg (robustester bei diesem N).

| Sub | Inhalt | Commit |
|---|---|---|
| 4.2.2/4.2.3 | Pre-Quali-Modell-Driver + Holdout-Eval | bd6c4ac |
| 4.2.4a–c | Dev-basierte Modellwahl, Next-Race-Feature-Builder, Inferenz + just-Targets | e4f5a21..a09dd00 |
| 4.2.5 | Streamlit Pre-Quali-Page (Modus-Vergleich, FP2-Delta) | 0643c63 |
| 4.2.6 | Pre-Quali-Stakes-Page + Shared-Kelly-Helper `src/utils/kelly.py` | fb95f8d |
| 4.2.7 | Quali-Markt-ROI: `just post-quali` / `roi run-quali` (settled gegen Feature-Table-Targets) | 1f44e33 |
| 4.2.8 | Composition-A/B: Podium-Modell mit predicted statt echter Quali (`just compose-prequali`) | d8740a5 |

**Erfolgskriterium / 4.2.8-Resultat (NEGATIV):** A/B-Holdout (827 Rows, 41 Races). Ein Pre-Race-Podium-Modell mit VORHERGESAGTER Quali (post_fp2-Features + 4 leakage-safe OOF-Pre-Quali-Probs) verliert signifikant gegen das Modell mit ECHTER Quali: Ensemble-Brier_raw 0.0872 (composed) vs. 0.0638 (real_grid), diff +0.0233, 95 % CI [+0.0131, +0.0330]. Es schlägt nicht einmal die triviale Top3Quali-Grid-Regel (0.0752; diff +0.0120, CI [−0.0024, +0.0267]). Bestätigt das Phase-1-Ergebnis empirisch: Grid-Position trägt das Podium-Signal — sobald die Quali vorhergesagt statt bekannt ist, bricht es weg.

**Konsequenz:** Pre-Quali taugt NICHT als Eingang für ein Podium-Modell. Der Nutzwert der Pre-Quali-Linie liegt in den **eigenständigen Quali-Märkten** (pole/top3/top10/teammate-Q) plus deren ROI-Loop (`post-quali`), nicht im Komponieren eines Podium-Modells aus vorhergesagter Quali.

### Offen / als Nächstes

- Live-Betrieb des Wett-Loops ab Monaco 2026-06-07 (Pre-Race + Pre-Quali-Märkte), erste echte ROI-Datenpunkte sammeln.
- DNF-Sub-Modell + Live-Updates bleiben offen (Phase 5+).

---

## 15. Phase 5 — Ranking-Modell (volle Grid-Reihenfolge, Quali + Race)

**Ziel:** Jedem Fahrer eine *exakte Platzierung* vorhersagen statt nur binärer Outcomes — einmal fürs Qualifying, einmal fürs Rennen. Bewusst eine andere ML-Aufgabe (Position 1..N statt 0/1, Rank-Metriken statt Brier, keine Kalibrierung); das ist der Lernwert.

**Zwei symmetrische Tasks** (passend zu den bestehenden Timing-Kontexten):

| Task | Ziel (`build.py`) | Features | Inferenz-Input | Baseline |
|---|---|---|---|---|
| quali | `target_quali_rank` (1..N) | Pre-Quali-Sets (pre_weekend / post_fp2) | `next_race_prequali.parquet` | RecentQualiForm (recent avg quali pos) |
| race | `target_race_rank` (1..N, DNF-aware) | volle `FEATURE_COLUMNS` | `next_race.parquet` | GridOrder (Startaufstellung) |

**Zwei ML-Methoden A/B-verglichen** (User-Entscheidung): (A) **Position-Regression** (Ridge/XGB/LGBM-Regressor + Ensemble → pro Rennen sortieren) vs. (B) **Learning-to-Rank / LambdaMART** (XGBRanker `rank:ndcg` / LGBMRanker `lambdarank`, `group=race`). Bewertet mit Positions-MAE, Spearman, Top-k; A/B-Verdikt per paired race-bootstrap CI auf der MAE-Differenz.

**Holdout-Resultate (827/823 Rows, 41 Races):**
- **race:** GridOrder-Baseline sehr stark (pos_mae **3.30**); bestes Modell Ridge 3.24 — **nicht signifikant** (CI [-0.22, +0.10]). Bestätigt den Phase-1-Befund empirisch erneut: die Startaufstellung trägt das Renn-Signal, Modelle holen kaum etwas obendrauf.
- **quali pre_weekend:** Ridge **3.42** schlägt RecentQualiForm 3.57 bei 95 %.
- **quali post_fp2:** LGBMReg **3.28** schlägt Baseline bei 95 %; FP2 hilft (3.42 → 3.28).
- **A/B-Verdikt:** **Position-Regression schlägt LambdaMART** auf Positions-MAE in jedem signifikanten Fall. Die Ranker gewinnen die Top-k-Set-Metriken (top1/top3 — sie optimieren die Reihenfolge direkt), platzieren das Mittelfeld aber ungenauer. Deployter Default: **Ridge** (dev-selektiert für race + pre_weekend, am interpretierbarsten).

**Neue Dateien:** `src/eval/rank_metrics.py`, `src/models/rank.py` (eval + select), `src/models/predict_rank.py` (Inferenz → `predictions/next_race_rank_{quali,race}.parquet`), `app/app_pages/rank.py` (Page „Grid & Finish Order"), `tests/test_rank_metrics.py` + `tests/test_rank_targets.py`. just-Targets: `eval-rank`, `select-rank`, `predict-rank YEAR ROUND`.

---

## 16. Phase 5.2 — DNF-Sub-Modell (P(Fahrer fällt aus))

**Stand: ✅ DONE 2026-06-01 (Negativ-Resultat).** Eigenständiger Binär-Markt P(DNF), drei Timing-Modi (`pre_weekend` / `post_fp2` / `race`). Aufbau spiegelt die Pre-Quali-Linie: neues Target `target_dnf` (= `dnf` als float, NaN-sicher für die next-race-Synth-Zeilen — in **beiden** next-race-Buildern genullt), Stack LogReg/XGB/LGBM/Ensemble mit Default-Params (kein Optuna — Phase-3.7-Lehre), walk-forward OOF → isotonic → Holdout-Brier. Zwei Baselines: ConstantRate (Brier-Floor) + **TeamReliability** (1-Feature-LogReg auf `team_form_dnf_rate_l10`, das DNF-Analog zu RecentQualiForm).

| Stufe | Inhalt | Artefakt |
|---|---|---|
| A1 | Target + Guards | build.py `TARGET_DNF`, beide next-race-Builder nullen es, `tests/test_dnf_target.py` |
| A2 | Modell + Eval + Verdikt | `src/models/dnf.py`, just `eval-dnf` / `select-dnf` |
| B1 | Inferenz | `src/models/predict_dnf.py` → `predictions/next_race_dnf.parquet`, just `predict-dnf` |
| B2 | Streamlit-Page | `app/app_pages/dnf.py` „DNF Risk" + Nav + HF-README |
| C1 | Integrations-Ablation | `src/models/compose_dnf.py`, just `compose-dnf`, `tests/test_compose_dnf.py` |

**Verdikt (Holdout, 827 Zeilen / 41 Rennen, Base-Rate 0.133):** Kein Modell schlägt die TeamReliability-Baseline (brier_raw 0.1155, brier_cal 0.1149, beste Kalibrierung ECE 0.0057). Bestes Modell je Modus = LogReg (brier_raw 0.1149–0.1152), aber **diff zu TeamReliability nicht signifikant** (CIs überspannen 0; race diff −0.0003, CI [−0.0047, +0.0040]). ConstantRate (0.1158) praktisch gleichauf. Dev-Selektion wählt TeamReliability in allen drei Modi. Mehr Wochenend-Signal (post_fp2/race) hilft nicht. → Ausfälle sind jenseits der Team-Zuverlässigkeitsrate **nicht vorhersagbar** (irreduzibles Chaos: Crashes, Zufallsdefekte). Gleiche Lehre wie 3.7 / 4.2.8 / 5.1-race.

**C1-Ablation:** Ein explizites, leak-frei cross-gefittetes `pred_dnf`-Feature ins Podium-Modell bringt **keinen signifikanten Effekt** (Podium-Ensemble base 0.0638 vs aug 0.0649, diff +0.0010, CI [−0.0009, +0.0029]). `team_form_dnf_rate_l10` ist bereits Feature und das Podium-Target schließt DNFs ohnehin aus.

**C2 (DNF als ROI-Wettmarkt) bewusst NICHT gebaut** (User-Entscheidung 2026-06-01: „kann eh nicht auf DNFs wetten"). Default deployt = **TeamReliability** (ehrlichste Wahl: beste Kalibrierung, variiert pro Team, anders als der flache ConstantRate). Die App-Page zeigt das Negativ-Verdikt prominent.

---

## 17. Phase 5.3 — Track-Overtaking-Index (Streckenbewertung)

**Stand: ✅ DONE 2026-06-01.** User-Frage: „bei welchen Strecken finden wie viele Überholungen statt?" — bisher gab es nur statische Track-Attribute (Länge, Kurven, DRS-Zonen, street/permanent), keine gemessene Überhol-Statistik. Neu aus den bereits ingesteten `laps.parquet` rekonstruiert.

**Messung — `src/process/overtakes.py` (neuer L2-Prozessor):** Paarweise On-Track-Pass-Erkennung aus der Per-Runde-`Position`-Spalte. Ein echter Pass = zwei Autos tauschen zwischen zwei aufeinanderfolgenden **grünen** Runden die Position; ausgeschlossen: Runde 1 (Startgewühl), Boxenrunden (PitIn/Out dieser oder Vorrunde, je Auto), Nicht-Grün (SC/VSC/Rot via TrackStatus). Ordered-Pairs statt summierter Positions-Deltas ist der Kern: Boxenstopp-Zyklen und Backmarker-Lapping zählen strukturell **nicht** mit (der Delta-Proxy überschätzt ~3–5×). Output `data/processed/overtakes.parquet`, eine Zeile pro Rennen; unbrauchbare Rennen → `is_usable=False`.

**Validierung:** Ø **33,1** / Median 32 / Range 0–81 Überholungen pro Rennen (178 Rennen, 176 nutzbar) — exakt im offiziellen DHL-Bereich (~30–50). Streckenrangfolge deckt sich punktgenau mit Domänenwissen: Monaco ~7 (unten), Albert Park/Hungaroring/Singapur niedrig, Vegas/Interlagos/Red Bull Ring/Bahrain ~50 (oben). Spa min=0 ist kein Bug (Regen-/SC-Rennen 2021).

**Feature — `track_overtakes_prior_mean` in `build.py`:** Expanding-Mean der Überholungen aus den **strikt vorherigen** Rennen je Strecke, `.shift(1)` = leakage-sicher (gleiches Muster wie `driver_track_finish_l3`). Erstauftritt einer Strecke → NaN → Median-Imputation. Neuer Guard in `tests/test_no_leakage.py` rechnet jeden Wert unabhängig aus `overtakes.parquet` nach. `just build-l2` + beide next-race-Recipes um `overtakes build` erweitert.

| Stufe | Inhalt | Artefakt |
|---|---|---|
| A | Messung + Validierung | `src/process/overtakes.py`, `overtakes.parquet` |
| B | Feature + Guard | `build.py` (`track_overtakes_prior_mean`), `tests/test_no_leakage.py`, justfile |
| C | Ablation (3 Targets) | `ablation.py` `- overtaking`-Variante + Race-Rank-A/B |

**Verdikt (Ablation, Holdout 41 Rennen):** Wirkt **genau dort, wo die Theorie es vorhersagt**. Pace-dominierte Brier-Targets profitieren nicht — Podium d_ens −0.0002, Teammate −0.0007 (beide Rauschen-Niveau, die anderen Track-Features proxen die Strecke bereits). Das **Race-Rank-Modell** dagegen richtungskonsistent besser: RegEnsemble Position-MAE 3.415→**3.364** (+0.051), XGBReg +0.041, kein Baum-Modell schlechter — aber paired-bootstrap-CI [−0.014, +0.120] **knapp nicht signifikant** auf 41 Rennen. **Entscheidung (User): global behalten (Option 1)** statt per-Modell-Selektion — Letzteres würde auf Rausch-Deltas optimieren (Overfitting-Falle, vgl. 3.7). Bei größerem Holdout neu bewerten.

---

## 18. Phase 5.4 — Team-Execution-Residual (Strategie-/Umsetzungs-Güte)

**Stand: ✅ DONE 2026-06-01.** User-Idee: „Teams mit historisch besseren Strategien einen besseren Score geben — gucken wie Pace und Ergebnis waren, ob man schlechter war als man sein sollte." Lücke: die bestehenden Team-Features sagen *wo* ein Team landet (`team_form_finish_l5`), aber keines, ob das **besser/schlechter als die Pace hergab** war.

**Feature — `team_exec_residual_l5` / `_l10` in `build.py` (`_add_team_execution_residual`):** Pro Vor-Rennen Finisher nach echter Renn-Pace (`race_clean_median_lap_ms`, grüne Runden ohne Pit/SC — bereits in `sessions.parquet`, sonst Leakage) ranken, `residual = pace_rank − finish_position`. Positiv = besser umgesetzt als die Pace verdiente (Pit-Wall/Box-Crew/Start). Pro Constructor beide Autos kollabiert, gelaggter Rolling-Mean L5/L10, `.shift(1)` = leakage-sicher (wie `_add_team_form`). DNFs ausgeschlossen (= Zuverlässigkeit, steckt schon in `team_form_dnf_rate_l10`). Kein neues Artefakt — inline aus Sessions.

**Floor/Ceiling-Falle (zentrale Lehre):** Roh war das Maß **schädlich** (Race-Rank Ensemble −0.060, Ranker −0.111) und die Team-Rangliste unsinnig (Spitzenteams Red Bull/McLaren *negativ*, Hinterbänkler Sauber/Williams *positiv*). Grund: Ein Pace-Führer kann nicht besser als P1 finishen → Residual nach oben gedeckelt; ein Hinterbänkler kann nur Plätze gewinnen → nach unten gedeckelt. Das rohe Residual ist damit ein **invertierter Proxy der Pace-Klasse**, die das Modell schon kennt. **Fix:** pro Pace-Rang den Erwartungswert abziehen (`_resid_raw − mean(_resid_raw | pace_rank)`) → übrig bleibt die team-spezifische Abweichung. Danach plausible Rangfolge (Ferrari/Mercedes oben, Haas/AlphaTauri unten).

**Verdikt (Ablation + Rank-A/B, detrended):** Marginal, nicht signifikant, aber für die **produktiv genutzten Rank-Modelle leicht positiv**: Ridge (Race-Default) Position-MAE +0.010, LambdaMART-Ranker +0.024/+0.031; nur die Tree-Regressionen verlieren (−0.04, nicht Default). Brier-Targets neutral (Podium −0.0002, Teammate −0.0010). **Entscheidung (User): global behalten (Option 1)**, analog Overtaking. Neuer Guard in `tests/test_no_leakage.py` rechnet jeden Wert (inkl. Detrend) unabhängig nach; `- team execution`-Variante in `ablation.py`.

**Lehre:** Gute Domänen-Intuition ≠ gutes Maß. Ein Residual gegen eine **begrenzte** Zielgröße (Position 1..N) ist an den Rändern strukturell verzerrt — vor der Verwendung de-biasen.

**Leakage-Fix (2026-06-01, Phase 5.5):** Der Detrend-Mittelwert pro Pace-Rang war initial global (sah das Zielrennen) → Round-Trip-Test `test_no_leakage_next.py` + `test_no_leakage_prequali.py` fingen es. Fix: expanding().shift(1) pro Pace-Rang. Zusätzlicher Tiebreaker `["race_date","driver_id"]` in `sort_values` für deterministische `rank(method="first")` über Rennen.

---

## 19. Phase 5.5 — Fahrer/Team-Skill-Features (Teammate-Gap, Pit-Crew, Start)

**Stand: ✅ DONE 2026-06-01.** User-Auftrag: neue Feature-Ideen gebaut aus dem Backlog-Plan — 4 gebaut, 1 als Falle entlarvt.

**Features (alle in `build.py`, Sessions um 2 Spalten erweitert, kein neues L2-Artefakt):**

| Feature | Beschreibung | Detrend | Coverage |
|---|---|---|---|
| `driver_teammate_quali_gap_l5` | ms-Differenz zu Teamkollege in Q, Rolling L5, `.shift(1)` | nein | 98.8% |
| `team_pit_speed_resid_l5` | Pit-Lane-Median vs. Feld-Median **dieses Rennens** (within-race), Cons. kollabiert, Rolling L5 | within-race | 99% |
| `driver_start_pos_gain_l5` | Grid − Lap-1-Position, Rolling L5 | nein | 99% |

**Sessions.parquet erweitert** um `race_pit_lane_median_ms` + `race_lap1_position` (PitIn/Out-Delta bzw. LapNumber==1 aus `laps.parquet`). `race_tyre_deg_ms_per_lap` gebaut, dann entfernt (s.u.).

**Verdikt (Podium + Teammate Ablation + Rank-A/B):**
- `teammate_quali`: **Bester Gewinn** — Podium-Brier +0.0003, Rank-Ridge +0.007, Rank-Ensemble +0.041. Face-Validity: Verstappen −1.07s / Pérez +1.16s.
- `pit_crew` + `start`: Neutral auf Brier, leicht negativ auf Rank-Ensemble (−0.04/−0.05) — **trotzdem behalten** (User-Entscheidung: Omitted-Variable-Kontrolle, neutrales Feature reduziert Falschsignal).
- Alle 3 zusammen: Rank-Ensemble neutral (−0.019 auf Ensemble, aber Ridge +0.007).

**Tyre-Degradation-Falle (⚠️ wichtige Warnung):** `race_tyre_deg_ms_per_lap` (Stint-Slope) gebaut, spektakuläre Verbesserung (Rank-MAE 3.43→2.83). Permutation-Test + NaN-Analyse entlarvt als Fingerabdruck-Leak: NaN-Zeilen (7%) haben mittleren Zielrang 18 vs. 10 bei vorhandenen Werten → Bäume lernen Auto-/Fahrer-Identität auswendig. Null-Effekt auf Podium-Brier (Kontrast!) = eindeutiges Zeichen. **Entfernt.** Generelle Regel: spektakulärer Einzel-Gewinn + keine Wirkung auf andere Targets + nur Bäume → sofort Permutation + NaN-Korrelation prüfen.

**Tests:** 3 neue No-Leakage-Guards (pit/start je unabhängig nachgerechnet; team_exec-Test an neuen Detrend angepasst). Volle Suite 134 Tests grün.

---

## 20. Phase 5.6 — Constructor-Saison-Trajektorie (Backlog #10, ⚠️ DROPPED)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-01 (Negativ-Resultat).** Backlog-Idee #10: ist ein
Team über die laufende Saison auf- oder absteigend? Feature `team_season_pos_traj_l3` =
`team_season_pos_pre_race[vor 3 Rennen] − [dieses Rennen]` je (Jahr, Constructor), positiv =
WCC-Plätze gewonnen. Billigste denkbare Variante: rein aus dem bereits gelaggten Standings-Feld
abgeleitet (kein neues Artefakt), leakage-sicher per Konstruktion (beide Terme sind Pre-Race-Stände).
20.5% NaN (erste 3 Saisonrunden). No-Leakage-Guard (unabhängiger Recompute + Team-Level) + Round-Trip
grün.

**Verdikt (Holdout, 41 Rennen):** Hilft **keinem deployten Modell**. Race-Rank: deployt Ridge
3.221→3.231 (schlechter), nur nicht-deployte XGBReg/RegEnsemble besser (3.449→3.345, 3.434→3.364).
Podium-Ensemble (primärer Wettmarkt) `d_ens −0.0010` (leicht schädlich, LogReg −0.0013); Teammate
neutral (+0.0001); Quali-Rank pre_weekend Ridge minimal besser (+0.005), post_fp2 gemischt. Dieselbe
marginal-nicht-signifikante Signatur wie Overtaking (5.3) / Team-Execution (5.4) — **aber schwächer**:
jene halfen ihren deployten Rank-Modellen, diese nicht.

**Entscheidung (Claude, vom User delegiert „was uns am meisten bringt"): DROP.** Der
Omitted-Variable-Keep aus 5.5 galt für *neutrale* Features (pit_crew/start neutral auf Brier);
die Trajektorie ist netto leicht negativ auf dem deployten/primären Pfad. Die einzigen Gewinne liegen
auf nicht-deployten Modellen — die zu verfolgen ist die Overfitting-Falle (Lehre 3.7). Code vollständig
zurückgebaut (Working Tree byte-identisch zu HEAD); nur dieses Negativ-Resultat dokumentiert, damit die
Idee nicht erneut gebaut wird. Bei deutlich größerem Holdout neu erwägbar.

**Nächster Backlog-Einstieg:** #6 Nässe-Skill-Delta (klar abgrenzbares Signal, Daten liegen).

---

## 21. Phase 5.7 — Nässe-Skill-Delta (Backlog #6, ✅ KEPT)

**Stand: ✅ DONE 2026-06-02 (behalten).** Backlog-Idee #6: manche Fahrer sind Regenspezialisten,
manche nicht — die Level-Form-Features mitteln Nass+Trocken zusammen und können das nicht trennen.
Feature `driver_wet_skill_delta` = (gelaggter Per-Fahrer-Expanding-Mean der Finish-Position-Proxy in
**trockenen** Rennen) − (in **nassen** Rennen), Split via `weather_is_wet_race_hour`. Positiv = finisht
im Nassen besser = Regenspezialist. Standing-Skill-Rating für jedes Rennen; das Modell kombiniert es
selbst mit dem Nass-Forecast des Zielrennens. Läuft nach `_add_weather_features`; beide Means
.shift(1)-gelaggt je Fahrer (no-lookahead). 10.9% NaN (braucht ≥1 vorheriges Nass- UND Trockenrennen).
Kein neues Artefakt. No-Leakage-Guard (unabhängiger Recompute + First-Race-NaN) + Round-Trip grün.

**Verdikt (Holdout, 41 Rennen) — stärkster Race-Rank-Gewinn aller Backlog-Features bisher:** Race-Rank
deployt Ridge 3.221→3.212 (−0.009), RegEnsemble 3.434→**3.320** (−0.114), XGBReg −0.061, LGBMReg −0.077;
nur die Ranker verlieren. Brier: Podium `d_ens −0.0014`, Teammate `+0.0005` (Markt leicht positiv).
Quali-Rank pre_weekend flach (3.424→3.422), post_fp2 leicht schlechter.

**Entscheidung (Claude, vom User delegiert): KEEP (global, Option 1)** — analog Overtaking (5.3) /
Team-Execution (5.4), die wegen Race-Rank-Gewinnen behalten wurden; dieses Feature hilft der deployten
Ridge + Regressions-Ensemble **deutlich klarer** als jene. Das Podium-Minus ist kein spezifischer
Schaden: bei diesem gesättigten Target zeigen fast ALLE Features negatives d_ens (Wetter −0.0017,
WCC −0.0013) — Grid+Pace dominieren, Zusatzfeatures fügen nur Varianz hinzu; −0.0014 ist Standard-
Rauschen. **Caveats:** (1) Small-N-Rauschen in der Fahrer-Rangliste — Verstappen +1.88 / Russell +2.88
plausibel, aber Backmarker/Rookies (Hartley +4.8, Bortoleto −5.7) sind 1–2-Rennen-Artefakte; (2) das aus
Renn-Finishes gebaute Feature ist in den Quali-Rank-Sets konzeptionell fehlplatziert (leichtes
post_fp2-Minus) — späteres Refinement (z.B. eigener Quali-Nässe-Split oder Quali-Set-Ausschluss).

**Nächster Backlog-Einstieg:** #11 Safety-Car-Exposition (echter Strategie-Wert, `laps.parquet` reicht).

---

## 22. Phase 5.8 — Safety-Car-Strategie (Backlog #11, ⚠️ DROPPED, Floor/Ceiling-Confound)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #11: profitiert ein Team
systematisch von SC/VSC-Phasen? Neuer L2-Roh-Signalbau in `process/fastf1._race_sc_position_delta`:
pro Fahrer das Netto-Klassifizierungs-Positionsdelta quer durch jede Neutralisationsphase (TrackStatus
4=SC, 6/7=VSC; Rot=5 ausgeschlossen), gemessen letzte grüne Runde *vor* vs. erste grüne Runde *nach* der
Phase, summiert über die Phasen des Rennens → `race_sc_pos_delta` in `sessions.parquet` (49.7% definiert,
~zero-sum über das Feld). Feature `team_sc_strategy_l10` = team-kollabierter, .shift(1)-gelaggter
10-Rennen-Rolling-Mean. SC-Phasen-Erkennung + No-Leakage-Guard (unabhängiger Recompute) gebaut.

**Verdikt — Floor/Ceiling-Confound (zentral, 2. Bestätigung von 5.4):** Face-Validity **invertiert** —
Sauber +2.6 (oben), Mercedes −1.36 (unten). Ein Führender kann quer durch eine SC kaum Plätze *gewinnen*
(nur verlieren), ein Hinterbänkler kaum *verlieren*. Das rohe SC-Delta ist damit primär ein invertierter
Proxy der Pace-Klasse, die das Modell via echte Pace-Features schon hat — genau die Floor/Ceiling-Falle
aus Phase 5.4. **Ablation (roh):** Brier-Märkte leicht *positiv* (Podium d_ens +0.0011, Teammate +0.0009),
aber im Rausch-Band (Baseline driftet ~0.001 zwischen Läufen) und aus dem konfundierten Inverted-Pace-
Proxy. Race-Rank: deployt Ridge flach (3.212→3.214), Regressions-Familie *leicht schlechter*
(RegEnsemble 3.320→3.352) — also genau die Modelle, die #6 verbesserte, verlieren hier. Quali-Rank leicht
schlechter. Anders als 5.4-roh (klar schädlich, −0.06…−0.11) nur Rausch-Band-gemischt, weil das SC-Signal
selten/schwach ist (selbst sein Confound ist schwach).

**Entscheidung (Claude-Empfehlung, vom User bestätigt): DROP.** Konfundiert + Rausch-Band, hilft keinem
deployten Modell klar. Der saubere Test der Hypothese bräuchte einen 5.4-Stil-De-Bias (per-Periode
Entry-Position erfassen, Erwartungswert je Position abziehen) — aber der De-Bias-Payoff wäre laut 5.4 nur
marginal und das Roh-Signal ist bereits auf den deployten Modellen Rausch-Band, also nicht den Umbau wert.
Code (inkl. L2-SC-Prozessor) vollständig zurückgebaut, Working Tree byte-identisch zu HEAD; nur dieses
Negativ-Resultat + die De-Bias-Option dokumentiert.

**Lehre (Wiederholung 5.4):** Jedes Maß als Positionsänderung gegen eine begrenzte Zielgröße (Position
1..N) ist an den Rändern strukturell verzerrt → vor Verwendung de-biasen ODER verwerfen, wenn das
Roh-Signal ohnehin im Rauschen liegt.

**Nächster Backlog-Einstieg:** #13 Renn-Restart-Performance (ergänzt SC logisch — aber ACHTUNG: gleiche
Floor/Ceiling-Falle, dort von vornherein de-biasen) oder #9 Form-Momentum (billig, aus bestehenden Spalten).

---

## 23. Phase 5.9 — Form-Momentum (Backlog #9, ✅ KEPT)

**Stand: ✅ DONE 2026-06-02 (behalten).** Backlog-Idee #9: Trend statt Niveau — ist ein Fahrer im Auf-
oder Abwind? Feature `driver_form_momentum_l3_l10` = `driver_form_finish_l10 − l3` (gelaggter 3- vs.
10-Rennen-Mean der Finish-Position-Proxy je Fahrer). Positiv = jüngste Form besser als die längere
Baseline = aufsteigend. Beide Terme bereits `.shift(1)`-gelaggt → leakage-sicher; berechnet inline in
`_add_driver_form`, kein neues Artefakt. Distinkt von #10 (das war Team-WCC-Trend und scheiterte; dies ist
Fahrer-Form-Streak). No-Leakage-Guard (unabhängiger l10−l3-Recompute + First-Race-NaN) + Round-Trip grün.

**Verdikt (Holdout, 41 Rennen):** Race-Rank deployt Ridge 3.212→**3.202** (−0.010, gleich stark wie das
behaltene #6); XGBRanker −0.077; Tree-Regressionen gemischt (LGBMReg +0.053). Brier: Podium d_ens +0.0015
(geholfen), Teammate −0.0004 (flach). Quali pre_weekend Ridge flach, post_fp2 leicht besser. Das
Podium-Plus ist Rausch-Band (in diesem Lauf hilft fast jedes Feature leicht, Baseline 0.0650 niedrig) —
die Keep-Basis ist die **deployte-race-Ridge-Verbesserung + saubere neutrale Märkte**, exakt das Kriterium
von #6/Overtaking/Team-Exec.

**Entscheidung (Claude-Empfehlung, vom User delegiert): KEEP (global).** Klarste positive Bilanz nach #6,
**sauberes Signal ohne Floor/Ceiling** (ein Richtungs-Maß ist nicht bounded — Mittelfeldfahrer haben beide
Vorzeichen; Face-Validity plausibel: Antonelli/Hamilton im Aufwind, Verstappen 2026 im Abwind). Bestätigt
die Session-Lehre: Features, die etwas isolieren, das Pace/Niveau NICHT proxen (Trend, Wet-Skill,
Teammate-Gap), sind die Gewinner; bounded Positionsresiduen (Team-Exec, SC) die Verlierer.

**Nächster Backlog-Einstieg:** #16 H2H vs. direkte Konkurrenten (aus bestehenden Spalten) oder #13
Restart-Performance (mit Pflicht-De-Bias).

---

## 24. Phase 5.10 — H2H vs. Konkurrenten (Backlog #16, ⚠️ DROPPED, Pace-Redundanz)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #16: Racecraft isolieren
über H2H gegen Fahrer derselben Pace-Klasse. Feature `driver_h2h_beat_rate_l20` = pro Rennen Anteil der
Grid-Nachbarn (±3 Startplätze, klassifizierte Finisher), die der Fahrer im Ziel schlägt; gelaggter
Rolling-Mean l20 je Fahrer. Grid-Nachbar als „Pace-Klasse" = pre-race bekannt → nicht zirkulär; Beat-
Outcome = Renn-Result → speist nur das gelaggte Feature. No-Leakage-Guard + Round-Trip grün.

**Verdikt (Holdout, 41 Rennen) — eindeutig negativ + redundant:** **corr(beat_rate, driver_form_finish_l10)
= −0.71** → das Feature ist zu 71% ein Pace-Proxy (schnelle Fahrer schlagen ihre tieferen Grid-Nachbarn
leicht), face-validity gemischt (echtes Racecraft-Restsignal bei Albon/Hülkenberg in langsamen Autos, aber
pace-dominiert: Norris/Verstappen/Piastri oben). Race-Rank: deployt Ridge 3.202→3.212 (+0.010 schlechter)
UND die **gesamte Regressions-Familie deutlich schlechter** (RegEnsemble 3.328→3.439 +0.111, LGBMReg +0.089,
XGBReg +0.077) — der schlechteste Race-Rank-Effekt der Session. Brier: Podium −0.0009 (schadet), Teammate
+0.0015 („alles-hilft"-Lauf, mid-pack, Rauschen).

**Entscheidung (Claude-Empfehlung, vom User delegiert): DROP.** 71% redundant mit der schon vorhandenen
Form; schleppt kollineares Rauschen in die Regressionsmodelle und verschlechtert den deployten race-Ridge
+ die ganze Regressions-Familie. Das schwache Racecraft-Residuum (Albon-Typ) überwiegt die Pace-Redundanz-
Last nicht. Code byte-identisch zurückgebaut; Negativ-Resultat dokumentiert.

**Lehre (3. Drop-Muster):** Ein Feature, das stark (|corr|>0.7) mit einem bereits vorhandenen Feature
korreliert, ist meist redundant + schädlich für lineare/Regressionsmodelle — vor dem Bau die Korrelation
zur bestehenden Form/Pace prüfen, nicht nur Leakage. Reine, gering-korrelierte Skill-Isolatoren (#6, #9,
Teammate-Gap) bleiben die Gewinner. (Nuance siehe §25: hohe Korrelation ist ein Risiko-Flag, kein
Auto-Drop — wenn das Signal trotzdem dichter/sauberer ist als das Original, kann es benigne helfen.)

---

## 25. Phase 5.11 — Track-Cluster-Form (Backlog #8, ✅ KEPT)

**Stand: ✅ DONE 2026-06-02 (behalten, knappe Keep-Entscheidung).** Backlog-Idee #8: `driver_track_finish_l3`
(exakte Strecke) ist ~31% NaN; generalisiere auf den Strecken-CLUSTER. Feature `driver_cluster_finish_l5` =
gelaggter 5-Rennen-Rolling-Mean der Finish-Position-Proxy über die Vor-Rennen des Fahrers an Strecken
desselben Clusters. Cluster (`_track_cluster`) = `{street|perm}_{twisty|fast}`, split bei 3 Kurven/km — eine
statische, interpretierbare Track-Eigenschaft (kein Leakage). Gelaggt per (Fahrer, Cluster). No-Leakage-Guard
(unabhängiger Recompute inkl. Cluster) + Round-Trip grün.

**Coverage-Gewinn (Kernziel erreicht):** NaN 30.9% (exakt-Track) → **4.8%** (Cluster).

**Verdikt (Holdout, 41 Rennen) — mixed-positiv:** Race-Rank deployt Ridge 3.202→**3.190** (−0.012,
Session-Bestwert); Quali post_fp2 LGBMReg 3.283→**3.217** (−0.066, Session-Bestwert); aber Quali
pre_weekend Ridge 3.422→3.441 (+0.019 schlechter), Tree-Regressionen leicht schlechter (RegEnsemble +0.034).
Brier: Podium −0.0005, Teammate +0.0008 (Rausch). **corr(cluster, driver_form_finish_l5) = 0.798** — über
der 0.6-Redundanzschwelle.

**Entscheidung (Claude-Empfehlung, vom User delegiert): KEEP.** Trotz 0.798-Korrelation **kein** #16-artiger
Regressions-Kollaps — die Redundanz ist hier *benigne*: das deployte race-Ridge + post_fp2-Quali bekommen
ihre Session-Bestwerte, plus der reale Coverage-Fix (31%→5% NaN). Knapper als #6/#9 (pre_weekend-Wobble +
Podium-Rausch-negativ), aber die deployten Hauptsignale + Coverage überwiegen.

**Nuance zur Korrelations-Checkliste:** |corr|>0.7 ist ein Risiko-Flag, KEIN Auto-Drop. #16 (Beat-Rate,
corr −0.71) war redundant UND schädlich (Signal war nur ein verrauschter Pace-Proxy). #8 (corr 0.798) ist
redundant ABER hilfreich, weil es ein dichteres, sauberes Track-Typ-Signal trägt, das die Gesamt-Form
nicht hat. → Korrelation prüfen, aber die Ablation entscheiden lassen.

**Nächster Backlog-Einstieg:** #7 Driver-ELO (sauberste Fahrer-Isolation, aufwändig) oder #13 Restart MIT
Pflicht-De-Bias.

---

## 26. Phase 5.12 — Driver-ELO (Backlog #7, ⚠️ DROPPED, redundant mit Form)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #7: pairwise Multiplayer-ELO,
rennweise chronologisch über die ganze Historie aktualisiert, soll Fahrer-Skill opponent-strength-gewichtet
isolieren. Feature `driver_elo_pre_race` = Rating ENTERING das Rennen (vor dem Update erfasst → leakage-
sicher), Debütanten bei 1500. Pairwise-Update (i vs j: 1/0.5/0 nach Zielreihenfolge, K=24/n_opp). No-
Leakage-Guard (unabhängiger chronologischer Replay + Pre-Update-Property + Debüt=1500) + Round-Trip grün
(globales sequenzielles Feature ist round-trip-sicher). Range 1343–1885, Verstappen oben — face-valide.

**Verdikt (Holdout, 41 Rennen) — eindeutig negativ + maximal redundant:** **corr(elo, driver_form_finish_l10)
= −0.846** (höchste aller getesteten Features). Race-Rank deployt Ridge 3.190→**3.226** (+0.036 schlechter);
Quali post_fp2 verliert den #8-Gewinn (LGBMReg 3.217→3.295, +0.073); pre_weekend leicht besser. Brier:
Podium −0.0002 (neutral), Teammate −0.0009 (Rausch). Ein Finishing-ELO ist im Kern eine geglättete
Finishing-Form — es trägt KEINE neue Dimension, nur Redundanz, und die Opponent-Strength-Gewichtung reicht
nicht. **Das Backlog-Versprechen „ELO isoliert Fahrer vom Auto" ist falsch:** ein Finishing-ELO ist
auto-dominiert (schnelles Auto → schlägt alle → hohes ELO).

**Entscheidung (Claude-Empfehlung, vom User delegiert): DROP.** Höchste Redundanz aller Kandidaten,
verschlechtert die deployten race+post_fp2-Modelle. Code (inkl. ELO-Maschinerie) byte-identisch zurückgebaut.

**Checklisten-Verfeinerung (#8 vs. #7, zentral):** Hohe Korrelation (|corr|>0.7) ist OK, WENN das Feature
eine NEUE Dimension trägt — #8 Track-Cluster (corr 0.798, aber Track-Typ-Info → benigne hilfreich, KEPT).
Ein hoch-korreliertes Feature in DERSELBEN Dimension wie bestehende (#7 ELO ≈ Form, #16 Beat-Rate ≈ Form)
ist reine Redundanz → schädlich, DROP. Vor dem Bau fragen: korreliert hoch UND misst dasselbe? → nicht bauen.

**Nächster Backlog-Einstieg:** #13 Restart MIT Pflicht-De-Bias oder #5 Top-Speed-Profil (mit
Fingerprint-Check) — beide tragen potenziell eine neue Dimension (Strategie-/Auto-Charakter).

---

## 27. Phase 5.13 — Top-Speed-Profil (Backlog #5, ⚠️ DROPPED, neue Dimension aber kein deployter Gewinn)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #5: Auto-Charakter
(Motor-Power vs. Drag-Trim) aus der Geraden-Geschwindigkeit. Neuer L2-Wert `race_top_speed_kph` in
`fastf1.py._race_pace_rows` = p95 von `SpeedST` (Speed-Trap auf längster Gerade) je Fahrer im Rennen
(p95 statt max = robust gegen einzelne Windschatten-Runden). Feature `team_top_speed_resid_l5` in
`build.py._add_top_speed_profile` = within-race-relativ (minus Renn-Median, entfernt die Strecke — Monza
≫ Monaco), pro Constructor kollabiert (Motor+Aero geteilt), .shift(1)-gelaggter 5-Rennen-Rolling-Mean
(Top-Speed ist Renn-Result → wie `team_pit_speed_resid_l5`). No-Leakage-Guard (unabhängiger Recompute) +
Round-Trip grün.

**Bestes Profil auf dem Papier aller getesteten Features:** **corr ~0** mit Form/Pace (|corr| < 0.04 zu
finish_l10/l5, team_form, q_gap, grid) — die sauberste Entkopplung überhaupt, echte neue Dimension, nicht
bounded (kein Floor/Ceiling), NaN nur 1.0% (Constructor-Cold-Start, überwiegend 2018 + neue Teams; mildes
Target-Delta race_rank 11.8 vs 10.5 strukturell durch Cold-Start, KEIN Auto-Fingerprint wie Tyre-Deg).
Face-Validity ein **Charakter-Achse, kein Qualitäts-Proxy:** Williams/Haas (Low-Downforce, schnell auf
Geraden) oben, McLaren/Red Bull (High-Downforce-Kurvenautos) unten — exakt wie 2024/25 real.

**Verdikt (Holdout, 41 Rennen) — kein deployter Gewinn, post_fp2-Schaden:**
- **race Ridge (Flaggschiff): 3.190 → 3.190 exakt neutral.** Ridge ist linear; Top-Speed-Charakter ist
  im Mittel weder gut noch schlecht (Haupteffekt ~0), das Signal steckt nur in der **Track-Interaktion**
  (Low-Drag-Auto relativ gut auf Power-Strecken, schlecht in Monaco) — die ein lineares Modell nicht
  abbilden kann. Nicht-deployte race-Bäume gemischt: LGBMReg 3.412→3.381 (−0.031), RegEnsemble −0.013,
  XGBReg 3.410→3.432 (+0.022 schlechter).
- **quali post_fp2 LGBMReg (deployt): 3.217 → 3.332 (+0.115 deutlich schlechter)** — race-abgeleitetes
  Feature im Quali-Set fehlplatziert; reproduzierbar (Rank-Eval deterministisch), kein Run-Rauschen.
- quali pre_weekend Ridge 3.441→3.429 (−0.012, marginal).
- Brier: Podium d_ens +0.0005, Teammate −0.0009 — beides Rausch-Band (in diesem Lauf fast alle Features so).

**Entscheidung (Claude-Empfehlung, vom User delegiert): DROP.** Hilft KEINEM deployten Modell (race-Ridge
exakt neutral), schadet dem deployten post_fp2-Quali, Gewinne nur auf nicht-deployten race-Bäumen — exakt
die Overfitting-Falle (Lehre 3.7, Drop-Grund bei #10). Code (inkl. L2-`race_top_speed_kph`) byte-identisch
zurückgebaut, mvp/sessions.parquet neu gebaut = HEAD.

**Lehre (zentral, neu):** Saubere Entkopplung (neue Dimension, corr~0) ist **notwendig, aber nicht
hinreichend.** Wenn der prädiktive Gehalt eines Features eine **Interaktion** braucht, die das deployte
(lineare) Modell nicht darstellen kann UND die Bäume bei N≈2700 nicht zuverlässig extrahieren, bleibt es
neutral-bis-schädlich — egal wie sauber die Dimension ist. Die Gewinner (#6 Wet-Skill, #9 Momentum, #8
Cluster) trugen alle einen nutzbaren **Haupteffekt** (besser/schlechter finishen), keine reine Interaktion.
**Mögliches künftiges Refinement:** explizites Interaktions-Feature (`top_speed_resid × track_straightness`
bzw. `× n_drs_zones`), das den Charakter für die linearen Modelle als Haupteffekt verfügbar macht — aber
durch dieselbe Small-N-Noise begrenzt.

**Nächster Backlog-Einstieg:** #13 Restart MIT Pflicht-De-Bias oder #14 FP2-Longrun-Teammate-Gap
(Race-Pace-Analog zum behaltenen Quali-Teammate-Gap 5.5, trägt sauberen Haupteffekt — besseres Profil als
ein reines Interaktions-Feature).

---

## 28. Phase 5.14 — FP2-Longrun-Teammate-Gap (Backlog #14, ⚠️ DROPPED, hilft nur pre_weekend)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #14: Race-Pace-Analog zum
behaltenen Quali-Teammate-Gap (5.5) — Fahrer-Sonntags-Pace vom Auto isoliert. Feature
`driver_teammate_fp2_pace_gap_l5` in `build.py._add_fp2_teammate_pace_gap` = pro Rennen die ms-Differenz der
FP2-Longrun-Pace (`fp2_long_run_gap_ms`) zum Teamkollegen (nur 2-Auto-Constructor, beide mit gültigem
Longrun ~66% der Zeilen), .shift(1)-gelaggter 5-Rennen-Rolling-Mean je Fahrer, negativ = schneller als
Garagenseite. Kein neues L2-Artefakt. No-Leakage-Guard (unabhängiger Recompute + Antisymmetrie) + Round-Trip
grün.

**Artefakt-Tail-Falle (wichtige Lehre):** Der Roh-Gap hatte std 3.3s und einen ~6%-Tail bis ±40s/Runde —
**Compound-Confound:** der Longrun-Picker (`_fp2_rows`) wählt den längsten Stint OHNE Compound-Matching, also
vergleicht er teils Hard- vs. Soft-Longruns = Reifenwahl, nicht Fahrer-Skill. Anders als der Quali-Gap
(Single-Lap, gleicher Soft, ~3.6s-bound) ist der Longrun-Gap strukturell kontaminiert. Fix: |Gap| > 3000ms
auf NaN (nicht-vergleichbare Longruns), behält 94% der validen Vergleiche; danach physikalisch plausible
Range ±2.84s. Face-Validity konsistent (Teamkollegen spiegeln: Bottas −2.1s / Zhou +2.5s). corr ~0 mit
Form/Pace (saubere neue Dimension), NaN 2.1%.

**Verdikt (Holdout, 41 Rennen) — neutral auf Brier, schadet deployten race+post_fp2, hilft nur pre_weekend:**
- **race Ridge (Flaggschiff): 3.190 → 3.231 (+0.041 schlechter)** — schlechtester race-Ridge-Hit der Session.
- **quali post_fp2 (deployt LGBMReg): 3.217 → 3.351 (+0.134); bester post_fp2 jetzt RegEnsemble 3.293 (+0.076).**
- **quali pre_weekend Ridge (deployt): 3.441 → 3.388 (−0.053, jetzt signifikant vs Baseline)** — einziger Gewinn.
- Brier: Podium d_ens +0.0000 (exakt neutral), Teammate +0.0001 (Rausch).

**Mechanismus (lehrreich):** Das Signal ist echt, aber konzentriert auf das **datenarme pre_weekend-Regime**
(nur Historie verfügbar → Fahrer-Skill-Delta zählt). Wo das Modell schon die AKTUELLE Pace dieses Rennens hat
(race: echtes Grid+Quali+FP2; post_fp2: FP2), ist die gelaggte Teammate-Pace redundant + varianzerhöhend und
verschlechtert genau das deployte race-Ridge.

**Entscheidung (Claude-Empfehlung, vom User delegiert): DROP.** Verschlechtert das Flaggschiff-race-Ridge
(+0.041) — exakt der Disqualifikator von #7 (+0.036) / #16 (+0.010); **kein behaltenes Feature (#6/#8/#9) hat
je das race-Ridge verschlechtert.** Neutral auf beiden Brier-Märkten. Der einzelne pre_weekend-Gewinn wiegt
das Schaden an zwei anderen deployten Modellen nicht auf. Code byte-identisch zurückgebaut, mvp.parquet = HEAD.

**Refinement-Idee (nicht gebaut, Per-Set-Surgery bewusst gemieden):** Ein **pre_weekend-quali-only**-Feature
(aus race/post_fp2-Sets ausgeschlossen) würde den −0.053-Gewinn isolieren ohne den race-Schaden — aber per-Set-
Selektion auf Holdout-Deltas ist die Overfitting-Falle (3.7); zudem ist offen, ob die pre_weekend-BINÄR-Märkte
(pole/top3/top10, die echten Wetten) auch profitieren oder nur der Rank. Bei größerem Holdout neu erwägbar.

**Lehre (ergänzt 5.13):** Ein sauberer Haupteffekt (corr~0, neue Dimension) ist auch dann nicht genug, wenn
sein Informationsgehalt von reicheren AKTUELL-Features (current-race Pace) bereits abgedeckt wird — dann fügt
er den informationsreichen deployten Modellen nur Varianz hinzu. Gewinner tragen Information, die kein anderes
Feature trägt UND die im deployten Kontext nicht schon vorhanden ist.

---

## 29. Phase 5.15 — Quali-Programm-Effizienz (Backlog #15, ⚠️ DROPPED, schadet post_fp2-Quali)

**Stand: ✅ EVALUIERT, VERWORFEN 2026-06-02 (Negativ-Resultat).** Backlog-Idee #15: misst, wie gut ein Fahrer
seine bestmögliche Quali-Runde zusammensetzt. Neuer L2-Wert `q_exec_gap_frac` in `fastf1.py._quali_exec_gap`
= `(best_actual_lap − best_possible_sectors_lap) / best_possible` aus den Q-Lap-Sektorzeiten (min S1 + min S2
+ min S3 über die sauberen fliegenden Runden), **≥2 fliegende Runden erforderlich** (sonst theoretical=actual
= Schein-0). Als Bruchteil = cross-track-vergleichbar. Feature `driver_quali_exec_gap_l5` = .shift(1)-gelaggter
5-Rennen-Rolling-Mean je Fahrer (Execution-Skill-Rating, in allen Sets legal). No-Leakage-Guard + Round-Trip grün.

**Sauberes Profil:** corr ~0 mit Form/Pace/Quali (|corr| < 0.05, neue Dimension), NaN 1.8%, Range 0–0.57%,
**gute Face-Validity** — Elite-Qualifier Leclerc/Verstappen/Alonso unter den besten „Lap-Assemblern", Zhou/
Hülkenberg hinten.

**Verdikt (Holdout, 41 Rennen) — schadet dem deployten post_fp2-Quali, sonst neutral:**
- **quali post_fp2 LGBMReg (deployt, sein „Zuhause"): 3.217 → 3.315 (+0.098 schlechter)** — reproduzierbar.
- quali pre_weekend Ridge (deployt): 3.441 → 3.436 (−0.005, marginal).
- race Ridge (deployt): 3.190 → 3.197 (+0.007, marginal schlechter).
- Brier: Podium d_ens −0.0002, Teammate +0.0004 (beides Rausch-Band).

**Entscheidung (Claude-Empfehlung, vom User delegiert): DROP.** Selbst auf den Quali-Markt gezielt schadet es
ausgerechnet dem deployten post_fp2-Modell und hilft keinem deployten Modell. Das Execution-Signal (40–150ms,
0.04–0.15% der Rundenzeit) ist zu klein/verrauscht gegenüber der Pace, die die Modelle schon tragen → reine
Varianz für die sensitive LGBMReg. Code (inkl. L2-`q_exec_gap_frac`) byte-identisch zurückgebaut.

**Sättigungs-Befund (zentral, nach 3 Drops in Folge):** Top-Speed (5.13), FP2-Teammate-Gap (5.14) und Quali-
Exec (5.15) waren alle drei **saubere neue Dimensionen** (corr~0, gute Face-Validity, leakage-sicher) — und
ALLE drei scheiterten auf den deployten Modellen. Das ist kein Zufall mehr: **der Feature-Satz ist gesättigt.**
Grid + Quali + Pace + die behaltenen Skill-Deltas (Wet #6, Cluster #8, Momentum #9) fangen das vorhersagbare
Signal bei N≈2700 / 41-Holdout-Rennen ab; weitere Mikro-Skill-Features fügen den deployten (überwiegend
linearen) Modellen Varianz statt Signal hinzu. Die Gewinner kamen alle FRÜH (vor der Sättigung) und trugen
größere, gröbere Effekte (Nässe-Spezialist, Track-Typ-Form, Auf-/Abwärtstrend). **Empfehlung: Feature-Jagd
beenden, auf Konsolidierung / mehr Daten (Saison 2026 wächst) / andere Hebel (Kalibrierung, Ensemble-Gewichte,
Live-ROI-Loop) umschwenken.** Verbleibende Backlog-Kandidaten (#13 Restart bounded, #12 Undercut komplex) sind
niedriger-EV und mit hoher Wahrscheinlichkeit ebenfalls Drops.

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
