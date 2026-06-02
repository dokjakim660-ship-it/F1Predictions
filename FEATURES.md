# Genutzte Features — F1 Prediction

> Stand: 2026-06-02 (nach Phase 5.16). Quelle: `src/features/build.py` → `FEATURE_COLUMNS`.
> **52 numerische Features + 1 kategorisches (`track_id`) = 53 Modell-Inputs.**

## Grundprinzipien

- **Granularität:** eine Zeile = ein Fahrer × ein Rennen (long format).
- **Pre-Race-Vertrag:** jedes Feature ist Samstagabend (nach Quali, vor dem Rennen) bekannt.
  Renn-Ergebnis-Spalten (Finish, DNF, Punkte, Renn-Pace) speisen nur die Targets und die
  `.shift(1)`-gelaggte rollende Historie — niemals das eigene Rennen als Feature.
- **Lagging:** alle Form-/Rolling-Features nutzen `.shift(1)` innerhalb der Fahrer-/Team-/
  (Fahrer,Strecke)-Gruppe → Rennen N sieht nur Rennen strikt davor (No-Lookahead-Invariante,
  bewacht von `tests/test_no_leakage.py`).
- **Cross-Track-Vergleichbarkeit:** rohe Rundenzeiten sind zwischen Strecken nicht vergleichbar,
  daher nur GAP-Features (z.B. Abstand zur Pole) statt absoluter Zeiten.
- **NaN:** Cold-Start-Lücken (erste Rennen eines Fahrers/Teams) werden vom Modell median-imputiert;
  XGB/LGBM behandeln NaN nativ.

---

## A) Startaufstellung / Grid (post-quali → pre-race legal)

| Feature | Was es misst |
|---|---|
| `grid_effective` | Startposition; Boxengassen-Start (grid=0) → als P21 (schlechtester Slot) behandelt. |
| `grid_log` | `log(grid_effective)` — komprimiert den Abstand vorne (P1→P2 wiegt mehr als P15→P16). |
| `is_pole` | 1 wenn von P1 gestartet. |
| `is_top3_grid` | 1 wenn aus den Top-3 gestartet. |

## B) Qualifying (dieses Rennen)

| Feature | Was es misst |
|---|---|
| `q_position` | Qualifying-Endposition. |
| `q_gap_to_pole_ms` | Zeitabstand der besten Quali-Runde zur Pole (ms) — cross-track-vergleichbar. |
| `quali_beat_teammate` | 1 wenn den Teamkollegen in der Quali geschlagen (nur 2-Auto-Teams, beide mit Zeit). |

## C) FP2-Pace (dieses Rennen; auf Sprint-WE NaN, `has_fp2` flaggt es)

| Feature | Was es misst |
|---|---|
| `fp2_long_run_gap_ms` | Abstand der FP2-Longrun-Median-Pace zum schnellsten Longrun der Session — der „heimliche Star" (Race-Pace-Proxy). |
| `fp2_short_run_gap_ms` | Abstand der besten FP2-Einzelrunde zur schnellsten (Quali-Sim-Proxy). |
| `fp2_long_run_lap_count` | Länge des erfassten Longruns (Konfidenz-Indikator der Pace-Schätzung). |
| `has_fp2` | 1 wenn FP2-Daten vorhanden (0 auf Sprint-Wochenenden). |

## D) Sprint-Ergebnis (dieses Rennen; ~85% NaN, `has_sprint` flaggt es)

| Feature | Was es misst |
|---|---|
| `sprint_position` | Sprint-Endposition. |
| `sprint_gap_to_winner_ms` | Zeitabstand zum Sprint-Sieger. |
| `sprint_minus_quali_pos` | Positionen im Sprint gewonnen/verloren vs. Quali (Renn-Racecraft-Hinweis). |
| `has_sprint` | 1 wenn Sprint-Wochenende. |

## E) Fahrer-Form (gelaggt, rollend)

| Feature | Was es misst |
|---|---|
| `driver_form_finish_l5` | Mittlere Finish-Position der letzten 5 Rennen (DNF = Worst-Case-Proxy P20). |
| `driver_form_finish_l10` | Dasselbe über 10 Rennen (stabileres Niveau). |
| `driver_form_podium_rate_l10` | Podestquote der letzten 10 Rennen. |
| `driver_form_points_l5` | Mittlere Punkte der letzten 5 Rennen. |
| `driver_form_dnf_rate_l10` | Ausfallquote der letzten 10 Rennen (Fahrer-Zuverlässigkeit). |
| `driver_form_quali_pos_l5` | Mittlere Quali-Position der letzten 5 Rennen. |
| `driver_career_races` | Anzahl bisheriger Rennen (Erfahrung; 0 im Debüt). |
| `driver_age_years` | Alter am Renntag. |
| `driver_form_momentum_l3_l10` | **Trend:** `finish_l10 − l3`; positiv = jüngste Form besser als Langzeit-Niveau = im Aufwind. *(Phase 5.9, behalten.)* |

## F) Fahrer-Skill, vom Auto isoliert (gelaggt)

| Feature | Was es misst |
|---|---|
| `driver_teammate_quali_gap_l5` | Gelaggter Quali-Zeitabstand zum eigenen Teamkollegen (gleiches Auto → Delta = Fahrer). Negativ = schneller als die Garagenseite. *(Phase 5.5, behalten.)* |
| `driver_wet_skill_delta` | Gelaggtes (Trocken-Mittel − Nass-Mittel) der Finish-Position; positiv = Regenspezialist. Das Modell kombiniert es mit dem Nass-Forecast. *(Phase 5.7, behalten.)* |
| `driver_start_pos_gain_l5` | Gelaggte Plätze gewonnen Grid→Runde 1 (Launch/Start-Racecraft, distinkt vom Overtaking-Index der Runde 1 ausschließt). *(Phase 5.5, behalten.)* |

## G) Team / Constructor-Form (gelaggt; ein Wert für beide Autos)

| Feature | Was es misst |
|---|---|
| `team_form_points_l5` | Mittlere Team-Punkte (beide Autos) der letzten 5 Rennen. |
| `team_form_finish_l5` | Mittlere Team-Finish-Position der letzten 5 Rennen. |
| `team_form_podium_rate_l10` | Podestquote pro Auto der letzten 10 Rennen. |
| `team_form_dnf_rate_l10` | Team-Ausfallquote der letzten 10 Rennen (Auto-Zuverlässigkeit). |
| `team_form_quali_gap_pole_l5` | Mittlerer Quali-Abstand des Teams zur Pole (Auto-Pace). |
| `team_season_points_pre_race` | WCC-Punkte-Stand vor diesem Rennen (innerhalb der Saison kumuliert, gelaggt). |
| `team_season_pos_pre_race` | WCC-Rang vor diesem Rennen. |
| `team_exec_residual_l5` / `_l10` | **Strategie/Umsetzungs-Güte:** Renn-Pace-Rang minus Finish-Position, floor/ceiling-de-biast, gelaggt. Positiv = besser umgesetzt als die Pace hergab (Pit-Wall, Box-Crew, Start). *(Phase 5.4, behalten.)* |
| `team_pit_speed_resid_l5` | Gelaggte Boxenstopp-Standzeit des Teams vs. Feld-Median dieses Rennens (entfernt die Strecken-Pitlane-Länge); negativ = schnellere Crew. *(Phase 5.5, behalten.)* |

## H) Strecken-Attribute (statisch / gelaggt)

| Feature | Was es misst |
|---|---|
| `track_length_km` | Streckenlänge. |
| `track_n_corners` | Anzahl Kurven. |
| `track_n_drs_zones` | Anzahl DRS-Zonen. |
| `track_is_street` | 1 wenn Stadtkurs. |
| `track_overtakes_prior_mean` | **Überhol-Index:** mittlere On-Track-Überholungen in den früheren Rennen dieser Strecke (gelaggt, no-lookahead). *(Phase 5.3, behalten.)* |
| `track_id` *(kategorisch)* | Strecken-Identität — LGBM nativ kategorisch / XGB one-hot. Lernt strecken-spezifische Muster, die die Attribute nicht erfassen. |

## I) Fahrer × Strecke (gelaggt)

| Feature | Was es misst |
|---|---|
| `driver_track_finish_l3` | Gelaggte Finish-Form des Fahrers auf der **exakten** Strecke (letzte 3 Besuche). ~31% NaN (rotierende Kalender, Debüts). |
| `driver_cluster_finish_l5` | Gelaggte Form auf Strecken desselben **Charakter-Clusters** (`{street\|perm}_{twisty\|fast}`). Dichtere Füllung (NaN ~5%) der exakt-Track-Idee. *(Phase 5.11, behalten.)* |

## J) Wetter (Forecast pre-race, Archiv post-race — leakage-sicher per Quelle)

| Feature | Was es misst |
|---|---|
| `weather_temp_c_race_hour` | Lufttemperatur zur Rennstunde. |
| `weather_is_wet_race_hour` | 1 wenn nass zur Rennstunde. |
| `weather_precip_mm_day_total` | Niederschlagssumme des Renntags. |
| `weather_wind_kph_race_hour` | Windgeschwindigkeit zur Rennstunde. |
| `weather_temp_c_day_max` | Tages-Höchsttemperatur. |

## K) Saison / Ära

| Feature | Was es misst |
|---|---|
| `season_progress` | `round / Saisonrunden` — normierter Saisonfortschritt (vergleichbar über 21- vs. 24-Rennen-Kalender). |
| `era_2022plus` | 1 ab 2022 (Ground-Effect-Reglement). |
| `era_2026plus` | 1 ab 2026 (neues Motoren-/Aero-Reglement). |

---

## Timing-Varianten (Pre-Quali-Modelle)

Aus `FEATURE_COLUMNS` abgeleitete reduzierte Sichten für Modelle, die VOR der Quali vorhersagen:

- **`post_fp2` (41 Features):** alles außer den quali-abgeleiteten + Sprint-Features (FP2 bleibt).
- **`pre_weekend` (37 Features):** zusätzlich ohne FP2 — nichts vom Renn-Wochenende.

Das Pre-Race-Set (volle 52+`track_id`) sieht Grid + Quali.

## Targets (keine Features — was vorhergesagt wird)

- `target_podium` — P(Top 3). **Primärer Wettmarkt.**
- `target_beat_teammate` — P(schlägt Teamkollegen im Rennen).
- `target_dnf` — P(Ausfall). *(Default deployt = TeamReliability-Baseline, Phase 5.2.)*
- `target_pole` / `target_top3_quali` / `target_top10_quali` / `target_quali_beat_teammate` — Quali-Märkte (Phase 4.2).
- `target_quali_rank` / `target_race_rank` — volle Reihenfolge 1..N (Ranking-Modelle, Phase 5.1; deployt: Ridge für race + pre_weekend, LGBMReg für post_fp2).

---

## Herkunft & Feature-Hunt-Bilanz (Phase 5)

Über die MVP-Basis hinaus wurden in Phase 5 systematisch Zusatz-Features getestet (Build → No-Leakage-Test →
Ablation → Keep/Drop). **Behalten** (oben mit Phasen-Tag markiert): Überhol-Index (5.3), Team-Execution (5.4),
Teammate-Quali-Gap + Pit-Crew + Start (5.5), Nässe-Skill (5.7), Form-Momentum (5.9), Track-Cluster (5.11).

**Verworfen** (Details in `PLANNING.md` §16–§30): DNF-Sub-Modell (5.2, TeamReliability unschlagbar),
Constructor-Trajektorie (5.6), Safety-Car-Strategie (5.8), H2H-Konkurrenten (5.10), Driver-ELO (5.12),
Top-Speed-Profil (5.13), FP2-Teammate-Gap (5.14), Quali-Programm-Effizienz (5.15), Restart-Performance
(5.16, NaN-Fingerprint-Leak).

**Stand:** Der Feature-Satz ist **gesättigt** — Grid + Quali + Pace + die behaltenen Skill-Deltas fangen das
vorhersagbare Signal ab; weitere Mikro-Features fügen den deployten Modellen Varianz statt Signal hinzu.
Die Feature-Jagd ist damit abgeschlossen (siehe `PLANNING.md` §29/§31).
