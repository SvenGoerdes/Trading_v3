# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Aktueller Champion: **EXP-005 (mean val Sharpe −1.12)** — Turnover-Penalty 0.004 (auf Band 0.10)

---

## EXP-006 — 2026-06-12 — Cross-Sectional-Momentum-Feature (Observation, momentum_window 12)
**Hypothese:** Alle TIs sind PRO ASSET rolling-z-skaliert → der Agent sieht nie die
  QUERSCHNITTLICHE relative Stärke (Asset X vs die anderen 9). Ein quer über die 10 Assets
  z-skaliertes 12-Bar-Momentum (1h, nur Vergangenheit) gibt dem Actor genau dieses fehlende
  Ranking-Signal. Erwartung: Trades konzentrieren sich auf relative Gewinner statt gleichmäßig
  zu streuen; win_rate/profit_factor lösen sich vom 0.5-Münzwurf.
**Änderung:** environment.cross_sectional_momentum: false → true (momentum_window 12)   (Basis: EXP-005)
**Basis:** Commit 18882e0, data_fingerprint `ec2e07548f555da2` (Daten gepaart mit EXP-001..005;
  ACHTUNG: Obs-Dimensionalität +n_assets → auf der Observation-Achse NICHT mit der Kosten-Kette
  gepaart, frischer Struktur-Baseline für den Signal-Branch, Vergleich nur auf Headline-Metrik).
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−1.33** (std 0.53) | MaxDD **0.36** | CPR 0.85
  Pro Fold: S42 F0 −1.51 / F1 −2.21 · S123 F0 −0.16 / F1 −1.45
  Gepaart (indikativ, da Obs-Dim geändert): 2:2 (S123 beide besser, S42 beide schlechter)
**Entscheidung:** ❌ ABGELEHNT — Delta −0.21 < +0.05-Schwelle, MaxDD unverändert (0.360 vs 0.355),
  kein Crash/NaN. Champion bleibt EXP-005. **NICHT als vielversprechend markiert** (Begründung unten).
**Modelle:** `models/exp_006/`
**MLflow:** Runs `seed_42` (d703684a), `seed_123` (29ef5f50), Experiment `td3_crypto_trading`
**Learnings:** Das Feature wirkt NICHT wie hypothetisiert. **Kern-Befund: Trade-Konzentration
  unverändert** — per-asset Trade-HHI 0.109/0.112 vs EXP-005 0.105/0.110 (1/10 = 0.10 = uniform);
  der Agent streut Trades weiter gleichmäßig über alle 10 Assets = derselbe Churn wie die ganze
  Kette. Kein Winner-Konzentrations-Effekt. Signal-Metriken nur im Rauschen: win_rate 0.47/0.43
  (EXP-005 0.40/0.46), profit_factor 0.59/0.62 (0.49/0.61), SMA20-Timing 50–54 % (Münzwurf).
  total_cost weiter gesunken (988/930 → 831/802) und n_trades (297/286 → 257/247), aber das ist
  Konvergenz-Drift, kein Edge. **Decisive Diagnose = UNTERTRAINING, nicht fehlendes Signal:** über
  die letzten 40k Steps BEIDER Seeds wandert der Actor noch — actions/mean steigt (s42 0.57→0.64,
  s123 0.58→0.65), actions/std fällt (0.74→0.69), reward_mean kriecht weiter Richtung 0. Die
  Politik ist beim 200k-Cutoff NICHT konvergiert. critic_loss sauber (2–5e-6) → kein Instabilitäts-/
  LR-Problem, schlicht zu wenig Optimierungszeit. Dazu Seed-Varianz explodiert (std 0.53 vs 0.41;
  S42/F0 −1.51 vs S123/F0 −0.16) = unterkonvergierte Actors landen pro Seed in anderen Basins.
  **Zwei offene Hypothesen für die Ablehnung:** (A) Kette ist bei 200k untertrainiert, (B) Signal
  nutzlos / Bottleneck ist Policy/Algorithmus. Lassen sich nicht in einem Experiment trennen
  („Momentum + längeres Training" = zwei Änderungen). **Verzweigung → EXP-007:** isoliert Variable
  (A) AUF DER CHAMPION-Observation (Momentum aus): total_timesteps 200k → 400k. Hilft 400k dem
  Champion → Kette war untertrainiert → EXP-008 = Momentum AN + 400k (Signal-Branch mit adäquatem
  Budget erneut testen). Hilft 400k NICHT → 200k reichte, EXP-006-Ablehnung ist echt, Bottleneck
  ist Repräsentation/Algorithmus (net_arch oder SAC), nicht Zeit. Anmerkung: `perf/*`-Metriken
  weiter ohne `step=` geloggt → Endwert = fold_1-Snapshot (fold_0-Signal nicht direkt beobachtbar).

---

## EXP-005 — 2026-06-11 — Turnover-Penalty höher (turnover_penalty_coef 0.002 → 0.004)
**Hypothese:** Der Kosten-Hebel klingt ab, ist aber noch nicht erschöpft (Spiegel des
  Band-Schritts 0.05→0.10). Verdopplung des Coef auf 0.004 senkt Trades/Kosten weiter und
  hebt Sharpe erneut — SOFERN der Agent die Spar-/Schaden-Schwelle nicht überschreitet.
**Änderung:** environment.turnover_penalty_coef: 0.002 → 0.004   (Basis: EXP-004)
**Basis:** Commit 30578b4, data_fingerprint `ec2e07548f555da2` (gepaart mit EXP-001..004)
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−1.12** (std 0.41) | MaxDD **0.36** | CPR 0.87
  Pro Fold: S42 F0 −0.20 / F1 −1.21 · S123 F0 −0.78 / F1 −2.27
  **Alle 4 Läufe besser** (gepaart): S42/F0 +0.93, S42/F1 +2.39, S123/F0 +0.55, S123/F1 +0.09
**Entscheidung:** ⭐ ADOPTIERT → neuer Champion. Delta +0.99 ≫ 0.05-Schwelle, MaxDD besser
  (0.355 vs 0.361), kein Crash/NaN. Vorläufig bis Confirmation (1M Steps, 5 Seeds).
**Modelle:** `models/exp_005/`
**MLflow:** Runs `seed_42` (b9033167), `seed_123` (56730aa2), Experiment `td3_crypto_trading`
**Freeze-Guard (eigenes Kill-Kriterium aus dem EXP-005-Vorschlag) → FEUERT NICHT:** Bei coef
  0.004 übersteigt der Penalty auf High-Turnover-Steps das Return-Signal — trotzdem **kein
  Freeze, kein All-Cash, kein Action-Kollaps.** Trades (fold_1-Snapshot) 413/355 → **297/286**
  (−28 %/−19 %, kein Asset auf null, Minimum 26 Trades), `actions/std` **0.79 → 0.76/0.71**
  (nur leicht gesunken), `actions/mean` **0.48 → 0.56/0.62** (Agent hält GRÖSSERE Positionen,
  geht NICHT in Cash), total_cost 1275/1156 → **988/930** (−22 %/−20 %). critic_loss sauber
  (5–6e-6). → ADOPT erfüllt, Freeze-Guard bestätigt aktive Politik.
**Learnings:** Der Gewinn ist **Regime + Kostensenkung, nicht Direktions-Edge.** Zerlegung
  fold_1 (Brutto vs Kosten): S42 brutto −24.0 % → **−10.4 %** UND Kosten −12.8pp → −9.9pp
  (beides echt verbessert); S123 brutto −16.7 % → **−18.9 %** (SCHLECHTER), nur Kosten besser
  → S123/F1 ist kosten-only und sub-Schwelle (+0.09). fold_0 CPR 0.99/0.96 = aktives, mild
  profitables Halten im Up-Markt (KEINE Inaktivität — siehe Trades/std). win_rate 0.40–0.46,
  profit_factor 0.49–0.61, SMA20-Timing 48–57 % = weiterhin **Münzwurf**. **Kosten-Hebel
  ERSCHÖPFT:** total_cost 5163 (EXP-001) → ~960 (EXP-005), **−81 %**; Spar-/Schaden an der
  Schwelle (S123/F1 schon kosten-only). **Verzweigung jetzt zwingend → Signal/Observation:**
  Wurzelursache des fehlenden Edge ist strukturell — alle TIs sind PRO ASSET rolling-z-skaliert,
  der Agent sieht nie die QUERSCHNITTLICHE relative Stärke (Asset X vs die anderen 9). EXP-006:
  Cross-Sectional-Momentum-Feature in die Observation (strukturell/Code, defaults OFF). Anmerkung:
  `perf/*`-Metriken werden ohne `step=` geloggt → Endwert = letzter Fold (fold_1); fold_0-Kosten/
  Brutto nicht direkt beobachtbar (kleiner Logging-Zusatz wäre nützlich, nicht blockierend).

---

## EXP-004 — 2026-06-11 — Turnover-Penalty im Reward (turnover_penalty_coef 0.0 → 0.002)
**Hypothese:** Das No-Trade-Band wirkt nur mechanisch; der Kosten-Term im Log-Return-Reward
  (~−4e-5/Step) liegt ~50× unter dem Reward-Rauschen (~2e-3/Step), der Gradient „sieht" die
  Kosten nicht. Ein expliziter Turnover-Penalty macht die Handelskosten lernbar — der Critic
  kann Verlust dem Handeln zuordnen, der Actor lernt Halten.
**Änderung:** environment.turnover_penalty_coef: 0.0 → 0.002   (Basis: EXP-003)
**Basis:** Commit 553f59c, data_fingerprint `ec2e07548f555da2` (gepaart mit EXP-001/002/003)
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−2.11** (std 0.26) | MaxDD **0.36** | CPR 0.81
  Pro Fold: S42 F0 −1.13 / F1 −3.61 · S123 F0 −1.34 / F1 −2.36
  3 von 4 Läufen besser; einzige gepaarte Regression S42/F1 (−3.37 → −3.61, −0.23, Down-Markt-Fold)
**Entscheidung:** ⭐ ADOPTIERT → neuer Champion. Delta +1.56 ≫ 0.05-Schwelle, MaxDD besser
  (0.36 vs 0.42), kein Crash/NaN. Vorläufig bis Confirmation (1M Steps, 5 Seeds).
**Modelle:** `models/exp_004/`
**MLflow:** Runs `seed_42` (be3e5928), `seed_123` (1cc469ef), Experiment `td3_crypto_trading`
**Learnings:** Der Penalty wirkt **gelernt, nicht mechanisch** (Band unverändert 0.10): Trades
  690/714 → 413/355 (−45 %), total_cost 2170/2197 → 1275/1156 (−45 %). Kosten-Term jetzt im
  Reward sichtbar (reward_mean −3.1e-4 → −7.4e-4 früh, ~22× Rauschen statt 50× darunter),
  critic_loss konvergiert sauber (3–6e-6, keine Instabilität). **Keine Über-Bestrafung / kein
  Freeze:** actions/std ~0.79, action_mean 0.32 → 0.48 (Agent **hält größere Positionen**, geht
  nicht in Cash), Trades fallen gleichmäßig über alle 10 Assets, keines auf null. **Brutto nicht
  robust besser** (fold_1: S42 brutto −18.7 → −24.0 schlechter, S123 −18.9 → −16.7) — Gewinn kommt
  aus Kostensenkung + Regime-Rückenwind auf fold_0, **nicht aus Direktions-Edge** (win_rate
  0.42–0.46, profit_factor 0.58–0.67, Timing ~48–53 % = Zufall). **Fold-Asymmetrie = Regime:**
  fold_0 (Aug11–Sep11, mild aufwärts: SOL +23 %, ETH +2 %) → Sharpe −1.1/−1.3, weil Halten belohnt
  wird; fold_1 (Sep11–Okt11, breiter Abwärtstrend: ETH −12 %, SOL −16 %) → bleibt schwach −3.6/−2.4,
  weil der long-biased Agent ohne Edge in den Selloff gezogen wird und Kostensenkung das nicht
  heilt. **Verzweigung:** Kosten-Hebel decaying aber **noch nicht erschöpft** (355–413 Trades/Fold
  Spielraum) → EXP-005 Penalty **0.002 → 0.004** (Hebel-Verlängerung, mirror Band-0.05→0.10).
  Danach zwingend Richtung **Signal/Observation** (Direktions-Edge), nicht weiter Kosten-Shaping.
  Anmerkung: `info["turnover"]` wird pro Step exponiert, aber nicht als `perf/turnover` aggregiert;
  perf/*-Metriken werden pro Run nur einmal (= fold_1) geloggt → fold_0-Kosten/Brutto nicht direkt
  beobachtbar. Kleiner Logging-Zusatz wäre nützlich (nicht blockierend).

---

## EXP-003 — 2026-06-11 — No-Trade-Band breiter (rebalance_threshold 0.05 → 0.10)
**Hypothese:** Kosten dominierten in EXP-002 den Restverlust (32 von 46 %pt). Ein breiteres
  Band (10 %pt statt 5) sollte die Kosten erneut um ~ein Drittel senken und Sharpe weiter heben.
**Änderung:** environment.rebalance_threshold: 0.05 → 0.10   (Basis: EXP-002)
**Basis:** Commit 1b8abb7, data_fingerprint `ec2e07548f555da2` (gepaart mit EXP-001/002)
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−3.67** (std 0.18) | MaxDD **0.42** | CPR 0.71
  Pro Fold: S42 F0 −3.61 / F1 −3.37 · S123 F0 −3.63 / F1 −4.08
  3 von 4 Läufen besser; **S123/F1 regressiert −3.19 → −4.08 (erste gepaarte Regression der Band-Serie)**
**Entscheidung:** ⭐ ADOPTIERT → neuer Champion. Delta +2.31 ≫ 0.05-Schwelle, MaxDD besser
  (0.42 vs 0.54), kein Crash/NaN. Vorläufig bis Confirmation (1M Steps, 5 Seeds).
**Modelle:** `models/exp_003/`
**MLflow:** Runs `seed_42` (6491c1f7), `seed_123` (23307f41), Experiment `td3_crypto_trading`
**Learnings:** Kosten-Hebel **am Abklingen, aber noch nicht erschöpft**. Kosten/Brutto-Zerlegung
  (fold_1-Snapshot): total_cost **3.181 → 2.184** (−31 %), Trades **~1.8k → ~702/Fold** (Agent
  handelt nur noch 0.8 % aller Asset-Step-Gelegenheiten). **Aber** Brutto-Rendite weiter schlechter
  ~−14.5 % → −18.8 %. Spar-/Schaden-Verhältnis pro Band-Schritt **5.15 (0→0.05) → 2.33 (0.05→0.10)**.
  Kosten dominieren den Rest **nicht mehr** (22 % Kapital Kosten vs 19 % Brutto-Verlust): selbst bei
  Nullkosten verliert der Agent ~19 %, weil **kein Richtungs-Edge** (win_rate 0.47–0.49, Timing ~48/49
  vs SMA20, Trades gleichmäßig über alle 10 Assets = reiner Churn). Regime-Sharpe Up/Down (−89/−75)
  ist Annualisierungs-Rauschen (14/31 Steps); Headline = Sideways-Regime (8.545 Steps). **Verzweigung
  jetzt:** Band mechanisch ausgereizt → nächstes Experiment **Turnover-Penalty im Reward** (EXP-004,
  strukturell/Code), damit der Gradient die Handelskosten endlich „sieht" und Halten gelernt wird.

---

## EXP-002 — 2026-06-11 — No-Trade-Band (rebalance_threshold 0.05)
**Hypothese:** Verlust war in EXP-001 fast vollständig kostengetrieben (~52 % Kapital in
  Gebühren, ~41k Trades/Fold). Ein No-Trade-Band (Asset nur rebalancieren, wenn
  |Ziel−Ist-Gewicht| > 5 %pt) killt den Per-Step-Micro-Churn und senkt die Kosten drastisch.
**Änderung:** environment.rebalance_threshold: 0.0 → 0.05   (Basis: EXP-001)
**Basis:** Commit 00c71bf, data_fingerprint `ec2e07548f555da2` (gepaart mit EXP-001)
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−5.99** (std 0.16) | MaxDD **0.54** | CPR 0.59
  Pro Fold: S42 F0 −7.77 / F1 −4.52 · S123 F0 −8.47 / F1 −3.19 (alle 4 Läufe besser als EXP-001)
**Entscheidung:** ⭐ ADOPTIERT → neuer Champion. Delta +3.09 ≫ 0.05-Schwelle, MaxDD besser
  (0.54 vs 0.62), kein Crash/NaN. Vorläufig bis Confirmation (1M Steps, 5 Seeds).
**Modelle:** `models/exp_002/`
**MLflow:** Runs `seed_42` (adf3a5eb), `seed_123` (4b48caa6), Experiment `td3_crypto_trading`
**Learnings:** Das Band wirkt rein **mechanisch** (Env-Filter, keine gelernte Politik —
  reward_mean halbiert auf −4.5e-5, Reward-SNR sogar gesunken 0.056 → 0.021, Actions weiter
  bei ±1 gesättigt). total_cost **5.162 → 3.181** (−38 %), Trades **~41.5k → ~1.8k/Fold**
  (~23× weniger). **Aber:** Brutto-Rendite (vor Kosten) ~−10.6 % → −14.5 % leicht
  schlechter; Restverlust weiter **kostendominiert** (32 von 46 %pt), gleichzeitig **kein
  Direktions-Edge**: win_rate 0.47, profit_factor 0.74, SMA20-Timing ~52/47, Regime-Sharpe
  Uptrend ~−89 / Downtrend ~−80 / Sideways ~−4. → Kosten-Hebel noch nicht ausgereizt:
  nächstes Experiment **Band 0.05 → 0.10** (EXP-003). Danach Verzweigung Richtung
  Signal/Reward-Shaping, sobald das Band plateaut.

---

## EXP-001 — 2026-06-11 — Baseline (365-Tage-Daten)
**Hypothese:** — (Baseline-Etablierung, keine Änderung)
**Änderung:** keine; champion.yml mit Screening-Budget (200k Steps, Seeds [42, 123], max_folds 2)
**Basis:** Commit 2583dd1, data_fingerprint `ec2e07548f555da2` (365 Tage, 7 Folds verfügbar)
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~19 min/Fold auf M1 Pro MPS, Gesamt 1h18m)
**Status:** ✅ ABGESCHLOSSEN — wird per Definition Champion
**Ergebnis:** mean val Sharpe **−9.08** (std 0.023) | MaxDD 0.62 | CPR 0.46
  Pro Fold: S42 F0 −9.86 / F1 −8.26 · S123 F0 −12.11 / F1 −6.10
**Entscheidung:** ⭐ CHAMPION (Baseline) — Referenz für alle künftigen Challenger
**Modelle:** `models/exp_001/`
**MLflow:** Experiment `td3_crypto_trading`, Runs `seed_42` (36447a17), `seed_123` (99b5a196)
**Learnings:** Verlust ist **fast vollständig durch Transaktionskosten** verursacht.
  total_cost ≈ 5.100–5.225 von 10.000 Startkapital (~52 %), total_return ≈ −62 %.
  ~41.000 Trades / 8.590 Steps × 10 Assets → Agent rebalanciert quasi jeden Step jedes
  Asset. Kein Edge: win_rate ~0.50, profit_factor ~0.82, Timing 51/49 vs SMA20 → brutto
  break-even, Kosten machen daraus −62 %. **Kein Lernproblem:** critic_loss konvergiert
  sauber (7e-4 → 5e-6), buffer/reward_mean plateaut ab ~110k bei −0.00012/Step (≈18× kleiner
  als Reward-Rauschen 0.0021 → Kostensignal im Reward zu schwach, um Churn zu bestrafen).
  → Nächster Hebel ist strukturell (No-Trade-Band / Turnover-Penalty), nicht LR/Exploration.

**Anmerkung Vorlauf:** Erster Startversuch am 2026-06-10 auf 94-Tage-Daten (nur 1 Fold) wurde
abgebrochen (Laptop unterwegs); danach Datenbasis auf 365 Tage erweitert. Ein zweiter Lauf
crashte nach Seed 42 an einem Off-by-one in der Evaluations-Diagnostik (gefixt in 2583dd1);
der finale Lauf reproduzierte die Seed-42-Metriken exakt (Determinismus bestätigt).
