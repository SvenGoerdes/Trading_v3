# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Aktueller Champion: **EXP-002 (mean val Sharpe −5.99)** — No-Trade-Band 0.05

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
