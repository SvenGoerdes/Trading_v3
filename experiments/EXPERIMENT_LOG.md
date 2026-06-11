# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Aktueller Champion: **EXP-001 (mean val Sharpe −9.08)**

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
