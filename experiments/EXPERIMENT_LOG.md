# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Aktueller Champion: **noch keiner** — EXP-001 etabliert die Baseline.

---

## EXP-001 — 2026-06-11 — Baseline (365-Tage-Daten)

**Hypothese:** — (Baseline-Etablierung, keine Änderung)
**Änderung:** keine; champion.yml mit Screening-Budget (200k Steps, Seeds [42, 123], max_folds 2)
**Basis:** Commit fc80fc8, data_fingerprint `ec2e07548f555da2` (365 Tage, 7 Folds verfügbar)
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~19 min/Fold auf M1 Pro MPS, Gesamt ~75 min)
**Status:** 🟡 LÄUFT
**Ergebnis (bisher):** Fold 0 / Seed 42: Sharpe −9.86, CPR 0.58, MaxDD 0.47
**Entscheidung:** —
**Modelle:** `models/exp_001/`
**MLflow:** Experiment `td3_crypto_trading`, Runs `seed_42`, `seed_123`
**Learnings:** —

**Anmerkung Vorlauf:** Erster Startversuch am 2026-06-10 auf 94-Tage-Daten (nur 1 Fold) wurde
abgebrochen (Laptop unterwegs); danach Datenbasis auf 365 Tage erweitert. Frühe Erkenntnis aus
Fold 0: Baseline-Agent verliert deutlich (CPR 0.58 = −42 % im Validierungsmonat) — viel Raum
für Verbesserung, genau dafür ist der Loop da.
