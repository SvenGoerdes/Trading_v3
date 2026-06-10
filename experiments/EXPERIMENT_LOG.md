# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Aktueller Champion: **noch keiner** — EXP-001 etabliert die Baseline.

---

## EXP-001 — 2026-06-10 — Baseline

**Hypothese:** — (Baseline-Etablierung, keine Änderung)
**Änderung:** keine; champion.yml mit Screening-Budget (200k Steps, Seeds [42, 123], max_folds 2)
**Basis:** conf/parameters.yml (Stand fcc1603 + MLflow-Refactor)
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~1h 40m auf M1 Pro MPS, ~135 steps/s)
**Status:** 🟡 LÄUFT
**Ergebnis:** —
**Entscheidung:** —
**Modelle:** `models/exp_001/`
**MLflow:** Experiment `td3_crypto_trading`, Runs `seed_42`, `seed_123`
**Learnings:** —
