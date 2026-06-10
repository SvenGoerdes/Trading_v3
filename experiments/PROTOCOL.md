# Experiment-Loop — Protokoll

Autonomer Optimierungs-Loop für den TD3-Trading-Agenten. Claude überwacht den Loop:
trainieren → evaluieren → loggen → entscheiden → nächstes Experiment.

## Idee (Champion/Challenger)

Es gibt immer genau einen **Champion** — die beste bisher gefundene Konfiguration
(`experiments/champion.yml`). Jedes Experiment ist ein **Challenger**: eine Kopie des
Champions mit **genau einer Änderung** (Hypothese). Der Challenger wird trainiert und
auf den Validierungs-Folds evaluiert.

- **Gewinnt der Challenger** → er wird neuer Champion, die Änderung wird übernommen.
- **Verliert er** → Ergebnis wird geloggt, die Änderung wird NICHT übernommen.

So entsteht eine monotone Verbesserungskette, und jede Entscheidung ist dokumentiert
und reproduzierbar.

## Hardware-Realität (M1 Pro, MPS)

Gemessen: **~135 steps/sec** → 1M Steps ≈ 2h pro Fold. Voller Lauf (5 Seeds × 3 Folds)
≈ 30h. Darum zwei Budget-Stufen:

| Stufe | Steps | Seeds | Folds | Dauer/Experiment | Zweck |
|---|---|---|---|---|---|
| **Screening** | 200k | 2 (42, 123) | 2 | ~1h 40m | Jeder Challenger |
| **Confirmation** | 1M | 5 | alle | ~30h | Nur adoptierte Champions, gelegentlich |

Screening-Läufe sind über alle Experimente **gepaart**: identische Seeds, identische
Fold-Zeiträume, identische Daten (Splits werden während des Loops nicht neu erzeugt).
Nur so sind Vergleiche zwischen Experimenten fair.

## Entscheidungsregel

Primärmetrik: **mittlere Validierungs-Sharpe-Ratio** über alle Seed×Fold-Läufe.

Adoptiert wird ein Challenger nur, wenn:
1. `mean_val_sharpe(challenger) > mean_val_sharpe(champion) + 0.05` (Rauschschwelle), UND
2. `max_drawdown` nicht mehr als 20 % (relativ) schlechter als der Champion, UND
3. kein Lauf abgestürzt ist / keine NaN-Metriken.

Bei knappen Ergebnissen (Differenz < Schwelle): NICHT adoptieren, aber als
"vielversprechend" markieren — Kandidat für Wiederholung mit größerem Budget.

**Ehrlichkeitsregel:** 2 Seeds × 2 Folds = 4 Datenpunkte → verrauscht. Die Schwelle
schützt vor Zufallstreffern, garantiert aber nichts. Adoptionen sind vorläufig, bis
eine Confirmation mit vollem Budget sie bestätigt.

## Suchraum (was variiert werden darf)

Eine Änderung pro Experiment, grob nach erwartetem Hebel sortiert:

1. **Lernraten** (actor/critic), LR-Schedule (linear/cosine/konstant)
2. **Reward-Shaping**: reward_scaling, ggf. Kosten-Penalty
3. **Exploration**: action_noise_std, learning_starts
4. **Netzarchitektur**: net_arch pi/qf ([256,128] → größer/kleiner)
5. **Replay**: batch_size, buffer_size, train_freq, policy_delay
6. **Environment**: window_size, max_position
7. **Algorithmus-Wechsel**: SAC statt TD3 (größerer Umbau, eigenes Experiment-Cluster)

Nicht angefasst werden: Daten-Splits, Fold-Definitionen, Gebühren/Slippage,
Testdaten (bleiben bis zum Schluss unberührt).

## Ablauf eines Loop-Zyklus

1. **Propose** — Claude formuliert eine Hypothese und erzeugt
   `experiments/configs/exp_NNN_<slug>.yml` (Kopie von champion.yml + 1 Änderung).
2. **Train** — Training läuft als Hintergrundprozess:
   `TRADING_CONFIG=experiments/configs/exp_NNN.yml uv run python -m src.run_pipeline --pipeline=training`
   Claude wird bei Prozessende benachrichtigt (Fallback: Wakeup-Heartbeat).
3. **Evaluate** — Ergebnis-JSON aus `experiments/results/exp_NNN.json` lesen
   (vom Trainings-Pipeline geschrieben), mit Champion vergleichen.
4. **Decide** — Entscheidungsregel anwenden, Adoption oder Ablehnung.
5. **Log** — Eintrag in `EXPERIMENT_LOG.md` (menschenlesbar) und `ledger.jsonl`
   (maschinenlesbar). Bei Adoption: champion.yml ersetzen + Git-Commit.
6. **Repeat** — nächste Hypothese, informiert durch alle bisherigen Ergebnisse.

## Verzeichnisstruktur

```
experiments/
├── PROTOCOL.md           # dieses Dokument
├── EXPERIMENT_LOG.md     # Journal: ein Eintrag pro Experiment
├── ledger.jsonl          # maschinenlesbar: 1 JSON-Zeile pro Experiment
├── champion.yml          # aktuell beste Konfiguration (vollständige Kopie)
├── configs/
│   ├── exp_001_baseline.yml
│   └── exp_002_<slug>.yml
└── results/
    └── exp_001.json      # aggregierte Metriken (vom Training geschrieben)
```

## Log-Format

### EXPERIMENT_LOG.md (pro Experiment)

```markdown
## EXP-002 — 2026-06-10 — höhere Critic-LR
**Hypothese:** Critic lernt mit 3e-4 zu langsam; 1e-3 beschleunigt Q-Konvergenz.
**Änderung:** td3.learning_rate.critic: 3e-4 → 1e-3   (Basis: EXP-001)
**Budget:** 200k Steps, Seeds [42, 123], Folds 0–1
**Ergebnis:** mean val Sharpe 0.31 (Champion: 0.44) | MaxDD 0.21 | CPR 0.98
**Entscheidung:** ❌ ABGELEHNT — Sharpe deutlich schlechter, Training instabil ab ~120k
**MLflow:** seed_42 / seed_123 unter Experiment td3_crypto_trading
**Learnings:** Critic-LR nach oben bringt nichts; eher Richtung kleinerer Actor-LR testen.
```

### ledger.jsonl (pro Experiment, eine Zeile)

```json
{"id": "exp_002", "date": "2026-06-10", "base": "exp_001", "change": {"td3.learning_rate.critic": [0.0003, 0.001]}, "budget": {"steps": 200000, "seeds": [42, 123], "folds": 2}, "metrics": {"mean_val_sharpe": 0.31, "mean_max_drawdown": 0.21, "mean_cpr": 0.98}, "champion_sharpe": 0.44, "decision": "rejected", "promising": false}
```

## Nötige Code-Änderungen (minimal)

1. **Config-Override**: `get_config()` liest Pfad aus Env-Var `TRADING_CONFIG`
   (Fallback: `conf/parameters.yml`). Eine Zeile in `src/utils/config.py`.
2. **Ergebnis-Export**: Training schreibt am Ende die aggregierten Cross-Seed-Metriken
   als JSON nach `experiments/results/<EXPERIMENT_ID>.json`, wenn Env-Var
   `EXPERIMENT_ID` gesetzt ist. Kleiner Block in `src/pipelines/training/pipeline.py`.
3. **Screening-Parameter**: keine Code-Änderung nötig — total_timesteps, seeds und
   CV-Monate stehen bereits in der YAML und werden pro Experiment-Config gesetzt.

## Modell-Persistenz (nie verwerfen)

Jedes Experiment bekommt ein eigenes Modellverzeichnis — nichts wird überschrieben:

```
models/
└── exp_001/
    ├── td3_seed42_best.zip     # bestes Modell pro Seed (nach Val-Sharpe)
    ├── td3_seed123_best.zip
    ├── config_used.yml         # exakte Config dieses Laufs (Snapshot)
    └── provenance.json         # git_commit, config_source, experiment_id, Zeitstempel
```

Läufe ohne `EXPERIMENT_ID` landen in `models/run_<UTC-Zeitstempel>/`. Die Ergebnis-JSON
(`experiments/results/<exp_id>.json`) verweist auf `model_dir` und die Modellpfade pro
Seed. Modelle sind gitignored (Binärdateien); die Traceability-Kette läuft über Git
(Configs, Results, Log) + `provenance.json` neben dem Modell.

## Git-Workflow (Nachvollziehbarkeit)

Ziel: Aus der Git-History ist ablesbar, **welche Änderung zu welchem besseren Modell
geführt hat**.

1. **Vor jedem Experiment-Start**: sauberer Tree; Experiment-Config wird committet:
   `exp: EXP-NNN <slug> — start`. Der Commit-Hash wandert via provenance.json und
   MLflow-Param `git_commit` in alle Artefakte des Laufs.
2. **Nach der Auswertung**: Ergebnis-JSON, Log-Eintrag, Ledger-Zeile committen:
   `exp: EXP-NNN <slug> — <ADOPTED|REJECTED> (val Sharpe X.XX vs Y.YY)`.
3. **Bei Adoption** zusätzlich: champion.yml im selben Commit aktualisieren und Tag
   setzen: `git tag champion-exp-NNN`. Die Tag-Folge ist die Verbesserungskette.

So gilt: Modell auf Platte → provenance.json → Commit-Hash → exakter Code- und
Config-Stand. Und umgekehrt: `git log --oneline | grep ADOPTED` zeigt die Kette der
erfolgreichen Änderungen.

## Sicherheits-Regeln

- `conf/parameters.yml` wird vom Loop **nie** direkt verändert; Adoption = Commit von
  champion.yml-Änderungen, nachvollziehbar in Git.
- Testdaten (`*_test.parquet`) werden im Loop nicht angefasst — nur Validierungs-Folds.
- Jedes Experiment loggt den Git-Commit-Hash mit (Reproduzierbarkeit).
- Abgestürzte Läufe zählen als Ablehnung mit Grund "crash", nie stillschweigend ignorieren.
