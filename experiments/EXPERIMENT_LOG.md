# Experiment-Log

Journal aller Experimente des autonomen Optimierungs-Loops. Regeln: siehe [PROTOCOL.md](PROTOCOL.md).
Maschinenlesbare Version: [ledger.jsonl](ledger.jsonl). Modelle: `models/<exp_id>/` (nie überschrieben).

Champion **5m-Kette** (fp ec2e07548f555da2): **EXP-007 (mean val Sharpe −0.85)** — EXP-005-Config, 400k Steps
Champion **1h-Kette** (fp b279adaf24c6ed85): **EXP-014 (mean val Sharpe −1.83)** — 1h, scaled Cash-Ausweg
> ACHTUNG: Die beiden Ketten haben unterschiedliche data_fingerprints (5m vs 1h) → Sharpe-Werte NICHT
> direkt vergleichbar. Innerhalb jeder Kette gilt der jeweilige Champion. 5m-RL-Raum erschöpft (12 Exp,
> TD3+SAC); 1h-Kette ab EXP-013 offen.

---

## EXP-015 — 2026-06-16 — 1h SIGNAL: cross_sectional_momentum AN, 1-Tages-Trend-Horizont (momentum_window 24)
**Hypothese:** EXP-014 belegte, dass der Agent jetzt RISIKO MANAGT (de-riskt 98→51 % investiert
  in den fold_1-Selloff, MaxDD 0.246→0.105), aber weiter keinen GRUND hat, Gewinner zu wählen
  (win_rate 0.57–0.63 = Münzwurf, alle 4 Folds negativ). Alle TIs sind PRO ASSET rolling-z-skaliert
  → der Agent sieht nie die QUERSCHNITTLICHE relative Stärke. Cross-Sectional-Momentum (quer über die
  10 Assets z-skaliert, window 24 = 1 Tag bei 1h) gibt ihm erstmals ein Relative-Stärke-Ranking — UND
  diesmal auf einem Substrat, das WIRKLICH teilnimmt (~50 %) und Cash NUTZT (anders als 5m EXP-006/010,
  die auf kosten-vergifteten bzw. ~90%-Cash-Substraten liefen). Zweiseitiges Kill-Kriterium vorab
  registriert: (a) NO-LIFT/NO-CONCENTRATION (HHI ~0.10, win_rate ~0.5, Sharpe nicht > −1.83 + 0.05)
  → Signal auch bei 1h nicht prädiktiv, Observation NICHT der Bottleneck; (b) ECHTES Signal (HHI ≫ 0.10
  und/oder win_rate löst sich von 0.5, Sharpe-Lift) → erstmals Direktions-Edge → ADOPT.
**Änderung:** environment.cross_sectional_momentum: false → true (momentum_window 12 → 24)   (Basis: EXP-014, 1h-Champion)
**Basis:** Commit 7097ed8, data_fingerprint `b279adaf24c6ed85` (GEPAART mit EXP-013/EXP-014). ACHTUNG:
  csm=true erweitert den Obs-Vektor um n_assets → STRUKTURELLE/Observations-Änderung, auf der
  Observation-Achse NICHT mit EXP-014 gepaart (frischer Struktur-Baseline, gleiches Caveat wie 5m
  EXP-006/010). Headline-Vergleich indikativ; die WITHIN-Experiment-Diagnostik (HHI/win_rate) ist primär.
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~694 val Candles/Fold — 4-Datenpunkt-Readout, std 0.26)
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−1.61** (std 0.26) | MaxDD **0.093** | CPR 0.956
  Pro Fold: S42 F0 **−0.90** (CPR 0.991, MaxDD 0.048) / F1 −1.81 (CPR 0.947, MaxDD 0.107)
            S123 F0 −1.20 (CPR 0.982, MaxDD 0.077) / F1 **−2.54** (CPR 0.904, MaxDD 0.140)
  Gepaart 2:2 vs EXP-014: besser S42/F1 (+1.01), S123/F0 (+0.47); schlechter S42/F0 (−0.45),
  S123/F1 (−0.14). ALLE 4 Folds weiter negativ. Delta +0.22 = ~0.85σ (std 0.26) → im Rauschen.
**Entscheidung:** ❌ ABGELEHNT (Rausch-Ablehnung). **Mechanisch ADOPTIERT die Regel** (Delta +0.22 > 0.05;
  MaxDD 0.093 vs 0.105 BESSER; kein Crash/NaN), ABER das vorab registrierte **Kill-Kriterium (a)
  NO-LIFT/NO-CONCENTRATION FEUERT** und übersteuert per Ehrlichkeitsregel das verrauschte Headline-Plus:
  **Per-Asset-Trade-HHI 0.109 / 0.149** auf nur **16 / 22 Trades** (~2 Trades/Asset = Integer-Count-
  Rauschen, identisch zum uniformen 0.10-Floor der GESAMTEN Kette). **win_rate 0.83 / 0.44** auf 6/9
  Trades = Geldmarkt-Artefakt, KEIN Lift vom 0.5-Münzwurf. `actions/mean` bei −0.05 / +0.02 gepinnt
  (derselbe Near-Zero-Attraktor wie EXP-014), critic konvergiert sauber 1.3–1.5e-6 = NICHT untertrainiert.
  **cash_ratio_mean 0.559 / 0.508** (< 0.85) → das Feature drückt den Agenten NICHT zurück in All-Cash;
  die EXP-014-Teilnahme bleibt erhalten. Der Agent SIEHT das Ranking und IGNORIERT es. **Champion bleibt EXP-014.**
**Modelle:** `models/exp_015/`
**MLflow:** Runs `seed_42` (8031a755), `seed_123` (bb3ac823), Experiment `td3_crypto_trading`
**Learnings — STRATEGISCHER ABSCHLUSS:** **Cross-Sectional-Momentum erzeugt auf KEINEM Horizont und
  KEINEM Substrat Winner-Konzentration** — bestätigt über 5m (renormalize EXP-006, ~90%-Cash scaled
  EXP-010) UND 1h (echtes ~50%-Teilnahme-Substrat EXP-015). Das war der sauberste mögliche Signal-Test
  (risikomanagender Agent, der Cash NUTZT und teilnimmt) — und er feuert NO-LIFT. Damit ist die
  **Observation/Signal-Achse für dieses Feature über beide Timeframes erschöpft.** Gesamtbild über 15
  Experimente: der RL-Agent extrahiert AUSSCHLIESSLICH Risikomanagement (Cash-Timing, EXP-014), NIE
  Security-Selection (win_rate ~0.5, HHI ~0.10 überall, Critics konvergiert, über TD3+SAC, beide
  Substrate, mit/ohne Signal). **FORMALER NO-EDGE-SCHLUSS für RL-on-TI in diesem 10-Asset-Universum** —
  kein fehlender Hyperparameter, der TI-Feature-Satz trägt keine cross-asset-prädiktive Information.
  **Empfehlung beim Wiederaufnehmen (kein EXP-016 gestartet, Loop-Ende auf User-Wunsch):** RL-Tuning
  STOPPEN. Zuerst eine billige OFFLINE-SUPERVISED-PROBE — kann IRGENDEIN Modell 1-Tages-Forward-Cross-
  Sectional-Returns aus den Features über Zufall vorhersagen? Wenn nein, findet die RL es auch nie.
  Wenn ja → neue Signalquelle in die Observation (Funding-Rates zuerst, dann Cross-Asset-Korrelation /
  Regime-Labels / supervised Edge-Modell). EXP-014-Confirmation (1M/5 Seeds) erst NACH der strategischen
  Richtungsentscheidung. Strukturelle Änderung → Orchestrator-Sign-off.

---

## EXP-014 — 2026-06-16 — 1h CASH-AUSWEG: allocation_mode renormalize → scaled
**Hypothese:** EXP-013 belegte: der 1h-Baseline-Verlust ist BRUTTO/DIREKTIONAL, nicht Kosten
  (renormalize ~98%/~83% investiert, total_cost nur 33/122, dennoch −21%/−20% Return, fold_1
  MaxDD 0.31–0.33). `scaled` (fester Divisor n_assets·max_position, Rest bleibt Cash) gibt dem
  Agenten einen echten De-Risk-Hebel, den die renormalize-Summe aufhebt. Zweiseitiges Kill-
  Kriterium vorab registriert: (a) Cash-Kollaps wie 5m (cash_ratio_mean > 0.85) → Artefakt;
  (b) echtes De-Risking (fold_0 investiert, fold_1 Cash, MaxDD/Sharpe besser) → ADOPT.
**Änderung:** environment.allocation_mode: "renormalize" → "scaled"   (Basis: EXP-013, 1h-Champion)
**Basis:** Commit b118a66, data_fingerprint `b279adaf24c6ed85` (GEPAART mit EXP-013, NICHT mit der
  5m-Kette). Strukturelle Execution-Änderung → Headline-Vergleich indikativ, per-fold-Diagnostik primär.
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~721 val Candles/Fold — 4-Datenpunkt-Readout, Vorsicht)
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−1.83** (std 0.20) | MaxDD **0.105** | CPR 0.944
  Pro Fold: S42 F0 **−0.45** (CPR 0.996, MaxDD 0.046) / F1 −2.82 (CPR 0.892, MaxDD 0.170)
            S123 F0 −1.67 (CPR 0.982, MaxDD 0.038) / F1 −2.40 (CPR 0.907, MaxDD 0.165)
  3 von 4 Folds besser als EXP-013 (gepaart: S42/F0 +1.39, S42/F1 +0.58, S123/F1 +1.18;
  einzige Regression S123/F0 −1.16). ALLE 4 Folds weiter negativ.
**Entscheidung:** ⭐ ADOPTIERT → neuer 1h-Champion. Delta +0.50 > 0.05-Schwelle, MaxDD 0.105 vs
  0.246 (−57 %, weit innerhalb der 20-%-Guard), kein Crash/NaN. **Kill-Kriterium (a) Cash-Kollaps
  FEUERT NICHT — und damit GERADE KEIN 5m-Artefakt:** `perf/cash_ratio_mean` **0.514 / 0.508**
  (< 0.85, cash_min 0.246/0.140) → im Down-Markt-Fold ~50 % investiert, ~50 % Cash. (5m-scaled lag
  bei 0.85–0.97 → All-Cash-Artefakt, EXP-008 trotz Headline-Sieg abgelehnt. Hier NICHT.)
  **Anti-Skill-Verdacht WIDERLEGT:** per-fold CPR zeigt fold_0 (Up/Sideways) ~flach (0.996/0.982,
  MaxDD 0.038–0.046 = niedrige Teilnahme, kein Verlust) und fold_1 (Down) kontrollierter ~50%-
  Verlust mit HALBIERTEM Drawdown — also regime-gerechte DIFFERENZIERTE Exposure-Senkung, KEIN
  Hineinkaufen in den Selloff. Vorläufig bis Confirmation (1M Steps, 5 Seeds).
**Modelle:** `models/exp_014/`
**MLflow:** Runs `seed_42` (83bef5b2), `seed_123` (3fcc70ed), Experiment `td3_crypto_trading`
**Learnings:** **`scaled` ist am 1h-Horizont echtes Risikomanagement, NICHT das 5m-All-Cash-Artefakt.**
  Der Unterschied zur 5m-Kette: dort umging `scaled` ein KOSTEN-Problem → All-Cash trivial optimal;
  bei 1h sind Kosten bereits vernachlässigbar (total_cost 25/30 vs 5160 bei 5m), also verdient sich
  der Cash-Ausweg nur als ECHTE Drawdown-Vermeidung — und tut das (fold_1 −21%→−11% Return,
  MaxDD 0.33→0.17). **Aber weiter KEIN Direktions-Edge:** win_rate 0.57–0.63, profit_factor 0.53–1.99
  auf 7–8 Trades, alle Folds negativ, critic konvergiert (~1.5–2.0e-6). Der Agent MANAGT jetzt Risiko,
  hat aber keinen GRUND, Gewinner zu wählen. **Verzweigung → EXP-015 (Signal-Branch):**
  cross_sectional_momentum AN, Trend-Horizont momentum_window 12 → 24 (= 1 Tag bei 1h). Anders als
  5m EXP-006/010 läuft der Signal-Test jetzt auf einem Substrat, das WIRKLICH teilnimmt (~50 %) und
  Cash NUTZT → die Bedingungen, ein Cross-Sectional-Ranking auszudrücken, existieren erstmals. Feuert
  auch das NO-LIFT (HHI ~0.10, win_rate ~0.5) → No-Edge-Schluss über BEIDE Timeframes × BEIDE Substrate
  + Signal-Hebel → ESKALATION: formaler No-Edge für RL-on-TI, strukturelles Umdenken statt RL-Tuning.

---

## EXP-013 — 2026-06-16 — DATEN-ACHSE: 1h-Baseline (Champion EXP-007 Env-Block @ 1h)
**Hypothese:** — (Baseline-Etablierung der NEUEN 1h-Daten-Achse, eine Änderung vs EXP-007:
  Timeframe 5m → 1h). Ehrliche Edge-Sonde: renormalize (Zwang ~Vollinvestition) KANN kein Cash
  halten, daher liest die 1h-Val-Sharpe direkt ab, ob ein vollinvestierter Agent bei ~12× geringerem
  Kosten-Drag am 1h-Horizont einen Direktions-Edge findet.
**Änderung:** timeframe: "5m" → "1h"   (Basis: Champion EXP-007 Env-Block, 200k Screening-Budget)
**Basis:** Commit 85ff837, data_fingerprint `b279adaf24c6ed85` (NEUER Fingerprint — NICHT mit
  EXP-001..012 gepaart; die 5m-Kette nutzte `ec2e07548f555da2`. EXP-013 IST der 1h-Referenzpunkt.
  Annualisierung timeframe-korrekt: periods_per_year 8760 (1h) statt 105120 (5m) → Sharpe-Zahlen im
  MEANING vergleichbar zu den 5m-Sharpes, aber NICHT gepaart).
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~1489 train / ~721 val Candles/Fold — 4-Datenpunkt-Readout
  noch verrauschter als bei 5m, Schwelle mit Vorsicht anwenden)
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−2.33** (std 0.29) | MaxDD **0.246** | CPR 0.876
  Pro Fold: S42 F0 **−1.84** (CPR 0.936, MaxDD 0.177) / F1 **−3.40** (CPR 0.788, MaxDD 0.329)
            S123 F0 **−0.51** (CPR 0.980, MaxDD 0.172) / F1 **−3.58** (CPR 0.800, MaxDD 0.306)
  ALLE 4 Folds negativ; fold_1 (Down-Markt) deutlich schlechter als fold_0 auf beiden Seeds —
  dieselbe Regime-Asymmetrie wie am 5m-Horizont.
**Entscheidung:** ⭐ CHAMPION (1h-Baseline) — Referenz für die gesamte 1h-Kette, per Definition
  (analog zu EXP-001 für die 5m-Kette). Kein Crash/NaN, deterministisch, beide Seeds vollständig.
  Der 1h-Pivot ist KEIN Totalausfall: −2.33 / MaxDD 0.246 ist deutlich gesünder als die 5m-renormalize-
  Baseline EXP-001 (−9.08 / MaxDD 0.62), die Kosten-Pathologie ist weg.
**Modelle:** `models/exp_013/`
**MLflow:** Runs `seed_42` (8c0647a4), `seed_123` (4abc2cfb), Experiment `td3_crypto_trading`
**Learnings — DEKOMPOSITION:**
  **(1) ECHTE, vollinvestierte Verluste — KEIN Near-Cash-Artefakt.** `perf/cash_ratio_mean`
  **0.020 (S42) / 0.173 (S123)**, cash_min 0.0 → Buch zu ~98 %/~83 % deployed. Per-Asset-Action-Means
  0.27–0.95 (alle long). Der Agent IST im Markt.
  **(2) Der Verlust ist BRUTTO/DIREKTIONAL, NICHT Kosten.** `perf/total_cost` **33 / 122** auf
  **7 / 38 Trades** — vs ~5160 Kosten / ~41k Trades bei der 5m-renormalize-Baseline EXP-001 (Kosten-Drag
  ~12× geringer, wie erwartet). TROTZDEM total_return **−21.2 % / −20.0 %**. Nach Abzug der trivialen
  Kosten verliert der Agent weiter ~20 % → KEIN Long-Edge am 1h-Horizont.
  **(3) Kein Direktions-Skill.** win_rate 0.14/0.29, profit_factor 0.13/0.17, SMA-Timing 20–50 %
  (Münzwurf) — gleiche edge-lose Signatur wie alle 12 5m-Experimente.
  **(4) Konvergiert, NICHT untertrainiert.** critic_loss 1e-4 → ~1.3e-6, flach ab ~105k.
  **STRATEGISCHER SCHLUSS:** Bei 1h mit ~12× geringerem Kosten-Drag verliert ein vollinvestierter Agent
  IMMER NOCH ~20 % auf Down-Markt-Folds → Failure-Mode = brutto/direktional MIT fehlendem Cash-Ausweg
  unter renormalize. **Verzweigung → EXP-014: allocation_mode renormalize → scaled.** Anders als bei 5m
  (EXP-008, wo scaled ein KOSTEN-Problem umging → All-Cash-Artefakt) kauft der Cash-Ausweg bei 1h nichts
  auf der Kosten-Achse — er kann sich nur als echtes Drawdown-Vermeiden auf Down-Markt-Folds verdienen
  (Risikomanagement, auch ohne Long-Edge wertvoll). Zweiseitiges Kill-Kriterium in EXP-014 vorab registriert.

---

## EXP-012 — 2026-06-16 — Algorithmus-Swap TD3 → SAC (Max-Entropy-Actor)
**Hypothese:** EXP-011 feuerte Kill-(a) NO-LIFT — verdoppelte Actor-Kapazität ([256,128]→[400,300])
  hob den Agenten NICHT vom Near-Cash-Attraktor; Repräsentations-Kapazität ist NICHT der Bottleneck.
  Letzter ungetesteter Hebel im Policy/Algorithmus-Branch: die Algorithmus-KLASSE. TD3s
  deterministischer Actor kollabiert bei edge-loser Reward-Landschaft auf die varianz-minimale
  Cash-Politik. SACs Maximum-Entropy-Ziel belohnt eine stochastische Politik und ist weniger
  kollaps-anfällig — falls IRGENDEINE handelbare Struktur existiert, ist die Entropie-Bonus + Twin-
  Soft-Q der wahrscheinlichste Cross-Familien-Hebel, sie sichtbar zu machen. Erwartung: cash_ratio_mean
  < 0.85, Trade-HHI ≫ 0.10, positive Sharpe über BEIDE Seeds.
**Änderung:** td3.algorithm: "td3" → "sac"   (Basis: EXP-010 `scaled` + Momentum-AN, net_arch [256,128])
**Basis:** Commit 7bb8e99, data_fingerprint `ec2e07548f555da2` (SAUBER mit EXP-010 gepaart auf jeder
  Achse außer der Algorithmus-Klasse). Caveats wie EXP-008..011: Budget 200k vs Champion-Headline 400k
  (EXP-007), `scaled`-Execution, SAC NICHT mit dem renormalize-Champion auf der Algorithmus-Achse
  gepaart → Headline-Vergleich nur indikativ.
**Budget:** 200k Steps × 2 Seeds × 2 Folds (~8h Wall-Clock: SAC 30 steps/sec, 7–8× langsamer als TD3)
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−0.61** (std 0.65) | MaxDD **0.066** | CPR 0.981
  Pro Fold: S42 F0 **−1.41** (CPR 0.986, MaxDD 0.042) / F1 −1.10 (CPR 0.954, MaxDD 0.139)
            S123 F0 **+0.97** (CPR 1.007, MaxDD 0.023) / F1 −0.90 (CPR 0.978, MaxDD 0.060)
  1 von 4 Folds profitabel; große Seed-Divergenz auf fold_0 (S42 −1.41 vs S123 +0.97 = der einzige
  positive Fold reproduziert NICHT über Seeds).
**Entscheidung:** ❌ ABGELEHNT — **Kill-Kriterium (a) NO-LIFT (vorab registriert) FEUERT.** SACs
  Entropie-Ziel brach den Near-Cash-Attraktor NICHT: `perf/cash_ratio_mean` **0.894 / 0.936** (beide
  > 0.85, cash_min 0.70/0.80 → ~6–11 % netto investiert, identisch zur TD3-scaled-Linie EXP-008..010),
  `perf/n_trades` (full-eval) **10**, per-asset Trades **11/10** über 200k Steps × 10 Assets,
  `actions/mean` **+0.002 / −0.034** über alle 200k Steps in BEIDEN Seeds gepinnt (gleicher konvergierter
  Attraktor wie TD3), critic_loss → **1e-9** ab ~200k = sauber konvergiert, NICHT untertrainiert.
  Trade-HHI 0.157/0.180 liegt zwar über uniform 0.10, ist aber auf nur ~10 Trades Integer-Count-Rauschen,
  KEINE Winner-Konzentration. Der S42/F1-MaxDD 0.139 ist NICHT mehr Deployment (cash dort 0.894), sondern
  EIN adverser Move (reward_skew −9.0, kurtosis 2891) auf winziger Position — gleiches Single-Adverse-
  Move-Muster wie EXP-009/010. win_rate 1.0 / profit_factor sind Geldmarktfonds-Artefakt auf ~4 Trades.
  Headline −0.61 schlägt zwar EXP-007 −0.85 (+0.24 > 0.05), aber das ist die Sharpe eines ~90 %-Cash-
  Buchs und doppelt un-gepaart gegen den Champion → Teilnahme-Bar übersteuert. **Champion bleibt EXP-007.**
**Modelle:** `models/exp_012/`
**MLflow:** Runs `seed_42` (aa7e73c1), `seed_123` (652545b3), Experiment `td3_crypto_trading`
**Learnings:** **Der Near-Cash-Attraktor überlebt über BEIDE Actor-Critic-Familien** — den
  deterministischen (TD3, EXP-001..011) UND den Maximum-Entropy-stochastischen (SAC, EXP-012), beide mit
  sauber konvergierten Critics. Beide Policy-Klassen lernen „fast alles in Cash halten" als
  risiko-/return-optimale Antwort auf eine edge-lose Reward-Landschaft mit positiven Kosten. Damit ist
  der **POLICY/ALGORITHMUS-BRANCH ERSCHÖPFT.** Die akkumulierte Evidenz über 12 Experimente (win_rate ~0.5
  auf echten Trade-Counts, Timing Münzwurf, HHI ~0.10 auf jeder Variante) ist ein publizierbares
  Negativ-Resultat: **kein handelbarer Direktions-Edge in diesem 10-Asset-Universum AM 5m-Timeframe mit
  diesen Features** — NICHT (bewiesen) bei jedem Horizont. **Verzweigung → ESKALATION AN USER (strategische
  Gabel, bricht Paarung):** (A) DATEN-Achse: Timeframe 5m → 1h (Pipeline-Re-Run, NEUER Fingerprint,
  beendet Paarung mit EXP-001..012; billig auf TD3 ~80min; Config bereit: `exp_013_data_axis_1h.yml`);
  (B) formaler No-Edge-Schluss für das 5m-Universum (RL-Suche stoppen). **Strategist-Empfehlung: A zuerst.**

---

## EXP-011 — 2026-06-12 — Breitere Netzarchitektur (net_arch pi/qf [256,128] → [400,300])
**Hypothese:** EXP-010 zeigte: die Observation ist nicht der Bottleneck (Cross-Sectional-Momentum
  erzeugt auf keinem Horizont Winner-Konzentration). Eskalation in den Policy/Algorithmus-Branch,
  cheapest-first (pre-registriert in EXP-010-Kill-(a)): testet, ob der Near-Cash-Kollaps ein
  Kapazitäts-Bottleneck ist — ob ein [256,128]-MLP schlicht zu klein ist, die 13-Feature-pro-Asset-
  Observation in eine selektive Allokation zu mappen. [400,300] = kanonische größere TD3/DDPG-Breite
  (Fujimoto et al. 2018), ~2× Parameter, reine Config, kein Code. Erwartung: cash_ratio_mean < 0.85,
  Trade-HHI ≫ 0.10, positive Sharpe über BEIDE Seeds.
**Änderung:** td3.net_arch.pi/qf: [256,128] → [400,300]   (Basis: EXP-010 `scaled` + Momentum-AN)
**Basis:** Commit 4579be1, data_fingerprint `ec2e07548f555da2` (SAUBER mit EXP-010 gepaart auf jeder
  Achse außer der Kapazität — beste Paarung der Struktur-Kette. Caveats wie EXP-008..010: Budget 200k
  vs Champion-Headline 400k (EXP-007), `scaled`-Execution → Headline-Vergleich indikativ).
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **+1.46** (std 0.47) | MaxDD **0.041** | CPR 0.991
  Pro Fold: S42 F0 **+6.48** (CPR 1.019, MaxDD 0.007) / F1 **−2.63** (CPR 0.937, MaxDD 0.134)
            S123 F0 +0.91 (CPR 1.005, MaxDD 0.014) / F1 +1.08 (CPR 1.003, MaxDD 0.009)
  3 von 4 Folds profitabel; die fold_0-Seed-Konsistenz von EXP-010 ist WEG (S42/F0 +6.48 vs
  S123/F0 +0.91) — der breitere, weiter edge-lose Actor landet pro Seed in anderen Basins.
**Entscheidung:** ❌ ABGELEHNT — **Kill-Kriterium (a) NO-LIFT (vorab registriert) FEUERT** (mit Kill-(b)-
  Churn-no-Edge-Geruch auf S42/F1). Die verdoppelte Actor-Kapazität hob den Agenten NICHT vom Near-Cash-
  Attraktor: `actions/mean` über alle 200k Steps in BEIDEN Seeds bei [−0.03, +0.01] gepinnt (identischer
  konvergierter Attraktor wie EXP-008/009/010), critic_loss → 0 ab ~105k = konvergiert. **Per-Asset-
  Trade-HHI 0.105 / 0.122** (uniform, identisch zu EXP-006/010). `n_trades` 18 / 28 (fold_1), total_cost
  27 / 42. Der +1.46 ist erneut ein Geldmarktfonds-Artefakt (profit_factor 9.6 / 412 auf 18–28 Trades
  eines ~90%-Cash-Buchs). **EINZIGES Lebenszeichen:** S42/F1 senkte cash_ratio_mean auf 0.795 (< 0.85 =
  mehr Kapitaleinsatz) — aber in einen VERLUST mit 13× größerem Drawdown (−6.3%, MaxDD 0.134) auf
  uniformem Buch → Kapazität kaufte Deployment, NICHT Edge. **Champion bleibt EXP-007.**
**Modelle:** `models/exp_011/`
**MLflow:** Runs `seed_42` (a5254825), `seed_123` (5de011d5), Experiment `td3_crypto_trading`
**Learnings:** **Repräsentations-Kapazität ist NICHT der Bottleneck.** Damit ist die Policy/
  Repräsentations-Achse ERSCHÖPFT, neben Kosten (EXP-002..005), Trainingslänge (EXP-007), Cash-Mechanik
  (EXP-008), Penalty (EXP-009) und Signal/Observation (EXP-006/010). Die akkumulierte Evidenz über 11
  Experimente (win_rate ~0.5, Timing Münzwurf, HHI ~0.10 auf JEDER Variante, überall sauber konvergierte
  Critics) deutet stark auf **keinen handelbaren Direktions-Edge in diesem 10-Asset-5m-Universum mit
  diesen Features** — nicht auf einen fehlenden Hyperparameter. **Verzweigung → ESKALATION AN USER:**
  (A) SAC-Swap (pre-registriert, ~25 Zeilen Code + 1 Screening-Lauf, letzter ungetesteter Algorithmus-
  Familien-Hebel; Config bereit: `exp_012_sac_swap.yml`); (B) formaler No-Edge-Schluss (RL-Suche stoppen);
  (C) Daten-Achse: gröberer Timeframe (1h) — Pipeline-Re-Run + neuer Fingerprint, bricht Paarung mit
  allen 11 Experimenten. **Strategist-Empfehlung: A zuerst** (billig, liefert die entscheidende
  Cross-Familien-Diagnose vor dem teuren Ledger-Reset von C).

---

## EXP-010 — 2026-06-12 — Cross-Sectional-Momentum AN, Trend-Horizont (momentum_window 12 → 48, ~4h)
**Hypothese:** EXP-009 zeigte: der Turnover-Penalty war NICHT bindend, die edge-lose Reward-
  Landschaft ist es — bei fehlendem Direktions-Edge plus Kosten IST Cash-Halten optimal. Eskalation
  in den Signal/Observation-Branch (pre-registriert): Cross-Sectional-Momentum AN, aber auf
  TREND-Horizont (`momentum_window 48` = ~4h statt 1h-Microstructure-Noise von EXP-006), auf der
  `scaled`-Basis wo Cash eine echte Option ist. Erwartung: der Actor lenkt Kapital SELEKTIV auf
  Cross-Sectional-Gewinner (cash_ratio < 0.85, Trade-HHI ≫ 0.10), positive Sharpe über BEIDE Seeds.
**Änderung:** environment.cross_sectional_momentum: false → true (momentum_window 48)   (Basis: EXP-009 `scaled`)
**Basis:** Commit 653f84b, data_fingerprint `ec2e07548f555da2` (STRUKTURELL/Code, frischer Baseline
  auf der Observation-Achse, Obs-Dim +n_assets — NICHT mit EXP-001..007 auf der Execution-Achse und
  NICHT mit EXP-009 auf der Observation-Achse gepaart. Caveats: Budget 200k vs Champion-Headline 400k
  (EXP-007), `scaled`-Execution → Headline-Vergleich nur indikativ).
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **+2.92** (std 1.47) | MaxDD **0.027** | CPR 1.010
  Pro Fold: S42 F0 **+6.34** (CPR 1.021, MaxDD 0.008) / F1 +2.45 (CPR 1.012, MaxDD 0.016)
            S123 F0 +5.58 (CPR 1.039, MaxDD 0.015) / F1 **−2.67** (CPR 0.969, MaxDD 0.069)
  3 von 4 Folds profitabel; fold_0 endlich seed-KONSISTENT (+6.34/+5.58), fold_1-Instabilität bleibt
  (S42 +2.45 vs S123 −2.67 — gleiches Muster wie EXP-009).
**Entscheidung:** ❌ ABGELEHNT — **Kill-Kriterium (a) NO-LIFT (vorab registriert) FEUERT.** Der Trend-
  Horizont-Signal hob den Agenten NICHT vom Near-Cash-Floor: `perf/cash_ratio_mean` **0.904 / 0.887**
  (beide > 0.85), `perf/n_trades` **13 / 12** (einstellig-zweistellig wie EXP-009 12/9), **per-asset
  Trade-HHI 0.112 / 0.117** (uniform 1/10 = 0.100 → KEINE Winner-Konzentration), total_cost 39/38 vs
  Champion-Kette ~960. `actions/mean` oszilliert in [−0.05, +0.06] über alle 200k Steps in BEIDEN Seeds
  (identischer konvergierter Near-Cash-Attraktor wie EXP-008/009), critic_loss → 0 ab ~105k = konvergiert,
  NICHT untertrainiert. Der +2.92 ist erneut ein Geldmarktfonds-Artefakt: profit_factor 18.9/26.6 und
  win_rate 0.83/0.85 auf nur 12–13 Trades; der S123/F1 −2.67 ist EIN adverser Move (reward_skew −40.05,
  kurtosis 2884) auf einer winzigen, undiversifizierten Position — gleiches Single-Adverse-Move-Muster
  wie EXP-009 (−3.46). DIREKTVERGLEICH zu EXP-006 (Momentum @ window 12, renormalize-Basis): HHI dort
  0.109 / 0.112 — DIESELBE uniforme Streuung. Über BEIDE Momentum-Varianten (12-bar/renorm UND
  48-bar/scaled) null Winner-Konzentration. Feature ist korrekt verdrahtet (Params geloggt csm=True
  window=48, Obs-Vektor wird erweitert; der Actor nutzt es schlicht nicht) → **die OBSERVATION ist
  nicht der Bottleneck. Champion bleibt EXP-007. `scaled` + Momentum-AN als Basis behalten.**
**Modelle:** `models/exp_010/`
**MLflow:** Runs `seed_42` (89ff90d1), `seed_123` (d956c53c), Experiment `td3_crypto_trading`
**Learnings:** **Cross-Sectional-Momentum erzeugt auf KEINEM Horizont (1h/renorm, 4h/scaled)
  Winner-Konzentration** — die Trade-Verteilung bleibt uniform (HHI ~0.11), der Direktions-Edge bleibt
  Münzwurf. Damit ist die Observation/Signal-Achse für dieses Feature erschöpft: das Problem ist nicht,
  dass der Agent das Ranking nicht SIEHT, sondern dass er keinen GRUND hat, ihm zu folgen (Signal
  nicht prädiktiv ODER Policy kann es nicht ausdrücken). **Verzweigung → EXP-011 (pre-registriert in der
  EXP-010-Kill-(a)): Policy/Algorithmus-Branch, cheapest-first.** net_arch [256,128] → [400,300] (reine
  Config, kein Code; sauber mit EXP-010 gepaart) testet, ob die Near-Cash-Kollaps ein Kapazitäts-
  Bottleneck ist. Feuert auch dort Kill (a) → Algorithmus-Swap SAC (Code nötig: `create_sac_agent` +
  `CustomSAC`-Subklasse) ODER formaler Schluss, dass dieses 10-Asset-5m-Universum mit diesen Features
  keinen handelbaren Edge trägt. Anmerkung: `perf/*` weiter ohne `step=` → Endwert = fold_1-Snapshot.

---

## EXP-009 — 2026-06-12 — Turnover-Penalty gesenkt (turnover_penalty_coef 0.004 → 0.001)
**Hypothese:** EXP-008 zeigte: unter `scaled` geht der edge-lose Agent in ~all-cash, weil der
  aggressiv getunte Turnover-Penalty (optimiert als Cash noch UNMÖGLICH war) jetzt die Inaktivität
  belohnt. Senkung 0.004 → 0.001 (4×) soll den Teilnahme-Anreiz wiederherstellen (0.001 > 0, Kosten
  bleiben im Gradienten sichtbar) und die entscheidende Diagnose liefern: ist `scaled` ein echter
  Edge oder nur eine teure Art, Cash zu halten? Zweiseitiges Kill-Kriterium vorab registriert.
**Änderung:** environment.turnover_penalty_coef: 0.004 → 0.001   (Basis: EXP-008 `scaled`)
**Basis:** Commit 052c466, data_fingerprint `ec2e07548f555da2` (STRUKTURELL/Code, `scaled`-Linie wie
  EXP-008 — NICHT auf der Execution-Achse mit EXP-001..007 gepaart. Vergleichs-Caveats wie EXP-008:
  Budget 200k vs Champion-Headline 400k (EXP-007) UND geänderte Execution-Semantik (scaled vs
  renormalize) → Headline-Vergleich nur indikativ).
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **+2.84** (std 2.89) | MaxDD **0.024** | CPR 1.002
  Pro Fold: S42 F0 **+8.89** (CPR 1.030, MaxDD 0.006) / F1 +2.58 (CPR 1.006, MaxDD 0.008)
            S123 F0 +3.35 (CPR 1.011, MaxDD 0.012) / F1 **−3.46** (CPR 0.961, MaxDD 0.072)
  3 von 4 Folds profitabel; fold_1-Seed-Instabilität bleibt (S42 +2.58 vs S123 −3.46).
**Entscheidung:** ❌ ABGELEHNT — **Kill-Kriterium (b) STILL-CASH (vorab registriert) FEUERT.** Der
  4×-Penalty-Schnitt hob die Teilnahme nur um Rausch-Beträge: `perf/cash_ratio_mean` 0.969/0.876
  (EXP-008) → **0.932/0.893** (EXP-009, beide > 0.85), `perf/n_trades` 10/10 → **12/9** (einstellig),
  netto investiert ~3–12% → ~7–11%. `actions/mean` blieb über alle 200k Steps in BEIDEN Seeds bei
  ~0.00 gepinnt (gleicher konvergierter Attraktor wie EXP-008), `actions/std` ~0.918, critic_loss → 0
  = konvergiert, NICHT untertrainiert. Der +2.84 ist ein Geldmarktfonds-Artefakt: profit_factor 16–140
  und win_rate 0.75–0.78 auf nur 9–12 Trades; der S123/F1 −3.46 ist EIN adverser Move (total_return
  −3.9%, reward_skew −40) auf einer winzigen, undiversifizierten Position. total_cost 36/29 vs Champion-
  Kette ~960 → Kill (a) RE-CHURN feuert NICHT. Headline +2.84 schlägt zwar numerisch −0.80, aber das
  Kill-Kriterium übersteuert (Artefakt + doppelte Vergleichs-Caveats). **Champion bleibt EXP-007.
  `scaled` wird als Struktur-Basis beibehalten.**
**Modelle:** `models/exp_009/`
**MLflow:** Runs `seed_42` (d9ae7743), `seed_123` (d61436e1), Experiment `td3_crypto_trading`
**Learnings:** **Der Turnover-Penalty war NIE der bindende Constraint auf Teilnahme — die edge-lose
  Reward-Landschaft ist es.** Bei fehlendem Direktions-Edge (win_rate ~0.5, Timing Münzwurf, bestätigt
  EXP-001..006) plus positiven Kosten IST Cash-Halten die Sharpe-/Return-maximale Politik, unabhängig
  vom Penalty-Niveau. Der `scaled`-Mechanik fehlt nicht der Anreiz, sondern dem Agenten der GRUND zu
  investieren. Damit ist die Kosten-Achse endgültig erschöpft (EXP-002..005 + EXP-008/009). **Verzweigung
  → EXP-010 (pre-registriert): Signal/Observation-Branch.** Cross-Sectional-Momentum ON, aber mit
  TREND-Horizont `momentum_window 12 → 48` (~4h statt 1h-Microstructure-Noise von EXP-006), auf der
  `scaled`-Basis wo Cash eine echte Option ist — gibt dem Actor das fehlende Relative-Stärke-Ranking,
  um Kapital SELEKTIV auf Cross-Sectional-Gewinner zu lenken. Wenn auch das nicht teilnimmt → Policy/
  Algorithmus-Branch (net_arch / SAC). Anmerkung: `perf/*` weiter ohne `step=` → Endwert = fold_1-
  Snapshot (fold_0 nicht direkt beobachtbar).

---

## EXP-008 — 2026-06-12 — Cash-fähige Allokation (allocation_mode "renormalize" → "scaled")
**Hypothese:** EXP-007 fand die Wurzelursache: `compute_target_holdings` renormalisiert über
  `sum` → in JEDEM Experiment war das Portfolio zwangs-zu-~100%-investiert, ~0% Cash. Ein fester
  Divisor (`n_assets · max_position`) gibt dem Agenten einen lernbaren Cash-Ausweg: niedrigere
  Actions de-risken im Down-Markt wirklich (nutzbarer Gradient), statt von der Renorm aufgehoben
  zu werden. Erwartung: fold_1 (Down-Markt) MaxDD/Sharpe besser, fold_0 unverändert.
**Änderung:** environment.allocation_mode: "renormalize" → "scaled"   (Basis: Champion EXP-005)
**Basis:** Commit 1db4664, data_fingerprint `ec2e07548f555da2` (STRUKTURELL/Code, wie EXP-006 —
  NICHT auf der Execution-Achse mit EXP-001..007 gepaart; frischer Struktur-Baseline für den
  Cash-Branch. ACHTUNG zwei Vergleichs-Caveats: Budget 200k vs Champion-Headline 400k (EXP-007),
  UND geänderte Execution-Semantik → Headline-Vergleich nur indikativ).
**Budget:** 200k Steps × 2 Seeds × 2 Folds
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **+1.42** (std 0.61) | MaxDD **0.026** | CPR 1.000
  — ERSTES POSITIVES MEAN DER KETTE und ERSTER positiver Down-Markt-Fold (S42/F1 +1.45).
  Pro Fold: S42 F0 +0.17 (CPR 1.001, MaxDD 0.02) / F1 +1.45 (CPR 1.005, MaxDD 0.01)
            S123 F0 +5.82 (CPR 1.017, MaxDD 0.007) / F1 −1.75 (CPR 0.976, MaxDD 0.065)
  Extreme Cross-Seed-Varianz auf beiden Folds; F1-Vorzeichen kippt (S42 +1.45 vs S123 −1.75).
**Entscheidung:** ❌ ABGELEHNT — **Kill-Kriterium (im EXP-008-Vorschlag vorab registriert) FEUERT:
  Agent ist quasi all-cash.** `perf/cash_ratio_mean` **0.969 / 0.876** (88–97 % CASH im Mittel),
  `cash_ratio_min` 0.30 / 0.60, `perf/n_trades` **10** über ~17.8k Steps × 10 Assets (Champion-
  Kette ~250–300/Fold), `actions/mean` **−0.006 / −0.018** (Champion ~0.84) → netto ~5 % investiert.
  total_cost **30 / 32** vs Champion ~960 (−97 %). Der +1.42 ist die Sharpe eines Geldmarktfonds
  (top Risk-adjusted-Ratio auf ~null Kapitaleinsatz), KEIN Handels-Edge. Champion bleibt EXP-007.
  **ABER:** `scaled`-Mechanik wird als Execution-Fix BEIBEHALTEN (Basis für EXP-009) — sie hat
  ihren Zweck erfüllt (fold_1 MaxDD ~0.41 → 0.01–0.06, Down-Markt-Hilflosigkeit weg), nur ins
  Nicht-Handeln über-korrigiert.
**Modelle:** `models/exp_008/`
**MLflow:** Runs `seed_42` (dca5d4f3), `seed_123` (b492230c), Experiment `td3_crypto_trading`
**Learnings — DREI BEFUNDE:**
  **(1) `scaled` funktioniert mechanisch — und genau deshalb geht der Agent in Cash.** Sobald
  Cash eine freie, lernbare Option ist, IST bei fehlendem Edge (win_rate ~0.5, Timing ~Münzwurf,
  bestätigt EXP-001..006) plus positiven Kosten die Sharpe-/Return-maximale Politik = NICHT handeln.
  Der Agent hat das korrekt gelernt. fold_1 MaxDD-Kollaps (0.41 → 0.01–0.06) belegt: die Cash-
  Fähigkeit heilt zwar die Down-Markt-Hilflosigkeit, aber durch Nicht-Teilnahme, nicht durch Edge.
  **(2) Stabiler Attraktor, KEIN Untertraining.** `actions/mean` fällt bis Step 10k auf ~0.00 und
  bleibt dort über alle 200k Steps in BEIDEN Seeds flach (Gegensatz zu EXP-006/007, wo der Mean
  weiterstieg). `actions/std` ~0.92 (Einzel-Actions spannen [−1,1], mitteln sich aber zu ~0 →
  scaled → ~5 % netto). critic_loss sauber (~1e-6), actor_loss flach. Die Politik IST konvergiert
  — auf „fast nichts tun". Mehr Training würde das NICHT ändern.
  **(3) Headline-Sharpe ist seed-VERRAUSCHT, Verhalten seed-STABIL.** Die std 0.61 kommt fast nur
  aus fold_0 (S123/F0 +5.82 = annualisierte Sharpe einer winzigen, glücklichen ~Cash-Position auf
  wenigen Trades) und dem F1-Sign-Flip — beides Artefakt von ~10 Trades/Episode, nicht von echter
  Varianz im Verhalten. Der Cash-Anteil ist über Seeds stabil.
  **Bindender Hebel jetzt:** der Turnover-Penalty (0.004), getunt als Cash noch UNMÖGLICH war,
  belohnt jetzt aktiv die Inaktivität → er ist der bindende Constraint auf Teilnahme.
  **Verzweigung → EXP-009:** turnover_penalty_coef 0.004 → 0.001 auf der `scaled`-Basis. Erzwingt
  Teilnahme und liefert die entscheidende Diagnose, die der +1.42 nicht geben kann: ist `scaled`
  ein echter Edge oder nur eine teure Art, Cash zu halten? Zweiseitiges Kill-Kriterium: (a) Re-Churn
  (n_trades ~250+, Kosten ~960, Sharpe negativ) → Penalty war tragend, edge-loser Korb kann nicht
  profitabel handeln → REJECT; (b) Immer-noch-Cash (cash_ratio_mean > 0.85, n_trades einstellig) →
  Penalty war NICHT bindend, das edge-lose Reward-Landschaft ist es → REJECT + Eskalation in den
  Signal/Observation-Branch (Agent hat keinen GRUND zu investieren, weil kein Edge zum Ausdrücken).

---

## EXP-007 — 2026-06-12 — Längeres Training (total_timesteps 200k → 400k)
**Hypothese:** EXP-006 diagnostizierte UNTERTRAINING: bei 200k wanderte der Actor noch
  (actions/mean steigend, actions/std fallend). Isoliere Variable (A) auf der Champion-
  Observation (Momentum AUS): verdoppeltes Training konvergiert die Politik und hebt Sharpe.
**Änderung:** td3.total_timesteps: 200000 → 400000   (Basis: EXP-005, Momentum aus)
**Basis:** Commit 9143dea, data_fingerprint `ec2e07548f555da2` (ACHTUNG: total_timesteps IST
  hier die experimentelle Variable → NICHT mit EXP-001..006 auf der Trainingslängen-Achse
  gepaart; Vergleich nur auf Headline-Metrik gegen Champion EXP-005, gleiche Seeds/Folds/Daten).
**Budget:** 400k Steps × 2 Seeds × 2 Folds (~2× Wall-Clock, ~2h30)
**Status:** ✅ ABGESCHLOSSEN
**Ergebnis:** mean val Sharpe **−0.85** (std 0.15) | MaxDD **0.284** | CPR 0.90
  Pro Fold: S42 F0 **+0.47** (CPR 1.025) / F1 −2.48 · S123 F0 **+0.58** (CPR 1.030) / F1 −1.98
  Gepaart 3:1 (S42/F0 +0.67, S123/F0 +1.36, S123/F1 +0.29 besser; **S42/F1 −1.27 schlechter**)
  **ERSTE PROFITABLE VAL-FOLDS DER GESAMTEN KETTE** (beide fold_0).
**Entscheidung:** ⭐ ADOPTIERT → neuer Champion. Delta +0.27 > 0.05-Schwelle, MaxDD besser
  (0.284 vs 0.355), kein Crash/NaN. **ABER:** Adoption ruht allein auf den zwei fold_0-Profiten
  (Up-Markt-Rückenwind auf Long-Basket = evtl. Regime-Glück, kein Edge); S42/F1 regressiert stark.
  **400k wird NICHT als Screening-Budget übernommen** (Begründung unten) — EXP-008 zurück auf 200k.
**Modelle:** `models/exp_007/`
**MLflow:** Runs `seed_42` (7d16f40c), `seed_123` (3bfc4d90), Experiment `td3_crypto_trading`
**Learnings — DREI BEFUNDE:**
  **(1) Actor bei 400k NICHT konvergiert.** actions/mean steigt über die letzten 50k beider
  Seeds weiter (s123 0.859→0.874), actions/std fällt weiter (0.423→0.391). Über 200k→400k:
  actions/mean +0.25 (0.59→0.84-0.87), actions/std −0.34 (0.73→0.39-0.45). critic_loss sauber
  (~1e-6). **Längeres Training konvergiert die Politik NICHT — es härtet nur den Long-Bias.**
  **(2) „Hold everything long" wird stärker, nicht klüger.** Alle 10 Per-Asset-Action-Means
  0.64-0.93. Die Extra-Steps verschieben nur die Tilt innerhalb eines Dauer-Long-Korbs.
  **(3) WURZELURSACHE GEFUNDEN — der Agent KANN strukturell kein Cash halten.**
  `compute_target_holdings` renormalisiert die Gewichte durch `sum`, sobald `sum > 1.0`. Bei
  10 Assets ist `sum > 1.0`, sobald actions/mean > **−0.80** — wo der Agent NIE war (selbst
  EXP-001 ~0.48 → sum 7.4). **→ In JEDEM Experiment seit EXP-001 ist das Portfolio auf ~100 %
  investiert / ~0 % Cash zwangs-renormalisiert.** Das ist die mechanische Wurzel der „Down-Markt-
  Hilflosigkeit": ein edge-loser Long-Korb wird durch jeden fold_1-Selloff gezogen, und mehr
  Training (EXP-007) macht es SCHLECHTER (Long-Bias härter → S42/F1 −1.21→−2.48). Einzelne
  Actions zu senken bringt nichts — die Renorm hebt es auf. **Verzweigung → EXP-008:** der
  EXP-006-Gate („400k hilft → Momentum AN + 400k") ist durch diese Evidenz überholt — 400k
  konvergiert nicht und Momentum ändert das Handelsverhalten nachweislich nicht. Der bindende
  Hebel ist die **Cash-Fähigkeit**: `allocation_mode "renormalize" → "scaled"` (fester Divisor
  n_assets·max_position statt Summe) gibt dem Agenten einen lernbaren Cash-Ausweg im Down-Markt.
  Anmerkung: `perf/*`-Metriken weiter ohne `step=` → Endwert = fold_1-Snapshot (fold_0 nicht
  direkt beobachtbar); EXP-008 soll `cash_ratio` in die Eval-Diagnostik aufnehmen.

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
