# FINDINGS — review campaign 2026-09-22 (DRAFT, updated as runs finish)

Scope: server work specified in `SERVER_HANDOFF.md` (review of September 2026). Branch `review-2026-09-22`,
results root `results/review_2026_09_22/`. All old results under `results/event_bench/` are untouched.
Paper-facing name: TSD (`--model faithful` remains the executable selector). Facts below distinguish
**supported**, **null/negative**, **outstanding**, and **manuscript changes**. Nothing planned is
described as completed; every table cell traces to a `results.csv` / CSV produced by the scripts in
`exp/review/` at the commit recorded in `CODE_COMMIT.txt`.

## 0. P0 — score-validity audit (query_validity.csv)

Instrumentation (`exp/temporal_benchmark_utils.py::QueryValidityAudit`): for every val/test query, BEFORE
numerical substitution and before the pad mask, the positive's raw validity, the counts of NaN / +inf / -inf
negative logits, the candidate count, the pad-mask count, the transformed-score count and the action taken;
per-split exact denominators; a bounded failing-query list with replay ids (global edge id, split, snapshot,
timestamp, relation); per-snapshot finite checks of memory / spatial state and of the REC caches.
Three reciprocal ranks are accumulated: TGB semantics on the raw scores, the guarded scores the harness
reports, and a conservative diagnostic that assigns reciprocal rank 0 to any affected query (full denominator).
The audit runs in EVERY new evaluation and in the checkpoint replays of the retained headline runs
(`audit/<run>/`); parity of the three metrics is recorded per split (`query_audit_parity`).

Status: see REVIEW_SUMMARY.md "Score-validity audit" and "Checkpoint-replay audits" (`replay_minus_retained_test`
is the replayed test MRR minus the metric stored in the retained run). Completed replays so far: every one has
**zero** affected queries (no non-finite positive or negative logit before substitution; no non-finite state or
REC entry at any snapshot) and the three metrics agree exactly (parity). Replay reproducibility: tgbl-wiki
seeds 43/46/47 and thgl-software 43/46 reproduce the retained test MRR to 1e-6 or better; tkgl-smallpedia
seed 47 reproduces it to 8e-4 (retained 0.61118; replay 1 0.61041; an independent replay 2 0.61031), i.e.
the smallpedia evaluation is non-deterministic at the 1e-4 to 1e-3 level (its relation-aggregate input uses
float `index_add_` on CUDA, whose summation order is not deterministic), and the retained value lies at the
edge of that band. This is a numerical-reproducibility caveat, not an invalid-score finding.

Historical training skip frequency for the retained event runs remains **unknown** (no counter existed);
new runs serialise `train_steps`, `train_nonfinite_loss`, `train_nonfinite_grad`, `train_clipped`,
`train_skipped_steps` (numerical_events.csv).

## 1. P0 — provenance and node-property observation schedules

`provenance/manifest.json`: repository SHA + dirty-diff hash, Python/PyTorch/PyG/CUDA/py-tgb versions and
`pip freeze`, installed TGB source hashes (loaders, preprocessing, evaluators, negative sampler, label
generation scripts), raw/processed dataset file SHA-256 (all nine datasets), split-mask hashes, negative-set
file hashes, label-timestamp hashes, hardware.

Installed package: py-tgb 2.2.0. Its `nodeproppred/dataset.py` matches the bundle's upstream reference except
one print statement; `utils/pre_process.py` differs from upstream in the thg edge-type-id handling (not used
by the node tasks).

### tgbn-trade (`provenance/trace_trade_w1/`, TGB-exact yearly windows; `trace_trade_paper/`, 3-year windows + 2048-edge prefix)
- **Label semantics (reconstructed from raw edges, all 30 label years):** label(Y, u) equals the row-normalised
  vector of year-Y edge weights of u (mean max-abs error 5e-9, exact for 100% of 6543 node-years); the
  next-year and previous-year windows do not match (errors 0.19). The stored label year IS the generating year.
- **Observation frontier:** under both our runner and the official TGN example (batch 200, deferred cursor),
  every label Y is scored after ALL year-Y edges have been ingested and before any year-(Y+1) edge
  (`ingested_t_eq_label` > 0, `ingested_t_gt_label` = 0 for every scored label, both runners). Same-period
  exposure is a property of the benchmark's release schedule, not of our runner.
- **Split assignment differs at the boundaries:** ours scores val = {2010..2013}, test = {2014..2016};
  the official example scores val = {2009..2012}, test = {2013..2015} (each split's last pending label is
  deferred to the next split) and never scores 2016 (no later batch). Train: ours 1987..2009, official 1987..2008.
- **Diagnostic predictors (NDCG@10, exact scored labels):** copy from currently available edges = **1.000**
  under BOTH schedules (val and test); last released label 0.867 (ours val) / 0.845 (ours test); forecast from
  the previous completed period 0.867 / 0.845; mean of previous three periods 0.865 / 0.842. Official-schedule
  values: 0.860 / 0.854 (last label), 1.000 (copy).
- Consequence for reporting: tgbn-trade scores are state read-out scores under a same-period schedule shared
  with the reference example; the label SETS per split differ by one boundary year from the official example,
  so leaderboard ranks are reported with that caveat (rank withheld as "matched-protocol" until label sets
  are aligned; a runner option that reproduces the deferred-cursor assignment is a one-line change but was not
  used for any reported number).

### tgbn-genre (`provenance/trace_genre_daily/`, daily windows, uncapped)
- **Label sets identical** between our runner and the official example in every split (1250 / 163 / 166 label
  timestamps; identical label hashes; nothing left unscored).
- **Frontier:** before scoring the label of day D both runners have ingested edges up to shortly after D
  (median 1.0 h for ours, 0.2 h for the official example, i.e. part of day D's first hour); 80% of labels have
  identical `ingested_t_gt_label` counts, the rest differ by at most 5242 edges out of ~10^7.
- **Label semantics:** the label of day D is best matched by the aggregate of the NEXT seven days [D, D+7d)
  (NDCG@10 of a window-copy vs the label 0.66 by edge weight; previous seven days 0.48; single next day 0.31),
  but no tested window reproduces the label values exactly (mean max-abs error >= 0.31), so the exact target
  transformation is not recoverable from the edge weights available to the runner — the task is a genuine
  forecast: **copy from available edges scores 0.014**; last label 0.293 / 0.300 (val / test); previous
  completed day 0.294 / 0.301; mean of previous three days 0.402 / 0.410 (leaderboard: Persistent Forecast
  0.357, Moving Average 0.509; ours 0.448 +/- 0.003).

## 2. Clocks, saturation, batching, query-time response

**Fixed-state future-query probe (`clock/probe_wiki_s43/`, retained tgbl-wiki seed-43 checkpoint):** the
state after the training replay was cloned and the identical (source, relation, candidate) tuples of the
first 20 validation batches (700 queries) were scored at their original query time and at +1 h, +1 day,
+30 days without ingestion.

| shift | neural score changed | mean |delta| neural | mean REC bonus change (positives) | MRR | MRR change |
|---|---|---|---|---|---|---|
| 0 | 0 of 700 | 0 | 0 | 0.6729 | 0 |
| +1 h | 0 of 700 | 0 | -0.065 | 0.6686 | -0.0043 |
| +1 day | 0 of 700 | 0 | -0.381 | 0.6666 | -0.0063 |
| +30 days | 0 of 700 | 0 | -1.128 | 0.6632 | -0.0097 |

**Supported:** the non-REC score is exactly invariant to the query time given fixed state and inputs; only the
REC recency feature responds (83% of these positives have a seen key). A query-time projection of the backbone
does not exist in the evaluated model (consistent with the implementation audit); it was not implemented.

**Fixed-membership gap intervention (`clock/gap_wiki_s43/gap_intervention.csv`, same checkpoint):** the
physical gap supplied to the step selector was scaled by 0.25, 0.5, 1, 2, 4 (through the non-learned
`delta_scale`), with batches, events, REC recency and query times unchanged, and the first 30k validation
edges re-scored after a full training replay.

| gap factor | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| val MRR | 0.7516 | 0.7518 | 0.7521 | 0.7548 | 0.7553 |
| val Hits@10 | 0.8718 | 0.8717 | 0.8726 | 0.8718 | 0.8721 |

**Null:** the trained tgbl-wiki model is insensitive to the magnitude of the gap it is given (range 0.004 MRR
over a 16x span, monotone in the direction of longer apparent gaps), i.e. the timing channel contributes
little on this benchmark at inference; this is consistent with the earlier no-gap ablation (-0.007) being a
training-time effect. The saturation diagnostics of the replayed wiki checkpoints show the step cap active
for about 5-6% of endpoint updates and under 1% of closure updates (closure updates outnumber endpoint
updates 7:1), so the cap is not the binding constraint on wiki.

**Saturation of the step selector (`clock_diagnostics.csv`; exact per-update counters over the full training
replay, validation and test):**

| run | lr | split | activity | updates | fraction capped at 0.25 | mean uncapped step | mean capped step |
|---|---|---|---|---|---|---|---|
| retained wiki seed 47 (replay) | 3e-4 | train replay | endpoint | 127,665 | 0.056 | 0.18 | 0.041 |
| retained wiki seed 47 (replay) | 3e-4 | train replay | closure | 894,276 | 0.006 | 0.022 | 0.018 |
| new wiki TSD seed 43 (global clock) | 1e-3 | train replay | endpoint | 127,665 | **0.9999** | 284 | 0.25 |
| new wiki TSD seed 43 (global clock) | 1e-3 | train replay | closure | 894,276 | **0.9999** | 189 | 0.25 |
| new wiki TSD seed 43 (global clock) | 1e-3 | val / test | all | 367,557 | 1.000 | 190-282 | 0.25 |
| new wiki TSD seed 43 (node-interaction clock) | 1e-3 | train replay | endpoint | 127,665 | 0.973 | 1756 | 0.244 |

**Saturation in the retained headline checkpoints (training-replay updates; `audit/*/clock_diagnostics.csv`):**

| dataset (lr) | seed | endpoint updates capped | closure updates capped | mean uncapped step (endpoint) |
|---|---|---|---|---|
| tgbl-wiki (3e-4) | 43 / 46 / 47 | 10% / 1% / 6% | 32% / 0% / 1% | 0.11 / 0.05 / 0.18 |
| thgl-software (1e-3) | 43 / 46 / 47 | 26% / 25% / 10% | 0% / 0% / 0% | 0.17 / 0.18 / 0.10 |
| thgl-forum (1e-3) | 43 / 46 / 47 | 6% / **98%** / **100%** | 0% / 2% / 0% | 0.42 / 2.9 / 4.6 |
| tkgl-polecat (3e-3) | 43 / 46 / 47 | **100%** / 16% / **100%** | **100%** / 0% / **100%** | 6399 / 1.8 / 3139 |
| tkgl-smallpedia (1e-2) | 43 / 46 / 47 | **77%** / 1% / **62%** | 26% / 0% / **88%** | 58 / 0.03 / 23 |
| tkgl-wikidata (3e-3) | 43 / 46 / 47 | 0% / 0% / 1% | 0% / 0% / 0% | 0.01 / 0.01 / 0.10 |
| tkgl-icews (3e-3) | 43 / 46 / 47 | pending | pending | pending |

Saturation is seed- and dataset-dependent: on forum, polecat and smallpedia the selector collapses to the
cap on two of three seeds (uncapped means of 10^1 to 10^3 against a cap of 0.25), while the same seeds' test
scores are indistinguishable from the unsaturated seeds (forum 0.628/0.641/0.619; polecat 0.240/0.246/0.246;
smallpedia 0.616/0.603/0.611). The learned timing weight stays near its initial value 1.0 in every run
(0.84-1.29). Zero physical gaps are rare everywhere (< 0.2% of updates; 1.7% on wikidata endpoints).

**Finding:** at the learning rate that every arm selected on tracking validation (1e-3), the selector
pre-activation grows until **every** memory update takes the capped step: the content- and gap-dependent
timing channel is inactive in the best-scoring TSD configuration (test MRR 0.7597 at seed 43 vs 0.7357 for the
retained lr 3e-4 model whose selector was only 5.6% capped). Closure updates outnumber endpoint updates 7:1;
zero physical gaps are rare (< 0.1%). Under the node-interaction clock 5.9% of endpoint updates have a zero gap
(first observation, seen mask false) and the supplied gaps are ~10x larger on average (log10 3.6 vs 2.8 in
seconds), but the selector is still 97% capped. **Consequence for the manuscript:** claims about selective,
physically timed memory steps are not supported by the evaluated predictor at its selected learning rate; the
model behaves as a fixed-step recurrence with capped ZOH transitions.

**Clock comparison (wiki, seed 43, lr 1e-3, identical batching; `clock/`):** global batch gap 0.7597
(selector 99.99% capped), time since the node's last interaction 0.7650 (97% capped), time since the node's
last memory update 0.7611 (selector **0.06%** capped, mean uncapped step 0.0018, i.e. the opposite regime).
Differences of +0.001 to +0.005 on one seed are not evidence for either node clock; the striking fact is that
the same accuracy is reached whether the selected steps are all at the cap or all near zero, so the memory
step regime is immaterial for wiki accuracy. **Batching diagnostic (same seed, same raw splits and queries; `clock/wiki_tsd_tw*`):** batch width 300 s
gives test MRR **0.7835** (6,209 training snapshots), 600 s (the benchmark setting) 0.7597 (3,105), 1200 s
0.7322 (1,553). Halving the width gains +0.024 and doubling it loses −0.027: the observation frontier (how
recent the ingested events are when a batch is scored) and the number of processed updates are a
first-order factor on wiki, larger than any architectural contrast in the matched matrix. Cross-method
comparisons must therefore state the batch width; our benchmark numbers use 600 s. Cross-seed confirmation of the clock variants was not run (budget); they are single-seed pilots.

(All aggregates, histograms of steps and gaps by activity class and split, and the learned timing parameters
are in `clock_diagnostics.csv` and each run's `clock_*.csv` / `clock_learned_timing.json`.)

## 3. Matched comparisons (tgbl-wiki five seeds; thgl-forum five seeds)

**tgbl-wiki, primary seven-arm comparison plus core-off (`matched/`; seeds 43-47 paired; identical splits,
features, 50k-edge context, 600 s batches, state warm-up, REC typed channel, scorer, softplus loss, K = 32
negatives drawn from a seed-locked generator independent of the architecture, tracking-validation selection
on the first 8,000 validation edges, 8 epochs; learning rate locked per arm on the seed-43 tracking validation
from {3e-4, 1e-3}: every arm chose 1e-3; official negatives and evaluator for the reported test MRR):**

| arm | active params | test MRR mean ± SD | Hits@10 | Δ arm − TSD (mean ± SD; 95% t) | per-seed Δ (43..47) | signs |
|---|---|---|---|---|---|---|
| TSD (memory-conditioned incidence maps) | 444,686 | **0.7623 ± 0.0030** | 0.852 | – | 0.7575 / 0.7646 / 0.7631 / 0.7647 / 0.7616 | – |
| current-only maps | 448,846 | 0.7601 ± 0.0075 | 0.852 | −0.0022 ± 0.0085; ±0.011 | +0.008 / +0.003 / −0.001 / −0.011 / −0.011 | 2+ 3− |
| identity maps | 444,558 | 0.7630 ± 0.0019 | 0.852 | +0.0007 ± 0.0032; ±0.004 | +0.005 / +0.000 / +0.002 / −0.003 / −0.001 | 3+ 2− |
| GRU + ordinary propagation | 33,338 | 0.7640 ± 0.0060 | 0.852 | +0.0017 ± 0.0077; ±0.010 | +0.014 / +0.005 / −0.002 / −0.005 / −0.004 | 2+ 3− |
| diagonal SSM + ordinary propagation | 10,126 | 0.7616 ± 0.0060 | 0.853 | −0.0007 ± 0.0068; ±0.008 | +0.008 / +0.004 / −0.005 / −0.003 / −0.008 | 2+ 3− |
| history-conditioned edge gates | 444,815 | 0.7548 ± 0.0049 | 0.852 | −0.0075 ± 0.0054; ±0.007 | −0.005 / −0.004 / −0.008 / −0.017 / −0.003 | 0+ 5− |
| node-frame geometry | 444,750 | 0.7543 ± 0.0060 | 0.852 | −0.0080 ± 0.0084; ±0.010 | +0.007 / −0.010 / −0.015 / −0.012 / −0.010 | 1+ 4− |
| core-off + REC (head only) | 8,419 | 0.7349 ± 0.0066 | 0.845 | −0.0274 ± 0.0084; ±0.010 | −0.014 / −0.031 / −0.032 / −0.026 / −0.035 | 0+ 5− |

"Active params" excludes parameters unused by construction (the SSM input selector B holds 435,584 of TSD's
parameters). The retained lr 3e-4 TSD (0.7331 ± 0.0026 over seeds 43/46/47) is superseded by this locked
configuration; the tracking-validation gap between the two learning rates was 0.766 vs 0.734 at seed 43.

**Supported (wiki):** the recurrent core contributes (+0.027 over the head-only model, every seed). **Null:**
memory-conditioned maps do not beat current-only maps, identity maps, a GRU or a diagonal SSM with ordinary
propagation; the last two reach the same accuracy with 13x and 44x fewer active parameters. **Negative:**
node-frame geometry and history-conditioned edge gates are slightly but consistently worse than identity
transport (−0.008; 4-5 of 5 seeds). **Stratified (identical test queries, seeds 44-47; `matched/wiki_strata/`):** 89.2% of wiki test queries
are recurring (source, destination) pairs. Every arm scores ~0.85 MRR on recurring pairs and ~0.01 on novel
pairs (TSD 0.854 / 0.012; core-off 0.819 / 0.014; GRU 0.853 / 0.011; identity 0.854 / 0.010). The core's gain
over the head-only model is entirely on recurring pairs (+0.035, 4 of 4 seeds) and on the longest
source-inactivity quartile (gap > 8,162 s: +0.040, 4 of 4); on novel pairs core-off is marginally *better*
(+0.002, 3 of 4). Between cores no stratum separates TSD from GRU, identity or diagonal SSM; node-frame and
edge gates lose on recurring pairs (−0.013 and −0.009, 4 of 4). The wiki benchmark therefore rewards ranking
among recurring destinations after long inactivity, not generalisation to unseen pairs.


**thgl-forum matched matrix (complete; `forum/` plus the replay-verified retained seeds; same head, REC
channels, data, batching, negatives regime, budget and selection rule; paired per seed; `paired_contrasts.csv`):**

| REC | arm | seeds | TSD mean | arm mean | Δ arm − TSD (mean ± SD; 95% t half-width) | signs |
|---|---|---|---|---|---|---|
| on | GRU + ordinary | 5 | 0.625 | 0.642 | +0.017 ± 0.015; ±0.019 | 5+ 0− |
| on | identity maps | 5 | 0.625 | 0.640 | +0.015 ± 0.009; ±0.012 | 5+ 0− |
| on | current-only maps | 5 | 0.625 | 0.634 | +0.009 ± 0.023; ±0.028 | 4+ 1− |
| on | diagonal SSM + ordinary | 3 | 0.629 | 0.646 | +0.017 ± 0.015; ±0.037 | 3+ 0− |
| on | node-frame geometry | 3 | 0.629 | 0.635 | +0.006 ± 0.002; ±0.005 | 3+ 0− |
| on | attention gates | 3 | 0.629 | 0.627 | −0.002 ± 0.011; ±0.026 | 1+ 2− |
| on | TSD no-memory (retained) | 3 | 0.629 | 0.650 | +0.021 ± 0.014; ±0.035 | 3+ 0− |
| on | TSD no-gap (retained) | 3 | 0.629 | 0.634 | +0.004 ± 0.020; ±0.049 | 2+ 1− |
| on | core-off (head only) | 5 | 0.625 | 0.615 | −0.010 ± 0.010; ±0.012 | 0+ 5− |
| off | GRU + ordinary | 5 | 0.353 | 0.376 | +0.023 ± 0.084; ±0.104 | 2+ 3− |
| off | current-only maps | 4 | 0.354 | 0.393 | +0.038 ± 0.079; ±0.126 | 2+ 2− |
| off | identity maps | 5 | 0.353 | 0.321 | −0.032 ± 0.097; ±0.120 | 2+ 3− |
| off | diagonal SSM + ordinary | 3 | 0.381 | 0.364 | −0.016 ± 0.093; ±0.231 | 2+ 1− |
| off | node-frame geometry | 3 | 0.381 | 0.378 | −0.003 ± 0.009; ±0.021 | 2+ 1− |
| off | attention gates | 3 | 0.381 | 0.313 | −0.068 ± 0.050; ±0.125 | 0+ 3− |
| off | TSD no-memory (retained) | 2 | 0.397 | 0.396 | −0.002 ± 0.023 | 1+ 1− |
| off | TSD no-gap (retained) | 3 | 0.381 | 0.367 | −0.013 ± 0.049; ±0.123 | 1+ 2− |
| off | core-off (head only) | 5 | 0.353 | 0.226 | −0.127 ± 0.070; ±0.087 | 0+ 5− |

Per-seed TSD: REC on 0.628 / 0.622 / 0.640 / 0.641 / 0.619 (seeds 43-47); REC off 0.394 / 0.232 / 0.391 /
0.401 / 0.347.

**Supported (forum):** (i) a recurrent core adds to the head (core-off is below TSD on 5 of 5 seeds in both
REC settings: −0.010 with the head, −0.127 without); (ii) the *kind* of core and the *kind* of transport do
not matter with the REC head: a GRU or a diagonal SSM with identity transport, identity maps, current-only
maps and node-frame geometry are all at or slightly above TSD on every paired seed (+0.006 to +0.021); the
retained no-memory ablation is also above TSD. **Negative:** memory-conditioned incidence-specific maps give no
paired benefit over current-only maps, identity maps or node frames on forum. **Without the head** no control
differs from TSD consistently except core-off and attention gates (both worse); TSD itself is seed-unstable
under the 4-epoch budget (seed 44: 0.232, slow start; see below), and the node-frame control is the most
stable close match (SD of the paired difference 0.009). The earlier claim that learned restriction maps are
"the robust mechanism" on forum is withdrawn: the low identity-map scores were a trainability effect of the
SSM core without the head, not evidence for learned transport.

Seed 44 makes the trainability effect visible: TSD REC-off scores 0.232 (GRU 0.394) with zero non-finite
events and modest clipping; its tracking validation rises 0.134 → 0.135 → 0.154 → 0.246 over the fixed
4-epoch budget and is still climbing at the cut-off, whereas the GRU core is at 0.39 after epoch 1.

## 4. Synthetic history-dependent task (synthetic/)

**Generator (`exp/review/synthetic_history.py`, "cued target drift"):** 400 nodes in 8 hidden communities, 60k
events with i.i.d. exponential gaps (irregular time), fixed random node features that encode nothing; each
node's hidden target community is set by its most recent CUE event (an observable interaction with one of 8
cue nodes; p = 0.03 per event; initial targets unannounced); other events pick a destination in the target
community with a hidden popularity weight observable only through in-degree history. Val/test destinations
are novel (never paired with the source) for a preregistered ~50% of events (realised: 65% val, 62% test,
`generator_meta.json`); 32 fixed uniform negatives per val/test query; batches are 20 s windows (~20 events).
Permitted information for the oracle = events with t < the query batch's window start. Data SHA-256 and the
hidden sidecar (communities, popularity, targets, novelty) are stored with the data.

**Sanity instance (`synthetic/gen_s1/oracle_summary.csv`):** permitted-history oracle MRR 0.585 (test;
novel 0.504, recurring 0.715; cue known for 97.5% of test queries), latent oracle 0.600, chance 0.059 —
the rule is inferable from permitted history. A predictor without history sees identical inputs across
histories and cannot infer the target by construction.

**Variant 1 (`gen_s1`, cue visible only through the cue node's identity), REC off, five paired training
seeds, lr 1e-3, 6 epochs, identical data / negatives / batching / selection
(`synthetic/synthetic_analysis/per_arm_stratified.csv`; test MRR on interactions, cue events excluded):**

| arm | n | MRR | SD | novel | recurring | Δ vs TSD (paired) | signs |
|---|---|---|---|---|---|---|---|
| TSD | 5 | 0.2098 | 0.0008 | 0.135 | 0.332 | – | – |
| current-only maps | 5 | 0.2098 | 0.0025 | 0.135 | 0.331 | +0.0000 ± 0.0026 | 2+ 3- |
| identity maps | 4 | 0.2059 | 0.0046 | 0.134 | 0.323 | −0.0040 ± 0.0045 | 0+ 4- |
| GRU + ordinary | 5 | 0.2091 | 0.0022 | 0.135 | 0.330 | −0.0007 ± 0.0026 | 3+ 2- |
| diagonal SSM + ordinary | 5 | 0.2098 | 0.0017 | 0.135 | 0.331 | +0.0001 ± 0.0015 | 1+ 4- |
| attention gates | 5 | 0.2069 | 0.0026 | 0.135 | 0.323 | −0.0029 ± 0.0028 | 1+ 4- |
| node-frame | 3 | 0.2117 | 0.0019 | 0.136 | 0.335 | +0.0018 ± 0.0015 | 3+ 0- |
| core-off (no memory) | 5 | 0.2125 | 0.0020 | 0.136 | 0.337 | +0.0027 ± 0.0025 | 5+ 0- |

**Null result:** no arm exceeds the memory-free core-off model; the "recurring" advantage (0.33 vs 0.13) is a
popularity prior that the memory-free model shows equally, and MRR on queries whose cue is in permitted
history equals the overall MRR for every arm. Under this budget (one-step gradient truncation, 6 epochs) none
of the evaluated recurrent cores — including the history-in-values comparators — extracts the cue when it is
observable only through the cue node's random feature vector. This variant therefore supports neither memory
nor memory-in-maps; it is kept in the record and was not tuned further.

**Variant 2 (`gen_s1r`, cue carried as an observable relation id 1+k, all arms with `--relation-in-input`
so the cue enters the memory input directly; `synthetic/synthetic_analysis_r/`):** same null. Test MRR on
interactions: TSD 0.2108 ± 0.0016 (3 seeds so far), current-only 0.2132 ± 0.0046, identity 0.2136 ± 0.0023,
GRU 0.2114 ± 0.0046, diagonal SSM 0.2127 ± 0.0013, attention 0.2120 ± 0.0035, node-frame 0.2129 ± 0.0019,
core-off 0.2155 ± 0.0010 (core-off − TSD = +0.005 on 3 of 3 paired seeds); permitted-history oracle 0.585.
Making the cue an explicit input does not let any evaluated core use it under one-step gradient truncation.

**Variant 3 (`gen_s2`, 100 nodes / 5 communities / typed cues, so a node's next interaction follows its cue
after a few batches instead of ~20; five essential arms, five paired seeds; `synthetic/synthetic_analysis_s2/`;
novel-pair fraction only 11% here, so this variant tests memory rather than novelty transfer):** TSD 0.265 ±
0.007, current-only 0.270 ± 0.010 (+0.005, 3+ 2−), identity 0.276 ± 0.023 (+0.011, 3+ 2−), GRU + ordinary
0.281 ± 0.009 (**+0.016 on 5 of 5 seeds**), core-off 0.268 ± 0.005 (+0.003, 2+ 3−); permitted-history oracle
0.531, latent oracle 0.590, chance 0.059. Shortening the delay lets the GRU core extract a small, seed-consistent
amount of history that the SSM core does not, but no arm comes close to the oracle.

**REC-on stratum (variant 1, separately labelled shortcut quantification; five seeds per arm):** with the
recurrence head every arm rises to 0.269-0.280 (TSD 0.279 ± 0.002, core-off 0.280 ± 0.002, identity 0.278,
current-only 0.276, GRU 0.274, node-frame 0.273, attention 0.272, diagonal SSM 0.269): MRR on recurring
pairs jumps from ~0.33 to ~0.62 while MRR on novel pairs falls from ~0.135 to ~0.067, i.e. the head is a
recurrence shortcut that trades novel-pair ranking for recurring-pair ranking, and it again leaves no room for
a core contribution (core-off − TSD = +0.001, 3+ 2−; all other arms within −0.010).

**Synthetic conclusion:** across three generator variants, none of the evaluated cores learns the delayed-cue
rule under the paper's training protocol (persistent state, one-step gradient truncation, fixed epoch budget);
the only seed-consistent difference is a small GRU advantage on the short-delay variant, and TSD is never above
its current-only-map or identity-map controls. The task therefore provides no support for putting history into
the restriction maps, and — because the history-in-values comparators also fail — it does not establish that
memory as implemented captures delayed cues either. The sanity conditions (a) and (b) hold; (c) holds only
weakly (GRU on variant 3). We report this as a negative result and did not tune the generator further.

## 5. Not evaluated (explicit)
- **ICEWS matrix**: not retrained (a run costs 20+ h); the three retained seeds were replayed for the P0 audit
  (`audit/icews_eval_s4x`). The retained ICEWS factorial (core-off+REC 0.319/0.311 vs TSD 0.339/0.334;
  REC-off arms 0.02-0.03) is reused as descriptive evidence only; GRU / diagonal-SSM / attention / node-frame
  cells on ICEWS are **not evaluated**.
- **Forum**: diagonal SSM, attention gates and node-frame have three seeds (43/46/47), not five.
- **Clock comparison and batching widths**: single-seed pilots on wiki (seed 43); not expanded to five seeds.
- **Heat-style transport, relation-conditioned edge maps, BPTT-length and initialisation controls** (handoff
  section 5): not run.
- **TSNN official reproduction**: not run; cite with published numbers marked as published-only.
- **Trade/genre matched-protocol ranks**: withheld (label-set differences at the split boundaries for trade;
  the label transformation of genre is not exactly reproducible from the edge weights available to the runner).

## 6. Manuscript changes implied by this evidence
1. **Central claim narrowed.** Memory-conditioned incidence maps give no paired benefit over current-only maps,
   identity maps or node-frame geometry on wiki (five seeds, locked lr, isolated RNG) or forum (five seeds
   with the REC head, retained seeds replay-verified), and a GRU or a diagonal SSM with ordinary propagation is
   at least as good. The supportable claim is that *a recurrent core adds to the recurrence head* (core-off is
   below TSD on every paired seed: wiki −0.027, forum −0.010 with REC and −0.127 without), not that
   temporal conditioning of geometry or learned transport is the mechanism.
2. **Remove the forum "learned restriction maps are the robust mechanism" narrative** (`COMPONENT_ATTRIBUTION.md`,
   `appendix_faithful_model.tex` factorial discussion): the low identity-map REC-off scores are a slow-start
   trainability effect of the SSM core under the 4-epoch budget; the GRU with identity transport matches TSD.
3. **Timing claims.** State that the step selector saturates at the cap for most or all updates on several
   datasets/seeds (forum 46/47, polecat 43/47, smallpedia 43/47; all wiki runs at lr 1e-3), that the trained
   wiki model is insensitive to the magnitude of the supplied gap (16x span, 0.004 MRR), that the neural score
   is exactly invariant to the query time, and that the no-gap control is an order-only recurrence whose
   difference from TSD is within seed noise on every dataset. Do not claim selective or physically timed memory.
4. **Benchmark numbers.** The wiki TSD configuration used for the leaderboard row should be the locked one
   (lr 1e-3, 600 s batches): five-seed mean 0.762 (rank 5 of 22 by the existing rule) instead of 0.733; state
   the batch width and that 300 s batches give 0.784 on seed 43. Node-property rows: report as descriptive
   scores under the documented schedule (Appendix paragraph in `appendix_protocol_trace.tex`), no rank claims.
5. **Appendix A additions** from the bundle's `scorer_spec.tex` plus the numerical-reporting paragraph:
   every new run serialises non-finite loss/gradient/skipped-step counters and per-query score validity; the
   retained runs' training skip rates remain unknown; the P0 audit found zero affected queries in 22 replays.
6. **Cost reporting.** Core-off reference vs bypass (156 s vs 3.9 s per pass, identical outputs); the vectorised
   builder changed forum wall-clock from ~16 h to ~80 min per run (timings before/after are not comparable;
   scores are). Parameter counts by component with unused-by-construction parameters excluded from "active".
7. **Synthetic task.** Report the three variants as a negative result on delayed-cue learning under one-step
   truncation; do not present the earlier nearly-all-recurring generator as a mechanism test.
8. **Related work**: cite TSNN and ST-Sheaf GNN as in `novelty_audit.md`; the node-frame control here is an
   internal ablation, not a TSNN reproduction.

## Code identity note
Runs launched before commit `2ca9fe5` used the original per-edge Python index builder in `lib/laplace.py`;
later runs use a vectorised implementation that is exactly equivalent (the original is retained as
`_compute_left_right_map_index_reference` and equality is pinned by
`models/test_review_controls.py::test_vectorised_left_right_map_index_matches_reference`). The change affects
wall-clock only (profiled synthetic step: 0.514 s -> 0.034 s), so timings across the two code states are
not comparable; scores are.
