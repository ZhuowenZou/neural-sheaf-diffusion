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

(Clock/saturation aggregates from all checkpoint replays and the node-clock / batching-width runs are
collected in `clock_diagnostics.csv`.)

## 3. Matched comparisons (tgbl-wiki five seeds; thgl-forum subset)
(wiki: filled from per_seed_results.csv / paired_contrasts.csv when wave 2 completes)

**thgl-forum, principal simple temporal baseline (GRU + ordinary propagation, same head/REC/data/protocol as
the retained TSD seeds; old-protocol negative RNG regime to match those seeds; `forum/`):**

| REC | seed | TSD (retained) | GRU + ordinary | Δ (GRU − TSD) |
|---|---|---|---|---|
| on | 43 | 0.6280 | 0.6511 | +0.023 |
| on | 46 | 0.6412 | 0.6418 | +0.001 |
| on | 47 | 0.6187 | 0.6414 | +0.023 |
| on | mean ± SD | 0.629 ± 0.011 | 0.645 ± 0.006 | +0.016 ± 0.013 (3+ 0−) |
| off | 43 | 0.3940 | 0.3827 | −0.011 |
| off | 46 | 0.4007 | 0.3932 | −0.008 |
| off | 47 | 0.3470 | 0.3807 | +0.034 |
| off | mean ± SD | 0.381 ± 0.029 | 0.386 ± 0.007 | +0.005 ± 0.025 (1+ 2−) |

**Finding (forum):** a GRU memory with identity transport equals TSD without the REC head and exceeds it on
every seed with the REC head. The retained SSM-with-identity-maps ablation (REC off: 0.220 / 0.332 / 0.326)
is therefore not evidence that learned restriction maps are necessary on forum: the low identity-map scores
are specific to the SSM core (unstable across seeds), and a simpler core with the same identity transport
recovers TSD's accuracy. The earlier claim "learned restriction maps are the robust mechanism on forum" must be
withdrawn or narrowed to "within the SSM core". Diagonal-SSM, attention-gate and node-frame cells and seeds
44/45 are being added (`forum/`); the fast builder makes a forum run ~80 min.

**Core-off anchor, executed cost (`matched/coreoff_cost_wiki/coreoff_cost.csv`):** on the same 20k-edge
tgbl-wiki slice (561 snapshots, same GPU, three repeats) the reference core-off path (`--no-memory --layers 0`,
which still evaluates the unused SSM transition and map decoder) takes 156 s per pass; the verified bypass
(`--fast-core-off`) takes 3.9 s with bit-identical outputs (max |diff| = 0). The shortcut is labelled
"core off + REC (head only)": it retains the pointwise input encoder, the current-input spatial projection
Z0 = P_z[x; 0], the scorer and REC; it is not a REC-only predictor. All new core-off runs use the bypass.

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
after a few batches instead of ~20):** running on the five essential arms; results in
`synthetic/synthetic_analysis_s2/` (core-off 0.267 ± 0.004 on 4 seeds; permitted oracle 0.531; the novel-pair
fraction is only 11% here because 100 nodes recur heavily, so this variant tests memory, not novelty transfer).

## 5. Not evaluated (explicit)
- ICEWS matrix beyond the retained seeds: not retrained (cost 20+ h per run); checkpoint-replay audits of the
  three retained seeds are in `audit/`.
- Forum: only the GRU-ordinary baseline (REC on and REC off, seeds 43/46/47) and single-seed attention /
  node-frame pilots were launched; diagonal-SSM and five-seed forum cells are **not evaluated**.
- Heat-style transport, relation-conditioned maps, BPTT-length and initialisation controls (section 5 of the
  handoff): not run.
- TSNN official reproduction: not run; cite with published numbers marked as such.

## Code identity note
Runs launched before commit `2ca9fe5` used the original per-edge Python index builder in `lib/laplace.py`;
later runs use a vectorised implementation that is exactly equivalent (the original is retained as
`_compute_left_right_map_index_reference` and equality is pinned by
`models/test_review_controls.py::test_vectorised_left_right_map_index_matches_reference`). The change affects
wall-clock only (profiled synthetic step: 0.514 s -> 0.034 s), so timings across the two code states are
not comparable; scores are.
