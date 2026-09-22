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

Status: (filled by collect_review — see REVIEW_SUMMARY.md "Score-validity audit" and "Checkpoint-replay audits";
the replays also report `replay_minus_retained_test`, the difference between the replayed test MRR and the
metric stored in the retained run.)

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

(Clock/saturation aggregates from the checkpoint replays and the node-clock / batching-width runs are
collected in `clock_diagnostics.csv`; the fixed-membership gap intervention is in `clock/gap_wiki_s43/`.)

## 3. Matched comparisons (tgbl-wiki five seeds; thgl-forum subset)
(filled from per_seed_results.csv / paired_contrasts.csv)

**Core-off anchor, executed cost (`matched/coreoff_cost_wiki/coreoff_cost.csv`):** on the same 20k-edge
tgbl-wiki slice (561 snapshots, same GPU, three repeats) the reference core-off path (`--no-memory --layers 0`,
which still evaluates the unused SSM transition and map decoder) takes 156 s per pass; the verified bypass
(`--fast-core-off`) takes 3.9 s with bit-identical outputs (max |diff| = 0). The shortcut is labelled
"core off + REC (head only)": it retains the pointwise input encoder, the current-input spatial projection
Z0 = P_z[x; 0], the scorer and REC; it is not a REC-only predictor. All new core-off runs use the bypass.

## 4. Synthetic history-dependent task (synthetic/)
(filled from synthetic/gen_s1/generator_meta.json and synthetic/runs/)

## 5. Not evaluated (explicit)
- ICEWS matrix beyond the retained seeds: not retrained (cost 20+ h per run); checkpoint-replay audits of the
  three retained seeds are in `audit/`.
- Forum: only the GRU-ordinary baseline (REC on and REC off, seeds 43/46/47) and single-seed attention /
  node-frame pilots were launched; diagonal-SSM and five-seed forum cells are **not evaluated**.
- Heat-style transport, relation-conditioned maps, BPTT-length and initialisation controls (section 5 of the
  handoff): not run.
- TSNN official reproduction: not run; cite with published numbers marked as such.
