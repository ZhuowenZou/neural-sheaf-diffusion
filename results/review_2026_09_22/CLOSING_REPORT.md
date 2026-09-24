# Closing report — review campaign 2026-09-22/24

Branch `review-2026-09-22`; results root `results/review_2026_09_22/`; narrative and all tables in `FINDINGS.md`.
Collector: collected 298 finished runs, 0 unfinished; fingerprint 3888628027

## (a) Completion
| block | finished | failed |
|---|---|---|
| P0 checkpoint replays (7 datasets x 3 seeds + 1) | 22 / 22 | 0 |
| tgbl-wiki matched matrix (8 arms x 5 seeds, lr pilots, clock/batching x 5 seeds) | complete | 0 |
| thgl-forum matched matrix (both REC settings, all arms x 5 seeds) | complete | 0 |
| tkgl-icews (TSD, current-only, identity, GRU, core-off x 5 seeds) | 18 / 18 new | 0 |
| tkgl-smallpedia (8 arms x seeds 43/46/47, + current-only s44) | 25 / 25 | 0 |
| thgl-software new arms x 3 seeds | 15 / 15 | 0 |
| tkgl-polecat, tkgl-wikidata GRU + core-off x 3 seeds | 12 / 12 | 0 |
| synthetic task (3 generator variants + REC-on stratum) | 145 / 145 | 0 |
| trimmed for time (not run) | smallpedia seeds 44/45 (15 runs, in `hold/`) | – |

## (b) Headline findings (numbers from FINDINGS.md)
- **P0 score validity:** all 22 replays have zero non-finite positive or negative logits and reproduce the retained
  test MRR to 6e-5 or better (smallpedia within 1e-3, CUDA non-determinism). Every reused number passes the gate.
- **Protocol:** tgbn-trade label Y equals normalised year-Y edges; our runner and the official TGN example both see
  year-Y edges before scoring label Y, but the official split label sets are shifted by one year; copy predictor 1.000.
  tgbn-genre label sets are identical; it is a genuine forecast (copy 0.014). Node-property ranks are withheld.
- **Central question — history in maps vs in values:** on no dataset does TSD beat current-only maps, identity maps,
  a GRU or a diagonal SSM with ordinary propagation. Wiki (5 seeds, lr locked 1e-3): TSD 0.762 ± 0.003, the four
  controls within ±0.002 with mixed signs (diagonal SSM uses 10k vs 445k active parameters). Forum with REC:
  every simpler control is at or above TSD on every paired seed (+0.009 to +0.022). ICEWS, smallpedia, software,
  polecat, wikidata: all core differences within ±0.012 and without a consistent sign in TSD's favour.
- **What is supported:** the recurrent core adds to the REC head — core-off is below TSD on wiki (−0.027, 5/5),
  forum (−0.010 with REC, 5/5; −0.127 without), ICEWS (−0.019, 4/5), polecat (−0.019, 3/3), software (−0.009, 3/3).
- **Timing:** the step selector saturates at the cap for up to 100% of updates (all wiki runs at lr 1e-3; two of three
  seeds on forum, polecat, smallpedia); the neural score is exactly query-time invariant; node clocks change wiki MRR by
  only +0.001 to +0.003, while halving the batch width adds +0.019 on every seed.
- **Synthetic:** no evaluated core learns the delayed-cue rule under one-step truncated training (all arms at the
  no-memory level, oracle 0.53-0.58); reported as a negative result.
- Manuscript changes are listed in FINDINGS.md section 6 (narrow the central claim; withdraw "learned restriction maps are
  the robust mechanism"; drop selective/physically-timed memory claims; wiki row 0.762 at 600 s batches).

## (c) Return package
`provenance/manifest.json`, `protocol_trace.csv`, `protocol_verdict.md`, `clock_diagnostics.csv`,
`per_seed_results.csv`, `paired_contrasts.csv`, `costs.csv`, `numerical_events.csv`, `query_validity.csv`,
`synthetic/` (generators, oracles, stratified analyses), `matched/wiki_strata/`, `tables_review.tex`,
`appendix_protocol_trace.tex`, `FINDINGS.md`, `REVIEW_SUMMARY.md`, `CODE_COMMIT.txt`, `run_event_benchmark_help.txt`.

## (d) Resume
`results/monitor/STATUS.md` (daemon and campaign agent have exited; relaunch with `setsid nohup bash
results/event_bench/queues/gpu_monitor.sh &` if more runs are queued), `python -m exp.review.collect_review`,
`bash exp/review/finalize.sh`. Held smallpedia seeds: move `hold/sp_*_s4[45].cmd` back to `queue/`.
