# Where the numbers for the paper update live (snapshot 2026-09-16)

All benchmark numbers are from the **leak-free protocol** (predict-then-update, official TGB splits / negatives /
evaluators, metric read by name, Delta_k in units of the median training gap, corrected node-label cursor,
per-timestamp NDCG). Every earlier number (Hits@10-as-MRR era, leaky protocol, mis-paired labels) is superseded
and is marked as such in the documents below.

| what | file |
|---|---|
| Leaderboard standings with ranks (source of truth for every headline number) | `event_bench/LEADERBOARD.md` |
| LaTeX results table (leak-free, with ranks and head components) | `event_bench/results_table_v3.tex` |
| Appendix draft: faithful implementation, protocol corrections, empirical standing, factorial table | `event_bench/appendix_faithful_model.tex` |
| Component / mechanism attribution (per dataset; forum & icews head-vs-core factorials) | `event_bench/COMPONENT_ATTRIBUTION.md` |
| Audit ledger: every hidden assumption, its test, and its resolution (rows 1-36) | `analytic/audit/AUDIT.md` |
| Analytic (synthetic) study design, oracle ceilings, v1-v4 history, conclusions | `analytic/DESIGN.md` |
| Timeline / GPU plan to the deadline | `monitor/TIMELINE.md` |
| Live monitor state (regenerated every 3 min by the daemon) | `monitor/STATUS.md` |

## Raw results
* `event_bench/leakfree2/<run>/results.csv` (+ `history.csv`): one row per benchmark / ablation run — test and val metric,
  Hits@10, config JSON. `<run>.log` holds the `FINAL val_mrr=... test_mrr=... test_hits10=...` line and per-epoch lines.
  Naming: `<dataset>_f_s<seed>` champion; `<dataset>_abl_<identity|nodelta|curonly|nomem>[_s<seed>]` mechanism ablations
  with the REC head; `<dataset>_norec_<...>` without the REC head; `<dataset>_coreoff_<rec|norec>` temporal core off.
* `leakfree_nodeprop/*.csv|.log`: tgbn-trade / tgbn-genre reruns under the corrected label protocol.
* `analytic/synth_v4/runs/<run>/summary.json`: synthetic analytic campaign (v4 generator); `analytic/synth_v4/synth_*/meta.json`
  carries the oracle ceilings. `analytic/forum_factorial/<arm>/summary.json`: per-event stratified analyses
  (novel vs recurrent events, inter-event-gap quartiles) on the forum factorial checkpoints.
* Checkpoints (`best.pt`) and per-event dumps (`events.csv`) are NOT committed (0.3 GB + 0.57 GB); they are on the
  cluster under the same paths.

## Code entry points
* `exp/run_event_benchmark.py` — link / TKG / THG runner (flags: `--predict-from-previous`, `--recurrency-decoder`,
  `--recurrency-untyped`, `--relation-in-input`, `--node-type-emb`, `--sheaf-identity`, `--no-delta-t`, `--no-memory`,
  `--sheaf-conditioning current_only`, `--layers 0`, `--delta-time-scale`, `--reserve-gpu-mb`, `--save-checkpoint`, `--eval-only`).
* `exp/run_faithful_trade.py` — node-property runner (checkpoint/resume, per-epoch log, non-finite guard).
* `exp/temporal_benchmark_utils.py` — the single TGB-faithful harness (predict-then-update, label drain, guards).
* `exp/analyze_event_model.py`, `exp/run_synthetic_sheaf.py`, `exp/synthetic_temporal_sheaf.py` — analytic study.
* `models/faithful_event_model.py`, `models/temporal_sheaf_ssm.py` — the faithful Sec-3.2 model.
* Tests: `models/test_audit_assumptions.py` (deciding tests for every audited assumption), `models/test_faithful_event_model.py`,
  `models/test_temporal_sheaf_ssm.py`, `exp/test_temporal_benchmark_utils.py`.
* Queue / monitor infrastructure: `results/event_bench/queues/` (`wait_launch.sh`, `gpu_monitor.sh`, campaign scripts).
