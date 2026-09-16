> **AUDIT CORRECTION (2026-09-04, supersedes the Hits@10 banner):** two further protocol defects were found and fixed. (1) Link tasks: embeddings were scored on the snapshot that contained the events being predicted (both models, training and evaluation); the leak-free predict-from-previous protocol is now the default for retraining and ALL link numbers below are pending re-measurement. (2) Node-property tasks: the TGB label cursor was reset per split, pairing tgbn-trade val/test with the 1987-1990 labels (persistence: 0.87 correct vs 0.46 mis-paired; static train-average: 0.65-0.73 correct vs 0.79 mis-paired), so the tgbn-trade/genre numbers are invalid pending re-run. See results/analytic/audit/AUDIT.md rows 20-29.

# Multi-dataset native-protocol benchmark campaign (2026-08-01)

Goal: extend the improved raw-data protocol (uncapped train, finest feasible
time resolution, tuned lr, patient model selection) from tgbn-trade to the
remaining datasets, for BOTH the original and the faithful model.

Trade reference (native res, seeds 43/46/47):
  original b_long 0.8656 +/- 0.0066 | faithful readout+lr0.005 0.8520 +/- 0.0153

## Lanes
Phase B (tuning, seed 43, shortened budgets):
  smallpedia: lr {3e-4, 1e-3, 3e-3} x {original, faithful}  [exact yearly ts]
  tgbl-wiki:  lr {1e-3, 3e-3, 1e-2} x {original, faithful}  [window from timing smoke]
  genre:      lr {0.0125, 0.005} x {faithful, original}     [weekly 604800; daily if cheap]
  wikidata:   lr {3e-4, 3e-3} x {original, faithful}        [2M suffix protocol, 4 ep]
Phase C (final): 3 seeds {43,46,47} at best config per dataset x model.

## Protocol notes
- tgbl/tkgl/genre data under DEFAULT TGB root (unset TGB_ROOT);
  wikidata requires TGB_ROOT=$PWD/datasets.
- Event runner: exp/run_event_benchmark.py (dst-range negatives for wiki,
  context graph = first 50k train edges, grad clip 1.0, state reset before
  every replay; float64 delta_t in both models).
- Genre/nodeprop: exp/run_faithful_trade.py / run_baseline_trade.py --dataset.
- Selection: best tracking-val epoch; faithful needs patient selection
  (trade lesson: transient degradation then recovery).
