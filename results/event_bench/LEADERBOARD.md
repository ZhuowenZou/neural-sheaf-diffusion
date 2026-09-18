# TGB leaderboards (fetched 2026-09-05 from tgb.complexdatalab.com) and our standing

Ranking rule used below: insert our leak-free test metric into the public table by test score; "rank k of n" counts the public entries plus ours. Our numbers are from the corrected leak-free campaign (`results/event_bench/leakfree2/`, predict-then-update, true MRR by name; node-property with the corrected label cursor).

## Link property prediction (test MRR)

**tgbl-wiki-v2** (22 with ours): TPNet 0.827 | Heuristic(LocalGlobal) 0.821 | HyperEvent 0.810 | DyGFormer 0.798 | NAT 0.749 | Base3 0.743 | DyGMamba 0.739 | TNCN 0.718 | CAWN 0.711 | CTAN 0.668 | EdgeBank(tw) 0.571 | EdgeBank(unl) 0.495 | HTGN(UTG) 0.464 | EGCNo(UTG) 0.398 | TGN 0.396 | GCLSTM(UTG) 0.374 | GCN(UTG) 0.336 | TCL 0.207 | TGAT 0.141 | GraphMixer 0.118 | DyRep 0.050
- Ours (faithful + REC-x): s43 0.7357, s46 0.7305, s47 0.7331 -> 3-seed mean **0.7331 +/- 0.0026** (Hits@10 0.850) -> rank **8 of 22** (between DyGMamba 0.739 and TNCN 0.718).
- Ablations (leak-free, 3 seeds each; full 0.7331 +/- 0.0026): identity maps 0.7292 +/- 0.0037, no-Delta 0.7264 +/- 0.0027 (every seed below every full seed), current-only 0.7310 +/- 0.0050; EMB-head 0.7145 (s43). Only the physical gap shows a seed-consistent effect (-0.007).

## Temporal knowledge graphs (test MRR)

**tkgl-smallpedia** (8 with ours): CEN 0.612 | Recurrency Baseline(train) 0.605 | TLogic 0.595 | RE-GCN 0.594 | Recurrency Baseline(default) 0.486 | EdgeBank(tw) 0.353 | EdgeBank(unl) 0.333
- Ours (faithful champion REL + REC typed/untyped/sym): s43 0.6164, s46 0.6033, s47 0.6112 -> 3-seed mean **0.6103 +/- 0.0066** -> rank **2 of 8** (0.002 behind CEN 0.612, ahead of Recurrency Baseline 0.605); best single seed would rank 1.
- Ablations (leak-free, 3 seeds; full 0.6103 +/- 0.0066): identity maps 0.6133 +/- 0.0063, no-Delta 0.6109 +/- 0.0053, current-only maps 0.6054 +/- 0.0082 (all within noise).

**tkgl-polecat** (7): TLogic 0.228 | Recurrency Baseline(train) 0.198 | CEN 0.184 | RE-GCN 0.175 | Recurrency Baseline(default) 0.167 | EdgeBank(tw) 0.056 | EdgeBank(unl) 0.045
- Ours (faithful REL + REC typed/untyped): s43 0.2403, s46 0.2461, s47 0.2462 -> 3-seed mean **0.2442 +/- 0.0034** (val 0.247 / 0.252 / 0.253; Hits@10 0.39-0.40) -> rank **1 of 8** (ahead of TLogic 0.228 / val 0.236).
- Ablations (leak-free, s43; full 0.2403): identity maps 0.2449, no-Delta 0.2441, current-only 0.2476 (all within seed noise; the head carries the result on daily polecat, as on smallpedia).

**tkgl-icews** (8 with ours): Recurrency Baseline(train) 0.211 | Recurrency Baseline(default) 0.206 | CEN 0.187 | TLogic 0.186 | RE-GCN 0.182 | EdgeBank(tw) 0.020 | EdgeBank(unl) 0.009
- Ours (faithful REL + REC typed/untyped, 2M-edge train suffix, full official evaluation): s43 0.3386, s46 0.3342, s47 0.3326 -> 3-seed mean **0.3351 +/- 0.0031** (val 0.309-0.324; Hits@10 0.53-0.54) -> rank **1 of 8** (leader Recurrency Baseline 0.211 / val 0.270). COMPLETE. Factorial: REC on full 0.337 (2 seeds), identity 0.337, no-Delta 0.338, no-memory 0.328 (2 seeds; -0.008 seed-consistent), current-only 0.329, core OFF 0.319; REC off: every arm 0.02-0.03 -> recurrence-dominated dataset; the core adds +0.02 on top of the head.

**tkgl-wikidata** (3 with ours): EdgeBank(tw) 0.535 | EdgeBank(unl) 0.535
- Ours (faithful REL + REC typed/untyped/sym, 2M-edge train suffix): s43 0.5371, s46 0.5500, s47 0.5323 -> 3-seed mean **0.5398 +/- 0.0092** (val 0.64; Hits@10 0.60-0.62) -> rank **1 of 3** (above EdgeBank 0.535 on every seed; the only learned entry).
- Ablations (leak-free, s43; full 0.5371, 3-seed std 0.009): identity maps 0.5336, no-Delta 0.5362, current-only maps 0.5381 (all within noise).

## Temporal heterogeneous graphs (test MRR)

**thgl-software** (7 with ours): STHN 0.731 | EdgeBank(unl) 0.449 | TGN(edge type) 0.424 | TGN 0.324 | EdgeBank(tw) 0.288 | Recurrency Baseline(default) 0.099
- Ours (faithful TYPE + REL + REC typed/untyped): s43 0.4360, s46 0.4400, s47 0.4369 -> 3-seed mean **0.4376 +/- 0.0017** -> rank **3 of 7** (behind EdgeBank(unl) 0.449, ahead of TGN(edge type) 0.424).
- Ablations (leak-free, 3 seeds; full 0.4376 +/- 0.0021): identity maps 0.4392 +/- 0.0030, no-Delta 0.4393 +/- 0.0003, current-only maps 0.4388 +/- 0.0043 (all within noise).

**thgl-forum** (5): TGN(edge type) 0.729 | TGN 0.649 | EdgeBank(unl) 0.617 | Recurrency Baseline(default) 0.561 | EdgeBank(tw) 0.534
- Ours (faithful TYPE + REL + REC typed/untyped, 4M-edge train suffix): s43 0.6280, s46 0.6412, s47 0.6187 -> 3-seed mean **0.6293 +/- 0.0113** (Hits@10 0.683 / 0.694 / 0.653) -> rank **3 of 6** (behind TGN 0.649, ahead of EdgeBank(unl) 0.617).
- Ablations with the REC head on (2 seeds; full 0.635): identity maps 0.641, no-Delta 0.635, current-only maps 0.629, no-memory 0.649 (s43) -> all within seed noise (std 0.011). Head-vs-core factorial (s43, test MRR; REC off column): full 0.394, identity maps 0.220, no-Delta 0.361, no-memory 0.409, current-only 0.384, core OFF 0.237; core OFF + REC 0.615. Seed 46 (REC off): full 0.401, identity 0.333, no-Delta 0.351, no-memory 0.383, current-only 0.411, core OFF 0.220; Seed 47 (REC off): full 0.347, identity 0.326, no-Delta 0.390, core OFF 0.232. FINAL (REC off 3 seeds): full 0.381 +/- 0.029, identity 0.293 +/- 0.063, no-Delta 0.367 +/- 0.020, core OFF 0.230 +/- 0.009: the head masks the core (+0.02 with it); without it the core adds +0.15 on every seed and the learned restriction maps are the robust mechanism (identity loses on every seed); Delta_k and memory are not robust on forum. Per-event analysis: the core's gain sits on novel pairs (0.138 vs 0.090 with the head) and grows with the inter-event gap.

**thgl-github** (2): EdgeBank(unl) 0.413 | EdgeBank(tw) 0.374 — not run (20-random-negative protocol is non-discriminative; our smoke saturated at 0.9999).
**thgl-myket** (2): EdgeBank(unl) 0.456 | EdgeBank(tw) 0.245 — same caveat (our leaky smoke 0.9994).

## Node property prediction (test NDCG@10)

**tgbn-trade** (8 with ours): NAVIS 0.863 | Persistent Forecast 0.855 | Moving Average 0.823 | TGNv2 0.735 | DyGFormer 0.388 | TGN 0.374 | DyRep 0.374
- Ours, PAPER protocol (3-year windows, 2048-edge training cap, 5 seeds, fixed label pairing 2026-09-11): **0.351 +/- 0.012** (val 0.39) -> rank **7 of 8** (below DyGFormer 0.388, above TGN/DyRep 0.374; heuristics 0.82-0.86). The 0.821 +/- 0.011 reported 09-05..09-11 is WITHDRAWN: its evaluation path still reset the label cursor per split, so it rewarded a static training average against 1987-1988 labels. TGB-exact protocol (yearly windows, uncapped training, 5 seeds) rerunning.
- Ours, TGB-exact protocol (yearly windows, uncapped training, memory readout; fixed label pairing): seeds 43-47 test 0.656 / 0.665 / 0.667 / 0.671 / 0.659 -> 5-seed mean **0.6636 +/- 0.0061** (val 0.74-0.76; best epochs 58-176) -> rank **5 of 8** (below TGNv2 0.735 and the three heuristics 0.82-0.86; far above DyGFormer 0.388 / TGN 0.374). COMPLETE.

**tgbn-genre** (8 with ours): NAVIS 0.528 | Moving Average 0.509 | TGNv2 0.469 | TGN 0.367 | DyGFormer 0.365 | Persistent Forecast 0.357 | DyRep 0.351
- Ours (weekly windows, uncapped, memory readout, fixed label drain, s43, 40 epochs, best epoch 34): **0.4617** (val 0.476) -> rank **4 of 8** (below NAVIS 0.528, Moving Average 0.509, TGNv2 0.469; above TGN 0.367). Daily windows (TGB-exact, one label per snapshot), seeds 43/46/47: 0.4482 / 0.4452 / 0.4502 -> 3-seed mean **0.4479 +/- 0.0025** (val 0.465-0.466, best epochs 4-8) -> rank **4 of 8** (below NAVIS 0.528, Moving Average 0.509, TGNv2 0.469; above TGN 0.367). COMPLETE. Pre-audit 0.513 was mis-paired and is invalid.
