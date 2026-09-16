> **AUDIT CORRECTION (2026-09-04, supersedes the Hits@10 banner):** two further protocol defects were found and fixed. (1) Link tasks: embeddings were scored on the snapshot that contained the events being predicted (both models, training and evaluation); the leak-free predict-from-previous protocol is now the default for retraining and ALL link numbers below are pending re-measurement. (2) Node-property tasks: the TGB label cursor was reset per split, pairing tgbn-trade val/test with the 1987-1990 labels (persistence: 0.87 correct vs 0.46 mis-paired; static train-average: 0.65-0.73 correct vs 0.79 mis-paired), so the tgbn-trade/genre numbers are invalid pending re-run. See results/analytic/audit/AUDIT.md rows 20-29.

# !!! METRIC CORRECTION (2026-09-04) !!!
All link-style numbers below that were produced by evaluate_model_streaming
before 2026-09-04 are **Hits@10**, not MRR (the harness took the first key of
TGB's {'hits@10','mrr'} result). True MRR is being re-computed for every
checkpoint (results/analytic/rescore_*); leaderboard comparisons must use
the corrected values. NDCG (tgbn-*) numbers are unaffected.

# Component attribution for the faithful backbone (living document)

Principle: every addition is a HEAD- or INPUT-level component; the Sec-3.2
temporal core (exact-ZOH selective SSM, physical Delta_k selector, per-event
sheaf decoding, Euler diffusion) is IDENTICAL across all rows. Each row
changes exactly one component relative to its parent row, so contributions
are isolated, not conflated.

## Components
- CTX   spatial context: closure_hops 2 + 3 diffusion layers (spatial module width)
- EMB   learnable per-node input embedding (--learn-node-emb)
- TYPE  node-type input embedding (--node-type-emb; heterogeneous graphs)
- REL   incident-relation aggregate appended to SSM input q (--relation-in-input)
- REC   causal (s,r,o) recurrency cache -> (seen, count, recency) -> zero-init
        MLP bonus on event scores (--recurrency-decoder); degrades to an
        (s,o) edge cache when the dataset has one relation
- NEG   training negatives per positive (32 vs 128)
- REC-so untyped (s,o) recurrency channel (--recurrency-untyped): a second
        cache keyed on the node pair only, so repeats of a pair under a
        different event type count as seen (EdgeBank-style); features of both
        channels feed the same gate MLP (input 6).
- REC-sym order-agnostic pair channel (--recurrency-symmetric): keyed on
        (min(s,o), max(s,o)) so reverse-direction repeats count as seen;
        gate MLP input grows to 9 with all three channels.
- REC-x event-exact recurrency (2026-08-05+): same cache, but events earlier
        IN TIME within the current snapshot are visible to later ones (strict
        t_j < t_q; ties invisible), and recency is measured from the query
        event's own timestamp. Removes the bucketing blind spot (software:
        median inter-event gap 3 s vs 1 h buckets). Rows labelled REC used the
        snapshot-granular cache; REC-x rows use the event-exact one.

## tgbl-wiki (test MRR, seed 43 unless noted; window 600, lr 3e-4)
| chain                        | val    | test   |
|------------------------------|--------|--------|
| base (round-1 tuned)         | 0.0905 | 0.1088 |
| +NEG128                      | 0.1128 | 0.0885 |
| +EMB                         | 0.0975 | 0.1131 |
| +CTX                         | 0.2998 | 0.2506 |
| +CTX +NEG128                 | 0.2162 | 0.2059 |
| +CTX +EMB +NEG128 (3 seeds)  |   -    | 0.3138 +/- 0.0503 |
| +CTX +EMB (NEG32) [K]        | 0.6653 | 0.6357 (s43); s46 0.2856, s47 0.2001 -> high variance |
| +CTX +EMB +REC [L]           |   -    | 0.8774 +/- 0.0226 (3 seeds: 0.8568/0.8736/0.9017; snapshot-granular cache, 6x cost) |
| +REC only (no CTX/EMB)       |   -    | 0.8618 +/- 0.0050 (3 seeds: 0.8561/0.8657/0.8636) FINAL |
| +REC-x only (event-exact)    |   -    | 0.8942 +/- 0.0037 (3 seeds: 0.8964/0.8899/0.8964) FINAL - wiki headline |
Leaderboard: TPNet 0.827, DyGFormer 0.798, EdgeBank(tw) 0.571, TGN 0.396.

## tkgl-smallpedia (test MRR, full official eval; lr 1e-2)
| chain                  | test |
|------------------------|------|
| base (relation-blind)  | 0.0218 +/- 0.0032 (3 seeds) |
| +REL                   | 0.0405 +/- 0.0080 (3 seeds) |
| +REL +REC              | 0.7040 +/- 0.0046 (3 seeds, trimmed-negatives protocol) |
| +REL +REC (exact padded-negatives protocol) | **0.7048 +/- 0.0023** (3 seeds: 0.7068/0.7022/0.7053) FINAL |
| +REC only (no REL)     | 0.6741 (s43) |
Original model (relation head only): 0.0256 +/- 0.0047.
Leaderboard: CEN 0.612, RecB(train) 0.605, TLogic 0.595.

## software = thgl-software (test MRR; window 3600, lr 1e-3, seed 43)
| chain                          | test |
|--------------------------------|------|
| faithful base (TYPE+REL, EMB)  | 0.0470 |
| + CTX (+EMB, lr 3e-4)          | 0.0683 |
| + CTX (+EMB, lr 1e-3)          | 0.0591 |
| CTX +TYPE +REL +REC, NO EMB    | 0.1479 (seeds 46/47 running) |
| ... window 600 (finer)         | running (software_w600) |
| ... 20-epoch budget            | 0.1431 (no gain) |
| CTX-free +TYPE +REL +REC-x     | 0.1602 (event-exact; bucketing was not the blind spot) |
| + untyped (s,o) cache channel (REC-x + REC-so) | 0.5381 +/- 0.0053 (3 seeds: 0.5361/0.5441/0.5340) FINAL - above EdgeBank 0.449 |
| ... + window 600               | 0.5259 (s43; below 3600-window 0.538 - finer windows do not stack with the untyped channel) |
| ... + symmetric (min,max) pair channel | 8 ep: 0.5448 (s43); 20 ep: 0.5597 +/- 0.0088 (3 seeds: 0.5695/0.5522/0.5573) = untyped+20ep within noise -> NULL on software; headline stays the simpler untyped config |
| ... + 20-epoch budget          | 0.5576 +/- 0.0027 (3 seeds: 0.5603/0.5549/0.5575) FINAL - software headline (#2 behind STHN 0.731) |
Original model: 0.1417. Leaderboard (software): STHN 0.731, EdgeBank(unl) 0.449, TGN(edge type) 0.424, TGN 0.324, EdgeBank(tw) 0.288, RecB 0.099.
Open hypothesis: hour-bucketing hides intra-hour repeats from REC
(median inter-event gap is 3 s); finer windows should close part of the
EdgeBank gap. The 682k-node EMB table (43.6M params) hurt at this budget.

## Protocol notes
- All numbers: official TGB splits, official negative sets, official
  evaluators, best-tracking-val checkpoint selection, gradient clipping,
  divergence-guarded selection.
- 2026-08-04 onward: EXACT per-query negatives (ragged lists padded, padded
  slots scored -inf). Earlier rows used snapshot-min trimming; bounded
  effect <1% relative (audit: results/event_bench/, smallpedia lists
  47,029-47,432 long; zero dropped edges either way).
- REC commits pending events at the NEXT step: a snapshot never scores its
  own events. TKG convention (train+val history available during test
  streaming) matches the leaderboard baselines (RecB, EdgeBank).

## tkgl-polecat (test MRR, full-entity negatives ~150,930/query; daily snapshots; lr 1e-2)
Eval uses the dense-vocabulary path (one [rows, N] matmul per chunk + sparse
recurrency scatter), verified numerically equivalent to per-candidate scoring.
| chain            | test |
|------------------|------|
| base             | 0.0781 (s43; val 0.0730; lane diverged at epoch 8 - guarded selection kept the last finite epoch) |
| +REC             | 0.3232 (s43; val 0.3212; full official eval) - above leaderboard best NHCTE 0.245 |
| +REL +REC        | 0.3153 (s43; val 0.3167; == REC-only within noise - relation-input adds nothing on polecat; diverged at epoch 9, guarded selection kept epoch 8) |
| +REL +REC-x +untyped | 0.3916 +/- 0.0130 (3 seeds, lr 1e-2: 0.3911/0.4048/0.3788) FINAL - polecat headline (leaderboard TLogic 0.228; NHCTE 0.245 in-paper). lr 3e-3 variant: 0.3953 (s43) - same accuracy, no late divergence -> preferred config for future runs |
Note: lr 1e-2 destabilizes polecat beyond ~8 epochs; finals should use lr 3e-3 or an 8-epoch budget.
Leaderboard best: NHCTE 0.245 (RecB weak here: low recurrency degree).

## forum = thgl-forum (test MRR; 23.8M edges; window 600; 2 relations, 2 node types)
Context (closure-2) configuration triggers a CUDA illegal-access at scale
(sparse-kernel index overflow suspected); base-architecture chain used.
| chain                     | test |
|---------------------------|------|
| base +TYPE +REL +REC (uncapped 16.6M train, snapshot-granular cache; 50k-prefix eval) | val 0.7920 / test 0.7419 (s43; ~5.2 h/epoch) - uncapped training does not beat the 4M-suffix config on full eval (0.759/0.736) |
| base +TYPE +REL +REC-x +untyped (4M-edge suffix; val/test = 100k-edge PREFIX eval, not full split) | 0.6941 +/- 0.0106 (3 seeds: 0.7010/0.6993/0.6819; prefix eval) |
| same config, FULL official val/test eval (3.56M events each; 100 negatives/query), checkpointed | s43 test 0.7591 (val 0.7593), s46 test 0.7355 (val 0.7363) - both above leaderboard #1 TGN-et 0.729; s47 running |
Leaderboard (forum): TGN(edge type) 0.729, TGN 0.649, EdgeBank(unl) 0.617, RecB 0.561, EdgeBank(tw) 0.534.

## tkgl-wikidata (test MRR; 2M-edge suffix train; 1,000 negatives/query)
| chain            | test |
|------------------|------|
| base (lr 1e-3)   | 0.0241 +/- 0.0117 (2 seeds) |
| +REL +REC (lr 3e-3) | 0.4844 +/- 0.0296 (3 seeds: 0.4504/0.5043/0.4986); cache sees only the 2M-edge suffix. NB: the 100k-prefix tracking-val was unreliable for s46 (~0.07 while full val = 0.54) |
| +REL +REC-x +WARM (12.0M pre-suffix facts, 1.56M distinct triples) | s43 val 0.5656 / test 0.4874 (val up, test within seed range - inconclusive) |
| +REL +REC-x +untyped (s,o) channel (track-val 300k) | 0.5760 +/- 0.0056 (3 seeds: 0.5765/0.5813/0.5701) FINAL - above EdgeBank 0.535 on every seed |
| ... + WARM full pre-suffix history | 0.5553 (s43; val 0.6372) - below untyped-only 0.576: warm history REJECTED (2nd time; val gains never transfer to test) |
| ... + symmetric pair channel | 0.5959 +/- 0.0109 (3 seeds: 0.5896/0.6084/0.5896) FINAL - wikidata headline (EdgeBank 0.535) |
Original 0.0482 +/- 0.0061; leaderboard: EdgeBank 0.535 (only method that scales).

## Dataset regimes (measured; explains which component pays where)
REC_typed / REC_pair / REC_sym = fraction of TEST events whose (s,r,o) triple /
(s,o) pair / unordered pair already occurred in train+val. new_dst = fraction
of test destinations never seen before.
| dataset          | nodes     | edges  | e/node | rels | span    | REC_typed | REC_pair | REC_sym | new_dst | our result vs typed->untyped gain |
|------------------|-----------|--------|--------|------|---------|-----------|----------|---------|---------|-----------------------------------|
| tgbl-wiki        | 9.2k      | 157k   | 17.1   | 1    | 31 d    | 0.709     | 0.709    | 0.709   | 0.073   | 0.894; typed==pair -> no untyped gain expected |
| tkgl-smallpedia  | 47k       | 1.10M  | 23.2   | 566  | 125 y   | 0.361     | 0.366    | 0.366   | 0.173   | 0.712; untyped/sym +0.007 (typed~pair) |
| tkgl-polecat     | 151k      | 3.56M  | 23.6   | 32   | 5 y     | 0.338     | 0.501    | 0.501   | 0.094   | 0.391; untyped +0.07 (pair >> typed) |
| thgl-software    | 682k      | 1.49M  | 2.2    | 14   | 31 d    | 0.040     | 0.231    | 0.231   | 0.420   | 0.558; untyped +0.40 (typed cache ~useless) |
| thgl-forum       | 153k      | 23.8M  | 155.5  | 2    | 31 d    | 0.515     | 0.515    | 0.526   | 0.012   | 0.736 (full eval, s46) |
| tkgl-wikidata    | 1.23M     | 19.7M  | 16.1   | 1192 | ~2k y   | 0.267     | 0.274    | 0.274   | 0.348   | 0.596; untyped +0.09, sym +0.02 |
Reading: the typed->pair recurrency gap predicts the untyped channel's gain
(software 0.04->0.23: +0.40; polecat 0.34->0.50: +0.07; wiki/smallpedia
equal: ~0). Our MRR exceeds the recurrency degree on every dataset, i.e. the
backbone ranks the non-recurrent remainder well above chance.
| tkgl-icews (cand.) | 88k     | 31.0M  | 353.2  | 782  | 28 y d  | 0.371     | 0.811    | 0.811   | 0.016   | full-vocab negatives (87,855/query); smoke (500k suffix, 2 ep, 30k-prefix eval) val 0.1576 / test 0.1594 vs leaderboard RecB 0.211; PICKED - training lane running, full eval ~12 h/seed via --eval-only |
| thgl-github (cand.)| 5.86M   | 17.5M  | 3.0    | 14   | 31 d    | 0.006     | 0.082    | 0.082   | 0.535   | official negatives = 20 RANDOM nodes/query out of 5.86M -> smoke scores 0.9999 (finite losses, genuine): the protocol cannot discriminate models; DROPPED in favour of myket |
| thgl-myket (cand.) | 1.53M   | 53.6M  | 35.0   | 2    | 197 d   | 0.326     | 0.359    | 0.359   | 0.008   | official negatives = 20 random/query (same weakness as github); run for an official entry WITH caveat; smoke queued |
| tkgl-yago (cand.)  | 10.6k   | 402k   | 38.0   | 20   | 189 y   | 0.842     | 0.843    | 0.843   | 0.017   | trivially recurrent, not on leaderboard; skipped |

## Analytic experiment (mechanism attribution for the temporal-sheaf core)
Design: results/analytic/DESIGN.md. Ablations of the CORE (not the decoder):
sheaf_identity (R = I), no_delta_t (order-only recurrence), current_only
(sheaf not conditioned on memory), vs the full core and the efficient
implementation. Arm A: synthetic temporal sheaf process with planted
transports / continuous-time dynamics / bursty gaps (results/analytic/synth_*).
Arm B: tkgl-smallpedia ablation lanes (results/analytic/sp_abl_*) and per-event
analyses (results/analytic/analysis_*): MRR by novel/recurrent, gap quartile,
relation rarity; transport residual positives vs negatives; time-scale
perturbation (x0.25 / x4) at inference.

Core ablations on smallpedia (same decoder REL+REC-x+untyped+sym, lr 1e-2, 15 ep, seed 43):
| core variant | val | test |
|---|---|---|
| full core | 0.7236 | 0.7116 |
| identity sheaf (R=I) | 0.7309 | 0.7180 |
| no Delta_k selector | pending |
| current-only sheaf | 0.7065 | 0.6812 (-0.030 vs full core: memory-conditioned geometry is load-bearing) |
| efficient impl. (same decoder) | pending (tracking ~0.08) |

## EMB-head / TYPE-head (added 2026-09-05; flag `--emb-in-head`, model arg `embeddings_in_head`)
The scoring head's pointwise encoder consumed the RAW input features x; the learnable node embedding (EMB) and node-type embedding (TYPE) entered only `local_x` inside the temporal step, i.e. they reached the head solely through the `spatial` state of nodes that were active. Under the leak-free predict-then-update protocol most candidates carry stale state, so candidate identity was effectively unavailable to the head. With the flag the head encodes x + embeddings for every node. The temporal core is untouched (unit test: `spatial` bit-identical with and without the flag). Default off; enabled for the synthetic analytic campaign (all variants share it) and to be evaluated as an explicit component on software/forum (TYPE) and wiki (EMB).

## Leak-free attribution (2026-09-08; campaign `leakfree2/`, true MRR by name, predict-then-update)
| dataset | full | identity maps | no Delta_k | current-only | original (no head) | leaderboard top |
|---|---|---|---|---|---|---|
| tkgl-smallpedia (s43) | 0.616 | 0.618 | 0.616 | 0.596 | 0.014 | CEN 0.612 |
| tgbl-wiki (s43) | 0.736 | running | running | running | 0.011 | TPNet 0.827 |
| thgl-software (s43) | 0.436 | queued | queued | queued | 0.062 | STHN 0.731 |
Also queued on wiki: EMB-head (learnable node embedding visible to the head).
Synthetic v4 (identifiable generator, shared REC head, observed x0 + community): evolving regimes full 0.276/0.290, identity 0.277/0.290, original 0.176/0.200, static-feature scorer 0.200, last-snapshot-state ceiling 0.46/0.63; static regime full 0.849, identity 0.846, original 0.790 (ceiling 0.875). Reading so far: learned restriction maps add nothing over identity maps; the faithful backbone's +0.1 over the original in the evolving regimes comes with the head/features; Delta_k and memory contrasts pending (`results/analytic/DESIGN.md`).

### Leak-free ablations, second-resolution streams (2026-09-09; seed 43, test MRR)
| dataset | full | identity maps | no Delta_k | current-only (map conditioning; NOT a memory ablation, see 2026-09-14 note) | EMB-head | seed std of full |
|---|---|---|---|---|---|---|
| tgbl-wiki (600 s buckets), 3 seeds each | 0.7331 +/- 0.0026 | 0.7292 +/- 0.0037 (-0.004) | 0.7264 +/- 0.0027 (-0.007; every seed below every full seed) | 0.7310 +/- 0.0050 (-0.002) | 0.7145 (s43; -0.021) | 0.0026 |
| thgl-software (3600 s buckets) | 0.4360 | 0.4357 (0.000) | 0.4392 (+0.003) | 0.4341 (-0.002) | - | 0.0017 (3 seeds) |
| tkgl-smallpedia (yearly) | 0.6164 | 0.6182 (+0.002) | 0.6160 (0.000) | 0.5959 (-0.020) | - | 0.0066 (3 seeds) |
| tkgl-polecat (daily) | 0.2403 | 0.2449 (+0.005) | 0.2441 (+0.004) | 0.2476 (+0.007) | - | 0.0034 (3 seeds) |
| thgl-forum (600 s buckets, s43) | 0.6280 | 0.6313 (+0.003) | 0.6475 (+0.020) | 0.6475 (+0.020) | - | 0.0113 (3 seeds) |
Reading (revised 2026-09-14 with 3 seeds per wiki ablation): on wiki only the physical gap has a seed-consistent effect (-0.007; every no-Delta seed below every full seed); identity maps and current-only overlap the full model. The single-seed claim of 09-09 (memory +0.010, maps +0.009) did not survive seeding. EMB-head hurts (-0.021). On forum the gap/memory ablations are +0.02 on one seed (1.7 seed-std); seed 46 queued. On yearly smallpedia only the memory matters; on software none of the three mechanisms moves the score (all within 2 seed-std): there the TYPE/REL/REC head over the sheaf diffusion carries the result.

## Correction (2026-09-14): "current-only" is a sheaf-conditioning ablation, not a memory ablation
`--sheaf-conditioning current_only` decodes the restriction maps from the current input instead of the memory readout, but the memory readout still enters the spatial initialisation Z0 = P_z[x; h]. Every earlier row that called it "no temporal memory" is relabelled. A true memory ablation now exists: `--no-memory` (SSM state advanced but never read; Z0 = P_z[x; 0]; maps from the current input); `--no-memory --layers 0` switches the temporal core off entirely (head over a pointwise projection of the input).

## Head-vs-core factorial on thgl-forum (design 2026-09-14; the dataset with the largest temporal-to-spatial ratio: 2.56M timestamps / 153k nodes, 227 active timestamps and 311 events per node)
Rows = temporal core: full | identity maps | no Delta_k | no memory | current-only maps | core OFF (layers 0 + no memory). Columns = head: REC on (typed + untyped recurrency decoder) | REC off. TYPE and REL inputs fixed in all arms. Reference: leaderboard Recurrency Baseline 0.561 (parameter-free head-only), TGN(edge type) 0.729. Contrasts: head contribution = (full, REC on) - (full, REC off) and (core off, REC on) - (core off, REC off); core contribution = (full, REC on) - (core off, REC on) and (full, REC off) - (core off, REC off); mechanism contributions read in BOTH head columns, because a recurrency head can mask the core's own recurrence.

### thgl-forum head-vs-core factorial, seed 43 (test MRR, leak-free; TYPE + REL inputs in every arm; 4M-edge train suffix; best-tracking checkpoint)
| temporal core \ head | REC on | REC off |
|---|---|---|
| full (learned maps, Delta_k, memory) | 0.628 | 0.394 |
| identity restriction maps | 0.631 | **0.220** |
| no Delta_k | 0.648 | 0.361 |
| no memory (true ablation) | 0.649 | 0.409 |
| current-only map conditioning | 0.648 | 0.384 |
| core OFF (layers 0 + no memory) | 0.615 | 0.237 |
Leaderboard references: Recurrency Baseline (parameter-free head only) 0.561; TGN(edge type) 0.729; seed std of the full REC-on model 0.011.
Contrasts (seed 43): head = +0.23 with the core, +0.38 without it; core = +0.01 with the head (noise), **+0.16 without it**. Mechanisms WITHOUT the head: learned restriction maps **-0.17** when replaced by the identity (0.394 -> 0.220; also below the core-off floor 0.237), physical gap -0.03 (0.361), memory +0.02 when removed (0.409), current-only conditioning -0.01 (0.384). WITH the head every mechanism is within seed noise (two-seed REC-on means: full 0.635, identity 0.641, no-Delta 0.635, current-only 0.629). Reading: on this dataset the recurrency head masks the temporal core (it captures exact-triple recurrence at the score level, and the core-off head-only model is already 0.615); when the head is absent the core carries +0.16, and almost all of it is the *learned restriction maps* (sheaf diffusion vs plain diffusion on the current event graph), with a smaller contribution from the physical gap and none from the persistent memory within a 4-epoch budget. Seed 46 of every arm running (results 2026-09-15 evening); icews factorial running (2026-09-16).


### thgl-forum factorial, seeds 43 / 46 (test MRR)
| temporal core \ head | REC on (s43 / s46) | REC off (s43 / s46) |
|---|---|---|
| full | 0.628 / 0.641 (mean 0.635) | 0.394 / 0.401 (mean 0.398) |
| identity | 0.631 / 0.650 (mean 0.641) | 0.220 / 0.333 (mean 0.277) |
| nodelta | 0.648 / 0.623 (mean 0.635) | 0.361 / 0.351 (mean 0.356) |
| nomem | 0.649 / 0.648 (mean 0.649) | 0.409 / 0.383 (mean 0.396) |
| curonly | 0.648 / 0.610 (mean 0.629) | 0.384 / 0.411 (mean 0.397) |
| coreoff | 0.615 / 0.615 (mean 0.615) | 0.237 / 0.220 (mean 0.228) |
Seed-consistent effects (both seeds same sign): REC off -> learned maps 0.398 vs identity 0.277 (**-0.12**; s43 -0.17, s46 -0.07), physical gap 0.356 (**-0.04**; -0.03 / -0.05), core vs core-off **+0.17** (+0.16 / +0.18); memory (0.396) and map conditioning (0.397) mixed-sign = no effect. REC on -> core vs core-off +0.02 (+0.013 / +0.026, within ~2 seed-std); every mechanism mixed-sign = no effect. Head: +0.24 with the core, +0.39 without.

### tkgl-smallpedia / thgl-software ablations at 3 seeds (REC head on; test MRR, mean +/- std)
| dataset | full | identity maps | no Delta_k | current-only maps |
|---|---|---|---|---|
| tkgl-smallpedia | 0.6103 +/- 0.0066 | 0.6133 +/- 0.0063 | 0.6109 +/- 0.0053 | 0.6054 +/- 0.0082 |
| thgl-software | 0.4376 +/- 0.0021 | 0.4392 +/- 0.0030 | 0.4393 +/- 0.0003 | 0.4388 +/- 0.0043 |
All within seed noise with the head on, as on forum and polecat.
