# Analytic experiment: what the temporal-sheaf dynamic captures

Goal: quantify, explain and demonstrate the advantage of the faithful
temporal-sheaf core (not the decoder components), by matching each of its
three ingredients to a measurable data property and an ablation that removes
exactly that ingredient.

| ingredient of the core                     | data property it targets                     | ablation (one change)                    |
|--------------------------------------------|----------------------------------------------|------------------------------------------|
| orthogonal restriction maps (transport)    | edge/relation-specific geometry              | `sheaf_identity`: R = I (plain diffusion)|
| exact ZOH over physical gap Delta_k        | heterogeneous inter-event gaps               | `no_delta_t`: selector ignores Delta_k   |
| sheaf decoded from memory (history)        | evolving interaction semantics               | `sheaf_conditioning=current_only`        |
Reference models: efficient (released) implementation; EdgeBank / RecB heuristics.

## Arm A - synthetic temporal sheaf process (ground truth, mechanism isolation)
Generator `exp/synthetic_temporal_sheaf.py`:
- N nodes in K communities; latent state x_u(t) in R^d follows dx/dt = A_c x
  (community-specific rotation+decay) -> x_u(t_k) = expm(A_c Delta) x_u(t_{k-1}) + noise
  when u is touched (state evolves ONLY through physical time, so ordering
  alone is insufficient).
- planted transport O_{c(u),c(v)} in O(d) per community pair (the sheaf).
- event generation: at time t (gaps ~ heavy-tailed, bursty), pick source u,
  destination v ~ softmax(-||O_{c(u)c(v)} x_u - x_v||^2 / tau) mixed with a
  recurrence component (repeat a past partner with prob rho).
- knobs: transport (identity vs rotations), dynamics (static vs evolving),
  gaps (uniform vs bursty), rho (recurrence).
Evaluation: MRR against ALL nodes (N small), stratified by gap bucket and
novel-vs-recurrent. Prediction: each ablation loses only in the regime of its
ingredient; the full core's margin over the efficient implementation appears
iff transports are non-trivial and gaps heterogeneous.

## Arm B - tkgl-smallpedia (real data)
Train under one protocol (REL + REC-x + untyped + symmetric; lr 1e-2; 15 ep)
the full core and the three ablations + the efficient implementation, all
checkpointed. Then `exp/analyze_event_model.py` (eval-only) records per test
event: rank, recurrent flag (triple / pair), Delta since the subject's last
event, relation frequency, and the sheaf residual of the positive vs the
mean over negatives. Reports:
1. MRR on NOVEL vs RECURRENT events (backbone contribution beyond memory).
2. MRR vs gap bucket (does accuracy hold over long gaps?).
3. MRR vs relation-frequency bucket (rare-relation transport).
4. Sheaf residual: positives vs negatives, learned maps vs identity.
5. Time-perturbation: rescale all gaps by {0.25, 1, 4} at inference; a
   Delta-aware core degrades, an order-only model is invariant.

## Interim observations (2026-09-04)
- Checkpoint/eval-only path verified: sp_untyped_sym_s43/best.pt reproduces
  val 0.7236 / test 0.7116 exactly. The per-event analysis loop initially
  under-scored (0.601 / 0.639 capped) vs the harness on identical state
  (0.7064 capped) -> analysis-loop discrepancy under diagnosis (residual
  side effect suspected); strata reported so far are provisional.
- Smallpedia core ablations (identity sheaf / no-Delta_k / current-only)
  all TRACK ~0.77 on the 50k-prefix val, i.e. at the full core's level: the
  headline metric is decoder-dominated on this dataset, so the core's
  contribution must be read from the stratified analysis (novel events,
  long gaps, rare relations) and from the synthetic arm, not from top-line MRR.
- Original (efficient) model with the same decoder budget tracks ~0.08.
- 2026-09-04: synthetic runs use a 300-event context graph (full train graph on 300 nodes is near-complete -> ~35k maps decoded per step; runs made no progress in 2 h). Grid relaunched.
- Arbiter: official --eval-only on the 20k test prefix = 0.7064 = harness
  cross-check; analyze_event_model's own loop = 0.6394 (with or without the
  residual diagnostic) -> the discrepancy is inside the analysis loop's
  scoring path; tensor-level bisect running. Per-event strata are on hold
  until the loop reproduces the official number.
- Identity-sheaf ablation (R = I, plain stalk-wise diffusion), official
  eval-only on its checkpoint: val 0.7309 / test 0.7180 vs full core 0.7236 /
  0.7116. On smallpedia's top-line MRR the learned transport is NOT load-
  bearing (decoder-dominated); the sheaf's contribution must be sought in
  the strata (novel events / long gaps / rare relations) and the synthetic arm.
- Full-split analysis (ts=1.0) with in-process cross-check: analysis loop
  0.6011 (own rank == official evaluator) vs harness 0.7116 on the identical
  post-val state -> the gap arises inside the test loop; the tensor bisect
  found only ~1e-3 score differences between paths, so the working
  hypothesis is near-tied candidate scores amplifying tiny numerical
  differences (bisect v3 measures near-ties per query). Strata remain provisional.
- Bisect v2: replaying validation through the harness (scoring) vs advance-only
  makes no difference; the analysis path still scores 0.65/0.63/0.07 on the
  first three test snapshots vs harness 0.7064, with <=1.4e-3 score deltas.
  Near-tie hypothesis (recurrency bonus creates identical top scores; fp32
  noise re-orders them) under test in bisect v3.
- Current-only sheaf ablation (maps decoded from current features, not
  memory), official eval: val 0.7065 / test 0.6812 vs full core 0.7236 /
  0.7116 -> history-conditioning of the sheaf IS load-bearing on smallpedia
  (-0.030 test), in contrast to the transport itself (identity maps +0.006).
- Bisect v3: the TGB evaluator applied to the harness's own captured score
  tensors (with the analysis mask) gives exactly the analysis numbers
  (0.6509/0.6308/0.0680); near-ties are rare (1.4-7 per query). So the
  scores are NOT the difference - the harness's reported 0.7064 must come
  from how it calls/aggregates the evaluator; bisect v4 instruments the
  evaluator calls inside evaluate_model_streaming.
- RESOLVED (2026-09-04): the analysis loop was correct all along. TGB's link
  evaluator returns {'hits@10','mrr'} and the harness took the FIRST value,
  so every benchmark "MRR" was Hits@10. True MRR of the full-core smallpedia
  checkpoint = 0.6011 (full split; the 0.7116 was Hits@10). All strata in
  analysis_* are true-MRR and stand. Harness fixed (by-name selection,
  hits@10 reported separately); all checkpoints being re-scored and the
  un-checkpointed headline configs retrained with checkpoints.
> **AUDIT CORRECTION (2026-09-04, supersedes the Hits@10 banner):** two further protocol defects were found and fixed. (1) Link tasks: embeddings were scored on the snapshot that contained the events being predicted (both models, training and evaluation); the leak-free predict-from-previous protocol is now the default for retraining and ALL link numbers below are pending re-measurement. (2) Node-property tasks: the TGB label cursor was reset per split, pairing tgbn-trade val/test with the 1987-1990 labels (persistence: 0.87 correct vs 0.46 mis-paired; static train-average: 0.65-0.73 correct vs 0.79 mis-paired), so the tgbn-trade/genre numbers are invalid pending re-run. See results/analytic/audit/AUDIT.md rows 20-29.

> **Protocol note (2026-09-04):** every `results/analytic/synth_*` result produced before this date used the leaky same-snapshot protocol and is superseded by `results/analytic/synth_lf/` (leak-free by construction; `--leaky` reproduces the old behaviour for the correction note only).

### Pre-audit per-event analyses on tkgl-smallpedia (leaky protocol + leaky checkpoints; record only, superseded)
| checkpoint | test-time scale | MRR | recurrent-triple MRR | novel-triple MRR | gap q1 / q4 |
|---|---|---|---|---|---|
| current_only (no memory) | 1.0 | 0.5379 | 0.7468 | 0.0116 | 0.599 / 0.540 |
| identity sheaf (old closed-form normalization) | 0.25 / 1.0 / 4.0 | 0.5962 / 0.6209 / 0.6262 | 0.826 / 0.860 / 0.868 | 0.018 | 0.661-0.691 / 0.582-0.608 |
Observations to re-test leak-free: (i) recurrent triples carry the score (0.86 vs 0.018 novel); (ii) the identity-sheaf ablation was NOT worse than the full model under the leaky protocol (0.6209 vs 0.6011), so the sheaf's contribution on smallpedia must be re-established on the leak-free checkpoints with the corrected (builder-normalized) identity ablation; (iii) the gap-quartile gradient (q1 > q4) persists across time scales.
| full champion (REL+REC typed/untyped/sym), leaky checkpoint | 0.25 / 1.0 / 4.0 | 0.5755 / 0.6011 / 0.4780 | 0.799 / 0.835 / 0.664 | 0.013 | 0.636-0.663 / 0.453-0.588 |
Pre-audit time-scale contrast (record only): compressing or stretching the test-period gaps moves the FULL model (0.576 / 0.601 / 0.478) while the identity-sheaf ablation is flat-to-rising (0.596 / 0.621 / 0.626), i.e. the learned sheaf + Δ_k path is what reacts to the physical time axis. Caveat: under the pre-audit Δ units the selector operated near its cap for stretched gaps, so the magnitude of this effect must be re-measured with `--delta-time-scale auto` and leak-free checkpoints (campaign `results/event_bench/leakfree/`, analyses to be re-run with `TSD_SCORE_FROM_PREVIOUS_STATE=1`). Also: the frequency-quartile pattern (rare/q2 relations ~0.65-0.70 vs frequent ~0.53) and gap-quartile gradient are stable across scales.

## Generator v2 (2026-09-05) — the v1 synthetic data had no signal
The v1 generator multiplied every node state by exp(-0.02*dt) at every event without renormalising, so all states decayed to ~0 and the destination kernel became uniform: the generator's OWN oracle (knows x(t) and the transports) scored MRR 0.0205 on the test period (chance 0.0033, popularity 0.023). Every v1 synthetic result (leaky-era or not) is therefore void — no model could have learned more than popularity. v2 keeps states on the unit sphere, sharpens the kernel (tau 0.1) and records oracle ceilings on the last 15% of events:
| regime | dynamics+transport oracle | static-x oracle | no-transport oracle | recurrent frac |
|---|---|---|---|---|
| full (rotations, evolving, bursty) | 0.314 (non-rec 0.432) | 0.041 | 0.055 | 0.304 |
| notrans (identity maps) | 0.320 | 0.051 | 0.320 (= dynamics) | 0.304 |
| static dynamics | 0.360 | 0.159 | 0.064 | 0.304 |
| uniform gaps | 0.320 | 0.042 | 0.068 | 0.296 |
So in the full regime a model must (i) track node state over time and (ii) apply the community transports to reach the 0.31-0.43 ceiling; a static embedding caps at 0.04 and dynamics-without-maps at 0.055. The next generator revision adds a "stale-bucket" oracle (state as of the previous snapshot boundary, i.e. what predict-then-update exposes without extrapolating the gap) and slower rates so that cross-bucket extrapolation via Delta_k is feasible.
Rate probe (full regime, v2 generator; "stale-bucket" = oracle restricted to the state at the previous snapshot boundary, i.e. what predict-then-update exposes without extrapolating the gap):
| rates (rad/unit) | bucket | event-time oracle | stale-bucket oracle | static-x | no-transport |
|---|---|---|---|---|---|
| 0.2-1.0 | 10 | 0.314 | 0.066 | 0.041 | 0.055 |
| 0.03-0.15 | 20 | 0.316 | 0.104 | 0.045 | 0.069 |
| **0.01-0.05** | **20** | 0.314 | 0.169 | 0.041 | 0.062 |
Chosen: rates 0.01-0.05, bucket 20 -> ladder of ceilings: static embedding 0.04 < dynamics without maps 0.06 < last-bucket state without gap extrapolation 0.17 < gap-extrapolating dynamics + transports 0.31 (0.43 on non-recurrent events). Each model variant maps onto one rung: current_only ~ static; identity ~ no-transport; nodelta ~ stale-bucket (order-only recurrence cannot extrapolate the physical gap); full ~ event-time. Data: `results/analytic/synth_v2/synth_{full,notrans,static,uniform}`; runs: `results/analytic/synth_v2/runs/`.

### v2 probes (full regime, faithful full variant, 2 epochs; 2026-09-05)
| protocol / setting | epoch-1 val MRR | epoch-2 val MRR |
|---|---|---|
| leaky (score on own snapshot), lr 1e-3 | 0.065 | 0.055 |
| leak-free, lr 1e-3 (campaign run, 5 epochs) | 0.020 | 0.021 ... 0.023 |
| leak-free, lr 1e-4 | 0.024 | 0.023 |
| static-embedding baseline (no dynamics; 30 epochs, CPU) | test 0.059 (recurrent 0.089, novel 0.016) | |
Reading: under the honest protocol the backbone alone cannot even reach the static-embedding baseline, because the scoring head only sees raw random features for candidates whose state is stale (most of the 300 nodes at any snapshot). Hypothesis under test: EMB-head (embeddings reach the head for every node) restores candidate identity; a recurrency head adds the 0.3-recurrence signal. Probes `probe/full_embhead`, `probe/full_rec` running.
Probe results: recurrency head (leak-free, lr 1e-3) epoch-1 val MRR **0.114** (vs backbone-only 0.020); embeddings-in-head alone 0.020 (no help). Finished pre-fix v2 shards (backbone only): full 0.023, notrans 0.033, uniform 0.029 — popularity level. Conclusion: with hidden latent states the backbone cannot learn the dynamics from events within budget; the analytic campaign moves to **v3**: node attributes = observed x(0) + community one-hot (system identification instead of latent inference), recurrency head shared by all variants, embeddings-in-head, 8 epochs. Backbone signal is read on NON-recurrent (novel) events, bounded by the oracle ladder (static 0.04 / no-transport 0.06 / stale-bucket 0.17 / dynamics+transport 0.31).

### v3 interim (2026-09-05 17:00): shared REC head, observed x0 + community, leak-free
| run | test MRR | novel | recurrent |
|---|---|---|---|
| full / full | 0.116 | 0.011 | 0.183 |
| notrans / full | 0.116 | 0.013 | 0.167 |
| notrans / nodelta | 0.116 | 0.013 | 0.167 |
| uniform / full | 0.112 | 0.012 | 0.177 |
| uniform / identity | 0.108 | 0.012 | 0.171 |
| uniform / nodelta | 0.110 | 0.012 | 0.174 |
Reading so far: the recurrency head accounts for the whole score (recurrent events 0.17-0.18, i.e. above the 0.30-recurrence prior); on NOVEL events every variant sits at 0.011-0.013 — below even the popularity baseline — so the backbone does not recover the rotating latent dynamics from sparse events within 8 epochs, even with x(0) and community observed. Variants are indistinguishable in the evolving regimes. The decisive remaining check is the STATIC-dynamics regime (no rotation; a model that learns the community transports can reach the 0.36 ceiling vs 0.064 without maps): `synth_static_{full,identity,...}` are queued in lane 2. If the static regime also shows no separation, the synthetic arm is reported as a negative result and the analytic claims rest on the real-data ablations.
Static-dynamics regime, first epoch (21:10): `synth_static_full` val MRR **0.287** after one epoch (ceiling 0.36 with transports, 0.064 without) versus ~0.11-0.12 for every variant in the evolving regimes — the backbone DOES learn the planted community transports when the latent states do not rotate. The sheaf contrast now rests on `synth_static_identity` (no maps) and `synth_static_current_only`, queued next. Note: the `original` variant ignores the recurrency flag (no REC head) and is not comparable (0.036-0.041).

### v3 final (2026-09-06 02:20): negative result for the backbone in every regime
| regime / variant | test MRR | novel | recurrent |
|---|---|---|---|
| static / full | 0.269 | 0.032 | 0.290 |
| static / identity | 0.269 | 0.030 | 0.290 |
| static / nodelta | 0.274 | 0.027 | 0.295 |
| static / current_only | 0.275 | 0.031 | 0.296 |
| full / {full, identity, nodelta, current_only} | 0.115-0.116 | 0.011 | 0.181-0.183 |
| full / full at test time-scale 0.25 / 4 | 0.116 / 0.116 | 0.011 | 0.182 / 0.181 |
| notrans / {full, identity, nodelta, current_only} | 0.115-0.116 | 0.012-0.013 | 0.166-0.167 |
| uniform / {full, identity, nodelta, current_only} | 0.108-0.113 | 0.012 | 0.171-0.179 |
Correction to the 21:10 note: the static-regime validation MRR of 0.287 was the OVERALL metric, driven by recurrence (87% of test pairs repeat in that regime); the novel-event MRR is 0.03 for every variant (static-x oracle 0.159, transport ceiling 0.36). Conclusion: with hidden_channels 8 the backbone + head learn nothing beyond recurrence in any regime; the variants are indistinguishable, and the synthetic arm is a NEGATIVE result as configured. Last check before closing the arm: representation width (the pair score must express x_u^T O_ab x_v with a community-pair-dependent map, which an 8-dim bilinear head cannot) — rerunning static full / identity / current_only with hidden 32 (`runs_h32/`).
Width check (static regime, hidden 32, `runs_h32/`): full 0.277 (novel 0.030), identity 0.276 (novel 0.029), current_only 0.275 (novel 0.031) — identical to hidden 8. Representation width is not the bottleneck; the backbone + streaming head do not learn the planted transports on novel events under the honest protocol. **The synthetic arm is closed as a negative result for the backbone** (recurrency head explains the entire score in every regime). Remaining diagnostic: is the transport structure learnable at all by a suitable static scorer trained i.i.d. (tensor-product features x0 (x) community, bilinear head)? If yes, the failure is attributable to the streaming model/head, not the data, and is reported as such.
Learnability check (static regime, i.i.d.-trained bilinear scorer on x0 (x) community tensor features, 40 epochs): test MRR 0.154, **novel 0.028**, recurrent 0.169 — even an unconstrained static scorer cannot recover the planted transports from these events (static-x oracle 0.159). With ~140 events per node of which 87% are repeats, the transport-determined preference is not identifiable from the novel events. So the v3 negative result is a DATA-identifiability limitation, not (only) a model one. Final attempt (v4): an identifiable generator — n=100, k=4, tau 0.03, rho 0.1, 100k events (~900 novel events per node), rates 0.01-0.05, bucket 20 — accepted only if the static scorer reaches the static-x oracle on the static regime; then 5 variants x {static, evolving}.

### v4 generator (identifiable; 2026-09-06): n=100, k=4, tau 0.03, rho 0.1, 100k events, rates 0.01-0.05, bucket 20, per-hit noise 0.002
| regime | dynamics+transport oracle | stale-bucket | static-x | no-transport | static tensor scorer (i.i.d.) |
|---|---|---|---|---|---|
| static | 0.875 | 0.875 | 0.767 | 0.196 | **0.807** (novel-pair 0.138) |
| full (evolving) | 0.821 | 0.461 | 0.178 | 0.186 | 0.200 (novel-pair 0.074) |
| notrans (evolving, identity maps) | 0.823 | 0.632 | 0.179 | 0.823 | - |
The structure is now learnable from events (static scorer reaches 0.81 of the 0.875 ceiling in the static regime) and the evolving regime leaves a ladder for temporal models: 0.20 (static features) < 0.46 (state at the last snapshot boundary) < 0.82 (gap-extrapolated dynamics). Campaign v4: {static, full, notrans} x {full, identity, nodelta, current_only, original}, shared REC head, observed features, hidden 32, leak-free (`results/analytic/synth_v4/runs/`).

### v4 first results (2026-09-06 21:00)
* static / full: **test 0.849** (best val 0.853; oracle ceiling 0.875; static-feature scorer 0.807; no-transport oracle 0.196) — the faithful backbone + head learns the planted community transports from observed features and events. Contrast runs (identity maps, no-Delta, current-only) in progress.
* evolving (full regime) / full: val 0.272 at epoch 7 (ladder: static features 0.18-0.20 < stale-bucket 0.46 < dynamics 0.82) — above the static-feature level, below the last-snapshot-state level; notrans / full: val 0.272 (its stale-bucket ceiling 0.63). Learning is slow but real; variant contrasts pending.

### v4 interim (2026-09-07 07:15)
| regime / variant | status | test MRR (novel-pair / recurrent) or last val |
|---|---|---|
| static / full | done | 0.849 (0.125 / 0.850) |
| static / identity | epoch 4 | val 0.838 |
| static / original | epoch 6 | val 0.774 |
| full (evolving) / full | done | 0.276 (0.022 / 0.277) |
| full / identity | epoch 1 | val 0.269 |
| full / original | epoch 5 | val 0.163 |
| notrans / full | done | 0.290 (0.021 / 0.291) |
| notrans / identity | epoch 2 | val 0.281 |
| notrans / original | epoch 5 | val 0.184 |
Reading: (i) static regime: with observed x0 + community the HEAD learns the transports as a bilinear map, so identity maps do as well as learned maps (0.838 vs 0.853 val) and even the original model reaches 0.77 — the static regime measures the head, not the sheaf; (ii) evolving regime: the faithful backbone (0.276 / 0.290) clearly beats the original model (val 0.163 / 0.184) and the static-feature scorer (0.200), but stays well below the last-snapshot-state ceiling (0.46 / 0.63) — partial state tracking; the decisive contrasts are `nodelta` (Delta_k) and `current_only` (memory), still queued.

### v4 interim (2026-09-08 06:50)
Evolving regimes, done: full 0.276 vs identity 0.277 (rotations); full 0.290 vs identity 0.290 (identity transports); original 0.176 / 0.200. Learned restriction maps add nothing over identity maps in either evolving regime. current_only at epoch 5: val 0.272 / 0.279 (= full's level), so the temporal memory is unlikely to be the source of the faithful-vs-original gap either; nodelta running. Static regime: nodelta val 0.8375, current_only val 0.830 (= full 0.853 within noise).

### v4 near-final (2026-09-09 07:00)
| regime | full | identity | no-Delta | current-only | original | static scorer / last-state ceiling / dynamics ceiling |
|---|---|---|---|---|---|---|
| static | 0.849 | 0.846 | 0.849 | 0.847 | 0.790 | 0.807 / 0.875 / 0.875 |
| evolving, rotations | 0.276 | 0.277 | val 0.272 (ep 6) | 0.276 | 0.176 | 0.200 / 0.46 / 0.82 |
| evolving, identity transports | 0.290 | 0.290 | 0.288 | 0.289 | 0.200 | 0.200 / 0.63 / 0.82 |
Conclusion: on the synthetic benchmark the faithful architecture beats the efficient implementation by ~0.1 in the evolving regimes and ~0.06 in the static one, but the four faithful variants are indistinguishable (within 0.003): learned restriction maps, the physical gap and the persistent memory are not what separates them here; the shared ingredients (sheaf diffusion on the current event graph, the observed features, the recurrency head) carry the gain, and the gap-extrapolation ceiling (0.46 -> 0.82) is not reached by any variant. The real-data ablations on wiki (memory +0.010, maps +0.009, Delta +0.006) are the evidence for the mechanisms; the synthetic result is reported as a bounded negative: the dynamics are learnable in principle (oracles) but not by these models within budget.
* 2026-09-09 17:50: `synth_full_nodelta` = 0.276 (novel 0.020) — identical to full / identity / current-only (0.276-0.277). The v4 evolving-rotations row is complete; only the test-time time-scale runs (ts 0.25 / 4) remain, for the Delta_k-sensitivity read-out.
* 2026-09-11: v4 test-time time-scale (full model, evolving-rotations): x0.25 -> 0.264, x1 -> 0.276, x4 -> 0.281 (novel 0.017 / 0.022 / 0.016). The model reacts to the physical time axis by ~0.01-0.02, i.e. Delta_k is used but weakly; the matching `nodelta` time-scale runs (in progress) give the control.

## Final summary (2026-09-17) — for the paper (LaTeX: results/event_bench/appendix_analytic.tex)
Synthetic (v4, identifiable): oracle ladder in the evolving regime 0.18 (static features) < 0.46 (last-snapshot state) < 0.82 (event-time state with transports); the faithful architecture reaches 0.276-0.290 there and 0.849 of 0.875 in the static regime, beating the efficient implementation by 0.06-0.10 in every regime, but its variants (identity maps / no Delta_k / current-only) are within 0.003 of each other with the recurrency head present, and the test-time gap rescaling (x0.25 / x4) moves the full model by -0.012 / +0.005. Bounded negative for the mechanisms, positive for the architecture. Real data (thgl-forum factorial + per-event analysis) supplies the positive mechanism evidence: without the head the core is +0.15 on every seed and learned restriction maps are the robust mechanism; with the head the core's gain is confined to novel pairs (0.138 vs 0.090) and grows with the inter-event gap. All earlier generator versions (v1-v3) and their results are void (documented above) and must not be cited.
