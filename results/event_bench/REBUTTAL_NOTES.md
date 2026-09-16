> **AUDIT CORRECTION (2026-09-04, supersedes the Hits@10 banner):** two further protocol defects were found and fixed. (1) Link tasks: embeddings were scored on the snapshot that contained the events being predicted (both models, training and evaluation); the leak-free predict-from-previous protocol is now the default for retraining and ALL link numbers below are pending re-measurement. (2) Node-property tasks: the TGB label cursor was reset per split, pairing tgbn-trade val/test with the 1987-1990 labels, so the tgbn-trade/genre numbers are invalid pending re-run. See results/analytic/audit/AUDIT.md rows 20-31.

> **METRIC CORRECTION (2026-09-04):** every link-style 'MRR' below produced before this date is HITS@10 (harness took the first key of TGB's {'hits@10','mrr'} result). NDCG numbers unaffected by this particular bug.

# Rebuttal-period findings: technical notes (2026-08-02)

Scope per author decision: EXCLUDES the faithful Sec-3.2 reimplementation and
the native-resolution ("raw data") trade protocol as reported contributions.
Dependencies of each claim on excluded work are marked [DEP].

## 1. Reviewer task (i): no-history-to-sheaf control
Original model (MambaSheafDiffusion), paper protocol (tgbn-trade, window 3,
train cap 2048 edges, d=4, hidden 64, layers 3, temporal_d_model 32, bptt 8,
lr 0.012488, epochs 200, patience 24, official TGB NDCG@10 evaluator,
seeds 43-47), sheaf learner conditioned on current features only vs full
history conditioning:

| variant             | val NDCG        | test NDCG       |
|---------------------|-----------------|-----------------|
| current_only (ctrl) | 0.8087 +/- 0.0415 | 0.8554 +/- 0.0455 |
| history (paper)     | 0.7974 +/- 0.0585 | 0.8345 +/- 0.0684 |

NOTE: under the paper protocol the control is NOT worse (difference within
seed noise; control mean higher). The favorable reversal (history wins 3/3
seeds, 0.8507 vs 0.8368) was obtained ONLY under the uncapped native-
resolution protocol with the faithful model [DEP: both exclusions].
There is no experiment showing history-conditioning helps under the paper's
own protocol with the paper's own model.

## 2. Reviewer task (ii): tkgl-wikidata benchmark (original model)
Scale-matched protocol: 1.23M entities, 19.7M temporal edges, 1192 relations;
train = most recent 2M train edges as yearly snapshots; bptt=1 streaming;
training loss = softplus pairwise ranking vs 32 uniform negatives/positive;
eval = full official val (2.87M) and test (2.88M) events, streaming with
carried state, dst-time-filtered official TGB negatives, official MRR
evaluator; best-tracking-val checkpoint selection; 69,611 parameters;
peak GPU ~7.5 GB; final eval ~27 min.

| run                       | val MRR | test MRR |
|---------------------------|---------|----------|
| lr 1e-3, 3 epochs (orig.) | 0.0563  | 0.0528   |
| lr 3e-3, 4 epochs (new)   | 0.0550  | 0.0525   |
| lr 3e-4, 4 epochs (new)   | 0.0237  | 0.0235   |

The headline number REPRODUCES at a second, independently tuned lr
(0.0525 vs 0.0528) - it is not a fragile single run. Longer training (8
epochs) degrades tracking val (0.0354) - report best-epoch selection as part
of the protocol.

## 3. Additional benchmarks of the ORIGINAL model (new datasets)
All with: event-scoring head, official TGB splits + negative sets, best-
tracking-val checkpointing, gradient clipping (max-norm 1.0), non-finite-
gradient step skip, divergence-guarded checkpoint selection, seed 43
(3-seed finals in progress). Protocol details per dataset:

- tgbn-genre (NDCG@10, 1,505 nodes, 17.9M edges): weekly snapshots
  (time_window 604800 s, matching weekly label cadence), UNCAPPED 12.5M
  train edges, lr 0.005 (tuned; lr 0.0125 gives 0.4029), patience 16.
  Result: val 0.4939 / test 0.4982 (best epoch 12).
- tgbl-wiki (MRR, 9,227 nodes, 157k edges): 600 s snapshot windows,
  uncapped splits, training negatives restricted to the 1000-id destination
  vocabulary (bipartite), lr 3e-3. Result: val 0.0666 / test 0.0735.
  lr 1e-3 and 1e-2 DIVERGE (see section 4).
- tkgl-smallpedia (MRR, 47,433 nodes, 1.1M edges, 566 relations): native
  yearly snapshots, uncapped, lr 3e-3. Result: val 0.0434 / test 0.0299.

[DEP caveat: these use uncapped training and dataset-appropriate windows;
if "improved sampling" is excluded wholesale, these numbers go with it.
The paper-era alternative numbers for tgbl-wiki are unusable - section 4.]

## 4. CRITICAL correction: historical tgbl-wiki results are invalid
All paper-era tgbl-wiki runs beyond ~3 epochs (including the Optuna-tuned
config) diverge to NaN; NaN scores are ranked rank-1 by the TGB evaluator,
pinning val/test MRR at exactly 1.0. Any such number is an artifact and
must not appear in the paper. The only honest paper-era result: val 0.0879 /
test 0.0215 (dense destination-vocab model, first 4096/512/512 edges,
exact-timestamp snapshots, 20 epochs, seed 43). Mitigations now in the
code: finite-guards, gradient clipping, non-finite-grad step skip, and
checkpoint selection requiring finite eval loss.

## 5. Training-stability findings (apply to BOTH models)
- Divergence mechanism (diagnosed step-level): recurrent state grows
  geometrically over long event streams once the learned dynamics leave the
  stable region; the loss stays small (normalized readout) until scores
  overflow -> NaN -> evaluator pins MRR ~1.0. [The Hurwitz-constrained
  generator fix belongs to the faithful model - DEP - but gradient clipping
  + guards suffice to keep the ORIGINAL model stable at tuned lrs.]
- Learning-rate sensitivity is real and per-dataset: the original diverges
  on tgbl-wiki at 1e-3 and 1e-2 but trains at 3e-3.
- Seed variance on tgbn-trade is large (test NDCG 0.72-0.89 across seeds
  43-47 under the paper protocol; +/-0.069 std). Multi-seed reporting is
  necessary; the paper's +/-0.009 is not reproducible as a seed std.
- CUDA torch_sparse.spmm is non-deterministic (atomics); exact
  reproducibility requires CPU or deterministic kernels.

## 6. Paper clarifications catalogued during the period
Delta_k appears in Eq. 11 but is absent from the Eq. 15 selector input;
Eq. 24's h_0 update is a no-op at Delta=0 under ZOH; Eqs. 10/23 use the
Hansen-Gebhart discrete layer map as a continuous ODE field; Table 3's C
readout is never consumed by the implementation; App. B Fubini claim vs
Remark B.3. (Worth addressing as camera-ready clarifications regardless of
what experiments are reported.)

## 7. Numbers pending (3-seed finals, seeds 43/46/47, in flight)
wiki original + faithful, genre original (lr 0.005) + faithful, smallpedia
both, wikidata seed-46 pair. Insert means +/- std when they land.
