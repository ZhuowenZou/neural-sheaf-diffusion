# Rebuttal-period experiments: technical summary

Two experiments were run during the rebuttal period on commit `da51010` (plus the
diffs described below): (A) a paired control ablation isolating the
history-conditioned sheaf interface on `tgbn-trade`, and (B) a scalability
benchmark of the memory-conditioned model on `tkgl-wikidata`, the largest TGB
2.0 temporal knowledge graph.

---

## A. "No-history-to-sheaf" control ablation (tgbn-trade)

**Question answered.** Table 2's SSM-only / sheaf-only ablations remove whole
components; they do not test the paper's *interface* claim (contribution 1-2:
temporal memory selects the spatial operator). This ablation keeps every
component and severs only the memory-to-restriction-map path.

**Design.** New flag `sheaf_conditioning` on `MambaSheafDiffusion`
(`models/mamba_models.py`):

- `history` (default): unchanged full model — restriction maps decoded from the
  persistent node memory `h_k` (verified bitwise-identical to the baseline).
- `current_only` (control): restriction maps decoded, at every diffusion layer,
  from a parallel trajectory seeded by the *current-snapshot node encoding*
  (`node_signal = enc(x_k)`, the same representation the sheaf-only ablation
  diffuses) and advanced by the same diffusion operators. The persistent memory
  still (i) initializes the diffused spatial latent, (ii) feeds the prediction
  head, (iii) receives lagged spatial feedback, and (iv) carries to the next
  event. So: **history shapes the features in both arms; it shapes the spatial
  operator only in the full model.**

**Parameter parity.** Both arms have exactly 157,825 parameters (the control
reuses the existing encoder; no new modules).

**Verified controls** (`exp/test_history_ablation_checks.py`, all pass; run on
CPU because CUDA spmm is non-deterministic and the map-invariance assertions are
bitwise):
1. history: perturbing the previous hidden state changes the decoded maps;
2. current_only: perturbing the previous hidden state leaves maps bitwise
   unchanged;
3. current_only: the hidden state still changes predictions;
4. current_only: lagged spatial feedback cannot reach the maps;
5. sheaf diffusion active in both arms (maps decoded every layer, output
   graph-dependent);
6. identical parameter counts;
7. `sheaf_conditioning=history` reproduces the unmodified baseline bitwise.
Additionally: gradient flow into the sheaf learner confirmed in both modes;
both arms share the identical snapshot bundle, official chronological TGB
masks, and the official `tgb` NDCG@10 evaluator (asserted in the runner).

**Protocol.** Exact Table-2 configuration from `make_trade_baseline_config()`:
snapshot window 3, diffusion layers 3, stalk dim d=4 (Householder-orthogonal
maps), hidden channels 64, temporal d_model 32, Adam lr 0.012488 / wd 1.98e-7,
BPTT 8, eval every 2, patience 24, max 200 epochs, split caps {train 2048,
val/test full}. No tuning of either arm; no seed was dropped or rerun.
Reproduction gate: seed-43 full model reached test NDCG 0.8181 (within the
historical range of the reported 0.812 +- 0.009) before the paired runs.

**Results** (paired seeds 43-47, official test NDCG@10):

| Variant | Validation | Test |
|---|---|---|
| Full (history-conditioned sheaf) | 0.7974 +- 0.0585 | 0.8345 +- 0.0684 |
| Control (current-only sheaf)     | 0.8087 +- 0.0415 | 0.8554 +- 0.0455 |

Per-seed test difference (full - control): seed 43: -0.0603, 44: -0.0259,
45: -0.0725, 46: +0.0400, 47: +0.0143. Mean paired difference **-0.0209**
(std 0.0479); the full model is better on 2/5 seeds.

**Interpretation (suggested rebuttal wording).**
> "A paired control in which restriction maps are decoded from the current
> snapshot only (temporal memory retained everywhere else, identical parameters
> and protocol) matches or exceeds the full model on tgbn-trade (mean paired
> difference -0.021 test NDCG over 5 shared seeds). This experiment does not
> isolate a benefit from memory-conditioned restriction maps on this benchmark.
> This is consistent with our own analysis that tgbn-trade is
> persistence-dominated (Sec. 5); we therefore narrow the claim: the
> history-to-sheaf interface is a well-typed causal mechanism whose empirical
> advantage must be established on tasks with genuine long-range dependencies,
> which we identify as future work."

Do **not** claim statistical significance in either direction from 5 seeds.

**Important side-finding — error bars.** Under the exact Table-2 protocol, the
*unmodified* full model's test NDCG across seeds 43-47 spans 0.72-0.89
(std ~0.068), far wider than the +-0.009 reported in Table 1/2. The reported
std likely reflects a different (narrower) source of variation. Recommend
recomputing the paper's error bars from true multi-seed reruns before the
rebuttal — a reviewer who reruns the pipeline could surface this.

**Artifacts.** `results/history_ablation_tgbn-trade/{results.csv, summary.csv,
paired_differences.csv}` — per-run columns: variant, seed, best_epoch,
validation_ndcg, test_ndcg, runtime_seconds (~360-550 s/run on one A100),
parameter_count, commit_hash, peak_gpu_mem_mb (~450), full config JSON.
Commands:
```
python exp/test_history_ablation_checks.py
TGB_ROOT=$PWD/datasets python exp/run_history_ablation.py \
    --seeds 43 44 45 46 47 --variants history      --out results/history_ablation_tgbn-trade
TGB_ROOT=$PWD/datasets python exp/run_history_ablation.py \
    --seeds 43 44 45 46 47 --variants current_only --out results/history_ablation_tgbn-trade_control
```

---

## B. tkgl-wikidata scalability benchmark (memory-conditioned model)

**Question answered.** Whether the claimed "local, sparse, online" properties
hold at the largest TGB 2.0 scale: tkgl-wikidata has **1,226,440 entities,
19.7M temporal edges** (13.97M / 2.87M / 2.88M train/val/test), **1,192
relations**, yearly timestamps (largest year ~315K events), and 71.9M static
triples (not used; see limitations).

**Model.** `EventTemporalMambaSheafDiffusion`
(`models/sparse_temporal_mamba.py`): identical memory-conditioned dynamics
(nodewise selective-SSM memory -> edgewise sheaf decoding -> local sheaf
diffusion), but snapshot-local computation and bilinear event scoring with
relation embeddings instead of a dense 1.23M-way softmax head (the only viable
output formulation at this scale). 69,611 parameters (d=2, 2 diffusion layers,
hidden channels 8, temporal d_model 64).

**Scale enablement (code changes, each verified by tests and profiling):**
1. **Sparse-gradient diffusion.** `torch.sparse.mm` -> `torch_sparse.spmm` in
   `_apply_local_diffusion`. The former's backward materializes a *dense*
   (n_local x n_local) gradient for the sheaf Laplacian: measured **22.3 GB**
   for a single 30K-edge snapshot (73,622 local rows), i.e. TB-scale for the
   largest wikidata years — guaranteed OOM. After the fix the backward is
   O(nnz): **2.4 GB** on the same snapshot, ~6.5 GB on a 175K-edge year.
2. **Lazy node encoding.** The event model previously encoded all 1.23M nodes
   every snapshot; the encoder is pointwise, so encoding only local nodes and
   queried candidate ids is mathematically identical and removes GB-scale
   per-snapshot activations from the autograd graph.
3. **Row+candidate chunked scoring.** Candidate scoring previously chunked only
   the candidate axis; a full val snapshot (289K events x ~1K filtered
   negatives) would materialize a ~39 GB representation tensor. Scoring is now
   tiled so rows x candidates <= `max_score_elements` (4M default).
4. **One-time negative-sample loading.** The official val/test negative-sample
   pickles are 14 GB each; they are now loaded once per process instead of per
   evaluation call.

**Protocol.** Train on the 2M most-recent train edges (years ~1969-1998, 30
yearly snapshots; recency-dominated suffix, documented deviation from the full
13.97M for compute), BPTT 1, Adam lr 1e-3, 32 uniform negatives per positive,
seed 43. Model selection on a capped validation prefix (100K events) for
tracking only. **Final evaluation is the full official protocol**: streaming
state train -> val -> test, all 2.87M val and 2.88M test events, official
dst-time-filtered negative samples, official TGB MRR evaluator.

**Results.**

| Run | Epochs | Full val MRR | Full test MRR |
|---|---|---|---|
| Primary (seed 43) | 3 | **0.0563** | **0.0528** |
| Rerun, identical config (seed 43) | 8 (best ckpt = ep. 1) | 0.0354 | 0.0479 |

Compute profile: ~10 min/train epoch (2M edges), peak GPU **6.7 GB** training /
**8.7 GB** evaluation on one A100, full val+test evaluation ~26 min, dataset
footprint 42 GB disk.

The rerun documents optimization instability: tracking-val MRR peaked at epoch
1 and degraded (0.033 -> 0.007-0.013) while training loss kept improving —
the pairwise ranking loss overfits the recency slice quickly, and CUDA
nondeterminism yields materially different trajectories between identical
launches. Present the result as a **feasibility/scalability benchmark**, not a
converged leaderboard entry.

**Suggested rebuttal wording.**
> "During the rebuttal period we ran the memory-conditioned model on
> tkgl-wikidata (1.23M entities, 19.7M temporal edges), which required making
> the diffusion backward sparse (O(nnz) instead of a dense quadratic Laplacian
> gradient) and lazily encoding only active/queried nodes. The model trains and
> is evaluated under the full official TGB 2.0 protocol within 7-9 GB of GPU
> memory on a single A100 (~10 min per 2M-edge epoch; full 5.7M-event
> evaluation in ~26 min), demonstrating that the architecture's local/sparse
> design extends to the largest TGB 2.0 TKG. The current accuracy
> (test MRR 0.053 after 3 epochs on a 2M-edge recency subset, single seed) is
> preliminary: training is undertrained and unstable at this scale, and closing
> the gap to heuristic baselines is left to future work."

If you cite leaderboard baselines for tkgl-wikidata in the rebuttal, verify the
current numbers directly from the TGB 2.0 leaderboard before quoting them; do
not quote from memory.

**Known limitations to state proactively:** recency-capped training subset;
single seed; no use of the 71.9M static triples; relation information only in
the scoring head (diffusion is relation-agnostic); no hyperparameter tuning at
this scale; observed run-to-run instability.

**Artifacts.** `results/wikidata_benchmark/` and `results/wikidata_benchmark_8ep/`
(`history.csv`, `results.csv` with commit hash and full config). Command:
```
TGB_ROOT=$PWD/datasets python exp/run_wikidata_benchmark.py \
    --epochs 3 --train-edges-cap 2000000 --out results/wikidata_benchmark
```

---

## C. Recommended additions before submitting the rebuttal

1. **Recompute Table 1/2 error bars** for tgbn-trade from true multi-seed runs;
   the current +-0.009 is inconsistent with observed seed variance (+-0.07).
   This is the single most exposed number in the paper.
2. **Align Sec. 3.2 text with the implementation.** The released code does not
   condition the temporal update on the physical event gap Delta_k (Eqs. 15-16
   ZOH): timestamps enter only snapshot bucketing and label alignment, and the
   selective-SSM step sizes are input-dependent but not gap-dependent. Either
   soften the text ("HiPPO-initialized selective SSM; discretization step
   selected from the input") or note the gap explicitly; a reviewer diffing
   text against code will find this.
3. **Frame the ablation as narrowing, not refuting.** The control result is
   consistent with Table 2 (sheaf-only >> SSM-only) and with the paper's own
   limitations paragraph; the rebuttal can honestly say the new experiment
   sharpens *where* the interface should matter (long-range-dependency
   regimes) rather than contradicting the architecture.
4. **Offer the reproducibility payload**: commit hash, seed lists, per-run
   CSVs, exact commands, and the diagnostic checks — the checklist currently
   answers "No" to open code, and these materials directly counter
   reproducibility objections.
5. **Claim the scalability engineering as a contribution** (sparse O(nnz)
   diffusion backward, lazy encoding, tiled scoring): it substantiates the
   abstract's "local, sparse" claim with measured memory numbers, which no
   current experiment in the paper does.
