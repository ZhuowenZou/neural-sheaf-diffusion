# Validating the historical-geometry memory (Appendix B constructions) — sensor array with changing mounts

Code: `exp/histgeom/` (core.py, sweep.py, analyze.py, figure.py, cost.py, exactness.py, test_core.py).
All numbers below come from the CSVs in this directory. Methods are scored on **recovery of the clean history y**
(never on the objective J_t), with α tuned per method, per cell and per scenario on 10 held-out generator draws
(criterion: whole-window error) and evaluated on 50 fresh draws per cell. Paired differences use the same draw for
every method and both scenarios (same noise); 95% intervals are bootstrap (4,000 resamples) over draws.

## Setup (as specified)
- n = 20 sensors, d_s = 2 (one d_s = 3 cell), 4-NN graph; restriction maps R_{e←u} = R_u(s); global sections are the
  physically consistent readings. Laplacian = sheaf Laplacian scaled by 1/max-degree (spectrum in [0,2], so
  I ⪯ K ⪯ (1+2α)I; unlike D^{-1/2}LD^{-1/2}, this keeps consistent readings exactly in ker L — noted deviation).
- Latent field g: 5 random sinusoids per component (frequencies ≤ 4 cycles/window; calibrated *before* comparing
  methods so that noise-free LegS error at N = 16 is ~6%); sensor deviations d_u at 0.2 scale; 400 held samples;
  noise at the stated SNR. Change at τ for a random 50% of sensors, rotation θ.
- **Scenario A (physical re-mount):** readings before τ in the old frame. **Scenario B (calibration correction):**
  readings in the new frame throughout. **Identical prescribed path** (old maps before τ, new after) in both.
- Methods: H_hist via the Gram recursion (Prop. B.6, Eq. 8 verbatim and a quadrature-equivalent form), H_hist via
  operator moments (Prop. B.7) + CG, H_cur = Q(t)⁻¹C(t) (identical to the Eq. 59 jump-corrected recursion, checked),
  H_in (Eq. 10), plain LegS (α = 0), pointwise filter Q(s)⁻¹X(s) without compression (reference).
- Base cell: θ = 45°, 50% re-mounted, τ/t = 0.5, N = 16, SNR 10 dB. Sweeps: θ ∈ {0,…,90}° (at 10 dB **and 0 dB**),
  fraction {0.1, 0.25, 0.5, 1}, τ/t {0.2, 0.5, 0.8}, N {4, 8, 16, 32}, SNR {−10, −5, 0, 10, 20, 30} dB, d_s = 3.

## Exactness (appendix table: `exactness_table.csv`, unit tests: `test_core.py`, 9/9 pass)
- Online recursion vs direct batch solve (per-interval Gauss–Legendre): max relative error ≤ 8e-13 for C and K
  (N = 8…32; jump, drift and many-re-mount paths). Paper Eq. (8) (matrix-exponential log-time form) vs batch:
  ≤ 2e-14. Operator moments (Eq. 56) reconstruct K to ≤ 4e-13. CG on the moments vs dense solve: ≤ 1e-12.
  Eq. (59) H_cur recursion vs the resolvent: ≤ 4e-14. Spectrum of K within [1, 1+2α] in every case.
- Closed forms reproduced exactly (to 1e-14): Eq. (11) H_hist = 4/9 v, H_cur = 2/5 v, H_in = 9/20 v; the two-node
  example (9/13, −√3/13) vs (1/2, 0); the recursion with held L converges to 9/13 at O(1/T).

## Main result — the scenario crossover (`fig_pre_change_vs_rotation_snr0.pdf`, 10 dB version alongside)
Pre-change segment (where historical and current geometry disagree), 0 dB SNR, tuned α, mean over 50 draws:

| θ | scen. | H_hist | H_in | H_cur | LegS | H_hist − H_cur [95% CI], signs |
|---|---|---|---|---|---|---|
| 0° | A = B | 0.1688 | 0.1688 | 0.1688 | 0.2110 | 0.0000 (identical by construction) |
| 10° | A | 0.1695 | 0.1695 | 0.1763 | 0.2115 | −0.0068 [−0.0078, −0.0058], 50/50 favour H_hist |
| 10° | B | 0.1751 | 0.1751 | 0.1686 | 0.2109 | +0.0064 [+0.0055, +0.0073], 49/50 favour H_cur |
| 45° | A | 0.1807 | 0.1807 | 0.2156 | 0.2204 | −0.0349 [−0.0393, −0.0305], 49/50 favour H_hist |
| 45° | B | 0.2024 | 0.2024 | 0.1681 | 0.2106 | +0.0343 [+0.0299, +0.0386], 47/50 favour H_cur |
| 90° | A | 0.2059 | 0.2051 | 0.2397 | 0.2407 | −0.0338 [−0.0379, −0.0293], 49/50 favour H_hist |
| 90° | B | 0.2065 | 0.2065 | 0.1679 | 0.2106 | +0.0386 [+0.0338, +0.0432], 49/50 favour H_cur |

- **H_hist wins Scenario A and H_cur wins Scenario B in 29 of the 32 sensor cells with θ > 0** (every θ at 10 and
  0 dB, every fraction and τ, N = 16 and 32, SNR −10 to 20 dB, d_s = 3, all four perturbations): the paired
  pre-change interval excludes zero in the predicted direction in both scenarios. Exceptions: N = 4 and N = 8
  (A favours H_hist on 50/50 draws, but B is unresolved; compression error dominates there) and 30 dB (no
  difference). All methods coincide at θ = 0.
- **Size depends on how much regularization is worth.** At the 10 dB base cell the gap is ±0.0016–0.0019
  (≈2% relative; tuned α ≈ 0.03–0.1 because compression already averages the noise); at 0 dB it is ±0.034 (17%);
  at −10 dB it is −0.172 (A) and +0.186 (B). At 20–30 dB the methods are indistinguishable. At N = 4–8 compression
  error dominates (0.45–0.77) and the gap is ≤ 0.001.
- **Post-change segment:** the same signs but smaller (e.g. 0 dB, θ = 45°: A −0.009, B +0.013), as expected.

## H_in ≈ H_hist (the pre-registered second outcome)
- H_hist − H_in is ≤ 1e-4 in magnitude in every cell at the tuned α, including the **well-specified Regime 1**
  (θ = 45°: H_hist 0.2950 vs H_in 0.2951 vs H_cur 0.3491; H_hist is Bayes-optimal there up to the ridge, so this is
  a correctness check, not evidence). The single exception is the lowest SNR (−10 dB, Scenario A), where H_in is
  *better* by 0.013 (43/50 draws), and θ = 90° at 0 dB (H_in better by 0.0008).
- α sweeps (`alpha_sweep_curves.csv`) show why: the two are numerically identical for α ≤ 3 and separate only in the
  over-regularised range α ≥ 10, where H_hist shrinks harder (e.g. 0 dB, A, α = 100: 0.328 vs 0.247).
- Write-up per the pre-registered rule: **"observation-time geometry is what matters; the joint Gram solve gives no
  measurable gain over filter-then-compress in these settings (including the one where it is Bayes-optimal)."**
  This remains a validated distinction from current filtering, and it says the Gram state is not worth its cost
  for reconstruction on this task.

## Robustness (Regime 3; 10 dB, θ = 45°; pre-change H_hist − H_cur)
| perturbation of the prescribed path | Scenario A | Scenario B |
|---|---|---|
| noisy maps (≈5° per sensor per epoch) | −0.0016 [−0.0019, −0.0012], 45/50 | +0.0014 [+0.0011, +0.0018], 44/50 |
| switch time misestimated (+0.1 t) | −0.0043 [−0.0049, −0.0036], 48/50 | +0.0017 [+0.0012, +0.0021], 47/50 |
| continuous drift (width 0.3 t) | −0.0012 [−0.0015, −0.0007], 44/50 | +0.0012 [+0.0007, +0.0016], 43/50 |
| five small re-mounts | −0.0015 [−0.0017, −0.0013], 49/50 | +0.0012 [+0.0010, +0.0014], 47/50 |

The crossover survives every perturbation at the base setting (magnitudes as at the unperturbed base, ±0.0012–0.0043).
With a misestimated switch the post-change segment in A reverses (+0.0034), as the wrong geometry is then applied
to post-change data.

## Cost (`cost_benchmark.csv`; single-threaded, T = 100 held intervals)
| p | N | Gram state | moment state (dense / nnz) | Gram recursion | moment recursion | dense solve (peak) | CG on moments (peak, iters α=1/10) |
|---|---|---|---|---|---|---|---|
| 40 | 16 | 3.1 MiB | 0.38 MiB / 6.1k | 5.0 s | 0.72 s | 0.035 s (9.4 MiB) | 0.014–0.026 s (0.33 MiB, 15/33) |
| 80 | 32 | 50 MiB | 3.1 MiB / 43k | 99 s | 3.4 s | 1.3 s (150 MiB) | 0.14–0.31 s (2.6 MiB, 16/43) |
| 150 | 32 | 176 MiB | 10.8 MiB / 112k | 55 s | 2.6 s | 2.1 s (527 MiB) | 0.04–0.12 s (4.8 MiB, 16/45) |

CG iteration counts depend on α only (15–16 at α = 1, 33–46 at α = 10) — as the bound cond(K) ≤ 1 + 2α predicts,
independent of p and N — and CG matches the dense solve to machine precision. H_cur and H_in cost < 0.3 s throughout.

## Not run
- Method 7 (the learned nodewise SSM of Section 4 trained to reconstruct the same history) — not implemented.
- The real-data (NOAA ISD) semi-synthetic variant — not run (no data download in this session).
- Only the paper's d_s-agnostic normalisation choice above was used; a D^{-1/2}LD^{-1/2} variant was not swept.
