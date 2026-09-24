# Stress tests towards the ideal hypotheses (E1, E2: exact memories; E3: learned TSD — see ../histgeom_e3_2026_09_24)

## Why H_hist and H_in tied in the first sensor experiment (derivation, checked numerically)
On any eigenmode whose penalty is λ(s), with q(s) = 1 + αλ(s),

  J_t(H) = ∫ q(s) · (Hπ(s) − x(s)/q(s))² dμ_t + const,

so **H_hist is the q-weighted polynomial projection of the pointwise-filtered signal Q⁻¹X, while H_in (Eq. 10)
is its unweighted projection.** They coincide when Q⁻¹X is polynomial or q is constant in time, and separate only
with truncation (small N / rough signal) *and* strong temporal variation of q (large α·Δλ). In the first experiment
tuned α was 0.03–1 and N = 16, hence the tie. Scalar probe (λ: 0 → 2 at τ, truth drawn from the historical prior):
H_hist − H_in = −0.006 / −0.069 / −0.283 at α₀ = 1 / 10 / 100 (N = 4) and −0.001 / −0.002 / −0.004 (N = 16).

## E1 — well-specified strong prior (the Gram memory's ideal case)
Truth H* drawn from the Gaussian prior whose MAP estimate is H_hist (precision (Mα₀/σ²)G + small ridge) on the
sensor sheaf with a 45° re-mount of half the sensors; α tuned per method on held-out draws (it recovers α₀ exactly
for H_hist and H_in in every cell). 50 evaluation draws, paired bootstrap 95% CI, whole window:

| α₀ | N | H_hist | H_in | H_cur | H_hist − H_in [95% CI], draws favouring H_hist |
|---|---|---|---|---|---|
| 1 | 4 / 8 / 16 | 0.330 / 0.306 / 0.303 | 0.330 / 0.306 / 0.303 | 0.356 / 0.338 / 0.333 | −0.0002 / −0.0001 / 0.0000 (n.s.) |
| 10 | 4 | 0.286 | 0.290 | 0.431 | −0.0032 [−0.0049, −0.0014], 36/50 |
| 10 | 16 | 0.221 | 0.222 | 0.345 | −0.0009 [−0.0013, −0.0005], 36/50 |
| 100 | 4 | 0.195 | 0.219 | 0.432 | **−0.0244 [−0.0313, −0.0176], 44/50** |
| 100 | 8 | 0.149 | 0.158 | 0.398 | −0.0091 [−0.0114, −0.0068], 43/50 |
| 100 | 16 | 0.138 | 0.144 | 0.339 | −0.0056 [−0.0072, −0.0042], 45/50 |

**Supported:** where the geometric prior is strong and acts on the whole polynomial trajectory, the joint Gram
solve beats filter-then-compress, by up to 11% relative at N = 4, and both beat current geometry by 0.20–0.25.

## E2 — physically ideal sensor array (perfectly consistent field, every sensor re-mounted by 90°)
Readings are exact global sections (no sensor deviation), so the geometry is maximally informative; low N and low SNR.

| N, SNR | Scenario A: H_hist / H_in / H_cur | H_hist − H_in (A) | Scenario B: H_hist / H_cur |
|---|---|---|---|
| 16, 0 dB | 0.192 / 0.161 / 0.251 | **+0.031 (50/50 favour H_in)** | 0.195 / 0.045 |
| 16, −10 dB | 0.278 / 0.211 / 0.527 | +0.067 (50/50 favour H_in) | 0.504 / 0.140 |
| 8, 0 dB | 0.501 / 0.489 / 0.508 | +0.012 (50/50 favour H_in) | 0.491 / 0.468 |
| 4, 0 dB | 0.787 / 0.782 / 0.788 | +0.005 (50/50 favour H_in) | 0.792 / 0.786 |

The scenario crossover (H_hist/H_in ≫ H_cur in A, reversed in B) is large here (e.g. 0 dB, N = 16: A −0.06,
B +0.15), **but H_in beats H_hist in Scenario A in every cell.** The endpoint (current value) behaves the same way
(0 dB, N = 16: 0.297 vs 0.186).

**Mechanism:** physical consistency holds pointwise in time, so the pointwise-filtered signal is the right local
estimate; the unweighted projection (H_in) approximates it best in the reported L² error, whereas H_hist's q-weighting
spends polynomial accuracy on directions at the times they are penalised (after the re-mount, the old-frame-consistent
direction is penalised 1 + 2α times more) at the expense of the rest.

## Conclusion for the exact memories
The historical Gram memory's advantage is real but conditional: **it is the better memory when the geometric prior
is a prior on the polynomial trajectory as a whole (E1), and the worse one when consistency is a pointwise-in-time
property of the signal (E2)** — the physically natural case. Both always beat current geometry when the history is
physical. The manuscript can state this precisely: the choice between H_hist and H_in is itself semantic (trajectory
vs. pointwise prior), with the q-weighting identity above as the explanation.

Files: `raw_errors.csv.gz`, `tuned_alpha.csv`, `tuned_errors.csv`, `paired_differences.csv`, `alpha_sweep_curves.csv`
(segments pre / post / all / end = last 2% of the window).
