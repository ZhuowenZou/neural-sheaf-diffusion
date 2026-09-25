# Synthetic reconstruction study: reproducibility changelog

- **071e1db** — code and records used for Section 5 / Appendix F (produced with NumPy 2.0.1 linked against MKL).
- **synthetic-v1** — first clean-environment reproduction (fresh conda env, pip wheels, OpenBLAS) failed on 7 of
  2,160 draws: `numpy.linalg.eigh` (LAPACK `*syevd`) did not converge on degenerate Gram matrices (drift and
  delayed-switch configurations). Fix: explicit symmetrisation + `scipy.linalg.eigh(driver="evr")`, with a
  regression test on one failing draw. No record changed.
- **synthetic-v2** — the same reproduction then matched every record except the Regime-1 Gaussian-prior check,
  whose truth was sampled as `V (z / sqrt(prec))` with eigenvectors of a highly degenerate matrix; the eigenvector
  basis inside degenerate eigenspaces differs between LAPACK builds, so the same seed gave different draws.
  Fix: the symmetric square root `V diag(prec^-1/2) V^T z` (basis-independent); MKL and OpenBLAS now agree to 5e-13.
  Regime-1 records (cells `R1_theta0`, `R1_theta45`) were regenerated; **all other records are bit-identical to
  071e1db**, and no number quoted in the paper comes from Regime 1. Regenerated Regime-1 pre-change errors
  (θ = 45°): H_hist 0.3069, H_in 0.3069, H_cur 0.3591, LegS 0.3872 (θ = 0°: all compressed methods 0.2977).
- **synthetic-v3** — checker only (no code or record change): the clean reproduction of v2 matched every record
  (raw errors to 3e-10 across MKL/OpenBLAS, paired intervals to 2e-11) except sign counts of 50 exact ties
  (|mean difference| <= 2e-13; methods coincide by construction). Sign counts are now compared on resolved
  contrasts only.
