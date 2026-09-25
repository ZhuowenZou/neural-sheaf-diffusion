# Clean-environment reproduction of the synthetic reconstruction study — tag synthetic-v3

Run on biaslab0.ics.uci.edu (AMD EPYC 7713 64-Core Processor, 96 worker processes, single-threaded BLAS), 2026-09-24.

Procedure (`bash reproduce/reproduce_synthetic.sh`, one command, from a fresh `git clone` of tag `synthetic-v3`):
1. new conda environment, Python 3.9, **only** `reproduce/requirements-synthetic.txt` installed from PyPI wheels
   (resolved: contourpy==1.3.0, cycler==0.12.1, exceptiongroup==1.3.1, fonttools==4.60.2, importlib_resources==6.5.2, iniconfig==2.1.0, kiwisolver==1.4.7, matplotlib==3.9.2, numpy==2.0.1, packaging==26.3, pandas==2.3.3, pillow==11.3.0, pluggy==1.6.0, pyparsing==3.3.3, pytest==8.3.3, python-dateutil==2.9.0.post0, pytz==2026.4, scipy==1.13.1, six==1.17.0, tabulate==0.9.0, tomli==2.4.1, typing_extensions==4.16.0, tzdata==2026.4, zipp==3.23.1); NumPy/SciPy PyPI wheels (bundled OpenBLAS)
   (the archived records were produced with MKL);
2. fresh clone of the repository at the tag; 3. unit tests; 4. full sweep (2,160 draws) + tuning/analysis;
5. exactness table, cost benchmark, figures; 6. comparison with the records committed at the tag.

Wall time 9 min 27 s. Step log:
```
[1/6] fresh environment in /tmp/claude-45663/-home-zhuowez1-project/4591a0e6-0b6e-4d41-9d9c-ee90f613e306/scratchpad/repro4_1708/run/env
[2/6] clean clone at synthetic-v3 (records from synthetic-v3)
[3/6] unit tests
10 passed in 7.65s
[4/6] sweep + analysis
done: 2160 tasks in 244s (median task 8.4s)
[5/6] exactness, cost, figures
[6/6] compare with archived records
done in 567 s; workdir /tmp/claude-45663/-home-zhuowez1-project/4591a0e6-0b6e-4d41-9d9c-ee90f613e306/scratchpad/repro4_1708/run
```

# Clean-environment reproduction: comparison with archived records

- [PASS] raw per-draw errors: identical row set: 630000 archived rows, 630000 regenerated
- [PASS] raw per-draw errors: values: max abs diff 3.15e-10, max rel diff 3.42e-09
- [PASS] selected regularisation (alpha) per cell/scenario/method: 350/350 identical
- [PASS] tuned_errors.csv: 1050 rows, max abs diff 6.30e-12
- [PASS] paired_differences.csv: 1050 rows, max abs diff 2.21e-11, sign counts identical on all 966 resolved contrasts (|mean| > 1e-9): True; 84 exact ties (|mean| <= 2.1e-13) not sign-compared
- [PASS] Table 15 means (32 entries): all match to 4 decimals
- [PASS] Section 5 paired differences (paper: -0.0349 [-0.0393,-0.0305]; +0.0343 [0.0299,0.0386]): A -0.0349 [-0.0393, -0.0305], B +0.0343 [+0.0299, +0.0386]
- [PASS] Appendix F.2 crossover count (paper: 28 of 31): 28 of 31
- [PASS] Appendix F.3 numerical-equivalence bounds: paper_eq8_vs_batch_C 1.5e-14 (<2.0e-14); paper_eq8_vs_batch_K 2.8e-15 (<2.0e-14); moments_eq56_vs_batch_K 2.5e-13 (<3.9e-13); recursion_vs_batch_C 7.8e-13 (<8.0e-13); recursion_vs_batch_K 3.0e-13 (<8.0e-13); cg_moments_vs_dense_H 1.1e-12 (<1.1e-12); hcur_eq59_vs_resolvent 3.6e-14 (<4.4e-14)
- [PASS] CG iteration counts (cost benchmark): [16, 33, 15, 33, 15, 35, 16, 43, 16, 43, 16, 43, 16, 46, 16, 46, 16, 45]

**ALL CHECKS PASS**

Timing and memory in cost_benchmark.csv are hardware-dependent and are not compared.

