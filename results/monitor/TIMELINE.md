# Timeline to the deadline (~2026-09-23), written 2026-09-14 07:50

## PRIORITY (user, 2026-09-14 08:30): temporal-graph mechanism attribution first
Chosen dataset: **thgl-forum** — the tkgl/thgl dataset with the largest temporal-to-spatial ratio (2.56M unique timestamps / 152,816 nodes = 16.7; 227 active timestamps and 311 events per node; icews is next at 0.12 / 117 / 706; polecat 0.012 / 6.7 / 47; smallpedia 0.003 / 14 / 46; software 1.0 / 3.4 / 4.4; wikidata 0.002 / 9.5 / 32).
Design (seed 43 first, then seed 46 for the decisive contrasts): core in {full, identity maps, no Delta_k, NO MEMORY (new true ablation), current-only maps, core OFF (layers 0 + no memory)} x head in {REC on, REC off}; TYPE/REL inputs fixed. 8 new arms x ~10 h at 16-22 GB -> all running by tonight as the s43 icews arms and the s46 forum arms finish; results by 2026-09-15 midday; second seed by 2026-09-16 evening. The 24 lower-priority ablation runs (smallpedia/software/wikidata/polecat/icews seeds) are HELD in leakfree2/hold/ and resume after the forum factorial; icews REC-off arms are the next priority if time allows (~22 h each).

## Status 2026-09-15 07:35
* Forum factorial seed 43: 3 of 8 new arms have test numbers, 5 are in their final evaluations (done by ~11:00). Seed 46 of all 8 arms launched 07:26 (2 on GPU 7, 6 on colleague GPUs 3-6 under the 1-h-idle rule); done ~2026-09-15 18:00-20:00.
* icews factorial (8 arms, seed 43) queued 07:35; ~20 h each -> 2026-09-16 morning-noon.
* Held (24 low-priority ablation seeds) resume after the icews factorial if headroom remains; hard stop for new GPU work 2026-09-19.

## GPU work (all queued; nothing else planned)
| job | state | expected finish |
|---|---|---|
| icews ablations x3 (identity / no-Delta / current-only; s43) | epoch 5 of 6 done; final eval ~10-13 h follows in-run | 2026-09-14 late evening to 09-15 ~05:00 |
| forum ablations seed 46 x3 (26 GB each) | running since 09-14 07:44 (gpu0 / gpu7 / gpu7) | ~10-12 h -> 2026-09-14 ~20:00 |
| **added 09-14 08:00 (user: fill GPU 7)** — 24 ablation runs to make the mechanism table >= 2 seeds on every dataset: smallpedia s46/47 (6 x 12 GB, ~1.5 h each), software s46/47 (6 x 10 GB, ~5 h), wikidata s43 (3 x 12 GB, ~11 h), icews s46 (3 x 16 GB, ~20 h), polecat s46 (3 x 14 GB, ~16 h), forum s47 (3 x 26 GB, ~11 h) | queued; the launcher fills GPUs 0/7 as memory frees (2-4 concurrent jobs per card) | ~330 GPU-hours; drains in ~2.5-3 days -> 2026-09-16/17 |
| icews_abl_nodelta (on colleague GPU 2, over budget) | training ends ~09:30; a watcher then stops it and runs the identical final evaluation on 0/7 via --eval-only | ~09-14 evening |
| everything else | COMPLETE (icews/polecat/wikidata/smallpedia/wiki/forum/software 3 seeds; trade 5 seeds; genre 3 seeds; wiki ablations 3 seeds; smallpedia/polecat/software ablations) | - |
GPU budget remaining: ~420 GPU-hours on our two cards, draining by 2026-09-17; 1-2 days of buffer for re-runs; hard stop for new GPU work 2026-09-19 so every number is in the documents by 09-20.

## Writing (no GPU)
| day | item |
|---|---|
| 09-14 | fold icews/forum ablations as they land; final attribution paragraph (done for wiki 3-seed); LEADERBOARD/table/appendix current |
| 09-15 | `results/analytic/DESIGN.md` final summary of the synthetic study (v4 ladder, negative result for variant separation, Delta_k time-scale probe) as App. "analytic" |
| 09-15 | trade/genre framing paragraph (state read-out task: copy-current-period predictor scores NDCG 1.0; persistence 0.85) |
| 09-16/17 | fold the multi-seed ablations into COMPONENT_ATTRIBUTION.md / appendix attribution paragraph as they land (means +/- std, seed-separation test) |
| 09-17 | consistency pass: every number in appendix_faithful_model.tex / results_table_v3.tex traced to results/event_bench/leakfree2/*/results.csv and results/leakfree_nodeprop/*.csv; protocol-correction subsection final |
| 09-18 -> 09-22 | user's paper writing; I stay on monitoring + any re-run requests |
| 09-23 | deadline |
