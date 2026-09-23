#!/bin/bash
# Final assembly of the review bundle (results/review_2026_09_22): collect, tables, synthetic analyses,
# provenance manifest at the current commit, then commit + push.  Idempotent.
cd /home/zhuowez1/project/neural-sheaf-diffusion || exit 1
RV=results/review_2026_09_22; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
set -e
$PY -m exp.review.collect_review | tail -1
$PY -m exp.review.make_tex | tail -1
$PY -m exp.review.synthetic_analysis --gen $RV/synthetic/gen_s1 --runs "$RV/synthetic/runs/synth_gen_s1_*" > /dev/null
$PY -m exp.review.synthetic_analysis --gen $RV/synthetic/gen_s1r --runs "$RV/synthetic/runs/synth_gen_s1r_*" --out $RV/synthetic/synthetic_analysis_r > /dev/null
$PY -m exp.review.synthetic_analysis --gen $RV/synthetic/gen_s2 --runs "$RV/synthetic/runs/synth_gen_s2_*" --out $RV/synthetic/synthetic_analysis_s2 > /dev/null
$PY -m exp.review.wiki_strata --runs "$RV/matched/wiki_*_lr1e-3" --out $RV/matched/wiki_strata > /dev/null
git rev-parse HEAD > $RV/CODE_COMMIT.txt
$PY -m exp.run_event_benchmark --help > $RV/run_event_benchmark_help.txt 2>&1
$PY -m exp.review.provenance --out $RV/provenance --datasets tgbn-trade tgbn-genre tgbl-wiki thgl-forum tkgl-icews tkgl-smallpedia tkgl-polecat tkgl-wikidata thgl-software --skip-large-hashes 2>&1 | tail -1
git add -A $RV exp/review && git commit -q -m "review: final assembly $(date '+%F %H:%M') (collect, tables, strata, manifest)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
git push -q origin review-2026-09-22 && echo "pushed $(git rev-parse --short HEAD)"
