# Work that needs large GPU headroom (updated 2026-09-11)

Placement (2026-09-11, agreed with colleagues): everything is routed to GPUs 0 and 7; GPUs 1-6 are used only after a full hour without any colleague's process on them. Launchers poll every 3 min and pre-reserve memory at start; the daemon shows per-GPU eligibility in STATUS.md.

| priority | job | memory | wall time | why |
|---|---|---|---|---|
| 1 | tkgl-icews full official evaluation — RUNNING on GPU 0 since 09-11 09:56 (18 GB reserved; silent until the FINAL line, ~12-20 h on the shared card) | 16 GB | ~12 h | potential rank 1 of 7: tracking-val 0.274 vs leader val 0.270 |
| 2 | tkgl-polecat seed 47 — RUNNING on GPU 0 since 09-11 09:53 (epoch 2 of 10 at 15:12; ~2.7 h/epoch on the shared card) | 14 GB | ~1 day | third seed for the rank-1 claim (2-seed mean 0.2432 vs TLogic 0.228) |
| 3 | ~~tkgl-wikidata s43~~ DONE 2026-09-10: 0.5371 (EdgeBank 0.535 -> rank 1/3); seeds 46/47 would need 45 GB each, ~10 h + eval |
| 4 | ~~thgl-forum seed 47~~ DONE 2026-09-10: 0.6187 (3-seed mean 0.6293 +/- 0.0113, rank 3/6) |
| 5 | tkgl-icews seeds 46/47 | 45 GB each | ~8 h + 12 h eval each | only if the s43 evaluation confirms rank 1 |
| 6 | polecat ablations (identity, no-Delta, current-only) — QUEUED 09-11 15:25 via .cmd (14 GB each); forum ablations (20 GB train / 60 GB eval) not yet queued | | ~1 day each | analytic evidence on daily / second-resolution knowledge and heterogeneous graphs |
| 7 | genre weekly (fixed label drain, checkpointed) — RUNNING on GPU 7 since 09-11 13:53, ~17 min/epoch, val 0.457 / test 0.446 at epoch 4 | 6 GB | ~10 h | corrected tgbn-genre number (leaderboard NAVIS 0.528, MovingAvg 0.509, TGNv2 0.469) |

Commands (from the repo root, conda env `nsd`):
```
WL=results/event_bench/queues/wait_launch.sh; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; L=results/event_bench/leakfree2
# forum seed 47
$WL 60000 $L/forum_f_s47.log $PY -m exp.run_event_benchmark --dataset thgl-forum --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --train-negatives-per-pos 32 --epochs 4 --patience 3 --min-epochs 2 --track-val-edges 20000 --model faithful --predict-from-previous --save-checkpoint --seed 47 --out $L/forum_f_s47
# polecat ablation example (identity maps)
$WL 40000 $L/polecat_abl_identity.log $PY -m exp.run_event_benchmark --dataset tkgl-polecat --epochs 10 --patience 5 --min-epochs 4 --track-val-edges 30000 --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --predict-from-previous --save-checkpoint --sheaf-identity --seed 43 --out $L/polecat_abl_identity
```
