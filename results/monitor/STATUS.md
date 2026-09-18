# Monitor status (2026-09-18 08:34:48)

## GPU free (GiB) and utilisation
0:14GB(100%) 1:26GB(100%) 2:22GB(100%) 3:24GB(100%) 4:24GB(100%) 5:13GB(100%) 6:5GB(100%) 7:11GB(100%) 
our GPU processes (pid:MiB): 675794:13532MiB 676024:13534MiB 1546691:18276MiB 676355:13530MiB 

## placement policy: ours = GPUs 0 7; colleagues' GPUs become eligible after 3600s without any other user's process
- gpu0: OURS
- gpu1: colleague busy
- gpu2: colleague busy
- gpu3: colleague busy
- gpu4: colleague busy
- gpu5: colleague busy
- gpu6: colleague busy
- gpu7: OURS
compensation: colleagues hold 111532 MiB on our GPUs; we hold 31810 MiB on theirs; budget left 79722 MiB (a job may go to a colleague GPU while it fits in this budget and in that card's free memory)
our launcher claims (pid:gpu/MiB): 

## leakfree2 benchmark runs
- DONE     forum_abl_curonly: FINAL val_mrr=0.6390 test_mrr=0.6475 test_hits10=0.7028 eval_sec=16469.5
- DONE     forum_abl_curonly_s46: FINAL val_mrr=0.5971 test_mrr=0.6099 test_hits10=0.6602 eval_sec=15402.1
- DONE     forum_abl_identity: FINAL val_mrr=0.6208 test_mrr=0.6313 test_hits10=0.6900 eval_sec=15757.5
- DONE     forum_abl_identity_s46: FINAL val_mrr=0.6438 test_mrr=0.6501 test_hits10=0.7030 eval_sec=13233.2
- DONE     forum_abl_nodelta: FINAL val_mrr=0.6411 test_mrr=0.6475 test_hits10=0.7021 eval_sec=14480.7
- DONE     forum_abl_nodelta_s46: FINAL val_mrr=0.6119 test_mrr=0.6233 test_hits10=0.6927 eval_sec=15436.5
- DONE     forum_abl_nomem: FINAL val_mrr=0.6425 test_mrr=0.6491 test_hits10=0.7006 eval_sec=12369.0
- DONE     forum_abl_nomem_s46: FINAL val_mrr=0.6405 test_mrr=0.6478 test_hits10=0.7022 eval_sec=13280.2
- DONE     forum_coreoff_norec: FINAL val_mrr=0.2536 test_mrr=0.2373 test_hits10=0.3462 eval_sec=13282.1
- DONE     forum_coreoff_norec_s46: FINAL val_mrr=0.2360 test_mrr=0.2202 test_hits10=0.3394 eval_sec=13394.2
- DONE     forum_coreoff_norec_s47: FINAL val_mrr=0.2488 test_mrr=0.2320 test_hits10=0.3502 eval_sec=15661.9
- DONE     forum_coreoff_rec: FINAL val_mrr=0.6020 test_mrr=0.6151 test_hits10=0.6475 eval_sec=12356.5
- DONE     forum_coreoff_rec_s46: FINAL val_mrr=0.6024 test_mrr=0.6151 test_hits10=0.6476 eval_sec=13589.2
- DONE     forum_f_s43: FINAL val_mrr=0.6230 test_mrr=0.6280 test_hits10=0.6826 eval_sec=14452.3
- DONE     forum_f_s46: FINAL val_mrr=0.6345 test_mrr=0.6412 test_hits10=0.6944 eval_sec=19345.1
- DONE     forum_f_s47: FINAL val_mrr=0.6019 test_mrr=0.6187 test_hits10=0.6526 eval_sec=14716.3
- DONE     forum_norec_curonly: FINAL val_mrr=0.3910 test_mrr=0.3842 test_hits10=0.5674 eval_sec=14971.1
- DONE     forum_norec_curonly_s46: FINAL val_mrr=0.4163 test_mrr=0.4113 test_hits10=0.5450 eval_sec=13630.3
- DONE     forum_norec_full: FINAL val_mrr=0.4061 test_mrr=0.3940 test_hits10=0.5448 eval_sec=12101.5
- DONE     forum_norec_full_s46: FINAL val_mrr=0.4091 test_mrr=0.4007 test_hits10=0.5359 eval_sec=13410.1
- DONE     forum_norec_full_s47: FINAL val_mrr=0.3554 test_mrr=0.3470 test_hits10=0.5586 eval_sec=15436.6
- DONE     forum_norec_identity: FINAL val_mrr=0.2277 test_mrr=0.2204 test_hits10=0.4598 eval_sec=12034.8
- DONE     forum_norec_identity_s46: FINAL val_mrr=0.3414 test_mrr=0.3325 test_hits10=0.5274 eval_sec=13654.8
- DONE     forum_norec_identity_s47: FINAL val_mrr=0.3536 test_mrr=0.3257 test_hits10=0.5538 eval_sec=14931.7
- CRASHED  forum_norec_nodelta.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.14 GiB. GPU 0 ha
- DONE     forum_norec_nodelta: FINAL val_mrr=0.3654 test_mrr=0.3608 test_hits10=0.5053 eval_sec=13979.9
- DONE     forum_norec_nodelta_s46: FINAL val_mrr=0.3619 test_mrr=0.3514 test_hits10=0.5307 eval_sec=13516.1
- DONE     forum_norec_nodelta_s47: FINAL val_mrr=0.3913 test_mrr=0.3900 test_hits10=0.5559 eval_sec=14536.0
- DONE     forum_norec_nomem: FINAL val_mrr=0.4172 test_mrr=0.4087 test_hits10=0.5558 eval_sec=12361.0
- DONE     forum_norec_nomem_s46: FINAL val_mrr=0.3913 test_mrr=0.3829 test_hits10=0.5354 eval_sec=13478.9
- DONE     icews_abl_curonly: FINAL val_mrr=0.3158 test_mrr=0.3285 test_hits10=0.5268 eval_sec=31006.0
- DONE     icews_abl_identity: FINAL val_mrr=0.3220 test_mrr=0.3369 test_hits10=0.5400 eval_sec=31352.3
- DONE     icews_abl_nodelta_eval: FINAL val_mrr=0.3217 test_mrr=0.3375 test_hits10=0.5405 eval_sec=30072.7
- STOPPED? icews_abl_nodelta: eval_sec': 9897.9, 'peak_gpu_mem_mb': 8031.1} checkpoint saved: results/event_bench/leakfr
- DONE     icews_abl_nomem: FINAL val_mrr=0.3173 test_mrr=0.3295 test_hits10=0.5243 eval_sec=30898.4
- DONE     icews_abl_nomem_s46: FINAL val_mrr=0.3123 test_mrr=0.3269 test_hits10=0.5220 eval_sec=43085.6
- DONE     icews_coreoff_norec: FINAL val_mrr=0.0199 test_mrr=0.0172 test_hits10=0.0386 eval_sec=27904.6
- DONE     icews_coreoff_norec_s46: FINAL val_mrr=0.0244 test_mrr=0.0256 test_hits10=0.0594 eval_sec=37343.9
- DONE     icews_coreoff_rec: FINAL val_mrr=0.2980 test_mrr=0.3191 test_hits10=0.5215 eval_sec=31623.2
- CRASHED  icews_coreoff_rec_s46.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.85 GiB. GPU 0 ha
- RUNNING  icews_coreoff_rec_s46: epoch 6 (patience 3)
- STOPPED? icews_eval_s43.fail1: A_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_seg
- DONE     icews_eval_s43: FINAL val_mrr=0.3235 test_mrr=0.3386 test_hits10=0.5405 eval_sec=48431.1
- DONE     icews_eval_s46: FINAL val_mrr=0.3227 test_mrr=0.3342 test_hits10=0.5288 eval_sec=39969.2
- DONE     icews_eval_s47: FINAL val_mrr=0.3088 test_mrr=0.3326 test_hits10=0.5270 eval_sec=35295.5
- TRAINED  icews_f_s43 (no final evaluation in this run): "best_track_val_mrr":0.273691654937466
- TRAINED  icews_f_s46 (no final evaluation in this run): "best_track_val_mrr":0.27220127267390487
- TRAINED  icews_f_s47 (no final evaluation in this run): "best_track_val_mrr":0.23980839967081943
- DONE     icews_norec_curonly: FINAL val_mrr=0.0253 test_mrr=0.0216 test_hits10=0.0567 eval_sec=30255.4
- STOPPED? icews_norec_curonly_s46.fail1: 28/magma-2.6.1/control/magma_internal.h:115: void magma_queue::setup_ptrArray(): Assertion
- DONE     icews_norec_curonly_s46: FINAL val_mrr=0.0305 test_mrr=0.0279 test_hits10=0.0596 eval_sec=36503.0
- DONE     icews_norec_full: FINAL val_mrr=0.0309 test_mrr=0.0280 test_hits10=0.0647 eval_sec=31366.9
- DONE     icews_norec_full_s46: FINAL val_mrr=0.0268 test_mrr=0.0204 test_hits10=0.0414 eval_sec=36045.2
- DONE     icews_norec_identity: FINAL val_mrr=0.0299 test_mrr=0.0250 test_hits10=0.0599 eval_sec=30654.2
- DONE     icews_norec_identity_s46: FINAL val_mrr=0.0308 test_mrr=0.0267 test_hits10=0.0608 eval_sec=39752.8
- DONE     icews_norec_nodelta: FINAL val_mrr=0.0297 test_mrr=0.0279 test_hits10=0.0705 eval_sec=31210.5
- DONE     icews_norec_nodelta_s46: FINAL val_mrr=0.0275 test_mrr=0.0240 test_hits10=0.0503 eval_sec=42848.8
- DONE     icews_norec_nomem: FINAL val_mrr=0.0304 test_mrr=0.0264 test_hits10=0.0563 eval_sec=30394.2
- DONE     icews_norec_nomem_s46: FINAL val_mrr=0.0230 test_mrr=0.0192 test_hits10=0.0478 eval_sec=38107.0
- DONE     polecat_abl_curonly: FINAL val_mrr=0.2544 test_mrr=0.2476 test_hits10=0.4007 eval_sec=8902.2
- RUNNING  polecat_abl_curonly_s46: epoch': 7, 'train_loss': 0.14372082104306394, 'track_val_mrr': 0.24102
- DONE     polecat_abl_identity: FINAL val_mrr=0.2508 test_mrr=0.2449 test_hits10=0.3971 eval_sec=8913.2
- RUNNING  polecat_abl_identity_s46: epoch': 7, 'train_loss': 0.14496895447435895, 'track_val_mrr': 0.24071
- DONE     polecat_abl_nodelta: FINAL val_mrr=0.2501 test_mrr=0.2441 test_hits10=0.3977 eval_sec=8365.3
- RUNNING  polecat_abl_nodelta_s46: epoch': 7, 'train_loss': 0.14094874445636502, 'track_val_mrr': 0.24564
- DONE     polecat_f_s43: FINAL val_mrr=0.2472 test_mrr=0.2403 test_hits10=0.3942 eval_sec=6834.8
- DONE     polecat_f_s46: FINAL val_mrr=0.2518 test_mrr=0.2461 test_hits10=0.3992 eval_sec=12421.2
- CRASHED  polecat_f_s46.oom: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 152.00 MiB. GPU 0 
- CRASHED  polecat_f_s47.fail1: torch.AcceleratorError: CUDA error: unspecified launch failure
- DONE     polecat_f_s47: FINAL val_mrr=0.2527 test_mrr=0.2462 test_hits10=0.3975 eval_sec=9563.2
- CRASHED  polecat_f_s47.oom: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.58 GiB. GPU 0 ha
- CRASHED  polecat_o_s43: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.70 GiB. GPU 0 ha
- DONE     sp_abl_curonly: FINAL val_mrr=0.6302 test_mrr=0.5959 test_hits10=0.7102 eval_sec=434.6
- CRASHED  sp_abl_curonly_s46.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.10 GiB. GPU 0 ha
- CRASHED  sp_abl_curonly_s46.fail2: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.10 GiB. GPU 0 ha
- DONE     sp_abl_curonly_s46: FINAL val_mrr=0.6414 test_mrr=0.6110 test_hits10=0.7118 eval_sec=445.4
- CRASHED  sp_abl_curonly_s47.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.44 GiB. GPU 0 ha
- DONE     sp_abl_curonly_s47: FINAL val_mrr=0.6430 test_mrr=0.6092 test_hits10=0.7119 eval_sec=432.3
- DONE     sp_abl_identity: FINAL val_mrr=0.6467 test_mrr=0.6182 test_hits10=0.7119 eval_sec=595.0
- DONE     sp_abl_identity_s46: FINAL val_mrr=0.6387 test_mrr=0.6062 test_hits10=0.7120 eval_sec=431.6
- CRASHED  sp_abl_identity_s47.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.33 GiB. GPU 0 ha
- DONE     sp_abl_identity_s47: FINAL val_mrr=0.6457 test_mrr=0.6154 test_hits10=0.7122 eval_sec=432.4
- DONE     sp_abl_nodelta: FINAL val_mrr=0.6453 test_mrr=0.6160 test_hits10=0.7129 eval_sec=447.2
- DONE     sp_abl_nodelta_s46: FINAL val_mrr=0.6386 test_mrr=0.6054 test_hits10=0.7125 eval_sec=460.9
- CRASHED  sp_abl_nodelta_s47.fail1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.66 GiB. GPU 0 ha
- DONE     sp_abl_nodelta_s47: FINAL val_mrr=0.6413 test_mrr=0.6112 test_hits10=0.7139 eval_sec=430.9
- DONE     sp_f_s43: FINAL val_mrr=0.6460 test_mrr=0.6164 test_hits10=0.7146 eval_sec=609.1
- DONE     sp_f_s46: FINAL val_mrr=0.6383 test_mrr=0.6033 test_hits10=0.7132 eval_sec=631.7
- DONE     sp_f_s47: FINAL val_mrr=0.6442 test_mrr=0.6112 test_hits10=0.7107 eval_sec=576.2
- DONE     sp_o_s43: FINAL val_mrr=0.0263 test_mrr=0.0139 test_hits10=0.0207 eval_sec=652.5
- DONE     sw_abl_curonly: FINAL val_mrr=0.3805 test_mrr=0.4341 test_hits10=0.4720 eval_sec=1881.3
- DONE     sw_abl_curonly_s46: FINAL val_mrr=0.3854 test_mrr=0.4425 test_hits10=0.4863 eval_sec=741.0
- DONE     sw_abl_curonly_s47: FINAL val_mrr=0.3830 test_mrr=0.4397 test_hits10=0.4790 eval_sec=752.6
- DONE     sw_abl_identity: FINAL val_mrr=0.3802 test_mrr=0.4357 test_hits10=0.4742 eval_sec=1772.6
- DONE     sw_abl_identity_s46: FINAL val_mrr=0.3837 test_mrr=0.4408 test_hits10=0.4851 eval_sec=801.5
- DONE     sw_abl_identity_s47: FINAL val_mrr=0.3844 test_mrr=0.4410 test_hits10=0.4828 eval_sec=770.3
- DONE     sw_abl_nodelta: FINAL val_mrr=0.3828 test_mrr=0.4392 test_hits10=0.4814 eval_sec=1856.4
- DONE     sw_abl_nodelta_s46: FINAL val_mrr=0.3830 test_mrr=0.4397 test_hits10=0.4814 eval_sec=768.4
- DONE     sw_abl_nodelta_s47: FINAL val_mrr=0.3854 test_mrr=0.4391 test_hits10=0.4826 eval_sec=769.8
- DONE     sw_f_s43: FINAL val_mrr=0.3796 test_mrr=0.4360 test_hits10=0.4736 eval_sec=996.0
- DONE     sw_f_s46: FINAL val_mrr=0.3844 test_mrr=0.4400 test_hits10=0.4796 eval_sec=850.8
- DONE     sw_f_s47: FINAL val_mrr=0.3798 test_mrr=0.4369 test_hits10=0.4744 eval_sec=720.8
- DONE     sw_o_s43: FINAL val_mrr=0.0532 test_mrr=0.0623 test_hits10=0.1280 eval_sec=954.1
- DONE     wd_abl_curonly: FINAL val_mrr=0.6378 test_mrr=0.5381 test_hits10=0.6091 eval_sec=2486.3
- DONE     wd_abl_identity: FINAL val_mrr=0.6421 test_mrr=0.5336 test_hits10=0.5950 eval_sec=2739.5
- DONE     wd_abl_nodelta: FINAL val_mrr=0.6434 test_mrr=0.5362 test_hits10=0.5982 eval_sec=2572.1
- DONE     wd_f_s43: FINAL val_mrr=0.6440 test_mrr=0.5371 test_hits10=0.6018 eval_sec=2615.6
- CRASHED  wd_f_s43.oom: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 h
- DONE     wd_f_s46: FINAL val_mrr=0.6408 test_mrr=0.5500 test_hits10=0.6206 eval_sec=2745.6
- DONE     wd_f_s47: FINAL val_mrr=0.6405 test_mrr=0.5323 test_hits10=0.5952 eval_sec=3026.9
- DONE     wd_o_s43: FINAL val_mrr=0.0259 test_mrr=0.0245 test_hits10=0.0453 eval_sec=2470.4
- DONE     wiki_abl_curonly: FINAL val_mrr=0.7375 test_mrr=0.7253 test_hits10=0.8494 eval_sec=4025.7
- DONE     wiki_abl_curonly_s46: FINAL val_mrr=0.7464 test_mrr=0.7345 test_hits10=0.8507 eval_sec=1182.2
- DONE     wiki_abl_curonly_s47: FINAL val_mrr=0.7463 test_mrr=0.7332 test_hits10=0.8515 eval_sec=1372.5
- DONE     wiki_abl_identity: FINAL val_mrr=0.7402 test_mrr=0.7267 test_hits10=0.8484 eval_sec=4030.4
- DONE     wiki_abl_identity_s46: FINAL val_mrr=0.7441 test_mrr=0.7334 test_hits10=0.8493 eval_sec=1298.5
- DONE     wiki_abl_identity_s47: FINAL val_mrr=0.7412 test_mrr=0.7275 test_hits10=0.8492 eval_sec=1375.2
- DONE     wiki_abl_nodelta: FINAL val_mrr=0.7387 test_mrr=0.7295 test_hits10=0.8499 eval_sec=4044.2
- DONE     wiki_abl_nodelta_s46: FINAL val_mrr=0.7394 test_mrr=0.7246 test_hits10=0.8489 eval_sec=1346.7
- DONE     wiki_abl_nodelta_s47: FINAL val_mrr=0.7370 test_mrr=0.7252 test_hits10=0.8492 eval_sec=1315.7
- DONE     wiki_embhead: FINAL val_mrr=0.7369 test_mrr=0.7145 test_hits10=0.8467 eval_sec=4028.1
- DONE     wiki_f_s43: FINAL val_mrr=0.7521 test_mrr=0.7357 test_hits10=0.8478 eval_sec=1252.7
- DONE     wiki_f_s46: FINAL val_mrr=0.7447 test_mrr=0.7305 test_hits10=0.8499 eval_sec=1281.3
- DONE     wiki_f_s47: FINAL val_mrr=0.7438 test_mrr=0.7331 test_hits10=0.8509 eval_sec=1520.7
- DONE     wiki_o_s43: FINAL val_mrr=0.0201 test_mrr=0.0110 test_hits10=0.0163 eval_sec=971.4

## node-property reruns
- genre_faithful_daily_s43: test NDCG mean±std: 0.4482 ± nan
- genre_faithful_daily_s46: test NDCG mean±std: 0.4452 ± nan
- genre_faithful_daily_s47: test NDCG mean±std: 0.4502 ± nan
- genre_faithful_s43_e40.misconfig: 
- genre_faithful_s43_e40.oldfail1: 
- genre_faithful_s43_e40.oldfail2: 
- genre_faithful_s43_e40.oldfail4: 
- genre_faithful_s43_e40.prefix: 
- genre_faithful_s43: 
- genre_faithful_weekly_s43: test NDCG mean±std: 0.4617 ± nan
- genre_original_s43.stopped: 
- trade_faithful: test NDCG mean±std: 0.8210 ± 0.0108
- trade_faithful_paper: test NDCG mean±std: 0.3506 ± 0.0119
- trade_faithful_uncapped.prefix: 
- trade_faithful_w1: test NDCG mean±std: 0.6636 ± 0.0061
- trade_faithful_w1.oldfail1: 
- trade_faithful_w1.oldfail2: 
- trade_faithful_w1.oldfail3: 
- trade_faithful_w1.oldfail4: 
- trade_faithful_w1.oldfail5: 
- trade_original: test NDCG mean±std: 0.8340 ± 0.0704

## synthetic v4
- DONE synth_full_current_only: test 0.276 novel 0.022 rec 0.277
- DONE synth_full_full: test 0.276 novel 0.022 rec 0.277
- DONE synth_full_full_ts0.25: test 0.264 novel 0.017 rec 0.265
- DONE synth_full_full_ts4: test 0.281 novel 0.016 rec 0.282
- DONE synth_full_identity: test 0.277 novel 0.022 rec 0.278
- DONE synth_full_nodelta: test 0.276 novel 0.020 rec 0.277
- synth_full_nodelta_ts0.25.fail1: epoch 3 loss 0.4472 val_mrr 0.2701 (6848
- DONE synth_full_nodelta_ts0.25: test 0.266 novel 0.016 rec 0.267
- synth_full_nodelta_ts0.25.moved: epoch 2 loss 0.4473 val_mrr 0.2699 (6136
- synth_full_nodelta_ts4.fail1: epoch 1 loss 0.4510 val_mrr 0.2694 (6852
- DONE synth_full_nodelta_ts4: test 0.281 novel 0.016 rec 0.282
- DONE synth_full_original: test 0.176 novel 0.029 rec 0.176
- DONE synth_notrans_current_only: test 0.289 novel 0.021 rec 0.290
- DONE synth_notrans_full: test 0.290 novel 0.021 rec 0.291
- DONE synth_notrans_identity: test 0.290 novel 0.021 rec 0.292
- DONE synth_notrans_nodelta: test 0.288 novel 0.021 rec 0.290
- DONE synth_notrans_original: test 0.200 novel 0.026 rec 0.201
- DONE synth_static_current_only: test 0.847 novel 0.128 rec 0.848
- DONE synth_static_full: test 0.849 novel 0.125 rec 0.850
- DONE synth_static_identity: test 0.846 novel 0.127 rec 0.847
- DONE synth_static_nodelta: test 0.849 novel 0.126 rec 0.850
- DONE synth_static_original: test 0.790 novel 0.151 rec 0.791

## recent actions
2026-09-17 13:59:53 (re)launching polecat_abl_identity_s46 via results/event_bench/leakfree2/polecat_abl_identity_s46.cmd (attempt 1)
2026-09-17 13:59:53 (re)launching polecat_abl_nodelta_s46 via results/event_bench/leakfree2/polecat_abl_nodelta_s46.cmd (attempt 1)
2026-09-17 13:59:53 (re)launching wd_abl_curonly via results/event_bench/leakfree2/wd_abl_curonly.cmd (attempt 1)
2026-09-17 13:59:53 (re)launching wd_abl_identity via results/event_bench/leakfree2/wd_abl_identity.cmd (attempt 1)
2026-09-17 13:59:53 (re)launching wd_abl_nodelta via results/event_bench/leakfree2/wd_abl_nodelta.cmd (attempt 1)
