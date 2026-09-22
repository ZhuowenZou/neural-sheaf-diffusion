# Campaign agent report

window: 2026-09-21 08:38:20 -> 2026-09-22 08:40:12

## Runs finished during the window
                   run    dataset  seed   head              core  val_mrr  test_mrr  test_hits10         finished
 forum_abl_curonly_s47 thgl-forum    47 REC on current-only maps 0.619579  0.633647     0.706861 2026-09-22 02:51
forum_abl_identity_s47 thgl-forum    47 REC on     identity maps 0.622341  0.632406     0.664629 2026-09-22 02:30
 forum_abl_nodelta_s47 thgl-forum    47 REC on        no Delta_k 0.616365  0.630245     0.682923 2026-09-21 23:39
   forum_abl_nomem_s47 thgl-forum    47 REC on         no memory 0.646772  0.653415     0.706355 2026-09-22 05:01
 forum_coreoff_rec_s47 thgl-forum    47 REC on          core OFF 0.604178  0.616564     0.648852 2026-09-22 02:47

## Still running at shutdown (left to finish; results NOT committed by the agent)
none

## GPU usage during the window (mean MiB per GPU: ours / others)
gpu0: 15900 / 30261
gpu1: 0 / 39441
gpu2: 0 / 59452
gpu3: 11622 / 28234
gpu4: 0 / 48970
gpu5: 18233 / 55780
gpu6: 0 / 59098
gpu7: 38228 / 16843

## Agent log (last 40 lines)
2026-09-22 02:48:04 tick: running [forum_abl_curonly_s47 forum_abl_nomem_s47 ] pending-launchers 2 free-GiB 54 51 22 80 24 4 22 55 
2026-09-22 02:58:08 committed 8 paths (collected 111 finished runs, 2 unfinished; fingerprint 2186501584)
2026-09-22 02:58:10 pushed to origin/new (349ce4c agent: results snapshot 2026-09-22 02:58 (collected )
2026-09-22 02:58:11 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 24 4 22 80 
2026-09-22 03:08:15 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 24 4 22 80 
2026-09-22 03:18:19 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 24 4 22 80 
2026-09-22 03:28:23 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 24 4 22 80 
2026-09-22 03:38:27 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 03:48:30 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 03:58:34 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:08:38 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:18:42 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:28:46 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:38:49 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:48:53 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 04:58:57 tick: running [forum_abl_nomem_s47 ] pending-launchers 1 free-GiB 54 51 22 80 51 4 22 80 
2026-09-22 05:09:01 committed 8 paths (collected 112 finished runs, 1 unfinished; fingerprint 1013548471)
2026-09-22 05:09:03 pushed to origin/new (05072ad agent: results snapshot 2026-09-22 05:09 (collected )
2026-09-22 05:09:04 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 05:19:07 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 05:29:11 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 05:39:14 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 05:49:18 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 05:59:21 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 06:09:25 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 51 26 22 80 
2026-09-22 06:19:28 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 06:29:31 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 06:39:35 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 06:49:38 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 06:59:41 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 07:09:44 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 07:19:48 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 07:29:51 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 07:39:55 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 07:49:58 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 08:00:01 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 08:10:04 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 08:20:08 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 08:30:11 tick: running [] pending-launchers 0 free-GiB 54 51 22 80 80 26 22 80 
2026-09-22 08:40:11 lifetime reached -> shutting down
