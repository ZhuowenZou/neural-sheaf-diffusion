# Review campaign summary (auto-generated 2026-09-22 10:35)

9 finished runs under `results/review_2026_09_22`; arms derived from resolved configs.

## Test MRR by dataset x arm x lr

| dataset   | group   | rec   | arm             |     lr |   n |     mean |           std | values          |
|:----------|:--------|:------|:----------------|-------:|----:|---------:|--------------:|:----------------|
| tgbl-wiki | matched | on    | core-off        | 0.001  |   1 | 0.744018 | nan           | 0.7440          |
| tgbl-wiki | smoke   | on    | attention-gates | 0.0003 |   1 | 0.537425 | nan           | 0.5374          |
| tgbl-wiki | smoke   | on    | core-off        | 0.0003 |   1 | 0.54059  | nan           | 0.5406          |
| tgbl-wiki | smoke   | on    | gru-sheaf       | 0.0003 |   1 | 0.525499 | nan           | 0.5255          |
| tgbl-wiki | smoke   | on    | tsd             | 0.0003 |   2 | 0.526591 |   0.000382704 | 0.5263 / 0.5269 |

## Checkpoint-replay audits (P0)

| run      | dataset         |   seed |   validation_mrr |   test_mrr |   retained_test_mrr |   replay_minus_retained_test |   query_audit_affected_total | query_audit_parity   |
|:---------|:----------------|-------:|-----------------:|-----------:|--------------------:|-----------------------------:|-----------------------------:|:---------------------|
| sp_f_s47 | tkgl-smallpedia |     47 |         0.644656 |   0.610411 |            0.61118  |                 -0.000769224 |                            0 | True                 |
| sw_f_s43 | thgl-software   |     43 |         0.379599 |   0.436019 |            0.436019 |                  1.37467e-07 |                            0 | True                 |
| sw_f_s46 | thgl-software   |     46 |         0.38443  |   0.43997  |            0.439968 |                  2.34667e-06 |                            0 | True                 |

## Score-validity audit (per split)

| group   | run                     | split   |   queries |   queries_affected |   pos_nonfinite |   neg_nan |   neg_posinf |   neg_neginf |   mrr_tgb_raw |   mrr_guarded |   mrr_conservative |   state_nonfinite_snapshots |   rec_nonfinite_snapshots |
|:--------|:------------------------|:--------|----------:|-------------------:|----------------:|----------:|-------------:|-------------:|--------------:|--------------:|-------------------:|----------------------------:|--------------------------:|
| audit   | sp_f_s47                | val     |    162066 |                  0 |               0 |         0 |            0 |            0 |      0.644656 |      0.644656 |           0.644656 |                           0 |                         0 |
| audit   | sp_f_s47                | test    |    163172 |                  0 |               0 |         0 |            0 |            0 |      0.610411 |      0.610411 |           0.610411 |                           0 |                         0 |
| audit   | sw_f_s43                | val     |    223469 |                  0 |               0 |         0 |            0 |            0 |      0.379599 |      0.379599 |           0.379599 |                           0 |                         0 |
| audit   | sw_f_s43                | test    |    223471 |                  0 |               0 |         0 |            0 |            0 |      0.436019 |      0.436019 |           0.436019 |                           0 |                         0 |
| audit   | sw_f_s46                | val     |    223469 |                  0 |               0 |         0 |            0 |            0 |      0.38443  |      0.38443  |           0.38443  |                           0 |                         0 |
| audit   | sw_f_s46                | test    |    223471 |                  0 |               0 |         0 |            0 |            0 |      0.43997  |      0.43997  |           0.43997  |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr1e-3 | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.754794 |      0.754794 |           0.754794 |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr1e-3 | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.744018 |      0.744018 |           0.744018 |                           0 |                         0 |
| smoke   | attention               | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.642936 |      0.642936 |           0.642936 |                           0 |                         0 |
| smoke   | attention               | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.537425 |      0.537425 |           0.537425 |                           0 |                         0 |
| smoke   | coreoff                 | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.641534 |      0.641534 |           0.641534 |                           0 |                         0 |
| smoke   | coreoff                 | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.54059  |      0.54059  |           0.54059  |                           0 |                         0 |
| smoke   | gru                     | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.609767 |      0.609767 |           0.609767 |                           0 |                         0 |
| smoke   | gru                     | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.525499 |      0.525499 |           0.525499 |                           0 |                         0 |
| smoke   | tsd                     | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.627036 |      0.627036 |           0.627036 |                           0 |                         0 |
| smoke   | tsd                     | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.526321 |      0.526321 |           0.526321 |                           0 |                         0 |
| smoke   | tsd_clockdiag           | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.626814 |      0.626814 |           0.626814 |                           0 |                         0 |
| smoke   | tsd_clockdiag           | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.526862 |      0.526862 |           0.526862 |                           0 |                         0 |

## Clock / saturation diagnostics (per run x split x activity class)

| group   | run           | clock   | split        | activity   |       n |   frac_capped |   frac_zero_gap_used |   n_first_update |   n_first_interaction |    mean_dt |   mean_dt_uncapped |   mean_log10_gap_used_pos |   delta_scale |
|:--------|:--------------|:--------|:-------------|:-----------|--------:|--------------:|---------------------:|-----------------:|----------------------:|-----------:|-------------------:|--------------------------:|--------------:|
| audit   | sp_f_s47      | global  | train_replay | endpoint   |  471683 |   0.617275    |          0.00505212  |            29692 |                 31556 | 0.156974   |        23.0432     |                   0       |             1 |
| audit   | sp_f_s47      | global  | train_replay | closure    |  119048 |   0.884761    |          0.00984477  |             1864 |                 15125 | 0.222796   |        37.4596     |                   0       |             1 |
| audit   | sp_f_s47      | global  | val          | endpoint   |   93744 |   0.848577    |          0           |             9339 |                  9339 | 0.214354   |        43.902      |                   0       |             1 |
| audit   | sp_f_s47      | global  | val          | closure    |   10570 |   0.88666     |          0           |                0 |                     0 | 0.222907   |        39.6129     |                   0       |             1 |
| audit   | sp_f_s47      | global  | test         | endpoint   |   95718 |   0.875342    |          0           |             6538 |                  6538 | 0.220999   |        43.2412     |                   0       |             1 |
| audit   | sp_f_s47      | global  | test         | closure    |   11857 |   0.879227    |          0           |                0 |                     0 | 0.221227   |        38.1172     |                   0       |             1 |
| audit   | sw_f_s43      | global  | train_replay | endpoint   | 1303199 |   0.257094    |          0.000412063 |           525811 |                534422 | 0.0806959  |         0.166753   |                   3.55627 |          3600 |
| audit   | sw_f_s43      | global  | train_replay | closure    | 1061697 |   0.00142884  |          0.00109259  |             8611 |                 36755 | 0.00316541 |         0.00331942 |                   3.55626 |          3600 |
| audit   | sw_f_s43      | global  | val          | endpoint   |  279765 |   0.255221    |          0           |            81054 |                 81054 | 0.0796798  |         0.164202   |                   3.5546  |          3600 |
| audit   | sw_f_s43      | global  | val          | closure    |  185742 |   0.00103369  |          0           |                0 |                     0 | 0.0027357  |         0.00283474 |                   3.55419 |          3600 |
| audit   | sw_f_s43      | global  | test         | endpoint   |  270528 |   0.255863    |          0           |            66451 |                 66451 | 0.0793651  |         0.166654   |                   3.55559 |          3600 |
| audit   | sw_f_s43      | global  | test         | closure    |  219058 |   0.000826265 |          0           |                0 |                     0 | 0.00262919 |         0.00270938 |                   3.55507 |          3600 |
| audit   | sw_f_s46      | global  | train_replay | endpoint   | 1303199 |   0.249077    |          0.000412063 |           525811 |                534422 | 0.0822498  |         0.177267   |                   3.55627 |          3600 |
| audit   | sw_f_s46      | global  | train_replay | closure    | 1061697 |   0.00280306  |          0.00109259  |             8611 |                 36755 | 0.0043743  |         0.00504024 |                   3.55626 |          3600 |
| audit   | sw_f_s46      | global  | val          | endpoint   |  279765 |   0.244091    |          0           |            81054 |                 81054 | 0.0809779  |         0.171203   |                   3.5546  |          3600 |
| audit   | sw_f_s46      | global  | val          | closure    |  185742 |   0.00171205  |          0           |                0 |                     0 | 0.0037614  |         0.00411092 |                   3.55419 |          3600 |
| audit   | sw_f_s46      | global  | test         | endpoint   |  270528 |   0.240918    |          0           |            66451 |                 66451 | 0.0801793  |         0.168496   |                   3.55559 |          3600 |
| audit   | sw_f_s46      | global  | test         | closure    |  219058 |   0.00131016  |          0           |                0 |                     0 | 0.00362975 |         0.00389548 |                   3.55507 |          3600 |
| smoke   | tsd_clockdiag | global  | train_replay | endpoint   |   23975 |   0           |          0.00200209  |              873 |                  3072 | 0.0371964  |         0.0371964  |                   2.77756 |           600 |
| smoke   | tsd_clockdiag | global  | train_replay | closure    |  123689 |   0           |          0.00126123  |             2199 |                 37138 | 0.0345548  |         0.0345548  |                   2.77762 |           600 |
| smoke   | tsd_clockdiag | global  | val          | endpoint   |    4606 |   0           |          0           |              265 |                   265 | 0.0368777  |         0.0368777  |                   2.77797 |           600 |
| smoke   | tsd_clockdiag | global  | val          | closure    |   20426 |   0           |          0           |                0 |                     0 | 0.0345546  |         0.0345546  |                   2.778   |           600 |
| smoke   | tsd_clockdiag | global  | test         | endpoint   |    4427 |   0.0112943   |          0           |              384 |                   384 | 0.0392191  |         0.0646758  |                   2.80767 |           600 |
| smoke   | tsd_clockdiag | global  | test         | closure    |   19269 |   0.0101718   |          0           |                0 |                     0 | 0.0367354  |         0.0591281  |                   2.80487 |           600 |

## Not finished

| run                       | state           |   last_epoch |   attempts |
|:--------------------------|:----------------|-------------:|-----------:|
| audit2_sp_f_s47           | crashed         |            0 |          1 |
| audit_forum_f_s43         | running/pending |            0 |          1 |
| audit_forum_f_s46         | running/pending |            0 |          1 |
| audit_forum_f_s47         | running/pending |            0 |          2 |
| audit_icews_eval_s43      | running/pending |            0 |          1 |
| audit_icews_eval_s46      | running/pending |            0 |          2 |
| audit_icews_eval_s47      | running/pending |            0 |          1 |
| audit_polecat_f_s43       | running/pending |            0 |          1 |
| audit_polecat_f_s46       | running/pending |            0 |          2 |
| audit_polecat_f_s47       | running/pending |            0 |          1 |
| audit_sp_f_s43            | running/pending |            0 |          3 |
| audit_sp_f_s46            | running/pending |            0 |          2 |
| audit_sw_f_s47            | running/pending |            0 |          1 |
| audit_wd_f_s43            | running/pending |            0 |          1 |
| audit_wd_f_s46            | running/pending |            0 |          1 |
| audit_wd_f_s47            | running/pending |            0 |          1 |
| audit_wiki_f_s43          | running/pending |            0 |          1 |
| audit_wiki_f_s46          | running/pending |            0 |          1 |
| audit_wiki_f_s47          | running/pending |            0 |          1 |
| forum_attention_rec_s43   | running/pending |            0 |          1 |
| forum_gru_norec_s43       | running/pending |            0 |          1 |
| forum_gru_norec_s46       | running/pending |            0 |          1 |
| forum_gru_norec_s47       | running/pending |            0 |          1 |
| forum_gru_rec_s43         | running/pending |            0 |          1 |
| forum_gru_rec_s46         | running/pending |            0 |          1 |
| forum_gru_rec_s47         | running/pending |            0 |          1 |
| forum_nodeframe_rec_s43   | running/pending |            0 |          1 |
| wiki_attention_s43_lr1e-3 | running/pending |            0 |          1 |
| wiki_attention_s43_lr3e-4 | running/pending |            1 |          1 |
| wiki_coreoff_s43_lr3e-4   | running/pending |            5 |          1 |
| wiki_curonly_s43_lr1e-3   | running/pending |            2 |          1 |
| wiki_curonly_s43_lr3e-4   | running/pending |            0 |          1 |
| wiki_diag_s43_lr1e-3      | running/pending |            1 |          1 |
| wiki_diag_s43_lr3e-4      | running/pending |            1 |          1 |
| wiki_gru_s43_lr1e-3       | running/pending |            2 |          1 |
| wiki_gru_s43_lr3e-4       | running/pending |            2 |          1 |
| wiki_identity_s43_lr1e-3  | running/pending |            0 |          1 |
| wiki_identity_s43_lr3e-4  | running/pending |            2 |          1 |
| wiki_nodeframe_s43_lr1e-3 | running/pending |            0 |          1 |
| wiki_nodeframe_s43_lr3e-4 | running/pending |            2 |          1 |
| wiki_tsd_s43_lr1e-3       | running/pending |            2 |          1 |
| wiki_tsd_s43_lr3e-4       | running/pending |            0 |          1 |

