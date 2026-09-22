# Review campaign summary (auto-generated 2026-09-22 15:35)

42 finished runs under `results/review_2026_09_22`; arms derived from resolved configs.

## Test MRR by dataset x arm x lr

| dataset                                                            | group   | rec   | arm               |     lr |   n |     mean |           std | values                            |
|:-------------------------------------------------------------------|:--------|:------|:------------------|-------:|----:|---------:|--------------:|:----------------------------------|
| synth-history:results/review_2026_09_22/synthetic/gen_s1r/data.npz | smoke   | off   | tsd               | 0.001  |   1 | 0.215083 | nan           | 0.2151                            |
| tgbl-wiki                                                          | matched | on    | core-off          | 0.0003 |   1 | 0.725545 | nan           | 0.7255                            |
| tgbl-wiki                                                          | matched | on    | core-off          | 0.001  |   1 | 0.744018 | nan           | 0.7440                            |
| tgbl-wiki                                                          | matched | on    | current-only-maps | 0.001  |   1 | 0.765613 | nan           | 0.7656                            |
| tgbl-wiki                                                          | matched | on    | diagssm-ordinary  | 0.001  |   1 | 0.765891 | nan           | 0.7659                            |
| tgbl-wiki                                                          | matched | on    | gru-ordinary      | 0.0003 |   1 | 0.733609 | nan           | 0.7336                            |
| tgbl-wiki                                                          | matched | on    | node-frame        | 0.0003 |   1 | 0.731724 | nan           | 0.7317                            |
| tgbl-wiki                                                          | smoke   | on    | attention-gates   | 0.0003 |   1 | 0.537425 | nan           | 0.5374                            |
| tgbl-wiki                                                          | smoke   | on    | core-off          | 0.0003 |   1 | 0.54059  | nan           | 0.5406                            |
| tgbl-wiki                                                          | smoke   | on    | gru-sheaf         | 0.0003 |   1 | 0.525499 | nan           | 0.5255                            |
| tgbl-wiki                                                          | smoke   | on    | tsd               | 0.0003 |   2 | 0.526591 |   0.000382704 | 0.5263 / 0.5269                   |
| thgl-forum                                                         | forum   | off   | attention-gates   | 0.001  |   1 | 0.347949 | nan           | 0.3479                            |
| thgl-forum                                                         | forum   | off   | core-off          | 0.001  |   1 | 0.22207  | nan           | 0.2221                            |
| thgl-forum                                                         | forum   | off   | diagssm-ordinary  | 0.001  |   2 | 0.337591 |   0.0916383   | 0.2728 / 0.4024                   |
| thgl-forum                                                         | forum   | off   | gru-ordinary      | 0.001  |   4 | 0.387535 |   0.00680891  | 0.3827 / 0.3932 / 0.3807 / 0.3936 |
| thgl-forum                                                         | forum   | off   | tsd               | 0.001  |   1 | 0.232276 | nan           | 0.2323                            |
| thgl-forum                                                         | forum   | on    | attention-gates   | 0.001  |   1 | 0.616149 | nan           | 0.6161                            |
| thgl-forum                                                         | forum   | on    | diagssm-ordinary  | 0.001  |   1 | 0.650478 | nan           | 0.6505                            |
| thgl-forum                                                         | forum   | on    | gru-ordinary      | 0.001  |   3 | 0.644765 |   0.00545284  | 0.6511 / 0.6418 / 0.6414          |
| thgl-forum                                                         | forum   | on    | node-frame        | 0.001  |   1 | 0.633638 | nan           | 0.6336                            |

## Paired contrasts (arm minus TSD, per-seed)

| dataset    | rec   | arm          |   arm_lr |   tsd_lr |   n |   tsd_mean |   arm_mean |   delta_arm_minus_tsd_mean |   delta_sd |   t95_halfwidth |   deltas | signs   |
|:-----------|:------|:-------------|---------:|---------:|----:|-----------:|-----------:|---------------------------:|-----------:|----------------:|---------:|:--------|
| thgl-forum | off   | core-off     |    0.001 |    0.001 |   1 |   0.232276 |   0.22207  |                 -0.0102063 |        nan |             nan |  -0.0102 | 0+ 1-   |
| thgl-forum | off   | gru-ordinary |    0.001 |    0.001 |   1 |   0.232276 |   0.393608 |                  0.161332  |        nan |             nan |   0.1613 | 1+ 0-   |

## Checkpoint-replay audits (P0)

| run              | dataset         |   seed |   validation_mrr |   test_mrr |   retained_test_mrr |   replay_minus_retained_test |   query_audit_affected_total | query_audit_parity   |
|:-----------------|:----------------|-------:|-----------------:|-----------:|--------------------:|-----------------------------:|-----------------------------:|:---------------------|
| forum_f_s47      | thgl-forum      |     47 |         0.601889 |   0.618727 |            0.618727 |                 -2.02006e-08 |                            0 | True                 |
| polecat_f_s43    | tkgl-polecat    |     43 |         0.247229 |   0.2403   |            0.2403   |                 -1.77451e-08 |                            0 | True                 |
| sp_f_s43         | tkgl-smallpedia |     43 |         0.646006 |   0.61644  |            0.616444 |                 -3.48359e-06 |                            0 | True                 |
| sp_f_s46         | tkgl-smallpedia |     46 |         0.638303 |   0.603251 |            0.603251 |                  0           |                            0 | True                 |
| sp_f_s47         | tkgl-smallpedia |     47 |         0.644656 |   0.610411 |            0.61118  |                 -0.000769224 |                            0 | True                 |
| sp_f_s47_replay2 | tkgl-smallpedia |     47 |         0.644553 |   0.610309 |          nan        |                nan           |                            0 | True                 |
| sw_f_s43         | thgl-software   |     43 |         0.379599 |   0.436019 |            0.436019 |                  1.37467e-07 |                            0 | True                 |
| sw_f_s46         | thgl-software   |     46 |         0.38443  |   0.43997  |            0.439968 |                  2.34667e-06 |                            0 | True                 |
| sw_f_s47         | thgl-software   |     47 |         0.379806 |   0.436862 |            0.436862 |                  5.24216e-09 |                            0 | True                 |
| wd_f_s43         | tkgl-wikidata   |     43 |         0.644031 |   0.537079 |            0.537079 |                 -2.47224e-07 |                            0 | True                 |
| wd_f_s46         | tkgl-wikidata   |     46 |         0.640765 |   0.550037 |            0.550037 |                 -3.99954e-07 |                            0 | True                 |
| wd_f_s47         | tkgl-wikidata   |     47 |         0.640469 |   0.532305 |            0.532306 |                 -4.82167e-07 |                            0 | True                 |
| wiki_f_s43       | tgbl-wiki       |     43 |         0.752119 |   0.735701 |            0.735701 |                  4.69272e-08 |                            0 | True                 |
| wiki_f_s46       | tgbl-wiki       |     46 |         0.744714 |   0.730548 |            0.730525 |                  2.31301e-05 |                            0 | True                 |
| wiki_f_s47       | tgbl-wiki       |     47 |         0.743804 |   0.733075 |            0.733075 |                  6.81311e-10 |                            0 | True                 |

## Score-validity audit (per split)

| group   | run                        | split   |   queries |   queries_affected |   pos_nonfinite |   neg_nan |   neg_posinf |   neg_neginf |   mrr_tgb_raw |   mrr_guarded |   mrr_conservative |   state_nonfinite_snapshots |   rec_nonfinite_snapshots |
|:--------|:---------------------------|:--------|----------:|-------------------:|----------------:|----------:|-------------:|-------------:|--------------:|--------------:|-------------------:|----------------------------:|--------------------------:|
| audit   | forum_f_s47                | val     |   3563658 |                  0 |               0 |         0 |            0 |            0 |      0.601889 |      0.601889 |           0.601889 |                           0 |                         0 |
| audit   | forum_f_s47                | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.618727 |      0.618727 |           0.618727 |                           0 |                         0 |
| audit   | polecat_f_s43              | val     |    533472 |                  0 |               0 |         0 |            0 |            0 |      0.247229 |      0.247229 |           0.247229 |                           0 |                         0 |
| audit   | polecat_f_s43              | test    |    532636 |                  0 |               0 |         0 |            0 |            0 |      0.2403   |      0.2403   |           0.2403   |                           0 |                         0 |
| audit   | sp_f_s43                   | val     |    162066 |                  0 |               0 |         0 |            0 |            0 |      0.646006 |      0.646006 |           0.646006 |                           0 |                         0 |
| audit   | sp_f_s43                   | test    |    163172 |                  0 |               0 |         0 |            0 |            0 |      0.61644  |      0.61644  |           0.61644  |                           0 |                         0 |
| audit   | sp_f_s46                   | val     |    162066 |                  0 |               0 |         0 |            0 |            0 |      0.638303 |      0.638303 |           0.638303 |                           0 |                         0 |
| audit   | sp_f_s46                   | test    |    163172 |                  0 |               0 |         0 |            0 |            0 |      0.603251 |      0.603251 |           0.603251 |                           0 |                         0 |
| audit   | sp_f_s47                   | val     |    162066 |                  0 |               0 |         0 |            0 |            0 |      0.644656 |      0.644656 |           0.644656 |                           0 |                         0 |
| audit   | sp_f_s47                   | test    |    163172 |                  0 |               0 |         0 |            0 |            0 |      0.610411 |      0.610411 |           0.610411 |                           0 |                         0 |
| audit   | sp_f_s47_replay2           | val     |    162066 |                  0 |               0 |         0 |            0 |            0 |      0.644553 |      0.644553 |           0.644553 |                           0 |                         0 |
| audit   | sp_f_s47_replay2           | test    |    163172 |                  0 |               0 |         0 |            0 |            0 |      0.610309 |      0.610309 |           0.610309 |                           0 |                         0 |
| audit   | sw_f_s43                   | val     |    223469 |                  0 |               0 |         0 |            0 |            0 |      0.379599 |      0.379599 |           0.379599 |                           0 |                         0 |
| audit   | sw_f_s43                   | test    |    223471 |                  0 |               0 |         0 |            0 |            0 |      0.436019 |      0.436019 |           0.436019 |                           0 |                         0 |
| audit   | sw_f_s46                   | val     |    223469 |                  0 |               0 |         0 |            0 |            0 |      0.38443  |      0.38443  |           0.38443  |                           0 |                         0 |
| audit   | sw_f_s46                   | test    |    223471 |                  0 |               0 |         0 |            0 |            0 |      0.43997  |      0.43997  |           0.43997  |                           0 |                         0 |
| audit   | sw_f_s47                   | val     |    223469 |                  0 |               0 |         0 |            0 |            0 |      0.379806 |      0.379806 |           0.379806 |                           0 |                         0 |
| audit   | sw_f_s47                   | test    |    223471 |                  0 |               0 |         0 |            0 |            0 |      0.436862 |      0.436862 |           0.436862 |                           0 |                         0 |
| audit   | wd_f_s43                   | val     |   2869900 |                  0 |               0 |         0 |            0 |            0 |      0.644031 |      0.644031 |           0.644031 |                           0 |                         0 |
| audit   | wd_f_s43                   | test    |   2877500 |                  0 |               0 |         0 |            0 |            0 |      0.537079 |      0.537079 |           0.537079 |                           0 |                         0 |
| audit   | wd_f_s46                   | val     |   2869900 |                  0 |               0 |         0 |            0 |            0 |      0.640765 |      0.640765 |           0.640765 |                           0 |                         0 |
| audit   | wd_f_s46                   | test    |   2877500 |                  0 |               0 |         0 |            0 |            0 |      0.550037 |      0.550037 |           0.550037 |                           0 |                         0 |
| audit   | wd_f_s47                   | val     |   2869900 |                  0 |               0 |         0 |            0 |            0 |      0.640469 |      0.640469 |           0.640469 |                           0 |                         0 |
| audit   | wd_f_s47                   | test    |   2877500 |                  0 |               0 |         0 |            0 |            0 |      0.532305 |      0.532305 |           0.532305 |                           0 |                         0 |
| audit   | wiki_f_s43                 | val     |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.752119 |      0.752119 |           0.752119 |                           0 |                         0 |
| audit   | wiki_f_s43                 | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.735701 |      0.735701 |           0.735701 |                           0 |                         0 |
| audit   | wiki_f_s46                 | val     |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.744714 |      0.744714 |           0.744714 |                           0 |                         0 |
| audit   | wiki_f_s46                 | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.730548 |      0.730548 |           0.730548 |                           0 |                         0 |
| audit   | wiki_f_s47                 | val     |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.743804 |      0.743804 |           0.743804 |                           0 |                         0 |
| audit   | wiki_f_s47                 | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.733075 |      0.733075 |           0.733075 |                           0 |                         0 |
| forum   | forum_attention_rec_s43    | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.603228 |      0.603228 |           0.603228 |                           0 |                         0 |
| forum   | forum_attention_rec_s43    | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.616149 |      0.616149 |           0.616149 |                           0 |                         0 |
| forum   | forum_attention_recoff_s46 | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.347924 |      0.347924 |           0.347924 |                           0 |                         0 |
| forum   | forum_attention_recoff_s46 | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.347949 |      0.347949 |           0.347949 |                           0 |                         0 |
| forum   | forum_attention_recon_s46  | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.626114 |      0.626114 |           0.626114 |                           0 |                         0 |
| forum   | forum_attention_recon_s46  | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.63677  |      0.63677  |           0.63677  |                           0 |                         0 |
| forum   | forum_coreoff_recoff_s44   | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.239776 |      0.239776 |           0.239776 |                           0 |                         0 |
| forum   | forum_coreoff_recoff_s44   | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.22207  |      0.22207  |           0.22207  |                           0 |                         0 |
| forum   | forum_diag_recoff_s43      | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.284913 |      0.284913 |           0.284913 |                           0 |                         0 |
| forum   | forum_diag_recoff_s43      | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.272793 |      0.272793 |           0.272793 |                           0 |                         0 |
| forum   | forum_diag_recoff_s47      | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.409269 |      0.409269 |           0.409269 |                           0 |                         0 |
| forum   | forum_diag_recoff_s47      | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.402389 |      0.402389 |           0.402389 |                           0 |                         0 |
| forum   | forum_diag_recon_s47       | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.640758 |      0.640758 |           0.640758 |                           0 |                         0 |
| forum   | forum_diag_recon_s47       | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.650478 |      0.650478 |           0.650478 |                           0 |                         0 |
| forum   | forum_gru_norec_s43        | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.384615 |      0.384615 |           0.384615 |                           0 |                         0 |
| forum   | forum_gru_norec_s43        | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.382652 |      0.382652 |           0.382652 |                           0 |                         0 |
| forum   | forum_gru_norec_s46        | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.397212 |      0.397212 |           0.397212 |                           0 |                         0 |
| forum   | forum_gru_norec_s46        | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.393172 |      0.393172 |           0.393172 |                           0 |                         0 |
| forum   | forum_gru_norec_s47        | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.382967 |      0.382967 |           0.382967 |                           0 |                         0 |
| forum   | forum_gru_norec_s47        | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.38071  |      0.38071  |           0.38071  |                           0 |                         0 |
| forum   | forum_gru_rec_s43          | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.640973 |      0.640973 |           0.640973 |                           0 |                         0 |
| forum   | forum_gru_rec_s43          | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.651057 |      0.651057 |           0.651057 |                           0 |                         0 |
| forum   | forum_gru_rec_s46          | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.634851 |      0.634851 |           0.634851 |                           0 |                         0 |
| forum   | forum_gru_rec_s46          | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.641819 |      0.641819 |           0.641819 |                           0 |                         0 |
| forum   | forum_gru_rec_s47          | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.630974 |      0.630974 |           0.630974 |                           0 |                         0 |
| forum   | forum_gru_rec_s47          | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.64142  |      0.64142  |           0.64142  |                           0 |                         0 |
| forum   | forum_gru_recoff_s44       | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.400271 |      0.400271 |           0.400271 |                           0 |                         0 |
| forum   | forum_gru_recoff_s44       | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.393608 |      0.393608 |           0.393608 |                           0 |                         0 |
| forum   | forum_nodeframe_rec_s43    | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.624445 |      0.624445 |           0.624445 |                           0 |                         0 |
| forum   | forum_nodeframe_rec_s43    | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.633638 |      0.633638 |           0.633638 |                           0 |                         0 |
| forum   | forum_nodeframe_recoff_s43 | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.402712 |      0.402712 |           0.402712 |                           0 |                         0 |
| forum   | forum_nodeframe_recoff_s43 | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.397977 |      0.397977 |           0.397977 |                           0 |                         0 |
| forum   | forum_tsd_recoff_s44       | val     |   3643658 |                  0 |               0 |         0 |            0 |            0 |      0.228773 |      0.228773 |           0.228773 |                           0 |                         0 |
| forum   | forum_tsd_recoff_s44       | test    |   3563653 |                  0 |               0 |         0 |            0 |            0 |      0.232276 |      0.232276 |           0.232276 |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr1e-3    | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.754794 |      0.754794 |           0.754794 |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr1e-3    | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.744018 |      0.744018 |           0.744018 |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr3e-4    | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.726325 |      0.726325 |           0.726325 |                           0 |                         0 |
| matched | wiki_coreoff_s43_lr3e-4    | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.725545 |      0.725545 |           0.725545 |                           0 |                         0 |
| matched | wiki_curonly_s43_lr1e-3    | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.754033 |      0.754033 |           0.754033 |                           0 |                         0 |
| matched | wiki_curonly_s43_lr1e-3    | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.765613 |      0.765613 |           0.765613 |                           0 |                         0 |
| matched | wiki_diag_s43_lr1e-3       | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.754338 |      0.754338 |           0.754338 |                           0 |                         0 |
| matched | wiki_diag_s43_lr1e-3       | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.765891 |      0.765891 |           0.765891 |                           0 |                         0 |
| matched | wiki_gru_s43_lr3e-4        | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.722226 |      0.722226 |           0.722226 |                           0 |                         0 |
| matched | wiki_gru_s43_lr3e-4        | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.733609 |      0.733609 |           0.733609 |                           0 |                         0 |
| matched | wiki_nodeframe_s43_lr3e-4  | val     |     87621 |                  0 |               0 |         0 |            0 |            0 |      0.72159  |      0.72159  |           0.72159  |                           0 |                         0 |
| matched | wiki_nodeframe_s43_lr3e-4  | test    |     23621 |                  0 |               0 |         0 |            0 |            0 |      0.731724 |      0.731724 |           0.731724 |                           0 |                         0 |
| smoke   | attention                  | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.642936 |      0.642936 |           0.642936 |                           0 |                         0 |
| smoke   | attention                  | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.537425 |      0.537425 |           0.537425 |                           0 |                         0 |
| smoke   | coreoff                    | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.641534 |      0.641534 |           0.641534 |                           0 |                         0 |
| smoke   | coreoff                    | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.54059  |      0.54059  |           0.54059  |                           0 |                         0 |
| smoke   | gru                        | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.609767 |      0.609767 |           0.609767 |                           0 |                         0 |
| smoke   | gru                        | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.525499 |      0.525499 |           0.525499 |                           0 |                         0 |
| smoke   | synth_r                    | val     |      2500 |                  0 |               0 |         0 |            0 |            0 |      0.210485 |      0.210485 |           0.210485 |                           0 |                         0 |
| smoke   | synth_r                    | test    |      1500 |                  0 |               0 |         0 |            0 |            0 |      0.215083 |      0.215083 |           0.215083 |                           0 |                         0 |
| smoke   | tsd                        | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.627036 |      0.627036 |           0.627036 |                           0 |                         0 |
| smoke   | tsd                        | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.526321 |      0.526321 |           0.526321 |                           0 |                         0 |
| smoke   | tsd_clockdiag              | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.626814 |      0.626814 |           0.626814 |                           0 |                         0 |
| smoke   | tsd_clockdiag              | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.526862 |      0.526862 |           0.526862 |                           0 |                         0 |

## Clock / saturation diagnostics (per run x split x activity class)

| group   | run           | clock   | split        | activity   |       n |   frac_capped |   frac_zero_gap_used |   n_first_update |   n_first_interaction |     mean_dt |   mean_dt_uncapped |   mean_log10_gap_used_pos |   delta_scale |
|:--------|:--------------|:--------|:-------------|:-----------|--------:|--------------:|---------------------:|-----------------:|----------------------:|------------:|-------------------:|--------------------------:|--------------:|
| audit   | forum_f_s47   | global  | train_replay | endpoint   | 3180356 |   0.999975    |          0.00192934  |           125128 |                136616 | 0.249997    |        4.5682      |                   2.77779 |           600 |
| audit   | forum_f_s47   | global  | train_replay | closure    | 8097681 |   0.000930019 |          0.00120047  |            11488 |                 29106 | 0.00192604  |        0.0025063   |                   2.77723 |           600 |
| audit   | forum_f_s47   | global  | val          | endpoint   | 2856301 |   1           |          0           |             7026 |                  7026 | 0.25        |        4.63004     |                   2.77777 |           600 |
| audit   | forum_f_s47   | global  | val          | closure    | 7467686 |   0.000814576 |          0           |                0 |                     0 | 0.00193225  |        0.00233779  |                   2.77748 |           600 |
| audit   | forum_f_s47   | global  | test         | endpoint   | 2871270 |   1           |          0           |             3097 |                  3097 | 0.25        |        4.62273     |                   2.77796 |           600 |
| audit   | forum_f_s47   | global  | test         | closure    | 6903909 |   0.000889062 |          0           |                0 |                     0 | 0.00198772  |        0.00246746  |                   2.77694 |           600 |
| audit   | polecat_f_s43 | global  | train_replay | endpoint   |  696804 |   1           |          0.000420491 |           109700 |                117435 | 0.25        |     6398.99        |                   4.93649 |         86400 |
| audit   | polecat_f_s43 | global  | train_replay | closure    | 4320507 |   0.999053    |          0.000662654 |             7735 |                 79201 | 0.249827    |     4057.16        |                   4.93649 |         86400 |
| audit   | polecat_f_s43 | global  | val          | endpoint   |  157299 |   1           |          0           |            17253 |                 17253 | 0.25        |     6496.6         |                   4.93658 |         86400 |
| audit   | polecat_f_s43 | global  | val          | closure    |  988285 |   1           |          0           |                0 |                     0 | 0.25        |     4040.86        |                   4.93657 |         86400 |
| audit   | polecat_f_s43 | global  | test         | endpoint   |  151286 |   1           |          0           |            16243 |                 16243 | 0.25        |     6412.58        |                   4.9365  |         86400 |
| audit   | polecat_f_s43 | global  | test         | closure    |  868432 |   1           |          0           |                0 |                     0 | 0.25        |     4070.92        |                   4.93651 |         86400 |
| audit   | sp_f_s43      | global  | train_replay | endpoint   |  471683 |   0.769286    |          0.00505212  |            29692 |                 31556 | 0.205082    |       58.0896      |                   0       |             1 |
| audit   | sp_f_s43      | global  | train_replay | closure    |  119048 |   0.264364    |          0.00984477  |             1864 |                 15125 | 0.0771727   |        2.27853     |                   0       |             1 |
| audit   | sp_f_s43      | global  | val          | endpoint   |   93744 |   0.666965    |          0           |             9339 |                  9339 | 0.187035    |       11.7702      |                   0       |             1 |
| audit   | sp_f_s43      | global  | val          | closure    |   10570 |   0.200946    |          0           |                0 |                     0 | 0.0622102   |        0.940173    |                   0       |             1 |
| audit   | sp_f_s43      | global  | test         | endpoint   |   95718 |   0.651957    |          0           |             6538 |                  6538 | 0.181579    |        7.07359     |                   0       |             1 |
| audit   | sp_f_s43      | global  | test         | closure    |   11857 |   0.190942    |          0           |                0 |                     0 | 0.0603203   |        0.650908    |                   0       |             1 |
| audit   | sp_f_s46      | global  | train_replay | endpoint   |  471683 |   0.00958483  |          0.00505212  |            29692 |                 31556 | 0.00398158  |        0.0311231   |                   0       |             1 |
| audit   | sp_f_s46      | global  | train_replay | closure    |  119048 |   0.00281399  |          0.00984477  |             1864 |                 15125 | 0.00120816  |        0.00346906  |                   0       |             1 |
| audit   | sp_f_s46      | global  | val          | endpoint   |   93744 |   0.0072538   |          0           |             9339 |                  9339 | 0.00357353  |        0.0128492   |                   0       |             1 |
| audit   | sp_f_s46      | global  | val          | closure    |   10570 |   0           |          0           |                0 |                     0 | 2.66421e-05 |        2.66421e-05 |                   0       |             1 |
| audit   | sp_f_s46      | global  | test         | endpoint   |   95718 |   0.00627886  |          0           |             6538 |                  6538 | 0.00302086  |        0.00945084  |                   0       |             1 |
| audit   | sp_f_s46      | global  | test         | closure    |   11857 |   0           |          0           |                0 |                     0 | 9.05863e-07 |        9.05863e-07 |                   0       |             1 |
| audit   | sp_f_s47      | global  | train_replay | endpoint   |  471683 |   0.617275    |          0.00505212  |            29692 |                 31556 | 0.156974    |       23.0432      |                   0       |             1 |
| audit   | sp_f_s47      | global  | train_replay | closure    |  119048 |   0.884761    |          0.00984477  |             1864 |                 15125 | 0.222796    |       37.4596      |                   0       |             1 |
| audit   | sp_f_s47      | global  | val          | endpoint   |   93744 |   0.848577    |          0           |             9339 |                  9339 | 0.214354    |       43.902       |                   0       |             1 |
| audit   | sp_f_s47      | global  | val          | closure    |   10570 |   0.88666     |          0           |                0 |                     0 | 0.222907    |       39.6129      |                   0       |             1 |
| audit   | sp_f_s47      | global  | test         | endpoint   |   95718 |   0.875342    |          0           |             6538 |                  6538 | 0.220999    |       43.2412      |                   0       |             1 |
| audit   | sp_f_s47      | global  | test         | closure    |   11857 |   0.879227    |          0           |                0 |                     0 | 0.221227    |       38.1172      |                   0       |             1 |
| audit   | sw_f_s43      | global  | train_replay | endpoint   | 1303199 |   0.257094    |          0.000412063 |           525811 |                534422 | 0.0806959   |        0.166753    |                   3.55627 |          3600 |
| audit   | sw_f_s43      | global  | train_replay | closure    | 1061697 |   0.00142884  |          0.00109259  |             8611 |                 36755 | 0.00316541  |        0.00331942  |                   3.55626 |          3600 |
| audit   | sw_f_s43      | global  | val          | endpoint   |  279765 |   0.255221    |          0           |            81054 |                 81054 | 0.0796798   |        0.164202    |                   3.5546  |          3600 |
| audit   | sw_f_s43      | global  | val          | closure    |  185742 |   0.00103369  |          0           |                0 |                     0 | 0.0027357   |        0.00283474  |                   3.55419 |          3600 |
| audit   | sw_f_s43      | global  | test         | endpoint   |  270528 |   0.255863    |          0           |            66451 |                 66451 | 0.0793651   |        0.166654    |                   3.55559 |          3600 |
| audit   | sw_f_s43      | global  | test         | closure    |  219058 |   0.000826265 |          0           |                0 |                     0 | 0.00262919  |        0.00270938  |                   3.55507 |          3600 |
| audit   | sw_f_s46      | global  | train_replay | endpoint   | 1303199 |   0.249077    |          0.000412063 |           525811 |                534422 | 0.0822498   |        0.177267    |                   3.55627 |          3600 |
| audit   | sw_f_s46      | global  | train_replay | closure    | 1061697 |   0.00280306  |          0.00109259  |             8611 |                 36755 | 0.0043743   |        0.00504024  |                   3.55626 |          3600 |
| audit   | sw_f_s46      | global  | val          | endpoint   |  279765 |   0.244091    |          0           |            81054 |                 81054 | 0.0809779   |        0.171203    |                   3.5546  |          3600 |
| audit   | sw_f_s46      | global  | val          | closure    |  185742 |   0.00171205  |          0           |                0 |                     0 | 0.0037614   |        0.00411092  |                   3.55419 |          3600 |
| audit   | sw_f_s46      | global  | test         | endpoint   |  270528 |   0.240918    |          0           |            66451 |                 66451 | 0.0801793   |        0.168496    |                   3.55559 |          3600 |
| audit   | sw_f_s46      | global  | test         | closure    |  219058 |   0.00131016  |          0           |                0 |                     0 | 0.00362975  |        0.00389548  |                   3.55507 |          3600 |
| audit   | sw_f_s47      | global  | train_replay | endpoint   | 1303199 |   0.0991729   |          0.000412063 |           525811 |                534422 | 0.0823272   |        0.0977986   |                   3.55627 |          3600 |
| audit   | sw_f_s47      | global  | train_replay | closure    | 1061697 |   0.000810024 |          0.00109259  |             8611 |                 36755 | 0.021325    |        0.0213742   |                   3.55626 |          3600 |
| audit   | sw_f_s47      | global  | val          | endpoint   |  279765 |   0.0988222   |          0           |            81054 |                 81054 | 0.0827203   |        0.0966902   |                   3.5546  |          3600 |
| audit   | sw_f_s47      | global  | val          | closure    |  185742 |   0.000446856 |          0           |                0 |                     0 | 0.0212197   |        0.0212488   |                   3.55419 |          3600 |
| audit   | sw_f_s47      | global  | test         | endpoint   |  270528 |   0.0944228   |          0           |            66451 |                 66451 | 0.0812836   |        0.096821    |                   3.55559 |          3600 |
| audit   | sw_f_s47      | global  | test         | closure    |  219058 |   0.000269335 |          0           |                0 |                     0 | 0.0213758   |        0.0213941   |                   3.55507 |          3600 |
| audit   | wd_f_s43      | global  | train_replay | endpoint   | 2374141 |   0           |          0.0173844   |           382887 |                384705 | 0.0102124   |        0.0102124   |                   0       |             1 |
| audit   | wd_f_s43      | global  | train_replay | closure    |  428260 |   0           |          0.00424508  |             1818 |                  1818 | 0.0112371   |        0.0112371   |                   0       |             1 |
| audit   | wd_f_s43      | global  | val          | endpoint   | 1613946 |   0           |          0           |           216363 |                216363 | 0.0104141   |        0.0104141   |                   0       |             1 |
| audit   | wd_f_s43      | global  | val          | closure    |  222249 |   0           |          0           |                0 |                     0 | 0.0107321   |        0.0107321   |                   0       |             1 |
| audit   | wd_f_s43      | global  | test         | endpoint   | 1604472 |   0           |          0           |           305272 |                305272 | 0.0106367   |        0.0106367   |                   0       |             1 |
| audit   | wd_f_s43      | global  | test         | closure    |  234774 |   0           |          0           |                0 |                     0 | 0.0105717   |        0.0105717   |                   0       |             1 |
| audit   | wd_f_s46      | global  | train_replay | endpoint   | 2374141 |   0           |          0.0173844   |           382887 |                384705 | 0.010375    |        0.010375    |                   0       |             1 |
| audit   | wd_f_s46      | global  | train_replay | closure    |  428260 |   0           |          0.00424508  |             1818 |                  1818 | 0.0120347   |        0.0120347   |                   0       |             1 |
| audit   | wd_f_s46      | global  | val          | endpoint   | 1613946 |   0           |          0           |           216363 |                216363 | 0.010657    |        0.010657    |                   0       |             1 |
| audit   | wd_f_s46      | global  | val          | closure    |  222249 |   0           |          0           |                0 |                     0 | 0.0120505   |        0.0120505   |                   0       |             1 |
| audit   | wd_f_s46      | global  | test         | endpoint   | 1604472 |   0           |          0           |           305272 |                305272 | 0.0109756   |        0.0109756   |                   0       |             1 |
| audit   | wd_f_s46      | global  | test         | closure    |  234774 |   0           |          0           |                0 |                     0 | 0.0119996   |        0.0119996   |                   0       |             1 |
| audit   | wd_f_s47      | global  | train_replay | endpoint   | 2374141 |   0.00678561  |          0.0173844   |           382887 |                384705 | 0.0988589   |        0.0990463   |                   0       |             1 |
| audit   | wd_f_s47      | global  | train_replay | closure    |  428260 |   0.000217158 |          0.00424508  |             1818 |                  1818 | 0.0872234   |        0.0872281   |                   0       |             1 |
| audit   | wd_f_s47      | global  | val          | endpoint   | 1613946 |   0.0250987   |          0           |           216363 |                216363 | 0.095956    |        0.0970512   |                   0       |             1 |
| audit   | wd_f_s47      | global  | val          | closure    |  222249 |   2.69967e-05 |          0           |                0 |                     0 | 0.0870248   |        0.0870253   |                   0       |             1 |
| audit   | wd_f_s47      | global  | test         | endpoint   | 1604472 |   0.00582809  |          0           |           305272 |                305272 | 0.0904322   |        0.0913112   |                   0       |             1 |
| audit   | wd_f_s47      | global  | test         | closure    |  234774 |   4.25942e-06 |          0           |                0 |                     0 | 0.0867182   |        0.0867183   |                   0       |             1 |
| audit   | wiki_f_s43    | global  | train_replay | endpoint   |  127665 |   0.103826    |          0.000148827 |             3503 |                  7475 | 0.0659904   |        0.11395     |                   2.77805 |           600 |
| audit   | wiki_f_s43    | global  | train_replay | closure    |  894276 |   0.318005    |          6.59752e-05 |             3972 |                132309 | 0.151736    |        0.229971    |                   2.77826 |           600 |
| audit   | wiki_f_s43    | global  | val          | endpoint   |   26656 |   0.102003    |          0           |              836 |                   836 | 0.0643684   |        0.112824    |                   2.778   |           600 |
| audit   | wiki_f_s43    | global  | val          | closure    |  136421 |   0.306705    |          0           |                0 |                     0 | 0.140328    |        0.222675    |                   2.77826 |           600 |
| audit   | wiki_f_s43    | global  | test         | endpoint   |   28587 |   0.118026    |          0           |              916 |                   916 | 0.0687254   |        0.12654     |                   2.77796 |           600 |
| audit   | wiki_f_s43    | global  | test         | closure    |  175893 |   0.331776    |          0           |                0 |                     0 | 0.148865    |        0.23376     |                   2.77824 |           600 |
| audit   | wiki_f_s46    | global  | train_replay | endpoint   |  127665 |   0.0118357   |          0.000148827 |             3503 |                  7475 | 0.0454231   |        0.0488664   |                   2.77805 |           600 |
| audit   | wiki_f_s46    | global  | train_replay | closure    |  894276 |   0.000193453 |          6.59752e-05 |             3972 |                132309 | 0.033705    |        0.0337288   |                   2.77826 |           600 |
| audit   | wiki_f_s46    | global  | val          | endpoint   |   26656 |   0.00712785  |          0           |              836 |                   836 | 0.0448337   |        0.0470762   |                   2.778   |           600 |
| audit   | wiki_f_s46    | global  | val          | closure    |  136421 |   0.000344522 |          0           |                0 |                     0 | 0.0340043   |        0.034021    |                   2.77826 |           600 |
| audit   | wiki_f_s46    | global  | test         | endpoint   |   28587 |   0.0126631   |          0           |              916 |                   916 | 0.0474092   |        0.0508305   |                   2.77796 |           600 |
| audit   | wiki_f_s46    | global  | test         | closure    |  175893 |   0.000187614 |          0           |                0 |                     0 | 0.0337025   |        0.0337105   |                   2.77824 |           600 |
| audit   | wiki_f_s47    | global  | train_replay | endpoint   |  127665 |   0.0564838   |          0.000148827 |             3503 |                  7475 | 0.0405183   |        0.180628    |                   2.77805 |           600 |
| audit   | wiki_f_s47    | global  | train_replay | closure    |  894276 |   0.00622403  |          6.59752e-05 |             3972 |                132309 | 0.0175493   |        0.0220584   |                   2.77826 |           600 |
| audit   | wiki_f_s47    | global  | val          | endpoint   |   26656 |   0.0365396   |          0           |              836 |                   836 | 0.0338193   |        0.118953    |                   2.778   |           600 |
| audit   | wiki_f_s47    | global  | val          | closure    |  136421 |   0.00613542  |          0           |                0 |                     0 | 0.0175875   |        0.0221466   |                   2.77826 |           600 |
| audit   | wiki_f_s47    | global  | test         | endpoint   |   28587 |   0.051352    |          0           |              916 |                   916 | 0.0378475   |        0.175124    |                   2.77796 |           600 |
| audit   | wiki_f_s47    | global  | test         | closure    |  175893 |   0.00645279  |          0           |                0 |                     0 | 0.0180415   |        0.0228511   |                   2.77824 |           600 |
| smoke   | tsd_clockdiag | global  | train_replay | endpoint   |   23975 |   0           |          0.00200209  |              873 |                  3072 | 0.0371964   |        0.0371964   |                   2.77756 |           600 |
| smoke   | tsd_clockdiag | global  | train_replay | closure    |  123689 |   0           |          0.00126123  |             2199 |                 37138 | 0.0345548   |        0.0345548   |                   2.77762 |           600 |
| smoke   | tsd_clockdiag | global  | val          | endpoint   |    4606 |   0           |          0           |              265 |                   265 | 0.0368777   |        0.0368777   |                   2.77797 |           600 |
| smoke   | tsd_clockdiag | global  | val          | closure    |   20426 |   0           |          0           |                0 |                     0 | 0.0345546   |        0.0345546   |                   2.778   |           600 |
| smoke   | tsd_clockdiag | global  | test         | endpoint   |    4427 |   0.0112943   |          0           |              384 |                   384 | 0.0392191   |        0.0646758   |                   2.80767 |           600 |
| smoke   | tsd_clockdiag | global  | test         | closure    |   19269 |   0.0101718   |          0           |                0 |                     0 | 0.0367354   |        0.0591281   |                   2.80487 |           600 |

## Not finished

| run                                    | state           |   last_epoch |   attempts |
|:---------------------------------------|:----------------|-------------:|-----------:|
| audit_forum_f_s43                      | running/pending |            0 |          2 |
| audit_forum_f_s46                      | running/pending |            0 |          2 |
| audit_icews_eval_s43                   | running/pending |            0 |          2 |
| audit_icews_eval_s46                   | running/pending |            0 |          3 |
| audit_icews_eval_s47                   | running/pending |            0 |          2 |
| audit_polecat_f_s46                    | running/pending |            0 |          3 |
| audit_polecat_f_s47                    | running/pending |            0 |          2 |
| forum_attention_recoff_s43             | running/pending |            0 |          1 |
| forum_attention_recoff_s47             | running/pending |            0 |          1 |
| forum_attention_recon_s46              | running/pending |            4 |          1 |
| forum_attention_recon_s47              | running/pending |            0 |          1 |
| forum_coreoff_recoff_s45               | running/pending |            0 |          1 |
| forum_coreoff_recon_s44                | running/pending |            4 |          1 |
| forum_coreoff_recon_s45                | running/pending |            0 |          1 |
| forum_curonly_recoff_s44               | running/pending |            0 |          1 |
| forum_curonly_recoff_s45               | running/pending |            0 |          2 |
| forum_curonly_recon_s44                | running/pending |            4 |          1 |
| forum_curonly_recon_s45                | running/pending |            1 |          2 |
| forum_diag_recoff_s46                  | running/pending |            0 |          1 |
| forum_diag_recon_s43                   | running/pending |            0 |          1 |
| forum_diag_recon_s46                   | running/pending |            4 |          1 |
| forum_gru_recoff_s45                   | running/pending |            0 |          1 |
| forum_gru_recon_s44                    | running/pending |            0 |          1 |
| forum_gru_recon_s45                    | running/pending |            3 |          1 |
| forum_identity_recoff_s44              | running/pending |            0 |          1 |
| forum_identity_recoff_s45              | running/pending |            0 |          2 |
| forum_identity_recon_s44               | running/pending |            0 |          3 |
| forum_identity_recon_s45               | running/pending |            0 |          2 |
| forum_nodeframe_recoff_s43             | running/pending |            4 |          2 |
| forum_nodeframe_recoff_s46             | running/pending |            2 |          1 |
| forum_nodeframe_recoff_s47             | running/pending |            0 |          1 |
| forum_nodeframe_recon_s46              | running/pending |            0 |          1 |
| forum_nodeframe_recon_s47              | running/pending |            0 |          1 |
| forum_tsd_recoff_s45                   | running/pending |            2 |          2 |
| forum_tsd_recon_s44                    | running/pending |            0 |          2 |
| forum_tsd_recon_s45                    | running/pending |            0 |          2 |
| synth_gen_s1r_identity_recoff_rel_s45  | running/pending |            3 |          1 |
| synth_gen_s1r_nodeframe_recoff_rel_s44 | running/pending |            4 |          1 |
| synth_gen_s1r_tsd_recoff_rel_s43       | running/pending |            5 |          1 |
| synth_gen_s1r_tsd_recoff_rel_s46       | running/pending |            6 |          1 |
| synth_gen_s2_curonly_recoff_rel_s45    | running/pending |            4 |          1 |
| synth_gen_s2_curonly_recoff_rel_s47    | running/pending |            5 |          1 |
| synth_gen_s2_gru_recoff_rel_s43        | running/pending |            0 |          1 |
| synth_gen_s2_gru_recoff_rel_s44        | running/pending |            5 |          1 |
| synth_gen_s2_gru_recoff_rel_s46        | running/pending |            3 |          1 |
| synth_gen_s2_gru_recoff_rel_s47        | running/pending |            4 |          1 |
| synth_gen_s2_identity_recoff_rel_s43   | running/pending |            6 |          1 |
| synth_gen_s2_identity_recoff_rel_s44   | running/pending |            4 |          1 |
| synth_gen_s2_identity_recoff_rel_s45   | running/pending |            3 |          1 |
| synth_gen_s2_identity_recoff_rel_s46   | running/pending |            4 |          1 |
| synth_gen_s2_tsd_recoff_rel_s43        | running/pending |            2 |          1 |
| synth_gen_s2_tsd_recoff_rel_s44        | running/pending |            4 |          1 |
| synth_gen_s2_tsd_recoff_rel_s45        | running/pending |            5 |          1 |
| synth_gen_s2_tsd_recoff_rel_s46        | running/pending |            2 |          1 |
| synth_gen_s2_tsd_recoff_rel_s47        | running/pending |            4 |          1 |
| wiki_attention_s43_lr1e-3              | running/pending |            6 |          1 |
| wiki_attention_s43_lr3e-4              | running/pending |            7 |          1 |
| wiki_coreoff_s44_lr1e-3                | running/pending |            8 |          1 |
| wiki_coreoff_s45_lr1e-3                | running/pending |            3 |          1 |
| wiki_coreoff_s46_lr1e-3                | running/pending |            7 |          1 |
| wiki_coreoff_s47_lr1e-3                | running/pending |            3 |          1 |
| wiki_curonly_s43_lr3e-4                | running/pending |            7 |          1 |
| wiki_diag_s43_lr3e-4                   | running/pending |            8 |          1 |
| wiki_gru_s43_lr1e-3                    | running/pending |            8 |          1 |
| wiki_identity_s43_lr1e-3               | running/pending |            6 |          1 |
| wiki_identity_s43_lr3e-4               | running/pending |            7 |          1 |
| wiki_nodeframe_s43_lr1e-3              | running/pending |            6 |          1 |
| wiki_tsd_s43_lr1e-3                    | running/pending |            8 |          1 |
| wiki_tsd_s43_lr3e-4                    | running/pending |            7 |          1 |

