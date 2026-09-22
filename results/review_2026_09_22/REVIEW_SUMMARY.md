# Review campaign summary (auto-generated 2026-09-22 09:54)

4 finished runs under `results/review_2026_09_22`; arms derived from resolved configs.

## Test MRR by dataset x arm x lr

| dataset   | group   | rec   | arm             |     lr |   n |     mean |   std |   values |
|:----------|:--------|:------|:----------------|-------:|----:|---------:|------:|---------:|
| tgbl-wiki | smoke   | on    | attention-gates | 0.0003 |   1 | 0.537425 |   nan |   0.5374 |
| tgbl-wiki | smoke   | on    | core-off        | 0.0003 |   1 | 0.54059  |   nan |   0.5406 |
| tgbl-wiki | smoke   | on    | gru-sheaf       | 0.0003 |   1 | 0.525499 |   nan |   0.5255 |
| tgbl-wiki | smoke   | on    | tsd             | 0.0003 |   1 | 0.526321 |   nan |   0.5263 |

## Paired contrasts (arm minus TSD, per-seed)

| dataset   | rec   | arm             |   arm_lr |   tsd_lr |   n |   tsd_mean |   arm_mean |   delta_arm_minus_tsd_mean |   delta_sd |   t95_halfwidth |   deltas | signs   |
|:----------|:------|:----------------|---------:|---------:|----:|-----------:|-----------:|---------------------------:|-----------:|----------------:|---------:|:--------|
| tgbl-wiki | on    | attention-gates |   0.0003 |   0.0003 |   1 |   0.526321 |   0.537425 |                0.0111045   |        nan |             nan |   0.0111 | 1+ 0-   |
| tgbl-wiki | on    | core-off        |   0.0003 |   0.0003 |   1 |   0.526321 |   0.54059  |                0.0142692   |        nan |             nan |   0.0143 | 1+ 0-   |
| tgbl-wiki | on    | gru-sheaf       |   0.0003 |   0.0003 |   1 |   0.526321 |   0.525499 |               -0.000821581 |        nan |             nan |  -0.0008 | 0+ 1-   |

## Score-validity audit (per split)

| group   | run       | split   |   queries |   queries_affected |   pos_nonfinite |   neg_nan |   neg_posinf |   neg_neginf |   mrr_tgb_raw |   mrr_guarded |   mrr_conservative |   state_nonfinite_snapshots |   rec_nonfinite_snapshots |
|:--------|:----------|:--------|----------:|-------------------:|----------------:|----------:|-------------:|-------------:|--------------:|--------------:|-------------------:|----------------------------:|--------------------------:|
| smoke   | attention | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.642936 |      0.642936 |           0.642936 |                           0 |                         0 |
| smoke   | attention | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.537425 |      0.537425 |           0.537425 |                           0 |                         0 |
| smoke   | coreoff   | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.641534 |      0.641534 |           0.641534 |                           0 |                         0 |
| smoke   | coreoff   | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.54059  |      0.54059  |           0.54059  |                           0 |                         0 |
| smoke   | gru       | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.609767 |      0.609767 |           0.609767 |                           0 |                         0 |
| smoke   | gru       | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.525499 |      0.525499 |           0.525499 |                           0 |                         0 |
| smoke   | tsd       | val     |      6000 |                  0 |               0 |         0 |            0 |            0 |      0.627036 |      0.627036 |           0.627036 |                           0 |                         0 |
| smoke   | tsd       | test    |      4000 |                  0 |               0 |         0 |            0 |            0 |      0.526321 |      0.526321 |           0.526321 |                           0 |                         0 |

## Not finished

| run                       | state           |   last_epoch |   attempts |
|:--------------------------|:----------------|-------------:|-----------:|
| audit_forum_f_s43         | running/pending |            0 |          1 |
| audit_forum_f_s46         | running/pending |            0 |          1 |
| audit_forum_f_s47         | crashed         |            0 |          1 |
| audit_icews_eval_s43      | running/pending |            0 |          1 |
| audit_icews_eval_s46      | crashed         |            0 |          1 |
| audit_icews_eval_s47      | running/pending |            0 |          1 |
| audit_polecat_f_s43       | running/pending |            0 |          1 |
| audit_polecat_f_s46       | crashed         |            0 |          1 |
| audit_polecat_f_s47       | running/pending |            0 |          1 |
| audit_sp_f_s43            | crashed         |            0 |          1 |
| audit_sp_f_s46            | crashed         |            0 |          1 |
| audit_sp_f_s47            | running/pending |            0 |          1 |
| audit_sw_f_s43            | running/pending |            0 |          1 |
| audit_sw_f_s46            | running/pending |            0 |          1 |
| audit_sw_f_s47            | running/pending |            0 |          1 |
| audit_wd_f_s43            | running/pending |            0 |          1 |
| audit_wd_f_s46            | running/pending |            0 |          1 |
| audit_wd_f_s47            | running/pending |            0 |          1 |
| audit_wiki_f_s43          | running/pending |            0 |          1 |
| audit_wiki_f_s46          | running/pending |            0 |          1 |
| audit_wiki_f_s47          | running/pending |            0 |          1 |
| wiki_attention_s43_lr1e-3 | running/pending |            0 |          1 |
| wiki_attention_s43_lr3e-4 | running/pending |            0 |          1 |
| wiki_coreoff_s43_lr1e-3   | running/pending |            4 |          1 |
| wiki_coreoff_s43_lr3e-4   | running/pending |            0 |          1 |
| wiki_curonly_s43_lr1e-3   | running/pending |            0 |          1 |
| wiki_curonly_s43_lr3e-4   | running/pending |            0 |          1 |
| wiki_diag_s43_lr1e-3      | running/pending |            0 |          1 |
| wiki_diag_s43_lr3e-4      | running/pending |            0 |          1 |
| wiki_gru_s43_lr1e-3       | running/pending |            0 |          1 |
| wiki_gru_s43_lr3e-4       | running/pending |            0 |          1 |
| wiki_identity_s43_lr1e-3  | running/pending |            0 |          1 |
| wiki_identity_s43_lr3e-4  | running/pending |            0 |          1 |
| wiki_nodeframe_s43_lr1e-3 | running/pending |            0 |          1 |
| wiki_nodeframe_s43_lr3e-4 | running/pending |            0 |          1 |
| wiki_tsd_s43_lr1e-3       | running/pending |            0 |          1 |
| wiki_tsd_s43_lr3e-4       | running/pending |            0 |          1 |

