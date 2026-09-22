# Review campaign summary (auto-generated 2026-09-22 10:05)

6 finished runs under `results/review_2026_09_22`; arms derived from resolved configs.

## Test MRR by dataset x arm x lr

| dataset   | group   | rec   | arm             |     lr |   n |     mean |           std | values          |
|:----------|:--------|:------|:----------------|-------:|----:|---------:|--------------:|:----------------|
| tgbl-wiki | matched | on    | core-off        | 0.001  |   1 | 0.744018 | nan           | 0.7440          |
| tgbl-wiki | smoke   | on    | attention-gates | 0.0003 |   1 | 0.537425 | nan           | 0.5374          |
| tgbl-wiki | smoke   | on    | core-off        | 0.0003 |   1 | 0.54059  | nan           | 0.5406          |
| tgbl-wiki | smoke   | on    | gru-sheaf       | 0.0003 |   1 | 0.525499 | nan           | 0.5255          |
| tgbl-wiki | smoke   | on    | tsd             | 0.0003 |   2 | 0.526591 |   0.000382704 | 0.5263 / 0.5269 |

## Score-validity audit (per split)

| group   | run                     | split   |   queries |   queries_affected |   pos_nonfinite |   neg_nan |   neg_posinf |   neg_neginf |   mrr_tgb_raw |   mrr_guarded |   mrr_conservative |   state_nonfinite_snapshots |   rec_nonfinite_snapshots |
|:--------|:------------------------|:--------|----------:|-------------------:|----------------:|----------:|-------------:|-------------:|--------------:|--------------:|-------------------:|----------------------------:|--------------------------:|
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

## Not finished

| run                       | state           |   last_epoch |   attempts |
|:--------------------------|:----------------|-------------:|-----------:|
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
| forum_attention_rec_s43   | running/pending |            0 |          1 |
| forum_gru_norec_s43       | running/pending |            0 |          1 |
| forum_gru_norec_s46       | running/pending |            0 |          1 |
| forum_gru_norec_s47       | running/pending |            0 |          1 |
| forum_gru_rec_s43         | running/pending |            0 |          1 |
| forum_gru_rec_s46         | running/pending |            0 |          1 |
| forum_gru_rec_s47         | running/pending |            0 |          1 |
| forum_nodeframe_rec_s43   | running/pending |            0 |          1 |
| wiki_attention_s43_lr1e-3 | running/pending |            0 |          1 |
| wiki_attention_s43_lr3e-4 | running/pending |            0 |          1 |
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

