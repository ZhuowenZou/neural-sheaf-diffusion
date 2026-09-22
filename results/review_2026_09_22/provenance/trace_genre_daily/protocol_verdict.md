# Protocol trace: tgbn-genre (time_window=86400, train_cap=-1; installed py-tgb)

Generated 2026-09-22 10:00 in 239s. Facts only.

## Label semantics (reconstruction from raw edges)

| candidate      |   offset_periods |   length_periods |   unit_sec_or_years |   label_days_checked |   pairs |   max_abs_err_mean |   max_abs_err_max |   frac_pairs_exact_1e6 |
|:---------------|-----------------:|-----------------:|--------------------:|---------------------:|--------:|-------------------:|------------------:|-----------------------:|
| same_period    |                0 |                1 |               86400 |                   24 |    3807 |           0.416326 |                 1 |            0           |
| next_period    |                1 |                1 |               86400 |                   24 |    3807 |           0.415982 |                 1 |            0.000525348 |
| prev_period    |               -1 |                1 |               86400 |                   24 |    3807 |           0.41855  |                 1 |            0           |
| next_7         |                0 |                7 |               86400 |                   24 |    3807 |           0.3077   |                 1 |            0           |
| next_8_incl    |                0 |                8 |               86400 |                   24 |    3807 |           0.309708 |                 1 |            0           |
| prev_7         |               -7 |                7 |               86400 |                   24 |    3807 |           0.360673 |                 1 |            0           |
| next_7_from_+1 |                1 |                7 |               86400 |                   24 |    3807 |           0.307404 |                 1 |            0           |
| window_-1_+7   |               -1 |                8 |               86400 |                   24 |    3807 |           0.309722 |                 1 |            0           |

Best-matching generating window: **next_7_from_+1** (offset 1 period(s), length 7), mean max-abs error 0.307.

## Scored label sets per split

| split        |   ours_n |   official_n | ours_only   | official_only   |    ours_first |     ours_last |   official_first |   official_last |
|:-------------|---------:|-------------:|:------------|:----------------|--------------:|--------------:|-----------------:|----------------:|
| train        |     1250 |         1250 | []          | []              |   1.10844e+09 |   1.21635e+09 |      1.10844e+09 |     1.21635e+09 |
| val          |      163 |          163 | []          | []              |   1.21644e+09 |   1.23044e+09 |      1.21644e+09 |     1.23044e+09 |
| test         |      166 |          166 | []          | []              |   1.23053e+09 |   1.24478e+09 |      1.23053e+09 |     1.24478e+09 |
| never_scored |        0 |            0 | []          | []              | nan           | nan           |    nan           |   nan           |

## Split-level examples

| runner               | split   |   cursor_index_at_seek |   split_first_edge_ts |   split_last_edge_ts |   snapshots |   labels_scored | note                                                                                  |   batches |   pending_label_at_split_end |   cursor_exhausted_break |
|:---------------------|:--------|-----------------------:|----------------------:|---------------------:|------------:|----------------:|:--------------------------------------------------------------------------------------|----------:|-----------------------------:|-------------------------:|
| ours                 | train   |                      0 |            1108357203 |           1216427761 |        1251 |            1250 | time_window=86400, train_cap=-1 (prefix of the training split)                        |       nan |                nan           |                      nan |
| ours                 | val     |                   1250 |            1216427762 |           1230448624 |         163 |             163 | time_window=86400, train_cap=-1 (prefix of the training split)                        |       nan |                nan           |                      nan |
| ours                 | test    |                   1413 |            1230448684 |           1245461220 |         174 |             166 | time_window=86400, train_cap=-1 (prefix of the training split)                        |       nan |                nan           |                      nan |
| official_tgn_example | train   |                    nan |            1108357203 |           1216427761 |         nan |            1250 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |     62505 |                  1.21644e+09 |                        0 |
| official_tgn_example | val     |                    nan |            1216427762 |           1230448624 |         nan |             163 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |     13394 |                  1.23053e+09 |                        0 |
| official_tgn_example | test    |                    nan |            1230448684 |           1245461220 |         nan |             166 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |     13394 |                  1.24478e+09 |                        1 |

## Boundary labels (first/last of every split, both runners)

| runner               | split   |   label_ts |   snapshot_ts |   frontier_t_max |   ingested_t_lt_label |   ingested_t_eq_label |   ingested_t_gt_label |   label_nodes | label_hash       | first_in_split   | last_in_split   |
|:---------------------|:--------|-----------:|--------------:|-----------------:|----------------------:|----------------------:|----------------------:|--------------:|:-----------------|:-----------------|:----------------|
| ours                 | train   | 1108443600 |    1108529945 |       1108529945 |                  1743 |                  1090 |                   377 |            26 | 0681c0c3276d9643 | True             | False           |
| ours                 | train   | 1216353600 |    1216357200 |       1216357200 |              12485107 |                  3499 |                     0 |           225 | 3f57c6299fd94a93 | False            | True            |
| ours                 | val     | 1216440000 |    1216514161 |       1216514161 |              12500454 |                 16001 |                   453 |           204 | cda37ef608ce01ef | True             | False           |
| ours                 | val     | 1230440400 |    1230448624 |       1230448624 |              15174582 |                  5056 |                     0 |           200 | c78f4f7b5bec7f7d | False            | True            |
| ours                 | test    | 1230526800 |    1230535083 |       1230535083 |              15189824 |                  5346 |                     0 |           222 | 57ad04e079c19295 | True             | False           |
| ours                 | test    | 1244779200 |    1244791080 |       1244791080 |              17838552 |                  1583 |                     0 |             5 | 8bee10d1f1fe88ed | False            | True            |
| official_tgn_example | train   | 1108443600 |    1108458543 |       1108458543 |                  1743 |                   457 |                     0 |            26 | 0681c0c3276d9643 | True             | False           |
| official_tgn_example | train   | 1216353600 |    1216353605 |       1216353605 |              12485107 |                  3093 |                     0 |           225 | 3f57c6299fd94a93 | False            | True            |
| official_tgn_example | val     | 1216440000 |    1216440903 |       1216440903 |              12500454 |                  3624 |                     0 |           204 | cda37ef608ce01ef | True             | False           |
| official_tgn_example | val     | 1230440400 |    1230440883 |       1230440883 |              15174582 |                  3696 |                     0 |           200 | c78f4f7b5bec7f7d | False            | True            |
| official_tgn_example | test    | 1230526800 |    1230528183 |       1230528183 |              15189824 |                  4214 |                     0 |           222 | 57ad04e079c19295 | True             | False           |
| official_tgn_example | test    | 1244779200 |    1244781242 |       1244779140 |              17838552 |                   788 |                     0 |             5 | 8bee10d1f1fe88ed | False            | True            |

## Diagnostic predictors (NDCG@10, mean over scored label timestamps)

| runner               | split   | predictor            |   label_timestamps |   ndcg_mean |
|:---------------------|:--------|:---------------------|-------------------:|------------:|
| ours                 | val     | last_label           |                163 |   0.29348   |
| ours                 | val     | prev_1_period        |                163 |   0.293601  |
| ours                 | val     | prev_3_periods       |                163 |   0.401665  |
| ours                 | val     | copy_available_edges |                163 |   0.0138543 |
| ours                 | test    | last_label           |                166 |   0.299839  |
| ours                 | test    | prev_1_period        |                166 |   0.301408  |
| ours                 | test    | prev_3_periods       |                166 |   0.410452  |
| ours                 | test    | copy_available_edges |                166 |   0.0137196 |
| official_tgn_example | val     | last_label           |                163 |   0.29348   |
| official_tgn_example | val     | prev_1_period        |                163 |   0.293601  |
| official_tgn_example | val     | prev_3_periods       |                163 |   0.401665  |
| official_tgn_example | val     | copy_available_edges |                163 |   0.0138543 |
| official_tgn_example | test    | last_label           |                166 |   0.299839  |
| official_tgn_example | test    | prev_1_period        |                166 |   0.301408  |
| official_tgn_example | test    | prev_3_periods       |                166 |   0.410452  |
| official_tgn_example | test    | copy_available_edges |                166 |   0.0137196 |
