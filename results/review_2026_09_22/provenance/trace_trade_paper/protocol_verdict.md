# Protocol trace: tgbn-trade (time_window=3, train_cap=2048; installed py-tgb)

Generated 2026-09-22 09:57 in 15s. Facts only.

## Label semantics (reconstruction from raw edges)

| candidate   |   offset_periods |   length_periods |   unit_sec_or_years |   label_days_checked |   pairs |   max_abs_err_mean |   max_abs_err_max |   frac_pairs_exact_1e6 |
|:------------|-----------------:|-----------------:|--------------------:|---------------------:|--------:|-------------------:|------------------:|-----------------------:|
| same_period |                0 |                1 |                   3 |                   30 |    6543 |           0.109898 |          0.666667 |             0.0449335  |
| next_period |                1 |                1 |                   3 |                   30 |    6543 |           0.246248 |          1        |             0          |
| prev_period |               -1 |                1 |                   3 |                   30 |    6543 |           0.188268 |          1        |             0.00168119 |

Best-matching generating window: **same_period** (offset 0 period(s), length 1), mean max-abs error 0.11.

## Scored label sets per split

| split        |   ours_n |   official_n | ours_only                                                                                                                                                                                                                                                                                                                        | official_only                                                                                                                  |   ours_first |   ours_last |   official_first |   official_last |
|:-------------|---------:|-------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|-------------:|------------:|-----------------:|----------------:|
| train        |        1 |           22 | []                                                                                                                                                                                                                                                                                                                               | [1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008] |         1987 |        1987 |             1987 |            2008 |
| val          |        4 |            4 | [2013]                                                                                                                                                                                                                                                                                                                           | [2009]                                                                                                                         |         2010 |        2013 |             2009 |            2012 |
| test         |        3 |            3 | [2016]                                                                                                                                                                                                                                                                                                                           | [2013]                                                                                                                         |         2014 |        2016 |             2013 |            2015 |
| never_scored |       22 |            1 | [np.int64(1988), np.int64(1989), np.int64(1990), np.int64(1991), np.int64(1992), np.int64(1993), np.int64(1994), np.int64(1995), np.int64(1996), np.int64(1997), np.int64(1998), np.int64(1999), np.int64(2000), np.int64(2001), np.int64(2002), np.int64(2003), np.int64(2004), np.int64(2005), np.int64(2006), np.int64(2007)] | [np.int64(2016)]                                                                                                               |          nan |         nan |              nan |             nan |

## Split-level examples

| runner               | split   |   cursor_index_at_seek |   split_first_edge_ts |   split_last_edge_ts |   snapshots |   labels_scored | note                                                                                  |   batches |   pending_label_at_split_end |   cursor_exhausted_break |
|:---------------------|:--------|-----------------------:|----------------------:|---------------------:|------------:|----------------:|:--------------------------------------------------------------------------------------|----------:|-----------------------------:|-------------------------:|
| ours                 | train   |                      0 |                  1986 |                 1987 |           1 |               1 | time_window=3, train_cap=2048 (prefix of the training split)                          |       nan |                          nan |                      nan |
| ours                 | val     |                     23 |                  2010 |                 2013 |           2 |               4 | time_window=3, train_cap=2048 (prefix of the training split)                          |       nan |                          nan |                      nan |
| ours                 | test    |                     27 |                  2014 |                 2016 |           1 |               3 | time_window=3, train_cap=2048 (prefix of the training split)                          |       nan |                          nan |                      nan |
| official_tgn_example | train   |                    nan |                  1986 |                 2009 |         nan |              22 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |      1687 |                         2009 |                        0 |
| official_tgn_example | val     |                    nan |                  2010 |                 2013 |         nan |               4 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |       366 |                         2013 |                        0 |
| official_tgn_example | test    |                    nan |                  2014 |                 2016 |         nan |               3 | batch_size=200; label fires when batch last t > pending label ts; one label per batch |       290 |                         2016 |                        0 |

## Boundary labels (first/last of every split, both runners)

| runner               | split   |   label_ts |   snapshot_ts |   frontier_t_max |   ingested_t_lt_label |   ingested_t_eq_label |   ingested_t_gt_label |   label_nodes | label_hash       | first_in_split   | last_in_split   |
|:---------------------|:--------|-----------:|--------------:|-----------------:|----------------------:|----------------------:|----------------------:|--------------:|:-----------------|:-----------------|:----------------|
| ours                 | train   |       1987 |          1987 |             1987 |                     0 |                 16384 |                     0 |           201 | 64a8806b1be54c1f | True             | True            |
| ours                 | val     |       2010 |          2012 |             2012 |                 16384 |                 54437 |                     0 |           213 | 2f1e91a6d10d2a8c | True             | False           |
| ours                 | val     |       2013 |          2013 |             2013 |                 70821 |                 18733 |                     0 |           214 | cda3e5d665433d0c | False            | True            |
| ours                 | test    |       2014 |          2016 |             2016 |                 70821 |                 57377 |                 19207 |           225 | bbfe440052a49927 | True             | False           |
| ours                 | test    |       2016 |          2016 |             2016 |                128198 |                 19207 |                     0 |           224 | 7f3d4c6273452735 | False            | True            |
| official_tgn_example | train   |       1987 |          1988 |             1987 |                     0 |                 18801 |                     0 |           201 | 64a8806b1be54c1f | True             | False           |
| official_tgn_example | train   |       2008 |          2009 |             2008 |                283986 |                 35450 |                     0 |           214 | 8d915e7bcff854a2 | False            | True            |
| official_tgn_example | val     |       2009 |          2010 |             2009 |                283986 |                 53238 |                     0 |           212 | 71f02d65e13f9027 | True             | False           |
| official_tgn_example | val     |       2012 |          2013 |             2012 |                337224 |                 54437 |                     0 |           213 | d205cab81896fdb8 | False            | True            |
| official_tgn_example | test    |       2013 |          2014 |             2013 |                391661 |                 18733 |                     0 |           214 | cda3e5d665433d0c | True             | False           |
| official_tgn_example | test    |       2015 |          2016 |             2015 |                391661 |                 57377 |                     0 |           226 | 0452d512a98429c2 | False            | True            |

## Diagnostic predictors (NDCG@10, mean over scored label timestamps)

| runner               | split   | predictor            |   label_timestamps |   ndcg_mean |
|:---------------------|:--------|:---------------------|-------------------:|------------:|
| ours                 | val     | last_label           |                  4 |    0.867286 |
| ours                 | val     | prev_1_period        |                  4 |    0.843834 |
| ours                 | val     | prev_3_periods       |                  4 |    0.817598 |
| ours                 | val     | copy_available_edges |                  4 |    0.979091 |
| ours                 | test    | last_label           |                  3 |    0.844826 |
| ours                 | test    | prev_1_period        |                  3 |    0.80625  |
| ours                 | test    | prev_3_periods       |                  3 |    0.783801 |
| ours                 | test    | copy_available_edges |                  3 |    0.975142 |
| official_tgn_example | val     | last_label           |                  4 |    0.86039  |
| official_tgn_example | val     | prev_1_period        |                  4 |    0.832993 |
| official_tgn_example | val     | prev_3_periods       |                  4 |    0.807432 |
| official_tgn_example | val     | copy_available_edges |                  4 |    1        |
| official_tgn_example | test    | last_label           |                  3 |    0.85406  |
| official_tgn_example | test    | prev_1_period        |                  3 |    0.810092 |
| official_tgn_example | test    | prev_3_periods       |                  3 |    0.790729 |
| official_tgn_example | test    | copy_available_edges |                  3 |    1        |
