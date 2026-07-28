# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pandas as pd

import exp.temporal_tgbl_studies as tgbl_studies


def test_rerank_configs_skips_failed_candidates(monkeypatch):
    class DummyContext:
        def get_snapshot_bundle(self, time_window):
            return {"time_window": time_window, "train_snapshots": [1], "val_snapshots": [1]}

    def fake_train_single_config(*, config, **kwargs):
        if config["name"] == "bad":
            raise ValueError("Encountered non-finite values in training logits.")
        return {
            "metric_name": "mrr",
            "best_epoch": 3,
            "best_train_metric": float("nan"),
            "best_val_metric": 0.25 if config["name"] == "good-a" else 0.5,
            "best_test_metric": 0.1,
            "best_train_loss": float("nan"),
            "best_val_loss": 1.0,
            "best_test_loss": 1.1,
        }

    monkeypatch.setattr(tgbl_studies, "train_single_config", fake_train_single_config)

    rerank_df = tgbl_studies.rerank_configs(
        DummyContext(),
        [
            {"name": "bad", "time_window": None},
            {"name": "good-a", "time_window": None},
            {"name": "good-b", "time_window": 10},
        ],
        progress=False,
    )

    assert list(rerank_df["status"]) == ["ok", "ok", "failed"]
    assert list(rerank_df["best_val_metric"].iloc[:2]) == [0.5, 0.25]
    failed_row = rerank_df.iloc[2]
    assert failed_row["error_type"] == "ValueError"
    assert "non-finite" in failed_row["error_message"]
    assert json.loads(failed_row["config_json"])["name"] == "bad"
