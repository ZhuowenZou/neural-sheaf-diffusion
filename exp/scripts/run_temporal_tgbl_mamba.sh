#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-tgbl-wiki-v2}"

"${CONDA_PREFIX}/bin/python" -m exp.run_temporal_tgbl \
  --model=TemporalMambaSheaf \
  --dataset="${DATASET_NAME}" \
  --temporal_dataset="${DATASET_NAME}" \
  --stateful_temporal=True \
  --closure_hops=1 \
  --temporal_epochs=20 \
  --temporal_train_edges=4096 \
  --temporal_val_edges=512 \
  --temporal_test_edges=512 \
  --temporal_bptt_steps=2 \
  --temporal_eval_every=2 \
  --temporal_skip_train_eval=True
