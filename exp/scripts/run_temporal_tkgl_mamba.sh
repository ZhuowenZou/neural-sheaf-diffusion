#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-tkgl-smallpedia}"

PYTHONPATH=. "${CONDA_PREFIX}/bin/python" -m exp.run_temporal_tkgl \
  --dataset="${DATASET_NAME}" \
  --temporal_dataset="${DATASET_NAME}" \
  --model=EventTemporalMambaSheaf \
  --stateful_temporal=True \
  --closure_hops=0 \
  --temporal_include_static_context=True \
  --temporal_epochs=20 \
  --temporal_train_edges=2048 \
  --temporal_val_edges=256 \
  --temporal_test_edges=256 \
  --temporal_train_negatives_per_pos=32 \
  --temporal_candidate_chunk_size=2048 \
  --temporal_eval_every=2 \
  --temporal_skip_train_eval=True \
  --temporal_bptt_steps=1
