#!/usr/bin/env bash
# set -euo pipefail

# DATASET_NAME="${1:-tgbl-wiki-v2}"

# "${CONDA_PREFIX}/bin/python" -m exp.run_temporal_tgbl \
#   --model=TemporalMambaSheaf \
#   --dataset="${DATASET_NAME}" \
#   --temporal_dataset="${DATASET_NAME}" \
#   --stateful_temporal=True \
#   --closure_hops=1 \
#   --temporal_epochs=20 \
#   --temporal_train_edges=4096 \
#   --temporal_val_edges=512 \
#   --temporal_test_edges=512 \
#   --temporal_bptt_steps=2 \
#   --temporal_eval_every=2 \
#   --temporal_skip_train_eval=True


export TGB_SHEAF_D=4
export TGB_LAYERS=1
export TGB_HIDDEN_CHANNELS=8
export TGB_DROPOUT=0.15000000000000002
\
PYTHONPATH=. "${CONDA_PREFIX}/bin/python" -m exp.run_temporal_tgbl \
    --dataset=tgbl-wiki-v2 \
    --temporal_dataset=tgbl-wiki-v2 \
    --model=TemporalMambaSheaf \
    --lr=0.01047799575121039 \
    --weight_decay=6.794457218175247e-08 \
    --stateful_temporal=True \
    --closure_hops=1 \
    --temporal_d_model=64 \
    --temporal_epochs=100 \
    --temporal_train_edges=4096 \
    --temporal_val_edges=512 \
    --temporal_test_edges=512 \
    --temporal_eval_every=2 \
    --temporal_skip_train_eval=True \
    --temporal_bptt_steps=4