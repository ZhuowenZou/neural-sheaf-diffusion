#!/bin/sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nsd

PYTHONPATH=. "$CONDA_PREFIX/bin/python" -m exp.run_temporal \
    --dataset=tgbn-trade \
    --temporal_dataset=tgbn-trade \
    --model=TemporalMambaSheaf \
    --stateful_temporal=True \
    --closure_hops=1 \
    --temporal_epochs=200 \
    --temporal_train_edges=2048 \
    --temporal_val_edges=256 \
    --temporal_test_edges=256 \
    --entity="${ENTITY}"
