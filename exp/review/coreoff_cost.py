"""Executed-cost comparison of the core-off anchor: reference path (no_memory + layers 0, SSM transition and
map decoder still computed) versus the verified fast bypass, on the same data slice and device, with output
parity checked (server handoff 2026-09-22, section 3).

    python -m exp.review.coreoff_cost --dataset tgbl-wiki --time-window 600 --train-edges-cap 20000 --out results/review_2026_09_22/matched/coreoff_cost_wiki
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from exp import temporal_benchmark_utils as bu
from exp.run_event_benchmark import _context_edge_index, _make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbl-wiki"); ap.add_argument("--time-window", type=float, default=600)
    ap.add_argument("--train-edges-cap", type=int, default=20000); ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec, dataset, td = bu.load_temporal_data(a.dataset)
    num_nodes = max(int(td.src.max()), int(td.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    feats = bu.make_node_features(dataset, num_nodes).to(device)
    train_ids, _, _, _ = bu.split_edge_ids(dataset, td)
    train_ids = train_ids[-a.train_edges_cap:]
    ctx = _context_edge_index(td, train_ids, 50000)
    snaps = bu.build_snapshots(td, train_ids, feats, time_window=a.time_window)
    base = dict(model="faithful", feedback_dim=16, no_memory_readout=False, sheaf_conditioning="history", learn_node_emb=False,
                relation_in_input=False, recurrency_decoder=True, recurrency_untyped=False, recurrency_symmetric=False,
                sheaf_identity=False, no_delta_t=False, no_memory=True, emb_in_head=False, d=2, layers=0, hidden_channels=8,
                temporal_d_model=64, closure_hops=1, train_negatives_per_pos=32, candidate_chunk_size=1024,
                max_score_elements=4_000_000, backbone="tsd", spatial=None, clock="global")
    rows = []
    outs = {}
    for name, fast in (("reference", False), ("fast_bypass", True)):
        torch.manual_seed(0)
        args = argparse.Namespace(**dict(base, fast_core_off=fast))
        model = _make_model(ctx, feats, num_nodes, device, args, num_relations)
        if name == "fast_bypass":
            model.load_state_dict(outs["reference_sd"])
        model.eval()
        for rep in range(a.repeats):
            model.reset_temporal_state()
            torch.cuda.synchronize() if device.type == "cuda" else None
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                state = None
                for s in snaps:
                    outputs, state = model.forward_sequence([s], initial_state=state)
            torch.cuda.synchronize() if device.type == "cuda" else None
            rows.append(dict(path=name, repeat=rep, snapshots=len(snaps), edges=int(train_ids.numel()), sec=time.perf_counter() - t0,
                             peak_alloc_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")))
        outs[name] = outputs[0]["spatial"].detach().cpu()
        outs[name + "_sd"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parity = float((outs["reference"] - outs["fast_bypass"]).abs().max())
    df = pd.DataFrame(rows); df["output_max_abs_diff"] = parity
    df.to_csv(os.path.join(a.out, "coreoff_cost.csv"), index=False)
    print(df.groupby("path").sec.agg(["mean", "min"]).to_string(), f"\nfinal spatial max|diff| = {parity:.3e}")


if __name__ == "__main__":
    main()
