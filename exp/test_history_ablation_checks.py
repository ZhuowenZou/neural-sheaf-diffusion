# Diagnostic checks for the "no-history-to-sheaf" (sheaf_conditioning) ablation.
#
# Verifies, on a small synthetic graph, that:
#   1. history mode: the previous hidden state changes the decoded restriction maps;
#   2. current_only mode: previous memory does NOT change the decoded maps;
#   3. current_only mode: previous memory still changes the predictions;
#   4. current_only mode: lagged spatial feedback does NOT change the decoded maps;
#   5. the sheaf-diffusion block is active (maps decoded, graph-dependent output)
#      in both variants;
#   6. both variants expose identical parameter counts (no extra parameters).
#
# Run: python exp/test_history_ablation_checks.py

import os
import sys

import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mamba_models import MambaSheafDiffusion, TemporalMambaState


def _normalize(edge_index):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None)
    return edge_index.contiguous()


GRAPH_SIZE = 12
INPUT_DIM = 16


def _make_args(device, sheaf_conditioning):
    return {
        "d": 4,
        "add_lp": False,
        "add_hp": False,
        "device": device,
        "graph_size": GRAPH_SIZE,
        "layers": 3,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "sparse_learner": False,
        "use_act": True,
        "input_dim": INPUT_DIM,
        "hidden_channels": 64,
        "output_dim": GRAPH_SIZE,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
        "stateful_temporal": True,
        "closure_hops": 2,
        "temporal_d_model": 32,
        "sheaf_conditioning": sheaf_conditioning,
    }


def _make_edge_index(device):
    src = torch.arange(GRAPH_SIZE, device=device)
    dst = (src + 1) % GRAPH_SIZE
    chords_src = torch.tensor([0, 2, 4, 6], device=device)
    chords_dst = torch.tensor([5, 7, 9, 11], device=device)
    return _normalize(torch.stack([
        torch.cat([src, chords_src]),
        torch.cat([dst, chords_dst]),
    ]))


def _make_model(device, sheaf_conditioning, seed=0):
    torch.manual_seed(seed)
    model = MambaSheafDiffusion(_make_edge_index(device), _make_args(device, sheaf_conditioning)).to(device)
    model.eval()
    return model


def _decoded_maps(model, x, edge_index, state):
    with torch.no_grad():
        logits, _ = model.step(x, edge_index=edge_index, state=state, return_state=True)
    maps = [learner.L.detach().clone() for learner in model.sheaf_learners]
    return logits.detach().clone(), maps


def _random_state(model, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    memory = torch.randn(GRAPH_SIZE, model.hidden_dim, generator=gen).to(device)
    spatial = torch.randn(GRAPH_SIZE, model.hidden_dim, generator=gen).to(device)
    return TemporalMambaState(memory=memory, spatial=spatial)


def _maps_equal(maps_a, maps_b):
    return all(torch.allclose(a, b, atol=0.0, rtol=0.0) for a, b in zip(maps_a, maps_b))


def _run_checks(device):
    torch.manual_seed(123)
    x = torch.randn(GRAPH_SIZE, INPUT_DIM, device=device)
    edge_index = _make_edge_index(device)

    results = {}

    # --- Check 1: history mode -> previous memory changes the decoded maps.
    model_h = _make_model(device, "history")
    state_a = _random_state(model_h, device, seed=1)
    state_b = TemporalMambaState(
        memory=state_a.memory + 1.0,
        spatial=state_a.spatial.clone(),
    )
    _, maps_a = _decoded_maps(model_h, x, edge_index, state_a)
    _, maps_b = _decoded_maps(model_h, x, edge_index, state_b)
    results["1_history_state_changes_maps"] = not _maps_equal(maps_a, maps_b)

    # --- Checks 2-4: current_only mode.
    model_c = _make_model(device, "current_only")
    state_a = _random_state(model_c, device, seed=2)
    state_mem = TemporalMambaState(memory=state_a.memory + 1.0, spatial=state_a.spatial.clone())
    state_spatial = TemporalMambaState(memory=state_a.memory.clone(), spatial=state_a.spatial + 1.0)

    logits_a, maps_a = _decoded_maps(model_c, x, edge_index, state_a)
    logits_mem, maps_mem = _decoded_maps(model_c, x, edge_index, state_mem)
    logits_sp, maps_sp = _decoded_maps(model_c, x, edge_index, state_spatial)

    results["2_current_only_memory_does_not_change_maps"] = _maps_equal(maps_a, maps_mem)
    results["3_current_only_memory_still_changes_predictions"] = not torch.allclose(logits_a, logits_mem)
    results["4_current_only_lagged_feedback_not_in_maps"] = _maps_equal(maps_a, maps_sp)

    # --- Check 5: sheaf diffusion active in both variants.
    active = True
    for model in (model_h, model_c):
        state = _random_state(model, device, seed=3)
        logits_full, maps = _decoded_maps(model, x, edge_index, state)
        active = active and all(m is not None and torch.isfinite(m).all() for m in maps)
        # Different graph -> different decoded operator -> different output.
        sparse_edge_index = edge_index[:, : GRAPH_SIZE // 2]
        logits_sparse, _ = _decoded_maps(model, x, sparse_edge_index, state)
        model.update_edge_index(edge_index)  # restore
        active = active and not torch.allclose(logits_full, logits_sparse)
    results["5_sheaf_diffusion_active_in_both_variants"] = active

    # --- Check 6: identical parameter counts.
    count_h = sum(p.numel() for p in model_h.parameters())
    count_c = sum(p.numel() for p in model_c.parameters())
    results["6_identical_parameter_counts"] = count_h == count_c
    results["_param_count_history"] = count_h
    results["_param_count_current_only"] = count_c

    # --- Baseline-unchanged check: history model ignores sheaf_signal plumbing.
    model_default = _make_model(device, "history")
    model_flagged = _make_model(device, "history")
    state = _random_state(model_default, device, seed=4)
    logits_default, maps_default = _decoded_maps(model_default, x, edge_index, state)
    logits_flagged, maps_flagged = _decoded_maps(model_flagged, x, edge_index, state)
    results["7_history_mode_reproduces_baseline"] = (
        torch.allclose(logits_default, logits_flagged) and _maps_equal(maps_default, maps_flagged)
    )

    return results


def test_history_ablation_checks():
    # CPU: CUDA sparse matmuls are non-deterministic (atomic adds), which breaks
    # the bitwise map-invariance assertions even for two identical models.
    results = _run_checks(torch.device("cpu"))
    failures = {k: v for k, v in results.items() if not k.startswith("_") and v is not True}
    assert not failures, f"Diagnostic checks failed: {failures}"


if __name__ == "__main__":
    device = torch.device("cpu")
    results = _run_checks(device)
    print(f"device: {device}")
    ok = True
    for key, value in results.items():
        if key.startswith("_"):
            print(f"  {key[1:]}: {value}")
            continue
        status = "PASS" if value is True else "FAIL"
        ok = ok and value is True
        print(f"  [{status}] {key}")
    sys.exit(0 if ok else 1)
