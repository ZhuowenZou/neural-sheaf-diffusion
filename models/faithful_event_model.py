"""Faithful (Sec. 3.2) event-scoring temporal sheaf diffusion for large TKGs.

Subclasses ``EventTemporalMambaSheafDiffusion`` so the event-scoring head,
candidate chunking, and sparse localization machinery are inherited unchanged
(scores stay directly comparable), while the temporal/spatial core is replaced
with the faithful discrete event-time algorithm:

  - selective SSM memory with exact ZOH discretization and HiPPO-LegS init,
    updated only on the snapshot-local (active + closure) node set;
  - physical event gap Delta_k = t_k - t_{k-1} conditioning the step-size
    selector (timestamps are threaded through forward_sequence);
  - lagged local spatial feedback computed on the PREVIOUS snapshot's graph;
  - edgewise sheaf decoded ONCE per event from the memory readout and held
    fixed while the local spatial latent evolves;
  - z_u(t_k) = P_z[x_u; h_u] initialization and forward-Euler integration of
    f_diff(Z) = sigma((I - L)(I (x) W1) Z W2) with a single tied (W1, W2).

The recurrent state remains the parent's ``TemporalMambaState`` (memory,
spatial) so the benchmark harness's ``detach_temporal_state`` keeps working;
the previous graph and timestamp are cached on the module, which is valid for
the strictly sequential streaming protocol used by the TGB runners.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn
from torch_geometric.utils import degree

from .mamba_models import TemporalMambaState
from .sheaf_models import LocalConcatSheafLearner
from .sparse_temporal_mamba import EventTemporalMambaSheafDiffusion
from .temporal_sheaf_ssm import DiagonalZOHSSM, GRUMemory, SelectiveZOHSSM


class _RecurrencyStore:
    """Sorted (key -> count, last_t) cache with a pending buffer that commits at
    the next event step, plus an intra-snapshot index for event-exact lookups
    (same-key events with strictly earlier timestamps)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.keys = None
        self.count = None
        self.last_t = None
        self.pending = None
        self.intra = None

    @property
    def size(self):
        return 0 if self.keys is None else self.keys.numel()

    def stash(self, keys, t_events, build_intra=True):
        self.pending = (keys, t_events)
        self.intra = None
        if build_intra and keys.numel():
            uk, inv = torch.unique(keys, return_inverse=True)
            t = t_events.double()
            t_min = float(t.min())
            span = float(t.max() - t.min()) + 1.0
            comp = inv.double() + (t - t_min) / span
            order = torch.argsort(comp)
            self.intra = (uk, comp[order], t[order], t_min, span)

    def advance(self):
        if self.pending is None:
            return
        new_keys, t_events = self.pending
        self.pending = None
        self.intra = None
        uk, inv, uc = torch.unique(new_keys, return_inverse=True, return_counts=True)
        uc = uc.double()
        tmax = torch.full((uk.numel(),), float("-inf"), dtype=torch.float64, device=uk.device)
        tmax = tmax.scatter_reduce(0, inv, t_events.double(), reduce="amax", include_self=True)
        if self.keys is None:
            self.keys, self.count, self.last_t = uk, uc, tmax
            return
        pos = torch.searchsorted(self.keys, uk).clamp_max(self.keys.numel() - 1)
        found = self.keys[pos] == uk
        fp = pos[found]
        self.count[fp] += uc[found]
        self.last_t[fp] = torch.maximum(self.last_t[fp], tmax[found])
        nk = uk[~found]
        if nk.numel():
            keys = torch.cat([self.keys, nk])
            cnt = torch.cat([self.count, uc[~found]])
            lt = torch.cat([self.last_t, tmax[~found]])
            order = torch.argsort(keys)
            self.keys, self.count, self.last_t = keys[order], cnt[order], lt[order]

    def lookup(self, keys_q, t_q=None):
        """(count, last_t) per query key: committed entries plus, when t_q is
        given, strictly-earlier pending events of the current snapshot."""
        device = keys_q.device
        count = torch.zeros(keys_q.shape, dtype=torch.float64, device=device)
        last = torch.full(keys_q.shape, float("-inf"), dtype=torch.float64, device=device)
        if self.size:
            pos = torch.searchsorted(self.keys, keys_q).clamp_max(self.size - 1)
            found = self.keys[pos] == keys_q
            count[found] = self.count[pos[found]]
            last[found] = self.last_t[pos[found]]
        if t_q is not None and self.intra is not None:
            uk, comp_sorted, t_sorted, t_min, span = self.intra
            rank = torch.searchsorted(uk, keys_q).clamp_max(uk.numel() - 1)
            present = uk[rank] == keys_q
            q_comp = rank.double() + ((t_q.double() - t_min) / span).clamp(0.0, 0.999999999)
            pos = torch.searchsorted(comp_sorted, q_comp)
            base = torch.searchsorted(comp_sorted, rank.double())
            c_i = (pos - base).double() * present
            has = c_i > 0
            count = count + c_i
            if has.any():
                l_i = torch.full_like(last, float("-inf"))
                l_i[has] = t_sorted[(pos[has] - 1).clamp_min(0)]
                last = torch.maximum(last, l_i)
        return count, last

    def block_rows(self, base, N, t_q=None):
        """For query rows with key blocks [base, base+N): (row_idx, obj,
        count, last_t) entries from the committed cache and, when t_q is
        given, strictly-earlier pending events."""
        device = base.device
        rows = torch.arange(base.numel(), device=device)
        out_rows, out_obj, out_cnt, out_last = [], [], [], []
        if self.size:
            lo = torch.searchsorted(self.keys, base)
            hi = torch.searchsorted(self.keys, base + N)
            cnts = hi - lo
            total = int(cnts.sum())
            if total:
                row_idx = torch.repeat_interleave(rows, cnts)
                starts = torch.cumsum(cnts, 0) - cnts
                entry = lo[row_idx] + torch.arange(total, device=device) - torch.repeat_interleave(starts, cnts)
                out_rows.append(row_idx); out_obj.append(self.keys[entry] - base[row_idx])
                out_cnt.append(self.count[entry]); out_last.append(self.last_t[entry])
        if t_q is not None and self.intra is not None:
            uk, comp_sorted, t_sorted, t_min, span = self.intra
            lo = torch.searchsorted(uk, base)
            hi = torch.searchsorted(uk, base + N)
            plo = torch.searchsorted(comp_sorted, lo.double())
            phi = torch.searchsorted(comp_sorted, hi.double())
            cnts = phi - plo
            total = int(cnts.sum())
            if total:
                row_idx = torch.repeat_interleave(rows, cnts)
                starts = torch.cumsum(cnts, 0) - cnts
                entry = plo[row_idx] + torch.arange(total, device=device) - torch.repeat_interleave(starts, cnts)
                t_e = t_sorted[entry]
                earlier = t_e < t_q.double()[row_idx]
                if earlier.any():
                    row_e, entry_e, t_e = row_idx[earlier], entry[earlier], t_e[earlier]
                    out_rows.append(row_e); out_obj.append(uk[comp_sorted[entry_e].floor().long()] - base[row_e])
                    out_cnt.append(torch.ones_like(t_e)); out_last.append(t_e)
        if not out_rows:
            return None
        return (torch.cat(out_rows), torch.cat(out_obj), torch.cat(out_cnt), torch.cat(out_last))


class FaithfulEventTemporalSheafDiffusion(EventTemporalMambaSheafDiffusion):

    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)

        # Drop the original temporal/spatial modules the faithful core replaces
        # (keeps parameter counts honest; the event head, lin1 encoder, and
        # localization helpers are retained).
        del self.nodewise_learner
        del self.sheaf_learners
        del self.weight_learners
        del self.lin_left_weights
        del self.lin_right_weights
        del self.epsilons
        del self.lin2

        self.d_h = int(args.get("temporal_d_model", 64))
        self.d_z = int(args.get("feedback_dim", 16))
        self.psi_dim = 2
        d_q = self.input_dim + self.d_z + self.psi_dim
        self.d_q = d_q

        self._extra_q_dim = (int(args.get("relation_input_dim", 16))
                             if (bool(args.get("relation_in_input", False)) and int(args.get("num_relations", 0) or 0) > 0)
                             else 0)
        # Temporal backbone (review handoff 2026-09-22, section 3): the paper's
        # selective ZOH SSM ("tsd"), a stable diagonal SSM with fixed B and the
        # same step selector ("diag_ssm"), or a GRU with the same inputs plus
        # the gap feature ("gru").  Everything downstream (readout, maps, spatial
        # block, head, REC) is shared.
        self.backbone = str(args.get("backbone", "tsd"))
        backbone_cls = {"tsd": SelectiveZOHSSM, "diag_ssm": DiagonalZOHSSM, "gru": GRUMemory}
        if self.backbone not in backbone_cls:
            raise ValueError(f"backbone must be one of {sorted(backbone_cls)}, got {self.backbone!r}")
        self.ssm = backbone_cls[self.backbone](
            self.d_h,
            d_q + self._extra_q_dim,
            dt_min=float(args.get("dt_min", 1e-3)),
            dt_max=float(args.get("dt_max", 0.1)),
            dt_cap=float(args.get("dt_cap", 0.25)),
        )
        # Node clock (section 2): "global" = gap between consecutive processed
        # snapshots (the evaluated convention); "node_update" = time since this
        # node's last memory update (closure updates included); "node_interaction"
        # = time since this node's last observed interaction (closure-only updates
        # excluded).  First observation: seen mask False -> gap 0 (explicit; no
        # sentinel).  Metadata is float64 and reset with the temporal state.
        self.clock = str(args.get("clock", "global"))
        if self.clock not in ("global", "node_update", "node_interaction"):
            raise ValueError("clock must be global | node_update | node_interaction")
        self._clock_last_update = None
        self._clock_last_interaction = None
        # Verified fast bypass for the core-off anchor (no_memory + layers == 0):
        # skips the unused SSM transition and map decoding; outputs are identical
        # (test_review_controls.test_fast_core_off_matches_reference_outputs).
        self.fast_core_off = bool(args.get("fast_core_off", False))
        self.diag = None   # models.diagnostics.ClockDiagnostics, when attached
        # Memory readout C (Eq. 5): on by default here — on this protocol the
        # recurrence is trained, where the readout was decisively better on
        # tgbn-trade at native resolution.
        self.use_memory_readout = bool(args.get("memory_readout", True))
        self.memory_readout = nn.LayerNorm(self.d_h) if self.use_memory_readout else nn.Identity()

        # Optional learnable per-node input embedding (input representation
        # only; the Sec.-3.2 temporal core is untouched). Useful on datasets
        # whose node features are random fallbacks (e.g. tgbl-wiki).
        self.use_learnable_node_features = bool(args.get("learnable_node_features", False))
        if self.use_learnable_node_features:
            self.node_feature_embedding = nn.Embedding(self.graph_size, self.input_dim)
            nn.init.normal_(self.node_feature_embedding.weight, std=0.1)

        # Optional node-type embedding for heterogeneous graphs (thgl-*):
        # adds a learned per-type vector to the input signal. Input
        # representation only; the temporal core is untouched.
        node_types = args.get("node_types")
        if node_types is not None:
            self.register_buffer("node_type_ids", node_types.long())
            self.node_type_embedding = nn.Embedding(int(node_types.max()) + 1, self.input_dim)
            nn.init.normal_(self.node_type_embedding.weight, std=0.1)
        else:
            self.node_type_ids = None
        # EMB-head / TYPE-head (head-level component, documented separately):
        # by default the scoring head encodes the RAW input features x, so the
        # learnable node / node-type embeddings reach the head only through the
        # `spatial` state of nodes that were active.  Under predict-then-update
        # most candidates carry stale state, so with this flag the head's
        # pointwise encoder sees x + embeddings for every node instead.
        self.embeddings_in_head = bool(args.get("embeddings_in_head", False))

        # Optional relation-aggregate input: mean embedding of the relations
        # incident to each node in the current event graph, appended to the
        # SSM input q (an edge descriptor in the spirit of psi_e).
        self.relation_input_dim = 0
        num_relations = int(args.get("num_relations", 0) or 0)
        if bool(args.get("relation_in_input", False)) and num_relations > 0:
            self.relation_input_dim = int(args.get("relation_input_dim", 16))
            self.relation_input_embedding = nn.Embedding(num_relations, self.relation_input_dim)
            nn.init.normal_(self.relation_input_embedding.weight, std=0.1)

        # no_memory (true memory ablation, 2026-09-14): the SSM state is still
        # advanced but NEVER read -- its readout is zeroed before entering the
        # spatial initialisation Z0 = P_z[x; 0] and the sheaf learner is
        # conditioned on the current input.  NOTE: `sheaf_conditioning =
        # current_only` alone only changes what the restriction maps are
        # decoded from; the memory still enters Z0 through h_out, so it is NOT
        # a memory ablation (earlier documents that called it "no temporal
        # memory" have been corrected).
        self.no_memory = bool(args.get("no_memory", False))
        if self.no_memory:
            args = dict(args, sheaf_conditioning="current_only")
        self.sheaf_conditioning = str(args.get("sheaf_conditioning", "history"))
        if self.sheaf_conditioning not in ("history", "current_only"):
            raise ValueError("sheaf_conditioning must be 'history' or 'current_only'")
        if self.sheaf_conditioning == "current_only":
            self.sheaf_input_proj = nn.Linear(self.input_dim, self.d_h)

        # Analytic ablations (each removes exactly one ingredient of the core):
        #  sheaf_identity: restriction maps fixed to I -> plain (stalk-wise)
        #    normalized graph diffusion instead of sheaf diffusion.
        #  no_delta_t: the step-size selector ignores the physical gap Delta_k
        #    (dt_time_weight frozen at 0) -> order-only recurrence.
        self.sheaf_identity = bool(args.get("sheaf_identity", False))
        self.no_delta_t = bool(args.get("no_delta_t", False))

        self.sheaf_learner = LocalConcatSheafLearner(
            self.d_h, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act
        )
        # Spatial-operator variant (section 3):
        #   sheaf      : incidence-specific orthogonal maps decoded from [h_u; h_v] (the model)
        #   identity   : all maps +/-I, same builder/normalisation (== sheaf_identity)
        #   node_frame : ONE orthogonal frame per node decoded from h_u; R_{e<-u} = U_u
        #                for every incident e (node-factorised transport, cycle products = I)
        #   attention  : identity transport with a history-conditioned symmetric edge
        #                gate w_uv = sigmoid(g([h_u;h_v]) + g([h_v;h_u])) in (0,1), applied
        #                through the builder's edge_weights path (same support/normalisation)
        self.spatial_variant = str(args.get("spatial", "identity" if self.sheaf_identity else "sheaf"))
        if self.spatial_variant not in ("sheaf", "identity", "node_frame", "attention"):
            raise ValueError("spatial must be sheaf | identity | node_frame | attention")
        if self.spatial_variant == "identity":
            self.sheaf_identity = True
        if self.spatial_variant == "node_frame":
            self.frame_decoder = nn.Linear(self.d_h, self.get_param_size(), bias=False)
        if self.spatial_variant == "attention":
            self.gate_decoder = nn.Linear(2 * self.d_h, 1, bias=True)
            nn.init.zeros_(self.gate_decoder.weight)
            nn.init.constant_(self.gate_decoder.bias, 2.0)   # w = sigmoid(4) ~ 0.98 at init (near identity control)
        self.feedback_proj = nn.Linear(self.hidden_dim, self.d_z)
        self.P_z = nn.Linear(self.input_dim + self.d_h, self.hidden_dim)
        self.W1 = nn.Linear(self.final_d, self.final_d, bias=False)
        nn.init.eye_(self.W1.weight)
        self.W2 = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        nn.init.orthogonal_(self.W2.weight)
        self.log_tau = nn.Parameter(torch.zeros(self.layers))

        self._prev_event_edge_index: Optional[torch.Tensor] = None
        self._prev_timestamp: Optional[torch.Tensor] = None

        # Recurrency-augmented decoder (task head only): a strictly causal
        # streaming cache of (s, r, o) triples; (seen, count, recency)
        # features feed a small MLP whose output is added to the base score.
        # Pending events commit at the START of the next step, so scoring a
        # snapshot never sees its own events. Requires bptt_steps=1 streams.
        self.use_recurrency_decoder = bool(args.get("recurrency_decoder", False))
        # Channels: typed (s, r, o) always; optional untyped (s, o) channel
        # (EdgeBank-style repeat memory across event types, for datasets
        # whose relation labels vary across repeats of the same pair).
        self.recurrency_untyped = bool(args.get("recurrency_untyped", False))
        self._rec_stores = []
        if self.use_recurrency_decoder:
            self._rec_R = max(int(args.get("num_relations", 0) or 0), 1)
            # composite key (src*R + rel)*N + dst must stay inside int64
            if int(self.graph_size) * self._rec_R * int(self.graph_size) >= 2 ** 62:
                raise ValueError("recurrency key packing overflows int64 for N^2 * R this large")
            self._rec_stores = [("typed", _RecurrencyStore())]
            if self.recurrency_untyped:
                self._rec_stores.append(("untyped", _RecurrencyStore()))
            # Order-agnostic pair channel: (min(s,o), max(s,o)) so a reverse-
            # direction repeat of the same pair also counts as seen.
            self.recurrency_symmetric = bool(args.get("recurrency_symmetric", False))
            if self.recurrency_symmetric:
                self._rec_stores.append(("symmetric", _RecurrencyStore()))
            self.recurrency_mlp = nn.Sequential(
                nn.Linear(3 * len(self._rec_stores), 16), nn.ReLU(), nn.Linear(16, 1)
            )
            nn.init.zeros_(self.recurrency_mlp[-1].weight)
            nn.init.zeros_(self.recurrency_mlp[-1].bias)
        self._rec_history = None
        if self.no_delta_t:
            with torch.no_grad():
                self.ssm.dt_time_weight.zero_()
            self.ssm.dt_time_weight.requires_grad_(False)

    # ------------------------------------------------------------------
    def set_recurrency_history(self, src, dst, rel, t):
        """Observed (s, r, o, t) facts that precede the training stream; they
        are committed to the recurrency cache on every reset so the cache
        covers the full history even when training uses a suffix cap."""
        self._rec_history = (src.long(), dst.long(), None if rel is None else rel.long(), t.double())

    def _warm_recurrency_cache(self):
        if not self.use_recurrency_decoder or self._rec_history is None:
            return
        device = self.event_bias.device
        src, dst, rel, t = self._rec_history
        for name, store in self._rec_stores:
            keys = self._rec_channel_keys(name, src.to(device), None if rel is None else rel.to(device), dst.to(device))
            store.stash(keys, t.to(device), build_intra=False)
            store.advance()

    def reset_temporal_state(self):
        super().reset_temporal_state()
        self._prev_event_edge_index = None
        self._prev_timestamp = None
        self._clock_last_update = None
        self._clock_last_interaction = None
        for _, store in self._rec_stores:
            store.reset()
        self._warm_recurrency_cache()

    # ----- clocks (section 2) ------------------------------------------
    def _clock_tables(self, device):
        if self._clock_last_update is None:
            nan = float("nan")
            self._clock_last_update = torch.full((self.graph_size,), nan, dtype=torch.float64, device=device)
            self._clock_last_interaction = torch.full((self.graph_size,), nan, dtype=torch.float64, device=device)
        return self._clock_last_update, self._clock_last_interaction

    def _node_gaps(self, timestamp, local_nodes, device):
        """(gap vector or None, last_update, last_interaction, seen_update, seen_interaction)
        for the local node set; the gap is None under the global clock."""
        lu, li = self._clock_tables(device)
        lu_l, li_l = lu[local_nodes], li[local_nodes]
        seen_u, seen_i = torch.isfinite(lu_l), torch.isfinite(li_l)
        if timestamp is None or self.clock == "global":
            return None, lu_l, li_l, seen_u, seen_i
        t = timestamp.to(device=device, dtype=torch.float64)
        ref = lu_l if self.clock == "node_update" else li_l
        seen = seen_u if self.clock == "node_update" else seen_i
        gap = torch.where(seen, (t - ref).clamp_min(0.0), torch.zeros_like(ref)).float()
        return gap, lu_l, li_l, seen_u, seen_i

    def _advance_clocks(self, timestamp, local_nodes, typed_edge_index, edge_timestamps, device):
        if timestamp is None:
            return
        lu, li = self._clock_tables(device)
        t = timestamp.to(device=device, dtype=torch.float64)
        lu[local_nodes] = t
        ev = typed_edge_index
        if ev is None or ev.numel() == 0:
            return
        ev = ev.to(device)
        if edge_timestamps is not None and edge_timestamps.numel() == ev.size(1):
            te = edge_timestamps.to(device=device, dtype=torch.float64)
        else:
            te = t.expand(ev.size(1))
        endpoints = torch.cat([ev[0], ev[1]])
        times = torch.cat([te, te])
        cur = li[endpoints]
        # unseen -> -inf (identical value per duplicate index, so the write is
        # deterministic); then a deterministic amax over duplicate endpoints
        li[endpoints] = torch.where(torch.isfinite(cur), cur, torch.full_like(cur, float("-inf")))
        li.index_reduce_(0, endpoints, times, "amax", include_self=True)

    def component_parameter_counts(self):
        """Parameter counts by component, including parameters that a variant
        leaves unused by construction (section 6)."""
        groups = {"embeddings": ("node_feature_embedding", "node_type_embedding", "relation_input_embedding"),
                  "ssm_input_selector_B": ("ssm.B_selector",), "ssm_other": ("ssm",),
                  "map_decoder": ("sheaf_learner", "sheaf_input_proj", "frame_decoder", "gate_decoder"),
                  "spatial_block": ("P_z", "W1", "W2", "log_tau", "feedback_proj", "memory_readout"),
                  "scorer": ("lin1", "lin12", "event_"), "rec": ("recurrency_mlp",)}
        out = {k: 0 for k in groups}
        out["other"] = 0
        for name, prm in self.named_parameters():
            placed = False
            for g in ("ssm_input_selector_B", "ssm_other", "embeddings", "map_decoder", "spatial_block", "scorer", "rec"):
                if any(name.startswith(pfx) for pfx in groups[g]):
                    out[g] += prm.numel(); placed = True; break
            if not placed:
                out["other"] += prm.numel()
        out["total"] = sum(p.numel() for p in self.parameters())
        out["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        unused = 0
        if self.no_memory:
            unused += sum(p.numel() for n, p in self.named_parameters() if n.startswith(("ssm", "feedback_proj", "memory_readout")))
        if self.layers == 0:
            unused += sum(p.numel() for n, p in self.named_parameters() if n.startswith(("sheaf_learner", "W1", "W2", "log_tau", "frame_decoder", "gate_decoder")))
        elif self.sheaf_identity and self.spatial_variant == "identity":
            unused += sum(p.numel() for n, p in self.named_parameters() if n.startswith("sheaf_learner"))
        if self.no_delta_t:
            unused += sum(p.numel() for n, p in self.named_parameters() if n.startswith(("ssm.dt_time_weight", "ssm.dt_time_log_scale")))
        out["unused_by_construction"] = unused
        out["active"] = out["total"] - unused
        return out

    # Compatibility views onto the typed channel (tests / diagnostics).
    @property
    def _rec_keys(self):
        return self._rec_stores[0][1].keys if self._rec_stores else None

    @property
    def _rec_count(self):
        return self._rec_stores[0][1].count if self._rec_stores else None

    @property
    def _rec_last_t(self):
        return self._rec_stores[0][1].last_t if self._rec_stores else None

    @property
    def _rec_pending(self):
        return self._rec_stores[0][1].pending if self._rec_stores else None

    @property
    def _rec_intra(self):
        return self._rec_stores[0][1].intra if self._rec_stores else None

    # ----- recurrency cache -------------------------------------------
    def _rec_pack(self, src, rel, dst):
        rel = rel.long() if rel is not None else torch.zeros_like(src)
        return (src.long() * self._rec_R + rel) * self.graph_size + dst.long()

    def _rec_channel_keys(self, name, src, rel, dst):
        if name == "typed":
            return self._rec_pack(src, rel, dst)
        if name == "untyped":
            return self._rec_pack(src, None, dst)
        lo = torch.minimum(src.long(), dst.long())
        hi = torch.maximum(src.long(), dst.long())
        return self._rec_pack(lo, None, hi)

    def _recurrency_advance(self):
        for _, store in self._rec_stores:
            store.advance()

    def _initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.graph_size, self.d_h, device=device)

    # Rows processed per chunk in _recurrency_bonus: the full B x K key grid
    # plus broadcast intermediates OOMs on the largest full-eval snapshots.
    recurrency_chunk_rows = 4096

    def _rec_features(self, counts, lasts, now):
        """Per-channel (seen, log count, log recency) -> concatenated feats."""
        feats = []
        for c, l in zip(counts, lasts):
            seen = (c > 0).float()
            dt = (now - l).clamp_min(0.0)
            dt = torch.where(c > 0, dt, torch.zeros_like(dt)).float()
            feats += [seen, torch.log1p(c.float()), torch.log1p(dt)]
        return torch.stack(feats, dim=-1)

    def _recurrency_bonus(self, src, edge_type, dst, query_time=None):
        shape = dst.shape
        device = dst.device
        if not self._rec_stores or not any(s.size or (query_time is not None and s.intra is not None)
                                           for _, s in self._rec_stores):
            return torch.zeros(shape, device=device)
        dst2d = dst.reshape(src.numel(), -1)
        snap_now = float(self._prev_timestamp) if self._prev_timestamp is not None else 0.0
        out = torch.zeros(dst2d.shape, device=device)
        for start in range(0, src.numel(), self.recurrency_chunk_rows):
            stop = min(start + self.recurrency_chunk_rows, src.numel())
            d = dst2d[start:stop]
            s = src[start:stop].view(-1, 1).expand_as(d).reshape(-1)
            r = edge_type[start:stop].view(-1, 1).expand_as(d).reshape(-1) if edge_type is not None else None
            tq = query_time[start:stop].view(-1, 1).expand_as(d).reshape(-1) if query_time is not None else None
            counts, lasts = [], []
            for name, store in self._rec_stores:
                keys = self._rec_channel_keys(name, s, r, d.reshape(-1))
                c, l = store.lookup(keys, tq)
                counts.append(c); lasts.append(l)
            now = tq.double() if tq is not None else torch.full_like(counts[0], snap_now)
            seen = torch.stack(counts, 0).sum(0) > 0
            bonus = torch.zeros(s.numel(), device=device)
            if seen.any():
                feats = self._rec_features([c[seen] for c in counts], [l[seen] for l in lasts], now[seen])
                bonus[seen] = self.recurrency_mlp(feats.to(device)).squeeze(-1)
            out[start:stop] = bonus.view(d.shape)
        return out.view(shape)

    supports_query_time = True

    def score_event_pairs(self, output, src, dst, edge_type=None, query_time=None):
        scores = super().score_event_pairs(output, src, dst, edge_type=edge_type)
        if self.use_recurrency_decoder:
            scores = scores + self._recurrency_bonus(src, edge_type, dst, query_time=query_time)
        return scores

    # Candidate sets covering at least this fraction of the vocabulary
    # (full-entity TKG protocols) are scored through a dense [rows, N]
    # matmul + gather; per-candidate scoring materializes [rows, K, hidden].
    dense_vocab_fraction = 0.25
    dense_vocab_elements = 2e8

    def score_event_candidates(self, output, src, candidates, edge_type=None, query_time=None):
        if (candidates.dim() == 2 and candidates.size(1) >= self.dense_vocab_fraction * self.graph_size
                and src.numel() > 0):
            return self._score_candidates_dense_vocab(output, src, candidates, edge_type, query_time)
        scores = super().score_event_candidates(output, src, candidates, edge_type=edge_type)
        if self.use_recurrency_decoder:
            scores = scores + self._recurrency_bonus(src, edge_type, candidates, query_time=query_time)
        return scores

    def _score_candidates_dense_vocab(self, output, src, candidates, edge_type, query_time=None):
        device = output["spatial"].device
        N = self.graph_size
        all_ids = torch.arange(N, device=device)
        dst_proj_all = self.event_destination_proj(self._event_node_repr(output, all_ids))  # [N, h]
        src = src.to(device)
        candidates = candidates.to(device)
        # Each recurrency channel materializes two [rows, N] float64 buffers in
        # the dense bonus path, so scale the row chunk by the channel count.
        per_row = max(N, 1) * (1 + 2 * len(self._rec_stores)) * 2
        rows_per_chunk = max(1, min(4096, int(self.dense_vocab_elements // per_row)))
        out = torch.empty(candidates.shape, device=device, dtype=dst_proj_all.dtype)
        for start in range(0, src.numel(), rows_per_chunk):
            stop = min(start + rows_per_chunk, src.numel())
            s = src[start:stop]
            et = edge_type[start:stop] if edge_type is not None else None
            src_proj = self.event_source_proj(self._event_node_repr(output, s))  # [b, h]
            rel, rbias = self._relation_factors(et, target_shape=s.shape, device=device, dtype=src_proj.dtype)
            query = src_proj * rel if rel is not None else src_proj
            full = query @ dst_proj_all.t() + self.event_bias  # [b, N]
            if rbias is not None:
                full = full + rbias.unsqueeze(-1)
            if self.use_recurrency_decoder:
                qt = query_time[start:stop] if query_time is not None else None
                full = full + self._recurrency_bonus_dense_rows(s, et, N, device, full.dtype, query_time=qt)
            out[start:stop] = full.gather(1, candidates[start:stop].long())
        return out

    def _recurrency_bonus_dense_rows(self, src, edge_type, N, device, dtype, query_time=None):
        """[b, N] bonus for (src, rel) query rows, aggregating each channel's
        committed entries and strictly-earlier pending events per (row, obj)."""
        b = src.numel()
        rel = edge_type.long() if edge_type is not None else torch.zeros_like(src)
        counts, lasts = [], []
        for name, store in self._rec_stores:
            count = torch.zeros(b, N, dtype=torch.float64, device=device)
            last = torch.full((b, N), float("-inf"), dtype=torch.float64, device=device)
            if name == "symmetric":
                # (min,max) keys are not a contiguous block per row: brute-force
                # lookup over all objects, in row sub-chunks to bound memory
                # (each lookup materializes several rows x N int64/float64 tensors).
                sym_rows = max(1, int(2e7 // max(N, 1)))
                all_obj = torch.arange(N, device=device)
                for r0 in range(0, b, sym_rows):
                    r1 = min(r0 + sym_rows, b)
                    nb = r1 - r0
                    s_b = src[r0:r1].view(-1, 1).expand(nb, N).reshape(-1)
                    keys = self._rec_channel_keys(name, s_b, None, all_obj.view(1, N).expand(nb, N).reshape(-1))
                    tq = query_time[r0:r1].double().view(-1, 1).expand(nb, N).reshape(-1) if query_time is not None else None
                    c, l = store.lookup(keys, tq)
                    count[r0:r1] = c.view(nb, N); last[r0:r1] = l.view(nb, N)
            else:
                r = torch.zeros_like(src) if name == "untyped" else rel
                base = (src.long() * self._rec_R + r) * N
                entries = store.block_rows(base, N, query_time)
                if entries is not None:
                    row_idx, obj, cnt, lt = entries
                    count.index_put_((row_idx, obj), cnt, accumulate=True)
                    # amax reduction (index_put_ with duplicate (row, obj) pairs is
                    # nondeterministic on CUDA; audit finding).
                    last.view(-1).index_reduce_(0, row_idx * N + obj, lt.to(last.dtype), "amax")
            counts.append(count); lasts.append(last)
        if query_time is not None:
            now = query_time.double().view(-1, 1).expand(b, N)
        else:
            now = torch.full((b, N), float(self._prev_timestamp) if self._prev_timestamp is not None else 0.0,
                             dtype=torch.float64, device=device)
        bonus = torch.zeros(b, N, dtype=dtype, device=device)
        seen = torch.stack(counts, 0).sum(0) > 0
        if seen.any():
            feats = self._rec_features([c[seen] for c in counts], [l[seen] for l in lasts], now[seen])
            bonus[seen] = self.recurrency_mlp(feats.to(device)).squeeze(-1).to(dtype)
        return bonus

    def _localize_edge_types(self, edge_index, edge_type, local_index):
        if edge_index is None or edge_index.numel() == 0 or edge_type is None:
            return None, None
        device = local_index.device
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        row, col = edge_index
        mask = (local_index[row] >= 0) & (local_index[col] >= 0)
        return local_index[edge_index[:, mask]], edge_type[mask]

    def _relation_input_signal(self, local_dynamic, local_types, n_local, device, dtype):
        signal = torch.zeros(n_local, self.relation_input_dim, device=device, dtype=dtype)
        if local_dynamic is None or local_dynamic.numel() == 0 or local_types is None:
            return signal
        emb = self.relation_input_embedding(local_types.long().to(device))
        count = torch.zeros(n_local, 1, device=device, dtype=dtype)
        for endpoint in (local_dynamic[0], local_dynamic[1]):
            signal.index_add_(0, endpoint, emb)
            count.index_add_(0, endpoint, torch.ones(endpoint.numel(), 1, device=device, dtype=dtype))
        return signal / count.clamp_min(1.0)

    def _local_feedback(self, local_prev_spatial, local_prev_edge_index, n_local, device):
        if local_prev_spatial is None:
            return torch.zeros(n_local, self.d_z, device=device)
        if local_prev_edge_index is not None and local_prev_edge_index.numel() > 0:
            row, col = local_prev_edge_index
            summary = torch.zeros_like(local_prev_spatial)
            summary.index_add_(0, row, local_prev_spatial[col])
            summary = summary + local_prev_spatial
            deg = degree(row, num_nodes=n_local).clamp_min(0).unsqueeze(-1) + 1.0
            z_read = summary / deg
        else:
            z_read = local_prev_spatial
        return self.feedback_proj(z_read)

    def _head_input_features(self, x):
        """Features the scoring head encodes for every node: raw x, plus the
        learnable node / node-type embeddings when `embeddings_in_head` is on."""
        if not self.embeddings_in_head:
            return x
        out = x
        if self.use_learnable_node_features:
            out = out + self.node_feature_embedding.weight
        if self.node_type_ids is not None:
            out = out + self.node_type_embedding(self.node_type_ids)
        return out

    def _identity_laplacian(self, local_edge_index, n_local):
        """Symmetric-normalized graph Laplacian of the local graph, expanded
        stalk-wise (kron with I_d): the sheaf Laplacian with all R = I."""
        device = local_edge_index.device
        row, col = local_edge_index
        deg = degree(row, num_nodes=n_local).clamp_min(1.0)
        w = -1.0 / torch.sqrt(deg[row] * deg[col])
        d = self.final_d
        ar = torch.arange(d, device=device)
        rows = (row.view(-1, 1) * d + ar.view(1, -1)).reshape(-1)
        cols = (col.view(-1, 1) * d + ar.view(1, -1)).reshape(-1)
        vals = w.view(-1, 1).expand(-1, d).reshape(-1)
        diag = torch.arange(n_local * d, device=device)
        index = torch.stack([torch.cat([rows, diag]), torch.cat([cols, diag])])
        values = torch.cat([vals, torch.ones(n_local * d, device=device)])
        return index, values

    def _local_faithful_diffusion(self, Z0, sheaf_signal, local_edge_index, n_local):
        if local_edge_index.numel() == 0:
            return Z0
        builder = self._make_local_builder(local_edge_index, n_local)
        if self.spatial_variant == "node_frame":
            frames = torch.tanh(self.frame_decoder(sheaf_signal))          # (n_local, P): one frame per node
            maps = frames.index_select(0, local_edge_index[0])              # R_{e<-u} = U_u for every incident e
            L, trans_maps = builder(maps)
        elif self.spatial_variant == "attention":
            row, col = local_edge_index
            hu, hv = sheaf_signal.index_select(0, row), sheaf_signal.index_select(0, col)
            logit = self.gate_decoder(torch.cat([hu, hv], dim=-1)) + self.gate_decoder(torch.cat([hv, hu], dim=-1))
            gate = torch.sigmoid(logit)                                     # (E, 1), symmetric in (u, v)
            zero_maps = torch.zeros(local_edge_index.size(1), self.get_param_size(), device=Z0.device, dtype=Z0.dtype)
            L, trans_maps = builder(zero_maps, gate)                        # identity transport, gated support
        else:
            maps = self.sheaf_learner(sheaf_signal, local_edge_index)
            if self.sheaf_identity:
                # Ablation: SAME builder and (augmented) normalization as the learned
                # sheaf, with every restriction map fixed to +/-I (zero Householder
                # parameters give -I; L_uv = -R_u^T R_v is unchanged).  The former
                # closed-form I - D^-1/2 A D^-1/2 differed in normalization
                # (audit finding: it also removed the self-loop term).
                L, trans_maps = builder(torch.zeros_like(maps).detach())
            else:
                L, trans_maps = builder(maps)
                self.sheaf_learner.set_L(trans_maps)

        Z = Z0.view(n_local * self.final_d, -1)
        for layer in range(self.layers):
            Z_in = F.dropout(Z, p=self.dropout, training=self.training)
            y = Z_in.t().reshape(-1, self.final_d)
            y = self.W1(y)
            y = y.reshape(-1, n_local * self.final_d).t()
            y = self.W2(y)
            y = y - torch_sparse.spmm(L[0], L[1], y.size(0), y.size(0), y)
            if self.use_act:
                y = F.elu(y)
            Z = Z + F.softplus(self.log_tau[layer]) * y
        return Z.reshape(n_local, -1)

    # ----- one event step ---------------------------------------------
    def step(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        active_nodes: Optional[torch.Tensor] = None,
        timestamp: Optional[torch.Tensor] = None,
        state: Optional[TemporalMambaState] = None,
        return_state: bool = False,
        edge_type: Optional[torch.Tensor] = None,
        typed_edge_index: Optional[torch.Tensor] = None,
        edge_timestamps: Optional[torch.Tensor] = None,
    ):
        device = x.device
        self._recurrency_advance()
        dynamic_edge_index = self._normalize_event_edge_index(edge_index, device)
        context_edge_index = self.edge_index.to(device) if self.edge_index is not None else None
        prev_state = state if state is not None else self._temporal_state
        prev_memory = (
            prev_state.memory
            if prev_state is not None and prev_state.memory is not None
            else self._initial_memory(device)
        )
        prev_spatial = prev_state.spatial if prev_state is not None else None

        if timestamp is not None and self._prev_timestamp is not None:
            # Subtract in float64: raw epoch-second timestamps (~1e9) lose
            # second-scale gaps to fp32 rounding.
            delta_t = (
                timestamp.to(device=device, dtype=torch.float64)
                - self._prev_timestamp.to(device=device, dtype=torch.float64)
            ).clamp_min(0.0).float()
        else:
            delta_t = torch.zeros((), device=device)

        expansion_edge_index = context_edge_index
        if expansion_edge_index is None or expansion_edge_index.numel() == 0:
            expansion_edge_index = dynamic_edge_index
        active_mask = self._expand_active_nodes(expansion_edge_index, active_nodes)
        local_nodes, local_index = self._global_local_maps(active_mask)

        if local_nodes.numel() == 0:
            event_output = self._empty_event_output(x, prev_spatial)
            next_state = TemporalMambaState(memory=prev_memory, spatial=event_output["spatial"])
            if return_state:
                return event_output, next_state
            return event_output

        local_context = None
        if context_edge_index is not None and context_edge_index.numel() > 0:
            local_context = self._localize_edges(context_edge_index, local_index)
        local_dynamic = None
        if dynamic_edge_index is not None and dynamic_edge_index.numel() > 0:
            local_dynamic = self._localize_edges(dynamic_edge_index, local_index)
        local_edge_index = self._merge_local_edge_indices(
            local_context, local_dynamic, local_nodes.numel(), device
        )
        n_local = local_nodes.numel()

        self._last_x = x
        head_x = self._head_input_features(x)
        local_x = x.index_select(0, local_nodes)
        if self.use_learnable_node_features:
            local_x = local_x + self.node_feature_embedding(local_nodes)
        if self.node_type_ids is not None:
            local_x = local_x + self.node_type_embedding(self.node_type_ids[local_nodes])
        local_x = F.dropout(local_x, p=self.input_dropout, training=self.training)
        local_prev_memory = prev_memory.index_select(0, local_nodes)
        local_prev_spatial = None if prev_spatial is None else prev_spatial.index_select(0, local_nodes)

        # Eq. 13: readout over the PREVIOUS event graph.
        local_prev_edges = None
        if self._prev_event_edge_index is not None and self._prev_event_edge_index.numel() > 0:
            local_prev_edges = self._localize_edges(self._prev_event_edge_index.to(device), local_index)
        z_bar = self._local_feedback(local_prev_spatial, local_prev_edges, n_local, device)

        # psi_u(G_k) on the local subgraph.
        deg = degree(local_edge_index[0], num_nodes=n_local).unsqueeze(-1) if local_edge_index.numel() else torch.zeros(n_local, 1, device=device)
        psi = torch.cat([deg, torch.ones(n_local, 1, device=device, dtype=local_x.dtype)], dim=-1)

        # Eqs. 14-18: every local node is in the active/closure set by
        # construction, so all local memories update.
        q_parts = [local_x, z_bar, psi]
        if self.relation_input_dim > 0:
            # edge_type aligns with the raw event list (typed_edge_index), not
            # with the bidirectionalized dynamic_edge_index.
            typed = typed_edge_index
            if typed is None and edge_type is not None and dynamic_edge_index is not None \
                    and edge_type.numel() == dynamic_edge_index.size(1):
                typed = dynamic_edge_index
            local_dyn_typed, local_types = self._localize_edge_types(
                typed.to(device) if typed is not None else None, edge_type, local_index
            )
            q_parts.append(self._relation_input_signal(
                local_dyn_typed, local_types, n_local, device, local_x.dtype
            ))
        q = torch.cat(q_parts, dim=-1)
        node_gap, lu_l, li_l, seen_u, seen_i = self._node_gaps(timestamp, local_nodes, device)
        gap_supplied = node_gap if node_gap is not None else delta_t
        if self.diag is not None:
            raw_mask = torch.zeros(self.graph_size, dtype=torch.bool, device=device)
            if active_nodes is not None:
                if active_nodes.dtype == torch.bool:
                    raw_mask = active_nodes.to(device)
                else:
                    raw_mask[active_nodes.to(device)] = True
            else:
                raw_mask.fill_(True)
            self.diag.begin_step(float(timestamp) if timestamp is not None else float("nan"),
                                 float(self._prev_timestamp) if self._prev_timestamp is not None else None,
                                 lu_l, li_l, seen_u, seen_i, raw_mask[local_nodes])
        core_off_fast = self.fast_core_off and self.no_memory and self.layers == 0
        if core_off_fast:
            # verified bypass: the SSM transition and the map decoder are never read
            local_memory = local_prev_memory
            h_out = torch.zeros(n_local, self.d_h, device=device, dtype=local_x.dtype)
        else:
            local_memory = self.ssm(local_prev_memory, q, gap_supplied)
            h_out = self.memory_readout(local_memory)
            if self.no_memory:
                h_out = torch.zeros_like(h_out)   # memory advanced but never read
        if self.diag is not None:
            self.diag.end_step()

        # Eqs. 19-23 + Eq. 25, decoded once and held fixed for the interval.
        if self.sheaf_conditioning == "current_only":
            sheaf_signal = self.sheaf_input_proj(local_x)
        else:
            sheaf_signal = h_out
        Z0 = self.P_z(torch.cat([local_x, h_out], dim=-1))
        if core_off_fast:
            local_spatial = Z0
        else:
            local_spatial = self._local_faithful_diffusion(Z0, sheaf_signal, local_edge_index, n_local)
        self._advance_clocks(timestamp, local_nodes, typed_edge_index if typed_edge_index is not None else dynamic_edge_index,
                             edge_timestamps, device)

        next_memory = self._scatter_global_state(prev_memory, local_nodes, local_memory, device)
        next_spatial = self._scatter_global_state(prev_spatial, local_nodes, local_spatial, device)
        next_state = TemporalMambaState(memory=next_memory, spatial=next_spatial)
        event_output = {"spatial": next_spatial, "x": head_x, "node_signal": None}

        self._prev_event_edge_index = dynamic_edge_index.detach() if dynamic_edge_index is not None else None
        self._prev_timestamp = timestamp.detach() if timestamp is not None else None

        if self.use_recurrency_decoder:
            ev = typed_edge_index if typed_edge_index is not None else dynamic_edge_index
            if ev is not None and ev.numel() > 0:
                et = edge_type
                if et is not None and et.numel() != ev.size(1):
                    et = None
                t_now = float(timestamp) if timestamp is not None else 0.0
                n_ev = ev.size(1)
                if edge_timestamps is not None and edge_timestamps.numel() == n_ev:
                    t_events = edge_timestamps.to(device=device, dtype=torch.float64)
                else:
                    t_events = torch.full((n_ev,), t_now, dtype=torch.float64, device=device)
                for name, store in self._rec_stores:
                    r = None if et is None else et.to(device)
                    store.stash(self._rec_channel_keys(name, ev[0].to(device), r, ev[1].to(device)), t_events)

        if self.stateful_temporal and state is None:
            self._temporal_state = TemporalMambaState(
                memory=next_memory.detach(), spatial=next_spatial.detach()
            )

        if return_state:
            return event_output, next_state
        return event_output

    @torch.no_grad()
    def sheaf_residual(self, memory, spatial, src, dst):
        """Transport residual ||R_{e<-u} z_u - R_{e<-v} z_v|| for arbitrary node
        pairs, with maps decoded from the current memory readout exactly as
        for observed edges (identity maps under the ablation). The builder
        needs both directions of every edge, so the query pairs are
        symmetrized and each direction's map is looked up afterwards."""
        device = spatial.device
        src = src.to(device).long(); dst = dst.to(device).long()
        d = self.final_d
        zu = spatial[src].view(-1, d, self.hidden_channels)
        zv = spatial[dst].view(-1, d, self.hidden_channels)
        if self.sheaf_identity or self.spatial_variant == "attention":
            return (zu - zv).flatten(1).norm(dim=-1)
        nodes = torch.unique(torch.cat([src, dst]))
        n_loc = nodes.numel()
        local_index = torch.full((self.graph_size,), -1, dtype=torch.long, device=device)
        local_index[nodes] = torch.arange(n_loc, device=device)
        h = self.memory_readout(memory[nodes])
        sheaf_signal = self.sheaf_input_proj(self._last_x[nodes]) if self.sheaf_conditioning == "current_only" else h
        lu, lv = local_index[src], local_index[dst]
        keep = lu != lv
        und = torch.cat([torch.stack([lu[keep], lv[keep]]), torch.stack([lv[keep], lu[keep]])], dim=1)
        und = torch.unique(und, dim=1)  # sorted lexicographically by (row, col)
        if self.spatial_variant == "node_frame":
            maps = torch.tanh(self.frame_decoder(sheaf_signal)).index_select(0, und[0])
        else:
            maps = self.sheaf_learner(sheaf_signal, und)
        builder = self._make_local_builder(und, n_loc)
        # one orthogonal map per DIRECTED input edge (the builder itself keeps
        # only the lower-triangular half plus its paired reverse)
        E = und.size(1)
        tm = builder.orth_transform(maps).reshape(E, d, d)
        keys = und[0] * n_loc + und[1]  # sorted ascending (unique over dim=1 sorts columns)
        e_uv = torch.searchsorted(keys, lu * n_loc + lv).clamp_max(E - 1)
        e_vu = torch.searchsorted(keys, lv * n_loc + lu).clamp_max(E - 1)
        Ru, Rv = tm[e_uv], tm[e_vu]
        res = (torch.bmm(Ru, zu) - torch.bmm(Rv, zv)).flatten(1).norm(dim=-1)
        res = torch.where(keep, res, (zu - zv).flatten(1).norm(dim=-1))
        return res

    def forward_sequence(
        self,
        snapshots: Sequence[Union[dict, object]],
        initial_state: Optional[TemporalMambaState] = None,
    ):
        outputs = []
        state = initial_state if initial_state is not None else self._temporal_state
        for snapshot in snapshots:
            typed_edge_index = None
            edge_timestamps = None
            if isinstance(snapshot, dict):
                x = snapshot["x"]
                edge_index = snapshot.get("edge_index")
                active_nodes = snapshot.get("active_nodes")
                timestamp = snapshot.get("timestamp")
                edge_type = snapshot.get("edge_types")
                typed_edge_index = snapshot.get("edge_index") if edge_type is not None else None
                edge_timestamps = snapshot.get("edge_timestamps")
            else:
                x = snapshot.x
                edge_index = getattr(snapshot, "edge_index", None)
                active_nodes = getattr(snapshot, "active_nodes", None)
                timestamp = getattr(snapshot, "timestamp", None)
                edge_type = getattr(snapshot, "edge_types", None)
                edge_timestamps = getattr(snapshot, "edge_timestamps", None)
                raw_src = getattr(snapshot, "src", None)
                raw_dst = getattr(snapshot, "dst", None)
                if raw_src is not None and raw_dst is not None and raw_src.numel() > 0:
                    typed_edge_index = torch.stack([raw_src, raw_dst])
            if timestamp is not None and not torch.is_tensor(timestamp):
                timestamp = torch.tensor(timestamp)
            event_output, state = self.step(
                x,
                edge_index=edge_index,
                active_nodes=active_nodes,
                timestamp=timestamp,
                state=state,
                return_state=True,
                edge_type=edge_type,
                typed_edge_index=typed_edge_index,
                edge_timestamps=edge_timestamps,
            )
            outputs.append(event_output)

        if self.stateful_temporal:
            self._temporal_state = TemporalMambaState(
                memory=state.memory.detach() if state is not None and state.memory is not None else None,
                spatial=state.spatial.detach() if state is not None and state.spatial is not None else None,
            )
        return outputs, state
