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
from .temporal_sheaf_ssm import SelectiveZOHSSM


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

        self.ssm = SelectiveZOHSSM(
            self.d_h,
            d_q,
            dt_min=float(args.get("dt_min", 1e-3)),
            dt_max=float(args.get("dt_max", 0.1)),
            dt_cap=float(args.get("dt_cap", 0.25)),
        )
        # Memory readout C (Eq. 5): on by default here — on this protocol the
        # recurrence is trained, where the readout was decisively better on
        # tgbn-trade at native resolution.
        self.use_memory_readout = bool(args.get("memory_readout", True))
        self.memory_readout = nn.LayerNorm(self.d_h) if self.use_memory_readout else nn.Identity()

        self.sheaf_conditioning = str(args.get("sheaf_conditioning", "history"))
        if self.sheaf_conditioning not in ("history", "current_only"):
            raise ValueError("sheaf_conditioning must be 'history' or 'current_only'")
        if self.sheaf_conditioning == "current_only":
            self.sheaf_input_proj = nn.Linear(self.input_dim, self.d_h)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.d_h, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act
        )
        self.feedback_proj = nn.Linear(self.hidden_dim, self.d_z)
        self.P_z = nn.Linear(self.input_dim + self.d_h, self.hidden_dim)
        self.W1 = nn.Linear(self.final_d, self.final_d, bias=False)
        nn.init.eye_(self.W1.weight)
        self.W2 = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        nn.init.orthogonal_(self.W2.weight)
        self.log_tau = nn.Parameter(torch.zeros(self.layers))

        self._prev_event_edge_index: Optional[torch.Tensor] = None
        self._prev_timestamp: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset_temporal_state(self):
        super().reset_temporal_state()
        self._prev_event_edge_index = None
        self._prev_timestamp = None

    def _initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.graph_size, self.d_h, device=device)

    # ----- faithful local computations --------------------------------
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

    def _local_faithful_diffusion(self, Z0, sheaf_signal, local_edge_index, n_local):
        if local_edge_index.numel() == 0:
            return Z0
        builder = self._make_local_builder(local_edge_index, n_local)
        maps = self.sheaf_learner(sheaf_signal, local_edge_index)
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
    ):
        device = x.device
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
            delta_t = (timestamp.to(device).float() - self._prev_timestamp.to(device).float()).clamp_min(0.0)
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

        local_x = F.dropout(x.index_select(0, local_nodes), p=self.input_dropout, training=self.training)
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
        q = torch.cat([local_x, z_bar, psi], dim=-1)
        local_memory = self.ssm(local_prev_memory, q, delta_t)
        h_out = self.memory_readout(local_memory)

        # Eqs. 19-23 + Eq. 25, decoded once and held fixed for the interval.
        if self.sheaf_conditioning == "current_only":
            sheaf_signal = self.sheaf_input_proj(local_x)
        else:
            sheaf_signal = h_out
        Z0 = self.P_z(torch.cat([local_x, h_out], dim=-1))
        local_spatial = self._local_faithful_diffusion(Z0, sheaf_signal, local_edge_index, n_local)

        next_memory = self._scatter_global_state(prev_memory, local_nodes, local_memory, device)
        next_spatial = self._scatter_global_state(prev_spatial, local_nodes, local_spatial, device)
        next_state = TemporalMambaState(memory=next_memory, spatial=next_spatial)
        event_output = {"spatial": next_spatial, "x": x, "node_signal": None}

        self._prev_event_edge_index = dynamic_edge_index.detach() if dynamic_edge_index is not None else None
        self._prev_timestamp = timestamp.detach() if timestamp is not None else None

        if self.stateful_temporal and state is None:
            self._temporal_state = TemporalMambaState(
                memory=next_memory.detach(), spatial=next_spatial.detach()
            )

        if return_state:
            return event_output, next_state
        return event_output

    def forward_sequence(
        self,
        snapshots: Sequence[Union[dict, object]],
        initial_state: Optional[TemporalMambaState] = None,
    ):
        outputs = []
        state = initial_state if initial_state is not None else self._temporal_state
        for snapshot in snapshots:
            if isinstance(snapshot, dict):
                x = snapshot["x"]
                edge_index = snapshot.get("edge_index")
                active_nodes = snapshot.get("active_nodes")
                timestamp = snapshot.get("timestamp")
            else:
                x = snapshot.x
                edge_index = getattr(snapshot, "edge_index", None)
                active_nodes = getattr(snapshot, "active_nodes", None)
                timestamp = getattr(snapshot, "timestamp", None)
            if timestamp is not None and not torch.is_tensor(timestamp):
                timestamp = torch.tensor(timestamp)
            event_output, state = self.step(
                x,
                edge_index=edge_index,
                active_nodes=active_nodes,
                timestamp=timestamp,
                state=state,
                return_state=True,
            )
            outputs.append(event_output)

        if self.stateful_temporal:
            self._temporal_state = TemporalMambaState(
                memory=state.memory.detach() if state is not None and state.memory is not None else None,
                spatial=state.spatial.detach() if state is not None and state.spatial is not None else None,
            )
        return outputs, state
