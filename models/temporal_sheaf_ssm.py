"""Faithful implementation of the Temporal Sheaf Diffusion discrete event-time
algorithm (paper Section 3.2).

This module implements the update equations of the paper literally, replacing the
length-2 pseudo-sequence MambaBlock used by ``MambaSheafDiffusion``:

1. Lagged local spatial feedback (Eq. 13):
       z_bar_{u,k-1} = Readout_u(Z^-_k, G_{k-1})
   computed with the *previous* interval's terminal spatial latent on the
   *previous* graph.

2. Local recurrent input (Eq. 14):
       q_{u,k} = [x_{u,k}; z_bar_{u,k-1}; psi_u(G_k)]

3. Selective SSM with explicit zero-order-hold discretization (Eqs. 15-18):
       (Delta_{u,k}, B_{u,k}) = Gamma_eta(q_{u,k}; Delta_k)
       A_bar = exp(Delta_{u,k} A)
       B_bar q = A^{-1} (A_bar - I) B_{u,k} q_{u,k}
       h_{u,k} = A_bar h_{u,k-1} + B_bar q_{u,k}          for u in A_k
   with a shared continuous-time generator A initialized from HiPPO-LegS and the
   physical event gap Delta_k = t_k - t_{k-1} entering the step-size selector.
   The recurrence runs over the true event index k: node memory h carries the
   entire event history, not a per-step 2-token reconstruction.

4. Edgewise sheaf decoding (Eq. 19), performed ONCE per event time and held
   fixed while the spatial latent evolves on [t_k, t_{k+1}).

5. Spatial latent initialization (Eq. 25):
       z_u(t_k) = P_z [x_{u,k}; h_{u,k}]

6. Discretized sheaf diffusion of the paper's vector field (Eqs. 22-23):
       Zdot = sigma((I - L_tilde_k)(I (x) W1) Z W2)
   integrated with forward Euler steps Z <- Z + tau_l * f_diff(Z), with a single
   (W1, W2) pair held fixed within the interval (the field is autonomous).
   ``diffusion_update='direct'`` instead applies the Hansen-Gebhart discrete
   layer Z <- sigma((I - L_tilde)(I (x) W1) Z W2).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected

from . import laplacian_builders as lb
from .sheaf_models import LocalConcatSheafLearner


def hippo_legs_matrix(n: int) -> torch.Tensor:
    """The (negated) HiPPO-LegS generator A used as SSM initialization.

    A[i, j] = -sqrt((2i+1)(2j+1))  for i > j
              -(i+1)               for i == j
              0                    for i < j
    (Gu et al., 2020, Eq. for LegS; the sign makes the dynamics contractive.)
    """
    idx = torch.arange(n, dtype=torch.float64)
    scale = torch.sqrt(2.0 * idx + 1.0)
    A = scale.unsqueeze(1) * scale.unsqueeze(0)
    A = -torch.tril(A, diagonal=-1)
    A = A - torch.diag(idx + 1.0)
    return A.to(torch.float32)


def hippo_legs_b(n: int) -> torch.Tensor:
    """HiPPO-LegS input vector B[i] = sqrt(2i+1)."""
    idx = torch.arange(n, dtype=torch.float64)
    return torch.sqrt(2.0 * idx + 1.0).to(torch.float32)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


@dataclass
class TemporalSheafState:
    """Carries everything the next event step needs from the current one."""

    memory: Optional[torch.Tensor]  # (N, d_h) SSM states H_k
    spatial: Optional[torch.Tensor]  # (N, d_s * f) terminal spatial latent Z^-
    prev_edge_index: Optional[torch.Tensor] = None  # G_{k-1} for Readout_u
    prev_timestamp: Optional[torch.Tensor] = None  # t_{k-1} for Delta_k

    def detach(self) -> "TemporalSheafState":
        return TemporalSheafState(
            memory=self.memory.detach() if self.memory is not None else None,
            spatial=self.spatial.detach() if self.spatial is not None else None,
            prev_edge_index=self.prev_edge_index,
            prev_timestamp=self.prev_timestamp,
        )


class SelectiveZOHSSM(nn.Module):
    """Shared nodewise selective SSM with exact zero-order-hold discretization.

    State h in R^{d_h}; input q in R^{d_q}. The generator A (d_h x d_h) is
    shared across nodes and initialized from HiPPO-LegS. The selector produces
    a per-node positive step size Delta_{u,k} (conditioned on both q_{u,k} and
    the physical event gap Delta_k) and an input-dependent B_{u,k} in
    R^{d_h x d_q} whose bias is initialized from the HiPPO-LegS B broadcast
    over input channels.
    """

    def __init__(self, d_state: int, d_input: int, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_cap: float = 0.25):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input
        # Hard cap on Delta_{u,k}: the LegS generator is non-normal, and
        # exp(dt*A) exhibits transient amplification for moderate dt; the paper
        # only constrains Delta_{u,k} > 0, so a stability cap is admissible.
        self.dt_cap = float(dt_cap)

        self.A = nn.Parameter(hippo_legs_matrix(d_state))

        # Selector Gamma_eta: q -> (Delta_{u,k}, B_{u,k}).
        self.dt_proj = nn.Linear(d_input, 1)
        nn.init.zeros_(self.dt_proj.weight)
        dt_init = math.exp(
            torch.empty(1).uniform_(math.log(dt_min), math.log(dt_max)).item()
        )
        with torch.no_grad():
            self.dt_proj.bias.fill_(_inv_softplus(torch.tensor(dt_init)).item())
        # Physical-time conditioning: dt pre-activation gets a learned multiple
        # of log1p(Delta_k / s) with s > 0 (resolves the paper's Eq.11-vs-Eq.15
        # ambiguity by feeding Delta_k into the selector input).
        self.dt_time_weight = nn.Parameter(torch.ones(1))
        self.dt_time_log_scale = nn.Parameter(torch.zeros(1))

        self.B_selector = nn.Linear(d_input, d_state * d_input)
        nn.init.normal_(self.B_selector.weight, std=1e-3)
        with torch.no_grad():
            base_b = hippo_legs_b(d_state).unsqueeze(1).expand(d_state, d_input)
            self.B_selector.bias.copy_((base_b / math.sqrt(d_input)).reshape(-1))

    def step_size(self, q: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Delta_{u,k} > 0 per node, from input content and physical gap."""
        time_scale = F.softplus(self.dt_time_log_scale) + 1e-4
        time_term = self.dt_time_weight * torch.log1p(delta_t.clamp_min(0.0) / time_scale)
        dt = F.softplus(self.dt_proj(q).squeeze(-1) + time_term)
        return dt.clamp(max=self.dt_cap)

    # Largest node batch processed at once: exp(dt A) materializes an
    # (n, d_h, d_h) tensor plus autograd intermediates, which is prohibitive
    # for snapshot-local sets with hundreds of thousands of nodes. Chunks are
    # gradient-checkpointed during training so peak memory is one chunk.
    chunk_size = 32768

    def forward(self, h_prev: torch.Tensor, q: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        n = q.size(0)
        if n <= self.chunk_size:
            return self._forward_impl(h_prev, q, delta_t)
        outputs = []
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            h_c, q_c = h_prev[start:stop], q[start:stop]
            if self.training and torch.is_grad_enabled():
                out = torch.utils.checkpoint.checkpoint(
                    self._forward_impl, h_c, q_c, delta_t, use_reentrant=False
                )
            else:
                out = self._forward_impl(h_c, q_c, delta_t)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def _forward_impl(self, h_prev: torch.Tensor, q: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """One exact ZOH update h_k = exp(dA) h_{k-1} + A^{-1}(exp(dA)-I) B q.

        h_prev: (N, d_h), q: (N, d_q), delta_t: scalar tensor (physical gap).
        """
        n = q.size(0)
        dt = self.step_size(q, delta_t)  # (N,)

        A = self.A
        dA = dt.view(n, 1, 1) * A.unsqueeze(0)  # (N, d_h, d_h)
        A_bar = torch.linalg.matrix_exp(dA)

        B = self.B_selector(q).view(n, self.d_state, self.d_input)  # (N, d_h, d_q)
        Bq = torch.bmm(B, q.unsqueeze(-1)).squeeze(-1)  # (N, d_h)

        # dt * phi1(dt A) B q = A^{-1} (A_bar - I) B q, computed by solving
        # A y = (A_bar - I) B q. A is HiPPO-LegS-initialized (lower triangular
        # with nonzero diagonal), well-conditioned at this size.
        rhs = torch.bmm(A_bar - torch.eye(self.d_state, device=q.device, dtype=q.dtype).unsqueeze(0), Bq.unsqueeze(-1))
        Bbar_q = torch.linalg.solve(A.unsqueeze(0).expand(n, -1, -1), rhs).squeeze(-1)

        return torch.bmm(A_bar, h_prev.unsqueeze(-1)).squeeze(-1) + Bbar_q


class FaithfulTemporalSheafDiffusion(nn.Module):
    """Discrete event-time temporal sheaf diffusion, following paper Sec. 3.2."""

    def __init__(self, edge_index, args):
        super().__init__()
        self.graph_size = int(args["graph_size"])
        self.input_dim = int(args["input_dim"])
        self.output_dim = int(args["output_dim"])
        self.d = int(args["d"])  # stalk dimension d_s
        self.hidden_channels = int(args["hidden_channels"])  # feature channels f
        self.layers = int(args["layers"])  # number of Euler steps / layers
        self.dropout = float(args.get("dropout", 0.0))
        self.input_dropout = float(args.get("input_dropout", 0.0))
        self.use_act = bool(args.get("use_act", True))
        self.sheaf_act = str(args.get("sheaf_act", "tanh"))
        self.orth_trans = str(args.get("orth", "householder"))
        self.add_lp = bool(args.get("add_lp", False))
        self.add_hp = bool(args.get("add_hp", False))
        self.closure_hops = int(args.get("closure_hops", 1))
        self.stateful_temporal = bool(args.get("stateful_temporal", False))
        self.diffusion_update = str(args.get("diffusion_update", "euler"))
        if self.diffusion_update not in ("euler", "direct"):
            raise ValueError("diffusion_update must be 'euler' or 'direct'")

        self.d_h = int(args.get("temporal_d_model", 64))  # memory dim d_h
        self.d_z = int(args.get("feedback_dim", 32))  # lagged feedback dim d_z
        self.psi_dim = 2  # psi_u(G_k) = [degree, active flag]
        self.psi_log_degree = bool(args.get("psi_log_degree", False))

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1
        self.hidden_dim = self.hidden_channels * self.final_d  # d_s * f flattened

        d_q = self.input_dim + self.d_z + self.psi_dim
        self.d_q = d_q

        # Temporal module Psi_eta (Eqs. 15-18).
        self.ssm = SelectiveZOHSSM(
            self.d_h,
            d_q,
            dt_min=float(args.get("dt_min", 1e-3)),
            dt_max=float(args.get("dt_max", 0.1)),
            dt_cap=float(args.get("dt_cap", 0.25)),
        )

        # Optional SSM output map C (Eq. 5: o = C m): when enabled, downstream
        # modules consume a normalized readout of the memory rather than the
        # raw ZOH state, decoupling their input scale from the step-size regime.
        self.use_memory_readout = bool(args.get("memory_readout", False))
        self.memory_readout = nn.LayerNorm(self.d_h) if self.use_memory_readout else nn.Identity()

        # Reviewer control: 'current_only' removes ONLY the sheaf's history
        # conditioning — restriction maps are decoded from a history-free
        # encoding of the current observation, while the memory branch still
        # drives P_z and the lagged feedback. 'history' is the paper's model.
        self.sheaf_conditioning = str(args.get("sheaf_conditioning", "history"))
        if self.sheaf_conditioning not in ("history", "current_only"):
            raise ValueError("sheaf_conditioning must be 'history' or 'current_only'")
        if self.sheaf_conditioning == "current_only":
            self.sheaf_input_proj = nn.Linear(self.input_dim, self.d_h)

        # Readout_u projection for z_bar (Eq. 13).
        self.feedback_proj = nn.Linear(self.hidden_dim, self.d_z)

        # Edgewise decoder Phi_theta (Eq. 19): restriction maps from endpoint
        # memories, parameterized as orthogonal maps (as in the codebase).
        self.sheaf_learner = LocalConcatSheafLearner(
            self.d_h, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act
        )
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size,
            self._prepare_edge_index(edge_index),
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        # P_z (Eq. 25) and diffusion parameters (Eq. 23): one W1, one W2,
        # held fixed within the interval; per-step learnable Euler step tau_l.
        self.P_z = nn.Linear(self.input_dim + self.d_h, self.hidden_dim)
        self.W1 = nn.Linear(self.final_d, self.final_d, bias=False)
        nn.init.eye_(self.W1.weight)
        self.W2 = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        nn.init.orthogonal_(self.W2.weight)
        self.log_tau = nn.Parameter(torch.zeros(self.layers))

        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self._temporal_state: Optional[TemporalSheafState] = None

    # ----- utilities -------------------------------------------------------
    def get_param_size(self):
        if self.orth_trans in ["matrix_exp", "cayley"]:
            return self.d * (self.d + 1) // 2
        return self.d * (self.d - 1) // 2

    def grouped_parameters(self):
        sheaf, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name or "ssm" in name:
                sheaf.append(param)
            else:
                others.append(param)
        return sheaf, others

    def reset_temporal_state(self):
        self._temporal_state = None

    def _prepare_edge_index(self, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=self.graph_size)
        edge_index, _ = coalesce(edge_index, None, self.graph_size, self.graph_size)
        return edge_index.contiguous()

    def _expand_active_nodes(self, edge_index, active_nodes):
        active_mask = torch.zeros(self.graph_size, dtype=torch.bool, device=edge_index.device)
        if active_nodes is None:
            active_mask.fill_(True)
            return active_mask
        if active_nodes.dtype == torch.bool:
            active_mask = active_nodes.clone()
        else:
            active_mask[active_nodes] = True
        frontier = active_mask.clone()
        row, col = edge_index
        for _ in range(max(self.closure_hops, 0)):
            outgoing = frontier[row]
            if not torch.any(outgoing):
                break
            new_nodes = torch.unique(col[outgoing])
            new_nodes = new_nodes[~active_mask[new_nodes]]
            if new_nodes.numel() == 0:
                break
            active_mask[new_nodes] = True
            frontier = torch.zeros_like(active_mask)
            frontier[new_nodes] = True
        return active_mask

    # ----- Eq. 13: lagged local spatial feedback ---------------------------
    def _lagged_feedback(self, state: Optional[TemporalSheafState], device) -> torch.Tensor:
        if state is None or state.spatial is None:
            return torch.zeros(self.graph_size, self.d_z, device=device)
        z_prev = state.spatial
        edge_index = state.prev_edge_index
        if edge_index is not None and edge_index.numel() > 0:
            row, col = edge_index
            summary = torch.zeros_like(z_prev)
            summary.index_add_(0, row, z_prev[col])
            summary = summary + z_prev
            deg = degree(row, num_nodes=self.graph_size).clamp_min(0).unsqueeze(-1) + 1.0
            z_read = summary / deg
        else:
            z_read = z_prev
        return self.feedback_proj(z_read)

    # ----- psi_u(G_k) ------------------------------------------------------
    def _topology_descriptor(self, edge_index, active_mask):
        row = edge_index[0]
        deg = degree(row, num_nodes=self.graph_size)
        if self.psi_log_degree:
            deg = torch.log1p(deg)
        deg = deg.unsqueeze(-1)
        active = active_mask.to(dtype=deg.dtype).unsqueeze(-1)
        return torch.cat([deg, active], dim=-1)

    # ----- Eqs. 19-23: sheaf decode + diffusion -----------------------------
    def _decode_sheaf(self, memory, edge_index):
        if (
            self.laplacian_builder.edge_index.size() != edge_index.size()
            or not torch.equal(self.laplacian_builder.edge_index, edge_index)
        ):
            self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)
        maps = self.sheaf_learner(memory, edge_index)
        L, trans_maps = self.laplacian_builder(maps)
        self.sheaf_learner.set_L(trans_maps)
        return L

    def _stalk_mix(self, x):
        # (I_{n_k} (x) W1) Z: mix within each node's stalk block.
        y = x.t().reshape(-1, self.final_d)
        y = self.W1(y)
        return y.reshape(-1, self.graph_size * self.final_d).t()

    def _f_diff(self, Z, L):
        # sigma((I - L_tilde)(I (x) W1) Z W2), Eq. 23.
        y = self._stalk_mix(Z)
        y = self.W2(y)
        y = y - torch_sparse.spmm(L[0], L[1], y.size(0), y.size(0), y)
        if self.use_act:
            y = F.elu(y)
        return y

    def _diffuse(self, Z0, L):
        Z = Z0
        for layer in range(self.layers):
            Z_in = F.dropout(Z, p=self.dropout, training=self.training)
            field = self._f_diff(Z_in, L)
            if self.diffusion_update == "euler":
                tau = F.softplus(self.log_tau[layer])
                Z = Z + tau * field
            else:
                Z = field
        return Z

    # ----- one event step (Sec. 3.2) ---------------------------------------
    def step(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        active_nodes: Optional[torch.Tensor] = None,
        timestamp: Optional[torch.Tensor] = None,
        state: Optional[TemporalSheafState] = None,
        return_state: bool = False,
    ):
        device = x.device
        if edge_index is None:
            edge_index = self.laplacian_builder.edge_index
        edge_index = self._prepare_edge_index(edge_index).to(device)
        prev = state if state is not None else self._temporal_state

        # Delta_k = t_k - t_{k-1} (0 at k = 0 or without timestamps).
        if timestamp is not None and prev is not None and prev.prev_timestamp is not None:
            delta_t = (timestamp.to(device).float() - prev.prev_timestamp.to(device).float()).clamp_min(0.0)
        else:
            delta_t = torch.zeros((), device=device)

        x_in = F.dropout(x, p=self.input_dropout, training=self.training)

        # Eq. 13 + Eq. 14.
        z_bar = self._lagged_feedback(prev, device)
        active_mask = self._expand_active_nodes(edge_index, active_nodes)
        psi = self._topology_descriptor(edge_index, active_mask)
        q = torch.cat([x_in, z_bar, psi], dim=-1)

        # Eqs. 15-18, gated to active nodes (Eq. 11: u in A_k).
        h_prev = prev.memory if prev is not None and prev.memory is not None else torch.zeros(
            self.graph_size, self.d_h, device=device
        )
        h_updated = self.ssm(h_prev, q, delta_t)
        h = torch.where(active_mask.unsqueeze(-1), h_updated, h_prev)
        h_out = self.memory_readout(h)

        # Eq. 19-21: decode sheaf once, hold fixed for the interval.
        if self.sheaf_conditioning == "current_only":
            sheaf_signal = self.sheaf_input_proj(x_in)
        else:
            sheaf_signal = h_out
        L = self._decode_sheaf(sheaf_signal, edge_index)

        # Eq. 25 + Eqs. 22-23.
        Z0 = self.P_z(torch.cat([x_in, h_out], dim=-1))
        Z = self._diffuse(Z0.view(self.graph_size * self.final_d, -1), L)
        Z = Z.reshape(self.graph_size, -1)

        logits = F.log_softmax(self.lin2(Z), dim=1)

        next_state = TemporalSheafState(
            memory=h,
            spatial=Z,
            prev_edge_index=edge_index,
            prev_timestamp=timestamp.detach() if timestamp is not None else None,
        )
        if self.stateful_temporal and state is None:
            self._temporal_state = next_state.detach()

        if return_state:
            return logits, next_state
        return logits

    def forward(self, x, edge_index: Optional[torch.Tensor] = None):
        return self.step(x, edge_index=edge_index, return_state=False)

    def forward_sequence(
        self,
        snapshots: Sequence[Union[dict, object]],
        initial_state: Optional[TemporalSheafState] = None,
    ):
        outputs: List[torch.Tensor] = []
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
            logits, state = self.step(
                x,
                edge_index=edge_index,
                active_nodes=active_nodes,
                timestamp=timestamp,
                state=state,
                return_state=True,
            )
            outputs.append(logits)

        if self.stateful_temporal:
            self._temporal_state = state.detach() if state is not None else None
        return outputs, state
