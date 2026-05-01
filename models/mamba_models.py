from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils import degree

from .disc_models import DiscreteBundleSheafDiffusion

try:
    from mamba_ssm.modules.mamba_simple import Mamba as _RealMamba
except Exception:  # pragma: no cover - fallback used when mamba is unavailable.
    _RealMamba = None


@dataclass
class TemporalMambaState:
    memory: Optional[torch.Tensor]
    spatial: Optional[torch.Tensor]


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.use_real = _RealMamba is not None
        self.input_proj = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.impl = _RealMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if self.use_real else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl is not None and x.is_cuda:
            return self.impl(x)

        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = F.silu(x)
        x = self.output_proj(x)
        return self.norm(x)

class NodewiseMambaLearner(nn.Module):
    """Shared nodewise recurrent learner driven by a Mamba block."""

    def __init__(self, hidden_dim: int, topo_dim: int = 2, d_model: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim
        self.d_model = d_model or hidden_dim

        self.memory_proj = nn.Linear(hidden_dim, self.d_model)
        self.input_proj = nn.Linear(hidden_dim * 3 + topo_dim, self.d_model)
        self.sequence_model = MambaBlock(self.d_model)
        self.output_proj = nn.Linear(self.d_model, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.topology_proj = nn.Linear(topo_dim, topo_dim)

    def forward(
        self,
        previous_memory: torch.Tensor,
        node_signal: torch.Tensor,
        lagged_readout: torch.Tensor,
        topology: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topology = self.topology_proj(topology)
        fused = torch.cat([node_signal, lagged_readout, previous_memory, topology], dim=-1)
        fused = self.input_proj(fused)
        sequence = torch.stack([self.memory_proj(previous_memory), fused], dim=1)
        updated = self.sequence_model(sequence)[:, -1, :]
        updated = self.output_proj(updated)
        updated = self.output_norm(updated)

        if active_mask is not None:
            updated = torch.where(active_mask.unsqueeze(-1), updated, previous_memory)

        return updated


class EdgewiseMambaLearner(nn.Module):
    """Backwards-compatible edgewise learner used by the existing benchmark path."""

    def __init__(self, in_dim: int, out_dim: int, d_model: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.mamba = MambaBlock(d_model)
        self.output_proj = nn.Linear(d_model, out_dim)
        self.L = None

    def __len__(self):
        return len(self.L) if self.L is not None else 0

    def forward(self, x, edge_index):
        _, col = edge_index
        edge_feat = x[col]
        h = self.input_proj(edge_feat)
        h = h.unsqueeze(0)
        h = self.mamba(h).squeeze(0)
        return self.output_proj(h)

    def set_L(self, L):
        self.L = L


class Mamba_learner(EdgewiseMambaLearner):
    pass


class MambaSheafDiffusion(DiscreteBundleSheafDiffusion):
    """Discrete sheaf diffusion with a nodewise Mamba recurrent learner."""

    def __init__(self, edge_index, args):
        super(MambaSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.stateful_temporal = bool(args.get('stateful_temporal', False))
        self.closure_hops = int(args.get('closure_hops', 1))
        self.temporal_d_model = int(args.get('temporal_d_model', self.hidden_dim))

        self.nodewise_learner = NodewiseMambaLearner(
            hidden_dim=self.hidden_dim,
            topo_dim=2,
            d_model=self.temporal_d_model,
        )
        self._temporal_state: Optional[TemporalMambaState] = None

    def grouped_parameters(self):
        sheaf_learner_params, other_params = [], []
        for name, param in self.named_parameters():
            if any(key in name for key in ["sheaf_learner", "nodewise_learner", "temporal_learner"]):
                sheaf_learner_params.append(param)
            else:
                other_params.append(param)
        assert len(sheaf_learner_params) > 0
        assert len(sheaf_learner_params) + len(other_params) == len(list(self.parameters()))
        return sheaf_learner_params, other_params

    def reset_temporal_state(self):
        self._temporal_state = None

    def _prepare_edge_index(self, edge_index):
        return self.edge_index if edge_index is None else edge_index

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        return x

    def _expand_active_nodes(self, edge_index: torch.Tensor, active_nodes: Optional[torch.Tensor]) -> torch.Tensor:
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

    def _lagged_local_readout(self, spatial_state: Optional[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        if spatial_state is None:
            return torch.zeros(self.graph_size, self.hidden_dim, device=edge_index.device)

        row, col = edge_index
        summary = torch.zeros_like(spatial_state)
        summary.index_add_(0, row, spatial_state[col])
        summary = summary + spatial_state
        deg = degree(row, num_nodes=self.graph_size).clamp_min(0).unsqueeze(-1) + 1.0
        return summary / deg

    def _topology_descriptor(self, edge_index: torch.Tensor, active_mask: Optional[torch.Tensor]) -> torch.Tensor:
        row = edge_index[0]
        deg = degree(row, num_nodes=self.graph_size).unsqueeze(-1)
        if active_mask is None:
            active_mask = torch.ones(self.graph_size, dtype=torch.bool, device=edge_index.device)
        active = active_mask.to(dtype=deg.dtype).unsqueeze(-1)
        return torch.cat([deg, active], dim=-1)

    def _initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.graph_size, self.hidden_dim, device=device)

    def _apply_discrete_diffusion(self, node_memory: torch.Tensor, edge_index: torch.Tensor):
        x = node_memory.view(self.graph_size * self.final_d, -1)
        x0 = x
        L = None

        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0.0, training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, edge_index)
                edge_weights = self.weight_learners[layer](x_maps, edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)
            left = self.lin_left_weights[layer] if self.left_weights else None
            right = self.lin_right_weights[layer] if self.right_weights else None
            x = self.left_right_linear(x, left, right)
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        spatial_state = x.reshape(self.graph_size, -1)
        logits = self.lin2(spatial_state)
        return F.log_softmax(logits, dim=1), spatial_state

    def step(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        active_nodes: Optional[torch.Tensor] = None,
        state: Optional[TemporalMambaState] = None,
        return_state: bool = False,
    ):
        edge_index = self._prepare_edge_index(edge_index).to(x.device)
        prev_state = state if state is not None else self._temporal_state

        node_signal = self._encode_nodes(x)
        prev_memory = prev_state.memory if prev_state is not None and prev_state.memory is not None else self._initial_memory(x.device)
        lagged_readout = self._lagged_local_readout(prev_state.spatial if prev_state is not None else None, edge_index)
        active_mask = self._expand_active_nodes(edge_index, active_nodes)
        topology = self._topology_descriptor(edge_index, active_mask)

        node_memory = self.nodewise_learner(
            previous_memory=prev_memory,
            node_signal=node_signal,
            lagged_readout=lagged_readout,
            topology=topology,
            active_mask=active_mask,
        )

        logits, spatial_state = self._apply_discrete_diffusion(node_memory, edge_index)
        next_state = TemporalMambaState(memory=node_memory, spatial=spatial_state)

        if self.stateful_temporal and state is None:
            self._temporal_state = TemporalMambaState(
                memory=node_memory.detach(),
                spatial=spatial_state.detach(),
            )

        if return_state:
            return logits, next_state
        return logits

    def forward(self, x, edge_index: Optional[torch.Tensor] = None):
        return self.step(x, edge_index=edge_index, return_state=False)

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
                active_nodes = snapshot.get("active_nodes")
            else:
                x = snapshot.x
                active_nodes = getattr(snapshot, "active_nodes", None)

            logits, state = self.step(
                x,
                active_nodes=active_nodes,
                state=state,
                return_state=True,
            )
            outputs.append(logits)

        if self.stateful_temporal:
            self._temporal_state = TemporalMambaState(
                memory=state.memory.detach() if state is not None and state.memory is not None else None,
                spatial=state.spatial.detach() if state is not None and state.spatial is not None else None,
            )

        return outputs, state


class TemporalEdgeMambaSheafDiffusion(MambaSheafDiffusion):
    """Temporal Mamba sheaf diffusion with an explicit destination-label head.

    This keeps the same spatial-temporal node dynamics as ``MambaSheafDiffusion``
    while interpreting the output classes as a contiguous destination vocabulary
    rather than generic node-property classes.
    """

    def __init__(self, edge_index, args):
        args = dict(args)
        destination_offset = int(args.get("destination_offset", 0))
        destination_size = int(args.get("destination_size", args["output_dim"]))
        args["output_dim"] = destination_size
        super().__init__(edge_index, args)
        self.destination_offset = destination_offset
        self.destination_size = destination_size

    def global_to_local_destination(self, dst: torch.Tensor) -> torch.Tensor:
        return dst.long() - self.destination_offset

    def local_to_global_destination(self, dst_local: torch.Tensor) -> torch.Tensor:
        return dst_local.long() + self.destination_offset


# class MambaSheafLearner(SheafDiffusion):
#     def __init__(self, edge_index, args):
#         super(MambaSheafLearner, self).__init__(edge_index, args)
#         assert args['d'] > 0

#         self.lin_right_weights = nn.ModuleList()
#         self.lin_left_weights = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()

#         if self.right_weights:
#             for i in range(self.layers):
#                 self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
#                 nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
#         if self.left_weights:
#             for i in range(self.layers):
#                 self.lin_left_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        
#         self.sheaf_learners = Mamba_learner(
#             in_dim=self.hidden_dim,
#             out_dim=self.d
#         )

#         self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
#                                                          normalised=self.normalised,
#                                                          deg_normalised=self.deg_normalised,
#                                                          add_hp=self.add_hp, add_lp=self.add_lp)

#         self.epsilons = nn.ParameterList()
#         for i in range(self.layers):
#             self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

#         self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
#         if self.second_linear:
#             self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

#     def forward(self, x):
#         x = F.dropout(x, p=self.input_dropout, training=self.training)
#         x = self.lin1(x)
#         if self.use_act:
#             x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         if self.second_linear:
#             x = self.lin12(x)
#         x = x.view(self.graph_size * self.final_d, -1)

#         x0 = x
#         for layer in range(self.layers):
#             if layer == 0 or self.nonlinear:
#                 x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
#                 maps = self.sheaf_learners(x_maps.reshape(self.graph_size, -1), self.edge_index)  # ✅ 수정됨
#                 L, trans_maps = self.laplacian_builder(maps)
#                 self.sheaf_learners.set_L(trans_maps)  # 그대로 둠 (옵션)

#             x = F.dropout(x, p=self.dropout, training=self.training)

#             if self.left_weights:
#                 x = x.t().reshape(-1, self.final_d)
#                 x = self.lin_left_weights[layer](x)
#                 x = x.reshape(-1, self.graph_size, self.final_d).t()
#             if self.right_weights:
#                 x = self.lin_right_weights[layer](x)

#             x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

#             if self.use_act:
#                 x = F.elu(x)
            
#             coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
#             x0 = coeff * x0 - x
#             x = x0

#         x = x.reshape(self.graph_size, -1)
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=1)


class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # [B, N, D]
        x = self.in_proj(x)
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.out_proj(x)
        return self.norm(x)
