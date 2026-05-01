from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected

from . import laplacian_builders as lb
from .mamba_models import MambaSheafDiffusion, TemporalMambaState


class SparseTemporalMambaSheafDiffusion(MambaSheafDiffusion):
    """Temporal Mamba sheaf diffusion that operates on snapshot-local subgraphs.

    The existing temporal implementation keeps a global node state, but it rebuilds
    the sheaf Laplacian and applies diffusion over the full graph for every
    snapshot. This class keeps the global recurrent state while restricting the
    expensive sheaf computations to the nodes touched by each event snapshot and
    their optional closure neighborhood.
    """

    def _prepare_edge_index(self, edge_index):
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, self.graph_size, self.graph_size)
        return edge_index.contiguous()

    def _left_right_linear_local(self, x, left, right, local_graph_size):
        if self.left_weights and left is not None:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, local_graph_size * self.final_d).t()

        if self.right_weights and right is not None:
            x = right(x)

        return x

    def _local_lagged_readout(
        self,
        spatial_state: Optional[torch.Tensor],
        local_edge_index: torch.Tensor,
        local_graph_size: int,
    ) -> torch.Tensor:
        if spatial_state is None:
            return torch.zeros(local_graph_size, self.hidden_dim, device=local_edge_index.device)

        row, col = local_edge_index
        summary = torch.zeros_like(spatial_state)
        summary.index_add_(0, row, spatial_state[col])
        summary = summary + spatial_state
        deg = degree(row, num_nodes=local_graph_size).clamp_min(0).unsqueeze(-1) + 1.0
        return summary / deg

    def _make_local_builder(self, edge_index: torch.Tensor, local_graph_size: int):
        return lb.NormConnectionLaplacianBuilder(
            local_graph_size,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

    def _apply_local_diffusion(self, local_memory: torch.Tensor, local_edge_index: torch.Tensor):
        local_graph_size = local_memory.size(0)

        if local_edge_index.numel() == 0:
            return local_memory

        x = local_memory.view(local_graph_size * self.final_d, -1)
        x0 = x
        local_builder = self._make_local_builder(local_edge_index, local_graph_size)
        L = None

        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0.0, training=self.training)
                x_maps = x_maps.reshape(local_graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, local_edge_index)
                if self.use_edge_weights:
                    self.weight_learners[layer].update_edge_index(local_edge_index)
                    edge_weights = self.weight_learners[layer](x_maps, local_edge_index)
                else:
                    edge_weights = None
                L, trans_maps = local_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)
            left = self.lin_left_weights[layer] if self.left_weights else None
            right = self.lin_right_weights[layer] if self.right_weights else None
            x = self._left_right_linear_local(x, left, right, local_graph_size)
            x = torch.sparse.mm(
                torch.sparse_coo_tensor(
                    L[0],
                    L[1],
                    size=(x.size(0), x.size(0)),
                    device=x.device,
                ).coalesce(),
                x,
            )

            if self.use_act:
                x = F.elu(x)

            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(local_graph_size, 1)
            x0 = coeff * x0 - x
            x = x0

        return x.reshape(local_graph_size, -1)

    def _global_local_maps(self, active_mask: torch.Tensor):
        local_nodes = active_mask.nonzero(as_tuple=False).view(-1)
        local_index = torch.full(
            (self.graph_size,),
            -1,
            dtype=torch.long,
            device=active_mask.device,
        )
        local_index[local_nodes] = torch.arange(local_nodes.numel(), device=active_mask.device)
        return local_nodes, local_index

    def _localize_edges(self, edge_index: torch.Tensor, local_index: torch.Tensor):
        row, col = edge_index
        edge_mask = (local_index[row] >= 0) & (local_index[col] >= 0)
        edge_index = edge_index[:, edge_mask]
        if edge_index.numel() == 0:
            return edge_index
        return local_index[edge_index]

    def _scatter_global_state(
        self,
        previous_state: Optional[torch.Tensor],
        local_nodes: torch.Tensor,
        local_values: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if previous_state is None:
            output = torch.zeros(
                self.graph_size,
                local_values.size(-1),
                device=device,
                dtype=local_values.dtype,
            )
        else:
            output = previous_state.clone()
        output[local_nodes] = local_values
        return output

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
        prev_memory = prev_state.memory if prev_state is not None and prev_state.memory is not None else self._initial_memory(x.device)
        prev_spatial = prev_state.spatial if prev_state is not None else None

        active_mask = self._expand_active_nodes(edge_index, active_nodes)
        local_nodes, local_index = self._global_local_maps(active_mask)

        if local_nodes.numel() == 0:
            logits = torch.zeros(self.graph_size, self.output_dim, device=x.device)
            next_state = TemporalMambaState(memory=prev_memory, spatial=prev_spatial)
            if return_state:
                return logits, next_state
            return logits

        local_edge_index = self._localize_edges(edge_index, local_index)
        local_x = x.index_select(0, local_nodes)
        local_prev_memory = prev_memory.index_select(0, local_nodes)
        local_prev_spatial = None if prev_spatial is None else prev_spatial.index_select(0, local_nodes)

        node_signal = self._encode_nodes(local_x)
        lagged_readout = self._local_lagged_readout(local_prev_spatial, local_edge_index, local_nodes.numel())
        topology = torch.cat(
            [
                degree(local_edge_index[0], num_nodes=local_nodes.numel()).unsqueeze(-1),
                torch.ones(local_nodes.numel(), 1, device=x.device, dtype=node_signal.dtype),
            ],
            dim=-1,
        )

        local_memory = self.nodewise_learner(
            previous_memory=local_prev_memory,
            node_signal=node_signal,
            lagged_readout=lagged_readout,
            topology=topology,
            active_mask=None,
        )
        local_spatial = self._apply_local_diffusion(local_memory, local_edge_index)

        next_memory = self._scatter_global_state(prev_memory, local_nodes, local_memory, x.device)
        next_spatial = self._scatter_global_state(prev_spatial, local_nodes, local_spatial, x.device)

        logits = torch.zeros(self.graph_size, self.output_dim, device=x.device, dtype=local_spatial.dtype)
        logits[local_nodes] = F.log_softmax(self.lin2(local_spatial), dim=1)
        next_state = TemporalMambaState(memory=next_memory, spatial=next_spatial)

        if self.stateful_temporal and state is None:
            self._temporal_state = TemporalMambaState(
                memory=next_memory.detach(),
                spatial=next_spatial.detach(),
            )

        if return_state:
            return logits, next_state
        return logits

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
            else:
                x = snapshot.x
                edge_index = getattr(snapshot, "edge_index", None)
                active_nodes = getattr(snapshot, "active_nodes", None)

            logits, state = self.step(
                x,
                edge_index=edge_index,
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


class SparseTemporalEdgeMambaSheafDiffusion(SparseTemporalMambaSheafDiffusion):
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
