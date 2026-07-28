from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn
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
            # torch_sparse.spmm keeps the backward w.r.t. the Laplacian values
            # sparse; torch.sparse.mm materializes a dense (n x n) gradient,
            # which is quadratic in the local subgraph size.
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

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


class EventTemporalMambaSheafDiffusion(SparseTemporalMambaSheafDiffusion):
    """Sparse temporal sheaf diffusion with event-level scoring.

    This variant keeps the same nodewise recurrent updates and sparse local
    sheaf diffusion as ``SparseTemporalMambaSheafDiffusion``, but it avoids the
    dense ``num_nodes x num_nodes`` classification head. Instead, each snapshot
    produces node embeddings and event scores are computed only for the queried
    source/destination pairs (and sampled negatives).
    """

    supports_event_scoring = True

    def __init__(self, edge_index, args):
        args = dict(args)
        event_output_dim = int(args.get("event_output_dim", args["hidden_channels"] * args["d"]))
        args["output_dim"] = event_output_dim
        super().__init__(edge_index, args)

        self.num_relations = int(args.get("num_relations", 0))
        self.train_negatives_per_pos = int(args.get("train_negatives_per_pos", 32))
        self.candidate_chunk_size = int(args.get("candidate_chunk_size", 2048))
        # Cap on rows*candidates scored densely at once; keeps peak activation
        # memory bounded when a snapshot carries hundreds of thousands of events.
        self.max_score_elements = int(args.get("max_score_elements", 4_000_000))

        self.event_source_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.event_destination_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.event_signal_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.event_state_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.event_norm = nn.LayerNorm(self.hidden_dim)
        self.event_bias = nn.Parameter(torch.zeros(1))

        if self.num_relations > 0:
            self.event_relation_embeddings = nn.Embedding(self.num_relations, self.hidden_dim)
            self.event_relation_bias = nn.Embedding(self.num_relations, 1)
        else:
            self.event_relation_embeddings = None
            self.event_relation_bias = None

    def _empty_event_output(self, x: torch.Tensor, prev_spatial: Optional[torch.Tensor]):
        if prev_spatial is None:
            spatial = torch.zeros(self.graph_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            spatial = prev_spatial
        return {
            "spatial": spatial,
            "x": x,
            "node_signal": None,
        }

    def _normalize_event_edge_index(self, edge_index: Optional[torch.Tensor], device: torch.device):
        if edge_index is None:
            return None
        if edge_index.numel() == 0:
            return edge_index.to(device=device, dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index.to(device))
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, self.graph_size, self.graph_size)
        return edge_index.contiguous()

    def _merge_local_edge_indices(
        self,
        local_context_edge_index: Optional[torch.Tensor],
        local_event_edge_index: Optional[torch.Tensor],
        local_graph_size: int,
        device: torch.device,
    ):
        edge_parts = []
        for edge_index in (local_context_edge_index, local_event_edge_index):
            if edge_index is not None and edge_index.numel() > 0:
                edge_parts.append(edge_index)

        if not edge_parts:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        if len(edge_parts) == 1:
            return edge_parts[0].contiguous()

        merged = torch.cat(edge_parts, dim=1)
        merged, _ = coalesce(merged, None, local_graph_size, local_graph_size)
        return merged.contiguous()

    def _event_node_repr(self, output, node_ids: torch.Tensor):
        flat_node_ids = node_ids.reshape(-1).long()
        if output.get("node_signal") is not None:
            node_signal = output["node_signal"].index_select(0, flat_node_ids)
        else:
            # Lazy path for large graphs: the encoder is pointwise per node, so
            # encoding only the queried ids is equivalent to a full-graph encode.
            node_signal = self._encode_nodes(output["x"].index_select(0, flat_node_ids))
        spatial = output["spatial"].index_select(0, flat_node_ids)
        fused = self.event_signal_proj(node_signal) + self.event_state_proj(spatial)
        fused = self.event_norm(fused)
        return fused.view(*node_ids.shape, -1)

    def _relation_factors(self, edge_type: Optional[torch.Tensor], target_shape, device, dtype):
        if self.event_relation_embeddings is None or edge_type is None:
            return None, None

        flat_edge_type = edge_type.reshape(-1).long().to(device)
        relation = self.event_relation_embeddings(flat_edge_type).to(dtype=dtype)
        bias = self.event_relation_bias(flat_edge_type).to(dtype=dtype)
        relation = relation.view(*target_shape, self.hidden_dim)
        bias = bias.view(*target_shape)
        return relation, bias

    def _score_embeddings(self, src_repr, dst_repr, relation_repr=None, relation_bias=None):
        src_proj = self.event_source_proj(src_repr)
        dst_proj = self.event_destination_proj(dst_repr)
        if relation_repr is not None:
            scores = (src_proj * relation_repr * dst_proj).sum(dim=-1)
        else:
            scores = (src_proj * dst_proj).sum(dim=-1)
        scores = scores + self.event_bias
        if relation_bias is not None:
            scores = scores + relation_bias
        return scores

    def score_event_pairs(self, output, src: torch.Tensor, dst: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
        src_repr = self._event_node_repr(output, src)
        dst_repr = self._event_node_repr(output, dst)
        relation_repr, relation_bias = self._relation_factors(
            edge_type,
            target_shape=src.shape,
            device=src_repr.device,
            dtype=src_repr.dtype,
        )
        return self._score_embeddings(src_repr, dst_repr, relation_repr, relation_bias)

    def _score_event_candidates_dense(
        self,
        output,
        src: torch.Tensor,
        dst_candidates: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ):
        src_repr = self._event_node_repr(output, src).unsqueeze(1)
        dst_repr = self._event_node_repr(output, dst_candidates)
        relation_repr = None
        relation_bias = None
        if self.event_relation_embeddings is not None and edge_type is not None:
            flat_edge_type = edge_type.reshape(-1).long().to(dst_repr.device)
            relation_repr = self.event_relation_embeddings(flat_edge_type).to(dtype=dst_repr.dtype).unsqueeze(1)
            relation_repr = relation_repr.expand(-1, dst_candidates.size(1), -1)
            relation_bias = self.event_relation_bias(flat_edge_type).to(dtype=dst_repr.dtype).unsqueeze(1)
            relation_bias = relation_bias.expand(-1, dst_candidates.size(1), -1).squeeze(-1)
        return self._score_embeddings(src_repr, dst_repr, relation_repr, relation_bias)

    def score_event_candidates(
        self,
        output,
        src: torch.Tensor,
        dst_candidates: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ):
        if dst_candidates.dim() == 1:
            return self.score_event_pairs(output, src, dst_candidates, edge_type=edge_type)

        num_rows, num_candidates = dst_candidates.size(0), dst_candidates.size(1)
        cand_chunk = max(min(int(self.candidate_chunk_size), num_candidates), 1)
        row_chunk = max(int(self.max_score_elements) // cand_chunk, 1)

        if num_rows <= row_chunk and num_candidates <= cand_chunk:
            return self._score_event_candidates_dense(output, src, dst_candidates, edge_type=edge_type)

        row_outputs = []
        for row_start in range(0, num_rows, row_chunk):
            row_stop = min(row_start + row_chunk, num_rows)
            row_src = src[row_start:row_stop]
            row_edge_type = edge_type if edge_type is None else edge_type[row_start:row_stop]
            chunks = []
            for start in range(0, num_candidates, cand_chunk):
                stop = min(start + cand_chunk, num_candidates)
                chunks.append(
                    self._score_event_candidates_dense(
                        output,
                        row_src,
                        dst_candidates[row_start:row_stop, start:stop],
                        edge_type=row_edge_type,
                    )
                )
            row_outputs.append(torch.cat(chunks, dim=1))
        return torch.cat(row_outputs, dim=0)

    def step(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        active_nodes: Optional[torch.Tensor] = None,
        state: Optional[TemporalMambaState] = None,
        return_state: bool = False,
    ):
        dynamic_edge_index = self._normalize_event_edge_index(edge_index, x.device)
        context_edge_index = self.edge_index.to(x.device) if self.edge_index is not None else None
        prev_state = state if state is not None else self._temporal_state
        prev_memory = prev_state.memory if prev_state is not None and prev_state.memory is not None else self._initial_memory(x.device)
        prev_spatial = prev_state.spatial if prev_state is not None else None

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

        local_context_edge_index = None
        if context_edge_index is not None and context_edge_index.numel() > 0:
            local_context_edge_index = self._localize_edges(context_edge_index, local_index)
        local_dynamic_edge_index = None
        if dynamic_edge_index is not None and dynamic_edge_index.numel() > 0:
            local_dynamic_edge_index = self._localize_edges(dynamic_edge_index, local_index)
        local_edge_index = self._merge_local_edge_indices(
            local_context_edge_index,
            local_dynamic_edge_index,
            local_nodes.numel(),
            x.device,
        )
        local_prev_memory = prev_memory.index_select(0, local_nodes)
        local_prev_spatial = None if prev_spatial is None else prev_spatial.index_select(0, local_nodes)
        local_node_signal = self._encode_nodes(x.index_select(0, local_nodes))

        lagged_readout = self._local_lagged_readout(local_prev_spatial, local_edge_index, local_nodes.numel())
        topology = torch.cat(
            [
                degree(local_edge_index[0], num_nodes=local_nodes.numel()).unsqueeze(-1),
                torch.ones(local_nodes.numel(), 1, device=x.device, dtype=local_node_signal.dtype),
            ],
            dim=-1,
        )

        local_memory = self.nodewise_learner(
            previous_memory=local_prev_memory,
            node_signal=local_node_signal,
            lagged_readout=lagged_readout,
            topology=topology,
            active_mask=None,
        )
        local_spatial = self._apply_local_diffusion(local_memory, local_edge_index)

        next_memory = self._scatter_global_state(prev_memory, local_nodes, local_memory, x.device)
        next_spatial = self._scatter_global_state(prev_spatial, local_nodes, local_spatial, x.device)
        next_state = TemporalMambaState(memory=next_memory, spatial=next_spatial)
        event_output = {
            "spatial": next_spatial,
            "x": x,
            "node_signal": None,
        }

        if self.stateful_temporal and state is None:
            self._temporal_state = TemporalMambaState(
                memory=next_memory.detach(),
                spatial=next_spatial.detach(),
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
            else:
                x = snapshot.x
                edge_index = getattr(snapshot, "edge_index", None)
                active_nodes = getattr(snapshot, "active_nodes", None)

            event_output, state = self.step(
                x,
                edge_index=edge_index,
                active_nodes=active_nodes,
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
