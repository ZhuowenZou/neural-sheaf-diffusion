import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

from .sheaf_base import SheafDiffusion
from . import laplacian_builders as lb


class MambaSheafDiffusion(SheafDiffusion):
    """Discrete sheaf diffusion with Mamba-based sheaf learner."""

    def __init__(self, edge_index, args):
        super(MambaSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        # Linear projections
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            self.sheaf_learners.append(Mamba_learner(
                in_dim=self.hidden_dim,
                out_dim=self.d
            ))

        # Laplacian builder using diagonal restriction maps
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp
        )

        # Optional edge weight modules
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        if self.right_weights:
            for _ in range(self.layers):
                layer = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                nn.init.orthogonal_(layer.weight)
                self.lin_right_weights.append(layer)
        if self.left_weights:
            for _ in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))

        # Epsilon residual gating
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1))) for _ in range(self.layers)
        ])

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        x = x.view(self.graph_size * self.final_d, -1)
        x0 = x

        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_map_input = F.dropout(x, p=self.dropout if layer > 0 else 0.0, training=self.training)
                maps = self.sheaf_learners[layer](x_map_input.view(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()
            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


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


class Mamba_learner(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.mamba = SimpleMambaBlock(d_model)
        self.output_proj = nn.Linear(d_model, out_dim)
        self.L = None

    def __len__(self):
        return len(self.L) if self.L is not None else 0

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_feat = x[col]  # [num_edges, in_dim]

        h = self.input_proj(edge_feat)
        h = h.unsqueeze(0)                    # [1, num_edges, d_model]
        h = self.mamba(h).squeeze(0)          # [num_edges, d_model]
        out = self.output_proj(h)             # [num_edges, out_dim]
        return out

    def set_L(self, L):
        self.L = L
