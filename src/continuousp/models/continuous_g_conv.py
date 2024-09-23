import torch
import torch.nn.functional
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import (
    MessagePassing,
)

from continuousp.utils.logger import LOGGER


class ContinuousGConv(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.lin1 = spectral_norm(
            nn.Linear(
                self.node_feat_size,
                2 * self.node_feat_size,
            ),
        )
        self.lin2 = spectral_norm(
            nn.Linear(
                self.node_feat_size,
                2 * self.node_feat_size,
            ),
        )
        self.lin3 = nn.Linear(
            self.edge_feat_size,
            2 * self.node_feat_size,
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_nodes, node_feat_size]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_weight: torch.Tensor,  # [num_edges, 1]
        edge_attr: torch.Tensor,  # [num_edges, edge_feat_size]
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            size=(x.size(0), x.size(0)),
        )
        return torch.nn.functional.tanh(out) + x  # [num_nodes, node_feat_size]

    def message(
        self,
        x_i: torch.Tensor,  # [num_edges, node_feat_size]
        x_j: torch.Tensor,  # [num_edges, node_feat_size]
        edge_weight: torch.Tensor,  # [num_edges, 1]
        edge_attr: torch.Tensor,  # [num_edges, edge_feat_size]
    ) -> torch.Tensor:  # [num_edges, node_feat_size]
        z1 = self.lin1(x_i)
        z2 = self.lin2(x_j)
        z3 = self.lin3(edge_attr)
        z = (z1 + z2) * z3
        z1, z2 = z.chunk(2, dim=1)
        return edge_weight.repeat(1, self.node_feat_size) * torch.nn.functional.sigmoid(z1) * z2
