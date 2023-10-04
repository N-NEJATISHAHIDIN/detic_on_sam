import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv



class GCN(torch.nn.Module):
    def __init__(self, input_chanels, hidden_channels, output_channels):
        super().__init__( )
        # torch.manual_seed(1234567)

        self.conv1 = GCNConv(input_chanels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.box_embeddings = PositionEmbeddingLearned(4, 128)

    def forward(self, x, edge_index):
        box_proposal_embeding = self.box_embeddings(x[:,:,-4:]).permute( 0,2, 1)
        x = torch.cat((x[:,:,:-4],box_proposal_embeding), dim = 2)
        # import pdb; pdb.set_trace()

        x = self.conv1(x[:,0,:], edge_index)
        # x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        print(x)
        return x


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


# class GNN(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GNN, self).__init__(aggr='mean') # "Add" aggregation.

#         self.lin_node = torch.nn.Linear(in_channels, 512)
#         self.lin2_node = torch.nn.Linear(512, 128)
#         self.lin3_node = torch.nn.Linear(128, 4)

#         self.lin_edge = torch.nn.Linear(in_channels, 512)
#         self.lin2_edge = torch.nn.Linear(512, 128)
#         self.lin3_edge = torch.nn.Linear(128, 4)

#         self.box_embeddings = PositionEmbeddingLearned(4, 128)

#     def forward(self, x, edge_index, edge_attr):
#         # x has shape [num_nodes, in_channels]
#         # edge_index has shape [2, num_edges]

#         # Add self-loops to the adjacency matrix.
#         edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
#         # Compute the degree vector.

#         deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]


#         # Apply the GNN model with node and edge feature computation
#         x = self.lin_node(x)
#         x = F.relu(x)
#         edge_attr = self.lin_edge(edge_attr)
#         edge_attr = F.relu(edge_attr)
#         x = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
#         x = F.relu(x)
#         x = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
#         x = self.lin_out(x)

#         return x
    
#         # deg = degree(edge_index[0], x.size(0), dtype=x.dtype)

#         # # Apply the linear transformation.
#         # x = self.lin(x)
#         # x = self.lin2(x)
#         # x = self.lin3(x)
        
#         # # Start propagating messages.
#         # return self.propagate(edge_index, x=x, deg=deg)

#     def message(self, x_j, deg):
#         # x_j has shape [num_edges, out_channels]

#         # Normalize node features by their degree.
#         return x_j / deg.view(-1, 1)

#     def update(self, aggr_out):
#         # aggr_out has shape [num_nodes, out_channels]

#         # Apply ReLU activation.
#         return F.relu(aggr_out)
