import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class AMLGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels_x, in_channels_edge, hidden_channels, out_channels_node):
        super(AMLGraphSAGE, self).__init__()

        self.initial_lin = torch.nn.Linear(in_channels_x, in_channels_edge)
        self.conv1 = SAGEConv(in_channels_edge, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels_node)

        # UPDATED: prediction head now also takes edge_attr directly
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels_node + in_channels_edge, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.initial_lin(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        sender_embeddings = x[edge_index[0]]
        receiver_embeddings = x[edge_index[1]]

        # UPDATED: edge_attr (amount, tx_type) now feeds the prediction head
        combined = torch.cat([sender_embeddings, receiver_embeddings, edge_attr], dim=1)

        raw_score = self.pred_head(combined)
        suspicion_score = torch.sigmoid(raw_score)

        return raw_score.squeeze(), suspicion_score.squeeze()