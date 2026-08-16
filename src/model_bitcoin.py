import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class BitcoinGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels_x, hidden_channels, out_channels_node):
        super(BitcoinGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels_x, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, out_channels_node)
        self.bn2 = torch.nn.BatchNorm1d(out_channels_node)

        # residual projection from raw input features to final embedding size
        self.res_proj = torch.nn.Linear(in_channels_x, out_channels_node)

        self.pred_head = torch.nn.Linear(out_channels_node, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_in = x

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        # residual connection to reduce over-smoothing
        x = x + self.res_proj(x_in)

        raw_scores = self.pred_head(x)
        suspicion_scores = F.softmax(raw_scores, dim=1)[:, 1]

        return raw_scores, suspicion_scores