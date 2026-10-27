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

    def forward(self, data, idx=None):
        """
        Message passing runs over `data.mp_edge_index` when the graph defines it (for example only the
        training-period edges), otherwise over every edge. Scores are produced for the edges in
        `data.edge_index`, or only for the rows in `idx` when given (saves memory in full-batch training).
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        mp_edge_index = getattr(data, 'mp_edge_index', None)
        if mp_edge_index is None:
            mp_edge_index = edge_index

        x = self.initial_lin(x)

        x = self.conv1(x, mp_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, mp_edge_index)

        src, dst = edge_index[0], edge_index[1]
        if idx is not None:
            src, dst, edge_attr = src[idx], dst[idx], edge_attr[idx]
        sender_embeddings = x[src]
        receiver_embeddings = x[dst]

        # UPDATED: edge_attr (amount, tx_type) now feeds the prediction head
        combined = torch.cat([sender_embeddings, receiver_embeddings, edge_attr], dim=1)

        raw_score = self.pred_head(combined)
        suspicion_score = torch.sigmoid(raw_score)

        return raw_score.squeeze(), suspicion_score.squeeze()

