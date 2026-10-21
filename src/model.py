import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, SAGEConv

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


class EdgeAwareGNN(torch.nn.Module):
    """
    Edge-aware message passing: two GINE layers add the (projected) edge attributes to every neighbour message,
    so amounts, recency and structure features travel along edges instead of only entering the prediction head.
    Same graph conventions as AMLGraphSAGE: messages flow over `data.mp_edge_index` (with the matching rows of
    `edge_attr`, selected by `data.mp_edge_mask`) and scores are produced for `data.edge_index`, or for `idx` only.
    """

    def __init__(self, in_channels_x, in_channels_edge, hidden_channels, out_channels_node):
        super().__init__()
        self.initial_lin = torch.nn.Linear(in_channels_x, hidden_channels)

        def mlp(n_in, n_out):
            return torch.nn.Sequential(torch.nn.Linear(n_in, n_out), torch.nn.ReLU(), torch.nn.Linear(n_out, n_out))

        self.conv1 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=in_channels_edge)
        self.conv2 = GINEConv(mlp(hidden_channels, out_channels_node), edge_dim=in_channels_edge)
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels_node + in_channels_edge, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data, idx=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        mp_edge_index = getattr(data, 'mp_edge_index', None)
        if mp_edge_index is None:
            mp_edge_index, mp_edge_attr = edge_index, edge_attr
        else:
            mp_edge_attr = edge_attr[data.mp_edge_mask]

        x = F.relu(self.initial_lin(x))
        x = F.relu(self.conv1(x, mp_edge_index, mp_edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, mp_edge_index, mp_edge_attr)

        src, dst = edge_index[0], edge_index[1]
        if idx is not None:
            src, dst, edge_attr = src[idx], dst[idx], edge_attr[idx]
        raw_score = self.pred_head(torch.cat([x[src], x[dst], edge_attr], dim=1))
        return raw_score.squeeze(), torch.sigmoid(raw_score).squeeze()
