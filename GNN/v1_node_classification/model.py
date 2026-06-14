"""
GNN destination predictor.

Conditioning a graph on a per-event query is what makes this useful:
each event broadcasts (item, hour, day-of-week, found-at-mask) into every
station's hidden state before message passing. The readout is a per-station
logit → softmax over destinations.

The decision rule (decision.py) turns that distribution into a storage station;
the model only predicts the distribution.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """One step of symmetric-normalized graph convolution with residual + LN.

    Pre-built dense adj is fine here — the metro has ~180 nodes.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h: [B, N, d], adj: [N, N]
        h_agg = torch.einsum("ij,bjd->bid", adj, h)
        h_new = self.lin(h_agg)
        h_new = F.relu(self.norm(h_new))
        h_new = self.dropout(h_new)
        return h + h_new


class GNNDestinationPredictor(nn.Module):
    def __init__(self, n_stations, node_feat_dim, n_items,
                 hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_stations = n_stations
        self.hidden = hidden

        self.node_proj = nn.Linear(node_feat_dim, hidden)
        self.item_emb = nn.Embedding(n_items, hidden)
        self.dow_emb = nn.Embedding(7, hidden)
        self.hour_proj = nn.Linear(2, hidden)
        self.query_combine = nn.Linear(hidden * 3, hidden)
        self.found_mask_emb = nn.Embedding(2, hidden)

        self.gnn_layers = nn.ModuleList(
            [GCNLayer(hidden, dropout) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _encode_query(self, item_idx, hour, dow):
        item = self.item_emb(item_idx)
        dow_e = self.dow_emb(dow)
        hour_rad = hour.float() * (2 * math.pi / 24.0)
        hour_feat = torch.stack([torch.sin(hour_rad), torch.cos(hour_rad)], dim=-1)
        hour_e = self.hour_proj(hour_feat)
        q = self.query_combine(torch.cat([item, dow_e, hour_e], dim=-1))
        return q  # [B, H]

    def forward(self, node_feats, adj, found_idx, item_idx, hour, dow):
        """
        node_feats: [N, F]
        adj:        [N, N]
        found_idx:  [B]    station index where the item was found
        item_idx:   [B]
        hour:       [B]
        dow:        [B]
        returns logits [B, N] over destinations.
        """
        B = found_idx.shape[0]
        N = self.n_stations

        h0 = self.node_proj(node_feats)                       # [N, H]
        h = h0.unsqueeze(0).expand(B, N, -1).contiguous()     # [B, N, H]

        q = self._encode_query(item_idx, hour, dow)           # [B, H]
        h = h + q.unsqueeze(1)

        mask = torch.zeros(B, N, dtype=torch.long, device=h.device)
        mask[torch.arange(B, device=h.device), found_idx] = 1
        h = h + self.found_mask_emb(mask)

        for layer in self.gnn_layers:
            h = layer(h, adj)

        logits = self.readout(h).squeeze(-1)                  # [B, N]
        return logits
