"""
Heterogeneous GNN for link prediction over the (station, context) graph.

Architecture
  - Per node-type initial embedding: station from features, context from a
    learned embedding table.
  - K stacked layers. In each layer every relation contributes one message
    aggregated by sum into the destination node. Self-loop preserves the
    previous representation. LayerNorm + ReLU + residual after fusion.
  - Link prediction: bilinear score between a context vector and every
    station vector → softmax over stations.

Relations (each gets its own message function):
  metro:           station -> station
  found_at_fwd:    station -> context     (event found at this station)
  found_at_rev:    context -> station     (reverse pass)
  picked_up_fwd:   context -> station     (the supervised relation)
  picked_up_rev:   station -> context     (reverse pass)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_sum(messages, index, dim_size):
    """messages: [E, d], index: [E] of dest node indices → [dim_size, d] sum."""
    out = torch.zeros(dim_size, messages.shape[1], device=messages.device, dtype=messages.dtype)
    out.index_add_(0, index, messages)
    return out


class HetGNNLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin_self_st = nn.Linear(dim, dim)
        self.lin_self_ctx = nn.Linear(dim, dim)
        # one linear per relation
        self.lin_metro = nn.Linear(dim, dim)
        self.lin_found_fwd = nn.Linear(dim, dim)   # st -> ctx
        self.lin_found_rev = nn.Linear(dim, dim)   # ctx -> st
        self.lin_lost_fwd = nn.Linear(dim, dim)    # st -> ctx  (lost_at relation)
        self.lin_lost_rev = nn.Linear(dim, dim)    # ctx -> st
        self.lin_pick_fwd = nn.Linear(dim, dim)    # ctx -> st  (supervised)
        self.lin_pick_rev = nn.Linear(dim, dim)    # st -> ctx
        self.norm_st = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_st, h_ctx, g):
        N_st, N_ctx = h_st.shape[0], h_ctx.shape[0]

        # ── station-side aggregation ──────────────────────────────────────
        msg_st = self.lin_self_st(h_st)

        # metro neighbors → station
        m_idx = g["metro_edge_index"]                          # [2, E]
        w_m = g["metro_weight"].unsqueeze(-1)
        m_msg = h_st[m_idx[0]] * w_m
        msg_st = msg_st + self.lin_metro(_scatter_sum(m_msg, m_idx[1], N_st))

        # found_at reverse: context → station (info about which contexts saw this station)
        f_idx = g["found_at_edge_index"]                       # row0=st, row1=ctx
        if f_idx.shape[1] > 0:
            w_f = g["found_at_weight"].unsqueeze(-1)
            ctx_msg = h_ctx[f_idx[1]] * w_f
            msg_st = msg_st + self.lin_found_rev(_scatter_sum(ctx_msg, f_idx[0], N_st))

        # lost_at reverse: context → station (which contexts were *lost* at this station)
        l_idx = g.get("lost_at_edge_index")
        if l_idx is not None and l_idx.shape[1] > 0:
            w_l = g["lost_at_weight"].unsqueeze(-1)
            ctx_msg = h_ctx[l_idx[1]] * w_l
            msg_st = msg_st + self.lin_lost_rev(_scatter_sum(ctx_msg, l_idx[0], N_st))

        # pickup forward: context → station (the supervised relation)
        p_idx = g["picked_up_edge_index"]                      # row0=ctx, row1=st
        if p_idx.shape[1] > 0:
            w_p = g["picked_up_weight"].unsqueeze(-1)
            ctx_msg = h_ctx[p_idx[0]] * w_p
            msg_st = msg_st + self.lin_pick_fwd(_scatter_sum(ctx_msg, p_idx[1], N_st))

        new_st = self.norm_st(msg_st)
        new_st = F.relu(new_st)
        new_st = self.drop(new_st)
        new_st = h_st + new_st                                  # residual

        # ── context-side aggregation ──────────────────────────────────────
        msg_ctx = self.lin_self_ctx(h_ctx)

        # found_at forward: station → context
        if f_idx.shape[1] > 0:
            st_msg = h_st[f_idx[0]] * w_f
            msg_ctx = msg_ctx + self.lin_found_fwd(_scatter_sum(st_msg, f_idx[1], N_ctx))

        # lost_at forward: station → context
        if l_idx is not None and l_idx.shape[1] > 0:
            st_msg = h_st[l_idx[0]] * w_l
            msg_ctx = msg_ctx + self.lin_lost_fwd(_scatter_sum(st_msg, l_idx[1], N_ctx))

        # pickup reverse: station → context
        if p_idx.shape[1] > 0:
            st_msg = h_st[p_idx[1]] * w_p
            msg_ctx = msg_ctx + self.lin_pick_rev(_scatter_sum(st_msg, p_idx[0], N_ctx))

        new_ctx = self.norm_ctx(msg_ctx)
        new_ctx = F.relu(new_ctx)
        new_ctx = self.drop(new_ctx)
        new_ctx = h_ctx + new_ctx                               # residual

        return new_st, new_ctx


class HetGNN(nn.Module):
    """Encode the heterogeneous graph, then score (context, found-at, station) triples.

    Key design: the query for a prediction is the *concatenation* of the context
    embedding and the found-at station embedding — both run through the GNN.
    Without the found-at signal the model can only condition on (item, hour, day)
    and has no way to use the location of the loss.
    """

    def __init__(self, station_feat_dim, n_contexts, hidden=64, n_layers=2,
                 dropout=0.1, pickup_edge_dropout=0.3):
        super().__init__()
        self.station_proj = nn.Linear(station_feat_dim, hidden)
        self.context_emb = nn.Embedding(n_contexts, hidden)
        nn.init.normal_(self.context_emb.weight, std=0.1)
        self.layers = nn.ModuleList([HetGNNLayer(hidden, dropout) for _ in range(n_layers)])
        # Combine (context, found-station, lost-station) into a query vector
        self.query_combine = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Bilinear scorer (DistMult-style with a near-identity init)
        self.W_score = nn.Parameter(torch.eye(hidden) + 0.01 * torch.randn(hidden, hidden))
        self.bias_st = nn.Parameter(torch.zeros(1))
        self.found_at_bonus = nn.Parameter(torch.zeros(1))
        self.pickup_edge_dropout = pickup_edge_dropout

    def _maybe_drop_pickup_edges(self, g):
        """During training, randomly drop a fraction of picked_up_at edges from
        the graph before message passing. Prevents the encoder from trivially
        memorising the relation we are trying to predict."""
        if not self.training or self.pickup_edge_dropout <= 0.0:
            return g
        idx = g["picked_up_edge_index"]
        if idx.shape[1] == 0:
            return g
        keep_mask = torch.rand(idx.shape[1], device=idx.device) > self.pickup_edge_dropout
        if keep_mask.all():
            return g
        g2 = dict(g)
        g2["picked_up_edge_index"] = idx[:, keep_mask]
        g2["picked_up_weight"] = g["picked_up_weight"][keep_mask]
        return g2

    def encode(self, g):
        g = self._maybe_drop_pickup_edges(g)
        h_st = self.station_proj(g["station_feats"])
        h_ctx = self.context_emb.weight
        for layer in self.layers:
            h_st, h_ctx = layer(h_st, h_ctx, g)
        return h_st, h_ctx

    def score(self, h_ctx_query, h_st, found_st_idx, lost_st_idx=None):
        """
        h_ctx_query:  [B, d]   context vectors for events
        h_st:         [N, d]   all station vectors
        found_st_idx: [B]      station the event was found at
        lost_st_idx:  [B] or None — claimant-reported loss station. If None,
                       falls back to found_st_idx (legacy data).

        Returns logits [B, N] over all stations.
        """
        h_found = h_st[found_st_idx]                                  # [B, d]
        if lost_st_idx is None:
            h_lost = h_found
        else:
            h_lost = h_st[lost_st_idx]
        q = self.query_combine(torch.cat([h_ctx_query, h_found, h_lost], dim=-1))
        scores = q @ self.W_score @ h_st.t() + self.bias_st          # [B, N]

        B = scores.shape[0]
        bonus = torch.zeros_like(scores)
        bonus[torch.arange(B, device=scores.device), found_st_idx] = self.found_at_bonus
        return scores + bonus

    def forward(self, g, ctx_ids, found_st_idx, lost_st_idx=None):
        h_st, h_ctx = self.encode(g)
        return self.score(h_ctx[ctx_ids], h_st, found_st_idx, lost_st_idx)
