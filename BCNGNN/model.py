"""
Inductive GraphSAGE-style GNN for per-item claim-station prediction.

Each forward pass takes a batched bundle of per-item subgraphs and outputs,
for every node in each subgraph, a score = "this is the claim station".

We implement SAGEConv from scratch (mean aggregation):
    h_v' = ReLU( W_self · h_v  +  W_neigh · mean_{u in N(v)} h_u )
followed by L2-normalisation (the GraphSAGE-original normalisation step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(messages: torch.Tensor, index: torch.Tensor, dim_size: int):
    """Mean aggregation over `messages` indexed by `index` into `dim_size` rows."""
    out = torch.zeros(dim_size, messages.shape[1], device=messages.device, dtype=messages.dtype)
    out.index_add_(0, index, messages)
    counts = torch.zeros(dim_size, device=messages.device, dtype=messages.dtype)
    counts.index_add_(0, index, torch.ones(index.shape[0], device=messages.device, dtype=messages.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(-1)
    return out / counts


class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index[0] = src, edge_index[1] = dst — message flows src -> dst
        src, dst = edge_index[0], edge_index[1]
        agg = _scatter_mean(x[src], dst, dim_size=x.shape[0])
        h = self.lin_self(x) + self.lin_neigh(agg)
        h = self.norm(h)
        h = F.relu(h)
        return h


class StorageGNN(nn.Module):
    """Per-item subgraph -> node logits.

    The score for each candidate node is conditioned on the found-station
    embedding (we read it out via the is_found feature flag) so the head
    can answer the contextual question "given THIS found station, is THAT
    node the likely claim?". This significantly outperforms a bare per-node
    MLP head.
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.drop = nn.Dropout(dropout)
        # Score head takes (candidate_emb, found_emb) and returns a scalar.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_idx: torch.Tensor | None = None,
                found_global_idx: torch.Tensor | None = None) -> torch.Tensor:
        """If batch_idx + found_global_idx are provided, condition each
        node's score on the found-station embedding of its item. Otherwise
        (single graph, found-flag in features), use the per-node embedding
        of the found node read out from the is_found feature column."""
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.drop(h)

        if batch_idx is not None and found_global_idx is not None:
            # Broadcast each item's found-station embedding to all its nodes.
            found_emb_per_item = h[found_global_idx]              # [B, d]
            found_emb_per_node = found_emb_per_item[batch_idx]    # [N, d]
        else:
            # Fallback: single-graph inference — find the found node by feature.
            # Look at the "is_found" feature flag (zone+lines offset = NUM_ZONES+N_LINES).
            # Use mean over rows where the flag is 1.
            from network import NUM_ZONES
            from subgraph import N_LINES
            found_flag = x[:, NUM_ZONES + N_LINES]
            # weighted average; should pick exactly the found node
            w = (found_flag > 0.5).float().unsqueeze(-1)
            found_emb = (h * w).sum(0) / w.sum().clamp(min=1.0)
            found_emb_per_node = found_emb.unsqueeze(0).expand(h.shape[0], -1)

        joined = torch.cat([h, found_emb_per_node], dim=-1)
        return self.head(joined).squeeze(-1)


def per_item_softmax(logits: torch.Tensor, batch_idx: torch.Tensor, n_items: int):
    """Softmax over nodes within each item's subgraph.

    logits:    [total_nodes]
    batch_idx: [total_nodes] item id per node
    Returns:   probs [total_nodes] (each item's nodes sum to 1)
    """
    # subtract per-item max for stability
    max_per_item = torch.full((n_items,), float('-inf'), device=logits.device)
    max_per_item = max_per_item.scatter_reduce(0, batch_idx, logits, reduce="amax", include_self=True)
    shifted = logits - max_per_item[batch_idx]
    exp = shifted.exp()
    denom = torch.zeros(n_items, device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, batch_idx, exp)
    return exp / denom[batch_idx].clamp(min=1e-12)


def per_item_cross_entropy(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Cross-entropy over each item's subgraph nodes against y_local."""
    batch_idx = batch["batch_idx"]
    offsets   = batch["offsets"]
    y_local   = batch["y_local"]
    n_items   = offsets.shape[0]

    # global index of the target node per item
    tgt_global = offsets + y_local                                    # [B]

    # per-item log-softmax via the stable softmax above
    probs = per_item_softmax(logits, batch_idx, n_items).clamp(min=1e-12)
    log_p = probs.log()
    return -log_p[tgt_global].mean()
