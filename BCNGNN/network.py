"""
Barcelona metro network — graph construction, zone classification, distances.

Builds an undirected NetworkX graph from metro.py station data.
Classifies every station into one of 8 functional zones.
Pre-computes all-pairs shortest-path distances for MC simulation.
"""

import sys, os, torch
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from metro import ALL_LINES, LINE_COLORS, find_transfers, build_edges as _metro_edges, build_adjacency

# ── Build undirected graph ───────────────────────────────────────────────

G = nx.Graph()
for a, b, line in _metro_edges():
    G.add_edge(a, b)

STATION_LIST  = sorted(G.nodes())
STATION_INDEX = {s: i for i, s in enumerate(STATION_LIST)}
N_STN         = len(STATION_LIST)
TRANSFERS     = find_transfers(ALL_LINES)

# ── Shortest-path distances (pre-computed) ───────────────────────────────

DIST_MATRIX = dict(nx.all_pairs_shortest_path_length(G))

# ── Transfer-aware distances ─────────────────────────────────────────────
# Dijkstra on (station, line) state space. Moving along the same line
# costs 1 hop; switching lines at a transfer station costs 1 + penalty.

_ADJ = build_adjacency()
_STATION_LINES = {}
for line_name, stations in ALL_LINES.items():
    for s in stations:
        _STATION_LINES.setdefault(s, set()).add(line_name)


def transfer_aware_distances(source, transfer_penalty=1.0):
    """Shortest distance from source to every station, with line-switch cost.

    Returns dict: station_name -> float distance.
    """
    import heapq

    # State: (station, line) — start on every line that serves the source
    dist = {}
    heap = []
    for line in _STATION_LINES.get(source, set()):
        dist[(source, line)] = 0.0
        heapq.heappush(heap, (0.0, source, line))

    while heap:
        d, stn, cur_line = heapq.heappop(heap)
        if d > dist.get((stn, cur_line), float('inf')):
            continue

        for nbr, edge_line in _ADJ.get(stn, set()):
            # Same line → cost 1, different line → cost 1 + penalty
            cost = 1.0 if edge_line == cur_line else 1.0 + transfer_penalty
            nd = d + cost
            if nd < dist.get((nbr, edge_line), float('inf')):
                dist[(nbr, edge_line)] = nd
                heapq.heappush(heap, (nd, nbr, edge_line))

    # Collapse to best distance per station
    best = {}
    for (stn, _), d in dist.items():
        if stn not in best or d < best[stn]:
            best[stn] = d
    return best

# ── Edge index for GNN (undirected → both directions) ────────────────────

_src, _dst = [], []
for u, v in G.edges():
    ui, vi = STATION_INDEX[u], STATION_INDEX[v]
    _src += [ui, vi]
    _dst += [vi, ui]
STN_EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

# ── Zone classification ──────────────────────────────────────────────────
# Each station gets one primary zone.  Unmentioned stations → "residential".

_ZONE_DEFS = {
    "tourist": [
        "Sagrada Família", "Passeig de Gràcia", "Barceloneta",
        "Drassanes", "Liceu", "Jaume I", "Arc de Triomf",
        "Espanya", "Lesseps", "Ciutadella-Vila Olímpica",
    ],
    "business": [
        "Diagonal", "Universitat", "Urquinaona", "Catalunya",
        "Sant Antoni", "Tetuan", "Monumental", "Girona",
        "Glòries", "Plaça del Centre", "Entença",
    ],
    "university": [
        "Zona Universitària", "Palau Reial", "Maria Cristina", "Les Corts",
    ],
    "hospital": [
        "Hospital de Bellvitge", "Hospital Clínic",
        "Guinardó-Hospital de Sant Pau", "Sant Pau-Dos de Maig",
    ],
    "airport": [
        "Aeroport T1", "Aeroport T2", "Mas Blau", "Parc Nou",
        "Cèntric", "El Prat Estació",
    ],
    "leisure": [
        "Bogatell", "Fontana", "Poble Sec", "Vallcarca",
        "Montbau", "Mundet",
    ],
    "industrial": [
        "Zona Franca", "Port Comercial-La Factoria", "Ecoparc",
        "ZAL-Riu Vell", "Mercabarna", "Parc Logístic", "Foc",
        "Foneria", "Fira", "Europa-Fira",
    ],
}

STATION_ZONE = {}
for zone, stations in _ZONE_DEFS.items():
    for s in stations:
        if s in STATION_INDEX:
            STATION_ZONE[s] = zone
for s in STATION_LIST:
    if s not in STATION_ZONE:
        STATION_ZONE[s] = "residential"

ZONE_LIST  = sorted(set(STATION_ZONE.values()))
ZONE_INDEX = {z: i for i, z in enumerate(ZONE_LIST)}
NUM_ZONES  = len(ZONE_LIST)

# ── Candidate found-at stations (for efficient training) ─────────────────
# All transfer stations + all line termini.  Items are most likely found/
# turned in at busy interchange and end-of-line stations.

CANDIDATE_STATIONS = sorted(set(
    list(TRANSFERS.keys())
    + [line[0]  for line in ALL_LINES.values()]
    + [line[-1] for line in ALL_LINES.values()]
))

# ── Quick summary ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Stations : {N_STN}")
    print(f"Edges    : {G.number_of_edges()}")
    print(f"Transfers: {len(TRANSFERS)}")
    print(f"Zones    : {ZONE_LIST}")
    print(f"Candidate found-at stations: {len(CANDIDATE_STATIONS)}")
