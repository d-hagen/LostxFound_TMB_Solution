"""
Barcelona Metro Network (TMB - Transports Metropolitans de Barcelona)
=====================================================================
Complete station data for lines L1-L11 (TMB-operated lines).

Data sourced from TMB official site and Wikipedia (May 2026).

Each line is defined as an ordered list of station names.
Transfer stations appear on multiple lines with identical spelling
so they can be detected automatically when building the graph.

Also included: FGC urban lines L6, L7, L8 that share interchange
stations with the TMB network.
"""

# ── Line colours (official TMB / FGC) ─────────────────────────────────
LINE_COLORS = {
    "L1":      {"name": "Red",          "hex": "#E2001A"},
    "L2":      {"name": "Purple",       "hex": "#9B2791"},
    "L3":      {"name": "Green",        "hex": "#349036"},
    "L4":      {"name": "Yellow",       "hex": "#FFAA00"},
    "L5":      {"name": "Blue",         "hex": "#0078BE"},
    "L6":      {"name": "Violet",       "hex": "#7B76A8"},  # FGC
    "L7":      {"name": "Brown",        "hex": "#9B6733"},  # FGC
    "L8":      {"name": "Pink",         "hex": "#EA6BA5"},  # FGC
    "L9 Nord": {"name": "Orange",       "hex": "#F58220"},
    "L9 Sud":  {"name": "Orange",       "hex": "#F58220"},
    "L10 Nord":{"name": "Light Blue",   "hex": "#00ACED"},
    "L10 Sud": {"name": "Light Blue",   "hex": "#00ACED"},
    "L11":     {"name": "Light Green",  "hex": "#8FD400"},
}

# ── Station lists, in order from terminus A → terminus B ──────────────

LINES = {
    # ----------------------------------------------------------------
    # L1  Hospital de Bellvitge ↔ Fondo  (30 stations, Red)
    # Runs SW→NE: L'Hospitalet → Eixample → Sant Andreu → Santa Coloma
    # ----------------------------------------------------------------
    "L1": [
        "Hospital de Bellvitge",
        "Bellvitge",
        "Avinguda Carrilet",
        "Rambla Just Oliveras",
        "Can Serra",
        "Florida",
        "Torrassa",
        "Santa Eulàlia",
        "Mercat Nou",
        "Plaça de Sants",
        "Hostafrancs",
        "Espanya",
        "Rocafort",
        "Urgell",
        "Universitat",
        "Catalunya",
        "Urquinaona",
        "Arc de Triomf",
        "Marina",
        "Glòries",
        "Clot",
        "Navas",
        "La Sagrera",
        "Fabra i Puig",
        "Sant Andreu",
        "Torras i Bages",
        "Trinitat Vella",
        "Baró de Viver",
        "Santa Coloma",
        "Fondo",
    ],

    # ----------------------------------------------------------------
    # L2  Paral·lel ↔ Badalona Pompeu Fabra  (18 stations, Purple)
    # Runs SW→NE: Paral·lel → Eixample → Sant Martí → Badalona
    # ----------------------------------------------------------------
    "L2": [
        "Paral·lel",
        "Sant Antoni",
        "Universitat",
        "Passeig de Gràcia",
        "Tetuan",
        "Monumental",
        "Sagrada Família",
        "Encants",
        "Clot",
        "Bac de Roda",
        "Sant Martí",
        "La Pau",
        "Verneda",
        "Artigues-Sant Adrià",
        "Sant Roc",
        "Gorg",
        "Pep Ventura",
        "Badalona Pompeu Fabra",
    ],

    # ----------------------------------------------------------------
    # L3  Zona Universitària ↔ Trinitat Nova  (26 stations, Green)
    # Runs W→centre→N: Zona Universitària → Sants → Ciutat Vella →
    #   Gràcia → Horta-Guinardó → Nou Barris
    # ----------------------------------------------------------------
    "L3": [
        "Zona Universitària",
        "Palau Reial",
        "Maria Cristina",
        "Les Corts",
        "Plaça del Centre",
        "Sants Estació",
        "Tarragona",
        "Espanya",
        "Poble Sec",
        "Paral·lel",
        "Drassanes",
        "Liceu",
        "Catalunya",
        "Passeig de Gràcia",
        "Diagonal",
        "Fontana",
        "Lesseps",
        "Vallcarca",
        "Penitents",
        "Vall d'Hebron",
        "Montbau",
        "Mundet",
        "Valldaura",
        "Canyelles",
        "Roquetes",
        "Trinitat Nova",
    ],

    # ----------------------------------------------------------------
    # L4  Trinitat Nova ↔ La Pau  (22 stations, Yellow)
    # Runs N→centre→SE: Nou Barris → Gràcia → Eixample → Ciutat Vella →
    #   seafront → Besòs
    # ----------------------------------------------------------------
    "L4": [
        "Trinitat Nova",
        "Via Júlia",
        "Llucmajor",
        "Maragall",
        "Guinardó-Hospital de Sant Pau",
        "Alfons X",
        "Joanic",
        "Verdaguer",
        "Girona",
        "Passeig de Gràcia",
        "Urquinaona",
        "Jaume I",
        "Barceloneta",
        "Ciutadella-Vila Olímpica",
        "Bogatell",
        "Llacuna",
        "Poblenou",
        "Selva de Mar",
        "El Maresme-Fòrum",
        "Besòs Mar",
        "Besòs",
        "La Pau",
    ],

    # ----------------------------------------------------------------
    # L5  Cornellà Centre ↔ Vall d'Hebron  (27 stations, Blue)
    # Runs SW→N: Cornellà → L'Hospitalet → Sants → Eixample →
    #   Gràcia → Horta-Guinardó
    # ----------------------------------------------------------------
    "L5": [
        "Cornellà Centre",
        "Gavarra",
        "Sant Ildefons",
        "Can Boixeres",
        "Can Vidalet",
        "Pubilla Cases",
        "Ernest Lluch",
        "Collblanc",
        "Badal",
        "Plaça de Sants",
        "Sants Estació",
        "Entença",
        "Hospital Clínic",
        "Diagonal",
        "Verdaguer",
        "Sagrada Família",
        "Sant Pau-Dos de Maig",
        "Camp de l'Arpa",
        "La Sagrera",
        "Congrés",
        "Maragall",
        "Virrei Amat",
        "Vilapicina",
        "Horta",
        "El Carmel",
        "El Coll-La Teixonera",
        "Vall d'Hebron",
    ],

    # ----------------------------------------------------------------
    # L9 Nord  La Sagrera ↔ Can Zam  (9 stations, Orange)
    # Runs S→N: La Sagrera → Sant Andreu → Santa Coloma de Gramenet
    # ----------------------------------------------------------------
    "L9 Nord": [
        "La Sagrera",
        "Onze de Setembre",
        "Bon Pastor",
        "Can Peixauet",
        "Santa Rosa",
        "Fondo",
        "Església Major",
        "Singuerlín",
        "Can Zam",
    ],

    # ----------------------------------------------------------------
    # L9 Sud  Aeroport T1 ↔ Zona Universitària  (15 stations, Orange)
    # Runs S→NW: Airport → El Prat → Zona Franca → L'Hospitalet →
    #   Zona Universitària
    # ----------------------------------------------------------------
    "L9 Sud": [
        "Aeroport T1",
        "Aeroport T2",
        "Mas Blau",
        "Parc Nou",
        "Cèntric",
        "El Prat Estació",
        "Les Moreres",
        "Mercabarna",
        "Parc Logístic",
        "Fira",
        "Europa-Fira",
        "Can Tries-Gornal",
        "Torrassa",
        "Collblanc",
        "Zona Universitària",
    ],

    # ----------------------------------------------------------------
    # L10 Nord  La Sagrera ↔ Gorg  (6 stations, Light Blue)
    # Runs S→NE: La Sagrera → Sant Andreu → Badalona
    # ----------------------------------------------------------------
    "L10 Nord": [
        "La Sagrera",
        "Onze de Setembre",
        "Bon Pastor",
        "Llefià",
        "La Salut",
        "Gorg",
    ],

    # ----------------------------------------------------------------
    # L10 Sud  Collblanc ↔ ZAL-Riu Vell  (11 stations, Light Blue)
    # Runs NW→S: L'Hospitalet → Zona Franca → Port
    # ----------------------------------------------------------------
    "L10 Sud": [
        "Collblanc",
        "Torrassa",
        "Can Tries-Gornal",
        "Provençana",
        "Ciutat de la Justícia",
        "Foneria",
        "Foc",
        "Zona Franca",
        "Port Comercial-La Factoria",
        "Ecoparc",
        "ZAL-Riu Vell",
    ],

    # ----------------------------------------------------------------
    # L11  Trinitat Nova ↔ Can Cuiàs  (5 stations, Light Green)
    # Light metro, runs N: Trinitat Nova → Torre Baró → Montcada i Reixac
    # ----------------------------------------------------------------
    "L11": [
        "Trinitat Nova",
        "Casa de l'Aigua",
        "Torre Baró-Vallbona",
        "Ciutat Meridiana",
        "Can Cuiàs",
    ],
}

# ── FGC urban lines (included for interchange completeness) ───────────

FGC_LINES = {
    # L6  Catalunya ↔ Sarrià  (8 stations, Violet)
    # Runs centre→W through Sarrià-Sant Gervasi
    "L6": [
        "Catalunya",
        "Provença",
        "Gràcia",
        "Sant Gervasi",
        "Muntaner",
        "La Bonanova",
        "Les Tres Torres",
        "Sarrià",
    ],

    # L7  Catalunya ↔ Avinguda Tibidabo  (7 stations, Brown)
    # Runs centre→NW through Gràcia toward Tibidabo
    "L7": [
        "Catalunya",
        "Provença",
        "Gràcia",
        "Plaça Molina",
        "Pàdua",
        "El Putxet",
        "Avinguda Tibidabo",
    ],

    # L8  Espanya ↔ Molí Nou-Ciutat Cooperativa  (11 stations, Pink)
    # Runs centre→SW: Espanya → L'Hospitalet → Cornellà → Sant Boi
    "L8": [
        "Espanya",
        "Magòria-La Campana",
        "Ildefons Cerdà",
        "Europa-Fira",
        "Gornal",
        "Sant Josep",
        "Avinguda Carrilet",
        "Almeda",
        "Cornellà-Riera",
        "Sant Boi",
        "Molí Nou-Ciutat Cooperativa",
    ],
}


# ── Transfer stations ─────────────────────────────────────────────────
# Auto-detected: stations whose name appears on more than one line.

def find_transfers(line_dict):
    """Return dict: station_name -> set of line names it appears on."""
    station_lines = {}
    for line_name, stations in line_dict.items():
        for s in stations:
            station_lines.setdefault(s, set()).add(line_name)
    return {s: lines for s, lines in station_lines.items() if len(lines) > 1}


ALL_LINES = {**LINES, **FGC_LINES}
TRANSFERS = find_transfers(ALL_LINES)


# ── Build edge list ───────────────────────────────────────────────────

def build_edges(line_dict=None):
    """
    Return list of (station_a, station_b, line_name) for every adjacent
    pair of stations on every line.  Transfer edges (between lines at
    the same station) are NOT included here; they are implicit via
    shared station names.
    """
    if line_dict is None:
        line_dict = ALL_LINES
    edges = []
    for line_name, stations in line_dict.items():
        for i in range(len(stations) - 1):
            edges.append((stations[i], stations[i + 1], line_name))
    return edges


def build_adjacency(line_dict=None):
    """
    Return dict: station -> set of (neighbour, line_name).
    Includes both directions (undirected graph).
    """
    if line_dict is None:
        line_dict = ALL_LINES
    adj = {}
    for a, b, line in build_edges(line_dict):
        adj.setdefault(a, set()).add((b, line))
        adj.setdefault(b, set()).add((a, line))
    return adj


# ── Geographic zones (rough) ──────────────────────────────────────────
# Useful for assigning coordinates or clustering.

LINE_GEOGRAPHY = {
    "L1":       "SW-NE diagonal: L'Hospitalet de Llobregat → city centre "
                "(Eixample) → Sant Andreu → Santa Coloma de Gramenet",
    "L2":       "SW-NE: Paral·lel (Ciutat Vella) → Eixample → Sant Martí "
                "→ Badalona",
    "L3":       "W-centre-N: Zona Universitària (Les Corts) → Sants → "
                "Ciutat Vella → Gràcia → Horta-Guinardó → Nou Barris",
    "L4":       "N-centre-SE: Nou Barris → Gràcia → Eixample → Ciutat "
                "Vella (Barceloneta) → seafront → Besòs",
    "L5":       "SW-N arc: Cornellà → L'Hospitalet → Sants → Eixample → "
                "Gràcia → Horta-Guinardó",
    "L6":       "Centre-W: Plaça Catalunya → Sarrià (upper west Barcelona)",
    "L7":       "Centre-NW: Plaça Catalunya → Avinguda Tibidabo",
    "L8":       "Centre-SW: Espanya → L'Hospitalet → Cornellà → Sant Boi",
    "L9 Nord":  "S-N: La Sagrera → Sant Andreu → Santa Coloma de Gramenet",
    "L9 Sud":   "S-NW: Barcelona Airport (El Prat) → Zona Franca → "
                "L'Hospitalet → Zona Universitària",
    "L10 Nord": "S-NE: La Sagrera → Sant Andreu → Badalona",
    "L10 Sud":  "NW-S: L'Hospitalet → Zona Franca → Port of Barcelona",
    "L11":      "N: Trinitat Nova → Torre Baró → Montcada i Reixac "
                "(light metro, northern hills)",
}


# ── Quick summary ─────────────────────────────────────────────────────

if __name__ == "__main__":
    all_stations = set()
    for stations in ALL_LINES.values():
        all_stations.update(stations)

    edges = build_edges()
    print(f"Lines:    {len(ALL_LINES)}")
    print(f"Stations: {len(all_stations)}  (unique names)")
    print(f"Edges:    {len(edges)}  (adjacent pairs)")
    print(f"Transfers: {len(TRANSFERS)} stations\n")

    print("Transfer stations:")
    for station, lines in sorted(TRANSFERS.items()):
        print(f"  {station:35s} {', '.join(sorted(lines))}")
