"""
Draw the Barcelona metro network as modelled by the data.

Station positions use real geographic coordinates (WGS84) so the map
matches the actual layout of Barcelona.

Usage:
  python -m barcelona_demo.network_map
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from metro import ALL_LINES, LINE_COLORS
from network import G, STATION_LIST, STATION_INDEX, TRANSFERS, STATION_ZONE

# ── Real geographic coordinates (lat, lon) ───────────────────────────────
# Sources: OpenStreetMap, TMB, Wikipedia. Estimated for a few newer stations.

_GEO = {
    # L1
    "Hospital de Bellvitge": (41.344650, 2.107201),
    "Bellvitge": (41.350965, 2.110935),
    "Avinguda Carrilet": (41.358913, 2.102483),
    "Rambla Just Oliveras": (41.363965, 2.099872),
    "Can Serra": (41.367684, 2.102736),
    "Florida": (41.368323, 2.109832),
    "Torrassa": (41.368406, 2.116631),
    "Santa Eulàlia": (41.368799, 2.128605),
    "Mercat Nou": (41.373553, 2.134034),
    "Plaça de Sants": (41.375376, 2.137913),
    "Hostafrancs": (41.375242, 2.143419),
    "Espanya": (41.375569, 2.149524),
    "Rocafort": (41.379237, 2.154573),
    "Urgell": (41.382647, 2.158700),
    "Universitat": (41.385694, 2.164305),
    "Catalunya": (41.386547, 2.170084),
    "Urquinaona": (41.388085, 2.173311),
    "Arc de Triomf": (41.392114, 2.180909),
    "Marina": (41.395600, 2.186400),
    "Glòries": (41.402900, 2.187200),
    "Clot": (41.410800, 2.187200),
    "Navas": (41.416200, 2.187000),
    "La Sagrera": (41.422623, 2.187024),
    "Fabra i Puig": (41.429781, 2.183639),
    "Sant Andreu": (41.437300, 2.190800),
    "Torras i Bages": (41.443127, 2.190714),
    "Trinitat Vella": (41.448890, 2.193610),
    "Baró de Viver": (41.449947, 2.199539),
    "Santa Coloma": (41.451076, 2.208049),
    "Fondo": (41.451588, 2.218392),
    # L2
    "Paral·lel": (41.375180, 2.167774),
    "Sant Antoni": (41.379786, 2.163270),
    "Passeig de Gràcia": (41.389524, 2.168228),
    "Tetuan": (41.395024, 2.174654),
    "Monumental": (41.400524, 2.179455),
    "Sagrada Família": (41.403980, 2.174812),
    "Encants": (41.407075, 2.182523),
    "Bac de Roda": (41.414881, 2.195106),
    "Sant Martí": (41.419057, 2.200731),
    "La Pau": (41.424039, 2.205618),
    "Verneda": (41.429890, 2.209775),
    "Artigues-Sant Adrià": (41.433721, 2.217573),
    "Sant Roc": (41.435317, 2.227861),
    "Gorg": (41.440441, 2.233692),
    "Pep Ventura": (41.443939, 2.238052),
    "Badalona Pompeu Fabra": (41.448961, 2.244221),
    # L3
    "Zona Universitària": (41.384244, 2.111686),
    "Palau Reial": (41.385808, 2.117855),
    "Maria Cristina": (41.387940, 2.126222),
    "Les Corts": (41.384134, 2.130878),
    "Plaça del Centre": (41.381821, 2.135630),
    "Sants Estació": (41.381469, 2.141114),
    "Tarragona": (41.378444, 2.145368),
    "Poble Sec": (41.374980, 2.160572),
    "Drassanes": (41.376673, 2.175774),
    "Liceu": (41.381396, 2.173071),
    "Diagonal": (41.395956, 2.159963),
    "Fontana": (41.402056, 2.152979),
    "Lesseps": (41.406163, 2.150025),
    "Vallcarca": (41.411907, 2.144524),
    "Penitents": (41.417541, 2.141133),
    "Vall d'Hebron": (41.425300, 2.142600),
    "Montbau": (41.430595, 2.145035),
    "Mundet": (41.435542, 2.148383),
    "Valldaura": (41.438004, 2.156876),
    "Canyelles": (41.441770, 2.166439),
    "Roquetes": (41.447355, 2.175783),
    "Trinitat Nova": (41.449483, 2.184162),
    # L4
    "Via Júlia": (41.443900, 2.178900),
    "Llucmajor": (41.437009, 2.173364),
    "Maragall": (41.425028, 2.176504),
    "Guinardó-Hospital de Sant Pau": (41.416042, 2.174364),
    "Alfons X": (41.412100, 2.165000),
    "Joanic": (41.405560, 2.163060),
    "Verdaguer": (41.400091, 2.168428),
    "Girona": (41.395024, 2.170960),
    "Jaume I": (41.384001, 2.178706),
    "Barceloneta": (41.382117, 2.185482),
    "Ciutadella-Vila Olímpica": (41.387847, 2.193476),
    "Bogatell": (41.395099, 2.192040),
    "Llacuna": (41.399381, 2.197649),
    "Poblenou": (41.403642, 2.203275),
    "Selva de Mar": (41.408015, 2.209109),
    "El Maresme-Fòrum": (41.411775, 2.216678),
    "Besòs Mar": (41.415094, 2.216085),
    "Besòs": (41.420369, 2.209713),
    # L5
    "Cornellà Centre": (41.357303, 2.070552),
    "Gavarra": (41.357985, 2.079215),
    "Sant Ildefons": (41.362800, 2.084000),
    "Can Boixeres": (41.366582, 2.091518),
    "Can Vidalet": (41.371600, 2.099200),
    "Pubilla Cases": (41.373813, 2.107121),
    "Ernest Lluch": (41.376390, 2.111670),
    "Collblanc": (41.375874, 2.118405),
    "Badal": (41.375552, 2.128927),
    "Entença": (41.383982, 2.144960),
    "Hospital Clínic": (41.388162, 2.150543),
    "Sant Pau-Dos de Maig": (41.410200, 2.175300),
    "Camp de l'Arpa": (41.414686, 2.181013),
    "Congrés": (41.423416, 2.181285),
    "Virrei Amat": (41.429594, 2.174999),
    "Vilapicina": (41.430460, 2.168173),
    "Horta": (41.429690, 2.160316),
    "El Carmel": (41.423600, 2.155100),
    "El Coll-La Teixonera": (41.421657, 2.149362),
    # L9 Nord
    "Onze de Setembre": (41.429760, 2.193435),
    "Bon Pastor": (41.436126, 2.204808),
    "Can Peixauet": (41.444472, 2.210286),
    "Santa Rosa": (41.446700, 2.215900),
    "Església Major": (41.454759, 2.212402),
    "Singuerlín": (41.459458, 2.205593),
    "Can Zam": (41.457679, 2.198494),
    # L9 Sud
    "Aeroport T1": (41.288258, 2.071040),
    "Aeroport T2": (41.303826, 2.073121),
    "Mas Blau": (41.311319, 2.073422),
    "Parc Nou": (41.316707, 2.087351),
    "Cèntric": (41.321844, 2.093565),
    "El Prat Estació": (41.331667, 2.090129),
    "Les Moreres": (41.328985, 2.103072),
    "Mercabarna": (41.333570, 2.111229),
    "Parc Logístic": (41.341806, 2.127983),
    "Fira": (41.351923, 2.130679),
    "Europa-Fira": (41.357249, 2.125690),
    "Can Tries-Gornal": (41.360721, 2.118199),
    # L10 Nord
    "Llefià": (41.441303, 2.217548),
    "La Salut": (41.442733, 2.224576),
    # L10 Sud
    "Provençana": (41.361389, 2.124293),
    "Ciutat de la Justícia": (41.363244, 2.133009),
    "Foneria": (41.360457, 2.138656),
    "Foc": (41.355851, 2.142162),
    "Zona Franca": (41.342985, 2.144972),
    "Port Comercial-La Factoria": (41.335984, 2.140666),
    "Ecoparc": (41.330184, 2.137090),
    "ZAL-Riu Vell": (41.323746, 2.133117),
    # L11
    "Casa de l'Aigua": (41.451500, 2.185400),
    "Torre Baró-Vallbona": (41.459580, 2.179900),
    "Ciutat Meridiana": (41.460900, 2.174400),
    "Can Cuiàs": (41.462700, 2.171500),
    # L6 (FGC)
    "Provença": (41.392874, 2.157979),
    "Gràcia": (41.399757, 2.152155),
    "Sant Gervasi": (41.401094, 2.147103),
    "Muntaner": (41.398686, 2.142791),
    "La Bonanova": (41.397813, 2.136152),
    "Les Tres Torres": (41.397796, 2.130860),
    "Sarrià": (41.398604, 2.125702),
    # L7 (FGC)
    "Plaça Molina": (41.401320, 2.147424),
    "Pàdua": (41.403326, 2.143059),
    "El Putxet": (41.405812, 2.139159),
    "Avinguda Tibidabo": (41.409396, 2.137297),
    # L8 (FGC)
    "Magòria-La Campana": (41.367858, 2.139368),
    "Ildefons Cerdà": (41.361165, 2.130169),
    "Gornal": (41.354775, 2.117073),
    "Sant Josep": (41.360951, 2.110361),
    "Almeda": (41.353056, 2.085278),
    "Cornellà-Riera": (41.351389, 2.070833),
    "Sant Boi": (41.348056, 2.043333),
    "Molí Nou-Ciutat Cooperativa": (41.357525, 2.035278),
}


def get_positions():
    """Convert (lat, lon) to (x, y) plot coordinates for all stations."""
    pos = {}
    for stn in STATION_LIST:
        if stn in _GEO:
            lat, lon = _GEO[stn]
            pos[stn] = np.array([lon, lat])  # x=lon, y=lat
    return pos


# ── Zone colours for station dots ────────────────────────────────────────

_ZONE_COLORS = {
    "tourist":     "#E74C3C",
    "business":    "#3498DB",
    "university":  "#9B59B6",
    "hospital":    "#2ECC71",
    "airport":     "#F39C12",
    "leisure":     "#1ABC9C",
    "industrial":  "#7F8C8D",
    "residential": "#BDC3C7",
}


def main():
    pos = get_positions()

    fig, ax = plt.subplots(figsize=(22, 20))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F5F0")

    # ── Draw metro lines ─────────────────────────────────────────────
    for line_name, stations in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888888")
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            if s1 in pos and s2 in pos:
                ax.plot([pos[s1][0], pos[s2][0]],
                        [pos[s1][1], pos[s2][1]],
                        color=color, linewidth=2.8, alpha=0.55, zorder=1,
                        solid_capstyle='round')

    # ── Draw all station nodes ───────────────────────────────────────
    for stn in STATION_LIST:
        if stn not in pos:
            continue
        zone  = STATION_ZONE.get(stn, "residential")
        color = _ZONE_COLORS.get(zone, "#BDC3C7")
        size  = 40 if stn not in TRANSFERS else 80
        ax.scatter(*pos[stn], s=size, c=color, edgecolors='#333333',
                   linewidths=0.4, zorder=3)

    # ── Transfer station rings ───────────────────────────────────────
    for stn in TRANSFERS:
        if stn in pos:
            ax.scatter(*pos[stn], s=100, facecolors='none',
                       edgecolors='black', linewidths=1.2, zorder=4)

    # ── Label transfer stations + termini ────────────────────────────
    termini = set()
    for stations in ALL_LINES.values():
        termini.add(stations[0])
        termini.add(stations[-1])

    labelled = set(TRANSFERS.keys()) | termini
    for stn in labelled:
        if stn not in pos:
            continue
        ax.annotate(
            stn, pos[stn], fontsize=5, ha='center', va='bottom',
            xytext=(0, 5), textcoords='offset points',
            fontweight='bold', color='#222222',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='#CCCCCC', alpha=0.85, lw=0.5),
            zorder=5,
        )

    # ── Legend: metro lines ──────────────────────────────────────────
    line_patches = []
    for line_name in sorted(ALL_LINES.keys()):
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888888")
        line_patches.append(mpatches.Patch(color=color, label=line_name))

    leg1 = ax.legend(handles=line_patches, loc='upper left',
                     fontsize=7, title="Metro lines", title_fontsize=8,
                     framealpha=0.9, ncol=2)
    ax.add_artist(leg1)

    # ── Legend: station zones ────────────────────────────────────────
    zone_patches = []
    for zone in sorted(_ZONE_COLORS.keys()):
        zone_patches.append(
            mpatches.Patch(color=_ZONE_COLORS[zone], label=zone))
    ax.legend(handles=zone_patches, loc='upper right',
              fontsize=7, title="Station zones", title_fontsize=8,
              framealpha=0.9)

    # ── Title + stats ────────────────────────────────────────────────
    ax.set_title(
        f"Barcelona Metro Network  —  "
        f"{len(STATION_LIST)} stations, {G.number_of_edges()} edges, "
        f"{len(TRANSFERS)} transfers",
        fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.axis('off')

    fname = os.path.join(os.path.dirname(__file__), "barcelona_network.png")
    fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
