"""Dump station coordinates + metro line segments + transfers + zones
to a JSON file the TMB demo loads to draw the inline SVG metro chart.

Run once whenever the network changes:
    python3 dump_geom.py
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from network_map import _GEO
from metro import ALL_LINES, LINE_COLORS
from network import STATION_LIST, TRANSFERS, STATION_ZONE


OUT = Path(__file__).parent.parent / "TMB" / "network_geom.json"


def main():
    # lon = x, lat = y (we flip y in the SVG itself with a -1 transform).
    stations = {}
    for stn in STATION_LIST:
        if stn not in _GEO:
            continue
        lat, lon = _GEO[stn]
        stations[stn] = {
            "x": lon,
            "y": lat,
            "zone": STATION_ZONE.get(stn, "residential"),
            "transfer": stn in TRANSFERS,
        }

    lines = {}
    for line_name, sts in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888888")
        segs = []
        for a, b in zip(sts, sts[1:]):
            if a in stations and b in stations:
                segs.append([a, b])
        lines[line_name] = {"color": color, "segments": segs}

    out = {"stations": stations, "lines": lines}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}  ({len(stations)} stations, {len(lines)} lines)")


if __name__ == "__main__":
    main()
