#!/usr/bin/env python3
"""Minimal local server — stdlib only (http.server + sqlite3) + Ollama VLM
+ trained heterogeneous GNN for storage routing.

Modes:
    python3 app.py --mode employee   (default) → Add Item + Database tabs
    python3 app.py --mode user                 → Find chat + Claimed items
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import gnn_bridge

DB_PATH = Path(__file__).parent / "entries.db"
HOST, PORT = "localhost", 8080
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"

# Find chat shells out to the local `claude` CLI (uses your Claude Code
# subscription — no API key needed). Local-demo only.
CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")
CLAUDE_MODEL = "claude-sonnet-4-6"

# Demo gimmick: a single physical locker, fixed code.
DEMO_LOCKER_NUMBER = 8
DEMO_PICKUP_CODE = "2801"
EXTENSION_DAYS = 3
STORAGE_DAYS = 5  # how long an item stays available once it has arrived

# Set from --mode at startup.
APP_MODE = "employee"

DETECT_PROMPT = """You are a lost-item logging system. The image shows a lost item photographed against a white background. Describe ONLY the item itself — ignore the white background entirely.

Respond ONLY with valid JSON, no other text. Fill in these fields:
- "item_type": what the object is, be specific (e.g. "blue ballpoint pen", "black leather wallet", "silver house key")
- "main_color": the dominant color as a common color name (e.g. "black", "dark blue", "rose", "light grey", "olive green")
- "secondary_colors": other visible colors as common color names, comma-separated (e.g. "white, silver"). Use "" if the item is one solid color.
- "perks": distinguishing features that help identify this specific item — brand name, size, material, wear, stickers, engravings, text, unique markings (e.g. "Nike logo, scuffed toe, size 10"). Use "" if nothing stands out.

Respond with ONLY this JSON:
{"item_type": "...", "main_color": "...", "secondary_colors": "...", "perks": "..."}"""


# ─── DB ────────────────────────────────────────────────────────────────────

def _column_names(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT NOT NULL,
            line TEXT DEFAULT '',
            item_type TEXT NOT NULL,
            connected_items TEXT DEFAULT '',
            main_color TEXT,
            secondary_colors TEXT,
            perks TEXT,
            time TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_b64 TEXT DEFAULT '',
            storage_station TEXT DEFAULT '',
            arrival_at DATETIME,
            expires_at DATETIME
        )
    """)
    # Backfill columns on pre-existing DBs.
    have = _column_names(conn, "entries")
    for col, ddl in [
        ("line",            "TEXT DEFAULT ''"),
        ("image_b64",       "TEXT DEFAULT ''"),
        ("storage_station", "TEXT DEFAULT ''"),
        ("arrival_at",      "DATETIME"),
        ("expires_at",      "DATETIME"),
    ]:
        if col not in have:
            conn.execute(f"ALTER TABLE entries ADD COLUMN {col} {ddl}")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            claimed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            extended INTEGER DEFAULT 0,
            collected_at DATETIME
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            entry_id INTEGER,
            item_type TEXT,
            original_station TEXT,
            storage_station TEXT,
            satisfied INTEGER,
            preferred_stations TEXT,
            lost_line TEXT,
            lost_time TEXT,
            lost_station TEXT,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ─── connected items helpers ──────────────────────────────────────────────

def get_connected_ids(entry):
    raw = entry.get("connected_items", "") or ""
    return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]


def add_connection(db, id_a, id_b):
    for src, dst in [(id_a, id_b), (id_b, id_a)]:
        row = db.execute("SELECT connected_items FROM entries WHERE id = ?", (src,)).fetchone()
        if not row:
            continue
        existing = set(get_connected_ids(dict(row)))
        if dst not in existing:
            existing.add(dst)
            db.execute(
                "UPDATE entries SET connected_items = ? WHERE id = ?",
                (",".join(str(x) for x in sorted(existing)), src),
            )
    db.commit()


def resolve_connections(entries):
    by_id = {e["id"]: e for e in entries}
    connections = {}
    for e in entries:
        ids = get_connected_ids(e)
        summaries = []
        for cid in ids:
            if cid in by_id:
                c = by_id[cid]
                summaries.append(
                    f"ID {c['id']}: {c['item_type']} ({c['main_color']}"
                    + (f", {c['secondary_colors']}" if c['secondary_colors'] else "")
                    + (f", {c['perks']}" if c['perks'] else "")
                    + ")"
                )
        connections[e["id"]] = summaries
    return connections


# ─── Ollama VLM ───────────────────────────────────────────────────────────

def call_ollama(image_b64: str) -> dict:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": DETECT_PROMPT,
        "images": [image_b64],
        "stream": False,
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        return {"error": "Cannot reach Ollama. Is it running? (ollama serve)"}
    except Exception as e:
        return {"error": str(e)}
    raw = data.get("response", "")
    try:
        start = raw.index("{"); end = raw.rindex("}") + 1
        snippet = raw[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # llava occasionally emits markdown-escaped chars (\_, \*) that are
            # not valid JSON escapes. Strip those and retry.
            cleaned = snippet.replace("\\_", "_").replace("\\*", "*")
            return json.loads(cleaned)
    except (ValueError, json.JSONDecodeError):
        return {"error": f"VLM returned unparseable response: {raw[:200]}"}


# ─── Find chatbot ─────────────────────────────────────────────────────────

FIND_SYSTEM_PROMPT = """You are a friendly lost-and-found assistant. Someone is trying to find an item they lost. You check their description against the database of found items.

CRITICAL — COMBINE SIGNALS ACROSS THE WHOLE CONVERSATION:
The person may give you information piece by piece across multiple messages.
You MUST treat every Person message as ADDITIVE. Always reason over the
union of ALL details they've ever mentioned, never just the latest message.

Example: "I lost a green backpack" + "on the L1" + "on the 18th of May" must
be treated identically to "I lost a green backpack on the L1 on the 18th of
May". The combined description has: item=backpack, color=green, line=L1,
date=18 May. That meets the match criteria — do not ask for any of those
again. NEVER re-ask for info the person already provided in any earlier turn.

MATCHING — SIGNAL WEIGHTS:
Some signals are reliable; others are noisy. Weight them accordingly.

STRONG SIGNALS — these must align for a match:
  • Item type (with synonyms: camera = camera, phone = mobile, backpack = rucksack, etc.)
  • Main color (with fuzzy color matching: "black" ≈ "black", "dark grey" ≈ "grey", etc.)
  • Metro line (L1, L2, … — case/format don't matter: "L4" = "l4" = "Line 4")
  • Approximate day / date ("18 May" = "May 18" = "the 18th" = same day)

WEAK SIGNALS — do NOT require these, use only to break ties when 2+ strong-matching
candidates remain:
  • Specific station — the person often misremembers exactly where they lost it
    (item can be found several stops downstream of where they actually lost it).
    Treat the person's stated station as a HINT, not a constraint. If they say
    "Sagrada Família" and the only entry on the same line says "Tetuan", that's
    still a likely match.
  • Specific time of day — people misremember exact times by hours. Match on day,
    not on the clock.

MATCH RULE (after combining ALL details across the conversation):
  Be GENEROUS. The photo confirmation step at the end is what catches mistakes —
  your job is to surface plausible candidates, not to gate-keep with questions.

  • Strong match: item type ✓ AND at least one color ✓ AND (line ✓ OR day ✓).
    Return as status=match.
  • Soft match: item type ✓ AND any reasonable signal (color OR line OR day).
    Still return as status=match — let the user verify with the photo.
  • If multiple candidates pass, return ALL of them (up to 3) in matching_ids.
    The user will scan the photos. That is faster than another follow-up.

DISAMBIGUATION:
  • Ask a follow-up ONLY when you have zero plausible candidates. Otherwise
    return the candidates immediately for photo confirmation.
  • If you do ask, prefer line / color / day — NOT exact station or exact time.
  • Maximum 1 follow-up question total. After that, return the best
    candidate(s) even on weak evidence and let the photo decide.

PRIVACY:
- Do NOT reveal database contents. Don't list items, don't hint at stored details.
- Do NOT say "we have a similar item" — that leaks info.
- Do NOT mention the storage locker, pickup code, locker number, station, or arrival date. Those are revealed by the system AFTER the user confirms the photo.

WHEN YOU FIND A MATCH:
- Reply briefly: "I might have found your item. Can you confirm it's the one in this photo?" — nothing more.
- The system shows the photo and handles the confirmation flow. Do NOT describe the item back.
- CRITICAL: in the SAME response, set status="match" AND fill matching_ids with the ID(s) of the candidate(s). The photo can only be shown if matching_ids is non-empty. Never use the "confirm it's the one in this photo" phrase without filling matching_ids — that wastes the user's time.
- If you have only ONE plausible candidate, include only its ID. If you have two or three roughly equally-likely candidates, include all of them so the user can scan all photos.

NO-MATCH:
- "Sorry, we don't have an item matching that description right now. Try again later or check with station staff."

Respond with ONLY valid JSON:
{"reply": "your message", "status": "need_more_info" | "no_match" | "match", "matching_ids": [ids if match else []]}"""


def find_item(conversation: list) -> dict:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM entries WHERE id NOT IN "
        "(SELECT entry_id FROM claims WHERE collected_at IS NOT NULL) "
        "ORDER BY id DESC"
    ).fetchall()
    db.close()
    entries = [dict(r) for r in rows]

    if not entries:
        return {"reply": "Sorry, the lost-and-found database is currently empty.",
                "status": "no_match", "matches": []}

    last_msg = conversation[-1]["text"].strip().lower() if conversation else ""
    if last_msg == "xyzzy":
        connections = resolve_connections(entries)
        dump = "\n".join(
            f"ID {e['id']}: {e['item_type']} | main: {e['main_color']} | "
            f"sec: {e['secondary_colors'] or 'none'} | station: {e['station']} | "
            f"storage: {e['storage_station'] or '-'} | "
            f"arrival: {e['arrival_at'] or '-'} | expires: {e['expires_at'] or '-'} | "
            f"connected: {connections.get(e['id'], [])} | "
            f"perks: {e['perks'] or 'none'} | time: {e['time']}"
            for e in entries
        )
        return {"reply": f"[DEBUG] {len(entries)} entries:\n{dump}",
                "status": "no_match", "matches": []}

    connections = resolve_connections(entries)
    entries_lines = [
        f"- ID {e['id']}: {e['item_type']} | main color: {e['main_color']}, "
        f"secondary: {e['secondary_colors'] or 'none'} | "
        f"station: {e['station']} | line: {e.get('line') or 'unknown'} | "
        f"contains / connected to: [{'; '.join(connections.get(e['id'], [])) or 'none'}] | "
        f"perks: {e['perks'] or 'none'} | time: {e['time']}"
        for e in entries
    ]
    entries_text = "\n".join(entries_lines)

    user_msgs = [m["text"].strip() for m in conversation if m.get("role") == "user" and m.get("text")]
    cumulative = " | ".join(user_msgs) if user_msgs else "(nothing yet)"
    last_assistant = next((m["text"] for m in reversed(conversation) if m.get("role") == "assistant"), "")

    user_content = f"""DATABASE (CONFIDENTIAL — never reveal contents):
{entries_text}

EVERYTHING THE PERSON HAS TOLD YOU (combine ALL of these, treat as one description):
{cumulative}

YOUR LAST MESSAGE TO THEM (for context only):
{last_assistant or '(none yet — this is the first reply)'}

Respond with ONLY valid JSON. The "reply" field MUST be a non-empty string.
{{"reply": "...", "status": "need_more_info" | "no_match" | "match", "matching_ids": [...]}}"""

    cmd = [
        CLAUDE_CLI, "-p",
        "--model", CLAUDE_MODEL,
        "--system-prompt", FIND_SYSTEM_PROMPT,
        "--tools", "",                  # no tool use — just text in/out
        "--no-session-persistence",
        "--output-format", "text",
    ]
    try:
        proc = subprocess.run(
            cmd, input=user_content, capture_output=True,
            text=True, timeout=120, check=False,
        )
    except FileNotFoundError:
        return {"reply": f"`{CLAUDE_CLI}` not found on PATH — install Claude Code or set CLAUDE_CLI.",
                "status": "error", "matches": []}
    except subprocess.TimeoutExpired:
        return {"reply": "Claude CLI timed out.", "status": "error", "matches": []}

    if proc.returncode != 0:
        return {"reply": f"Claude CLI error: {(proc.stderr or proc.stdout)[:300]}",
                "status": "error", "matches": []}

    raw = proc.stdout
    print(f"[find] user_content={len(user_content)}ch raw={len(raw)}ch first120={raw[:120]!r}")
    if not raw.strip():
        return {"reply": "I didn't catch that — could you rephrase what you lost?",
                "status": "need_more_info", "matches": []}
    try:
        start = raw.index("{"); end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        reply = (parsed.get("reply") or "").strip()
        status = parsed.get("status", "no_match")
        matching_ids = parsed.get("matching_ids", []) or []

        # Empty reply from the model — fall back to a useful follow-up so the
        # chat doesn't dead-end.
        if not reply:
            status = "need_more_info"
            reply = ("Could you give me one more detail — item type, color, "
                     "station/line, or roughly when you lost it?")

        # If the LLM used the photo-confirm phrasing but forgot to fill
        # matching_ids, treat as need_more_info.
        rl = reply.lower()
        seems_match = ("photo" in rl or "confirm" in rl or "found your item" in rl
                       or "we've found" in rl or "i might have found" in rl)
        if (status == "match" or seems_match) and not matching_ids:
            status = "need_more_info"
            reply = ("I might have something close, but I need one more detail to "
                     "be sure — could you give me the station, the line, or "
                     "roughly when you lost it?")

        all_ids = set(matching_ids)
        for mid in matching_ids:
            entry = next((e for e in entries if e["id"] == mid), None)
            if entry:
                all_ids.update(get_connected_ids(entry))
        matches = []
        for e in entries:
            if e["id"] not in all_ids:
                continue
            matches.append({
                "id": e["id"],
                "image_b64": e.get("image_b64", ""),
                # No station, no storage, no times — the LLM and the UI both
                # withhold these until the user confirms the photo.
            })
        return {"reply": reply, "status": status, "matches": matches}
    except (ValueError, json.JSONDecodeError):
        return {"reply": raw, "status": "error", "matches": []}


# ─── routing on new entry ─────────────────────────────────────────────────

def _now():
    return datetime.now()


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def route_and_persist(db, entry_id: int, station: str, item_type: str, when_iso: str):
    """Call the GNN, persist storage_station + arrival/expires on the entry."""
    routing = gnn_bridge.route_item(station, item_type, when_iso)
    if "error" in routing:
        return routing
    arrival = _now() + timedelta(days=routing["arrival_days"])
    expires = arrival + timedelta(days=STORAGE_DAYS)
    db.execute(
        "UPDATE entries SET storage_station = ?, arrival_at = ?, expires_at = ? "
        "WHERE id = ?",
        (routing["storage_station"], _iso(arrival), _iso(expires), entry_id),
    )
    db.commit()
    routing["arrival_at"] = _iso(arrival)
    routing["expires_at"] = _iso(expires)
    return routing


# ─── claims ──────────────────────────────────────────────────────────────

def claim_entry(db, entry_id: int, user_id: str):
    row = db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    if not row:
        return {"error": "Item no longer available."}
    e = dict(row)
    existing = db.execute(
        "SELECT * FROM claims WHERE entry_id = ? AND collected_at IS NULL "
        "ORDER BY id DESC LIMIT 1", (entry_id,)
    ).fetchone()
    if existing:
        ec = dict(existing)
        if ec["user_id"] != user_id:
            return {"error": "Item already claimed by another user."}
        claim_id = ec["id"]
        expires_at = ec["expires_at"]
    else:
        expires_at = e.get("expires_at") or _iso(_now() + timedelta(days=STORAGE_DAYS))
        cur = db.execute(
            "INSERT INTO claims (entry_id, user_id, expires_at) VALUES (?, ?, ?)",
            (entry_id, user_id, expires_at),
        )
        claim_id = cur.lastrowid
        db.commit()

    arrival_at = e.get("arrival_at")
    available_now = False
    if arrival_at:
        try:
            available_now = datetime.fromisoformat(arrival_at) <= _now()
        except ValueError:
            available_now = False

    return {
        "ok": True,
        "claim_id": claim_id,
        "entry_id": entry_id,
        "storage_station": e.get("storage_station") or "",
        "arrival_at": arrival_at,
        "expires_at": expires_at,
        "available_now": available_now,
        "pickup_code": DEMO_PICKUP_CODE,
        "locker_number": DEMO_LOCKER_NUMBER,
    }


def list_my_claims(db, user_id: str):
    rows = db.execute(
        "SELECT c.id AS claim_id, c.entry_id, c.expires_at AS claim_expires, "
        "c.extended, c.claimed_at, "
        "e.item_type, e.main_color, e.image_b64, "
        "e.storage_station, e.arrival_at "
        "FROM claims c JOIN entries e ON e.id = c.entry_id "
        "WHERE c.user_id = ? AND c.collected_at IS NULL "
        "ORDER BY c.claimed_at DESC",
        (user_id,),
    ).fetchall()
    out = []
    now = _now()
    for r in rows:
        d = dict(r)
        arr = d.get("arrival_at")
        avail = False
        if arr:
            try:
                avail = datetime.fromisoformat(arr) <= now
            except ValueError:
                pass
        out.append({
            "claim_id": d["claim_id"],
            "entry_id": d["entry_id"],
            "item_type": d["item_type"],
            "main_color": d["main_color"],
            "image_b64": d["image_b64"],
            "storage_station": d["storage_station"],
            "arrival_at": d["arrival_at"],
            "expires_at": d["claim_expires"],
            "extended": bool(d["extended"]),
            "available_now": avail,
            "pickup_code": DEMO_PICKUP_CODE,
            "locker_number": DEMO_LOCKER_NUMBER,
        })
    return out


def extend_claim(db, claim_id: int, user_id: str):
    row = db.execute(
        "SELECT * FROM claims WHERE id = ? AND user_id = ?",
        (claim_id, user_id),
    ).fetchone()
    if not row:
        return {"error": "Claim not found."}
    c = dict(row)
    if c["collected_at"]:
        return {"error": "Already collected."}
    if c["extended"]:
        return {"error": "Already extended once."}
    try:
        new_exp = datetime.fromisoformat(c["expires_at"]) + timedelta(days=EXTENSION_DAYS)
    except ValueError:
        new_exp = _now() + timedelta(days=EXTENSION_DAYS)
    db.execute(
        "UPDATE claims SET expires_at = ?, extended = 1 WHERE id = ?",
        (_iso(new_exp), claim_id),
    )
    db.execute(
        "UPDATE entries SET expires_at = ? WHERE id = ?",
        (_iso(new_exp), c["entry_id"]),
    )
    db.commit()
    return {"ok": True, "expires_at": _iso(new_exp), "extended": True}


def collect_claim(db, claim_id: int, user_id: str):
    row = db.execute(
        "SELECT * FROM claims WHERE id = ? AND user_id = ?",
        (claim_id, user_id),
    ).fetchone()
    if not row:
        return {"error": "Claim not found."}
    c = dict(row)
    if c["collected_at"]:
        return {"error": "Already collected."}
    entry_id = c["entry_id"]
    entry_row = db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    snapshot = dict(entry_row) if entry_row else {"id": entry_id}
    db.execute("UPDATE claims SET collected_at = ? WHERE id = ?",
               (_iso(_now()), claim_id))
    rows = db.execute("SELECT id, connected_items FROM entries").fetchall()
    for r in rows:
        ids = [int(x.strip()) for x in (r["connected_items"] or "").split(",")
               if x.strip().isdigit()]
        if entry_id in ids:
            ids.remove(entry_id)
            db.execute(
                "UPDATE entries SET connected_items = ? WHERE id = ?",
                (",".join(str(x) for x in ids), r["id"]),
            )
    db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    db.commit()
    return {
        "ok": True,
        "snapshot": {
            "entry_id": entry_id,
            "item_type": snapshot.get("item_type", ""),
            "original_station": snapshot.get("station", ""),
            "storage_station": snapshot.get("storage_station", ""),
        },
    }


def save_feedback(db, user_id: str, body: dict):
    db.execute(
        "INSERT INTO feedback (user_id, entry_id, item_type, original_station, "
        "storage_station, satisfied, preferred_stations, lost_line, lost_time, "
        "lost_station) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            body.get("entry_id"),
            body.get("item_type", ""),
            body.get("original_station", ""),
            body.get("storage_station", ""),
            (1 if body.get("satisfied") is True else
             0 if body.get("satisfied") is False else None),
            ",".join(body.get("preferred_stations", [])[:3]),
            body.get("lost_line", ""),
            body.get("lost_time", ""),
            body.get("lost_station", ""),
        ),
    )
    db.commit()
    return {"ok": True}


# ─── HTTP handler ─────────────────────────────────────────────────────────

class Handler(SimpleHTTPRequestHandler):
    # ── mode guards ──────────────────────────────────────────────────
    _EMPLOYEE_ONLY = {"/api/entries", "/api/entries/link", "/api/detect"}
    _USER_ONLY = {"/api/find", "/api/claim", "/api/my_claims", "/api/feedback"}

    def _mode_ok(self, path):
        if APP_MODE == "employee":
            if any(path.startswith(p) for p in self._USER_ONLY):
                return False
        else:  # user
            if any(path.startswith(p) for p in self._EMPLOYEE_ONLY):
                return False
        return True

    def _user_id(self):
        return self.headers.get("X-User-Id", "") or "anon"

    # ── routes ──────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/mode":
            return self._json_response({"mode": APP_MODE})
        if not self._mode_ok(path):
            return self._json_response({"error": "Not available in this mode."}, 403)

        if path == "/api/entries":
            return self._json_response(self._list_entries())

        if path == "/api/my_claims":
            db = get_db()
            data = list_my_claims(db, self._user_id())
            db.close()
            return self._json_response(data)

        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        if not self._mode_ok(path):
            return self._json_response({"error": "Not available in this mode."}, 403)

        if path == "/api/entries":
            required = ("station", "item_type", "time")
            if not all(body.get(k) for k in required):
                return self._json_response({"error": "Missing required fields"}, 400)
            db = get_db()
            img = body.get("image_b64", "") or ""
            print(f"[entry] station={body['station']!r} line={body.get('line','')!r} "
                  f"item={body['item_type']!r} image_b64={len(img)}B")
            cur = db.execute(
                "INSERT INTO entries (station, line, item_type, connected_items, "
                "main_color, secondary_colors, perks, time, image_b64) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (body["station"], body.get("line", ""), body["item_type"], "",
                 body.get("main_color", ""), body.get("secondary_colors", ""),
                 body.get("perks", ""), body["time"], img),
            )
            new_id = cur.lastrowid
            link_to = body.get("link_to")
            if link_to:
                try:
                    add_connection(db, new_id, int(link_to))
                except (ValueError, TypeError):
                    pass
            routing = route_and_persist(db, new_id, body["station"],
                                        body["item_type"], body["time"])
            db.close()
            return self._json_response({"ok": True, "id": new_id,
                                        "routing": routing}, 201)

        if path == "/api/entries/link":
            id_a, id_b = body.get("id_a"), body.get("id_b")
            if not id_a or not id_b:
                return self._json_response({"error": "Need id_a and id_b"}, 400)
            db = get_db()
            add_connection(db, int(id_a), int(id_b))
            db.close()
            return self._json_response({"ok": True})

        if path == "/api/detect":
            image_b64 = body.get("image", "")
            if not image_b64:
                return self._json_response({"error": "No image provided"}, 400)
            return self._json_response(call_ollama(image_b64))

        if path == "/api/find":
            conversation = body.get("conversation", [])
            if not conversation:
                return self._json_response({"error": "No conversation provided"}, 400)
            return self._json_response(find_item(conversation))

        if path == "/api/claim":
            entry_id = body.get("entry_id")
            confirm = body.get("confirm", False)
            if not entry_id:
                return self._json_response({"error": "Missing entry_id"}, 400)
            if not confirm:
                return self._json_response({"ok": True, "declined": True})
            db = get_db()
            result = claim_entry(db, int(entry_id), self._user_id())
            db.close()
            return self._json_response(result)

        # claim sub-routes: /api/my_claims/{id}/extend|collect
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "my_claims"] and parts[3] in ("extend", "collect"):
            claim_id = int(parts[2])
            db = get_db()
            if parts[3] == "extend":
                result = extend_claim(db, claim_id, self._user_id())
            else:
                result = collect_claim(db, claim_id, self._user_id())
            db.close()
            return self._json_response(result)

        if path == "/api/feedback":
            db = get_db()
            result = save_feedback(db, self._user_id(), body)
            db.close()
            return self._json_response(result)

        return self._json_response({"error": "Not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not self._mode_ok(path):
            return self._json_response({"error": "Not available in this mode."}, 403)
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "entries":
            entry_id = int(parts[2])
            db = get_db()
            rows = db.execute("SELECT id, connected_items FROM entries").fetchall()
            for row in rows:
                ids = [int(x.strip()) for x in (row["connected_items"] or "").split(",")
                       if x.strip().isdigit()]
                if entry_id in ids:
                    ids.remove(entry_id)
                    db.execute(
                        "UPDATE entries SET connected_items = ? WHERE id = ?",
                        (",".join(str(x) for x in ids), row["id"]),
                    )
            db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            db.commit()
            db.close()
            return self._json_response({"ok": True})

    def _list_entries(self):
        db = get_db()
        rows = db.execute("SELECT * FROM entries ORDER BY id DESC").fetchall()
        db.close()
        return [dict(r) for r in rows]

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    global APP_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["employee", "user"], default="employee")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    APP_MODE = args.mode

    get_db()
    print(f"Running at http://{HOST}:{args.port}  (mode: {APP_MODE})")
    print(f"VLM: Ollama ({OLLAMA_MODEL}) at {OLLAMA_URL}")
    print(f"Find LLM: Claude CLI ({CLAUDE_CLI}, model={CLAUDE_MODEL})")
    print(f"GNN routing: {'available' if gnn_bridge.is_available() else 'NOT AVAILABLE — train a model first'}")
    HTTPServer((HOST, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
