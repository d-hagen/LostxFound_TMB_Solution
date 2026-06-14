# TMB Lost & Found — Demo

AI-powered lost item tracker for public transport. Staff photograph found items, a vision model auto-describes them, and passengers search via an LLM chatbot.

## Requirements

- **Python 3.10+** (uses only stdlib — no pip install needed)
- **Ollama** — local LLM runtime ([ollama.com](https://ollama.com))
  - `llava` model (vision, ~4.7 GB)
  - `llama3.2` model (text, ~2.0 GB)
- ~8 GB free RAM for both models
- macOS, Linux, or Windows (with WSL)

## Setup

```bash
# 1. Install Ollama from https://ollama.com

# 2. Pull the two models (one-time download)
ollama pull llava
ollama pull llama3.2

# 3. Start Ollama
ollama serve

# 4. In a second terminal, start the app
python3 app.py                    # default: --mode employee
python3 app.py --mode user        # passenger-facing
```

Open **http://localhost:8080** in your browser.

> If port 8080 is busy: `lsof -ti:8080 | xargs kill`

## How to Use

Two modes share the same server and database:

- **Employee** (default): *Add Item* + *Database* tabs. Each saved item is auto-routed by the GNN to a storage station, with arrival ETA, expiry, locker and pickup code persisted.
- **User**: *Find* (LLM chat) + *Claimed items* tabs. After confirming a match in chat, the ticket reveals locker #8 + code 2801; the user can extend once (+3 days) or mark collected (which frees the locker and deletes the entry).

The GNN routing requires a trained model in `../GNN/artifacts/models/`. If it's missing, items are still saved but no storage station is assigned.

### Add Item
1. Tap **Take Photo** (mobile) or **Upload Photo** to capture a found item.
2. The vision model analyzes the image and auto-fills: item type, main color, secondary colors, and distinguishing features.
3. Select the station and adjust any fields if needed.
4. Hit **Save**.
5. To link related items (e.g. a bag and its contents), use the **Connected Items** workflow — each item gets linked bidirectionally.

### Find (LLM Search)
1. A passenger (or staff member) describes what they lost in the chat.
2. The LLM checks the description against the database.
3. It will only confirm a match if the passenger provides enough detail (item type + color + station or date). It never reveals what's stored.
4. Debug: type `xyzzy` to dump the full database.

### Database
- Browse all stored entries.
- Delete items inline.

## Test Images

The `TestData/` folder contains sample images you can upload to try the detection.

## Project Structure

```
TMB/
  app.py         — Python HTTP server + SQLite + Ollama integration
  index.html     — Single-page frontend (HTML/CSS/JS, TMB-branded)
  entries.db     — SQLite database (auto-created on first run)
  TestData/      — Sample images for testing
```

## Notes

- Zero external Python dependencies — everything uses the standard library.
- The database file (`entries.db`) is auto-created on first run. Delete it to reset.
- Both LLM models run locally via Ollama — no API keys, no cloud, fully offline.
