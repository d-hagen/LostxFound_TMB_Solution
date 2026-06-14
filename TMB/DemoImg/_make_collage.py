#!/usr/bin/env python3
"""Build a coherent overview collage of the TMB app: Staff side vs Passenger side."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "demo_overview.png"

IMGS = {
    "addItem":  "Screenshot 2026-06-13 at 19.14.34.png",
    "gnn":      "Screenshot 2026-06-13 at 19.28.24.png",
    "db":       "Screenshot 2026-06-13 at 19.17.36.png",
    "find1":    "Screenshot 2026-06-13 at 19.16.13.png",
    "find2":    "Screenshot 2026-06-13 at 19.19.47.png",
    "claimed":  "Screenshot 2026-06-13 at 19.16.22.png",
    "feedback": "Screenshot 2026-06-13 at 19.20.41.png",
}
def load(k): return Image.open(HERE / IMGS[k]).convert("RGB")

# ---- palette ----
BG        = (236, 241, 239)
CARD      = (255, 255, 255)
CARD_BD   = (223, 228, 226)
TILE_BD   = (228, 231, 230)
STAFF     = (217, 56, 57)
PASS      = (52, 150, 138)
TITLE_TXT = (34, 41, 39)
SUB_TXT   = (108, 118, 115)
CAP_TXT   = (74, 84, 81)

# ---- fonts ----
AR  = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def F(p, s): return ImageFont.truetype(p, s)
f_title = F(ARB, 92); f_sub = F(AR, 40)
f_hdr = F(ARB, 58);   f_hsub = F(AR, 33)
f_cap = F(AR, 31);    f_badge = F(ARB, 34)

# ---- geometry ----
MARGIN, GAP = 100, 80
CARD_R, TILE_R = 40, 22
CARD_PAD, INNER_Y = 70, 60
HEADER_H = 152
TILE_PAD, ROW_GAP, COL_GAP = 26, 48, 40
CAP_GAP, CAP_H, CAP_LH = 18, 84, 40
BADGE_R = 30

CONTENT_W = 1400                      # inner content width of each card
WP = (CONTENT_W - COL_GAP) // 2       # half-width tile

def wrap(draw, text, font, maxw, maxlines=2):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
            if len(lines) == maxlines: break
    if cur and len(lines) < maxlines: lines.append(cur)
    return lines[:maxlines]

_scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))

def make_tile(key, tile_w, caption, num, accent):
    img = load(key)
    iw = tile_w - 2 * TILE_PAD
    ih = round(iw * img.height / img.width)
    img = img.resize((iw, ih), Image.LANCZOS)
    tile_h = TILE_PAD + ih + CAP_GAP + CAP_H + TILE_PAD
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle([0, 0, tile_w - 1, tile_h - 1], TILE_R, fill=CARD, outline=TILE_BD, width=2)
    m = Image.new("L", (iw, ih), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, iw - 1, ih - 1], 14, fill=255)
    tile.paste(img, (TILE_PAD, TILE_PAD), m)
    d.rounded_rectangle([TILE_PAD, TILE_PAD, TILE_PAD + iw - 1, TILE_PAD + ih - 1], 14,
                        outline=(214, 218, 217), width=2)
    # caption strip: numbered badge (left) + caption text — kept off the screenshot
    cap_y = TILE_PAD + ih + CAP_GAP
    bcx, bcy = TILE_PAD + BADGE_R, cap_y + CAP_H // 2
    d.ellipse([bcx - BADGE_R, bcy - BADGE_R, bcx + BADGE_R, bcy + BADGE_R], fill=accent)
    n = str(num)
    nb = d.textbbox((0, 0), n, font=f_badge)
    d.text((bcx - (nb[2] - nb[0]) / 2 - nb[0], bcy - (nb[3] - nb[1]) / 2 - nb[1]), n,
           font=f_badge, fill=(255, 255, 255))
    text_x = TILE_PAD + 2 * BADGE_R + 20
    lines = wrap(_scratch, caption, f_cap, tile_w - TILE_PAD - text_x)
    ty = cap_y + (CAP_H - len(lines) * CAP_LH) // 2
    for ln in lines:
        d.text((text_x, ty), ln, font=f_cap, fill=CAP_TXT); ty += CAP_LH
    return tile

def shadow(canvas, box, radius, blur=26, alpha=58, off=(0, 14)):
    x0, y0, x1, y1 = box
    pad = blur * 3
    sh = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([pad, pad, pad + (x1 - x0), pad + (y1 - y0)],
                                         radius, fill=(20, 30, 28, alpha))
    canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)), (x0 - pad + off[0], y0 - pad + off[1]))

# ---- build tiles, grouped into rows ----
staff_rows = [
    [make_tile("addItem", WP, "Snap an item — the vision model auto-fills the details", 1, STAFF),
     make_tile("gnn",     WP, "A GNN reroutes it to the best storage station", 2, STAFF)],
    [make_tile("db", CONTENT_W, "Routed database — arrival, expiry, locker & pickup code", 3, STAFF)],
]
pass_rows = [
    [make_tile("find1",   WP, "Describe what you lost in plain language", 1, PASS),
     make_tile("find2",   WP, "AI confirms the match — reveals locker & pickup code", 2, PASS)],
    [make_tile("claimed", WP, "Track the claim — extend or mark collected", 3, PASS),
     make_tile("feedback",WP, "Optional feedback tunes the routing model", 4, PASS)],
]

def block_size(rows):
    h = sum(max(t.height for t in r) for r in rows) + ROW_GAP * (len(rows) - 1)
    return h

staff_h, pass_h = block_size(staff_rows), block_size(pass_rows)
content_h = max(staff_h, pass_h)
card_w = CONTENT_W + 2 * CARD_PAD
card_h = HEADER_H + 2 * INNER_Y + content_h

TITLE_H, GAP_TC = 300, 54
canvas_w = MARGIN + card_w + GAP + card_w + MARGIN
canvas_h = MARGIN + TITLE_H + GAP_TC + card_h + MARGIN
cv = Image.new("RGBA", (canvas_w, canvas_h), BG + (255,))
draw = ImageDraw.Draw(cv)

# ---- title band ----
tb = [MARGIN, MARGIN, canvas_w - MARGIN, MARGIN + TITLE_H]
shadow(cv, tb, CARD_R)
draw.rounded_rectangle(tb, CARD_R, fill=CARD, outline=CARD_BD, width=2)
ax = MARGIN + 60
draw.rounded_rectangle([ax, MARGIN + 70, ax + 26, MARGIN + TITLE_H - 70], 13, fill=STAFF)
draw.rounded_rectangle([ax + 38, MARGIN + 70, ax + 64, MARGIN + TITLE_H - 70], 13, fill=PASS)
tx = ax + 110
draw.text((tx, MARGIN + 78), "TMB Lost & Found", font=f_title, fill=TITLE_TXT)
draw.text((tx + 4, MARGIN + 188),
          "AI-powered lost & found for public transit — staff catalog items, passengers retrieve them.",
          font=f_sub, fill=SUB_TXT)

def draw_card(x, title, sub, accent, rows, block_h):
    box = [x, MARGIN + TITLE_H + GAP_TC, x + card_w, MARGIN + TITLE_H + GAP_TC + card_h]
    shadow(cv, box, CARD_R)
    draw.rounded_rectangle(box, CARD_R, fill=CARD, outline=CARD_BD, width=2)
    draw.rounded_rectangle([box[0], box[1], box[2], box[1] + HEADER_H], CARD_R, fill=accent)
    draw.rectangle([box[0], box[1] + CARD_R, box[2], box[1] + HEADER_H], fill=accent)
    draw.text((box[0] + CARD_PAD, box[1] + 30), title, font=f_hdr, fill=(255, 255, 255))
    draw.text((box[0] + CARD_PAD + 4, box[1] + 100), sub, font=f_hsub, fill=(255, 255, 255))
    cx0 = box[0] + CARD_PAD
    y = box[1] + HEADER_H + INNER_Y + (content_h - block_h) // 2
    for row in rows:
        rh = max(t.height for t in row)
        row_w = sum(t.width for t in row) + COL_GAP * (len(row) - 1)
        rx = cx0 + (CONTENT_W - row_w) // 2
        for t in row:
            cv.alpha_composite(t, (rx, y + (rh - t.height) // 2)); rx += t.width + COL_GAP
        y += rh + ROW_GAP

draw_card(MARGIN, "STAFF", "Catalog found items · GNN storage routing", STAFF, staff_rows, staff_h)
draw_card(MARGIN + card_w + GAP, "PASSENGER", "Find · claim · collect · feedback", PASS, pass_rows, pass_h)

final = cv.convert("RGB")
target_w = 2400
final = final.resize((target_w, round(canvas_h * target_w / canvas_w)), Image.LANCZOS)
final.save(OUT, "PNG")
print("saved", OUT, final.size)
