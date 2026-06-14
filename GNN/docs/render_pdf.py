"""Render all .md files in this directory to PDF using markdown + weasyprint."""
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

css = """
@page { size: A4; margin: 18mm 18mm 20mm 18mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 18pt; margin: 0 0 6pt 0; color: #111; border-bottom: 2px solid #444; padding-bottom: 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 4pt 0; color: #1a3d6b; border-bottom: 1px solid #cfd9e6; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt 0; color: #1a3d6b; }
p  { margin: 4pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 4pt 16pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #c8c8c8; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfd; }
code { background: #f4f4f4; padding: 1pt 3pt; border-radius: 2pt; font-size: 9.5pt; }
pre { background: #f4f4f4; padding: 6pt; border-radius: 3pt; font-size: 9pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
strong { color: #111; }
a { color: #1a3d6b; text-decoration: none; }
blockquote { border-left: 3px solid #cfd9e6; margin: 6pt 0; padding: 2pt 8pt; color: #444; }
"""

here = Path(__file__).parent
for src in sorted(here.glob("*.md")):
    out = src.with_suffix(".pdf")
    md_text = src.read_text()
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{src.stem}</title></head>
<body>{html_body}</body></html>"""
    HTML(string=html, base_url=str(here)).write_pdf(out, stylesheets=[CSS(string=css)])
    print(f"Wrote {out.name}  ({out.stat().st_size/1024:.1f} KB)")
