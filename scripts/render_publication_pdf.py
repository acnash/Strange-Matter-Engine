#!/usr/bin/env python3
"""Convert publication Markdown to a print-ready PDF through headless Edge."""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "publication" / "publication.md"
TEMP = ROOT / "tmp" / "pdfs"
OUTPUT = ROOT / "output" / "pdf" / "molecule_space_time_publication.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


CSS = r"""
@page { size: A4; margin: 18mm 17mm 18mm 19mm; }
* { box-sizing: border-box; }
html { font-family: "Segoe UI", Arial, sans-serif; color: #17202a; }
body { margin: 0; font-size: 10.2pt; line-height: 1.48; }
h1 { font-size: 25pt; line-height: 1.13; margin: 0 0 8mm; color: #0d2638; }
h2 { font-size: 17pt; margin: 9mm 0 3.5mm; color: #0d4f68; border-bottom: 1px solid #b8d6df; padding-bottom: 1.5mm; break-after: avoid; }
h3 { font-size: 13.2pt; margin: 6mm 0 2.5mm; color: #17657c; break-after: avoid; }
h4 { font-size: 11.3pt; margin: 4.5mm 0 2mm; color: #263f4c; break-after: avoid; }
p { margin: 0 0 3.2mm; orphans: 3; widows: 3; }
strong { color: #102f40; }
a { color: #006d8f; text-decoration: none; }
img { display: block; max-width: 100%; max-height: 205mm; width: auto; height: auto; margin: 4mm auto 2.5mm; break-inside: avoid; }
p:has(> img) { break-after: avoid; margin-bottom: 1.5mm; }
p:has(> img) + p { break-before: avoid; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; font-size: 8.7pt; break-inside: avoid; }
thead { display: table-header-group; background: #e8f3f6; }
th, td { border: 0.5pt solid #aebfc7; padding: 1.7mm 2mm; vertical-align: top; }
th { color: #123849; font-weight: 650; }
tr { break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.7pt; background: #f1f4f5; padding: 0.2mm 0.7mm; border-radius: 1mm; }
pre { white-space: pre-wrap; background: #f1f4f5; padding: 3mm; border-left: 2mm solid #38a8c4; break-inside: avoid; }
blockquote { margin: 4mm 7mm; padding-left: 4mm; border-left: 1.5mm solid #55a9bd; color: #334b56; }
ul, ol { margin: 2mm 0 4mm 7mm; padding-left: 5mm; }
li { margin-bottom: 1mm; }
.author { font-size: 12pt; color: #536a75; margin-top: -5mm; margin-bottom: 12mm; }
.MathJax { font-size: 96% !important; }
mjx-container[display="true"] { margin: 4mm 0 !important; break-inside: avoid; overflow: visible; }
"""


def localize_images(html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if re.match(r"^[a-zA-Z]+:", value):
            return match.group(0)
        path = (SOURCE.parent / value).resolve()
        return f'src="{path.as_uri()}"'
    return re.sub(r'src="([^"]+)"', replace, html)


def main() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html_body = localize_images(html_body)
    html_body = re.sub(r"(<h1>.*?</h1>)\s*<p><strong>(.*?)</strong></p>",
                       r"\1<p class=\"author\">\2</p>", html_body, count=1,
                       flags=re.DOTALL)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Molecule Space-Time</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true}},
  svg: {{fontCache: 'global'}}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head><body>{html_body}</body></html>"""
    html_path = TEMP / "publication_print.html"
    html_path.write_text(document, encoding="utf-8")
    profile = TEMP / f"edge-profile-{uuid.uuid4().hex}"
    profile.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    started = time.time()
    command = [
        str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-data-dir={profile.resolve()}",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=20000",
        f"--print-to-pdf={OUTPUT.resolve()}", html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    deadline = time.time() + 10.0
    while not OUTPUT.exists() and time.time() < deadline:
        time.sleep(0.2)
    if (completed.returncode or not OUTPUT.exists()
            or OUTPUT.stat().st_mtime < started):
        raise RuntimeError(
            f"Edge failed to create the PDF (exit={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r})"
        )
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
