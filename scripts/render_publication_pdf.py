#!/usr/bin/env python3
"""Convert publication Markdown to a preview-safe, print-ready PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "publication" / "publication.md"
TEMP = ROOT / "tmp" / "pdfs"
PRINT_OUTPUT = ROOT / "output" / "pdf" / "molecule_space_time_publication.pdf"
OUTPUT = ROOT / "output" / "pdf" / "molecule_space_time_publication_preview.pdf"
VECTOR_OUTPUT = PRINT_OUTPUT

BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)


CSS = r"""
@page { size: A4; margin: 18mm 17mm 18mm 19mm; }
* { box-sizing: border-box; }
html { font-family: "STIX Two Text", "Times New Roman", serif; color: #17202a; }
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
math[display="block"] { display: block; margin: 4mm auto; text-align: center; font-size: 96%; break-inside: avoid; }
p:has(> math[display="block"]) { break-inside: avoid; }
p:has(> math[display="block"]) + p:has(> math[display="block"]) { break-before: avoid; }
#appendix-a-mathematical-definition-of-the-graph-cellular-automaton { break-before: page; }
#references { break-before: page; }
"""


def localize_images(html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if re.match(r"^[a-zA-Z]+:", value):
            return match.group(0)
        path = (SOURCE.parent / value).resolve()
        return f'src="{path.as_uri()}"'
    return re.sub(r'src="([^"]+)"', replace, html)


def find_browser() -> Path:
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    raise RuntimeError("No supported Chrome or Edge executable was found")


def render_markdown() -> str:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Pandoc is required to typeset the manuscript mathematics")
    completed = subprocess.run(
        [pandoc, str(SOURCE), "-f", "gfm", "-t", "html5", "--mathml"],
        cwd=SOURCE.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return localize_images(completed.stdout)


def create_preview_safe_pdf(source: Path, destination: Path) -> None:
    """Rebuild each rendered page in a minimal PDF 1.4 container.

    The Codex inline previewer has been unreliable with tagged PDFs and
    browser-specific object structures.  A page-image PDF avoids those
    structures while preserving the appearance of the print rendering.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required to create the preview-safe PDF")

    with tempfile.TemporaryDirectory(prefix="publication-pages-", dir=TEMP) as folder:
        prefix = Path(folder) / "page"
        subprocess.run(
            [
                pdftoppm,
                "-jpeg",
                "-r", "85",
                "-jpegopt", "quality=50,optimize=y,progressive=n",
                str(source),
                str(prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pages = sorted(
            Path(folder).glob("page-*.jpg"),
            key=lambda path: int(path.stem.rsplit("-", 1)[1]),
        )
        if not pages:
            raise RuntimeError("The vector PDF produced no rendered pages")

        if destination.exists():
            destination.unlink()
        width, height = A4
        pdf = canvas.Canvas(
            str(destination),
            pagesize=A4,
            pageCompression=1,
            pdfVersion=(1, 4),
        )
        pdf.setTitle("Molecule Space-Time")
        pdf.setAuthor("Anthony Nash")
        pdf.setCreator("Strange Matter Engine preview-safe renderer")
        for page in pages:
            pdf.drawImage(
                str(page), 0, 0,
                width=width, height=height,
                preserveAspectRatio=False,
                mask="auto",
            )
            pdf.showPage()
        pdf.save()


def main() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html_body = render_markdown()
    html_body = re.sub(r"(<h1>.*?</h1>)\s*<p><strong>(.*?)</strong></p>",
                       r"\1<p class=\"author\">\2</p>", html_body, count=1,
                       flags=re.DOTALL)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Molecule Space-Time</title>
<style>{CSS}</style>
</head><body>{html_body}</body></html>"""
    html_path = TEMP / "publication_print.html"
    html_path.write_text(document, encoding="utf-8")
    browser = find_browser()
    if VECTOR_OUTPUT.exists():
        VECTOR_OUTPUT.unlink()
    with tempfile.TemporaryDirectory(prefix="chrome-profile-", dir=TEMP) as profile:
        started = time.time()
        command = [
            str(browser), "--headless=new", "--disable-gpu",
            "--disable-background-networking", "--disable-component-update",
            "--no-first-run", "--no-pdf-header-footer",
            f"--user-data-dir={Path(profile).resolve()}",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={VECTOR_OUTPUT.resolve()}",
            html_path.resolve().as_uri(),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 60.0
        previous_size = -1
        stable_checks = 0
        while time.time() < deadline:
            if VECTOR_OUTPUT.exists():
                current_size = VECTOR_OUTPUT.stat().st_size
                if current_size > 0 and current_size == previous_size:
                    stable_checks += 1
                else:
                    stable_checks = 0
                previous_size = current_size
                if stable_checks >= 5:
                    break
            if process.poll() is not None and not VECTOR_OUTPUT.exists():
                break
            time.sleep(0.2)

        if process.poll() is None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        else:
            stdout, stderr = process.communicate()

        if (not VECTOR_OUTPUT.exists()
                or VECTOR_OUTPUT.stat().st_mtime < started
                or VECTOR_OUTPUT.stat().st_size == 0):
            raise RuntimeError(
                f"The browser failed to create the vector PDF "
                f"(exit={process.returncode}, stdout={stdout!r}, "
                f"stderr={stderr!r})"
            )

    create_preview_safe_pdf(VECTOR_OUTPUT, OUTPUT)
    print(f"Print PDF: {PRINT_OUTPUT.resolve()}")
    print(f"Preview PDF: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
