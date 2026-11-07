#!/usr/bin/env python3
"""Render book.md to PDF using the same KaTeX HTML as scripts/build_pages.py.

Usage:
    python3 scripts/build_pdf.py
    python3 scripts/build_pdf.py --out book.pdf --format Letter

Requires Playwright in the Python you run this with (not Homebrew python3):

    python3 -m venv .venv
    .venv/bin/pip install playwright
    .venv/bin/playwright install chromium
    .venv/bin/python scripts/build_pdf.py
"""

import argparse
import functools
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def _serve(docs: Path) -> tuple[ThreadingHTTPServer, str]:
    handler = functools.partial(_QuietHandler, directory=str(docs))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address[:2]
    return httpd, f"http://{host}:{port}/"


def _build_pages() -> None:
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "build_pages.py")],
        cwd=str(ROOT),
    )


PRINT_CSS = """
.site-banner, #status { display: none !important; }
html, body {
  padding: 0 !important;
  max-width: none !important;
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
}
#content.markdown-body {
  max-width: 980px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}
"""


def _render_pdf(url: str, out: Path, page_format: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright is not installed for this Python.\n"
            "  python3 -m pip install playwright\n"
            "  python3 -m playwright install chromium"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 1600})
        page.set_default_timeout(180_000)
        # Screen CSS, not @media print: GitHub markdown print styles drop boxes.
        page.emulate_media(media="screen")
        page.goto(url, wait_until="load")
        page.wait_for_selector("#content.markdown-body:not([hidden])")
        page.wait_for_selector("#content[data-pdf-ready='1']")
        page.wait_for_function("() => document.querySelectorAll('.katex').length > 0")
        page.wait_for_function(
            "() => [...document.images].every(img => img.complete)"
        )
        page.add_style_tag(content=PRINT_CSS)
        out.parent.mkdir(parents=True, exist_ok=True)
        page.pdf(
            path=str(out),
            format=page_format,
            print_background=True,
            prefer_css_page_size=False,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="font-size:9px;width:100%;text-align:center;'
                'color:#57606a;font-family:-apple-system,Helvetica,sans-serif;">'
                '<span class="pageNumber"></span></div>'
            ),
            margin={
                "top": "0.7in",
                "bottom": "0.8in",
                "left": "0.75in",
                "right": "0.75in",
            },
        )
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "book.pdf",
        help="output PDF path (default: book.pdf)",
    )
    parser.add_argument(
        "--format",
        default="Letter",
        choices=("Letter", "A4"),
        help="page size",
    )
    args = parser.parse_args()

    _build_pages()
    httpd, url = _serve(DOCS)
    try:
        print(f"Rendering {url} → {args.out}")
        _render_pdf(url, args.out, args.format)
    finally:
        httpd.shutdown()
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
