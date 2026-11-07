#!/usr/bin/env python3
"""Build a KaTeX GitHub Pages site from book.md (Cursor-like rendering).

Native github.com file preview uses MathJax and ignores custom CSS, so formulas
and algorithm boxes never match Cursor. This copies book.md + CSS + figures into
docs/ and serves an HTML shell that renders with KaTeX (same engine Cursor uses)
and github-markdown-light.css (.impl-note / .aside-note boxes).

Usage:
  python3 scripts/build_pages.py

Then: repo Settings → Pages → Deploy from branch → /docs on main
Site: https://<user>.github.io/<repo>/
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reinforcement Learning: A Compact Mathematical and Python Guide</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.8.1/github-markdown-light.min.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css" />
  <link rel="stylesheet" href="github-markdown-light.css" />
  <style>
    html, body { margin: 0; padding: 0; background: #fff; }
    .site-banner {
      max-width: 980px; margin: 0 auto; padding: 12px 24px 0;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      color: #57606a;
    }
    .site-banner a { color: #0969da; }
    #content.markdown-body {
      box-sizing: border-box; min-width: 200px; max-width: 980px;
      margin: 0 auto; padding: 24px 24px 64px;
    }
    #status { max-width: 980px; margin: 24px auto; padding: 0 24px; color: #57606a; }
    /* Keep Cursor/site look when printing or writing PDF (Chromium @media print
       otherwise drops box backgrounds and clips pre/katex overflow). */
    html, body {
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }
    pre, pre code {
      overflow: visible !important;
      white-space: pre-wrap !important;
      word-break: break-word !important;
    }
    .katex-display {
      overflow: visible !important;
      max-width: 100% !important;
    }
    .katex-display > .katex {
      max-width: 100%;
      white-space: normal !important;
    }
    img { max-width: 100% !important; height: auto !important; }
    .book-figure img { max-height: 4.2in; }
    table { font-size: 0.95em; }
    #content.markdown-body > h1 { break-before: page; }
    #content.markdown-body > h1:first-of-type { break-before: auto; }
    .book-figure, .impl-note, .aside-note, pre, table, .katex-display {
      break-inside: avoid;
    }
    @media print {
      .site-banner, #status { display: none !important; }
      html, body { padding: 0 !important; max-width: none !important; }
      #content.markdown-body { padding: 0 !important; margin: 0 auto !important; }
    }
  </style>
</head>
<body>
  <p class="site-banner">
    KaTeX site (Cursor-like math + algorithm boxes) ·
    <a href="https://github.com/dipeshgautam2012/reinforcement_learning/blob/main/book.md">book.md source</a>
  </p>
  <p id="status">Loading book…</p>
  <article id="content" class="markdown-body" hidden></article>

  <script src="https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"></script>
  <script>
    function protectMath(src) {
      const slots = [];
      const stash = (m) => {
        slots.push(m);
        return "%%MATH" + (slots.length - 1) + "%%";
      };
      // Display first, then inline (avoid eating $$ insides)
      let s = src.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (m) => stash(m));
      s = s.replace(/\\$([^\\n$]+?)\\$/g, (m) => stash(m));
      return { text: s, slots };
    }

    function restoreMath(html, slots) {
      return html.replace(/%%MATH(\\d+)%%/g, (_, i) => slots[Number(i)]);
    }

    async function main() {
      const status = document.getElementById("status");
      const content = document.getElementById("content");
      try {
        const res = await fetch("book.md", { cache: "no-cache" });
        if (!res.ok) throw new Error("Failed to fetch book.md (" + res.status + ")");
        let md = await res.text();
        // Remove GitHub-file-preview banner if present
        md = md.replace(/^> \\*\\*GitHub readers:\\*\\*[\\s\\S]*?\\n(?=> |\\n)/m, "");

        const { text, slots } = protectMath(md);
        const mdit = window.markdownit({
          html: true,
          linkify: true,
          breaks: false,
          typographer: false,
          highlight: function (str, lang) {
            if (!window.hljs) return "";
            try {
              if (lang && hljs.getLanguage(lang))
                return hljs.highlight(str, { language: lang }).value;
              return hljs.highlightAuto(str).value;
            } catch (e) {
              return "";
            }
          },
        });
        let html = mdit.render(text);
        html = restoreMath(html, slots);
        content.innerHTML = html;
        content.hidden = false;
        status.hidden = true;

        renderMathInElement(content, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ],
          throwOnError: false,
          ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        });
        try { await document.fonts.ready; } catch (e) {}
        content.setAttribute("data-pdf-ready", "1");
      } catch (err) {
        status.textContent = String(err);
      }
    }

    document.addEventListener("DOMContentLoaded", () => {
      // Wait for KaTeX auto-render to be available
      const wait = () => {
        if (window.renderMathInElement && window.markdownit) main();
        else setTimeout(wait, 30);
      };
      wait();
    });
  </script>
</body>
</html>
"""


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True)

    # Strip GitHub-readers banner from the published markdown copy
    book = (ROOT / "book.md").read_text()
    book = re.sub(
        r"^> \*\*GitHub readers:\*\*.*\n(?=> )?",
        "",
        book,
        count=1,
        flags=re.M,
    )
    (DOCS / "book.md").write_text(book)
    (DOCS / "index.html").write_text(INDEX_HTML)
    shutil.copy2(ROOT / "github-markdown-light.css", DOCS / "github-markdown-light.css")
    if (ROOT / "figures").exists():
        shutil.copytree(ROOT / "figures", DOCS / "figures")
    (DOCS / ".nojekyll").write_text("")

    print(f"Wrote {DOCS}/")
    print("Next: Settings → Pages → Deploy from branch → /docs (main)")
    print("URL:  https://dipeshgautam2012.github.io/reinforcement_learning/")


if __name__ == "__main__":
    main()
