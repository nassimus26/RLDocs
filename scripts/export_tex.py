#!/usr/bin/env python3
"""Convert book.md into a pdfLaTeX project under tex/.

Usage:
    python3 scripts/export_tex.py
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tex"


MAIN_TEX = r"""% pdfLaTeX project (compile from this directory with latexmk -pdf).
\documentclass[11pt,oneside,openany]{book}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{float}
\usepackage{fvextra}
\usepackage{tcolorbox}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\graphicspath{{figures/}}
\setlist[itemize]{leftmargin=1.4em,itemsep=0.25em,topsep=0.4em}
\setlist[enumerate]{leftmargin=1.6em,itemsep=0.25em,topsep=0.4em}
\emergencystretch=3em
\sloppy
\fvset{breaklines=true,breakanywhere=true,fontsize=\small,baselinestretch=1.05}

\newtcolorbox{implnote}[1][Note]{
  colback=green!4,colframe=green!45!black,fonttitle=\bfseries,title={#1},
  boxrule=0.6pt,left=6pt,right=6pt,top=4pt,bottom=4pt
}
\newtcolorbox{asidenote}[1][Aside]{
  colback=blue!3,colframe=blue!45!black,fonttitle=\bfseries,title={#1},
  boxrule=0.6pt,left=6pt,right=6pt,top=4pt,bottom=4pt
}

\title{Reinforcement Learning:\\A Compact Mathematical and Python Guide}
\author{}
\date{}

\begin{document}
\frontmatter
\maketitle
\tableofcontents
\mainmatter
"""

MAIN_END = r"""
\appendix
\input{appendix}
\backmatter
\input{references}
\end{document}
"""


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def _ph_store(chunks: list[str], latex: str, prefix: str = "XXPH") -> str:
    chunks.append(latex)
    return f"{prefix}{len(chunks) - 1}XX"


def _ph_restore(text: str, chunks: list[str], prefix: str = "XXPH") -> str:
    def repl(m: re.Match[str]) -> str:
        i = int(m.group(1))
        if i >= len(chunks):
            return m.group(0)
        return chunks[i]

    return re.sub(rf"{prefix}(\d+)XX", repl, text)


def escape_text(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("&", r"\&")
    s = s.replace("_", r"\_")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def tidy_unicode(s: str) -> str:
    repl = {
        "–": "--",
        "—": "---",
        "−": "-",
        "‘": "`",
        "’": "'",
        "“": "``",
        "”": "''",
        "×": r"$\times$",
        "≤": r"$\le$",
        "≥": r"$\ge$",
        "≠": r"$\ne$",
        "…": "...",
        "·": r"$\cdot$",
        "\u00a0": " ",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def display_math_to_tex(body: str) -> str:
    body = body.strip("\n")
    body = body.strip()
    if r"\tag{" in body:
        return "\\begin{equation}\n" + body + "\n\\end{equation}"
    return "\\[\n" + body + "\n\\]"


def _ascii_code(body: str) -> str:
    repl = {
        "×": "x",
        "←": "<-",
        "→": "->",
        "γ": "gamma",
        "π": "pi",
        "θ": "theta",
        "φ": "phi",
        "ϕ": "phi",
        "ε": "eps",
        "Σ": "sum",
        "ĥ": "Qhat",
        "—": "--",
        "–": "-",
        "…": "...",
        "·": "*",
        "ẋ": "xdot",
        "\u0307": "",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "̂": "",
    }
    for a, b in repl.items():
        body = body.replace(a, b)
    return body.encode("ascii", "replace").decode("ascii")


def code_to_tex(lang: str, body: str) -> str:
    body = _ascii_code(body.replace("\t", "    "))
    if not body.endswith("\n"):
        body += "\n"
    return (
        "\\begin{lstlisting}\n"
        + body
        + "\\end{lstlisting}"
    )


def figure_to_tex(block: str) -> str:
    src_m = re.search(r'src="([^"]+)"', block)
    alt_m = re.search(r'alt="([^"]*)"', block)
    width_m = re.search(r'width="([^"]+)"', block)
    id_m = re.search(r'id="([^"]+)"', block)
    cap_m = re.search(r"\*\*Figure[^*]*\*\*\.?\s*(.*)", block)
    if not src_m:
        return ""
    path = Path(src_m.group(1)).name
    alt = alt_m.group(1) if alt_m else ""
    cap = cap_m.group(1).strip() if cap_m else alt
    cap = re.sub(r"</?[^>]+>", "", cap)
    w = "0.72\\textwidth"
    if width_m:
        raw = width_m.group(1).strip()
        if raw.endswith("%"):
            try:
                w = f"{min(0.95, float(raw[:-1]) / 100):.2f}\\textwidth"
            except ValueError:
                pass
        else:
            try:
                px = float(raw)
                w = f"{min(0.95, px / 780):.2f}\\textwidth"
            except ValueError:
                pass
    label = ""
    if id_m:
        label = f"\\label{{{id_m.group(1)}}}"
    cap_tex = convert_inline(tidy_unicode(cap))
    return (
        "\\begin{figure}[ht]\n\\centering\n"
        f"\\includegraphics[width={w}]{{{path}}}\n"
        f"\\caption{{{cap_tex}}}{label}\n"
        "\\end{figure}"
    )


def note_title_and_body(inner: str) -> tuple[str, str]:
    inner = inner.strip("\n")
    title = "Note"
    lines = inner.split("\n")
    if lines and lines[0].startswith("**") and lines[0].endswith("**"):
        title = lines[0].strip("*").rstrip(".")
        inner = "\n".join(lines[1:]).strip("\n")
    return title, inner


def md_table_to_tex(block: str) -> str:
    rows = [r for r in block.strip().split("\n") if r.strip().startswith("|")]
    if len(rows) < 2:
        return block
    def cells(row: str) -> list[str]:
        parts = [c.strip() for c in row.strip().strip("|").split("|")]
        return parts

    header = cells(rows[0])
    body_rows = [cells(r) for r in rows[2:]]
    n = len(header)
    col = "l" * n
    lines = ["\\begin{center}", f"\\begin{{tabular}}{{{col}}}", "\\toprule"]
    lines.append(
        " & ".join(convert_inline(tidy_unicode(c)) for c in header) + " \\\\"
    )
    lines.append("\\midrule")
    for r in body_rows:
        r = (r + [""] * n)[:n]
        lines.append(
            " & ".join(convert_inline(tidy_unicode(c)) for c in r) + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(lines)


def convert_inline(text: str, chunks=None) -> str:
    local: list[str] = []

    def math_repl(m: re.Match[str]) -> str:
        return _ph_store(local, f"${m.group(1)}$", "XXLC")

    def code_repl(m: re.Match[str]) -> str:
        body = m.group(1).replace("\\", r"\textbackslash{}")
        body = body.replace("{", r"\{").replace("}", r"\}")
        body = body.replace("%", r"\%").replace("_", r"\_")
        body = body.replace("#", r"\#").replace("&", r"\&")
        return _ph_store(local, f"\\texttt{{{body}}}", "XXLC")

    text = re.sub(r"(?<!\$)\$(?!\$)((?:\\.|[^$\\])+)\$", math_repl, text)
    text = re.sub(r"`([^`\n]+)`", code_repl, text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = tidy_unicode(text)
    text = escape_text(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    return _ph_restore(text, local, "XXLC")


def convert_lists_and_par(text: str, chunks: list[str]) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    def is_ul(s: str):
        return re.match(r"^(\s*)[-*] (.+)$", s)

    def is_ol(s: str):
        return re.match(r"^(\s*)(\d+)\. (.+)$", s)

    while i < n:
        line = lines[i]
        if line.strip() == "---":
            out.append("")
            i += 1
            continue
        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            level = len(hm.group(1))
            title = hm.group(2).strip()
            title = re.sub(r"^Appendix [A-G]\.\s+", "", title)
            title = re.sub(r"^\d+\.\d+(\.\d+)*\s+", "", title)
            title = re.sub(r"^\d+\.\s+", "", title)
            title = convert_inline(title)
            cmd = {1: "chapter", 2: "section", 3: "subsection", 4: "subsubsection"}.get(
                level, "subsubsection"
            )
            out.append(f"\\{cmd}{{{title}}}")
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < n and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            block = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(md_table_to_tex("\n".join(block)))
            continue
        ul = is_ul(line)
        ol = is_ol(line)
        if ul or ol:
            stack: list[tuple[int, str]] = []

            def close_deeper(indent: int) -> None:
                while stack and stack[-1][0] > indent:
                    out.append(f"\\end{{{stack.pop()[1]}}}")

            while i < n:
                raw = lines[i]
                if not raw.strip():
                    i += 1
                    if i < n and not (is_ul(lines[i]) or is_ol(lines[i]) or re.match(r"^XXPH\d+XX$", lines[i].strip())):
                        break
                    continue
                um = is_ul(raw)
                om = is_ol(raw)
                if um or om:
                    m = um or om
                    indent = len(m.group(1))
                    kind = "itemize" if um else "enumerate"
                    item = m.group(2) if um else m.group(3)
                    close_deeper(indent)
                    if stack and stack[-1][0] == indent and stack[-1][1] != kind:
                        out.append(f"\\end{{{stack.pop()[1]}}}")
                    if not stack or stack[-1][0] < indent:
                        out.append(f"\\begin{{{kind}}}")
                        stack.append((indent, kind))
                    out.append("\\item " + convert_inline(item))
                    i += 1
                    continue
                if re.match(r"^XXPH\d+XX$", raw.strip()) and stack:
                    if raw[:1] in " \t":
                        out.append(raw.strip())
                        i += 1
                        continue
                    break
                if raw.startswith("  ") and stack:
                    out[-1] = out[-1] + " " + convert_inline(raw.strip())
                    i += 1
                    continue
                break
            while stack:
                out.append(f"\\end{{{stack.pop()[1]}}}")
            continue
        if line.startswith("> "):
            q = []
            while i < n and (lines[i].startswith("> ") or lines[i].strip() == ">"):
                q.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            qtex = convert_lists_and_par("\n".join(q), chunks)
            out.append("\\begin{quote}\n" + qtex + "\n\\end{quote}")
            continue
        if not line.strip():
            out.append("")
            i += 1
            continue
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^#{1,6} ", lines[i]) and not (
            is_ul(lines[i]) or is_ol(lines[i]) or lines[i].startswith("> ")
            or lines[i].strip().startswith("|") or lines[i].strip() == "---"
        ):
            para.append(lines[i])
            i += 1
        joined = " ".join(p.strip() for p in para)
        out.append(convert_inline(joined))
    return "\n".join(out)


def protect_and_convert(md: str) -> str:
    chunks: list[str] = []

    def fence_repl(m: re.Match[str]) -> str:
        lang = m.group(1) or ""
        return _ph_store(chunks, code_to_tex(lang, m.group(2)))

    md = re.sub(r"^[ \t]*```(\w*)[ \t]*\n(.*?)```", fence_repl, md, flags=re.S | re.M)

    def fig_repl(m: re.Match[str]) -> str:
        return _ph_store(chunks, figure_to_tex(m.group(0)))

    md = re.sub(
        r'<div class="book-figure"[^>]*>.*?</div>',
        fig_repl,
        md,
        flags=re.S,
    )

    def aside_repl(m: re.Match[str]) -> str:
        title, inner = note_title_and_body(m.group(1))
        inner_tex = protect_and_convert(inner)
        title_tex = convert_inline(title)
        return _ph_store(
            chunks,
            f"\\begin{{asidenote}}[{title_tex}]\n{inner_tex}\n\\end{{asidenote}}",
        )

    md = re.sub(
        r'<div class="[^"]*aside-note[^"]*"[^>]*>\s*(.*?)\s*</div>',
        aside_repl,
        md,
        flags=re.S,
    )

    def note_repl(m: re.Match[str]) -> str:
        title, inner = note_title_and_body(m.group(1))
        inner_tex = protect_and_convert(inner)
        title_tex = convert_inline(title)
        return _ph_store(
            chunks,
            f"\\begin{{implnote}}[{title_tex}]\n{inner_tex}\n\\end{{implnote}}",
        )

    md = re.sub(
        r'<div class="impl-note"[^>]*>\s*(.*?)\s*</div>',
        note_repl,
        md,
        flags=re.S,
    )

    def disp_repl(m: re.Match[str]) -> str:
        return _ph_store(chunks, display_math_to_tex(m.group(1)))

    md = re.sub(r"(?m)^[ \t]*\$\$[ \t]*\n(.*?)^[ \t]*\$\$[ \t]*$", disp_repl, md, flags=re.S)
    md = re.sub(r"(?m)^[ \t]*\$\$(.+?)\$\$[ \t]*$", disp_repl, md)
    md = re.sub(r"\$\$(.+?)\$\$", disp_repl, md, flags=re.S)

    def inl_repl(m: re.Match[str]) -> str:
        return _ph_store(chunks, f"${m.group(1)}$")

    md = re.sub(r"(?<!\$)\$(?!\$)((?:\\.|[^$\\])+)\$", inl_repl, md)

    tex = convert_lists_and_par(md, chunks)
    tex = _ph_restore(tex, chunks)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip() + "\n"


def split_book(src: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^(?=# )", src)
    skip = ("Reinforcement Learning: A Compact", "Table of Contents")
    out: list[tuple[str, str]] = []
    for part in parts[1:]:
        title = part.split("\n", 1)[0][2:].strip()
        if any(title.startswith(p) for p in skip):
            continue
        body = part.split("\n", 1)[1] if "\n" in part else ""
        out.append((title, body))
    return out


def chapter_include_name(n: int, title: str) -> str:
    t = re.sub(r"^\d+\.\s+", "", title)
    return f"{n:02d}-{slug(t)}"


def write_main(chapter_files: list[str]) -> str:
    body = MAIN_TEX
    for f in chapter_files:
        body += f"\\input{{chapters/{f}}}\n"
    body += MAIN_END
    return body


def copy_figures() -> None:
    dest = OUT / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    src = ROOT / "figures"
    if not src.exists():
        return
    for p in src.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            shutil.copy2(p, dest / p.name)


def main() -> None:
    src = (ROOT / "book.md").read_text()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "chapters").mkdir(parents=True)

    sections = split_book(src)
    chapter_files: list[str] = []
    appendix_parts: list[str] = []
    refs_tex = ""

    for title, body in sections:
        if title.startswith("Appendix "):
            appendix_parts.append(protect_and_convert("# " + title + "\n" + body))
            print("appendix", title)
            continue
        if title == "References":
            refs_tex = (
                "\\chapter*{References}\n"
                "\\addcontentsline{toc}{chapter}{References}\n\n"
                + protect_and_convert(body)
            )
            print("references")
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", title)
        if not m:
            print("skip", title)
            continue
        n = int(m.group(1))
        fname = chapter_include_name(n, title)
        tex = protect_and_convert("# " + title + "\n" + body)
        (OUT / "chapters" / f"{fname}.tex").write_text(tex)
        chapter_files.append(fname)
        print(f"chapter {n:02d}  {fname}.tex")

    (OUT / "appendix.tex").write_text("\n".join(appendix_parts) + "\n")
    (OUT / "references.tex").write_text(refs_tex)
    (OUT / "main.tex").write_text(write_main(chapter_files))
    (OUT / "latexmkrc").write_text("$pdf_mode = 1;\n")
    copy_figures()
    print(f"wrote {OUT}/  ({len(chapter_files)} chapters)")


if __name__ == "__main__":
    main()
