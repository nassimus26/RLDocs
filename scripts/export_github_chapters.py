#!/usr/bin/env python3
"""Export book.md -> chapters/*.md for GitHub math preview.

book.md stays Cursor/KaTeX-friendly ($$...$$, \\tag{n}).
Chapter files are transformed for github.com:
  - $$ displays -> fenced math blocks
  - \\tag{n} -> \\quad (n)  (avoids Chromium MathML vertical stack)
  - <div class="impl-note"> -> GitHub > [!NOTE] callout boxes
  - one chapter per page (GitHub MathJax budget)
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _protect_inline_code(text: str) -> tuple[str, list[str]]:
    chunks: list[str] = []

    def repl(m: re.Match[str]) -> str:
        chunks.append(m.group(0))
        return f"\0CODE{len(chunks) - 1}\0"

    return re.sub(r"`[^`\n]+`", repl, text), chunks


def _restore_inline_code(text: str, chunks: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        return chunks[int(m.group(1))]

    return re.sub(r"\0CODE(\d+)\0", repl, text)


def _fix_math_body(body: str) -> str:
    body = body.strip("\n")
    body = re.sub(r"\\tag\{([^}]+)\}", r"\\quad (\1)", body)
    body = re.sub(r"(?<!\\)\\\\\n", r"\\\\ \n", body)
    lines = body.split("\n")
    out: list[str] = []
    for line in lines:
        if line.strip() in ("=", "&=", "+", "-") and out:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def convert_display_math(text: str) -> str:
    text, codes = _protect_inline_code(text)

    def repl_block(m: re.Match[str]) -> str:
        body = _fix_math_body(m.group(1))
        return f"```math\n{body}\n```"

    # Multiline $$ blocks (optional indent)
    text = re.sub(
        r"(?m)^[ \t]*\$\$[ \t]*\n(.*?)^[ \t]*\$\$[ \t]*$",
        repl_block,
        text,
        flags=re.S,
    )

    # Single-line $$...$$ (optional indent)
    def repl_one(m: re.Match[str]) -> str:
        body = _fix_math_body(m.group(1))
        return f"```math\n{body}\n```"

    text = re.sub(r"(?m)^[ \t]*\$\$(.+?)\$\$[ \t]*$", repl_one, text)

    # Any remaining same-line $$...$$ (e.g. mid-paragraph)
    text = re.sub(r"\$\$(.+?)\$\$", repl_one, text, flags=re.S)

    text = _restore_inline_code(text, codes)

    # Export note: Cursor uses $$, chapters use math fences
    text = text.replace(
        "This guide uses `$...$` for inline math and `$$...$$`\n"
        "> for display math.",
        "This chapter is a GitHub export of `book.md`. Inline math uses `$...$`; "
        "display math uses fenced `math` blocks.",
    )
    return text


def merge_eq_continuations(text: str) -> str:
    pattern = re.compile(
        r"(```math\n)(.*?)(\n```)\n+(\n*)```math\n(=)", re.S
    )
    while True:
        new, c = pattern.subn(r"\1\2\n\5", text, count=1)
        if not c:
            break
        text = new
    return text


def convert_note_boxes(text: str) -> str:
    """Turn Cursor note boxes into GitHub alert callouts."""

    def make_repl(alert: str):
        def repl(m: re.Match) -> str:
            body = m.group(1).strip("\n")
            out = [f"> [!{alert}]"]
            for line in body.split("\n"):
                out.append("> " + line if line.strip() else ">")
            return "\n".join(out)

        return repl

    # Asides first (may also carry impl-note for Cursor preview styling)
    text = re.sub(
        r'<div class="[^"]*aside-note[^"]*"[^>]*>\s*(.*?)\s*</div>',
        make_repl("TIP"),
        text,
        flags=re.S,
    )
    text = re.sub(
        r'<div class="impl-note"[^>]*>\s*(.*?)\s*</div>',
        make_repl("NOTE"),
        text,
        flags=re.S,
    )
    return text


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def math_count(body: str) -> int:
    inl = len(re.findall(r"(?<!\$)\$(?!\$)((?:\\.|[^$\\])+)\$(?!\$)", body))
    return inl + body.count("```math")


def main() -> None:
    src = (ROOT / "book.md").read_text()
    gh = convert_note_boxes(merge_eq_continuations(convert_display_math(src)))
    if "$$" in gh:
        sample = re.search(r".{0,60}\$\$.{0,60}", gh)
        raise SystemExit(f"export still contains $$: {sample.group(0)!r}")
    if "\\tag{" in gh:
        raise SystemExit("export still contains \\tag")

    out = ROOT / "chapters"
    out.mkdir(exist_ok=True)
    for p in out.glob("*.md"):
        p.unlink()

    parts = re.split(r"(?m)^(?=# )", gh)
    skip = ("Reinforcement Learning: A Compact", "Table of Contents")
    entries: list[tuple[str, str, int]] = []
    for part in parts[1:]:
        title = part.split("\n", 1)[0][2:].strip()
        if any(title.startswith(p) for p in skip):
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", title)
        app = re.match(r"^Appendix ([A-D])\.\s+(.*)$", title)
        if m:
            fname = f"{int(m.group(1)):02d}-{slug(m.group(2))}.md"
        elif app:
            fname = f"appendix-{app.group(1).lower()}-{slug(app.group(2))}.md"
        else:
            fname = f"{slug(title)}.md"
        lines = part.split("\n")
        body = (
            lines[0]
            + "\n\n[← Chapter index](README.md) · [Edit full book](../book.md)\n\n"
            + "\n".join(lines[1:])
        )
        n = math_count(part)
        if n >= 950:
            raise SystemExit(
                f"{fname} has {n} math nodes (GitHub limit ~1000); split further"
            )
        (out / fname).write_text(body)
        entries.append((fname, title, n))
        print(f"{n:4d}  {fname}")

    index = [
        "# Book chapters (GitHub preview)",
        "",
        "Edit [`book.md`](../book.md) in Cursor — that file’s formatting is canonical.",
        "",
        "These files are generated by `scripts/export_github_chapters.py` so github.com",
        "can render the math (one chapter per page; `$$` / `\\tag` rewritten for GitHub).",
        "",
    ]
    for fname, title, _n in entries:
        index.append(f"- [{title}]({fname})")
    index.append("")
    (out / "README.md").write_text("\n".join(index))
    print(f"wrote {len(entries)} chapters")


if __name__ == "__main__":
    main()
