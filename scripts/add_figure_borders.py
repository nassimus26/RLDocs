"""Permanently add a thin light frame around raster (and ppo_clip.svg) figures.

Never crops the drawing: only expands with a white pad and a light stroke.
"""

from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1] / "figures"
STROKE = (214, 214, 214)  # #d6d6d6
STROKE_HEX = "#d6d6d6"
EXTS = {".jpg", ".jpeg", ".png"}


def sizes(w, h):
    m = min(w, h)
    pad = max(8, round(m * 0.015))
    stroke = max(1, round(m * 0.003))
    return pad, stroke


def near(p, target, tol=12):
    return max(abs(c - s) for c, s in zip(p, target)) < tol


def already_framed(im):
    px = im.convert("RGB").load()
    w, h = im.size
    samples = (px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1])
    return all(near(p, STROKE) for p in samples)


def border_raster(path):
    im = Image.open(path)
    if already_framed(im):
        print(f"{path.name}: skip (already light-framed)")
        return
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        src = im.convert("RGBA")
        bg.paste(src, mask=src.split()[-1] if src.mode == "RGBA" else None)
        im = bg
    else:
        im = im.convert("RGB")
    pad, stroke = sizes(*im.size)
    im = ImageOps.expand(im, border=pad, fill="white")
    im = ImageOps.expand(im, border=stroke, fill=STROKE)
    suf = path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        im.save(path, quality=95, optimize=True)
    else:
        im.save(path, optimize=True)
    print(f"{path.name}: pad={pad}px stroke={stroke}px {im.size[0]}x{im.size[1]}")


def border_svg(path):
    text = path.read_text()
    frame = (
        '<rect id="page-frame" x="1" y="1" width="758" height="418"'
        f' fill="none" stroke="{STROKE_HEX}" stroke-width="1.5"/>'
    )
    if 'id="page-frame"' in text:
        if STROKE_HEX in text:
            print(f"{path.name}: svg already light-framed")
            return
        raise SystemExit(f"unexpected existing svg frame: {path}")
    needle = '<rect width="760" height="420" fill="#ffffff"/>'
    if needle not in text:
        raise SystemExit(f"unexpected svg: {path}")
    path.write_text(text.replace(needle, needle + "\n  " + frame, 1))
    print(f"{path.name}: svg stroke added")


def main():
    for p in sorted(ROOT.iterdir()):
        if p.suffix.lower() in EXTS:
            border_raster(p)
    svg = ROOT / "ppo_clip.svg"
    if svg.exists():
        border_svg(svg)


if __name__ == "__main__":
    main()
