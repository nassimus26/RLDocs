"""Erase the cropped vertical 'resample' on the InstructGPT figure and redraw it."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(
    "/Users/dipesh/Downloads/Gemini_Generated_Image_yiciu2yiciu2yici.jpeg"
)
DST = ROOT / "figures" / "instructgpt_style_flow.jpg"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
INK = (30, 30, 30)
STROKE = (214, 214, 214)
EXTRA_RIGHT = 52


def light_frame(im):
    m = min(im.size)
    pad = max(8, round(m * 0.015))
    stroke = max(1, round(m * 0.003))
    im = ImageOps.expand(im, border=pad, fill="white")
    im = ImageOps.expand(im, border=stroke, fill=STROKE)
    return im


def vertical_label(text, size):
    font = ImageFont.truetype(FONT, size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    strip = Image.new("RGBA", (r - l + 8, b - t + 8), (255, 255, 255, 0))
    ImageDraw.Draw(strip).text((4 - l, 4 - t), text, font=font, fill=INK + (255,))
    # 90° CCW: first letter at the bottom (read bottom → top)
    return strip.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)


def main():
    im = Image.open(SRC).convert("RGB")
    ImageDraw.Draw(im).rectangle((1425, 360, 1454, 515), fill="white")
    im = ImageOps.expand(im, border=(0, 0, EXTRA_RIGHT, 0), fill="white")
    label = vertical_label("resample", 32)
    # just right of the return-path stroke (~1421–1423); center on π_ref
    x = 1430
    y = 408 + (75 - label.size[1]) // 2
    im.paste(label, (x, y), label)
    im = light_frame(im)
    im.save(DST, quality=95, optimize=True)
    print(f"wrote {DST} {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
