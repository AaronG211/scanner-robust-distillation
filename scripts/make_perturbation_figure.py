#!/usr/bin/env python3
"""
Build the "what we will actually do" slide figure: real PathMNIST patches put
through the same perturbation families the experiment will use, at increasing
strength.

Rows    = perturbation family (hue, saturation, contrast, blur, JPEG)
Columns = strength level (0 = untouched original, then 4 increasing steps)

Output: figures/perturbation_ladder.png
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "data" / "pathmnist_64.npz"
OUT = ROOT / "figures" / "perturbation_ladder.png"

CELL = 96          # display size per patch
PAD = 6
CAP_H = 16         # room under each patch for its strength label
LABEL_W = 150
HEADER_H = 34

# five strength levels; index 0 is always the untouched original
HUE = [0, 0.04, 0.08, 0.14, 0.20]          # fraction of the colour wheel
SAT = [1.0, 1.35, 1.7, 2.1, 2.6]           # saturation multiplier
CON = [1.0, 0.8, 0.62, 0.46, 0.32]         # contrast multiplier
BLUR = [0, 0.6, 1.2, 1.9, 2.8]             # gaussian sigma (px, at 64x64)
JPEG = [100, 40, 22, 12, 7]                # JPEG quality


def shift_hue(img: Image.Image, frac: float) -> Image.Image:
    if frac == 0:
        return img
    hsv = np.array(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(frac * 255)) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def jpeg(img: Image.Image, q: int) -> Image.Image:
    if q >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img: Image.Image, sigma: float) -> Image.Image:
    from PIL import ImageFilter
    return img if sigma == 0 else img.filter(ImageFilter.GaussianBlur(sigma))


ROWS = [
    ("Hue shift",   HUE,  lambda im, v: shift_hue(im, v),                    lambda v: f"+{v:.2f}"),
    ("Saturation",  SAT,  lambda im, v: ImageEnhance.Color(im).enhance(v),   lambda v: f"x{v:.2f}"),
    ("Contrast",    CON,  lambda im, v: ImageEnhance.Contrast(im).enhance(v),lambda v: f"x{v:.2f}"),
    ("Blur",        BLUR, lambda im, v: blur(im, v),                         lambda v: f"s={v:.1f}"),
    ("JPEG",        JPEG, lambda im, v: jpeg(im, v),                         lambda v: f"q={v}"),
]


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
              else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    d = np.load(NPZ)
    imgs, labels = d["train_images"], d["train_labels"].ravel()
    # class 8 = colorectal adenocarcinoma epithelium in PathMNIST -- visually rich
    idx = int(np.where(labels == 8)[0][7])
    base = Image.fromarray(imgs[idx]).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    print(f"using train patch #{idx}, class {labels[idx]}")

    n_col = 5
    W = LABEL_W + n_col * (CELL + PAD) + PAD
    H = HEADER_H + len(ROWS) * (CELL + PAD + CAP_H) + PAD + 20
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    f_lab, f_hdr, f_sm = font(15, True), font(14, True), font(12)

    for c in range(n_col):
        x = LABEL_W + c * (CELL + PAD)
        txt = "original" if c == 0 else f"level {c}"
        tw = dr.textlength(txt, font=f_hdr)
        dr.text((x + (CELL - tw) / 2, 8), txt, fill="black", font=f_hdr)
    dr.line([(LABEL_W - PAD, HEADER_H - 6), (W - PAD, HEADER_H - 6)], fill="#cccccc", width=1)

    for r, (name, levels, fn, fmt) in enumerate(ROWS):
        y = HEADER_H + r * (CELL + PAD + CAP_H)
        dr.text((PAD, y + CELL // 2 - 9), name, fill="black", font=f_lab)
        for c, v in enumerate(levels):
            x = LABEL_W + c * (CELL + PAD)
            canvas.paste(fn(base.copy(), v), (x, y))
            cap = fmt(v)
            cw = dr.textlength(cap, font=f_sm)
            dr.text((x + (CELL - cw) / 2, y + CELL + 2), cap, fill="#666666", font=f_sm)

    dr.text((PAD, H - 16),
            "PathMNIST 64x64 colon patch under the five perturbation families "
            "(strength range fixed before training).",
            fill="#444444", font=f_sm)

    OUT.parent.mkdir(exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
