"""Generate a FAKE ID card so you can test the pipeline without real documents.

Writes:
    samples/demo_flat.png   - the card itself, perfectly straight
    samples/demo_photo.jpg  - same card "photographed": tilted, on a table, noisy

The text is drawn inside the template's boxes, so card and template always agree.
"""

import sys
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from idocr.template import load_template  # noqa: E402

FAKE_DATA = {
    "id_number": "104738215",
    "last_name": "BENALI",
    "first_name": "YASMINE",
    "birth_date": "14.03.1998",
    "sex": "F",
    "expiry_date": "02.11.2031",
    "last_name_ar": "بن علي",
    "first_name_ar": "ياسمين",
    "birth_place_ar": "وهران",
}
LABELS = {
    "id_number": "Card No.", "last_name": "Surname", "first_name": "Given name",
    "birth_date": "Date of birth", "sex": "Sex", "expiry_date": "Expiry",
    "last_name_ar": "اللقب", "first_name_ar": "الاسم", "birth_place_ar": "مكان الميلاد",
}


def visual(text: str) -> str:
    """Prepare Arabic for drawing: join letters into their connected forms
    (reshaper), then flip to right-to-left display order (bidi). Pillow's basic
    text layout does neither on its own."""
    return get_display(arabic_reshaper.reshape(text))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_flat_card(template) -> np.ndarray:
    w, h = template.card_size
    card = Image.new("RGB", (w, h), (226, 236, 230))
    d = ImageDraw.Draw(card)

    # Background guilloche-ish lines, like real security printing.
    for i in range(0, w + h, 14):
        d.line([(i, 0), (i - h, h)], fill=(212, 226, 218), width=1)

    # Header band inset from the edge: real cards keep a light border, which
    # is what lets edge detection separate the card from a dark table.
    d.rectangle([12, 12, w - 12, int(0.15 * h)], fill=(24, 84, 64))
    d.text((int(0.04 * w), int(0.04 * h)), "DEMO REPUBLIC  -  IDENTITY CARD",
           font=font(30, bold=True), fill=(255, 255, 255))

    # Photo placeholder.
    d.rectangle([int(0.04 * w), int(0.22 * h), int(0.29 * w), int(0.84 * h)],
                fill=(180, 190, 186), outline=(120, 130, 126), width=2)
    d.text((int(0.11 * w), int(0.50 * h)), "PHOTO", font=font(24), fill=(110, 120, 116))

    for name, spec in template.fields.items():
        x, y, bw, bh = template.pixel_box(name)
        value_font = font(int(bh * 0.78), bold=True)
        if spec.lang == "arabic":
            # Arabic reads right-to-left, so align label and value to the box's right edge.
            d.text((x + bw, y - 4), visual(LABELS[name]), font=font(16), fill=(70, 90, 80), anchor="rb")
            d.text((x + bw - 2, y + 2), visual(FAKE_DATA[name]), font=value_font,
                   fill=(20, 20, 20), anchor="ra")
        else:
            d.text((x, y - 18), LABELS[name], font=font(15), fill=(70, 90, 80))
            d.text((x + 2, y + 2), FAKE_DATA[name], font=value_font, fill=(20, 20, 20))

    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


def fake_photo(card: np.ndarray) -> np.ndarray:
    """Place the card tilted on a dark 'table' with blur and sensor noise."""
    ch, cw = card.shape[:2]
    W, H = 1600, 1200
    rng = np.random.default_rng(7)

    table = np.full((H, W, 3), (48, 40, 36), np.uint8)
    table = cv2.add(table, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))

    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = np.float32([[260, 230], [1340, 170], [1390, 900], [210, 980]])  # perspective tilt
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, M, (W, H))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (W, H))

    photo = np.where(mask[..., None] > 0, warped, table)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    noise = rng.normal(0, 6, photo.shape)
    return np.clip(photo + noise, 0, 255).astype(np.uint8)


def main() -> None:
    template = load_template(ROOT / "templates" / "demo_card.json")
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)

    card = draw_flat_card(template)
    cv2.imwrite(str(out / "demo_flat.png"), card)
    cv2.imwrite(str(out / "demo_photo.jpg"), fake_photo(card), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Wrote {out / 'demo_flat.png'} and {out / 'demo_photo.jpg'}")


if __name__ == "__main__":
    main()
