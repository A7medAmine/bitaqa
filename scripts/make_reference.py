"""Build the reference image used for feature-matching alignment.

    python scripts/make_reference.py private/good_photo.jpg --template templates/dz_id_front.json

Takes a clear, straight-on photo of the card, aligns it with the outline
method, then paints over every field box and every `blank_boxes` area (the ID
photo etc.). What remains is what EVERY card of this type shares: printed
labels, header, flag, background pattern. That is exactly what matching
should lock onto - and the saved file contains no personal data.

The result is saved next to the template and linked via its "reference" key.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from idocr.align import align_card  # noqa: E402
from idocr.template import load_template  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--template", required=True)
    args = p.parse_args()

    template_path = Path(args.template)
    template = load_template(template_path)
    template.reference = None  # build from the outline method, not an old reference

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")
    card, method = align_card(img, template.card_size)
    if method != "outline":
        sys.exit("Card outline not found. Use a photo on a plain dark background, card upright.")

    # Fill with the card's typical background colour so no hard edges appear
    # (hard edges would themselves become keypoints).
    fill = tuple(int(v) for v in np.median(card.reshape(-1, 3), axis=0))
    cw, ch = template.card_size
    boxes = [f.box for f in template.fields.values()] + list(template.blank_boxes)
    for x, y, w, h in boxes:
        x0, y0 = int(x * cw) - 6, int(y * ch) - 6
        cv2.rectangle(card, (x0, y0), (int((x + w) * cw) + 6, int((y + h) * ch) + 6), fill, -1)

    out = template_path.with_name(f"{template.name}_ref.png")
    cv2.imwrite(str(out), card)

    raw = json.loads(template_path.read_text(encoding="utf-8"))
    raw["reference"] = out.name
    template_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out} and linked it in {template_path}")


if __name__ == "__main__":
    main()
