"""Build a template for YOUR card by drawing a box around each field.

    python scripts/calibrate.py my_card.jpg --name my_card \
        --fields id_number:digits last_name:name last_name_ar:name:arabic birth_date:date

For each field a window opens: drag a rectangle around where that field's
VALUE is printed, press ENTER/SPACE to confirm (C to skip). The result is
saved to templates/<name>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from idocr.align import align_card  # noqa: E402

CARD_SIZE = (856, 540)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--name", required=True)
    p.add_argument("--fields", nargs="+", required=True,
                   help="name:type[:lang], type in text|name|digits|date|sex, lang in latin|arabic")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")
    card, method = align_card(img, CARD_SIZE)
    if method == "none":
        print("Card outline not found; using the whole image. Crop tightly or shoot on a dark background.")

    fields = {}
    for item in args.fields:
        name, ftype, lang = (item.split(":") + ["", ""])[:3]
        x, y, w, h = cv2.selectROI(f"Draw box: {name}", card, showCrosshair=True)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            print(f"skipped {name}")
            continue
        cw, ch = CARD_SIZE
        fields[name] = {"box": [round(x / cw, 4), round(y / ch, 4), round(w / cw, 4), round(h / ch, 4)],
                        "type": ftype or "text", "lang": lang or "latin"}

    out = ROOT / "templates" / f"{args.name}.json"
    out.write_text(json.dumps({"name": args.name, "card_size": list(CARD_SIZE), "fields": fields},
                              indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
