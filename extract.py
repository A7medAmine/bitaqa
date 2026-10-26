"""Command-line entry point.

    python extract.py samples/demo_photo.jpg
    python extract.py my_card.jpg --template templates/my_card.json --debug out/
"""

import argparse
import sys

from idocr import extract_card, load_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured fields from an ID card image.")
    parser.add_argument("images", nargs="+", help="image file(s)")
    parser.add_argument("--template", default="templates/demo_card.json")
    parser.add_argument("--debug", help="folder to save aligned card + drawn boxes")
    args = parser.parse_args()

    # Windows consoles default to a legacy codepage that can't print Arabic.
    sys.stdout.reconfigure(encoding="utf-8")

    template = load_template(args.template)
    for path in args.images:
        result = extract_card(path, template, debug_dir=args.debug)
        print(result.model_dump_json(indent=2))
        if len(args.images) > 1:
            print(f"--- {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
