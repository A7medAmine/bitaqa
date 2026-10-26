"""Glue: photo -> align -> crop fields -> OCR -> parse -> CardResult."""

import time
from pathlib import Path

import cv2
import numpy as np

from .align import align_card
from .ocr import OcrEngine
from .parse import CardResult, parse_field
from .template import Template

PAD = 4  # extra pixels around each box so slightly-off text isn't cut

_engine: OcrEngine | None = None


def get_engine() -> OcrEngine:
    """Load the OCR model once and reuse it; loading is the slow part."""
    global _engine
    if _engine is None:
        _engine = OcrEngine()
    return _engine


def extract_card(image: str | Path | np.ndarray, template: Template,
                 debug_dir: str | Path | None = None) -> CardResult:
    start = time.perf_counter()

    img = cv2.imread(str(image)) if not isinstance(image, np.ndarray) else image
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image}")

    # Step 1: straighten the card.
    card, method = align_card(img, template.card_size, template.reference)

    engine = get_engine()
    fields = {}
    for name, spec in template.fields.items():
        # Step 2: cut out the field using the template box.
        x, y, w, h = template.pixel_box(name)
        crop = card[max(y - PAD, 0):y + h + PAD, max(x - PAD, 0):x + w + PAD]
        # Step 3: OCR.  Step 4: clean + validate.
        raw, conf = engine.read_line(crop, spec.lang)
        fields[name] = parse_field(raw, conf, spec.type, spec.pattern)

    elapsed = (time.perf_counter() - start) * 1000

    if debug_dir:
        save_debug(card, template, fields, Path(debug_dir))

    return CardResult(template=template.name, align_method=method,
                      fields=fields, elapsed_ms=round(elapsed, 1))


def save_debug(card: np.ndarray, template: Template, fields: dict, out: Path) -> None:
    """Save the aligned card with field boxes drawn: green = valid, red = not."""
    out.mkdir(parents=True, exist_ok=True)
    vis = card.copy()
    for name in template.fields:
        x, y, w, h = template.pixel_box(name)
        color = (0, 180, 0) if fields[name].valid else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, name, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(out / "aligned.png"), card)
    cv2.imwrite(str(out / "boxes.png"), vis)
