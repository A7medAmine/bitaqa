"""Card templates: where each field lives on a straightened card.

Boxes are stored as fractions of the card size (x, y, w, h in 0..1), so the
same template works no matter what resolution the aligned card ends up at.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

FieldType = Literal["text", "name", "digits", "date", "sex", "place"]
Lang = Literal["latin", "arabic"]


class FieldSpec(BaseModel):
    box: tuple[float, float, float, float]  # x, y, w, h as fractions
    type: FieldType = "text"
    lang: Lang = "latin"  # which OCR model reads this field
    pattern: str | None = None  # optional regex the cleaned value must match


class Template(BaseModel):
    name: str
    description: str = ""
    card_size: tuple[int, int] = (856, 540)  # ID-1 ratio (85.6 x 54 mm)
    fields: dict[str, FieldSpec]
    # Optional straightened image of a blank card, used for feature matching
    # (see align.py). Path is relative to the template file.
    reference: str | None = None
    # Extra areas to blank when building the reference (e.g. the ID photo),
    # in addition to the field boxes. Same fractional x, y, w, h format.
    blank_boxes: list[tuple[float, float, float, float]] = []

    def pixel_box(self, field: str) -> tuple[int, int, int, int]:
        """Convert a field's fractional box to pixel coords on the aligned card."""
        cw, ch = self.card_size
        x, y, w, h = self.fields[field].box
        return int(x * cw), int(y * ch), int(w * cw), int(h * ch)


def load_template(path: str | Path) -> Template:
    path = Path(path)
    template = Template.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if template.reference:
        template.reference = str(path.parent / template.reference)
    return template
