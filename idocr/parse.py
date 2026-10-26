"""Step 4: turn raw OCR text into clean, validated values.

OCR makes predictable mistakes (O vs 0, I vs 1, stray punctuation). Each field
type knows what it *should* look like, so we fix what we can and flag the rest.
"""

import re
from datetime import date

from pydantic import BaseModel

from .places import Place, match_place

# Letters OCR often returns where a digit was printed, plus Arabic-Indic
# digits (٠١٢٣٤٥٦٧٨٩) mapped to 0-9 so every number comes out the same way.
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1",
                             "|": "1", "Z": "2", "S": "5", "s": "5", "B": "8", "G": "6",
                             **{chr(0x0660 + i): str(i) for i in range(10)}})
# Tatweel (ـ, decorative stretch) and harakat (short-vowel marks) are noise for matching.
ARABIC_NOISE = re.compile(r"[ـً-ْ]")


class FieldResult(BaseModel):
    value: str | None  # cleaned value, None if unreadable
    raw: str           # exactly what OCR returned
    confidence: float  # OCR confidence 0..1
    valid: bool        # passed the type/pattern check
    # Dates only: True for a presumed date (تاريخ مفترض), printed with XX for the
    # unknown parts. value then keeps only the known parts: "1971" or "1971-05".
    presumed: bool | None = None
    # Places only: the matched commune/wilaya, None if no official name is close.
    place: Place | None = None


class CardResult(BaseModel):
    template: str
    align_method: str  # "features", "outline", or "none" (image used as-is)
    fields: dict[str, FieldResult]
    elapsed_ms: float


def clean_digits(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw.translate(DIGIT_FIXES))
    return digits or None


def clean_name(raw: str) -> str | None:
    # Keep letters (any alphabet, Arabic included), spaces, hyphens and apostrophes.
    name = re.sub(r"[^\w\s'-]|\d|_", "", ARABIC_NOISE.sub("", raw)).strip()
    return re.sub(r"\s+", " ", name).upper() or None


def clean_date(raw: str) -> str | None:
    """Accept dd.mm.yyyy / dd/mm/yyyy / yyyy-mm-dd etc., return ISO yyyy-mm-dd."""
    text = raw.translate(DIGIT_FIXES)
    for pattern, order in ((r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", "ymd"),
                           (r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", "dmy"),
                           (r"(\d{2})(\d{2})(\d{4})", "dmy")):
        if m := re.search(pattern, text):
            parts = dict(zip(order, map(int, m.groups())))
            try:
                return date(parts["y"], parts["m"], parts["d"]).isoformat()
            except ValueError:
                return None  # e.g. month 13: looks like a date but isn't
    return clean_presumed_date(text)


# What OCR returns for a printed "X": x, ×, K, or stray > < % when the X is blurry.
UNKNOWN_PART = re.compile(r"[Xx×Kk><%*]{1,2}")


def clean_presumed_date(text: str) -> str | None:
    """Presumed date (تاريخ مفترض): unknown day/month printed as XX.

    "1971.XX.XX" -> "1971",  "1971.05.XX" -> "1971-05",  "XX.XX.1971" -> "1971".
    The result is an ISO 8601 date with reduced precision.
    """
    parts = [p for p in re.split(r"[.\-/\s]+", text.strip()) if p]
    if len(parts) != 3:
        return None
    if re.fullmatch(r"\d{4}", parts[0]):
        year, month, day = parts
    elif re.fullmatch(r"\d{4}", parts[2]):
        day, month, year = parts
    else:
        return None

    def known(part: str) -> int | None:
        if re.fullmatch(r"\d{1,2}", part):
            return int(part)
        if UNKNOWN_PART.fullmatch(part):
            return None
        raise ValueError(part)  # neither a number nor XX: not a date

    try:
        m, d = known(month), known(day)
    except ValueError:
        return None
    if m is None:
        return year if d is None else None  # a known day without a month makes no sense
    if not 1 <= m <= 12:
        return None
    return f"{year}-{m:02d}"


def is_presumed(value: str) -> bool:
    """Full ISO dates are 10 chars (yyyy-mm-dd); shorter means parts were XX."""
    return len(value) < 10


ARABIC_SEX = {"ذكر": "M", "أنثى": "F", "انثى": "F", "أنثي": "F", "انثي": "F"}


def clean_sex(raw: str) -> str | None:
    text = ARABIC_NOISE.sub("", raw).strip(" :.")
    for word, sex in ARABIC_SEX.items():
        if word in text:
            return sex
    letter = text.upper()[:1]
    return letter if letter in ("M", "F") else None


CLEANERS = {
    "digits": clean_digits,
    "name": clean_name,
    "date": clean_date,
    "sex": clean_sex,
    "place": clean_name,  # then snapped to the official spelling in parse_field
    # Strip edge punctuation: a box drawn up to its label may catch the label's colon.
    "text": lambda raw: raw.strip(" :.-|") or None,
}


def parse_field(raw: str, confidence: float, field_type: str, pattern: str | None) -> FieldResult:
    value = CLEANERS[field_type](raw)
    place = match_place(value) if field_type == "place" and value else None
    if place:
        value = place.text
    valid = value is not None and (pattern is None or re.fullmatch(pattern, value) is not None)
    if field_type == "place":
        valid = valid and place is not None  # unknown place (e.g. abroad): flag for review
    presumed = is_presumed(value) if field_type == "date" and value else None
    return FieldResult(value=value, raw=raw, confidence=round(confidence, 3), valid=valid,
                       presumed=presumed, place=place)
