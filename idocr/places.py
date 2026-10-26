"""Algerian place names: snap OCR'd communes and wilayas to the official list.

Birth places and issuing authorities are real places, so a near miss such as
"ين وسارة" can be corrected to the only commune it can be ("عين وسارة"). Data: 69
wilayas and 1541 communes from github.com/kossa/algerian-cities (MIT licence,
see idocr/data/LICENSE).

The commune list still uses the 58-wilaya split: communes of the 11 newest
wilayas (59-69) point to the wilaya they belonged to before. The wilaya name
printed on the card is matched against all 69.
"""

import json
import re
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

from pydantic import BaseModel

DATA = Path(__file__).parent / "data"
# Below this similarity the text is left as read: it is probably not an
# Algerian place (e.g. born abroad) or too garbled to guess. A commune alone
# needs a closer match, because one letter off can be a different real place
# ("باريس" Paris vs "اريس" Arris); a printed wilaya that also matches confirms it.
MIN_SCORE = 0.8
MIN_SCORE_ALONE = 0.9
# Letters used interchangeably in print and by OCR: hamza forms of alef, ى/ي, ة/ه.
FOLD = str.maketrans("أإآٱىة", "اااايه")


class Place(BaseModel):
    text: str               # official Arabic spelling, laid out like the input
    commune_ar: str
    commune_fr: str | None  # None when the commune name exists in several wilayas
    post_code: str | None
    wilaya_code: int | None
    wilaya_ar: str | None
    wilaya_fr: str | None
    score: float            # similarity of the OCR text to the official name(s), 0..1


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(FOLD)).strip()


@cache
def communes() -> list[dict]:
    rows = json.loads((DATA / "Commune_Of_Algeria.json").read_text(encoding="utf-8"))
    return [{"ar": r["ar_name"], "fr": r["name"], "post_code": str(r["post_code"]).zfill(5),
             "wilaya": int(r["wilaya_id"]), "key": fold(r["ar_name"])} for r in rows]


@cache
def wilayas() -> dict[int, dict]:
    rows = json.loads((DATA / "Wilaya_Of_Algeria.json").read_text(encoding="utf-8"))
    return {int(r["code"]): {"ar": r["ar_name"], "fr": r["name"], "code": int(r["code"]),
                             "key": fold(r["ar_name"])} for r in rows}


def closest(text: str, rows: list[dict]) -> tuple[float, list[dict]]:
    """The rows whose name is most similar to text (ties kept), and that similarity."""
    scored = [(SequenceMatcher(None, text, r["key"]).ratio(), r) for r in rows]
    top = max(score for score, _ in scored)
    return top, [r for score, r in scored if score == top]


@cache
def closest_commune(text: str) -> tuple[float, list[dict]]:
    return closest(text, communes())


@cache
def closest_wilaya(text: str) -> tuple[float, list[dict]]:
    return closest(text, list(wilayas().values()))


def match_place(text: str) -> Place | None:
    """Match "commune" or "commune wilaya" text to the official names.

    Every split of the words into a commune part and an optional wilaya part
    is tried, and the reading where both parts match best wins. Returns None
    when nothing is close enough.
    """
    words = fold(text).split()
    best, best_score = None, 0.0
    # Commune only first, so it wins ties against a commune + wilaya reading.
    for i in range(len(words), 0, -1):
        commune_text, wilaya_text = " ".join(words[:i]), " ".join(words[i:])
        score, candidates = closest_commune(commune_text)
        wilaya = None
        if wilaya_text:
            w_score, w_rows = closest_wilaya(wilaya_text)
            if len(w_rows) > 1:
                continue
            score, wilaya = min(score, w_score), w_rows[0]
        if score < (MIN_SCORE if wilaya else MIN_SCORE_ALONE) or (best and score <= best_score):
            continue
        if len({c["ar"] for c in candidates}) > 1:
            continue  # equally close to different names: can't choose
        # Same name in several wilayas (e.g. عين البيضاء): the printed wilaya decides.
        if wilaya and len(candidates) > 1:
            candidates = [c for c in candidates if c["wilaya"] == wilaya["code"]] or candidates
        commune = candidates[0] if len(candidates) == 1 else None
        if wilaya is None and commune:
            wilaya = wilayas()[commune["wilaya"]]
        best, best_score = (candidates[0]["ar"], commune, wilaya, wilaya_text), score

    if best is None:
        return None
    commune_ar, commune, wilaya, wilaya_text = best
    return Place(
        text=f"{commune_ar} {wilaya['ar']}" if wilaya_text else commune_ar,
        commune_ar=commune_ar,
        commune_fr=commune and commune["fr"],
        post_code=commune and commune["post_code"],
        wilaya_code=wilaya and wilaya["code"],
        wilaya_ar=wilaya and wilaya["ar"],
        wilaya_fr=wilaya and wilaya["fr"],
        score=round(best_score, 3),
    )
