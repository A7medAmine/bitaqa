"""Step 3: read text from a small field crop (this is the AI part).

RapidOCR runs PaddleOCR's pretrained neural nets through ONNX Runtime, which
is fast on CPU and needs no GPU. Because each crop holds a single line of text,
we skip text *detection* and only run *recognition* - that is what makes
per-field OCR fast.

Each script needs its own recognition model: a model trained on Latin letters
cannot read Arabic. The template says which model each field uses.
"""

import cv2
import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

MIN_HEIGHT = 48  # recognizer works best on text roughly this tall

MODEL_PARAMS = {
    "latin": {},  # RapidOCR default: PP-OCRv6 (Latin letters, digits, Chinese)
    "arabic": {
        "Rec.lang_type": LangRec.ARABIC,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    },
}


def trim_to_text(crop: np.ndarray, margin: int = 6) -> np.ndarray:
    """Cut away empty background so the text fills the crop.

    Boxes are drawn wide because names vary in length. But the recognizer
    squeezes the whole crop into a fixed-height strip, so a short name inside
    a mostly empty box gets read badly. Dark pixels = ink; keep their bounding
    box plus a small margin.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop specks smaller than a dot of text (noise, background print).
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]
    if not keep:
        return crop
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    h, w = crop.shape[:2]
    return crop[max(y0 - margin, 0):min(y1 + margin, h), max(x0 - margin, 0):min(x1 + margin, w)]


class OcrEngine:
    def __init__(self) -> None:
        self.engines: dict[str, RapidOCR] = {}

    def _engine(self, lang: str) -> RapidOCR:
        """Load each language model only the first time a field needs it."""
        if lang not in self.engines:
            self.engines[lang] = RapidOCR(params=MODEL_PARAMS[lang])
        return self.engines[lang]

    def read_line(self, crop: np.ndarray, lang: str = "latin") -> tuple[str, float]:
        """Return (text, confidence 0..1) for a single-line crop."""
        crop = trim_to_text(crop)
        if crop.shape[0] < MIN_HEIGHT:
            f = MIN_HEIGHT / crop.shape[0]
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)

        out = self._engine(lang)(crop, use_det=False, use_cls=False, use_rec=True)
        texts, scores = out.txts or (), out.scores or ()
        if not texts:
            return "", 0.0
        return " ".join(texts).strip(), float(min(scores))
