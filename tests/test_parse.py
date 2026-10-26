"""Run with:  python -m pytest tests"""

import pytest

from idocr.parse import clean_date, clean_sex, parse_field


@pytest.mark.parametrize("raw, expected", [
    # Full dates, different printed formats.
    ("1998.03.14", "1998-03-14"),
    ("14.03.1998", "1998-03-14"),
    ("14/03/1998", "1998-03-14"),
    ("١٩٩٨.٠٣.١٤", "1998-03-14"),  # Arabic-Indic digits
    ("I998.O3.14", "1998-03-14"),  # OCR letter/digit mix-ups
    # Presumed dates (تاريخ مفترض): unknown parts printed as XX.
    ("1971.XX.XX", "1971"),
    ("1971.XX.X>", "1971"),        # blurry X read as >
    ("1971.xx.××", "1971"),
    ("XX.XX.1971", "1971"),
    ("1971.05.XX", "1971-05"),
    # Not dates.
    ("1971.XX.05", None),          # day known but month not: nonsense
    ("1998.13.01", None),          # month 13
    ("1971.AB.CD", None),
    ("", None),
])
def test_clean_date(raw, expected):
    assert clean_date(raw) == expected


def test_presumed_flag():
    exact = parse_field("1998.03.14", 1.0, "date", None)
    presumed = parse_field("1971.XX.XX", 1.0, "date", None)
    assert (exact.valid, exact.presumed) == (True, False)
    assert (presumed.value, presumed.valid, presumed.presumed) == ("1971", True, True)
    assert parse_field("104738215", 1.0, "digits", None).presumed is None


@pytest.mark.parametrize("raw, expected", [
    ("أنثى", "F"), ("انثى", "F"), ("ذكر", "M"), ("M", "M"), ("ثى", None),
])
def test_clean_sex(raw, expected):
    assert clean_sex(raw) == expected
