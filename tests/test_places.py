"""Run with:  python -m pytest tests"""

from idocr.parse import parse_field
from idocr.places import match_place


def test_commune_and_wilaya_with_missing_letter():
    place = match_place("ين وسارة الجلفة")  # OCR dropped the first letter of عين
    assert place.text == "عين وسارة الجلفة"
    assert (place.post_code, place.wilaya_code, place.wilaya_fr) == ("17031", 17, "Djelfa")


def test_commune_only_gets_its_wilaya():
    place = match_place("تيميمون")
    assert (place.text, place.commune_fr, place.wilaya_code) == ("تيميمون", "Timimoun", 49)


def test_multi_word_commune_with_alef_variant():
    place = match_place("اولاد سيدي ميهوب")  # أ printed or read without hamza
    assert (place.text, place.wilaya_code) == ("أولاد سيدي ميهوب", 48)


def test_same_name_in_two_wilayas():
    # عين البيضاء exists in Oum El Bouaghi (4) and Ouargla (30).
    assert match_place("عين البيضاء ورقلة").post_code == "30002"
    alone = match_place("عين البيضاء")
    assert (alone.commune_ar, alone.post_code, alone.wilaya_code) == ("عين البيضاء", None, None)


def test_new_wilaya_name_is_recognised():
    place = match_place("بوسعادة بوسعادة")
    assert (place.wilaya_code, place.wilaya_ar) == (68, "بوسعادة")


def test_unknown_place_is_left_alone():
    assert match_place("باريس") is None  # Paris, one letter from the commune اريس (Arris)
    assert match_place("اريس باتنة").commune_fr == "Arris"  # the printed wilaya confirms it
    field = parse_field("باريس", 0.99, "place", None)
    assert (field.value, field.valid, field.place) == ("باريس", False, None)


def test_parse_field_snaps_place():
    field = parse_field("ين وسارة", 0.95, "place", None)
    assert (field.value, field.valid, field.raw) == ("عين وسارة", True, "ين وسارة")
