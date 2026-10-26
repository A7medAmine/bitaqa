# bitaqa

[![tests](https://github.com/A7medAmine/bitaqa/actions/workflows/tests.yml/badge.svg)](https://github.com/A7medAmine/bitaqa/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Extract structured JSON from photos of ID cards, Arabic fields included.
Ships with a template for the **Algerian biometric national ID** (front side),
and works for any card with a fixed layout once you draw its field boxes.
CPU only, a few hundred ms per card once the OCR models are loaded.

```json
"birth_date":     { "value": "1971", "raw": "1971.XX.XX", "valid": true, "presumed": true },
"birth_place_ar": { "value": "عين وسارة", "raw": "ين وسارة", "valid": true,
                    "place": { "commune_fr": "Ain Oussera", "post_code": "17031", "wilaya_code": 17 } }
```

> **Privacy:** everything runs locally; no image or text leaves your machine.
> Never commit real card photos. `.gitignore` excludes all images by default.

## How it works

```
photo
  │  1. align.py     OpenCV matches the card to a reference (or finds its outline)
  │                  and warps it flat and upright (856x540)
  ▼
flat card
  │  2. template     each field is a fixed box on the flat card (templates/*.json)
  ▼
field crops
  │  3. ocr.py       RapidOCR (PaddleOCR models on ONNX Runtime) reads each crop
  ▼
raw text
  │  4. parse.py     clean OCR mistakes, normalise dates, validate with regex
  ▼
JSON (Pydantic CardResult)
```

Only step 3 is a neural network, and it is pretrained. Everything else is
classic image processing and plain Python. Because every card of one type has
the same layout, "where is the name?" is answered by a template, not by AI.

## Read the code in this order

1. `templates/demo_card.json`: what a layout looks like
2. `idocr/align.py`: edges, contours, perspective warp
3. `idocr/ocr.py`: calling the OCR model on one line of text
4. `idocr/parse.py`: turning messy text into clean values
5. `idocr/pipeline.py`: glues the four steps together
6. `extract.py`: the command-line wrapper

## Setup

Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Try it on the fake demo card

```powershell
python scripts/make_demo_card.py                       # writes samples/demo_photo.jpg
python extract.py samples/demo_photo.jpg --debug debug  # prints JSON
```

Open `debug/boxes.png` to see the straightened card with each field box drawn
(green = valid value, red = failed validation). This image is the first thing
to check whenever results look wrong.

## Alignment: two methods

1. **Feature matching** (when the template has a `reference` image). SIFT finds
   distinctive points (letter corners, flag, emblem, background pattern) on the
   reference card and in the photo, matches them, and RANSAC computes the
   homography that maps one onto the other. Works at **any rotation, including
   upside down**, on cluttered backgrounds, and when the card's edges are hidden.
2. **Outline detection** (fallback). Finds the card's 4 corners from edges or
   brightness. Needs a plain, contrasting background and an upright card.

`align_method` in the output says which one was used (`features`, `outline`, `none`).

Build a reference once per card type from a clear photo on a dark background:

```powershell
python scripts/make_reference.py private/good_photo.jpg --template templates/my_card.json
```

It blanks every field box plus `blank_boxes` (the ID photo), so only what all
cards share remains, and the reference contains no personal data.

## Included templates

- `templates/demo_card.json`: the fake demo card
- `templates/dz_id_front.json`: Algerian biometric national ID, front side
  (card number, NIN, Arabic names, birth date/place, sex, issue/expiry dates,
  issuing authority, blood type)

```powershell
python extract.py private/id.jpg --template templates/dz_id_front.json --debug debug
```

## Use it on your own card

1. Take a clear photo of your card on a dark, plain background, filling most of the frame.
2. Draw the field boxes:
   ```powershell
   python scripts/calibrate.py private/my_card.jpg --name my_card `
       --fields id_number:digits last_name:name first_name:name birth_date:date sex:sex expiry_date:date
   ```
   Drag a box around where each *value* is printed (not the label). Leave a
   little margin, because the next photo won't be perfectly identical.
3. Build the reference image for rotation-proof alignment (see "Alignment" above).
4. Optionally add a `"pattern"` regex to fields in `templates/my_card.json`
   (e.g. `"^\\d{9}$"` for a 9-digit number) so wrong reads get flagged.
5. Run:
   ```powershell
   python extract.py private/my_card.jpg --template templates/my_card.json --debug debug
   ```

Keep real card images in `private/`. `.gitignore` excludes all images so they
never get committed.

## Arabic fields

Each field has a `lang` in the template: `"latin"` (default) or `"arabic"`.
Each language uses its own recognition model, because a model trained on one
script cannot read another. Bilingual cards therefore get one box per
language, e.g. `last_name` and `last_name_ar`:

```json
"last_name_ar": { "box": [0.66, 0.36, 0.30, 0.08], "type": "name", "lang": "arabic" }
```

- Arabic output is in normal reading order (logical order), ready to store or compare.
- Arabic-Indic digits (٠١٢...) are converted to 0-9 in `digits` and `date` fields.
- Tatweel (ـ) and short-vowel marks are stripped from names.
- Read numbers and dates with `"latin"` even on Arabic cards when they are printed
  in 0-9; the Latin model is more accurate on digits.
- The first run downloads the Arabic model (about 10 MB), so it is slow once.

When calibrating, set the language with a third part: `last_name_ar:name:arabic`.

## Field types

| type     | cleaning                                         |
|----------|--------------------------------------------------|
| `digits` | fixes O→0, I→1, S→5 etc., keeps digits only       |
| `name`   | keeps letters/spaces/hyphens, uppercases         |
| `date`   | accepts dd.mm.yyyy, dd/mm/yyyy, yyyy-mm-dd; outputs ISO. Presumed dates (تاريخ مفترض, e.g. `1971.XX.XX`) give `"1971"` with `"presumed": true` |
| `sex`    | M or F                                           |
| `place`  | cleaned like `name`, then snapped to the official Algerian commune/wilaya spelling (see below) |
| `text`   | trimmed, unchanged                               |

## Algerian places

`place` fields (birth place, issuing authority) are matched against the
official list of 69 wilayas and 1541 communes (`idocr/places.py`, data in
`idocr/data/` from [kossa/algerian-cities](https://github.com/kossa/algerian-cities),
MIT). Text like `"commune"` or `"commune wilaya"` is fuzzy-matched, so OCR
slips get fixed: `ين وسارة الجلفة` becomes `عين وسارة الجلفة`. The match is added
to the field:

```json
"place": { "text": "عين وسارة الجلفة", "commune_ar": "عين وسارة", "commune_fr": "Ain Oussera",
           "post_code": "17031", "wilaya_code": 17, "wilaya_ar": "الجلفة",
           "wilaya_fr": "Djelfa", "score": 0.941 }
```

- Nothing close enough (e.g. born abroad): the value stays as read, `place` is
  `null` and `valid` is `false`, so it gets reviewed.
- A commune alone needs a closer match (0.9) than one confirmed by its printed
  wilaya (0.8), because one letter can turn one real place into another.
- Commune names found in several wilayas (e.g. عين البيضاء) are resolved by the
  printed wilaya; without one, `post_code` and `wilaya_*` stay `null`.
- The commune list still uses the 58-wilaya split: communes of wilayas 59-69
  report their former wilaya unless the card prints the new wilaya's name.

## Tests

```powershell
pip install pytest
python -m pytest tests
```

## Known limits / next steps

- **Card not found** (`align_method: "none"`): no reference match and no outline.
  Usually heavy glare, strong blur, or the card is tiny in the frame. The image
  is then used as-is, which only works for tight, upright scans.
- **Values shift between cards**: printers place text a few pixels differently.
  Draw boxes generously (up to the label) - `trim_to_text` in `ocr.py` cuts the
  empty space away before OCR.
- **Other scripts** (Cyrillic, Devanagari...): add an entry to `MODEL_PARAMS` in `ocr.py`
  and allow the new value in `Lang` in `template.py`.
- **Two-sided cards**: make one template per side.
- **MRZ** (`<<<` lines on the back): parse it and cross-check it against the front fields.
- **Train your own model**: once this works, its outputs become labels for
  fine-tuning a detector (YOLO) or an image-to-JSON model (Donut / small VLM).

## Credits

- OCR: [RapidOCR](https://github.com/RapidAI/RapidOCR) running
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models on ONNX Runtime
- Algerian wilayas and communes: [kossa/algerian-cities](https://github.com/kossa/algerian-cities)
  (MIT, see `idocr/data/LICENSE`)

## License

[MIT](LICENSE)

This project is for building legitimate document-processing tools (KYC,
digitising your own records). Only process ID cards you are authorised to
handle, and follow the data-protection law that applies to you (in Algeria,
Law 18-07 on the protection of personal data).
