"""Step 1: find the card in a photo and straighten it (pure OpenCV, no AI).

Two methods:

A. Feature matching (used when the template has a reference image).
   Find distinctive points (SIFT keypoints) on a clean reference card and in
   the photo, match them, and let RANSAC compute the homography that maps the
   photo onto the reference. Works at any rotation, on busy backgrounds and
   when the card's edges are hidden, because it uses what is printed ON the
   card, not its outline.

B. Outline detection (fallback). The card is the biggest 4-cornered shape in
   the picture. Find its corners and warp. Fast, but needs a background that
   contrasts with the card, and cannot tell which way is up.
"""

from functools import lru_cache

import cv2
import numpy as np

WORK_WIDTH = 800  # detect on a small copy: much faster, same corners
CARD_RATIO = 85.6 / 54  # ID-1 card format (bank cards, most national IDs)
RATIO_TOLERANCE = 0.15  # allow 15% off, since perspective distorts the ratio

MATCH_SIZE = 1200   # longest photo side used for feature matching
LOWE_RATIO = 0.75   # keep a match only if clearly better than the 2nd best
MIN_INLIERS = 25    # fewer RANSAC-agreeing matches than this = no card found


# ---------- A. Feature matching ----------

_sift = cv2.SIFT_create(nfeatures=4000)


@lru_cache(maxsize=8)
def reference_features(path: str):
    """Keypoints of the reference card, computed once per template."""
    ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(f"Cannot read reference image: {path}")
    keypoints, descriptors = _sift.detectAndCompute(ref, None)
    return np.float32([k.pt for k in keypoints]), descriptors


def match_to_reference(img: np.ndarray, reference: str) -> np.ndarray | None:
    """Return the 3x3 homography mapping full-size photo -> reference card, or None."""
    ref_pts, ref_desc = reference_features(reference)

    scale = min(1.0, MATCH_SIZE / max(img.shape[:2]))
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    keypoints, desc = _sift.detectAndCompute(gray, None)
    if desc is None or len(keypoints) < MIN_INLIERS:
        return None

    # For each photo point, find its 2 closest reference points. If the best
    # is not clearly closer than the second (Lowe's ratio test), it's ambiguous.
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, ref_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None

    src = np.float32([keypoints[m.queryIdx].pt for m in good]) / scale  # back to full size
    dst = ref_pts[[m.trainIdx for m in good]]
    # RANSAC: repeatedly fit a homography to random subsets and keep the one
    # most matches agree with. Wrong matches (background, glare) get ignored.
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inliers.sum() < MIN_INLIERS:
        return None
    return H


# ---------- B. Outline detection ----------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)               # x + y: smallest = top-left, largest = bottom-right
    d = np.diff(pts, axis=1).ravel()  # y - x: smallest = top-right, largest = bottom-left
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def edge_mask(gray: np.ndarray) -> np.ndarray:
    """Outline where brightness jumps (Canny edges)."""
    edges = cv2.Canny(gray, 50, 150)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)  # close small gaps


def bright_mask(gray: np.ndarray) -> np.ndarray:
    """Card = bright blob on a darker background.

    Otsu picks the brightness cut-off automatically. Morphological "open"
    removes thin bright specks (wood grain, dust); "close" fills dark text
    holes inside the card so it becomes one solid shape.
    """
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def quad_from_contour(c: np.ndarray) -> np.ndarray | None:
    """Reduce an outline to 4 corners, tolerating rounded card corners.

    Simplify the convex hull with a growing tolerance until 4 points remain.
    """
    hull = cv2.convexHull(c)
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx)
    return None


def has_card_ratio(corners: np.ndarray) -> bool:
    """Sanity check: shape must have roughly the card's width/height ratio.

    Catches cases like a dark header blending into a dark table, where only
    part of the card gets outlined.

    Landscape only: a portrait shape with a card-like ratio is usually the ID
    photo on the card itself, not the card.
    """
    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    return abs(w / h - CARD_RATIO) / CARD_RATIO < RATIO_TOLERANCE


def find_card_corners(img: np.ndarray) -> np.ndarray | None:
    """Locate the card's 4 corners in the full-size image, or None if not found."""
    scale = WORK_WIDTH / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # remove noise before finding shapes
    min_area = 0.1 * small.shape[0] * small.shape[1]  # card must fill >= 10% of photo

    for mask in (edge_mask(gray), bright_mask(gray)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < min_area:
                break
            corners = quad_from_contour(c)
            if corners is not None and has_card_ratio(corners):
                return corners / scale

    return None


# ---------- Entry point ----------

def align_card(img: np.ndarray, card_size: tuple[int, int],
               reference: str | None = None) -> tuple[np.ndarray, str]:
    """Return (flat card image, method used: "features", "outline" or "none").

    "none" means nothing was found and the image is assumed to already be a
    tight, upright scan of the card.
    """
    w, h = card_size

    if reference:
        H = match_to_reference(img, reference)
        if H is not None:
            return cv2.warpPerspective(img, H, (w, h)), "features"

    corners = find_card_corners(img)
    if corners is not None:
        target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners, target)
        return cv2.warpPerspective(img, matrix, (w, h)), "outline"

    return cv2.resize(img, (w, h)), "none"
