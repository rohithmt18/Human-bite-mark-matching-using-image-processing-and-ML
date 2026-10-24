"""
preprocessing.py — Reusable bite-mark image processing & feature extraction.

Enhanced Pipeline stages:
    1.  Load & validate image
    2.  Resize to standard working resolution (prevents scale-variance issues)
    3.  Background removal  (HSV + saturation fallback for pink/beige wax)
    4.  Auto-crop to mold bounding box
    5.  Specular glare / reflection removal  (inpaint bright highlights)
    6.  Dirt & dust removal  (morphological cleaning + area-based blob removal)
    7.  Shadow / uneven-illumination normalisation  (background division)
    8.  CLAHE contrast enhancement
    9.  NLM denoising (stronger than bilateral for textured dental wax)
   10.  Adaptive threshold + combined Canny+morphological-gradient edges
   11.  Improved contour filtering  (area, aspect, circularity, inner-mask, merge)
   12.  Per-impression & aggregate feature extraction
"""

import cv2
import math
import numpy as np
import os


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Working resolution — images are resized to this before any processing.
# Larger values preserve more detail but are slower.
WORK_SIZE = (1024, 1024)

# Minimum side of the mold after cropping below which we abort
MIN_MOLD_DIM = 80


# ===========================================================================
# 1. Load & validate
# ===========================================================================

def load_and_validate(image_path):
    """Load an image from disk. Returns (image, True) or (None, False)."""
    if not os.path.isfile(image_path):
        return None, False
    image = cv2.imread(image_path)
    if image is None:
        return None, False
    return image, True


# ===========================================================================
# 2. Resize to working resolution (keeps aspect ratio, pads remainder)
# ===========================================================================

def resize_to_work(image, target=(1024, 1024)):
    """
    Letterbox-resize `image` so the longest side equals target size.
    Background is filled with black.  Returns (resized_image, scale_factor).
    """
    th, tw = target
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, scale


# ===========================================================================
# 3. Background removal via HSV colour segmentation
# ===========================================================================

def remove_background(image):
    """
    Isolate the red/pink wax mold from the background.

    Handles:
      - Classic red wax   (hue 0-15 and 160-180)
      - Pink/light wax    (hue 0-20, lower saturation)
      - Beige/cream wax   (hue 15-30, low saturation)
      - Any high-V region that is clearly the mold

    Returns
    -------
    fg_mask  : uint8 binary mask (255 = foreground mold)
    isolated : BGR image with background blacked out
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Red (wraps in HSV) ---
    mask_r1 = cv2.inRange(hsv, np.array([0,  30, 60]), np.array([18, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([155, 30, 60]), np.array([180, 255, 255]))

    # --- Pink / light rose ---
    mask_pink = cv2.inRange(hsv, np.array([0, 15, 120]), np.array([20, 120, 255]))

    # --- Beige / cream ---
    mask_beige = cv2.inRange(hsv, np.array([15, 10, 130]), np.array([35, 80, 255]))

    fg_mask = cv2.bitwise_or(mask_r1, mask_r2)
    fg_mask = cv2.bitwise_or(fg_mask, mask_pink)
    fg_mask = cv2.bitwise_or(fg_mask, mask_beige)

    # Morphological closing to fill holes inside the mold
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)

    # Morphological opening to remove tiny specks of background colour
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open)

    # Keep only the single largest connected region (the mold itself)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = np.uint8(labels == largest) * 255

    isolated = cv2.bitwise_and(image, image, mask=fg_mask)
    return fg_mask, isolated


# ===========================================================================
# 4. Auto-crop to mold bounding box
# ===========================================================================

def autocrop_to_mold(image, fg_mask, pad=12):
    """
    Crop both the image and the mask to the tight bounding box of the mold.

    Returns
    -------
    roi        : cropped BGR image
    roi_mask   : cropped binary mask
    bbox       : (x, y, w, h) in the pre-crop image
    """
    coords = cv2.findNonZero(fg_mask)
    if coords is None:
        return image, fg_mask, (0, 0, image.shape[1], image.shape[0])

    bx, by, bw, bh = cv2.boundingRect(coords)
    H, W = image.shape[:2]

    bx = max(bx - pad, 0)
    by = max(by - pad, 0)
    bw = min(bw + 2 * pad, W - bx)
    bh = min(bh + 2 * pad, H - by)

    if bw < MIN_MOLD_DIM or bh < MIN_MOLD_DIM:
        return image, fg_mask, (0, 0, W, H)

    masked = cv2.bitwise_and(image, image, mask=fg_mask)
    roi      = masked[by:by + bh, bx:bx + bw]
    roi_mask = fg_mask[by:by + bh, bx:bx + bw]
    return roi, roi_mask, (bx, by, bw, bh)


# ===========================================================================
# 5. Specular glare / reflection removal
# ===========================================================================

def remove_glare(roi, roi_mask):
    """
    Detect specular highlights (very bright, near-white pixels) and
    inpaint them using the Telea fast-marching method.

    These highlights appear on shiny dental wax under camera flash or
    overhead lighting and corrupt edge detection completely.

    Returns
    -------
    cleaned  : BGR image with glare inpainted
    glare_mask : uint8 mask of inpainted regions (for debug)
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    # Specular highlights: very high brightness AND very low saturation
    glare_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    glare_mask[(v_ch > 230) & (s_ch < 40)] = 255

    # Restrict to mold area only
    glare_mask = cv2.bitwise_and(glare_mask, roi_mask)

    # Dilate slightly so the inpainting fills the full highlight blob
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.dilate(glare_mask, k, iterations=1)

    if cv2.countNonZero(glare_mask) == 0:
        return roi.copy(), glare_mask

    cleaned = cv2.inpaint(roi, glare_mask, inpaintRadius=5,
                          flags=cv2.INPAINT_TELEA)
    # Re-apply mold mask so inpainting doesn't bleed into background
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=roi_mask)
    return cleaned, glare_mask


# ===========================================================================
# 6. Dirt, dust, and small-artifact removal
# ===========================================================================

def remove_dirt(roi, roi_mask):
    """
    Remove small dirt specks, dust, and isolated noise blobs from the image.

    Strategy:
    a) Convert to grayscale and threshold very dark regions (dirt/scratches)
       that are NOT tooth impressions.
    b) Filter out blobs smaller than a minimum area via connected-components.
    c) Inpaint the tiny dark blobs.

    Returns
    -------
    cleaned   : BGR image with small dirt inpainted
    dirt_mask : uint8 mask of removed regions (for debug)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=roi_mask)

    # Very small very-dark spots — candidate dirt
    _, dark = cv2.threshold(gray_masked, 18, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, roi_mask)

    # Remove blobs larger than 600 px² (those are real tooth impressions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark, connectivity=8
    )
    dirt_mask = np.zeros_like(dark)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 600:                         # small enough to be dirt
            dirt_mask[labels == lbl] = 255

    # Also catch mid-tone speckle using morphological top-hat on the gray image
    k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray_masked, cv2.MORPH_BLACKHAT, k_tophat)
    _, speckle = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    speckle = cv2.bitwise_and(speckle, roi_mask)

    # Only keep speckles smaller than 300 px²
    num_sp, lbl_sp, stats_sp, _ = cv2.connectedComponentsWithStats(
        speckle, connectivity=8
    )
    for lbl in range(1, num_sp):
        if stats_sp[lbl, cv2.CC_STAT_AREA] < 300:
            dirt_mask[lbl_sp == lbl] = 255

    if cv2.countNonZero(dirt_mask) == 0:
        return roi.copy(), dirt_mask

    cleaned = cv2.inpaint(roi, dirt_mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=roi_mask)
    return cleaned, dirt_mask


# ===========================================================================
# 7. Shadow / uneven-illumination normalisation
# ===========================================================================

def normalize_illumination(roi, roi_mask):
    """
    Correct uneven lighting and shadows using background-division normalisation.

    Method:
    - Estimate the illumination field by heavily blurring the image.
    - Divide the original by this illumination estimate (retinex-style).
    - This removes gradual brightness gradients (e.g. shadow on one side).

    Returns corrected BGR image.
    """
    # Work on L-channel (perceptual brightness)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    l_float = l_ch.astype(np.float32) + 1.0   # avoid division by zero

    # Illumination estimate: very large Gaussian blur
    blur_sigma = max(roi.shape[:2]) // 4
    blur_sigma = max(blur_sigma, 21)
    if blur_sigma % 2 == 0:
        blur_sigma += 1
    illumination = cv2.GaussianBlur(l_float, (0, 0), blur_sigma)

    # Normalize
    l_norm = (l_float / illumination) * 128.0
    l_norm = np.clip(l_norm, 0, 255).astype(np.uint8)

    lab_norm = cv2.merge([l_norm, a_ch, b_ch])
    normalised = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    normalised = cv2.bitwise_and(normalised, normalised, mask=roi_mask)
    return normalised


# ===========================================================================
# 8. CLAHE contrast enhancement
# ===========================================================================

def correct_lighting(roi, roi_mask, clip_limit=2.5, tile_grid=(8, 8)):
    """
    Apply CLAHE on the L-channel (LAB space) for contrast enhancement.
    Lower clip_limit than before (3.0 → 2.5) because illumination is already
    normalised in step 7, so we don't want to over-sharpen.
    """
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    corrected = cv2.bitwise_and(corrected, corrected, mask=roi_mask)
    return corrected


# ===========================================================================
# 9. Denoising (NLM + mild bilateral)
# ===========================================================================

def denoise(roi, roi_mask):
    """
    Two-stage denoising:
    1. fastNlMeansDenoisingColored — strong noise reduction, texture-preserving
    2. Bilateral filter — edge-preserving spatial smoothing
    """
    # NLM: h=8 is gentle; h=10 for noisier images
    nlm = cv2.fastNlMeansDenoisingColored(roi, None,
                                           h=8, hColor=8,
                                           templateWindowSize=7,
                                           searchWindowSize=21)
    # Bilateral for final edge preservation
    result = cv2.bilateralFilter(nlm, d=7, sigmaColor=50, sigmaSpace=50)
    result = cv2.bitwise_and(result, result, mask=roi_mask)
    return result


# ===========================================================================
# 10. Segment tooth impressions — edges + threshold + contour filtering
# ===========================================================================

def segment_impressions(denoised, roi_mask,
                        min_area=600, max_area_frac=0.06,
                        max_aspect=5.5, min_circularity=0.10,
                        erode_kernel_size=70):
    """
    Detect individual tooth impression contours with enhanced filtering.

    Improvements over v1:
    - Combined edge map: Canny + morphological gradient (OR'd together)
      → catches both hard edges and gradual dark-region boundaries
    - Circularity filter to reject worm-like or straight-line artifacts
    - Solidity filter to reject highly irregular shapes (likely noise)
    - Tighter inner-mask erosion kernel (70 vs 80) for more interior coverage

    Parameters
    ----------
    min_area         : minimum contour area in pixels
    max_area_frac    : maximum contour area as fraction of image area
    max_aspect       : maximum width/height aspect ratio
    min_circularity  : 4π·area/perimeter² — rejects non-tooth shapes
    erode_kernel_size: size of erosion kernel for inner-mask

    Returns
    -------
    good_contours, edges_combined, thresh, inner_mask, gray
    """
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

    # --- Gaussian blur for stable edge detection ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Edge map 1: Canny (sharp boundaries) ---
    edges_canny = cv2.Canny(blur, 25, 100)
    edges_canny = cv2.bitwise_and(edges_canny, edges_canny, mask=roi_mask)

    # --- Edge map 2: Morphological gradient (dark regions in wax surface) ---
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_morph = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k_morph)
    _, edges_morph = cv2.threshold(edges_morph, 20, 255, cv2.THRESH_BINARY)
    edges_morph = cv2.bitwise_and(edges_morph, edges_morph, mask=roi_mask)

    # --- Combined edge map ---
    edges_combined = cv2.bitwise_or(edges_canny, edges_morph)

    # --- Adaptive threshold on blurred gray ---
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    # Add edges into the threshold map (helps catch soft impressions)
    thresh = cv2.bitwise_or(thresh, edges_combined)

    # Morphological cleanup of the threshold map
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_clean)   # remove speckle
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_clean)   # fill micro-holes

    # Inner mask: only keep contours whose centre is well inside the wax
    k_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size)
    )
    inner_mask = cv2.erode(roi_mask, k_erode, iterations=1)

    # --- Find external contours ---
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = roi_mask.shape[0] * roi_mask.shape[1]
    max_area   = image_area * max_area_frac

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area or area >= max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # --- Circularity filter ---
        # Perfect circle = 1.0; elongated / noisy shape approaches 0
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # --- Aspect ratio filter ---
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue

        # --- Solidity filter: contour must be relatively compact ---
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.30:      # very irregular → reject
                continue

        # --- Centroid must be inside the eroded inner mask ---
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        H_m, W_m = inner_mask.shape[:2]
        if cy < 0 or cy >= H_m or cx < 0 or cx >= W_m:
            continue
        if inner_mask[cy, cx] == 0:
            continue

        candidates.append(contour)

    # Merge heavily-overlapping detections of the same tooth
    good_contours = _merge_nearby_contours(candidates, overlap_thresh=0.35)

    return good_contours, edges_combined, thresh, inner_mask, gray


# ===========================================================================
# Helpers for contour merging
# ===========================================================================

def _merge_nearby_contours(contours, overlap_thresh=0.35):
    """
    Merge contours whose bounding boxes overlap significantly.
    Prevents a single tooth impression from being counted multiple times.
    """
    if len(contours) <= 1:
        return contours

    bboxes = [cv2.boundingRect(c) for c in contours]
    merged = [False] * len(contours)
    result = []

    for i in range(len(contours)):
        if merged[i]:
            continue
        group = [contours[i]]
        merged[i] = True
        for j in range(i + 1, len(contours)):
            if merged[j]:
                continue
            if _bbox_overlap_ratio(bboxes[i], bboxes[j]) > overlap_thresh:
                group.append(contours[j])
                merged[j] = True

        if len(group) == 1:
            result.append(group[0])
        else:
            all_pts = np.vstack(group)
            hull    = cv2.convexHull(all_pts)
            result.append(hull)

    return result


def _bbox_overlap_ratio(box1, box2):
    """IoU-style overlap between two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1 = max(x1, x2);  iy1 = max(y1, y2)
    ix2 = min(x1+w1, x2+w2);  iy2 = min(y1+h1, y2+h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area1  = w1 * h1
    area2  = w2 * h2
    return inter / max(min(area1, area2), 1)


# ===========================================================================
# 11. Feature extraction
# ===========================================================================

def extract_per_impression_features(contours):
    """
    Extract shape features for each individual tooth impression contour.
    Returns a list of dicts.
    """
    features = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area       = cv2.contourArea(contour)
        perimeter  = cv2.arcLength(contour, True)

        aspect_ratio  = round(w / h, 3) if h > 0 else 0
        rect_area     = w * h
        extent        = round(area / rect_area, 3) if rect_area > 0 else 0

        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity  = round(area / hull_area, 3) if hull_area > 0 else 0

        circularity = (
            round((4 * math.pi * area) / (perimeter ** 2), 4)
            if perimeter > 0 else 0
        )

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        hu = cv2.HuMoments(M).flatten()

        features.append({
            "id":           i + 1,
            "area":         round(area, 2),
            "perimeter":    round(perimeter, 2),
            "width":        w,
            "height":       h,
            "aspect_ratio": aspect_ratio,
            "solidity":     solidity,
            "extent":       extent,
            "circularity":  circularity,
            "centroid_x":   cx,
            "centroid_y":   cy,
            "hu1": hu[0], "hu2": hu[1], "hu3": hu[2], "hu4": hu[3],
            "hu5": hu[4], "hu6": hu[5], "hu7": hu[6],
        })
    return features


def extract_aggregate_features(impression_features, sample_id):
    """
    Compute aggregate (bite-level) features from per-impression features.
    Returns (summary_dict, distances_list).
    """
    if not impression_features:
        return {
            "sample_id":       sample_id,
            "impression_count": 0,
            "bite_width": 0, "bite_height": 0,
            "avg_spacing": 0, "total_area": 0, "total_perimeter": 0,
            "avg_aspect_ratio": 0, "avg_solidity": 0, "avg_circularity": 0,
        }, []

    # Sort by centroid-x for inter-tooth spacing
    sorted_imp = sorted(impression_features, key=lambda f: f["centroid_x"])

    distances = []
    for j in range(len(sorted_imp) - 1):
        a, b = sorted_imp[j], sorted_imp[j + 1]
        dist = math.sqrt(
            (b["centroid_x"] - a["centroid_x"]) ** 2 +
            (b["centroid_y"] - a["centroid_y"]) ** 2
        )
        distances.append(round(dist, 2))

    avg_spacing = round(sum(distances) / len(distances), 2) if distances else 0

    all_cx = [f["centroid_x"] for f in impression_features]
    all_cy = [f["centroid_y"] for f in impression_features]
    bite_width  = max(all_cx) - min(all_cx) if len(all_cx) >= 2 else 0
    bite_height = max(all_cy) - min(all_cy) if len(all_cy) >= 2 else 0

    n              = len(impression_features)
    total_area     = sum(f["area"]        for f in impression_features)
    total_perimeter= sum(f["perimeter"]   for f in impression_features)
    avg_solidity   = round(sum(f["solidity"]    for f in impression_features) / n, 3)
    avg_aspect     = round(sum(f["aspect_ratio"]for f in impression_features) / n, 3)
    avg_circ       = round(sum(f["circularity"] for f in impression_features) / n, 4)

    # Average Hu moments across all impressions
    avg_hu = {}
    for k in range(1, 8):
        key = f"hu{k}"
        avg_hu[f"avg_{key}"] = round(
            sum(f[key] for f in impression_features) / n, 6
        )

    summary = {
        "sample_id":         sample_id,
        "impression_count":  n,
        "bite_width":        bite_width,
        "bite_height":       bite_height,
        "avg_spacing":       avg_spacing,
        "total_area":        round(total_area, 2),
        "total_perimeter":   round(total_perimeter, 2),
        "avg_aspect_ratio":  avg_aspect,
        "avg_solidity":      avg_solidity,
        "avg_circularity":   avg_circ,
    }
    summary.update(avg_hu)

    # Individual inter-impression spacing columns
    for j, dist in enumerate(distances):
        summary[f"spacing_{j+1}_{j+2}"] = dist

    return summary, distances


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def draw_contour_boxes(roi, contours):
    """Draw numbered bounding boxes + contour outlines on a copy of the ROI."""
    output = roi.copy()
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(output, [contour], -1, (0, 200, 255), 1)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output, str(i + 1), (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2
        )
    return output


def save_debug_images(output_dir, stages):
    """Save a dict of {filename: image_array} to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for fname, img in stages.items():
        cv2.imwrite(os.path.join(output_dir, fname), img)


# ===========================================================================
# Full pipeline: image path → features + debug images
# ===========================================================================

def process_single_image(image_path, output_dir=None):
    """
    Run the complete enhanced pipeline on a single bite-mark image.

    Parameters
    ----------
    image_path : str  — path to the dental mold photograph
    output_dir : str  — if given, debug images are saved here

    Returns
    -------
    summary       : dict of aggregate features  (None on failure)
    imp_features  : list of per-impression feature dicts
    distances     : list of inter-impression distances
    contours      : list of contour arrays
    """
    image, ok = load_and_validate(image_path)
    if not ok:
        print(f"  [SKIP] Cannot load: {image_path}")
        return None, [], [], []

    sample_id = os.path.splitext(os.path.basename(image_path))[0]

    # ── 1. Resize to standard working size ──────────────────────────────────
    image_work, _ = resize_to_work(image, WORK_SIZE)

    # ── 2. Background removal ────────────────────────────────────────────────
    fg_mask, isolated = remove_background(image_work)

    # ── 3. Auto-crop to mold ─────────────────────────────────────────────────
    roi, roi_mask, bbox = autocrop_to_mold(image_work, fg_mask)

    # ── 4. Glare / reflection removal ────────────────────────────────────────
    roi_no_glare, glare_mask = remove_glare(roi, roi_mask)

    # ── 5. Dirt / dust removal ───────────────────────────────────────────────
    roi_clean, dirt_mask = remove_dirt(roi_no_glare, roi_mask)

    # ── 6. Shadow / illumination normalisation ───────────────────────────────
    roi_normalised = normalize_illumination(roi_clean, roi_mask)

    # ── 7. CLAHE contrast enhancement ────────────────────────────────────────
    roi_clahe = correct_lighting(roi_normalised, roi_mask)

    # ── 8. Denoising (NLM + bilateral) ───────────────────────────────────────
    denoised = denoise(roi_clahe, roi_mask)

    # ── 9. Segment impressions ───────────────────────────────────────────────
    contours, edges, thresh, inner_mask, gray = segment_impressions(
        denoised, roi_mask
    )

    # ── 10. Feature extraction ───────────────────────────────────────────────
    imp_features = extract_per_impression_features(contours)
    summary, distances = extract_aggregate_features(imp_features, sample_id)

    # ── 11. Save debug images ────────────────────────────────────────────────
    if output_dir:
        output_img = draw_contour_boxes(roi, contours)
        save_debug_images(output_dir, {
            "01_original.jpg":          image_work,
            "02_background_mask.jpg":   fg_mask,
            "03_isolated.jpg":          isolated,
            "04_roi_autocropped.jpg":   roi,
            "05_glare_mask.jpg":        glare_mask,
            "06_no_glare.jpg":          roi_no_glare,
            "07_dirt_mask.jpg":         dirt_mask,
            "08_no_dirt.jpg":           roi_clean,
            "09_illumination_norm.jpg": roi_normalised,
            "10_clahe.jpg":             roi_clahe,
            "11_denoised.jpg":          denoised,
            "12_gray.jpg":              gray,
            "13_edges_combined.jpg":    edges,
            "14_threshold.jpg":         thresh,
            "14b_inner_mask.jpg":       inner_mask,
            "15_contours.jpg":          output_img,
        })

    return summary, imp_features, distances, contours
