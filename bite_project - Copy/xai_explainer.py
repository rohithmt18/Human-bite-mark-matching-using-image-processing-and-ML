"""
xai_explainer.py — Explainable AI module for bite-mark forensic matching.

Generates visual explanations for why a query matched (or didn't match) a
suspect, including:
    - Side-by-side contour overlay comparison
    - Inter-tooth spacing diagrams
    - Radar / spider chart of per-metric similarity
    - Feature-by-feature diff table (as image)
    - Composite forensic report image
"""

import cv2
import math
import os
import numpy as np

from preprocessing import (
    process_single_image,
    draw_contour_boxes,
    load_and_validate,
    remove_background,
    autocrop_to_mold,
    correct_lighting,
    denoise,
    segment_impressions,
    extract_per_impression_features,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

CLR_BG       = (30, 30, 40)        # dark background
CLR_TEXT      = (220, 220, 230)     # light text
CLR_ACCENT   = (0, 200, 200)       # cyan accent
CLR_GOOD     = (0, 200, 100)       # green — close match
CLR_WARN     = (0, 180, 255)       # orange — moderate diff
CLR_BAD      = (0, 80, 255)        # red — large diff
CLR_QUERY    = (255, 180, 0)       # blue-ish for query contours
CLR_MATCH    = (0, 255, 0)         # green for match contours
CLR_GRID     = (60, 60, 80)        # grid lines

SUBJECT_NAMES = {
    "P001": "Rohith",
    "P002": "Shravya",
    "P003": "Drithika",
    "P004": "Vyshnavi",
    "P005": "Sonika",
    "P006": "Akash",
    "P007": "Rameeza",
    "P008": "Yashas",
    "P009": "Rishi",
    "P010": "Vishmaya",
    "P011": "Mounesh",
    "P012": "Prateeksha",
}

def get_person_name(subject_id, sample_id=""):
    """Retrieve person name for a subject, with automatic fallback from sample filenames."""
    import re
    sid = str(subject_id).strip()
    if sid in SUBJECT_NAMES:
        return SUBJECT_NAMES[sid]
    if sample_id:
        base = os.path.splitext(os.path.basename(str(sample_id)))[0]
        cleaned = re.sub(r'(?i)(copy\s*of|upper|lower|arch|sample|_|-|\d+)', '', base).strip()
        if cleaned:
            return cleaned.title()
    return f"Subject {sid}"


# ---------------------------------------------------------------------------
# 1. Radar / spider chart
# ---------------------------------------------------------------------------

def generate_radar_chart(breakdown, size=500):
    """
    Draw a radar chart showing per-metric similarity scores.

    breakdown: dict with keys 'structural', 'hu_moments', 'spacing', 'count', and optionally 'coatnet'
    Returns: BGR image (size x size)
    """
    if "coatnet" in breakdown:
        categories = ["Structural", "Hu Moments", "Spacing", "Count", "CoAtNet"]
        values = [
            breakdown.get("structural", 0) / 100,
            breakdown.get("hu_moments", 0) / 100,
            breakdown.get("spacing", 0) / 100,
            breakdown.get("count", 0) / 100,
            breakdown.get("coatnet", 0) / 100,
        ]
    else:
        categories = ["Structural", "Hu Moments", "Spacing", "Count"]
        values = [
            breakdown.get("structural", 0) / 100,
            breakdown.get("hu_moments", 0) / 100,
            breakdown.get("spacing", 0) / 100,
            breakdown.get("count", 0) / 100,
        ]

    img = np.full((size, size, 3), CLR_BG, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.35)
    n = len(categories)

    # Draw concentric rings (25%, 50%, 75%, 100%)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = int(radius * frac)
        cv2.circle(img, (cx, cy), r, CLR_GRID, 1, cv2.LINE_AA)
        # Label
        label = f"{int(frac * 100)}%"
        cv2.putText(img, label, (cx + 3, cy - r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_GRID, 1, cv2.LINE_AA)

    # Compute vertex positions
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    vertices = []
    for angle in angles:
        vx = int(cx + radius * math.cos(angle))
        vy = int(cy + radius * math.sin(angle))
        vertices.append((vx, vy))

    # Draw axis lines
    for vx, vy in vertices:
        cv2.line(img, (cx, cy), (vx, vy), CLR_GRID, 1, cv2.LINE_AA)

    # Draw data polygon
    data_pts = []
    for i, val in enumerate(values):
        angle = angles[i]
        dx = int(cx + radius * val * math.cos(angle))
        dy = int(cy + radius * val * math.sin(angle))
        data_pts.append((dx, dy))

    # Fill polygon
    overlay = img.copy()
    pts_arr = np.array(data_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], (200, 180, 0))
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Draw polygon outline
    for i in range(n):
        p1 = data_pts[i]
        p2 = data_pts[(i + 1) % n]
        cv2.line(img, p1, p2, CLR_ACCENT, 2, cv2.LINE_AA)

    # Draw vertices and labels
    for i in range(n):
        cv2.circle(img, data_pts[i], 5, CLR_ACCENT, -1, cv2.LINE_AA)

        # Label position (outside the chart)
        lx = int(cx + (radius + 40) * math.cos(angles[i]))
        ly = int(cy + (radius + 40) * math.sin(angles[i]))

        text = f"{categories[i]} ({values[i]*100:.0f}%)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]

        # Center the label
        tx = lx - text_size[0] // 2
        ty = ly + text_size[1] // 2
        cv2.putText(img, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)

    # Title
    composite = breakdown.get("composite", 0)
    title = f"Similarity Breakdown — {composite}%"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(img, title, (cx - ts[0] // 2, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_ACCENT, 2, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# 2. Feature comparison table (as image)
# ---------------------------------------------------------------------------

def generate_feature_table(query_features, match_features, width=600):
    """
    Render a side-by-side feature comparison table as a BGR image.
    Colour-codes the difference: green = close, orange = moderate, red = large.
    """
    keys = [
        ("impression_count", "Impressions"),
        ("bite_width", "Bite Width"),
        ("bite_height", "Bite Height"),
        ("avg_spacing", "Avg Spacing"),
        ("total_area", "Total Area"),
        ("total_perimeter", "Total Perimeter"),
        ("avg_aspect_ratio", "Avg Aspect"),
        ("avg_solidity", "Avg Solidity"),
    ]

    row_h = 36
    header_h = 50
    height = header_h + row_h * len(keys) + 10

    img = np.full((height, width, 3), CLR_BG, dtype=np.uint8)

    # Column positions
    col0 = 20   # feature name
    col1 = 200  # query value
    col2 = 340  # match value
    col3 = 480  # diff

    # Header
    cv2.putText(img, "Feature", (col0, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "Query", (col1, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "Match", (col2, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "Diff %", (col3, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_ACCENT, 1, cv2.LINE_AA)

    cv2.line(img, (10, header_h - 5), (width - 10, header_h - 5), CLR_GRID, 1)

    for i, (key, label) in enumerate(keys):
        y = header_h + row_h * i + 28

        q_val = float(query_features.get(key, 0))
        m_val = float(match_features.get(key, 0))

        # Percent difference
        denom = max(abs(q_val), abs(m_val), 1)
        diff_pct = abs(q_val - m_val) / denom * 100

        # Colour based on diff
        if diff_pct < 15:
            clr = CLR_GOOD
        elif diff_pct < 40:
            clr = CLR_WARN
        else:
            clr = CLR_BAD

        cv2.putText(img, label, (col0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"{q_val:.1f}", (col1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"{m_val:.1f}", (col2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"{diff_pct:.1f}%", (col3, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(img, (10, y + 8), (width - 10, y + 8), CLR_GRID, 1)

    return img


# ---------------------------------------------------------------------------
# 3. Side-by-side contour comparison
# ---------------------------------------------------------------------------

def _resize_to_height(img, target_h):
    """Resize keeping aspect ratio to fit target height."""
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def generate_contour_comparison(query_path, match_path, target_h=500):
    """
    Process both images and place contour-annotated versions side-by-side.
    Returns a BGR image.
    """
    # Process query
    q_img, q_ok = load_and_validate(query_path)
    m_img, m_ok = load_and_validate(match_path)

    if not q_ok or not m_ok:
        placeholder = np.full((target_h, 400, 3), CLR_BG, dtype=np.uint8)
        cv2.putText(placeholder, "Image not available", (50, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_TEXT, 1, cv2.LINE_AA)
        return placeholder

    def _process_for_display(image, colour, label):
        fg_mask, _ = remove_background(image)
        roi, roi_mask, _ = autocrop_to_mold(image, fg_mask)
        corrected = correct_lighting(roi, roi_mask)
        denoised_img = denoise(corrected, roi_mask)
        contours, _, _, _, _ = segment_impressions(denoised_img, roi_mask)

        output = roi.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), colour, 2)

        output = _resize_to_height(output, target_h)

        # Add label at top
        cv2.putText(output, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
        return output

    q_display = _process_for_display(q_img, CLR_QUERY, "QUERY")
    m_display = _process_for_display(m_img, CLR_MATCH, "MATCH")

    # Separator bar
    sep = np.full((target_h, 4, 3), CLR_ACCENT, dtype=np.uint8)

    return np.hstack([q_display, sep, m_display])


# ---------------------------------------------------------------------------
# 4. Spacing diagram overlay
# ---------------------------------------------------------------------------

def generate_spacing_diagram(imp_features, roi_shape, label=""):
    """
    Draw inter-tooth spacing lines on a blank canvas sized to roi_shape.
    """
    h, w = roi_shape[:2]
    canvas = np.full((h, w, 3), CLR_BG, dtype=np.uint8)

    if len(imp_features) < 2:
        return _resize_to_height(canvas, 400)

    sorted_imp = sorted(imp_features, key=lambda f: f["centroid_x"])

    # Draw centroid dots
    for f in sorted_imp:
        cx, cy = int(f["centroid_x"]), int(f["centroid_y"])
        cv2.circle(canvas, (cx, cy), 6, CLR_ACCENT, -1, cv2.LINE_AA)

    # Draw spacing lines
    for j in range(len(sorted_imp) - 1):
        a = sorted_imp[j]
        b = sorted_imp[j + 1]
        pt1 = (int(a["centroid_x"]), int(a["centroid_y"]))
        pt2 = (int(b["centroid_x"]), int(b["centroid_y"]))

        dist = math.sqrt(
            (b["centroid_x"] - a["centroid_x"]) ** 2 +
            (b["centroid_y"] - a["centroid_y"]) ** 2
        )

        cv2.line(canvas, pt1, pt2, CLR_GOOD, 1, cv2.LINE_AA)

        # Label distance at midpoint
        mx = (pt1[0] + pt2[0]) // 2
        my = (pt1[1] + pt2[1]) // 2
        cv2.putText(canvas, f"{dist:.0f}", (mx, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_TEXT, 1, cv2.LINE_AA)

    if label:
        cv2.putText(canvas, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_ACCENT, 2, cv2.LINE_AA)

    return _resize_to_height(canvas, 400)


# ---------------------------------------------------------------------------
# 5. Full forensic report
# ---------------------------------------------------------------------------

def generate_report(query_path, match_info, output_dir):
    """
    Generate a composite forensic report image for a single match.

    Parameters
    ----------
    query_path : str — path to query image
    match_info : tuple of (db_row, score, breakdown) from matching_engine
    output_dir : str — directory to save the report

    Returns
    -------
    report_path : str — path to the saved report image
    """
    db_row, score, breakdown = match_info

    os.makedirs(output_dir, exist_ok=True)

    # Generate components
    radar = generate_radar_chart(breakdown, size=500)

    # We need query features for the table — process the query
    q_summary, q_imp, _, _ = process_single_image(query_path)
    if q_summary is None:
        q_summary = {}
        q_imp = []

    table = generate_feature_table(q_summary, db_row, width=600)

    # Resize radar to match table height
    radar_h = table.shape[0]
    radar_resized = cv2.resize(radar, (radar_h, radar_h), interpolation=cv2.INTER_AREA)

    # Combine radar + table side by side
    top_row = np.hstack([
        radar_resized,
        np.full((radar_h, 10, 3), CLR_BG, dtype=np.uint8),
        table
    ])

    # Title bar
    title_h = 60
    title_bar = np.full((title_h, top_row.shape[1], 3), CLR_BG, dtype=np.uint8)

    subject_id = db_row.get("subject_id", "Unknown")
    samp_lower = str(db_row.get("sample_id", "")).lower()
    arch_type = "Upper Arch" if "upper" in samp_lower else ("Lower Arch" if "lower" in samp_lower else "Sample")
    person_name = get_person_name(subject_id, samp_lower)

    title_text = f"Forensic Match Report: Subject ID: {subject_id} ({arch_type}) - Score: {score}%"
    cv2.putText(title_bar, title_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, CLR_ACCENT, 2, cv2.LINE_AA)

    # Footer bar at the end of the report showing identified person name
    footer_h = 50
    footer_bar = np.full((footer_h, top_row.shape[1], 3), (20, 25, 35), dtype=np.uint8)
    footer_text = f"Identified Person Name: {person_name}  |  Subject: {subject_id}  |  Accuracy Score: {score}%"
    cv2.putText(footer_bar, footer_text, (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    # Stack vertically: title, content, footer
    report = np.vstack([title_bar, top_row, footer_bar])

    # Save
    safe_name = f"Subject_{subject_id}_{arch_type.replace(' ', '_')}"
    report_path = os.path.join(output_dir, f"report_{safe_name}.jpg")
    cv2.imwrite(report_path, report)

    return report_path


def generate_all_reports(query_path, ranked_matches, output_dir, top_n=5):
    """Generate forensic reports for the top N matches."""
    reports = []
    for i, match_info in enumerate(ranked_matches[:top_n]):
        report_path = generate_report(query_path, match_info, output_dir)
        reports.append(report_path)
        db_row, score, _ = match_info
        # Also create a rank-indexed copy for UI convenience
        rank_report_path = os.path.join(output_dir, f"report_{i+1}.jpg")
        try:
            import shutil
            if os.path.abspath(report_path) != os.path.abspath(rank_report_path):
                shutil.copyfile(report_path, rank_report_path)
        except Exception:
            pass
        print(f"  Report #{i+1}: Subject ID: {db_row.get('subject_id', '?')} "
              f"({score}%) -> {report_path}")
    return reports
