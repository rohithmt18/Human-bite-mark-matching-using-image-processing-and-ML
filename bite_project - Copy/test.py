import cv2
import os
import csv
import math
import numpy as np

# --------------------------------
# Helper: resize for display
# --------------------------------

def resize_for_display(img, max_width=1000, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    return cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

# --------------------------------
# 1. Load image
# --------------------------------

image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "P001", "Rohith lower.jpg")
image = cv2.imread(image_path)

if image is None:
    print("[ERROR] Image not found")
    exit()

print("[OK] Image loaded")
print("Width:", image.shape[1])
print("Height:", image.shape[0])

# --------------------------------
# 2. Create results directory
# --------------------------------

os.makedirs("results", exist_ok=True)

# --------------------------------
# 3. Background removal using HSV color segmentation
# --------------------------------
# The dental mold is red/pink on a dark wood table.
# We isolate the mold by detecting the red/pink region.

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red wraps around in HSV, so we need two ranges
# Lower red range (0-10)
lower_red1 = np.array([0, 40, 50])
upper_red1 = np.array([15, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

# Upper red range (160-180)
lower_red2 = np.array([160, 40, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine both red masks
fg_mask = cv2.bitwise_or(mask1, mask2)

# Clean up the mask: close small holes, then open to remove noise
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

# Optional: find the largest connected component (the mold itself)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
if num_labels > 1:
    # Label 0 is background; find the largest foreground component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fg_mask = np.uint8(labels == largest_label) * 255

print("[OK] Background removed via HSV segmentation")

# Apply the mask to isolate the mold (black background)
isolated = cv2.bitwise_and(image, image, mask=fg_mask)

# --------------------------------
# 4. Auto-crop to the mold bounding box (full mold, no hardcoded ROI)
# --------------------------------

coords = cv2.findNonZero(fg_mask)
bx, by, bw, bh = cv2.boundingRect(coords)

# Add a small padding
pad = 10
bx = max(bx - pad, 0)
by = max(by - pad, 0)
bw = min(bw + 2 * pad, image.shape[1] - bx)
bh = min(bh + 2 * pad, image.shape[0] - by)

roi = isolated[by:by+bh, bx:bx+bw]
roi_mask = fg_mask[by:by+bh, bx:bx+bw]

print(f"[OK] Auto-cropped to mold region: {bw} x {bh} (full mold)")

# --------------------------------
# 5. Reflection / glare removal using CLAHE
# --------------------------------
# CLAHE equalises contrast locally, reducing bright glare spots

lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_corrected = clahe.apply(l_channel)

lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
roi_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

# Mask out background again after color-space round-trip
roi_corrected = cv2.bitwise_and(roi_corrected, roi_corrected, mask=roi_mask)

print("[OK] Reflections reduced via CLAHE")

# --------------------------------
# 6. Noise and dirt removal
# --------------------------------
# Bilateral filter preserves edges while smoothing noise/dirt
# Then morphological opening removes small remaining specks

denoised = cv2.bilateralFilter(roi_corrected, d=9, sigmaColor=75, sigmaSpace=75)

# Additional median blur to remove salt-and-pepper dirt
denoised = cv2.medianBlur(denoised, 5)

print("[OK] Noise and dirt removed")

# --------------------------------
# 7. Convert to grayscale
# --------------------------------

gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

# Mask background to pure black so it doesn't interfere with thresholding
gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

# --------------------------------
# 8. Edge detection
# --------------------------------

# Additional Gaussian blur for smoother edges
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 30, 120)

# Mask edges to foreground only
edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

# --------------------------------
# 9. Adaptive thresholding (better for uneven lighting on the mold)
# --------------------------------

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=35,
    C=12
)

# Remove background from threshold result
thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

# Morphological cleaning on threshold: remove tiny noise spots and close gaps
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)

# --------------------------------
# 9b. Create eroded inner mask to reject contours near edges / outside wax
# --------------------------------
# Erode the foreground mask heavily so only the interior of the wax remains.
# Any contour whose centroid is outside this inner mask is an outlier.

kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
inner_mask = cv2.erode(roi_mask, kernel_erode, iterations=1)

# --------------------------------
# 10. Find contours on thresholded image
# --------------------------------

contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter: keep only contours that are INSIDE the wax mold
good_contours = []
image_area = bw * bh

for contour in contours:
    area = cv2.contourArea(contour)

    # Skip very small noise (< 300px) and very large blobs (> 3% of image)
    if area <= 300 or area >= (image_area * 0.03):
        continue

    # Compute centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2

    # REJECT if centroid is outside the eroded inner mask (i.e. outside the wax)
    if cy < 0 or cy >= inner_mask.shape[0] or cx < 0 or cx >= inner_mask.shape[1]:
        continue
    if inner_mask[cy, cx] == 0:
        continue

    good_contours.append(contour)

print("Total contours:", len(contours))
print("Contours inside wax (after outlier removal):", len(good_contours))

# --------------------------------
# 11. Draw bounding boxes
# --------------------------------

output = roi.copy()

for i, contour in enumerate(good_contours):
    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        str(i + 1),
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# --------------------------------
# 12. Feature extraction (per impression candidate)
# --------------------------------
# NOTE: These are detected contour/impression CANDIDATES.
# They have not been validated as genuine tooth impressions yet.
# Visual validation against 08_contours.jpg is required.

impression_features = []

for i, contour in enumerate(good_contours):
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Aspect ratio
    aspect_ratio = round(w / h, 3) if h > 0 else 0

    # Extent = contour area / bounding rect area
    rect_area = w * h
    extent = round(area / rect_area, 3) if rect_area > 0 else 0

    # Solidity = contour area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = round(area / hull_area, 3) if hull_area > 0 else 0

    # Centroid from moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Hu Moments (7 shape invariants)
    hu = cv2.HuMoments(M).flatten()

    impression_features.append({
        "id": i + 1,
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "centroid_x": cx,
        "centroid_y": cy,
        "hu1": hu[0],
        "hu2": hu[1],
        "hu3": hu[2],
        "hu4": hu[3],
        "hu5": hu[4],
        "hu6": hu[5],
        "hu7": hu[6]
    })

print("\n--- Per-impression candidate features ---")
for f in impression_features:
    print(f"  Candidate {f['id']}: area={f['area']}, "
          f"perimeter={f['perimeter']}, "
          f"size={f['width']}x{f['height']}, "
          f"aspect={f['aspect_ratio']}, "
          f"solidity={f['solidity']}, "
          f"extent={f['extent']}, "
          f"centroid=({f['centroid_x']}, {f['centroid_y']})")

# --------------------------------
# 13. Spatial features (inter-impression distances)
# --------------------------------

# Sort impression candidates left-to-right by centroid x
sorted_impressions = sorted(impression_features, key=lambda f: f["centroid_x"])

distances = []
for j in range(len(sorted_impressions) - 1):
    a = sorted_impressions[j]
    b = sorted_impressions[j + 1]
    dist = math.sqrt(
        (b["centroid_x"] - a["centroid_x"]) ** 2 +
        (b["centroid_y"] - a["centroid_y"]) ** 2
    )
    distances.append(round(dist, 2))

avg_spacing = round(sum(distances) / len(distances), 2) if distances else 0

print(f"\n--- Spatial features ---")
print(f"  Inter-impression distances: {distances}")
print(f"  Average spacing: {avg_spacing}")

# --------------------------------
# 14. Aggregate bite-mark features for CSV
# --------------------------------

# Overall bite dimensions from all centroids
if len(sorted_impressions) >= 2:
    all_cx = [f["centroid_x"] for f in impression_features]
    all_cy = [f["centroid_y"] for f in impression_features]
    bite_width = max(all_cx) - min(all_cx)
    bite_height = max(all_cy) - min(all_cy)
else:
    bite_width = 0
    bite_height = 0

total_area = sum(f["area"] for f in impression_features)
total_perimeter = sum(f["perimeter"] for f in impression_features)
avg_solidity = round(
    sum(f["solidity"] for f in impression_features) / len(impression_features), 3
) if impression_features else 0
avg_aspect = round(
    sum(f["aspect_ratio"] for f in impression_features) / len(impression_features), 3
) if impression_features else 0

sample_id = os.path.splitext(os.path.basename(image_path))[0]

summary_row = {
    "sample_id": sample_id,
    "impression_count": len(good_contours),
    "bite_width": bite_width,
    "bite_height": bite_height,
    "avg_spacing": avg_spacing,
    "total_area": round(total_area, 2),
    "total_perimeter": round(total_perimeter, 2),
    "avg_aspect_ratio": avg_aspect,
    "avg_solidity": avg_solidity
}

# Add individual spacing distances (spacing_1_2, spacing_2_3, ...)
for j, dist in enumerate(distances):
    summary_row[f"spacing_{j+1}_{j+2}"] = dist

print(f"\n--- Summary for {sample_id} ---")
for key, val in summary_row.items():
    print(f"  {key}: {val}")

# Write CSV
csv_path = "results/features.csv"
file_exists = os.path.exists(csv_path)

with open(csv_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=summary_row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(summary_row)

print(f"\n[OK] Features appended to {csv_path}")

# --------------------------------
# 15. Save per-impression details CSV
# --------------------------------

detail_csv_path = f"results/{sample_id}_impressions.csv"

with open(detail_csv_path, "w", newline="") as csvfile:
    fieldnames = ["id", "area", "perimeter", "width", "height",
                  "aspect_ratio", "solidity", "extent", "centroid_x", "centroid_y",
                  "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for f in impression_features:
        writer.writerow(f)

print(f"[OK] Per-impression details saved to {detail_csv_path}")

# --------------------------------
# 16. Save all processing stages
# --------------------------------

cv2.imwrite("results/01_original.jpg", image)
cv2.imwrite("results/02_background_mask.jpg", fg_mask)
cv2.imwrite("results/03_isolated.jpg", isolated)
cv2.imwrite("results/04_roi_autocropped.jpg", roi)
cv2.imwrite("results/05_clahe_corrected.jpg", roi_corrected)
cv2.imwrite("results/06_denoised.jpg", denoised)
cv2.imwrite("results/07_gray.jpg", gray)
cv2.imwrite("results/08_edges.jpg", edges)
cv2.imwrite("results/09_threshold.jpg", thresh)
cv2.imwrite("results/09b_inner_mask.jpg", inner_mask)
cv2.imwrite("results/10_contours.jpg", output)

print("[OK] Results saved to results/ folder")

# --------------------------------
# 17. Display results (resized)
# --------------------------------

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Background Removed", cv2.WINDOW_NORMAL)
cv2.namedWindow("CLAHE Corrected", cv2.WINDOW_NORMAL)
cv2.namedWindow("Denoised", cv2.WINDOW_NORMAL)
cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detected Impressions", cv2.WINDOW_NORMAL)

cv2.imshow("Original", resize_for_display(image))
cv2.imshow("Background Removed", resize_for_display(isolated))
cv2.imshow("CLAHE Corrected", resize_for_display(roi_corrected))
cv2.imshow("Denoised", resize_for_display(denoised))
cv2.imshow("Edges", resize_for_display(edges))
cv2.imshow("Threshold", resize_for_display(thresh))
cv2.imshow("Detected Impressions", resize_for_display(output))

print("\n[INFO] Press any key on the OpenCV windows to close them.")
cv2.waitKey(0)
cv2.destroyAllWindows()