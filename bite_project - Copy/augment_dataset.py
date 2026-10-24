"""
augment_dataset.py — Data augmentation for bite-mark dental mold images.

For each original image in datasets/<SubjectID>/ this script generates
augmented variants specifically designed for forensic bite-mark matching:

    1.  Rotation          ±5°, ±10°, ±15°     (dental molds may be placed at angles)
    2.  Horizontal flip                         (upper/lower arch symmetry variants)
    3.  Brightness shift  ±20%, ±35%           (different lighting conditions)
    4.  Gaussian noise    light / medium        (camera sensor noise)
    5.  Mild blur                               (slightly out-of-focus shots)
    6.  CLAHE contrast variant                  (high/low contrast conditions)
    7.  Scale crop        90%, 95%              (different distances from camera)

After running this script, re-run:
    python batch_process.py
    python coatnet_database.py
to rebuild the feature databases with the augmented data.

Usage:
    python augment_dataset.py
    python augment_dataset.py --dry-run        # list files that would be created
    python augment_dataset.py --clean          # remove all previously augmented files
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Prefix added to all generated filenames so we can identify / clean them later
AUG_PREFIX = "aug_"

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Individual augmentation functions
# All functions accept a BGR numpy array and return a BGR numpy array.
# ---------------------------------------------------------------------------


def rotate_image(image, angle_deg):
    """Rotate image around its centre, keeping the original canvas size."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def flip_horizontal(image):
    """Mirror left <-> right."""
    return cv2.flip(image, 1)


def adjust_brightness(image, factor):
    """
    Multiply pixel values by `factor` (float).
    factor > 1.0  -> brighter
    factor < 1.0  -> darker
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(image, std):
    """Add zero-mean Gaussian noise with given standard deviation."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def mild_blur(image, ksize=3):
    """Apply a gentle Gaussian blur to simulate slight defocus."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def clahe_variant(image, clip_limit):
    """Apply CLAHE at a different clip limit to create contrast variant."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def scale_crop(image, scale):
    """
    Zoom in by `scale` factor (e.g. 0.90) and resize back to original size.
    Simulates the camera being closer / further from the mold.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Crop from centre
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    cropped = image[y0:y0 + new_h, x0:x0 + new_w]

    # Resize back to original dimensions
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Build augmentation plan
# ---------------------------------------------------------------------------

def build_augmentations(image):
    """
    Return a list of (suffix, augmented_image) tuples for a given source image.
    """
    augmentations = []

    # --- Rotations ---
    for angle in [-15, -10, -5, 5, 10, 15]:
        sign = "m" if angle < 0 else "p"
        aug = rotate_image(image, angle)
        augmentations.append((f"rot{sign}{abs(angle)}", aug))

    # --- Horizontal flip ---
    augmentations.append(("hflip", flip_horizontal(image)))

    # --- Brightness ---
    for factor, label in [(0.65, "dark35"), (0.80, "dark20"),
                          (1.20, "bright20"), (1.35, "bright35")]:
        augmentations.append((f"bright_{label}", adjust_brightness(image, factor)))

    # --- Gaussian noise ---
    for std, label in [(8, "noise_lt"), (18, "noise_md")]:
        augmentations.append((label, add_gaussian_noise(image, std)))

    # --- Mild blur ---
    augmentations.append(("blur3", mild_blur(image, ksize=3)))

    # --- CLAHE variants ---
    for clip, label in [(1.5, "clahe_lo"), (5.0, "clahe_hi")]:
        augmentations.append((label, clahe_variant(image, clip)))

    # --- Scale crop ---
    for scale, label in [(0.90, "scale90"), (0.95, "scale95")]:
        augmentations.append((label, scale_crop(image, scale)))

    return augmentations


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def augment_image(src_path, dry_run=False):
    """
    Generate and save all augmentations for a single source image.
    Returns (generated_count, skipped_count).
    """
    stem, ext = os.path.splitext(os.path.basename(src_path))
    out_dir    = os.path.dirname(src_path)
    save_ext   = ".jpg"   # always save as JPEG to save space

    image = cv2.imread(src_path)
    if image is None:
        print(f"  [WARN] Cannot read image: {src_path}")
        return 0, 0

    augmentations = build_augmentations(image)
    generated = 0
    skipped   = 0

    for suffix, aug_img in augmentations:
        out_name = f"{AUG_PREFIX}{stem}_{suffix}{save_ext}"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would create: {out_name}")
            generated += 1
            continue

        ok = cv2.imwrite(out_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            generated += 1
        else:
            print(f"  [ERROR] Failed to write: {out_path}")

    return generated, skipped


# ---------------------------------------------------------------------------
# Clean previously generated augmented files
# ---------------------------------------------------------------------------

def clean_augmented(datasets_dir):
    """Remove all files whose name starts with AUG_PREFIX."""
    removed = 0
    for root, dirs, files in os.walk(datasets_dir):
        for fname in files:
            if fname.startswith(AUG_PREFIX):
                fpath = os.path.join(root, fname)
                os.remove(fpath)
                print(f"  Removed: {fpath}")
                removed += 1
    print(f"\nRemoved {removed} augmented files.")


# ---------------------------------------------------------------------------
# Discover original (non-augmented) images
# ---------------------------------------------------------------------------

def discover_originals(datasets_dir):
    """Yield paths to original (non-augmented) images only."""
    for root, dirs, files in os.walk(datasets_dir):
        dirs.sort()
        for fname in sorted(files):
            if fname.startswith(AUG_PREFIX):
                continue   # skip previously generated files
            if fname.lower().startswith("copy of"):
                continue   # skip "Copy of" duplicates
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue
            yield os.path.join(root, fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment bite-mark dataset images to improve matching accuracy."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be created without writing them."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove all previously generated augmented files and exit."
    )
    args = parser.parse_args()

    if args.clean:
        print(f"=== Cleaning augmented files from: {DATASETS_DIR}\n")
        clean_augmented(DATASETS_DIR)
        return

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"=== Bite-Mark Dataset Augmentation [{mode}] ===")
    print(f"Dataset dir : {DATASETS_DIR}")
    print(f"Aug prefix  : '{AUG_PREFIX}'\n")

    originals = list(discover_originals(DATASETS_DIR))
    if not originals:
        print("[ERROR] No original images found. Check DATASETS_DIR.")
        sys.exit(1)

    print(f"Found {len(originals)} original image(s) to augment.\n")

    total_generated = 0
    total_skipped   = 0

    for src_path in originals:
        rel = os.path.relpath(src_path, DATASETS_DIR)
        print(f"  {rel}")
        gen, skip = augment_image(src_path, dry_run=args.dry_run)
        total_generated += gen
        total_skipped   += skip
        print(f"    -> {gen} generated, {skip} already exist")

    print(f"\n=== Augmentation Complete ===")
    print(f"  Total generated : {total_generated}")
    print(f"  Already existed : {total_skipped}")

    if not args.dry_run and total_generated > 0:
        print("\n Next steps to rebuild the feature database:")
        print("   1.  python batch_process.py")
        print("   2.  python coatnet_database.py")
        print("   Then upload a query image in the app to see improved matching.\n")


if __name__ == "__main__":
    main()
