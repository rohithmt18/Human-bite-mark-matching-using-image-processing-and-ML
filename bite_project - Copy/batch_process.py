"""
batch_process.py — Process all dental mold images across P001-P012.

Walks the datasets/ directory, runs the preprocessing pipeline on each image,
and writes consolidated feature CSVs to results/.
"""

import csv
import os
import sys

from preprocessing import process_single_image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Aggregate CSV for all samples
AGGREGATE_CSV = os.path.join(RESULTS_DIR, "dataset_features.csv")

# Skip duplicate "Copy of" files
SKIP_PREFIXES = ("Copy of",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_images(datasets_dir):
    """
    Walk datasets/ and yield (subject_id, image_path) tuples.
    Skips files starting with SKIP_PREFIXES.
    Only processes .jpg / .jpeg / .png files.
    """
    valid_exts = {".jpg", ".jpeg", ".png"}

    for subject_folder in sorted(os.listdir(datasets_dir)):
        subject_path = os.path.join(datasets_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for fname in sorted(os.listdir(subject_path)):
            # Skip duplicates
            if any(fname.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue

            yield subject_folder, os.path.join(subject_path, fname)


def write_aggregate_csv(all_summaries, csv_path):
    """Write all summary rows to a single CSV, using a union of all keys."""
    if not all_summaries:
        print("[WARN] No summaries to write.")
        return

    # Collect all possible fieldnames (union of all dicts)
    fieldnames = []
    seen = set()
    for row in all_summaries:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_summaries:
            writer.writerow(row)


def write_impression_csv(imp_features, csv_path):
    """Write per-impression details to a CSV."""
    if not imp_features:
        return
    fieldnames = [
        "id", "area", "perimeter", "width", "height",
        "aspect_ratio", "solidity", "extent", "circularity",
        "centroid_x", "centroid_y",
        "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in imp_features:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    images = list(discover_images(DATASETS_DIR))
    total = len(images)

    print(f"=== Bite-Mark Batch Processor ===")
    print(f"Found {total} images across datasets/\n")

    all_summaries = []
    success = 0
    failed = 0

    for idx, (subject_id, image_path) in enumerate(images, 1):
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"[{idx}/{total}] Processing {subject_id}/{sample_name} ...")

        # Per-subject output directory
        output_dir = os.path.join(RESULTS_DIR, subject_id, sample_name)

        summary, imp_features, distances, contours = process_single_image(
            image_path, output_dir=output_dir
        )

        if summary is None:
            failed += 1
            continue

        # Add subject_id to summary for traceability
        summary["subject_id"] = subject_id

        all_summaries.append(summary)
        success += 1

        # Save per-impression CSV
        imp_csv = os.path.join(
            RESULTS_DIR, subject_id, f"{sample_name}_impressions.csv"
        )
        os.makedirs(os.path.dirname(imp_csv), exist_ok=True)
        write_impression_csv(imp_features, imp_csv)

        print(f"         -> {summary['impression_count']} impressions detected")

    # Write aggregate CSV
    write_aggregate_csv(all_summaries, AGGREGATE_CSV)

    print(f"\n=== Batch Processing Complete ===")
    print(f"  Success: {success}/{total}")
    print(f"  Failed:  {failed}/{total}")
    print(f"  Aggregate features -> {AGGREGATE_CSV}")


if __name__ == "__main__":
    main()
