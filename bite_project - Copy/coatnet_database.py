"""
coatnet_database.py

Build a CoAtNet deep-feature database for all bite-mark images
inside the datasets/ directory.

Output:
    results/coatnet_features.csv

Each image gets:
    - subject_id
    - sample_id
    - 768 CoAtNet feature values
"""

import csv
import os

from coatnet_extractor import extract_coatnet_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "coatnet_features.csv"
)

SKIP_PREFIXES = ("Copy of",)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
}


# ---------------------------------------------------------
# Discover dataset images
# ---------------------------------------------------------

def discover_images():

    images = []

    for subject_folder in sorted(os.listdir(DATASETS_DIR)):

        subject_path = os.path.join(
            DATASETS_DIR,
            subject_folder
        )

        if not os.path.isdir(subject_path):
            continue

        for filename in sorted(os.listdir(subject_path)):

            if any(
                filename.startswith(prefix)
                for prefix in SKIP_PREFIXES
            ):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in VALID_EXTENSIONS:
                continue

            image_path = os.path.join(
                subject_path,
                filename
            )

            images.append(
                (
                    subject_folder,
                    filename,
                    image_path
                )
            )

    return images


# ---------------------------------------------------------
# Build database
# ---------------------------------------------------------

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    images = discover_images()

    total = len(images)

    print()
    print("=" * 50)
    print("       CoAtNet Feature Database Builder")
    print("=" * 50)
    print()

    print(f"Found {total} images.")
    print(f"Device will be selected automatically by CoAtNet.")
    print()

    if total == 0:
        print("[ERROR] No dataset images found.")
        return

    rows = []

    success = 0
    failed = 0

    for index, (subject_id, filename, image_path) in enumerate(
        images,
        1
    ):

        sample_id = os.path.splitext(filename)[0]

        print(
            f"[{index}/{total}] "
            f"{subject_id}/{sample_id}"
        )

        try:

            features = extract_coatnet_features(
                image_path
            )

            if features.shape[0] != 768:

                print(
                    f"    [ERROR] Expected 768 features, "
                    f"got {features.shape[0]}"
                )

                failed += 1
                continue

            row = {
                "subject_id": subject_id,
                "sample_id": sample_id,
            }

            for i, value in enumerate(features, 1):

                row[f"feature_{i}"] = float(value)

            rows.append(row)

            success += 1

            print(
                f"    -> Feature vector: {features.shape}"
            )

        except Exception as e:

            failed += 1

            print(
                f"    [ERROR] {e}"
            )

    # -----------------------------------------------------
    # Write CSV
    # -----------------------------------------------------

    if not rows:

        print()
        print("[ERROR] No features were generated.")
        return

    fieldnames = [
        "subject_id",
        "sample_id"
    ]

    fieldnames.extend(
        f"feature_{i}"
        for i in range(1, 769)
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("       CoAtNet Database Complete")
    print("=" * 50)

    print(
        f"Success: {success}/{total}"
    )

    print(
        f"Failed:  {failed}/{total}"
    )

    print()
    print(
        f"Database saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print()


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()