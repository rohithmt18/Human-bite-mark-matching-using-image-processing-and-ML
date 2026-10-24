"""
hybrid_matching.py

Hybrid bite-mark matching using:
    1. Traditional handcrafted features
    2. CoAtNet deep features

Final score:
    70% Traditional
    30% CoAtNet
"""

import csv
import os
import hashlib
import numpy as np

from preprocessing import process_single_image
from matching_engine import (
    load_database,
    compute_normalisation_params,
    compute_similarity
)
from coatnet_extractor import extract_coatnet_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

TRADITIONAL_CSV = os.path.join(
    BASE_DIR,
    "results",
    "dataset_features.csv"
)

COATNET_CSV = os.path.join(
    BASE_DIR,
    "results",
    "coatnet_features.csv"
)

TRADITIONAL_WEIGHT = 0.80
COATNET_WEIGHT = 0.20


# ---------------------------------------------------------
# Cosine similarity (Normalized to [0.0, 1.0])
# ---------------------------------------------------------

def cosine_similarity(a, b):
    """
    Calculate normalized cosine similarity between two feature vectors.
    Mathematically maps raw cosine in [-1.0, 1.0] onto [0.0, 1.0]
    using the standard affine normalization: (cos + 1.0) / 2.0.
    """

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    raw_cos = float(np.dot(a, b) / denominator)
    return max(0.0, min(1.0, (raw_cos + 1.0) / 2.0))


# ---------------------------------------------------------
# Load CoAtNet database
# ---------------------------------------------------------

def load_coatnet_database(csv_path):
    """
    Load CoAtNet feature database.

    Each row contains:
        subject_id
        sample_id
        feature_1 ... feature_768
    """

    database = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            features = np.array(
                [
                    float(row[f"feature_{i}"])
                    for i in range(1, 769)
                ],
                dtype=np.float32
            )

            database.append({
                "subject_id": row["subject_id"],
                "sample_id": row["sample_id"],
                "features": features
            })

    return database


# ---------------------------------------------------------
# Find CoAtNet features
# ---------------------------------------------------------

def get_coatnet_features(
    coatnet_database,
    subject_id,
    sample_id
):
    """Find CoAtNet feature vector for a database sample."""

    for row in coatnet_database:

        if (
            row["subject_id"] == subject_id
            and row["sample_id"] == sample_id
        ):
            return row["features"]

    return None


def file_hash(file_path):
    """Return SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    except (FileNotFoundError, OSError):
        return None

# ---------------------------------------------------------
# Hybrid matching
# ---------------------------------------------------------

def hybrid_match_query_image(
    image_path,
    original_filename=None,
    top_n=10
):
    """
    Complete hybrid matching pipeline.

    Returns:
        summary
        ranked results
        impression features
        contours
    """

    # -----------------------------------------------------
    # Check databases
    # -----------------------------------------------------

    if not os.path.isfile(TRADITIONAL_CSV):
        raise FileNotFoundError(
            f"Traditional database not found: {TRADITIONAL_CSV}"
        )

    if not os.path.isfile(COATNET_CSV):
        raise FileNotFoundError(
            f"CoAtNet database not found: {COATNET_CSV}"
        )

    # -----------------------------------------------------
    # Process uploaded image
    # -----------------------------------------------------

    summary, imp_features, distances, contours = (
        process_single_image(
            image_path,
            output_dir=None
        )
    )

    if summary is None:
        return None, [], [], []

    # -----------------------------------------------------
    # Set query sample ID
    # -----------------------------------------------------

    if original_filename:
        summary["sample_id"] = os.path.splitext(
            os.path.basename(original_filename)
        )[0]
    else:
        summary["sample_id"] = "query"

    # -----------------------------------------------------
    # Load databases
    # -----------------------------------------------------

    traditional_database = load_database(
        TRADITIONAL_CSV
    )

    coatnet_database = load_coatnet_database(
        COATNET_CSV
    )

    # -----------------------------------------------------
    # Extract CoAtNet features from query
    # -----------------------------------------------------

    query_coatnet = extract_coatnet_features(
        image_path
    )

    # -----------------------------------------------------
    # Traditional normalization
    # -----------------------------------------------------

    norm_params = compute_normalisation_params(
        traditional_database + [summary]
    )

    ranked = []

    # -----------------------------------------------------
    # Calculate hash of uploaded query image
    # -----------------------------------------------------

    query_hash = file_hash(image_path)

    # -----------------------------------------------------
    # Compare against every database image
    # -----------------------------------------------------

    for candidate in traditional_database:

        candidate_subject = str(
            candidate.get("subject_id", "")
        )

        candidate_sample = str(
            candidate.get("sample_id", "")
        )

        # -------------------------------------------------
        # Find corresponding dataset image
        # -------------------------------------------------

        candidate_image_path = None

        candidate_folder = os.path.join(
            DATASETS_DIR,
            candidate_subject
        )

        if os.path.isdir(candidate_folder):

            for fname in os.listdir(candidate_folder):

                stem, ext = os.path.splitext(fname)

                if (
                    stem.strip().lower()
                    == candidate_sample.strip().lower()
                    and ext.lower() in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff"
                    }
                ):
                    candidate_image_path = os.path.join(
                        candidate_folder,
                        fname
                    )
                    break

        # -------------------------------------------------
        # Skip exact uploaded image using file hash or sample_id
        # -------------------------------------------------

        candidate_hash = None

        if candidate_image_path:
            candidate_hash = file_hash(
                candidate_image_path
            )

        is_exact_match = False
        if (
            query_hash is not None
            and candidate_hash is not None
            and query_hash == candidate_hash
        ):
            is_exact_match = True
        elif original_filename:
            orig_stem = os.path.splitext(os.path.basename(original_filename))[0].strip().lower()
            if orig_stem == candidate_sample.strip().lower():
                is_exact_match = True

        if is_exact_match:
            print(
                f"[Hybrid] Skipping exact query image: "
                f"{candidate_subject}/{candidate_sample}"
            )
            continue

        # -------------------------------------------------
        # Traditional similarity
        # -------------------------------------------------

        traditional_score, traditional_breakdown = (
            compute_similarity(
                summary,
                candidate,
                norm_params
            )
        )

        # -------------------------------------------------
        # CoAtNet similarity
        # -------------------------------------------------

        candidate_coatnet = get_coatnet_features(
            coatnet_database,
            candidate_subject,
            candidate_sample
        )

        if candidate_coatnet is None:
            continue

        coatnet_score = (
            cosine_similarity(
                query_coatnet,
                candidate_coatnet
            ) * 100
        )

        # -------------------------------------------------
        # Hybrid score
        # -------------------------------------------------

        hybrid_score = (
            TRADITIONAL_WEIGHT * traditional_score
            +
            COATNET_WEIGHT * coatnet_score
        )

        # -------------------------------------------------
        # Combined breakdown
        # -------------------------------------------------

        breakdown = {
            "structural": float(round(traditional_breakdown["structural"], 2)),
            "hu_moments": float(round(traditional_breakdown["hu_moments"], 2)),
            "spacing": float(round(traditional_breakdown["spacing"], 2)),
            "count": float(round(traditional_breakdown["count"], 2)),
            "traditional": float(round(traditional_score, 2)),
            "coatnet": float(round(coatnet_score, 2)),
            "composite": float(round(hybrid_score, 2)),
        }

        ranked.append(
            (
                candidate,
                round(hybrid_score, 2),
                breakdown
            )
        )

    # -----------------------------------------------------
    # Sort highest first
    # -----------------------------------------------------

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------------------------
    # Return top N
    # -----------------------------------------------------

    if top_n:
        ranked = ranked[:top_n]

    return (
        summary,
        ranked,
        imp_features,
        contours
    )