"""
evaluate_coatnet.py

Evaluate pretrained CoAtNet on the complete bite-mark dataset.

For each image, compare it against all other images and check whether
the other impression from the same person is ranked highly.
"""

import os
import csv
import numpy as np

from coatnet_extractor import extract_coatnet_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
DATABASE_CSV = os.path.join(BASE_DIR, "results", "coatnet_features.csv")


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(a, b):
    """Calculate cosine similarity between two feature vectors."""

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ---------------------------------------------------------
# Load CoAtNet database
# ---------------------------------------------------------

def load_database(csv_path):

    database = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            features = np.array(
            [float(row[f"feature_{i}"]) for i in range(1, 769)],
            dtype=np.float32
       )

            database.append({
                "subject_id": row["subject_id"],
                "sample_id": row["sample_id"],
                "features": features
            })

    return database


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print()
    print("=" * 60)
    print("          CoAtNet Dataset Evaluation")
    print("=" * 60)

    print("\nLoading CoAtNet feature database...")

    database = load_database(DATABASE_CSV)

    print(f"Loaded {len(database)} images.")

    print("\nEvaluating all images...")

    correct = 0
    total = 0

    results = []

    for query in database:

        query_subject = query["subject_id"]
        query_sample = query["sample_id"]

        similarities = []

        for candidate in database:

            # Don't compare image with itself
            if (
                candidate["subject_id"] == query_subject
                and candidate["sample_id"] == query_sample
            ):
                continue

            similarity = cosine_similarity(
                query["features"],
                candidate["features"]
            )

            similarities.append({
                "subject_id": candidate["subject_id"],
                "sample_id": candidate["sample_id"],
                "similarity": similarity
            })

        # Highest similarity first
        similarities.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        top_match = similarities[0]

        is_correct = (
            top_match["subject_id"] == query_subject
        )

        if is_correct:
            correct += 1

        total += 1

        results.append({
            "query_subject": query_subject,
            "query_sample": query_sample,
            "top_subject": top_match["subject_id"],
            "top_sample": top_match["sample_id"],
            "similarity": top_match["similarity"],
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"

        print(
            f"{status} "
            f"{query_subject}/{query_sample} "
            f"→ "
            f"{top_match['subject_id']}/{top_match['sample_id']} "
            f"({top_match['similarity'] * 100:.2f}%)"
        )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    accuracy = (correct / total) * 100 if total else 0

    print()
    print("=" * 60)
    print("             CoAtNet Evaluation Result")
    print("=" * 60)

    print(f"\nTotal queries : {total}")
    print(f"Correct Top-1 : {correct}")
    print(f"Incorrect     : {total - correct}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()