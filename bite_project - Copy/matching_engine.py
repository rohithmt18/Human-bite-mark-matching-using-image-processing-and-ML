"""
matching_engine.py — Multi-metric similarity scoring for bite-mark forensics.

Given a query image (or pre-extracted feature vector), compare against the
stored subject database and return ranked suspect matches with composite
similarity scores (0–100%).

Metrics used:
    1. Euclidean distance on normalised structural features
    2. Cosine similarity on averaged Hu-moment vectors
    3. Pearson correlation on inter-tooth spacing sequences
    4. Impression-count proximity bonus
"""

import csv
import math
import os
import numpy as np

from preprocessing import process_single_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCTURAL_KEYS = [
    "impression_count", "bite_width", "bite_height",
    "avg_spacing", "total_area", "total_perimeter",
    "avg_aspect_ratio", "avg_solidity",
]

HU_KEYS = [f"avg_hu{k}" for k in range(1, 8)]

# Weights for the composite score
W_STRUCTURAL = 0.35
W_HU = 0.30
W_SPACING = 0.25
W_COUNT = 0.10


# ---------------------------------------------------------------------------
# Load database
# ---------------------------------------------------------------------------

def safe_float(val, default=0.0):
    """
    Safely convert any value to float.
    Returns `default` for empty strings, None, or non-numeric strings.
    """
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_database(csv_path):
    """Load dataset_features.csv into a list of dicts (string values → float)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, val in row.items():
                try:
                    parsed[key] = float(val)
                except (ValueError, TypeError):
                    parsed[key] = val  # keep strings (sample_id, subject_id etc.)
            rows.append(parsed)
    return rows


def _get_spacing_vector(row):
    """Extract the spacing_X_Y values as an ordered list."""
    spacings = []
    i = 1
    while True:
        key = f"spacing_{i}_{i + 1}"
        if key in row and row[key] != "" and row[key] is not None:
            try:
                spacings.append(float(row[key]))
            except (ValueError, TypeError):
                break
        else:
            break
        i += 1
    return spacings


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalisation_params(database):
    """Compute min/max for each structural feature across the database."""
    params = {}
    for key in STRUCTURAL_KEYS:
        vals = [row.get(key, 0) for row in database
                if isinstance(row.get(key, 0), (int, float))]
        if vals:
            params[key] = (min(vals), max(vals))
        else:
            params[key] = (0, 1)
    return params


def normalise_vector(feature_dict, params):
    """Min-max normalise structural features to [0, 1]."""
    vec = []
    for key in STRUCTURAL_KEYS:
        val = safe_float(feature_dict.get(key, 0))
        lo, hi = params[key]
        rng = hi - lo
        if rng == 0:
            vec.append(0.0)
        else:
            vec.append((val - lo) / rng)
    return np.array(vec)


# ---------------------------------------------------------------------------
# Individual similarity metrics
# ---------------------------------------------------------------------------

def euclidean_similarity(vec_a, vec_b):
    """Convert Euclidean distance to a 0–1 similarity score."""
    dist = np.linalg.norm(vec_a - vec_b)
    # Max possible distance in unit hypercube = sqrt(n_dimensions)
    max_dist = math.sqrt(len(vec_a))
    return max(0.0, 1.0 - dist / max_dist)


def cosine_similarity(vec_a, vec_b):
    """Standard cosine similarity, clamped to [0, 1]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return max(0.0, min(1.0, (sim + 1) / 2))  # shift from [-1,1] to [0,1]


def spacing_correlation(spacing_a, spacing_b):
    """
    Robust spacing similarity for bite-mark impressions.

    Handles:
    - Different numbers of detected gaps
    - Different overall bite sizes
    - Small positional shifts in the spacing sequence

    Returns:
        Similarity score in the range [0, 1]
    """

    if not spacing_a or not spacing_b:
        return 0.5

    # Convert to numpy arrays
    a = np.asarray(spacing_a, dtype=float)
    b = np.asarray(spacing_b, dtype=float)

    # Remove invalid values
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return 0.5

    # ---------------------------------------------------------
    # 1. Normalize each spacing sequence by its mean
    # ---------------------------------------------------------
    a = a / (np.mean(a) + 1e-8)
    b = b / (np.mean(b) + 1e-8)

    # ---------------------------------------------------------
    # 2. Compare sequences using the best alignment
    # ---------------------------------------------------------
    best_corr = -1.0

    # Allow the shorter sequence to slide across
    # the longer sequence.
    if len(a) <= len(b):
        short = a
        long = b
    else:
        short = b
        long = a

    difference = len(long) - len(short)

    for shift in range(difference + 1):

        segment = long[shift:shift + len(short)]

        if len(segment) < 2:
            continue

        if np.std(short) == 0 or np.std(segment) == 0:
            continue

        corr = np.corrcoef(short, segment)[0, 1]

        if np.isfinite(corr):
            best_corr = max(best_corr, corr)

    # ---------------------------------------------------------
    # 3. Also compare the full sequences after interpolation
    # ---------------------------------------------------------
    target_length = max(len(a), len(b))

    x_a = np.linspace(0, 1, len(a))
    x_b = np.linspace(0, 1, len(b))
    x_target = np.linspace(0, 1, target_length)

    a_interp = np.interp(x_target, x_a, a)
    b_interp = np.interp(x_target, x_b, b)

    if np.std(a_interp) > 0 and np.std(b_interp) > 0:

        full_corr = np.corrcoef(
            a_interp,
            b_interp
        )[0, 1]

        if np.isfinite(full_corr):
            best_corr = max(best_corr, full_corr)

    # ---------------------------------------------------------
    # 4. Convert correlation [-1, 1] → [0, 1]
    # ---------------------------------------------------------
    if best_corr < -1:
        return 0.5

    similarity = (best_corr + 1) / 2

    return max(0.0, min(1.0, similarity))


def count_similarity(count_a, count_b):
    """Similarity based on impression count proximity."""
    if count_a == 0 and count_b == 0:
        return 1.0
    diff = abs(count_a - count_b)
    max_count = max(count_a, count_b, 1)
    return max(0.0, 1.0 - diff / max_count)


def log_transform_hu(feature_dict):
    """
    Apply standard sign-preserving log10 transform to Hu moments:
        h_norm = -sign(h) * log10(|h| + 1e-15)
    This brings all 7 moments (which span 1e-1 to 1e-7) onto comparable scales
    so that higher-order moments contribute meaningfully to similarity scoring.
    """
    h_vals = []
    for k in HU_KEYS:
        val = safe_float(feature_dict.get(k, 0.0))
        h_vals.append(-np.sign(val) * math.log10(abs(val) + 1e-15))
    return np.array(h_vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Composite similarity
# ---------------------------------------------------------------------------

def compute_similarity(query_row, db_row, norm_params):
    """
    Compute composite similarity between a query and a database entry.

    Returns
    -------
    score     : float in [0, 100]
    breakdown : dict with per-metric scores
    """
    # 1. Structural similarity (Euclidean on normalised features)
    q_struct = normalise_vector(query_row, norm_params)
    d_struct = normalise_vector(db_row, norm_params)
    struct_sim = euclidean_similarity(q_struct, d_struct)

    # 2. Hu-moment similarity (Cosine on log-transformed Hu moments)
    q_hu = log_transform_hu(query_row)
    d_hu = log_transform_hu(db_row)
    hu_sim = cosine_similarity(q_hu, d_hu)

    # 3. Spacing correlation
    q_spacing = _get_spacing_vector(query_row)
    d_spacing = _get_spacing_vector(db_row)
    sp_sim = spacing_correlation(q_spacing, d_spacing)

    # 4. Count proximity
    cnt_sim = count_similarity(
        int(safe_float(query_row.get("impression_count", 0))),
        int(safe_float(db_row.get("impression_count", 0)))
    )

    # Weighted composite
    composite = (
        W_STRUCTURAL * struct_sim +
        W_HU * hu_sim +
        W_SPACING * sp_sim +
        W_COUNT * cnt_sim
    )

    score = round(composite * 100, 2)

    breakdown = {
        "structural": float(round(struct_sim * 100, 2)),
        "hu_moments": float(round(hu_sim * 100, 2)),
        "spacing": float(round(sp_sim * 100, 2)),
        "count": float(round(cnt_sim * 100, 2)),
        "composite": float(score),
    }

    return float(score), breakdown


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_matches(query_row, database, top_n=None, exclude_sample_id=None):
    """
    Rank all database entries by similarity to the query.

    Returns sorted list of (db_row, score, breakdown), highest first.
    """
    norm_params = compute_normalisation_params(database + [query_row])

    results = []

    for db_row in database:

        # Don't match the exact query sample
        if exclude_sample_id:
            db_sample_id = str(db_row.get("sample_id", "")).strip().lower()
            query_sample_id = str(exclude_sample_id).strip().lower()

            if db_sample_id == query_sample_id:
                continue

        score, breakdown = compute_similarity(
            query_row, db_row, norm_params
        )

        results.append((db_row, score, breakdown))

    results.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        results = results[:top_n]

    return results



# ---------------------------------------------------------------------------
# End-to-end: image path → ranked matches
# ---------------------------------------------------------------------------
def match_query_image(image_path, database_csv, original_filename=None):
    """
    Full pipeline: process a query image, extract features, rank against DB.
    """

    # Process the query image
    summary, imp_features, distances, contours = process_single_image(
        image_path, output_dir=None
    )

    if summary is None:
        return None, [], [], []

    # Use the original uploaded filename as sample_id.
    # This allows the matching engine to exclude the exact same
    # database sample from the results.
    if original_filename:
        summary["sample_id"] = os.path.splitext(
            os.path.basename(original_filename)
        )[0]

    # Load database
    database = load_database(database_csv)

    # Rank
    ranked = rank_matches(summary, database)

    return summary, ranked, imp_features, contours


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python matching_engine.py <query_image_path> [database_csv]")
        print("Example: python matching_engine.py datasets/P001/Rohith\\ lower.jpg")
        sys.exit(1)

    query_path = sys.argv[1]
    db_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "dataset_features.csv"
    )

    if not os.path.isfile(db_csv):
        print(f"[ERROR] Database CSV not found: {db_csv}")
        print("Run batch_process.py first to build the database.")
        sys.exit(1)

    print(f"=== Bite-Mark Matching Engine ===")
    print(f"Query: {query_path}")
    print(f"Database: {db_csv}\n")

    summary, ranked, imp_features, contours = match_query_image(
        query_path, db_csv
    )

    if summary is None:
        print("[ERROR] Could not process query image.")
        sys.exit(1)

    print(f"Query: {summary['sample_id']}")
    print(f"  Impressions detected: {summary['impression_count']}")
    print(f"  Bite dimensions: {summary['bite_width']} x {summary['bite_height']}")
    print()

    print(f"--- Top Matches ---")
    for i, (db_row, score, breakdown) in enumerate(ranked[:10], 1):
        sid = db_row.get("subject_id", "?")
        name = db_row.get("sample_id", "?")
        print(f"  #{i}  {sid}/{name}")
        print(f"       Composite Score: {score}%")
        print(f"       Structural: {breakdown['structural']}%  "
              f"Hu Moments: {breakdown['hu_moments']}%  "
              f"Spacing: {breakdown['spacing']}%  "
              f"Count: {breakdown['count']}%")
        print()


if __name__ == "__main__":
    main()
