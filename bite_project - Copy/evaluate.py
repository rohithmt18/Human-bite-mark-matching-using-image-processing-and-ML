"""
evaluate.py — Evaluation & cross-validation for the bite-mark matching system.

Performs leave-one-out cross-validation:
    For each sample in the database, treat it as a "query" and check
    whether the matching engine correctly identifies other samples from
    the same subject as top matches.

Reports:
    - Rank-1 accuracy  (correct subject is the #1 match)
    - Top-3 accuracy   (correct subject is in the top 3 matches)
    - Top-5 accuracy
    - Per-sample results table
    - Confusion-style analysis
"""

import csv
import os
import sys

from matching_engine import load_database, rank_matches, compute_normalisation_params


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_CSV = os.path.join(BASE_DIR, "results", "dataset_features.csv")
REPORT_PATH = os.path.join(BASE_DIR, "results", "evaluation_report.txt")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(db_csv=DB_CSV):
    """
    Leave-one-out cross-validation.

    For each sample in the DB, use it as a query against all OTHER samples.
    A match is "correct" if the top-ranked result comes from the same
    subject (e.g. same person, different jaw / different photo).
    """
    database = load_database(db_csv)
    n = len(database)

    if n < 2:
        print("[ERROR] Need at least 2 samples in the database.")
        return

    print(f"=== Bite-Mark Matching Evaluation ===")
    print(f"Database: {db_csv}")
    print(f"Samples:  {n}\n")

    results = []

    for i, query_row in enumerate(database):
        query_sid = query_row.get("subject_id", "")
        query_name = query_row.get("sample_id", "")

        # Build "other" set (everything except the query itself)
        others = [row for j, row in enumerate(database) if j != i]

        # Rank
        ranked = rank_matches(query_row, others)

        # Check if any of the top matches share the same subject_id
        top1_subject = ranked[0][0].get("subject_id", "") if ranked else ""
        top1_name = ranked[0][0].get("sample_id", "") if ranked else ""
        top1_score = ranked[0][1] if ranked else 0

        top3_subjects = [r[0].get("subject_id", "") for r in ranked[:3]]
        top5_subjects = [r[0].get("subject_id", "") for r in ranked[:5]]

        rank1_correct = (top1_subject == query_sid)
        top3_correct = (query_sid in top3_subjects)
        top5_correct = (query_sid in top5_subjects)

        # Find the actual rank of the correct subject
        correct_rank = None
        for rank_idx, (db_row, score, breakdown) in enumerate(ranked, 1):
            if db_row.get("subject_id", "") == query_sid:
                correct_rank = rank_idx
                break

        results.append({
            "query_subject": query_sid,
            "query_sample": query_name,
            "top1_match": f"{top1_subject}/{top1_name}",
            "top1_score": top1_score,
            "rank1_correct": rank1_correct,
            "top3_correct": top3_correct,
            "top5_correct": top5_correct,
            "correct_rank": correct_rank,
        })

        status = "Y" if rank1_correct else "N"
        print(f"  [{status}] {query_sid}/{query_name:20s} -> "
              f"Top-1: {top1_subject}/{top1_name} ({top1_score}%)  "
              f"Correct rank: {correct_rank or 'N/A'}")

    # Aggregate metrics
    rank1_acc = sum(1 for r in results if r["rank1_correct"]) / n * 100
    top3_acc = sum(1 for r in results if r["top3_correct"]) / n * 100
    top5_acc = sum(1 for r in results if r["top5_correct"]) / n * 100

    # Average correct rank
    valid_ranks = [r["correct_rank"] for r in results if r["correct_rank"] is not None]
    avg_rank = sum(valid_ranks) / len(valid_ranks) if valid_ranks else float('inf')

    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Samples evaluated:   {n}")
    print(f"  Rank-1 Accuracy:     {rank1_acc:.1f}%")
    print(f"  Top-3 Accuracy:      {top3_acc:.1f}%")
    print(f"  Top-5 Accuracy:      {top5_acc:.1f}%")
    print(f"  Avg Correct Rank:    {avg_rank:.2f}")
    print(f"{'=' * 50}")

    # Write report
    write_report(results, rank1_acc, top3_acc, top5_acc, avg_rank, n)

    return results, rank1_acc, top3_acc, top5_acc


def write_report(results, rank1_acc, top3_acc, top5_acc, avg_rank, n):
    """Save evaluation report to a text file."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  BITE-MARK FORENSIC MATCHING — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Samples evaluated:   {n}\n")
        f.write(f"Rank-1 Accuracy:     {rank1_acc:.1f}%\n")
        f.write(f"Top-3 Accuracy:      {top3_acc:.1f}%\n")
        f.write(f"Top-5 Accuracy:      {top5_acc:.1f}%\n")
        f.write(f"Avg Correct Rank:    {avg_rank:.2f}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'Query':30s} {'Top-1 Match':30s} {'Score':8s} {'R1':4s} {'Rank':5s}\n")
        f.write("-" * 60 + "\n")

        for r in results:
            r1 = "Y" if r["rank1_correct"] else "N"
            cr = str(r["correct_rank"]) if r["correct_rank"] else "N/A"
            query = f"{r['query_subject']}/{r['query_sample']}"
            f.write(f"{query:30s} {r['top1_match']:30s} {r['top1_score']:6.1f}% "
                    f"{r1:4s} {cr:5s}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n  Report saved -> {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DB_CSV

    if not os.path.isfile(csv_path):
        print(f"[ERROR] Database not found: {csv_path}")
        print("Run batch_process.py first to build the database.")
        sys.exit(1)

    run_evaluation(csv_path)
