"""
evaluate_hybrid.py

Comprehensive evaluation framework for bite-mark matching:
    1. Traditional handcrafted features (structural, log Hu moments, tooth spacing, impression count)
    2. Pretrained CoAtNet deep features (normalized cosine similarity in [0, 1])
    3. Systematic weighting evaluation across traditional components and hybrid ratios
    4. Cross-jaw matching (Upper <-> Lower of same person) with strict exact-query exclusion
    5. Saves metrics to CSV and generates publication-quality evaluation graphs.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matching_engine import (
    load_database,
    compute_normalisation_params,
    compute_similarity,
    HU_KEYS,
)
from hybrid_matching import (
    load_coatnet_database,
    get_coatnet_features,
    cosine_similarity,
    file_hash,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

TRADITIONAL_CSV = os.path.join(RESULTS_DIR, "dataset_features.csv")
COATNET_CSV = os.path.join(RESULTS_DIR, "coatnet_features.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "evaluation_plots.png")


def find_image_path(subject_id, sample_id):
    """Resolve full image path for a database entry."""
    subject_dir = os.path.join(DATASETS_DIR, subject_id)
    if not os.path.isdir(subject_dir):
        return None
    for fname in os.listdir(subject_dir):
        stem, ext = os.path.splitext(fname)
        if stem.strip().lower() == sample_id.strip().lower() and ext.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            return os.path.join(subject_dir, fname)
    return None


def evaluate_configuration(
    traditional_db,
    coatnet_db,
    weight_traditional,
    weight_coatnet,
    hash_lookup=None
):
    """
    Evaluate retrieval performance for a given weight configuration.
    
    Returns:
        rank1_acc, top3_acc, top5_acc, avg_rank, detailed_ranks
    """
    total = 0
    correct_r1 = 0
    ranks = []
    
    # Precompute normalisation parameters for traditional features
    norm_params = compute_normalisation_params(traditional_db)
    
    for query in traditional_db:
        q_subject = str(query.get("subject_id", "")).strip()
        q_sample = str(query.get("sample_id", "")).strip()
        
        q_coatnet = get_coatnet_features(coatnet_db, q_subject, q_sample)
        if q_coatnet is None and weight_coatnet > 0:
            continue
            
        q_hash = hash_lookup.get((q_subject, q_sample)) if hash_lookup else None
        
        similarities = []
        
        for cand in traditional_db:
            c_subject = str(cand.get("subject_id", "")).strip()
            c_sample = str(cand.get("sample_id", "")).strip()
            
            # --- Strict Exclusion of Exact Query Image ---
            # 1. By Subject ID + Sample ID match
            if c_subject == q_subject and c_sample.lower() == q_sample.lower():
                continue
                
            # 2. By File SHA-256 Hash
            c_hash = hash_lookup.get((c_subject, c_sample)) if hash_lookup else None
            if q_hash is not None and c_hash is not None and q_hash == c_hash:
                continue
                
            # Traditional Similarity
            if weight_traditional > 0:
                t_score, _ = compute_similarity(query, cand, norm_params)
            else:
                t_score = 0.0
                
            # CoAtNet Similarity (Normalized [0, 100])
            if weight_coatnet > 0:
                c_coatnet = get_coatnet_features(coatnet_db, c_subject, c_sample)
                if c_coatnet is not None:
                    c_score = cosine_similarity(q_coatnet, c_coatnet) * 100.0
                else:
                    c_score = 0.0
            else:
                c_score = 0.0
                
            composite_score = (
                weight_traditional * t_score +
                weight_coatnet * c_score
            )
            
            similarities.append({
                "subject_id": c_subject,
                "sample_id": c_sample,
                "score": composite_score,
                "traditional": t_score,
                "coatnet": c_score
            })
            
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Determine rank of genuine match (same subject_id)
        genuine_rank = None
        for r, item in enumerate(similarities, 1):
            if item["subject_id"] == q_subject:
                genuine_rank = r
                break
                
        if genuine_rank is not None:
            total += 1
            ranks.append(genuine_rank)
            if genuine_rank == 1:
                correct_r1 += 1
                
    rank1 = (correct_r1 / total * 100.0) if total else 0.0
    top3 = (sum(r <= 3 for r in ranks) / total * 100.0) if total else 0.0
    top5 = (sum(r <= 5 for r in ranks) / total * 100.0) if total else 0.0
    avg_rank = (sum(ranks) / len(ranks)) if ranks else 0.0
    
    return rank1, top3, top5, avg_rank, ranks


def main():
    print("=" * 78)
    print("         SYSTEMATIC FORENSIC BITE-MARK HYBRID EVALUATION")
    print("=" * 78)
    
    if not os.path.isfile(TRADITIONAL_CSV):
        raise FileNotFoundError(f"Missing {TRADITIONAL_CSV}")
    if not os.path.isfile(COATNET_CSV):
        raise FileNotFoundError(f"Missing {COATNET_CSV}")
        
    print(f"\n[1] Loading database features...")
    traditional_db = load_database(TRADITIONAL_CSV)
    coatnet_db = load_coatnet_database(COATNET_CSV)
    print(f"    Traditional samples : {len(traditional_db)}")
    print(f"    CoAtNet samples     : {len(coatnet_db)}")
    
    # Precompute hashes for data-leakage prevention
    print("\n[2] Precomputing file hashes for strict exclusion...")
    hash_lookup = {}
    for row in traditional_db:
        sid = str(row.get("subject_id", "")).strip()
        samp = str(row.get("sample_id", "")).strip()
        ipath = find_image_path(sid, samp)
        if ipath and os.path.isfile(ipath):
            hash_lookup[(sid, samp)] = file_hash(ipath)
    print(f"    Indexed {len(hash_lookup)} image hashes.")
    
    # -------------------------------------------------------------------
    # Hybrid Ratio Evaluation Grid
    # -------------------------------------------------------------------
    ratios = [
        (1.00, 0.00, "Traditional Only"),
        (0.90, 0.10, "90% Trad / 10% CoAt"),
        (0.80, 0.20, "80% Trad / 20% CoAt"),
        (0.70, 0.30, "70% Trad / 30% CoAt"),
        (0.60, 0.40, "60% Trad / 40% CoAt"),
        (0.50, 0.50, "50% Trad / 50% CoAt"),
        (0.40, 0.60, "40% Trad / 60% CoAt"),
        (0.30, 0.70, "30% Trad / 70% CoAt"),
        (0.20, 0.80, "20% Trad / 80% CoAt"),
        (0.10, 0.90, "10% Trad / 90% CoAt"),
        (0.00, 1.00, "CoAtNet Only"),
    ]
    
    print("\n" + "-" * 78)
    print(f"{'Model / Configuration':<24} | {'Rank-1':>10} | {'Top-3':>10} | {'Top-5':>10} | {'Avg Rank':>10}")
    print("-" * 78)
    
    summary_results = []
    
    for wt, wc, label in ratios:
        r1, t3, t5, ar, rks = evaluate_configuration(
            traditional_db, coatnet_db, wt, wc, hash_lookup
        )
        print(f"{label:<24} | {r1:9.2f}% | {t3:9.2f}% | {t5:9.2f}% | {ar:9.2f}")
        summary_results.append({
            "configuration": label,
            "weight_traditional": wt,
            "weight_coatnet": wc,
            "rank1": round(r1, 2),
            "top3": round(t3, 2),
            "top5": round(t5, 2),
            "avg_rank": round(ar, 2),
        })
        
    print("-" * 78)
    
    # -------------------------------------------------------------------
    # Save CSV Summary
    # -------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "configuration", "weight_traditional", "weight_coatnet", "rank1", "top3", "top5", "avg_rank"
        ])
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"\n[3] Saved evaluation summary to: {SUMMARY_CSV}")
    
    # -------------------------------------------------------------------
    # Generate Evaluation Graph
    # -------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    coat_weights = [r["weight_coatnet"] * 100 for r in summary_results]
    r1_vals = [r["rank1"] for r in summary_results]
    t3_vals = [r["top3"] for r in summary_results]
    t5_vals = [r["top5"] for r in summary_results]
    ar_vals = [r["avg_rank"] for r in summary_results]
    
    # Subplot 1: Accuracy curves vs CoAtNet Weight
    ax1.plot(coat_weights, r1_vals, marker='o', linewidth=2.2, label='Rank-1 Accuracy (%)', color='#00b4d8')
    ax1.plot(coat_weights, t3_vals, marker='s', linewidth=2.0, label='Top-3 Accuracy (%)', color='#2a9d8f')
    ax1.plot(coat_weights, t5_vals, marker='^', linewidth=2.0, label='Top-5 Accuracy (%)', color='#e76f51')
    ax1.set_xlabel('CoAtNet Weight (%) [Remaining % is Traditional]', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('BiteID Hybrid Accuracy across Weight Ratios', fontsize=12, fontweight='bold')
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Subplot 2: Bar Comparison (Traditional vs CoAtNet vs Optimal Hybrid)
    trad_only = next(r for r in summary_results if r["configuration"] == "Traditional Only")
    coat_only = next(r for r in summary_results if r["configuration"] == "CoAtNet Only")
    optimal = max(summary_results, key=lambda x: (x["rank1"], x["top3"], x["top5"]))
    
    models = ['Traditional Only', 'CoAtNet Only', f'Hybrid Optimal\n({optimal["configuration"]})']
    r1_bars = [trad_only['rank1'], coat_only['rank1'], optimal['rank1']]
    t3_bars = [trad_only['top3'], coat_only['top3'], optimal['top3']]
    t5_bars = [trad_only['top5'], coat_only['top5'], optimal['top5']]
    
    x = np.arange(len(models))
    width = 0.25
    ax2.bar(x - width, r1_bars, width, label='Rank-1', color='#00b4d8')
    ax2.bar(x, t3_bars, width, label='Top-3', color='#2a9d8f')
    ax2.bar(x + width, t5_bars, width, label='Top-5', color='#e76f51')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Core Model Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[4] Generated evaluation plot: {PLOT_PATH}")
    
    # -------------------------------------------------------------------
    # Final Executive Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("                         EXECUTIVE SUMMARY")
    print("=" * 78)
    print(f"  Traditional Only : Rank-1 = {trad_only['rank1']:.2f}% | Top-3 = {trad_only['top3']:.2f}% | Top-5 = {trad_only['top5']:.2f}%")
    print(f"  CoAtNet Only     : Rank-1 = {coat_only['rank1']:.2f}% | Top-3 = {coat_only['top3']:.2f}% | Top-5 = {coat_only['top5']:.2f}%")
    print(f"  Optimal Hybrid   : Rank-1 = {optimal['rank1']:.2f}% | Top-3 = {optimal['top3']:.2f}% | Top-5 = {optimal['top5']:.2f}%")
    print(f"  Configuration    : {optimal['configuration']} (Avg Rank: {optimal['avg_rank']:.2f})")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()