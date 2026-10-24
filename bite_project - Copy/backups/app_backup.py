"""
app.py — Flask web dashboard for bite-mark forensic analysis.

Routes:
    GET  /                    — Landing page with upload zone
    POST /upload              — Upload a query bite-mark image
    GET  /analyze/<job_id>    — Run analysis and redirect to results
    GET  /results/<job_id>    — View ranked matches with XAI
    GET  /database            — Browse all subjects in the database
    GET  /subject/<subject_id>— View individual subject details
"""

import csv
import json
import os
import uuid
import time

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_from_directory
)

from preprocessing import process_single_image, load_and_validate
from matching_engine import (
    load_database, rank_matches, match_query_image, STRUCTURAL_KEYS, HU_KEYS
)
from xai_explainer import (
    generate_radar_chart, generate_feature_table,
    generate_contour_comparison, generate_all_reports
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "bite-mark-forensic-2026"

UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
DB_CSV       = os.path.join(RESULTS_DIR, "dataset_features.csv")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def get_subjects_info():
    """Scan datasets/ for subject folders and their images."""
    subjects = []
    if not os.path.isdir(DATASETS_DIR):
        return subjects

    for folder in sorted(os.listdir(DATASETS_DIR)):
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        images = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.startswith("Copy of"):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_EXT:
                images.append(fname)

        # Get person name from first image filename
        person_name = ""
        if images:
            name_parts = os.path.splitext(images[0])[0]
            # Remove "upper" / "lower" suffix
            person_name = name_parts.replace(" upper", "").replace(" lower", "").strip()

        subjects.append({
            "id": folder,
            "name": person_name,
            "images": images,
            "image_count": len(images),
        })

    return subjects


def load_db_features():
    """Load the database CSV, return list of dicts."""
    if not os.path.isfile(DB_CSV):
        return []
    return load_database(DB_CSV)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page with upload zone and database overview."""
    subjects = get_subjects_info()
    db_exists = os.path.isfile(DB_CSV)
    db_count = len(load_db_features()) if db_exists else 0
    return render_template(
        "index.html",
        subjects=subjects,
        db_exists=db_exists,
        db_count=db_count,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle query image upload, start analysis."""
    if "query_image" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["query_image"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Use JPG, PNG, or BMP.", "error")
        return redirect(url_for("index"))

    # Save with unique job ID
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(job_dir, f"query{ext}")
    file.save(saved_path)

    # Store metadata
    meta = {
        "original_name": file.filename,
        "saved_path": saved_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(job_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return redirect(url_for("analyze", job_id=job_id))


@app.route("/analyze/<job_id>")
def analyze(job_id):
    """Run analysis pipeline and redirect to results."""
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    meta_path = os.path.join(job_dir, "meta.json")

    if not os.path.isfile(meta_path):
        flash("Job not found.", "error")
        return redirect(url_for("index"))

    with open(meta_path) as f:
        meta = json.load(f)

    query_path = meta["saved_path"]

    # Check database exists
    if not os.path.isfile(DB_CSV):
        flash("Database not built yet. Run batch_process.py first.", "error")
        return redirect(url_for("index"))

    # Run matching
    summary, ranked, imp_features, contours = match_query_image(
    query_path,
    DB_CSV,
    original_filename=meta["original_name"]
)

    if summary is None:
        flash("Could not process the uploaded image.", "error")
        return redirect(url_for("index"))

    # Save processing debug images
    debug_dir = os.path.join(job_dir, "debug")
    process_single_image(query_path, output_dir=debug_dir)

    # Generate XAI reports for top 5
    report_dir = os.path.join(job_dir, "reports")
    report_paths = generate_all_reports(
        query_path, ranked, report_dir, top_n=5
    )

    # Generate radar charts for top 5
    for i, (db_row, score, breakdown) in enumerate(ranked[:5]):
        radar = generate_radar_chart(breakdown)
        radar_path = os.path.join(report_dir, f"radar_{i+1}.jpg")
        import cv2
        cv2.imwrite(radar_path, radar)

    # Save results
    results_data = {
        "query_summary": summary,
        "query_impressions": len(imp_features),
        "matches": [],
    }
    for i, (db_row, score, breakdown) in enumerate(ranked[:10]):
        match_data = {
            "rank": i + 1,
            "subject_id": str(db_row.get("subject_id", "")),
            "sample_id": str(db_row.get("sample_id", "")),
            "score": score,
            "breakdown": breakdown,
            "impression_count": int(float(db_row.get("impression_count", 0))),
        }
        results_data["matches"].append(match_data)

    with open(os.path.join(job_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    meta["analyzed"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return redirect(url_for("results", job_id=job_id))


@app.route("/results/<job_id>")
def results(job_id):
    """Display analysis results with ranked matches and XAI."""
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    results_path = os.path.join(job_dir, "results.json")

    if not os.path.isfile(results_path):
        flash("Results not found. Please upload and analyze an image first.", "error")
        return redirect(url_for("index"))

    with open(results_path) as f:
        results_data = json.load(f)

    # Check for radar chart images
    report_dir = os.path.join(job_dir, "reports")
    radar_images = {}
    for i in range(1, 6):
        radar_path = os.path.join(report_dir, f"radar_{i}.jpg")
        if os.path.isfile(radar_path):
            radar_images[i] = f"/job_files/{job_id}/reports/radar_{i}.jpg"

    # Check for debug images
    debug_dir = os.path.join(job_dir, "debug")
    debug_images = {}
    if os.path.isdir(debug_dir):
        for fname in sorted(os.listdir(debug_dir)):
            if fname.endswith(".jpg"):
                debug_images[fname] = f"/job_files/{job_id}/debug/{fname}"

    return render_template(
        "results.html",
        job_id=job_id,
        results=results_data,
        radar_images=radar_images,
        debug_images=debug_images,
    )


@app.route("/database")
def database():
    """Browse all subjects in the database."""
    subjects = get_subjects_info()
    db_features = load_db_features()

    # Map features by subject_id
    feature_map = {}
    for row in db_features:
        sid = row.get("subject_id", "")
        if sid not in feature_map:
            feature_map[sid] = []
        feature_map[sid].append(row)

    return render_template(
        "database.html",
        subjects=subjects,
        feature_map=feature_map,
    )


@app.route("/subject/<subject_id>")
def subject_detail(subject_id):
    """View individual subject details."""
    subject_path = os.path.join(DATASETS_DIR, subject_id)
    if not os.path.isdir(subject_path):
        flash("Subject not found.", "error")
        return redirect(url_for("database"))

    images = []
    for fname in sorted(os.listdir(subject_path)):
        if fname.startswith("Copy of"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in ALLOWED_EXT:
            images.append(fname)

    # Load features
    db_features = load_db_features()
    subject_features = [
        r for r in db_features if r.get("subject_id") == subject_id
    ]

    # Check for processed images
    processed = {}
    results_subdir = os.path.join(RESULTS_DIR, subject_id)
    if os.path.isdir(results_subdir):
        for sample_dir in os.listdir(results_subdir):
            sample_path = os.path.join(results_subdir, sample_dir)
            if os.path.isdir(sample_path):
                imgs = [f for f in os.listdir(sample_path) if f.endswith(".jpg")]
                processed[sample_dir] = imgs

    person_name = ""
    if images:
        name_parts = os.path.splitext(images[0])[0]
        person_name = name_parts.replace(" upper", "").replace(" lower", "").strip()

    return render_template(
        "subject.html",
        subject_id=subject_id,
        person_name=person_name,
        images=images,
        features=subject_features,
        processed=processed,
    )


# ---------------------------------------------------------------------------
# Serve uploaded / generated files
# ---------------------------------------------------------------------------

@app.route("/job_files/<job_id>/<path:filepath>")
def serve_job_file(job_id, filepath):
    """Serve files from the job upload directory."""
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    return send_from_directory(job_dir, filepath)


@app.route("/dataset_image/<subject_id>/<filename>")
def serve_dataset_image(subject_id, filename):
    """Serve original dataset images."""
    subject_path = os.path.join(DATASETS_DIR, subject_id)
    return send_from_directory(subject_path, filename)


@app.route("/result_image/<subject_id>/<path:filepath>")
def serve_result_image(subject_id, filepath):
    """Serve processed result images."""
    result_path = os.path.join(RESULTS_DIR, subject_id)
    return send_from_directory(result_path, filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Bite-Mark Forensic Dashboard ===")
    print(f"Database: {DB_CSV}")
    print(f"Datasets: {DATASETS_DIR}")
    print(f"Starting server on http://127.0.0.1:5000\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
