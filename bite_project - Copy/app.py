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
import cv2
import json
import os
import uuid
import time
import shutil

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_from_directory
)

from preprocessing import process_single_image, load_and_validate
from matching_engine import (
    load_database, rank_matches, match_query_image, STRUCTURAL_KEYS, HU_KEYS
)

from hybrid_matching import hybrid_match_query_image
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
HISTORY_FILE = os.path.join(BASE_DIR, "search_history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


SUBJECT_NAMES = {
    "P001": "Rohith",
    "P002": "Shravya",
    "P003": "Drithika",
    "P004": "Vyshnavi",
    "P005": "Sonika",
    "P006": "Akash",
    "P007": "Rameeza",
    "P008": "Yashas",
    "P009": "Rishi",
    "P010": "Vishmaya",
    "P011": "Mounesh",
    "P012": "Prateeksha",
}

def resolve_person_name(subject_id, sample_id=""):
    """Lookup person name from mapping, or extract from sample filename."""
    import re
    sid = str(subject_id).strip()
    if sid in SUBJECT_NAMES:
        return SUBJECT_NAMES[sid]
    if sample_id:
        base = os.path.splitext(os.path.basename(str(sample_id)))[0]
        cleaned = re.sub(r'(?i)(copy\s*of|upper|lower|arch|sample|_|-|\d+)', '', base).strip()
        if cleaned:
            return cleaned.title()
    return f"Subject {sid}"


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

        subjects.append({
            "id": folder,
            "name": f"Subject {folder}",
            "images": images,
            "image_count": len(images),
        })

    return subjects


def load_db_features():
    """Load the database CSV, return list of dicts."""
    if not os.path.isfile(DB_CSV):
        return []
    return load_database(DB_CSV)


def load_history():
    """Load search history from JSON file."""
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history_entry(entry):
    """Append a new entry to the search history."""
    history = load_history()
    history.insert(0, entry)   # newest first
    history = history[:50]     # keep at most 50 entries
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


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

    print("\n[APP DEBUG] job_id:", job_id)
    print("[APP DEBUG] query_path:", query_path)
    print("[APP DEBUG] original_name:", repr(meta["original_name"]))

    # Check database exists
    if not os.path.isfile(DB_CSV):
        flash("Database not built yet. Run batch_process.py first.", "error")
        return redirect(url_for("index"))

    # Run hybrid matching
    summary, ranked, imp_features, contours = hybrid_match_query_image(
        query_path,
        original_filename=meta["original_name"],
        top_n=10
    )
    print(
        "[APP DEBUG] hybrid function file:",
        hybrid_match_query_image.__code__.co_filename
    )

    if summary is None:
        flash("Could not process the uploaded image.", "error")
        return redirect(url_for("index"))

    # Save processing debug images
    debug_dir = os.path.join(job_dir, "debug")
    process_single_image(query_path, output_dir=debug_dir)

    print("\n[APP] HYBRID RANKING RECEIVED:")
    for i, (db_row, score, breakdown) in enumerate(ranked[:10], 1):
        print(
            f"[APP] #{i}: "
            f"{db_row.get('subject_id')} / "
            f"{db_row.get('sample_id')} "
            f"= {score}%"
        )
    print("[APP] END HYBRID RANKING\n")

    # Generate XAI reports for top 5
    report_dir = os.path.join(job_dir, "reports")
    report_paths = generate_all_reports(
        query_path, ranked, report_dir, top_n=5
    )

    # Generate radar charts for top 5
    for i, (db_row, score, breakdown) in enumerate(ranked[:5]):
        radar = generate_radar_chart(breakdown)
        radar_path = os.path.join(report_dir, f"radar_{i+1}.jpg")
        cv2.imwrite(radar_path, radar)

    # Save results
    results_data = {
        "query_summary": summary,
        "query_impressions": len(imp_features),
        "matches": [],
    }

    for i, (db_row, score, breakdown) in enumerate(ranked[:10]):
        sid = str(db_row.get("subject_id", ""))
        samp_id = str(db_row.get("sample_id", ""))
        match_data = {
            "rank": i + 1,
            "subject_id": sid,
            "sample_id": samp_id,
            "person_name": resolve_person_name(sid, samp_id),
            "score": float(score),
            "breakdown": {k: float(v) for k, v in breakdown.items()},
            "impression_count": int(
                float(db_row.get("impression_count", 0))
            ),
        }

        results_data["matches"].append(match_data)

    with open(os.path.join(job_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    meta["analyzed"] = True

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # ---- Save to search history ----
    top_match = results_data["matches"][0] if results_data["matches"] else {}

    # Copy the query thumbnail for history display
    thumb_dir = os.path.join(BASE_DIR, "static", "history_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_filename = f"{job_id}.jpg"
    thumb_path = os.path.join(thumb_dir, thumb_filename)
    try:
        import cv2 as _cv2
        _img = _cv2.imread(query_path)
        if _img is not None:
            h, w = _img.shape[:2]
            scale = min(200 / w, 200 / h, 1.0)
            _img = _cv2.resize(_img, (int(w * scale), int(h * scale)))
            _cv2.imwrite(thumb_path, _img)
    except Exception:
        thumb_filename = ""

    history_entry = {
        "job_id": job_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_name": meta.get("original_name", ""),
        "query_impressions": results_data.get("query_impressions", 0),
        "thumb": thumb_filename,
        "top_match": {
            "person_name": top_match.get("person_name", ""),
            "subject_id":  top_match.get("subject_id",  ""),
            "sample_id":   top_match.get("sample_id",   ""),
            "score":       top_match.get("score",        0),
        },
        "total_matches": len(results_data["matches"]),
    }
    save_history_entry(history_entry)

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

    # Ensure every match has person_name (even for existing saved jobs)
    for m in results_data.get("matches", []):
        if not m.get("person_name"):
            m["person_name"] = resolve_person_name(m.get("subject_id", ""), m.get("sample_id", ""))

    # Check for radar chart and forensic report images
    report_dir = os.path.join(job_dir, "reports")
    radar_images = {}
    report_images = {}
    for i in range(1, 6):
        radar_path = os.path.join(report_dir, f"radar_{i}.jpg")
        if os.path.isfile(radar_path):
            radar_images[i] = f"/job_files/{job_id}/reports/radar_{i}.jpg"
        rep_path = os.path.join(report_dir, f"report_{i}.jpg")
        if os.path.isfile(rep_path):
            report_images[i] = f"/job_files/{job_id}/reports/report_{i}.jpg"

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
        report_images=report_images,
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

    return render_template(
        "subject.html",
        subject_id=subject_id,
        person_name=f"Subject {subject_id}",
        images=images,
        features=subject_features,
        processed=processed,
    )


# ---------------------------------------------------------------------------
# Search History
# ---------------------------------------------------------------------------

@app.route("/history")
def history():
    """Display the saved search history."""
    entries = load_history()
    return render_template("history.html", entries=entries)


@app.route("/history/clear", methods=["POST"])
def history_clear():
    """Clear all search history."""
    # Remove history file
    if os.path.isfile(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    # Remove thumbnails
    thumb_dir = os.path.join(BASE_DIR, "static", "history_thumbs")
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir)
    flash("Search history cleared.", "success")
    return redirect(url_for("history"))


@app.route("/history/delete/<job_id>", methods=["POST"])
def history_delete(job_id):
    """Delete a single history entry."""
    history = load_history()
    history = [e for e in history if e.get("job_id") != job_id]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)
    # Remove thumbnail
    thumb_path = os.path.join(BASE_DIR, "static", "history_thumbs", f"{job_id}.jpg")
    if os.path.isfile(thumb_path):
        os.remove(thumb_path)
    return redirect(url_for("history"))


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
