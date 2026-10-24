# BiteID — Forensic Bite-Mark Analysis System

An AI-assisted forensic bite-mark matching and Explainable AI (XAI) dashboard built with Flask, OpenCV, PyTorch, and CoAtNet.

---

## ⚡ How to Run (Quick Start)

### 🌟 New Easy Ways to Run:

| Method | Command / Action | Where to run |
|---|---|---|
| **🖱️ 1-Click (Easiest)** | **Double-click `run.bat`** | File Explorer (in root or project folder) |
| **💻 Command Prompt / PowerShell** | `.\run.bat` or `run` | Any terminal window |
| **⚡ PowerShell Script** | `.\run.ps1` | PowerShell terminal |
| **📦 NPM Shortcut** | `npm start` *(or `npm run dev`)* | Terminal |

> ✨ **What this does automatically:**
> 1. Detects and activates the Python virtual environment (`venv`).
> 2. Launches the Flask server on `http://127.0.0.1:5000`.
> 3. Automatically opens your default web browser to the dashboard!

---

## 🔄 Command Replacement Summary

| Old Complex Command (Replaced) | New Easy Replacement | Benefit |
|---|---|---|
| `cd "bite_project - Copy"`<br>`.\venv\Scripts\python.exe app.py` | **`.\run.bat`**<br>*(or double-click `run.bat`)* | No need to remember `venv\Scripts\python.exe` path or type long paths. 1-click execution. |
| Activating venv manually via PowerShell script | **`.\run.ps1`** or **`npm start`** | Works from both root and subfolder, opens browser automatically. |

---

## 📁 Project Structure & Alignment

```
bite_version2/
│
├── run.bat                          # ⚡ 1-Click Windows Batch launcher (Root)
├── run.ps1                          # ⚡ PowerShell launcher (Root)
├── package.json                     # 📦 npm start / npm run dev config
├── requirements.txt                 # 📋 Complete Python dependencies list
├── README.md                        # 📖 Project guide and documentation
│
└── bite_project - Copy/             # 📂 Main Application Directory
    │
    ├── run.bat                      # ⚡ 1-Click launcher (Direct folder)
    ├── run.ps1                      # ⚡ PowerShell launcher (Direct folder)
    ├── app.py                       # 🚀 Main Flask Web Application & Routes
    ├── requirements.txt             # 📋 Virtual environment dependencies
    ├── search_history.json          # 🕒 Persistent recent search history
    │
    ├── 🔬 Computer Vision & AI Pipeline:
    │   ├── preprocessing.py         # Image validation, CLAHE, segmentation, contour extraction
    │   ├── coatnet_extractor.py     # Deep learning CoAtNet feature extraction model
    │   ├── coatnet_database.py      # Database embeddings builder for CoAtNet
    │   ├── matching_engine.py       # Handcrafted shape, Hu moments & structural matching
    │   ├── hybrid_matching.py       # Hybrid engine (CoAtNet + Handcrafted weighted fusion)
    │   └── xai_explainer.py         # Explainable AI: radar charts, contour overlays & tables
    │
    ├── 📈 Evaluation & Testing:
    │   ├── evaluate.py              # Performance evaluation for handcrafted matching
    │   ├── evaluate_coatnet.py      # Performance evaluation for CoAtNet model
    │   ├── evaluate_hybrid.py       # Performance evaluation for hybrid model
    │   ├── batch_process.py         # Batch dataset preprocessing utility
    │   ├── test.py                  # Standalone pipeline unit tests
    │   └── test_coatnet_similarity.py# Quick similarity scoring verification
    │
    ├── 🌐 Web Templates (templates/):
    │   ├── base.html                # Global layout, header, footer & nav links
    │   ├── index.html               # Upload & forensic inquiry landing page
    │   ├── results.html             # Top ranked suspect matches, XAI radar & metrics
    │   ├── history.html             # Recent searches with custom confirm delete modal
    │   ├── database.html            # Suspect database catalog browser
    │   └── subject.html             # Detailed subject profile & impression details
    │
    ├── 🎨 Static Assets (static/):
    │   ├── css/
    │   │   └── style.css            # Custom CSS styling (dark/light, glassmorphism, modals)
    │   ├── js/
    │   │   └── app.js               # Client-side UI interactions & drag-and-drop
    │   └── history_thumbs/          # Cached thumbnails for recent search queries
    │
    ├── 💾 Data & Results:
    │   ├── datasets/                # Subject reference impressions (P001 - P012)
    │   ├── results/                 # Feature CSVs, evaluation plots & metrics
    │   └── uploads/                 # Temporary query uploads
    │
    ├── 🗄️ Backups & Archives:
    │   └── backups/
    │       └── app_backup.py        # Archived legacy code backup
    │
    └── 🐍 Python Environment:
        └── venv/                    # Isolated Python 3.13 Virtual Environment
```

---

## 🌐 Web Routes

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Main landing page & image upload portal |
| `/upload` | `POST` | Uploads bite-mark query and starts analysis job |
| `/results/<job_id>` | `GET` | Displays suspect ranking, matching scores & XAI charts |
| `/history` | `GET` | View recent searches and forensic queries |
| `/history/delete/<id>` | `POST` | Removes an individual search record |
| `/history/clear` | `POST` | Clears all search history records |
| `/database` | `GET` | Browse all 12 registered suspects in the database |
| `/subject/<id>` | `GET` | View details and reference images for a specific subject |
