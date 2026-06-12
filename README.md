# 🎬 Netflix Prize Dataset: Personalized Content Discovery Engine...

An end-to-end, production-ready modular Python pipeline engineered to process user-item interactions, train collaborative filtering models, evaluate absolute and ranking metrics, and generate personalized movie recommendations.

This repository implements and compares **Singular Value Decomposition (SVD Matrix Factorization)** and a closed-form **Biased Baseline (BaselineOnly)** model utilizing the historic Netflix Prize Dataset.

---

## 📂 Repository Structure

```text
netflix-recommender/
│
├── data/                       # Local data placeholder directory
│   └── README.md
│
├── src/                        # Core source code package
│   ├── __init__.py
│   ├── data_pipeline.py        # Data downloading, custom parsing, and downsampling
│   ├── model_pipeline.py       # Surprise framework dataset configuration & training
│   ├── evaluation.py           # Custom implementations for RMSE and MAP@10 metrics
│   └── inference.py            # Recommendation generation & user profiling module
│
├── main.py                     # Master execution orchestrator script
├── requirements.txt            # Package dependency manifest
└── README.md                   # System documentation and instructions
```

---

## 🛠️ System Requirements

### Hardware Prerequisites

| Component | Requirement |
|-----------|-------------|
| **CPU** | Minimum 1 Dual-Core (Intel Xeon / AMD EPYC or better) |
| **GPU** | T4 or equivalent recommended for MAP@10 evaluation sorting |
| **RAM** | Minimum **12 GB** (lazy parsing & row downsampling keep memory bounded) |
| **Storage** | Minimum **5 GB** free disk space for raw data, matrix cache, and dependencies |

### Software Stack

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04 LTS+), macOS, or Windows 10/11 via WSL2 / Anaconda |
| **Python** | `3.9` – `3.11` (optimized on `3.10`) |

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/lalit5700/NetflixRecommendationSystems.git
cd NetflixRecommendationSystems
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Pipeline

```bash
python main.py
```

---

## ⚙️ Pipeline Workflow

When `main.py` executes, the following stages run sequentially:

| Stage | Module | Description |
|-------|--------|-------------|
| **1. Data** | `data_pipeline.py` | Downloads raw Netflix datasets, parses custom movie line offsets, filters to a dense 500K-observation frame, and caches locally |
| **2. Modelling** | `model_pipeline.py` | Structures data into an 80/20 train-test split and fits both the SVD latent factorization engine and the BaselineOnly algorithm |
| **3. Evaluation** | `evaluation.py` | Computes RMSE and MAP@10; items with true ratings ≥ 3.5 are treated as relevant |
| **4. Inference** | `inference.py` | Profiles a target test user and outputs personalized recommendation carousels with algorithmic success examples |

---

## 📊 Models Implemented

### Singular Value Decomposition (SVD)
A latent factor collaborative filtering model that decomposes the user-item interaction matrix into lower-dimensional representations, capturing latent preference signals.

### Biased Baseline (BaselineOnly)
A closed-form model that estimates ratings using global, user-level, and item-level bias terms — serving as a strong interpretable baseline.

---

## 📈 Evaluation Metrics

- **RMSE** — Measures absolute point-error generalizability on held-out ratings.
- **MAP@10** — Measures ranking quality; evaluates how well relevant items (rating ≥ 3.5) are surfaced in the top 10 recommendation slots.

---

## 📝 Notes

- Pipeline progress is logged directly to stdout for real-time monitoring.
- Local dynamic caching avoids redundant recomputation on repeated runs.
- Memory usage is strictly bounded via lazy text parsing and row downsampling.
