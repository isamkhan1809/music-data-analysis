<div align="center">

```
███╗   ███╗██╗   ██╗███████╗██╗ ██████╗    ██████╗  █████╗ ████████╗ █████╗
████╗ ████║██║   ██║██╔════╝██║██╔════╝    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██╔████╔██║██║   ██║███████╗██║██║         ██║  ██║███████║   ██║   ███████║
██║╚██╔╝██║██║   ██║╚════██║██║██║         ██║  ██║██╔══██║   ██║   ██╔══██║
██║ ╚═╝ ██║╚██████╔╝███████║██║╚██████╗    ██████╔╝██║  ██║   ██║   ██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

         █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
        ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
        ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
        ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
        ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
        ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
```

### *Decode the DNA of Every Hit.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Spotify](https://img.shields.io/badge/Data-Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> **An end-to-end analysis of the Spotify Tracks Dataset — uncovering what makes songs popular, why genres cluster the way they do, and what separates a viral hit from obscurity.**

</div>

---

## ◈ Two Questions. Deep Answers.

```
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSIS PIPELINE                             │
│                                                                 │
│  Spotify Dataset (3,000+ tracks, 10 genres)                     │
│           │                                                     │
│     ┌─────┴─────┐                                              │
│     ▼           ▼                                              │
│  QUESTION 1   QUESTION 2                                        │
│  ─────────    ─────────                                         │
│  What drives  Do genres                                         │
│  popularity?  cluster                                           │
│               naturally?                                        │
│     │               │                                           │
│     ▼               ▼                                           │
│  Correlation    K-Means (k=5)                                   │
│  Analysis       DBSCAN                                          │
│  Scatter plots  PCA projection                                  │
│  Feature heatmap Genre validation                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ◈ Key Findings

> **Danceability and energy** are the strongest positive predictors of popularity.

> **Acousticness and instrumentalness** are strong negative predictors — highly acoustic or instrumental tracks score lower on average.

> **Explicit tracks** average ~6 popularity points higher than non-explicit tracks.

> **K-Means k=5** maps closely to real genre families:
> - Cluster 1 → Energetic / Electronic
> - Cluster 2 → Acoustic / Classical
> - Cluster 3 → Hip-Hop / R&B
> - Cluster 4 → Rock / Metal
> - Cluster 5 → Pop / Indie

> **Classical and jazz** are the most separable genres in PCA space. **Pop and indie** overlap substantially.

> **Tempo** alone is weak at predicting popularity but strong at separating **metal** (high BPM) from **classical** (variable BPM).

---

## ◈ Audio Features Decoded

| Feature | Range | What It Measures |
|---|---|---|
| `danceability` | 0.0–1.0 | Suitability for dancing |
| `energy` | 0.0–1.0 | Perceptual intensity |
| `acousticness` | 0.0–1.0 | Acoustic confidence |
| `instrumentalness` | 0.0–1.0 | Probability of no vocals |
| `valence` | 0.0–1.0 | Musical positiveness |
| `tempo` | BPM | Beats per minute |
| `loudness` | dB | Overall loudness |
| `speechiness` | 0.0–1.0 | Presence of spoken words |
| `popularity` | 0–100 | **Target variable** |

---

## ◈ Quick Start

```bash
# 1. Clone
git clone https://github.com/isamkhan1809/music-data-analysis.git
cd music-data-analysis

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch (runs on synthetic data by default — no Kaggle account needed)
jupyter notebook spotify_analysis.ipynb
```

### With Real Spotify Data

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/
unzip data/-spotify-tracks-dataset.zip -d data/
# In Cell 3: set USE_SYNTHETIC = False
```

---

## ◈ Notebook Structure

| Cell | Content |
|---|---|
| 1 | Introduction & problem framing |
| 2 | Imports |
| 3 | Data loading (synthetic or real) |
| 4 | Exploratory data analysis |
| 5 | Popularity correlation analysis |
| 6 | Feature distribution by genre |
| 7 | K-Means clustering + PCA projection |
| 8 | DBSCAN clustering |
| 9 | Radar charts — genre audio fingerprints |
| 10 | Conclusions & insights |

---

## ◈ Techniques

| Area | Methods |
|---|---|
| Wrangling | pandas, numpy |
| Visualisation | matplotlib, seaborn (heatmaps, boxplots, radar charts) |
| Clustering | K-Means, DBSCAN |
| Dimensionality Reduction | PCA (2-component) |
| Statistics | Pearson correlation, scipy |

---

## ◈ Project Structure

```
music-data-analysis/
├── spotify_analysis.ipynb   ← Full analysis (10 cells)
├── data/                    ← Place Kaggle CSV here
├── requirements.txt
└── README.md
```

---

<div align="center">

**The science of sound. The math behind the music.**

*MIT License*

</div>
