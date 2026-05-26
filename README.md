# Music Data Analysis — Spotify Dataset

An end-to-end exploratory data analysis and machine-learning project that investigates what makes songs popular and identifies genre-specific audio characteristics using the Spotify Tracks Dataset from Kaggle.

---

## Project Overview

This project answers two central questions:

1. **What audio features drive popularity?** — Using correlation analysis and scatter plots to find the strongest predictors of a track's popularity score.
2. **Do genres cluster naturally in audio-feature space?** — Using K-Means clustering (k=5) and DBSCAN to discover natural groupings, then validating them against ground-truth genre labels.

### Key Findings

- **Danceability and energy** show the strongest positive correlation with popularity across all genres.
- **Acousticness and instrumentalness** are strong negative predictors of popularity — highly acoustic or instrumental tracks score lower on average.
- **Explicit tracks** average ~6 popularity points higher than non-explicit tracks.
- **K-Means with k=5** produces clusters that map closely to broad genre families: (1) energetic/electronic, (2) acoustic/classical, (3) hip-hop/r&b, (4) rock/metal, (5) pop/indie.
- **Classical and jazz** are the most separable genres in PCA space; pop and indie overlap substantially.
- **Tempo** alone is a weak predictor of popularity but is a strong discriminator between metal (high BPM) and classical (variable BPM).

---

## Dataset

**Source:** [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

### Download Instructions

1. Create a free Kaggle account at https://www.kaggle.com if you do not already have one.
2. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
3. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (download it from Kaggle → Account → Create API Token).
4. Run:
   ```bash
   kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/
   unzip data/-spotify-tracks-dataset.zip -d data/
   ```
5. The CSV will appear at `data/dataset.csv`. Update the `DATA_PATH` variable in Cell 3 of the notebook to point to it instead of using the synthetic data path.

### Column Reference

| Column | Description |
|---|---|
| `track_id` | Spotify track URI |
| `artists` | Artist name(s) |
| `album_name` | Album name |
| `track_name` | Track name |
| `popularity` | 0–100 popularity score |
| `duration_ms` | Duration in milliseconds |
| `explicit` | Whether the track has explicit lyrics |
| `danceability` | 0.0–1.0 — suitability for dancing |
| `energy` | 0.0–1.0 — perceptual intensity |
| `key` | Musical key (0=C … 11=B) |
| `loudness` | dB loudness (typically −60 to 0) |
| `mode` | 1=major, 0=minor |
| `speechiness` | 0.0–1.0 — presence of spoken words |
| `acousticness` | 0.0–1.0 — acoustic confidence |
| `instrumentalness` | 0.0–1.0 — probability of no vocals |
| `liveness` | 0.0–1.0 — presence of a live audience |
| `valence` | 0.0–1.0 — musical positiveness |
| `tempo` | BPM |
| `time_signature` | Estimated beats per bar |
| `track_genre` | Genre label |

---

## Project Structure

```
music-data-analysis/
├── data/                        # Place the Kaggle CSV here
├── spotify_analysis.ipynb       # Main analysis notebook (10 cells)
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # This file
```

---

## Setup & Running

### Prerequisites

- Python 3.9 or later
- pip

### Installation

```bash
# Clone / navigate to the project folder
cd music-data-analysis

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Notebook

```bash
jupyter notebook spotify_analysis.ipynb
```

Or with JupyterLab:

```bash
pip install jupyterlab
jupyter lab spotify_analysis.ipynb
```

### Run without Kaggle Data

The notebook ships with a **synthetic data generator** (Cell 3) that produces 3,000 realistic tracks across 10 genres. You can run every cell end-to-end without downloading anything from Kaggle. When the real dataset is available, swap `USE_SYNTHETIC = True` to `False` in Cell 3.

---

## Techniques Used

| Area | Methods |
|---|---|
| Data Wrangling | pandas, numpy |
| Visualization | matplotlib, seaborn (heatmaps, boxplots, radar charts, scatter plots) |
| Clustering | K-Means (scikit-learn), DBSCAN |
| Dimensionality Reduction | PCA (2-component projection for cluster visualization) |
| Statistical Analysis | Pearson correlation, scipy descriptive stats |

---

## License

This project is for educational purposes. The Spotify Tracks Dataset is distributed under the terms stated on its [Kaggle page](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
