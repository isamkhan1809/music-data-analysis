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

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=1DB954&center=true&vCenter=true&width=700&lines=Decode+the+DNA+of+Every+Hit+%F0%9F%8E%B5;What+Makes+a+Song+Go+Viral%3F;K-Means+%7C+DBSCAN+%7C+PCA+Clustering;Spotify+Tracks+Dataset+%E2%80%94+Deep+Analysis" alt="Typing SVG" />

<img src="https://media.giphy.com/media/K6dv229cpUkP4TCQYu/giphy.gif" width="360" />

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Spotify](https://img.shields.io/badge/Data-Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

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
│  What drives  Do genres cluster naturally?                      │
│  popularity?                                                    │
│     │               │                                           │
│     ▼               ▼                                           │
│  Correlation    K-Means (k=5)  +  DBSCAN                        │
│  Heatmaps       PCA projection + Genre validation               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ◈ Key Findings

> **Danceability and energy** are the strongest positive predictors of popularity.

> **Acousticness and instrumentalness** are strong negative predictors.

> **Explicit tracks** average ~6 popularity points higher than non-explicit.

> **K-Means k=5** maps to: Energetic/Electronic · Acoustic/Classical · Hip-Hop/R&B · Rock/Metal · Pop/Indie

> **Classical and jazz** are the most separable in PCA space. **Pop and indie** overlap substantially.

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
| `popularity` | 0–100 | **Target variable** |

---

## ◈ Quick Start

```bash
git clone https://github.com/isamkhan1809/music-data-analysis.git
cd music-data-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook spotify_analysis.ipynb
# Runs on synthetic data by default — no Kaggle account needed
```

### With Real Spotify Data

```bash
pip install kaggle
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/
# In Cell 3: set USE_SYNTHETIC = False
```

---

## ◈ Notebook Structure

| Cell | Content |
|---|---|
| 1–3 | Intro, imports, data loading |
| 4–5 | EDA + popularity correlations |
| 6–7 | Genre distributions + K-Means + PCA |
| 8–9 | DBSCAN + radar charts |
| 10 | Conclusions |

---

## ◈ Project Structure

```
music-data-analysis/
├── spotify_analysis.ipynb   ← Full analysis
├── data/
├── requirements.txt
└── README.md
```

---

<div align="center">

**The science of sound. The math behind the music.**

*MIT License*

<br/>

Into music tech, Spotify data, or ML-powered audio analysis?<br/>
Let's connect — built by <a href="https://github.com/isamkhan1809">Isam Khan</a> &nbsp;|&nbsp;
<a href="https://linkedin.com/in/isam-khan-3a1260292"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white&labelColor=000000"/></a>
<a href="https://isamkhan.com"><img src="https://img.shields.io/badge/-isamkhan.com-00D9FF?style=flat-square&logo=googlechrome&logoColor=white&labelColor=000000"/></a>

</div>
