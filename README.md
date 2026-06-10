<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,4,8&height=200&section=header&text=Music%20Data%20Analysis&fontSize=68&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Decoding%20the%20DNA%20of%20Every%20Hit&descAlignY=60&descSize=20" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-1DB954?style=for-the-badge&logo=python&logoColor=white&labelColor=0D0D0D)](https://python.org)
[![Spotify](https://img.shields.io/badge/Spotify-Dataset-1DB954?style=for-the-badge&logo=spotify&logoColor=white&labelColor=0D0D0D)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%2B%20DBSCAN-1DB954?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=0D0D0D)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-1DB954?style=for-the-badge&labelColor=0D0D0D)](LICENSE)

<br/>

<a href="https://github.com/isamkhan1809/music-data-analysis">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=1DB954&center=true&vCenter=true&width=700&lines=What+Makes+a+Song+Go+Viral%3F;K-Means+%7C+DBSCAN+%7C+PCA+Clustering;Danceability+%C2%B7+Energy+%C2%B7+Acousticness;3%2C000+Tracks.+10+Genres.+Deep+Analysis." alt="Typing SVG" />
</a>

</div>

---

<br/>

<div align="center">

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Every hit has a fingerprint — a signature of energy,      ║
  ║   danceability, tempo, and valence that separates it        ║
  ║   from the tracks that never made it.                       ║
  ║                                                              ║
  ║       This project finds that fingerprint.                  ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

</div>

<br/>

## `>_ The Story`

> *Spotify assigns every track a hidden score — danceability, energy, acousticness, valence. These aren't metadata. They're a mathematical portrait of a song.*
>
> *This project asks two questions: what do those scores say about popularity? And do genres cluster naturally in audio-feature space, or do the boundaries blur?*
>
> *The answers are in the data. This notebook surfaces them.*

<br/>

## `>_ Key Findings`

<table>
<tr>
<td width="50%">

**Popularity drivers:**
```
↑ Danceability   — strongest positive signal
↑ Energy         — second strongest
↓ Acousticness   — strong negative predictor
↓ Instrumentalness — weak vocals = lower score
↑ Explicit       — avg +6 popularity points
```

</td>
<td width="50%">

**Genre clusters (K-Means k=5):**
```
Cluster 1 → Energetic / Electronic
Cluster 2 → Acoustic / Classical
Cluster 3 → Hip-Hop / R&B
Cluster 4 → Rock / Metal
Cluster 5 → Pop / Indie (overlap)
```

</td>
</tr>
</table>

<br/>

## `>_ The Pipeline`

```
┌─────────────────────────────────────────────────────────────┐
│                  MUSIC ANALYSIS PIPELINE                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Spotify Dataset  (3,000+ tracks · 10 genres)        │  │
│  └──────────────────────────┬─────────────────────────┘  │
│                              │                            │
│            ┌─────────────────┴─────────────────┐         │
│            ▼                                   ▼         │
│      QUESTION 1                          QUESTION 2       │
│  What drives popularity?            Do genres cluster?    │
│                                                           │
│  Pearson correlations               K-Means (k=5)         │
│  Scatter plots                      DBSCAN                │
│  Feature heatmaps                   PCA (2-component)     │
│  Explicit comparison                Genre validation      │
│                                                           │
│            └─────────────────┬─────────────────┘         │
│                               ▼                           │
│               Radar charts · Genre fingerprints           │
│               Conclusions + recommendations               │
└─────────────────────────────────────────────────────────────┘
```

<br/>

## `>_ Audio Features`

<div align="center">

| Feature | Range | What It Measures |
|---|---|---|
| `danceability` | 0.0–1.0 | Suitability for dancing |
| `energy` | 0.0–1.0 | Perceptual intensity |
| `acousticness` | 0.0–1.0 | Acoustic confidence |
| `instrumentalness` | 0.0–1.0 | Probability of no vocals |
| `valence` | 0.0–1.0 | Musical positiveness |
| `tempo` | BPM | Speed of the track |
| `loudness` | dB | Overall loudness |
| `popularity` | 0–100 | **Target variable** |

</div>

<br/>

## `>_ Get Running`

```bash
# Clone
git clone https://github.com/isamkhan1809/music-data-analysis.git
cd music-data-analysis

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch (synthetic data — no download needed)
jupyter notebook spotify_analysis.ipynb
```

**With real Spotify data:**
```bash
pip install kaggle
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/
# In Cell 3: set USE_SYNTHETIC = False
```

<br/>

## `>_ Tech Stack`

<div align="center">

| Layer | Technology |
|---|---|
| **Clustering** | K-Means, DBSCAN |
| **Dimensionality Reduction** | PCA (2-component) |
| **Statistics** | Pearson correlation, scipy |
| **Visualisation** | matplotlib, seaborn |
| **Notebook** | Jupyter |

</div>

<br/>

## `>_ Project Structure`

```
music-data-analysis/
├── spotify_analysis.ipynb   ← Full 10-cell analysis
├── data/                    ← Place Kaggle CSV here
└── requirements.txt
```

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,4,8&height=120&section=footer&animation=twinkling" width="100%"/>

<br/>

*The science of sound. The math behind the music.*
*3,000 tracks analysed. Every signal surfaced.*

<br/>

[![GitHub](https://img.shields.io/badge/github-isamkhan1809-1DB954?style=for-the-badge&logo=github&logoColor=white&labelColor=0D0D0D)](https://github.com/isamkhan1809)

</div>
