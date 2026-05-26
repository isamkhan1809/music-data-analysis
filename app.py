import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ──────────────────────────────────────────────
# Global state populated at startup
# ──────────────────────────────────────────────
df = None
pca_coords = None
genre_stats_cache = None
audio_features_cache = None
clusters_cache = None
popularity_drivers_cache = None

GENRES = [
    "pop", "rock", "hip-hop", "classical", "jazz",
    "electronic", "country", "r&b", "metal", "indie",
]

GENRE_COLORS = {
    "pop": "#ff6384",
    "rock": "#ff9f40",
    "hip-hop": "#ffcd56",
    "classical": "#4bc0c0",
    "jazz": "#9966ff",
    "electronic": "#c9cbcf",
    "country": "#36a2eb",
    "r&b": "#fd7f6f",
    "metal": "#7eb0d5",
    "indie": "#b2e061",
}

AUDIO_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "speechiness",
]

# Genre-specific distribution parameters (mean, std) for each audio feature
# Order: danceability, energy, valence, acousticness, instrumentalness, speechiness
GENRE_PARAMS = {
    "pop":        {"danceability": (0.72, 0.10), "energy": (0.68, 0.12), "valence": (0.65, 0.15),
                   "acousticness": (0.22, 0.12), "instrumentalness": (0.04, 0.06), "speechiness": (0.08, 0.04),
                   "tempo": (118, 15), "loudness": (-5.5, 2.5), "popularity": (65, 15)},
    "rock":       {"danceability": (0.52, 0.12), "energy": (0.80, 0.12), "valence": (0.55, 0.18),
                   "acousticness": (0.12, 0.10), "instrumentalness": (0.08, 0.10), "speechiness": (0.06, 0.03),
                   "tempo": (128, 18), "loudness": (-5.0, 2.5), "popularity": (58, 16)},
    "hip-hop":    {"danceability": (0.78, 0.10), "energy": (0.65, 0.12), "valence": (0.55, 0.18),
                   "acousticness": (0.15, 0.12), "instrumentalness": (0.03, 0.05), "speechiness": (0.22, 0.08),
                   "tempo": (95, 14), "loudness": (-5.8, 2.8), "popularity": (63, 16)},
    "classical":  {"danceability": (0.28, 0.10), "energy": (0.20, 0.10), "valence": (0.38, 0.18),
                   "acousticness": (0.88, 0.08), "instrumentalness": (0.82, 0.12), "speechiness": (0.04, 0.02),
                   "tempo": (100, 28), "loudness": (-16.0, 4.5), "popularity": (42, 15)},
    "jazz":       {"danceability": (0.55, 0.12), "energy": (0.42, 0.14), "valence": (0.52, 0.20),
                   "acousticness": (0.58, 0.16), "instrumentalness": (0.45, 0.20), "speechiness": (0.05, 0.03),
                   "tempo": (108, 22), "loudness": (-11.0, 3.5), "popularity": (45, 14)},
    "electronic": {"danceability": (0.78, 0.10), "energy": (0.82, 0.10), "valence": (0.60, 0.18),
                   "acousticness": (0.06, 0.06), "instrumentalness": (0.35, 0.22), "speechiness": (0.07, 0.04),
                   "tempo": (128, 14), "loudness": (-5.2, 2.8), "popularity": (60, 16)},
    "country":    {"danceability": (0.60, 0.12), "energy": (0.58, 0.14), "valence": (0.62, 0.18),
                   "acousticness": (0.42, 0.16), "instrumentalness": (0.02, 0.03), "speechiness": (0.05, 0.02),
                   "tempo": (112, 16), "loudness": (-7.5, 3.0), "popularity": (52, 15)},
    "r&b":        {"danceability": (0.72, 0.11), "energy": (0.60, 0.13), "valence": (0.58, 0.18),
                   "acousticness": (0.28, 0.14), "instrumentalness": (0.05, 0.07), "speechiness": (0.09, 0.05),
                   "tempo": (100, 15), "loudness": (-7.0, 3.0), "popularity": (60, 16)},
    "metal":      {"danceability": (0.38, 0.12), "energy": (0.92, 0.07), "valence": (0.32, 0.18),
                   "acousticness": (0.05, 0.06), "instrumentalness": (0.18, 0.18), "speechiness": (0.07, 0.04),
                   "tempo": (145, 20), "loudness": (-4.0, 2.0), "popularity": (48, 15)},
    "indie":      {"danceability": (0.58, 0.13), "energy": (0.55, 0.15), "valence": (0.50, 0.20),
                   "acousticness": (0.38, 0.18), "instrumentalness": (0.12, 0.14), "speechiness": (0.06, 0.03),
                   "tempo": (115, 18), "loudness": (-8.5, 3.0), "popularity": (50, 16)},
}

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Skylar", "Dakota", "Reese", "Cameron", "Drew", "Blake", "Sage", "Noel",
    "Luca", "Milo", "Aria", "Luna", "Zara", "Nova", "Echo", "Juno",
    "Kai", "River", "Phoenix", "Ember", "Orion", "Lyric",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Young", "Lee", "Walker", "Hall", "Chen", "Patel",
    "Rivera", "Cruz", "Torres", "Reyes", "Santos", "Flores", "Carter", "Moore",
]

TRACK_ADJECTIVES = [
    "Midnight", "Golden", "Electric", "Neon", "Crystal", "Shadow", "Velvet",
    "Broken", "Wild", "Faded", "Lost", "Burning", "Frozen", "Silent", "Falling",
    "Running", "Dreaming", "Dancing", "Shining", "Hollow", "Sacred", "Distant",
    "Wicked", "Sober", "Alive", "Restless", "Endless", "Perfect", "Heavy", "Soft",
]

TRACK_NOUNS = [
    "Heart", "Night", "Sky", "Road", "Fire", "Rain", "Storm", "Dream", "Soul",
    "Light", "Wave", "Echo", "Moon", "Star", "Wind", "River", "Ocean", "Mountain",
    "City", "Way", "Love", "Time", "World", "Mind", "Space", "Life", "Sound",
    "Voice", "Floor", "Eyes", "Hands", "Blood", "Bones", "Wings", "Roots",
]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def _clip_01(arr):
    return _clip(arr, 0.0, 1.0)


def generate_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tracks_per_genre = n // len(GENRES)
    remainder = n - tracks_per_genre * len(GENRES)

    rows = []
    for i, genre in enumerate(GENRES):
        count = tracks_per_genre + (1 if i < remainder else 0)
        params = GENRE_PARAMS[genre]

        # Build fake track names and artists
        adj = rng.choice(TRACK_ADJECTIVES, count)
        noun = rng.choice(TRACK_NOUNS, count)
        track_names = [f"{a} {b}" for a, b in zip(adj, noun)]

        first = rng.choice(FIRST_NAMES, count)
        last = rng.choice(LAST_NAMES, count)
        artists = [f"{f} {l}" for f, l in zip(first, last)]

        for j in range(count):
            row = {
                "track_name": track_names[j],
                "artist": artists[j],
                "genre": genre,
            }
            for feat in AUDIO_FEATURES:
                mu, sigma = params[feat]
                val = rng.normal(mu, sigma)
                row[feat] = float(_clip_01(np.array([val]))[0])

            t_mu, t_sig = params["tempo"]
            row["tempo"] = float(_clip(np.array([rng.normal(t_mu, t_sig)]), 60, 200)[0])

            l_mu, l_sig = params["loudness"]
            row["loudness"] = float(_clip(np.array([rng.normal(l_mu, l_sig)]), -30, 0)[0])

            p_mu, p_sig = params["popularity"]
            row["popularity"] = float(_clip(np.array([rng.normal(p_mu, p_sig)]), 0, 100)[0])

            rows.append(row)

    return pd.DataFrame(rows)


def run_analysis(data):
    global pca_coords

    feature_matrix = data[AUDIO_FEATURES].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # K-Means clustering (k=8)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    data = data.copy()
    data["cluster"] = kmeans.fit_predict(scaled)

    # PCA to 2 components for scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    data["pca_x"] = coords[:, 0]
    data["pca_y"] = coords[:, 1]

    return data


def precompute_caches(data):
    global genre_stats_cache, audio_features_cache, clusters_cache, popularity_drivers_cache

    # ── Genre stats ──────────────────────────────────────────────────────────
    stats = []
    for genre in GENRES:
        gdf = data[data["genre"] == genre]
        stats.append({
            "genre": genre,
            "count": int(len(gdf)),
            "avg_popularity": round(float(gdf["popularity"].mean()), 1),
            "avg_danceability": round(float(gdf["danceability"].mean()), 3),
            "avg_energy": round(float(gdf["energy"].mean()), 3),
            "color": GENRE_COLORS[genre],
        })
    genre_stats_cache = stats

    # ── Audio features (per-genre averages) ──────────────────────────────────
    af = {}
    for genre in GENRES:
        gdf = data[data["genre"] == genre]
        af[genre] = {feat: round(float(gdf[feat].mean()), 3) for feat in AUDIO_FEATURES}
    audio_features_cache = af

    # ── Clusters (scatter plot data) ─────────────────────────────────────────
    clusters = []
    sample = data.sample(min(len(data), 2000), random_state=42)
    for _, row in sample.iterrows():
        clusters.append({
            "x": round(float(row["pca_x"]), 4),
            "y": round(float(row["pca_y"]), 4),
            "genre": row["genre"],
            "track_name": row["track_name"],
            "artist": row["artist"],
            "popularity": round(float(row["popularity"]), 1),
            "cluster": int(row["cluster"]),
        })
    clusters_cache = clusters

    # ── Popularity drivers (correlations) ────────────────────────────────────
    corrs = []
    for feat in AUDIO_FEATURES + ["tempo", "loudness"]:
        c = float(data[feat].corr(data["popularity"]))
        corrs.append({"feature": feat, "correlation_with_popularity": round(c, 4)})
    corrs.sort(key=lambda x: abs(x["correlation_with_popularity"]), reverse=True)
    popularity_drivers_cache = corrs


# ──────────────────────────────────────────────
# Application startup
# ──────────────────────────────────────────────
with app.app_context():
    print("Generating synthetic Spotify dataset…")
    df = generate_dataset(2000)
    print("Running K-Means and PCA…")
    df = run_analysis(df)
    print("Pre-computing dashboard caches…")
    precompute_caches(df)
    print(f"Ready — {len(df)} tracks loaded.")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/genre-stats")
def genre_stats():
    return jsonify(genre_stats_cache)


@app.route("/api/audio-features")
def audio_features():
    return jsonify(audio_features_cache)


@app.route("/api/clusters")
def clusters():
    return jsonify(clusters_cache)


@app.route("/api/popularity-drivers")
def popularity_drivers():
    return jsonify(popularity_drivers_cache)


@app.route("/api/predict-popularity", methods=["POST"])
def predict_popularity():
    body = request.get_json(force=True)
    required = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"]

    for field in required:
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    user_vec = np.array([float(body[field]) for field in required])

    # Compute Euclidean distance from user vector to each genre's mean profile
    distances = {}
    for genre in GENRES:
        genre_vec = np.array([audio_features_cache[genre][feat] for feat in required])
        distances[genre] = float(np.linalg.norm(user_vec - genre_vec))

    sorted_genres = sorted(distances, key=lambda g: distances[g])
    best_genre = sorted_genres[0]
    similar_genres = sorted_genres[:3]

    # Weighted popularity prediction using genre means (inverse-distance weighted)
    total_weight = 0.0
    weighted_pop = 0.0
    for genre in GENRES:
        w = 1.0 / (distances[genre] + 1e-6)
        genre_avg_pop = next(s["avg_popularity"] for s in genre_stats_cache if s["genre"] == genre)
        weighted_pop += w * genre_avg_pop
        total_weight += w

    # Feature contributions: use correlation signs to nudge estimate
    feature_bonus = 0.0
    for feat, val in zip(required, user_vec):
        corr = next((c["correlation_with_popularity"] for c in popularity_drivers_cache if c["feature"] == feat), 0)
        genre_mean = audio_features_cache[best_genre][feat]
        delta = val - genre_mean
        feature_bonus += corr * delta * 20  # scale factor

    predicted = float(np.clip(weighted_pop / total_weight + feature_bonus, 0, 100))

    # Generate recommendation message
    energy_label = "high-energy" if user_vec[1] > 0.65 else ("low-energy" if user_vec[1] < 0.35 else "mid-tempo")
    dance_label = "very danceable" if user_vec[0] > 0.70 else ("not very danceable" if user_vec[0] < 0.35 else "moderately danceable")
    mood_label = "upbeat" if user_vec[2] > 0.65 else ("melancholic" if user_vec[2] < 0.35 else "balanced")

    recommendation = (
        f"This track has a {energy_label}, {dance_label}, {mood_label} profile "
        f"that fits {best_genre} music. "
        f"Tracks with this fingerprint typically score around {round(predicted)}/100 in popularity."
    )

    return jsonify({
        "predicted_popularity": round(predicted, 1),
        "similar_genres": similar_genres,
        "recommendation": recommendation,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
