import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ====== BASIC CONFIG ======
st.set_page_config(page_title="🎧 Song Recommender", layout="wide")

# ====== CSS STYLING ======
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: "Poppins", sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00b4db, #0083b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
    }
    .sub-title {
        text-align: center;
        color: #c9c9c9;
        font-size: 18px;
        margin-bottom: 2em;
    }
    .song-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 5px solid #00b4db;
    }
    </style>
""", unsafe_allow_html=True)

# ====== DATASET ======
df = pd.read_csv("spotify_tracks.csv")

# Rename columns according to your dataset
df = df.rename(columns={
    'track_name': 'Song',
    'artist_name': 'Artist',
    'danceability': 'Danceability',
    'energy': 'Energy',
    'tempo': 'Tempo',
    'valence': 'Valence',
    'genre': 'Genre',
    'popularity': 'Popularity'
})

# Keep only relevant columns
df = df[['Song', 'Artist', 'Genre', 'Danceability', 'Energy', 'Tempo', 'Valence', 'Popularity']]

# Handle missing and duplicate values
df = df.dropna(subset=['Danceability', 'Energy', 'Tempo', 'Valence'])
df = df.drop_duplicates(subset=['Song'])

# Optional: subsample for performance (if dataset is large)
if len(df) > 20000:
    df = df.sample(20000, random_state=42).reset_index(drop=True)

# ====== ML PART ======
features = ['Danceability', 'Energy', 'Tempo', 'Valence']
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])

def recommend(song_name, n=5):
    if song_name not in df['Song'].values:
        return []
    idx = df[df['Song'] == song_name].index[0]
    song_vec = scaled[idx].reshape(1, -1)
    scores = cosine_similarity(song_vec, scaled).flatten()
    rec_idx = np.argsort(scores)[::-1][1:n+1]
    recs = [(df.iloc[i]['Song'], df.iloc[i]['Artist'], round(scores[i], 3)) for i in rec_idx]
    return recs

# ====== UI ======
st.markdown('<h1 class="main-title">🎶 Song Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Machine Learning (Cosine Similarity)</p>', unsafe_allow_html=True)

song_choice = st.selectbox("🎧 Select a song you like:", sorted(df['Song'].unique()))

if st.button("🎵 Recommend Similar Songs"):
    recs = recommend(song_choice)
    if not recs:
        st.warning("⚠️ Song not found in dataset. Try another one.")
    else:
        st.subheader(f"Because you liked **{song_choice}**, you might also enjoy:")

        for song, artist, score in recs:
            st.markdown(
                f"""
                <div class="song-card">
                    <b>🎧 {song}</b> — {artist}<br>
                    <small>Similarity Score: {score}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====== Visualization ======
        fig, ax = plt.subplots()
        songs = [r[0] for r in recs]
        scores = [r[2] for r in recs]
        ax.barh(songs, scores, color='#00b4db')
        ax.set_xlabel("Cosine Similarity Score", color='white')
        ax.set_ylabel("Recommended Songs", color='white')
        ax.set_title(f"Top {len(songs)} Similar Songs", color='white')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.invert_yaxis()
        st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Built by <b>Greek Kumar</b> | Uses ML to find song similarity 🎧</p>", unsafe_allow_html=True)

