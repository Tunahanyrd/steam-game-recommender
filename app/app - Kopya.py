import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import requests

H5_URL = "https://huggingface.co/datasets/Tunahanyrd/steam-game-recommendation/blob/main/models/game_recommendation.h5"
H5_FILE = "game_recommendation.h5"

@st.cache_data
def download_h5():
    h5_path = Path(__file__).resolve().parent.parent / "data" / H5_FILE

    if not h5_path.exists():
        with st.spinner("📥 Model files downloading please wait a minute"):
            response = requests.get(H5_URL, stream=True)
            if response.status_code == 200:
                os.makedirs(h5_path.parent, exist_ok=True)  # data klasörü yoksa oluştur
                with open(h5_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                st.success("✅ File downloaded")
            else:
                st.error(f"🚨 HTTP error! Error code: {response.status_code}")
                return None
    return h5_path

<<<<<<< HEAD
# 📌 Load saved data
@st.cache_data
def load_hdf5():
    try:
        data_dir = Path(__file__).resolve().parent.parent / "data"

        hdf5_path = data_dir / "game_recommendation.h5"

        if not hdf5_path.exists():
            st.error("🚨 Required data file not found! Please check `game recommendation.h5` file.")
            return None, None

        with pd.HDFStore(hdf5_path, "r") as store:
=======
@st.cache_data
def load_hdf5():
    try:
        h5_path = download_h5()
        if h5_path is None:
            return None, None

        with pd.HDFStore(h5_path, "r") as store:
>>>>>>> 9efe386 (Updated H5 file handling & Hugging Face download integration)
            df = store["df"]
            similarity_matrix = store["similarity_matrix"].values  

            vector_columns = [
                "developers_vector", "publishers_vector", "category_vector", 
                "genre_vector", "tags_matrix", "tags_tfidf_matrix", 
                "feature_matrix", "final_feature_vectors", "short_desc_matrix"
            ]
            for col in vector_columns:
                if col in df.columns:
<<<<<<< HEAD
                    df[col] = df[col].astype(object) 
=======
                    df[col] = df[col].astype(object)
>>>>>>> 9efe386 (Updated H5 file handling & Hugging Face download integration)

        return df, similarity_matrix

    except Exception as e:
<<<<<<< HEAD
        st.error(f"⚠️ An error occurred while loading data: {e}")
=======
        st.error(f"⚠️ Data importing failed: {e}")
>>>>>>> 9efe386 (Updated H5 file handling & Hugging Face download integration)
        return None, None

df, similarity_matrix = load_hdf5()



df, similarity_matrix = load_hdf5()

if df is None or similarity_matrix is None:
    st.stop()


def recommend_games(game_id, top_n=10, min_similarity=0.5):
    """
    Suggests similar games for the specified `game_id`.
    
    Args:
        game_id (int): Steam App ID.
        top_n (int): Number of recommendation
        min_similarity (float)
        
    """
    if game_id not in df["app_id"].values:
        st.error("⚠️ Recommendations for this game cannot be calculated. Please try another game.!")
        return None

    app_id_to_index = {app_id: i for i, app_id in enumerate(df["app_id"].values)}

    target_idx = app_id_to_index.get(game_id, None)
    
    if target_idx is None:
        st.error("⚠️ Invalid game ID!")
        return None

    if target_idx >= similarity_matrix.shape[0]:
        st.error("⚠️ Recommendations for this game cannot be calculated. Please try another game.")
        return None

    sim_scores = list(enumerate(similarity_matrix[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sim_scores[1:]:  
        if score >= min_similarity:
            recommendations.append((df.iloc[idx]["app_id"], df.iloc[idx]["name"], score))
        if len(recommendations) >= top_n:
            break

    return recommendations


st.title("🎮 Game Recommendation")
st.markdown("**Enter your game Steam ID and see similar games!**")

game_id = st.text_input("Enter your game Steam ID:", "")

if st.button("Get Recommendation"):
    if game_id.isdigit():
        game_id = int(game_id)
        recommendations = recommend_games(game_id)

        if recommendations:
            st.markdown("### 📌 Recommended Games:")
            for app_id, name, similarity in recommendations:
                st.markdown(f"🔹 **{name}** (Similarity: {similarity:.3f})")
        else:
            st.error("ID is not found.")
    else:
        st.warning("Please enter a valid ID.!")
