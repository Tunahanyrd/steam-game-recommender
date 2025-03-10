import streamlit as st
import pandas as pd
import numpy as np
import h5py
import requests
from pathlib import Path

# Function to convert a key to a string.
def key_to_str(key):
    if isinstance(key, tuple):
        # If the key is a tuple, take its first element.
        return key[0]
    return key

# Download the HDF5 file from Hugging Face if it doesn't exist locally.
@st.cache_data
def download_h5():
    # Define the local path for the HDF5 file.
    h5_path = Path("game_recommendation.h5")
    
    # If the file already exists, return its path.
    if h5_path.exists():
        return h5_path

    hf_url = "https://huggingface.co/datasets/Tunahanyrd/steam-game-recommendation/resolve/main/models/game_recommendation.h5"
    
    try:
        # Download the file in streaming mode.
        with requests.get(hf_url, stream=True) as r:
            r.raise_for_status()
            with open(h5_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return h5_path
    except Exception as e:
        st.error(f"🚨 HDF5 file download failed: {e}")
        return None

# Load the HDF5 file and extract the DataFrame and similarity matrix.
@st.cache_data
def load_hdf5():
    try:
        h5_path = download_h5()
        if h5_path is None:
            return None, None

        with h5py.File(h5_path, "r") as file:
            # Initialize a dictionary to build the DataFrame.
            df_dict = {}
            similarity_key = None

            # Iterate through all keys in the file.
            for key in file.keys():
                key_str = key_to_str(key)
                if key_str == "similarity_matrix":
                    similarity_key = key_str
                    continue
                df_dict[key_str] = file[key_str][()]

            # Ensure the similarity matrix key was found.
            if similarity_key is None:
                st.error("⚠️ Similarity matrix not found in the HDF5 file.")
                return None, None

            # Create the DataFrame.
            df = pd.DataFrame(df_dict)
            
            # Decode byte columns if needed.
            for col in df.select_dtypes(include=[np.object_, bytes]):
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            
            # Convert specific columns to numpy arrays if they are stored as lists.
            vector_columns = [
                "developers_vector", "publishers_vector", "category_vector", 
                "genre_vector", "tags_matrix", "tags_tfidf_matrix", 
                "feature_matrix", "final_feature_vectors", "short_desc_matrix"
            ]
            for col in vector_columns:
                if col in df.columns and isinstance(df[col].iloc[0], list):
                    df[col] = df[col].apply(np.array)
            
            # Extract the similarity matrix using the found key.
            similarity_matrix = file[similarity_key][:]
            
        return df, similarity_matrix

    except Exception as e:
        st.error(f"⚠️ Data importing error: {e}")
        return None, None

# Load the data.
df, similarity_matrix = load_hdf5()

if df is None or similarity_matrix is None:
    st.stop()

# Function to recommend similar games based on a given game ID.
def recommend_games(game_id, top_n=10, min_similarity=0.5):
    """
    Recommend similar games based on a provided game ID.

    Args:
        game_id (int): The Steam game ID.
        top_n (int): The number of recommendations to return.
        min_similarity (float): The minimum similarity threshold.

    Returns:
        List of tuples with (app_id, game name, similarity score) for recommended games.
    """
    if game_id not in df["app_id"].values:
        st.error("⚠️ Recommendations for this game cannot be calculated. Please try another game.")
        return None

    # Create a mapping from app ID to DataFrame index.
    app_id_to_index = {app_id: i for i, app_id in enumerate(df["app_id"].values)}
    target_idx = app_id_to_index.get(game_id, None)

    if target_idx is None:
        st.error("⚠️ Invalid game ID!")
        return None

    if target_idx >= similarity_matrix.shape[0]:
        st.error("⚠️ Recommendations for this game cannot be calculated. Please try another game.")
        return None

    # Calculate similarity scores for the target game.
    sim_scores = list(enumerate(similarity_matrix[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    # Skip the first score (the game itself).
    for idx, score in sim_scores[1:]:
        if score >= min_similarity:
            recommendations.append((df.iloc[idx]["app_id"], df.iloc[idx]["name"], score))
        if len(recommendations) >= top_n:
            break

    return recommendations

# Streamlit UI
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
            st.error("Game ID not found.")
    else:
        st.warning("Please enter a valid ID.")
