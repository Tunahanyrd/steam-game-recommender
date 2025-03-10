import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

@st.cache_data
def load_hdf5():
    try:
        data_dir = Path(__file__).resolve().parent.parent / "data"

        hdf5_path = data_dir / "game_recommendation.h5"

        if not hdf5_path.exists():
            st.error("🚨 Required data file not found! Please check `game recommendation.h5` file.")
            return None, None

        with pd.HDFStore(hdf5_path, "r") as store:
            df = store["df"]
            similarity_matrix = store["similarity_matrix"].values  # DataFrame'den NumPy array'e çevir

            # Liste veya karmaşık veri tipleri içeren sütunları NumPy array formatına çevir
            vector_columns = [
                "developers_vector", "publishers_vector", "category_vector", 
                "genre_vector", "tags_matrix", "tags_tfidf_matrix", 
                "feature_matrix", "final_feature_vectors", "short_desc_matrix"
            ]
            for col in vector_columns:
                if col in df.columns:
                    df[col] = df[col].astype(object)  # NumPy array olarak sakla

        return df, similarity_matrix

    except Exception as e:
        st.error(f"⚠️ Veri yüklenirken hata oluştu: {e}")
        return None, None


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
        st.error("⚠️ Bu ID'ye sahip oyun bulunamadı!")
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

game_id = st.text_input("Enter your game Steam ID", "")

if st.button("Get Recommendation"):
    if game_id.isdigit():
        game_id = int(game_id)
        recommendations = recommend_games(game_id)

        if recommendations:
            st.markdown("### 📌 Recommended games:")
            for app_id, name, similarity in recommendations:
                st.markdown(f"🔹 **{name}** (Benzerlik: {similarity:.3f})")
        else:
            st.error("ID is not found.")
    else:
        st.warning(""Please enter a valid ID.!")
