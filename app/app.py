import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import requests
from pathlib import Path

@st.cache_data
def load_hdf5():
    try:
        # Huggingface URL'sinden dosyayı indiriyoruz.
        url = "https://huggingface.co/datasets/Tunahanyrd/steam-game-recommendation/resolve/main/models/game_recommendation.h5"
        response = requests.get(url)
        if response.status_code != 200:
            st.error("🚨 Veriler Huggingface'den alınamadı! Lütfen bağlantıyı kontrol edin.")
            return None, None

        # Güvenli geçici dosya işlemi için TemporaryDirectory kullanıyoruz.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "game_recommendation.h5"
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(response.content)
            # Geçici dosya üzerinden veriyi yüklüyoruz.
            with pd.HDFStore(tmp_path, "r") as store:
                df = store["df"]
                similarity_matrix = store["similarity_matrix"].values

                vector_columns = [
                    "developers_vector", "publishers_vector", "category_vector", 
                    "genre_vector", "tags_matrix", "tags_tfidf_matrix", 
                    "feature_matrix", "final_feature_vectors", "short_desc_matrix"
                ]
                for col in vector_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(object)
            # TemporaryDirectory bloğu bittiğinde dosya otomatik silinir.
            return df, similarity_matrix

    except Exception as e:
        st.error(f"⚠️ Veri yüklenirken hata oluştu: {e}")
        return None, None

df, similarity_matrix = load_hdf5()

if df is None or similarity_matrix is None:
    st.stop()

def recommend_games(game_id, top_n=10, min_similarity=0.5):
    """
    Belirtilen `game_id` için benzer oyunları önerir.
    
    Args:
        game_id (int): Steam App ID.
        top_n (int): Öneri sayısı
        min_similarity (float): Minimum benzerlik değeri
    """
    if game_id not in df["app_id"].values:
        st.error("⚠️ Bu ID'ye sahip bir oyun bulunamadı!")
        return None

    app_id_to_index = {app_id: i for i, app_id in enumerate(df["app_id"].values)}
    target_idx = app_id_to_index.get(game_id)
    
    if target_idx is None:
        st.error("⚠️ Geçersiz oyun ID'si!")
        return None

    if target_idx >= similarity_matrix.shape[0]:
        st.error("⚠️ Bu oyun için öneriler hesaplanamıyor. Lütfen başka bir oyun deneyin.")
        return None

    sim_scores = list(enumerate(similarity_matrix[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    # İlk skor hedef oyunu temsil ettiğinden atlanıyor.
    for idx, score in sim_scores[1:]:
        if score >= min_similarity:
            recommendations.append((df.iloc[idx]["app_id"], df.iloc[idx]["name"], score))
        if len(recommendations) >= top_n:
            break

    return recommendations

st.title("🎮 Oyun Önerisi")
st.markdown("**Steam ID'nizi girin ve benzer oyunları görün!**")

game_id = st.text_input("Oyun Steam ID'nizi girin", "")

if st.button("Önerileri Getir"):
    if game_id.isdigit():
        game_id = int(game_id)
        recommendations = recommend_games(game_id)

        if recommendations:
            st.markdown("### 📌 Önerilen Oyunlar:")
            for app_id, name, similarity in recommendations:
                st.markdown(f"🔹 **{name}** (Benzerlik: {similarity:.3f})")
        else:
            st.error("ID bulunamadı.")
    else:
        st.warning("Lütfen geçerli bir ID girin!")
