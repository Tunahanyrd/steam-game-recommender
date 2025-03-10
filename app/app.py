import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

@st.cache_data
def load_hdf5():
    try:
        # Hugging Face Hub'dan dosyayı indirin
        hdf5_path = hf_hub_download(
            repo_id="Tunahanyrd/steam-game-recommendation",  # Hugging Face'deki repo adınız
            filename="models/game_recommendation.h5"         # İndirilecek dosyanın yolu
        )
        
        # İndirilen dosya ile veriyi yükleyin
        with pd.HDFStore(hdf5_path, "r") as store:
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
                    
        return df, similarity_matrix

    except Exception as e:
        st.error(f"⚠️ Veri yüklenirken hata oluştu: {e}")
        return None, None

df, similarity_matrix = load_hdf5()

if df is None or similarity_matrix is None:
    st.stop()

# Öneri fonksiyonu
def recommend_games(game_id, top_n=10, min_similarity=0.5):
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
    for idx, score in sim_scores[1:]:
        if score >= min_similarity:
            recommendations.append((df.iloc[idx]["app_id"], df.iloc[idx]["name"], score))
        if len(recommendations) >= top_n:
            break

    return recommendations

# Uygulama arayüzü
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
