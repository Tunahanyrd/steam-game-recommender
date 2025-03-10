import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# 📌 KAYDEDİLEN VERİLERİ YÜKLE
@st.cache_data
def load_hdf5():
    try:
        # Klasör konumunu belirle (Bir üst klasöre çık ve 'data' klasörüne gir)
        data_dir = Path(__file__).resolve().parent.parent / "data"

        # Dosya yollarını oluştur
        hdf5_path = data_dir / "game_recommendation.h5"

        # Eğer dosya yoksa hata ver
        if not hdf5_path.exists():
            st.error("🚨 Gerekli veri dosyası bulunamadı! Lütfen `game_recommendation.h5` dosyasını kontrol edin.")
            return None, None

        # HDF5 dosyasını yükle
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

# Eğer veri yüklenemediyse, hata mesajı göster ve programı durdur
if df is None or similarity_matrix is None:
    st.stop()


def recommend_games(game_id, top_n=10, min_similarity=0.5):
    """
    Belirtilen `game_id` için en benzer oyunları önerir.
    
    Args:
        game_id (int): Steam App ID.
        top_n (int): Kaç öneri getirileceği.
        min_similarity (float): Minimum benzerlik skoru.
        
    Returns:
        List of tuples: [(önerilen_oyun_app_id, önerilen_oyun_adı, benzerlik_skoru), ...]
    """
    # Oyunun DataFrame içinde olup olmadığını kontrol edelim
    if game_id not in df["app_id"].values:
        st.error("⚠️ Bu ID'ye sahip oyun bulunamadı!")
        return None

    # `app_id` -> `index` haritası oluştur
    app_id_to_index = {app_id: i for i, app_id in enumerate(df["app_id"].values)}

    # Hedef oyunun index'ini al
    target_idx = app_id_to_index.get(game_id, None)
    
    if target_idx is None:
        st.error("⚠️ Geçersiz oyun ID!")
        return None

    # Eğer index, similarity_matrix boyutunu aşıyorsa hata önleyelim
    if target_idx >= similarity_matrix.shape[0]:
        st.error("⚠️ Bu oyun için öneri hesaplanamıyor. Lütfen başka bir oyun deneyin.")
        return None

    # Benzerlik skorlarını al
    sim_scores = list(enumerate(similarity_matrix[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Kendi oyununu çıkar ve en benzer oyunları al
    recommendations = []
    for idx, score in sim_scores[1:]:  # İlk oyun kendisi olduğu için atlıyoruz
        if score >= min_similarity:
            recommendations.append((df.iloc[idx]["app_id"], df.iloc[idx]["name"], score))
        if len(recommendations) >= top_n:
            break

    return recommendations


# Streamlit UI Başlangıç
st.title("🎮 Oyun Öneri Sistemi")
st.markdown("**Oyun Steam ID'sini girin ve benzer oyunları görün!**")

# Kullanıcının oyun ID girmesi için input kutusu
game_id = st.text_input("Oyun Steam ID'sini girin:", "")

if st.button("Önerileri Getir"):
    if game_id.isdigit():
        game_id = int(game_id)
        recommendations = recommend_games(game_id)

        if recommendations:
            st.markdown("### 📌 Önerilen Oyunlar:")
            for app_id, name, similarity in recommendations:
                st.markdown(f"🔹 **{name}** (Benzerlik: {similarity:.3f})")
        else:
            st.error("Bu ID'ye ait oyun bulunamadı veya öneri yapılamadı.")
    else:
        st.warning("Lütfen geçerli bir Steam ID girin!")
