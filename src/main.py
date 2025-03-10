#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 17:52:35 2025
Author: Tunahan

Description:
------------
Bu proje, Steam verilerinden oyun öneri sistemi oluşturmak için hazırlanmıştır.
Veriler "games.json" dosyasından okunur, temizlenir ve çeşitli özellikler vektörleştirilir:
    - İnceleme puanı hesaplaması: Wilson Score kullanılarak.
    - Metacritic skoru: Eksik değerler kullanıcı puanlarına dayalı doldurulur.
    - Yayın tarihi: Yıldan, oyunun çıkış yılından itibaren geçen süre (2025 - release_year) hesaplanır.
    - Kısa açıklamalar: TF-IDF ile vektörleştirilir.
    - Etiketler (tags): DictVectorizer veya TF-IDF kullanılarak sayısal vektöre dönüştürülür.
    - Developers, Publishers, Categories ve Genres: Word2Vec ile vektörleştirilip, ortalama vektörler hesaplanır.
Son olarak, tüm özellikler belirli ağırlıklarla birleştirilip tek bir birleşik özellik vektörü elde edilir ve cosine similarity üzerinden öneriler sunulur.
"""

import pandas as pd
import numpy as np
import ast
from scipy.stats import norm
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. DATA LOAD & CLEANING
# =============================================================================
# "games.json" dosyasını, index olarak app_id'leri kullanarak oku
df = pd.read_json(r"../data/games.json", orient="index")
df.index.rename("app_id", inplace=True)

# Kullanılmayacak sütunları kaldırıyoruz
drop_columns = [
    "dlc_count", "detailed_description", "about_the_game", "reviews",
    "recommendations", "header_image", "website", "required_age",
    "support_url", "support_email", "metacritic_url", "achievements",
    "full_audio_languages", "packages", "screenshots", "movies",
    "user_score", "score_rank", "average_playtime_forever", 
    "average_playtime_2weeks", "median_playtime_forever", 
    "median_playtime_2weeks", "peak_ccu", "notes", "supported_languages",
    "windows", "mac", "linux", "price"
]
df = df.drop(columns=drop_columns)

# Eğer index steam id içeriyorsa, index'i resetleyip "app_id" sütunu olarak saklayalım
df = df.reset_index()
df = df.rename(columns={"index": "app_id"})

# =============================================================================
# 2. ESTIMATED OWNERS İŞLEMLERİ
# =============================================================================
def transform_estimated_owners(x):
    """
    'estimated_owners' sütunundaki aralık değerlerini (ör. "20000 - 50000") 
    aralığın ortalamasına çevirir. "0 - 0" değerlerini NaN yapar.
    """
    if isinstance(x, str):
        x = x.strip()
        if x == "0 - 0":
            return np.nan
        try:
            lower, upper = x.split('-')
            lower = int(lower.strip())
            upper = int(upper.strip())
            return (lower + upper) / 2  # Aralığın ortalaması
        except Exception:
            return np.nan
    return np.nan

df['estimated_owners_numeric'] = df['estimated_owners'].apply(transform_estimated_owners)
df = df.dropna(subset=['estimated_owners_numeric'])

# Normalize edilmiş estimated_owners değeri (popülarite göstergesi)
scaler = MinMaxScaler()
df['estimated_owners_normalized'] = scaler.fit_transform(df[['estimated_owners_numeric']])
# İhtiyacımız kalmadığı için ilgili sütunları kaldırıyoruz
df = df.drop(columns=['estimated_owners', 'estimated_owners_numeric'])

# =============================================================================
# 3. RELEASE DATE -> RELEASE YEAR
# =============================================================================
# Release tarihi yıl olarak alınır, ardından 2025 - release_year hesaplanarak "yaş" elde edilir.
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df.drop(columns=['release_date'])
df['release_year'] = 2025 - df['release_year']

# =============================================================================
# 4. REVIEW SCORE VE WILSON SCORE HESAPLAMASI
# =============================================================================
# Toplam inceleme sayısı
df["total_reviews"] = df["positive"] + df["negative"]

# Pozitif oran (p) hesaplanır
df["p"] = df["positive"] / (df["total_reviews"] + 1e-6)
z = norm.ppf(0.975)  # %95 güven düzeyi için z değeri

# Penalty_factor ile düşük inceleme sayısına sahip oyunlara ekstra ceza uygulanır
penalty_factor = 2.5
df["positive_rate_wilson"] = (
    (df["p"] + (z**2) / (2 * df["total_reviews"])) -
    (penalty_factor * z * np.sqrt((df["p"] * (1 - df["p"]) + (z**2) / (4 * df["total_reviews"])) / (df["total_reviews"] + 1e-6)))
) / (1 + (z**2) / df["total_reviews"])

# İnceleme sayısının etkisini log ile ağırlıklandırarak final puan elde edilir.
df["positive_rate"] = df["positive_rate_wilson"] * np.log1p(df["total_reviews"])

# Kullanmayacağımız sütunları kaldırıyoruz.
drop_cols = ["total_reviews", "positive", "negative", "positive_rate_wilson", "p"]
df = df.drop(columns=drop_cols)

# =============================================================================
# 5. METACRITIC SCORE İŞLEMLERİ
# =============================================================================
# Sıfır olan metacritic_score değerlerini NaN yapıp, positive_rate üzerinden dolduruyoruz.
df["metacritic_score"] = df["metacritic_score"].replace(0, np.nan)
df["metacritic_score"] = df["metacritic_score"].fillna(df["positive_rate"] * 9 + 5)
# Ölçeklendirme: Burada 120’ye bölüyoruz (deneme-yanılma ile ayarlanabilir)
df["metacritic_score"] = df["metacritic_score"] / 120
# Belirli bir eşik değerinin altındaki oyunları filtreleyelim 
df = df[df["metacritic_score"] > 0.25]
# "positive_rate" sütununu artık kullanmayacağımız için kaldırıyoruz.
df = df.drop(columns="positive_rate")

# =============================================================================
# 6. KATEGORİ VE GENRE İŞLEMLERİ
# =============================================================================
# Sütunların string olarak gelmesi durumunda ast.literal_eval ile listeye çeviriyoruz.
df["categories"] = df["categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# =============================================================================
# 7. DEVELOPERS VE PUBLISHERS VECTÖRLEŞTİRİLMESİ (Word2Vec)
# =============================================================================
def tokenize_names(names):
    """
    Geliştirici veya yayıncı isimlerini normalize edip tokenize eder.
    Örnek: "Ubisoft Montreal" -> "ubisoft_montreal"
    """
    if isinstance(names, list):
        return [name.strip().lower().replace(" ", "_") for name in names]
    return []

# Developers için
df["developers_tokenized"] = df["developers"].apply(lambda x: tokenize_names(x))
dev_corpus = df["developers_tokenized"].tolist()
dev_model = Word2Vec(sentences=dev_corpus, vector_size=8, window=5, min_count=1, workers=4)
def get_average_vector(tokens, model, vector_size=8):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
df["developers_vector"] = df["developers_tokenized"].apply(lambda tokens: get_average_vector(tokens, dev_model))

# Publishers için
df["publishers_tokenized"] = df["publishers"].apply(lambda x: tokenize_names(x))
pub_corpus = df["publishers_tokenized"].tolist()
pub_model = Word2Vec(sentences=pub_corpus, vector_size=8, window=5, min_count=1, workers=4)
df["publishers_vector"] = df["publishers_tokenized"].apply(lambda tokens: get_average_vector(tokens, pub_model))

# =============================================================================
# 8. CATEGORIES VE GENRES VECTÖRLEŞTİRİLMESİ (Word2Vec)
# =============================================================================
# Eğer veriler liste değilse, boş liste kabul ediyoruz.
df["categories"] = df["categories"].apply(lambda x: x if isinstance(x, list) else [])
df["genres"] = df["genres"].apply(lambda x: x if isinstance(x, list) else [])

cat_corpus = df["categories"].tolist()
genre_corpus = df["genres"].tolist()
combined_corpus = cat_corpus + genre_corpus
cat_genre_model = Word2Vec(sentences=combined_corpus, vector_size=8, window=3, min_count=1, workers=4)
def get_vector_representation(tags, model, vector_size=8):
    vectors = [model.wv[tag] for tag in tags if tag in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
df["category_vector"] = df["categories"].apply(lambda x: get_vector_representation(x, cat_genre_model))
df["genre_vector"] = df["genres"].apply(lambda x: get_vector_representation(x, cat_genre_model))

# =============================================================================
# 9. TAGS VE SHORT DESCRIPTION VECTÖRLEŞTİRİLMESİ
# =============================================================================
# (A) SHORT DESCRIPTION: TF-IDF kullanılarak vektörleştiriliyor.
tfidf = TfidfVectorizer(max_features=100)
short_desc_matrix = tfidf.fit_transform(df['short_description'].fillna("")).toarray()

# (B) TAGS: Önce tags sütununu uygun formata çevirelim.
def convert_tags(x):
    """
    tags sütunundaki veriyi; dictionary, liste veya string formatında alıp 
    dictionary formatına çevirir.
    """
    if isinstance(x, dict):
        return x
    elif isinstance(x, list):
        return {tag: 1 for tag in x}
    else:
        try:
            converted = ast.literal_eval(x)
            if isinstance(converted, dict):
                return converted
            elif isinstance(converted, list):
                return {tag: 1 for tag in converted}
            else:
                return {}
        except:
            return {}
df["tags"] = df["tags"].apply(convert_tags)
# DictVectorizer ile tags'leri vektörleştiriyoruz.
vec = DictVectorizer(sparse=False)
tags_matrix = vec.fit_transform(df["tags"])

# Ayrıca, etiketlerin anahtarlarını bir metin haline getiren alternatif TF-IDF yöntemi:
def tags_to_text(tags):
    if isinstance(tags, dict):
        return " ".join(tags.keys())
    return ""
df["tags_text"] = df["tags"].apply(tags_to_text)
tfidf_tags = TfidfVectorizer(max_features=100)
tags_tfidf_matrix = tfidf_tags.fit_transform(df["tags_text"]).toarray()

# =============================================================================
# 10. ÖZELLİK BİRLEŞTİRME
# =============================================================================
# release_year sütununu normalize edelim:
scaler_year = MinMaxScaler()
df["release_year_norm"] = scaler_year.fit_transform(df[['release_year']].fillna(0))

# Ağırlık değerleri (deneme-yanılma ile ayarlanabilir)
weights = {
    "metacritic": 0.75,  
    "release_year": 1.0,  
    "short_desc": 2.5,  
    "tags": 3.5,  
    "developers": 1.2,  
    "publishers": 1.2,  
    "category": 1.2,  
    "genre": 2.4,  
}


def combine_features(row):
    """
    Scalar (metacritic_score, release_year_norm) ve Word2Vec tabanlı
    vektörleri (developers, publishers, category, genre) ağırlıklandırarak 
    birleştirir.
    """
    return np.concatenate([
        np.array([row["metacritic_score"]]) * weights["metacritic"],
        np.array([row["release_year_norm"]]) * weights["release_year"],
        row["developers_vector"] * weights["developers"],
        row["publishers_vector"] * weights["publishers"],
        row["category_vector"] * weights["category"],
        row["genre_vector"] * weights["genre"],
    ])

# Her oyunun temel vektörünü oluşturuyoruz.
basic_feature_vectors = df.apply(lambda row: combine_features(row), axis=1).tolist()

# Kısa açıklama ve tags vektörlerini ekleyerek nihai özellik vektörünü oluşturuyoruz.
final_feature_vectors = []
for i in range(len(df)):
    vec_basic = basic_feature_vectors[i]
    vec_short = short_desc_matrix[i] * weights["short_desc"]
    vec_tags = tags_matrix[i] * weights["tags"]
    full_vector = np.concatenate([vec_basic, vec_short, vec_tags])
    final_feature_vectors.append(full_vector)
df["feature_vector"] = final_feature_vectors

# =============================================================================
# 11. BENZERLİK MATRISİNİN HESAPLANMASI
# =============================================================================
# Her oyunun birleşik özellik vektörünü bir matris haline getiriyoruz.
feature_matrix = np.vstack(df["feature_vector"].values)
similarity_matrix = cosine_similarity(feature_matrix)

# =============================================================================
# 12. ÖNERİ FONKSİYONU
# =============================================================================
def recommend_games(game_index, top_n=10, min_similarity=0.5):
    """
    Belirtilen index'teki oyuna benzer oyunları cosine similarity matrisine
    göre sıralar.
    
    Args:
        game_index (int): DataFrame içindeki sıralı index (0,1,2,...).
        top_n (int): Döndürülecek öneri sayısı.
        min_similarity (float): Öneri için minimum benzerlik eşiği.
        
    Returns:
        List of tuples: [(önerilen_oyun_index, similarity_score), ...]
    """
    if game_index < 0 or game_index >= similarity_matrix.shape[0]:
        print("Geçersiz oyun index'i.")
        return []
    
    sim_scores = list(enumerate(similarity_matrix[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for idx, score in sim_scores[1:]:
        if score >= min_similarity:
            recommendations.append((idx, score))
        if len(recommendations) >= top_n:
            break
    return recommendations

# =============================================================================
# 13. ÖRNEK KULLANIM: KULLANICI İLE ETKİLEŞİM (While Loop)
# =============================================================================
if __name__ == '__main__':
    print("Oyun Öneri Sistemi Başlatıldı. Çıkmak için 'exit' yazın.\n")
    while True:
        target_input = input("Target ID giriniz: ").strip()
        if target_input.lower() == 'exit':
            break
        try:
            target = int(target_input)
        except ValueError:
            print("Geçersiz ID. Lütfen sayı giriniz.")
            continue
        target_game = df[df["app_id"] == target]
        if target_game.empty:
            print(f"{target} app_id'li oyun bulunamadı!")
        else:
            print("\nHedef Oyun:")
            print(target_game[["app_id", "name"]])
            
            # DataFrame içindeki sıralı pozisyonu alıyoruz.
            target_pos = df.index.get_loc(target_game.index[0])
            recommendations = recommend_games(target_pos, top_n=10, min_similarity=0.5)
            if not recommendations:
                print("\nÖnerilen oyun bulunamadı veya geçersiz index!")
            else:
                print("\nÖnerilen Oyunlar:")
                for rec_index, score in recommendations:
                    rec_game = df.iloc[rec_index][["app_id", "name"]]
                    print(f"{rec_game['app_id']} - {rec_game['name']} (Benzerlik: {score:.3f})")
        print("\n-----------------------------------\n")
        
