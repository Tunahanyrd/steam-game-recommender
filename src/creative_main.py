#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 17:52:35 2025
Author: Tunahan

Description:
------------
This project is designed to create a game recommendation system using Steam data.
The data is read from the "games.json" file, cleaned, and various features are vectorized:
    - Review score calculation: Using the Wilson Score.
    - Metacritic score: Missing values are filled based on user ratings.
    - Release date: The time elapsed since the game's release year is calculated as (2025 - release_year).
    - Short descriptions: Vectorized using TF-IDF.
    - Tags: Converted into numerical vectors using DictVectorizer or TF-IDF.
    - Developers, Publishers, Categories, and Genres: Vectorized with Word2Vec, and average vectors are computed.
Finally, all features are combined with specific weights to obtain a single unified feature vector, and recommendations are generated using cosine similarity.
"""
import requests
import pandas as pd
import ast
import numpy as np
from scipy.stats import norm
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

STEAM_API_KEY = ""

# =============================================================================
# 1. DATA LOAD & CLEANING
# =============================================================================
# Read the "games.json" file using app_ids as indexes
df = pd.read_json(r"../../steam-game-recommendation/data/games.json", orient="index")

# Remove unused columns
drop_columns = [
    "dlc_count", "detailed_description", "reviews",
    "recommendations", "header_image", "website", "required_age",
    "support_url", "support_email", "metacritic_url", "achievements",
    "full_audio_languages", "packages", "screenshots", "movies",
    "user_score", "score_rank", "average_playtime_forever", 
    "average_playtime_2weeks", "median_playtime_forever", 
    "median_playtime_2weeks", "peak_ccu", "notes", "supported_languages",
    "windows", "mac", "linux", "price"
]
df = df.drop(columns=drop_columns)

df.index.rename("app_id", inplace=True)
# =============================================================================
# 2. ESTIMATED OWNERS PROCESS
# =============================================================================
def transform_estimated_owners(x):
    """
    Converts ranges in column 'estimated_owners' to numerical (e.g. "20000 - 50000").
    Values "0 - 0" become NaN.
    """
    if isinstance(x, str):
        x = x.strip()
        if x == "0 - 0":
            return np.nan
        try:
            lower, upper = x.split('-')
            lower = int(lower.strip())
            upper = int(upper.strip())
            return (lower + upper) / 2  # Mean of the interval
        except Exception as e:
            print(f"An error occured from transforming estimated owners: {e}")
            return np.nan
    return np.nan

df['estimated_owners_numeric'] = df['estimated_owners'].apply(transform_estimated_owners)
df = df.dropna(subset=['estimated_owners_numeric'])

# Normalized estimated_owners values (popularity indicator)
scaler = MinMaxScaler()
df['estimated_owners_normalized'] = scaler.fit_transform(df[['estimated_owners_numeric']])
df = df.drop(columns=['estimated_owners', 'estimated_owners_numeric'])

# =============================================================================
# 3. RELEASE DATE -> RELEASE YEAR
# =============================================================================
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df.drop(columns=['release_date'])
df['release_year'] = 2025 - df['release_year']

# =============================================================================
# 4. REVIEW SCORE VE WILSON SCORE PROCESSİNG
# =============================================================================
df["total_reviews"] = df["positive"] + df["negative"]

df["p"] = df["positive"] / (df["total_reviews"] + 1e-6)
z = norm.ppf(0.975)  # 95% confidence

penalty_factor = 2.5
df["positive_rate_wilson"] = (
    (df["p"] + (z**2) / (2 * df["total_reviews"])) -
    (penalty_factor * z * np.sqrt((df["p"] * (1 - df["p"]) + (z**2) / (4 * df["total_reviews"])) / (df["total_reviews"] + 1e-6)))
) / (1 + (z**2) / df["total_reviews"])

df["positive_rate"] = df["positive_rate_wilson"] * np.log1p(df["total_reviews"])

drop_cols = ["total_reviews", "positive", "negative", "positive_rate_wilson", "p"]
df = df.drop(columns=drop_cols)

# =============================================================================
# 5. METACRITIC SCORE PROCESSING
# =============================================================================
df["metacritic_score"] = df["metacritic_score"].replace(0, np.nan)
df["metacritic_score"] = df["metacritic_score"].fillna(df["positive_rate"] * 9 + 5)
df["metacritic_score"] = df["metacritic_score"] / 121
df = df[df["metacritic_score"] > 0.25]
df = df.drop(columns="positive_rate")

# =============================================================================
# 6. CATEGORIES AND GENRES PROCESSING
# =============================================================================
df["categories"] = df["categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# =============================================================================
# 7. DEVELOPERS AND PUBLISHERS VECTORIZATıON (Word2Vec)
# =============================================================================
def tokenize_names(names):
    """
    Developers or publishers name normalizes and tokenizes.
    Sample: "Ubisoft Montreal" -> "ubisoft_montreal"
    """
    if isinstance(names, list):
        return [name.strip().lower().replace(" ", "_") for name in names]
    return []

df["developers_tokenized"] = df["developers"].apply(lambda x: tokenize_names(x))
dev_corpus = df["developers_tokenized"].tolist()
dev_model = Word2Vec(sentences=dev_corpus, vector_size=8, window=5, min_count=1, workers=4)

def get_average_vector(tokens, model, vector_size=8):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df["developers_vector"] = df["developers_tokenized"].apply(lambda tokens: get_average_vector(tokens, dev_model))

df["publishers_tokenized"] = df["publishers"].apply(lambda x: tokenize_names(x))
pub_corpus = df["publishers_tokenized"].tolist()
pub_model = Word2Vec(sentences=pub_corpus, vector_size=8, window=5, min_count=1, workers=4)
df["publishers_vector"] = df["publishers_tokenized"].apply(lambda tokens: get_average_vector(tokens, pub_model))

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
# 9. TAGS VE SHORT DESCRIPTION VECTORIZATION
# =============================================================================
tfidf = TfidfVectorizer(max_features=500)
short_desc_matrix = tfidf.fit_transform(df['short_description'].fillna("")).toarray()

def convert_tags(x):
    """
    Converts 'tags' column to dict format.
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

vec = DictVectorizer(sparse=False)
tags_matrix = vec.fit_transform(df["tags"])

def tags_to_text(tags):
    if isinstance(tags, dict):
        return " ".join(tags.keys())
    return ""

df["tags_text"] = df["tags"].apply(tags_to_text)
tfidf_tags = TfidfVectorizer(max_features=100)
tags_tfidf_matrix = tfidf_tags.fit_transform(df["tags_text"]).toarray()

# =============================================================================
# 10. FEATURE MERGING (BASIC FEATURES)
# =============================================================================
scaler_year = MinMaxScaler()
df["release_year_norm"] = scaler_year.fit_transform(df[['release_year']].fillna(0))

weights = {
    "metacritic": 1.0,  
    "release_year": 15.0,  
    "short_desc": 15.0,  
    "tags": 15.0,  
    "developers": 2.5,  
    "publishers": 4.0,  
    "category": 10.0,  
    "genre": 10.0,  
}

def combine_features(row):
    """
    Combines scalar features (metacritic_score, release_year_norm) 
    with Word2Vec-based vectors (developers, publishers, category, genre) 
    by applying weighted aggregation.
    """
    return np.concatenate([
        np.array([row["metacritic_score"]]) * weights["metacritic"],
        np.array([row["release_year_norm"]]) * weights["release_year"],
        row["developers_vector"] * weights["developers"],
        row["publishers_vector"] * weights["publishers"],
        row["category_vector"] * weights["category"],
        row["genre_vector"] * weights["genre"],
    ])

basic_feature_vectors = df.apply(lambda row: combine_features(row), axis=1).tolist()
basic_feature_matrix = np.vstack(basic_feature_vectors)

# =============================================================================
# 11. CALCULATING SIMILARITY MATRİX (SEPARATE FUSION)
# =============================================================================
basic_feature_matrix_norm = normalize(basic_feature_matrix, norm='l2')
short_desc_matrix_norm = normalize(short_desc_matrix, norm='l2')
tags_matrix_norm = normalize(tags_matrix, norm='l2')

sim_basic = cosine_similarity(basic_feature_matrix_norm)
sim_short = cosine_similarity(short_desc_matrix_norm)
sim_tags = cosine_similarity(tags_matrix_norm)

sim_weight_basic = 1.0
sim_weight_short = weights["short_desc"]
sim_weight_tags = weights["tags"]

final_similarity = (sim_weight_basic * sim_basic + 
                    sim_weight_short * sim_short + 
                    sim_weight_tags * sim_tags) 

# =============================================================================
# 12. RECOMMENDATION (Single ID)
# =============================================================================
def recommend_games(game_index, top_n=10, min_similarity=0.5):
    """
    Ranks similar games to the specified index based on the combined similarity matrix.
    
    Args:
        game_index (int): Sequential index in the DataFrame (0,1,2,...).
        top_n (int): Number of recommendations to return.
        min_similarity (float): Minimum similarity threshold for recommendations.
    
    Returns:
        List of tuples: [(recommended_game_index, similarity_score), ...]
    """
    if game_index < 0 or game_index >= final_similarity.shape[0]:
        print("\nInvalid game index.")
        return []
    
    sim_scores = list(enumerate(final_similarity[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for idx, score in sim_scores[1:]:
        if score >= min_similarity:
            recommendations.append((idx, score))
        if len(recommendations) >= top_n:
            break
    return recommendations

# =============================================================================
# 12-B. RECOMMENDATION (Multiple AppIDs or Steam UserId's)
# =============================================================================
# Bu fonksiyon, kullanıcı birden fazla app_id girerse, her birinin similarity row'unu toplayıp
# ortalama alır ve en benzer oyunları döndürür.
def recommend_multi_games(app_id_list, playtime_weights=None, top_n=10, min_similarity=0.1):
    """
    Takes a list of app_ids that user likes, sums/averages their similarity rows,
    and returns recommended games, considering playtime weights.

    Args:
        app_id_list (list): List of integer app_ids
        playtime_weights (list or None): Normalized playtime weights for each app_id
        top_n (int): Number of recommendations
        min_similarity (float): Minimum similarity threshold

    Returns:
        List of tuples: [(recommended_game_index, similarity_score), ...]
    """
    indexes = []
    for a_id in app_id_list:
        row_df = df[df["app_id"] == a_id]
        if not row_df.empty:
            seq_index = df.index.get_loc(row_df.index[0])  
            indexes.append(seq_index)
        else:
            print(f"⚠️ AppID {a_id} not found in dataset.")

    if not indexes:
        print("No valid AppIDs found.")
        return []

    if playtime_weights is None or len(playtime_weights) != len(indexes):
        playtime_weights = np.ones(len(indexes))  

    playtime_weights = np.array(playtime_weights) / np.sum(playtime_weights)

    combined_sim = np.zeros(final_similarity.shape[0])
    for weight, idx in zip(playtime_weights, indexes):
        combined_sim += final_similarity[idx] * weight

    sim_scores = list(enumerate(combined_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sim_scores:
        if idx not in indexes and score >= min_similarity:
            recommendations.append((idx, score))
        if len(recommendations) >= top_n:
            break
    return recommendations

def get_playtime_weights(app_id_list, playtime_data):
    playtimes = np.array([playtime_data.get(app_id, 0) for app_id in app_id_list])

    if playtimes.max() == 0:
        return np.ones(len(app_id_list))

    # Doğrudan maksimum değere göre normalize ediliyor
    playtime_weights = playtimes / playtimes.max()

    return playtime_weights



def get_steam_library(steam_id):
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={STEAM_API_KEY}&steamid={steam_id}&format=json&include_appinfo=true"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        games = data.get("response", {}).get("games", [])

        app_ids = [game["appid"] for game in games if game.get("playtime_forever", 0) > 0]
        playtime_data = {game["appid"]: game["playtime_forever"] for game in games}
        game_names = {game["appid"]: game["name"] for game in games}

        return app_ids, playtime_data, game_names
    else:
        print(f"❌ Error: Unable to fetch data from Steam API! Status Code: {response.status_code}")
        return None, None, None

def validate_input(user_input):
    user_input = user_input.strip()

    if "," in user_input:
        ids = user_input.split(",")
        if all(id.strip().isdigit() for id in ids):
            return "appid_list", [int(id.strip()) for id in ids]

    if user_input.isdigit():
        if len(user_input) == 17:
            return "steamid", user_input
        else:
            return "single_appid", [int(user_input)]

    return "unknown", None

if __name__ == '__main__':
    print("🎮 Game Recommendation System (Steam ID & AppIDs) Started.")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("Enter comma-separated AppIDs or a Steam ID (or 'exit'): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        input_type, processed_input = validate_input(user_input)

        if input_type == "unknown":
            print("⚠️ Invalid input. Please enter a valid Steam ID or numeric AppIDs separated by commas.")
            continue

        if input_type == "steamid":
            steam_id = processed_input
            app_id_list, playtime_data, game_names = get_steam_library(steam_id)

            if not app_id_list:
                print(f"⚠️ No games found for Steam ID {steam_id}. Make sure the profile is public.")
                continue

            print(f"✅ Steam ID recognized: {steam_id}. Found {len(app_id_list)} games.")
            playtime_weights = get_playtime_weights(app_id_list, playtime_data)
        
        else:
            app_id_list = processed_input
            playtime_weights = None

        recs = recommend_multi_games(app_id_list, playtime_weights=playtime_weights, top_n=10, min_similarity=0.2)

        if not recs:
            print("❌ No recommendations found for the given AppIDs.")
        else:
            print("\n🎯 Target Games:")
            for app_id in app_id_list:
                game_name = game_names.get(app_id, "Unknown Game")
                print(f"🔹 {app_id} - {game_name}")

            print("\n🔍 Recommended Games:")
            for rec_index, score in recs:
                rec_game = df.iloc[rec_index][["app_id", "name"]]
                print(f"🎮 {rec_game['app_id']} - {rec_game['name']} (Similarity: {score / final_similarity.max():.3f})")

        print("\n" + "-"*40 + "\n")
