**🎮 Steam Game Recommendation System**  
🔍 *An AI-powered recommendation system that suggests similar Steam games based on metadata, user playtime, and tags.*  

---

## **📖 About This Project**  
This project is an **AI-driven game recommendation system** that leverages **machine learning & NLP** to suggest games based on:  
✔ **User Playtime & Preferences**  
✔ **Developers & Publishers**  
✔ **Genres & Categories**  
✔ **Short Descriptions & Tags**  
✔ **Metacritic & User Review Scores**  
✔ **Real Steam Library Integration**  

It processes **Steam game data**, extracts meaningful features, and calculates similarities using **cosine similarity & Word2Vec**.  

---

## **⚙️ Features**
✅ **Suggests similar Steam games** based on a given game’s Steam ID  
✅ **Fetches a user's Steam library** and analyzes playtime for personalized recommendations  
✅ Uses **TF-IDF, DictVectorizer, Word2Vec & Cosine Similarity**  
✅ **Playtime-weighted recommendations** for accurate results  
✅ **Filters out low-playtime games** dynamically  
✅ **Web-based interface** using **Streamlit**  
✅ Lightweight and fast **recommendation engine**  

---

## **🚀 How to Use?**  
### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Web App**  
```bash
streamlit run app/app.py
```

### **3️⃣ Enter a Steam ID or App IDs**  
- Enter **your Steam ID** to fetch your library and get personalized recommendations  
- Enter **a list of Steam Game IDs** to get recommendations based on selected games  

---

## **📂 Project Structure**  
```
📂 steam-game-recommender /
│── 📂 app/
│   ├── 📜 app.py              # Streamlit Web App        
│   ├── 📜 app_local.py        # Local Streamlit App    
│
│── 📂 data/
│   ├── 📜 games.csv           # Original dataset csv file
│   ├── 📜 games.json          # Processed dataset json file
│
│── 📂 models/
│   ├── 📜 game_recommendation.h5 # Precomputed similarity matrix
│   ├── 📜 game_recommendation_creative.h5 # Creative model precomputed similarity matrix
│
│── 📂 src/
│   ├── 📜 main.py              # Core ML pipeline
│   ├── 📜 creative_main.py     # Creative ML pipeline
│
│── 📜 requirements.txt                 
│── 📜 README.md              
```

## **📌 Dataset & Model**
The complete dataset and model are hosted on Hugging Face:
👉 **[Hugging Face Dataset](https://huggingface.co/datasets/Tunahanyrd/steam-game-recommendation)**

### **📌 How to Use the Precomputed Model**
1. Download the `.h5` file from Hugging Face and load the model:
```python
import pandas as pd

with pd.HDFStore("game_recommendation.h5", "r") as store:
    df = store["df"]
    similarity_matrix = store["similarity_matrix"].values
```
2. Run the recommendation function:
```python
recommended_games = recommend_multi_games(app_id_list, playtime_weights=playtime_weights, top_n=10, min_similarity=0.2)
```

---

## **🛠 Technologies Used**  
- **Python (Pandas, NumPy, Scikit-learn, Gensim, Streamlit, Requests)**  
- **Machine Learning & NLP** (TF-IDF, Word2Vec, Cosine Similarity)  
- **Steam API Integration** (Fetching real user libraries)  
- **GitHub LFS** (for large dataset storage)   

---

**🚀 Ready to find your next favorite game? Let’s go!** 🎮  

---
> **Note:** This project uses data from the Steam Games Dataset by Fronkon Games, available under the MIT License.

