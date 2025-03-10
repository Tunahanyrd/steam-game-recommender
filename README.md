**🎮 Steam Game Recommendation System**  
🔍 *An AI-powered recommendation system that suggests similar Steam games based on metadata, user reviews, and tags.*  

---

### **📖 About This Project**  
This project is an **AI-driven game recommendation system** that uses **machine learning & NLP** to suggest similar games based on:  
✔ **Developers & Publishers**  
✔ **Genres & Categories**  
✔ **Short Descriptions & Tags**  
✔ **Metacritic & User Review Scores**  

It processes **Steam game data**, extracts meaningful features, and calculates similarities using **cosine similarity & Word2Vec**.  

---

### **⚙️ Features**
✅ **Suggests similar Steam games** based on a given game’s Steam ID  
✅ Uses **TF-IDF, DictVectorizer, Word2Vec & Cosine Similarity**  
✅ Processes **large-scale game metadata**  
✅ **Web-based interface** using **Streamlit**  
✅ Lightweight and fast **recommendation engine**  

---

### **🚀 How to Use?**  
1️⃣ Clone the repository  
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
3️⃣ Run the web app:  
```bash
streamlit run app/app.py
```
4️⃣ Enter a **Steam Game ID** and get similar game recommendations instantly!  

---

### **📂 Project Structure**  
```
📂 steam-game-recommender /
│── 📂 app/
│   ├── 📜 app.py              # for Streamlit        
│   ├── 📜 app_local.py        # for local Streamlit    
│── 📂 data/
│   ├── 📜 games.csv            # Original dataset csv file
│   ├── 📜 games.json           # Original dataset json file
│
│── 📂 models/
│   ├── 📜 game_recommendation.h5 # HDF5 model file
│
│── 📂 src/
│   ├── 📜 main.py              
│
│── 📜 requirements.txt                 
│── 📜 README.md              
```

## 📌 Dataset
The complete dataset and model are hosted on Hugging Face:
👉 **[Hugging Face Dataset](https://huggingface.co/datasets/Tunahanyrd/steam-game-recommendation)**

## 📌 How to Use
1. Run the Streamlit application:
```bash
streamlit run app/local_app.py
```
2. Download the `.h5` file from Hugging Face and load the model:
```python
import pandas as pd

with pd.HDFStore("game_recommendation.h5", "r") as store:
    df = store["df"]
    similarity_matrix = store["similarity_matrix"].values
```

---

### **🛠 Technologies Used**  
- **Python (Pandas, NumPy, Scikit-learn, Gensim, Streamlit)**  
- **Machine Learning & NLP** (TF-IDF, Word2Vec, Cosine Similarity)  
- **GitHub LFS** (for large dataset storage)   

---

**🚀 Ready to find your next favorite game? Let’s go!** 🎮  

---
