#  Fake News Classifier – Web Application

A **Flask-based Fake News Detection Web Application** that classifies news articles as **REAL** or **FAKE** using a Machine Learning model.  
The system supports both **manual text input** and **live news fetching** using NewsAPI, providing predictions along with **confidence scores**.

This project was developed as part of an academic and practical learning initiative in **Machine Learning and Web Development**.

---

##  Project Objective
The rapid spread of misinformation on digital platforms makes it difficult to distinguish between real and fake news.  
This project aims to **automatically detect fake news articles** using Natural Language Processing (NLP) and Machine Learning techniques.

---

##  Features
- Classify news text as **REAL** or **FAKE**
- Confidence score for each prediction
- Live news classification using NewsAPI
- Web-based user interface
- Fast prediction (< 100 ms per article)
- Clean and modular project structure

---

##  Tech Stack

### Backend
- Python 3
- Flask (Web Framework)

### Machine Learning & NLP
- Scikit-learn
- TF-IDF Vectorization
- XGBoost Classifier
- NLTK (text preprocessing)
- NumPy

### Frontend
- HTML5
- CSS3

### Data Sources
- ISOT Fake News Dataset
- NewsAPI (Live News)

---

##  Machine Learning Workflow
1. News text input (manual or live)
2. Text preprocessing:
   - Lowercasing
   - Removing HTML & special characters
   - Tokenization
   - Stopword removal
   - Lemmatization
3. TF-IDF feature extraction (top 10,000 features)
4. Prediction using trained XGBoost model
5. Output:
   - REAL / FAKE label
   - Confidence percentage

---

##  Model Performance
- **Accuracy:** 92%
- **Precision:** 91%
- **Recall:** 90%
- **F1-Score:** 90%
- Tested on 5,000+ news articles

---

## How to Run the Project Locally

1️ Clone the repository
->  ```bash
    git clone https://github.com/<your-username>/fake-news-classifier.git
    cd fake-news-classifier
    
2️ Install dependencies
pip install -r requirements.txt

3️ Set NewsAPI key (optional for live news)
set NEWS_API_KEY=your_api_key_here

4️ Run the application
python app.py

5️ Open in browser
http://127.0.0.1:5000


Example Predictions

REAL:
"Government announces new education policy for schools"
Confidence: 94%

FAKE:
"Secret cure doctors hate! Shocking revelation inside!"
Confidence: 91%
