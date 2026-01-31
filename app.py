from flask import Flask, render_template, request
import pickle
import requests
import numpy as np
import random

from data_preparation import clean_text

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


import os
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

NEWS_URL = "https://newsapi.org/v2/everything"


@app.route("/", methods=["GET", "POST"])  
def home():
    prediction = None
    text = ""
    live_news = None
    api_error = None
    
    if request.method == "POST":
        # Dataset prediction
        text = request.form.get("news", "").strip()
        if text:
            cleaned = clean_text(text)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = np.max(proba) * 100
            label = "TRUE" if pred == 1 else "FAKE"
            prediction = f"{label} ({confidence:.1f}%)"
    
    if request.args.get("live"):
        # Live news
        try:
            page = random.randint(1, 20)
            params = {
                "q": "india",
                "apiKey": NEWS_API_KEY,
                "language": "en",
                "sortBy": "popularity",
                "pageSize": 8,
                "page": page,
            }
            
            response = requests.get(NEWS_URL, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") == "ok":
                live_news = []
                for article in data.get("articles", [])[:6]:
                    title = article.get("title", "")
                    desc = article.get("description", "")
                    full_text = f"{title} {desc}".strip()
                    
                    if full_text:
                        cleaned = clean_text(full_text)
                        X = vectorizer.transform([cleaned])
                        pred = model.predict(X)[0]
                        proba = model.predict_proba(X)[0]
                        confidence = np.max(proba) * 100
                        label = "TRUE" if pred == 1 else "FAKE"
                        
                        live_news.append({
                            "title": title[:70] + "..." if len(title) > 70 else title,
                            "label": label,
                            "confidence": f"{confidence:.0f}%",
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "url": article.get("url", "#")
                        })
            else:
                api_error = data.get("message", "API Error")
                
        except Exception as e:
            api_error = str(e)

    return render_template("index.html",
                           prediction=prediction,
                           text=text,
                           live_news=live_news,
                           api_error=api_error)


if __name__ == "__main__":
    app.run(debug=True)
