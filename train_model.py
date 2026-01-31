

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_preparation import clean_text

print("🚀 XGBoost Training (Fixed)...")

# Load
train_df = pd.read_excel("TRAIN.1.xlsx")
valid_df = pd.read_excel("VALID.2.xlsx")
test_df = pd.read_excel("TEST.3.xlsx")

full_train = pd.concat([train_df, valid_df], ignore_index=True)

# Clean
full_train['text_clean'] = full_train['text'].apply(clean_text)
test_df['text_clean'] = test_df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    stop_words='english',
)

X = vectorizer.fit_transform(full_train['text_clean'])
X_test = vectorizer.transform(test_df['text_clean'])

y = full_train['label'].values
y_test = test_df['label'].values

# XGBoost
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.6, 
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

# Results
print("\n✅ TRAIN ACCURACY:", accuracy_score(y, model.predict(X)))
print("\n✅ TEST ACCURACY:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test), target_names=['FAKE(0)', 'TRUE(1)']))

# Samples
print("\n✅ SAMPLES:")
samples = [
    "Government announces new education policy",
    "Secret cure doctors hate!",
]

for s in samples:
    cleaned = clean_text(s)
    X_s = vectorizer.transform([cleaned])
    pred = model.predict(X_s)[0]
    proba = model.predict_proba(X_s)[0].max()*100
    label = "TRUE" if pred == 1 else "FAKE"
    print(f"'{s}' → {label} ({proba:.1f}%)")

# SAVE DIRECTLY (no dictionary)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n🎉 SAVED! Run: python app.py")
