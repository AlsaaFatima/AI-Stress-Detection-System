import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("stress_text_data.csv")

# Debug: confirm columns
print("CSV columns:", data.columns)

# =========================
# FEATURES & LABELS
# =========================
X = data["text"]
y = data["label"]

# =========================
# TF-IDF VECTORIZER
# =========================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_vec, y)

# =========================
# SAVE MODEL & VECTORIZER
# =========================
joblib.dump(model, "text_stress_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Text NLP model & vectorizer saved successfully")