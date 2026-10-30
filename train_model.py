
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# File paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "WELFake_Dataset.csv"
MODEL_PATH = BASE_DIR / "fake_news_model_new.joblib"

# 1. Load dataset
print("Loading WELFake dataset...")
df = pd.read_csv(DATA_PATH)

# 2. Prepare the data
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title and article text
df["news"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
).str.strip()

# Remove rows with no usable news text
df = df[df["news"].str.len() > 0]

# Remove duplicate articles to reduce evaluation leakage
df = df.drop_duplicates(subset=["news"])

X = df["news"]
y = df["label"].astype(int)

print("Usable records:", len(df))
print("\nLabel counts:")
print(y.value_counts().sort_index())

# 3. Split before fitting the vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=200000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train classifier
print("Training LinearSVC model...")
model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate on held-out test data
predictions = model.predict(X_test_tfidf)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (labels ordered as 0, 1):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 6. Save new model without overwriting the old one
joblib.dump(
    {"vectorizer": tfidf, "model": model},
    MODEL_PATH,
)

print(f"\nNew model saved to: {MODEL_PATH}")
print("Original model has NOT been overwritten.")
