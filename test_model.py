
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. File paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "WELFake_Dataset.csv"
MODEL_PATH = BASE_DIR / "fake_news_model_new.joblib"

# 2. Load the existing saved model
saved_objects = joblib.load(MODEL_PATH)
model = saved_objects["model"]
vectorizer = saved_objects["vectorizer"]

print("Loaded model:", type(model).__name__)

# 3. Load and clean the original dataset
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["news"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
).str.strip()

df = df[df["news"].str.len() > 0]
df = df.drop_duplicates(subset=["news"])

X = df["news"]
y = df["label"].astype(int)

# 4. Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Transform test data using the SAVED vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# 6. Predict using the SAVED model
predictions = model.predict(X_test_tfidf)

# 7. Evaluate model performance
print("\n===== MODEL EVALUATION =====")

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nEvaluation completed successfully!")
