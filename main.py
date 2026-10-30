
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fake_news_model_new.joblib"
STATIC_DIR = BASE_DIR / "static"

# Load the trained model and TF-IDF vectorizer
saved_objects = joblib.load(MODEL_PATH)
tfidf = saved_objects["vectorizer"]
model = saved_objects["model"]

print("Loaded model from:", MODEL_PATH.name)
print("Model type:", type(model).__name__)
print("Model classes:", model.classes_)

# Create the API
app = FastAPI(
    title="Fake News Classifier API",
    description="Predict whether an article resembles fake or real news.",
    version="1.0.0",
)

# Serve CSS and JavaScript from the static folder
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


class NewsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)


# Show the frontend website at the home page
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# Predict whether the submitted text resembles fake or real news
@app.post("/predict")
def predict_news(news: NewsRequest):
    article = news.text.strip()

    if not article:
        return {"error": "Please enter some news text."}

    # Convert article text into TF-IDF features
    features = tfidf.transform([article])

    # Predict using the trained LinearSVC model
    prediction = int(model.predict(features)[0])
    decision_score = float(model.decision_function(features)[0])
    print("NEW CODE RUNNING! Decision score:", decision_score)

    if prediction == 0:
        result = "Predicted Fake News"
    else:
        result = "Predicted Real News"

    return {
    "prediction": prediction,
    "result": result,
    "decision_score": decision_score,
    "note": (
        "This is a model prediction, not independent "
        "verification of the article."
    ),
}