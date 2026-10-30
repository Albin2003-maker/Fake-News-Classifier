# Fake News Classification System

An end-to-end Machine Learning project that classifies news articles as **Fake News** or **Real News** using Natural Language Processing (NLP).

## Features
- Text preprocessing and cleaning
- TF-IDF feature extraction using unigrams and bigrams
- Linear Support Vector Classification (LinearSVC)
- FastAPI backend for predictions
- Interactive web interface built with HTML, CSS and JavaScript
- Model saved and loaded using Joblib

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- LinearSVC
- FastAPI
- HTML, CSS and JavaScript
- Joblib

## Model Performance

The model achieved **96.98% accuracy** on a held-out test split of the WELFake dataset.

This result reflects performance on that test split. It does not guarantee the same accuracy on new or real-world articles.

## Project Structure

```text
fake-news-classification-system/
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── main.py
├── train_model.py
├── test_model.py
├── fake_news_model_new.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

3. Open this address in your browser:

```text
http://127.0.0.1:8000
```

## Dataset

This project uses the WELFake dataset.

The dataset file is not included in this repository. Obtain it from its original source and place `WELFake_Dataset.csv` in the project root before running the training scripts.

## Important Disclaimer

This project is an educational machine learning demonstration. Its predictions are based on patterns learned from training data. It does not independently verify facts or determine whether an article is true.