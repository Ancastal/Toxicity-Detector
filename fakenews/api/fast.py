import sys
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from fakenews.model import initialize_model
from fakenews.preprocess import preprocess_text

app = FastAPI()

app.state.vectorizer = TfidfVectorizer(ngram_range=(1, 1))
app.state.vectorizer, app.state.model = initialize_model(app.state.vectorizer)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# localhost:8000/predict?sentence=hello
@app.get("/predict")
def predict(
        sentence: str
    ):
    """
    Make a single course prediction.
    """
    X_processed = preprocess_text(sentence)
    X_processed = app.state.vectorizer.transform([X_processed])
    y_pred = app.state.model.predict(X_processed)

    return {
        sentence: y_pred.tolist()[0],
    }

@app.get("/")
def root():
   return {
        'API': '200 [GET /predict?sentence=hello]'
    }
