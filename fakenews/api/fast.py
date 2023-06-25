import logging
# from simpletransformers.classification import ClassificationModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from sklearn.feature_extraction.text import TfidfVectorizer
# from fakenews.model import load_models
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder
import os
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Setting up application...")

app.state.tokenizer = AutoTokenizer.from_pretrained("Ancastal/bert-bias-detector")
app.state.model = AutoModelForSequenceClassification.from_pretrained("Ancastal/bert-bias-detector")

# app.state.model = load_models()
logging.info("✅ Model loaded.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/reset")
def reset():
    """
    Reset the model.
    """
    logging.info("🧑‍💻 Resetting model...")
    app.state.model = None
    logging.info("✅ Model reset.")

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    roberta_dir = os.path.join(parent_dir, "models", "roberta")

    try:
        os.rmdir(roberta_dir)
    except OSError as e:
        logging.info(e)

    return {
        'status': 'ok'
    }


@app.get("/predict")
def predict(
        sentence: str
    ):
    """
    Make a single prediction.
    """
    logging.info("🧑‍💻 Making prediction...")

    inputs = app.state.tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        logits = app.state.model(**inputs).logits

    y_pred = logits.argmax().item()

    logging.info("✅ Prediction made.")

    return y_pred


@app.get("/")
def root():
    return {
        'API': '200 [GET /predict?sentence=hello]'
    }
