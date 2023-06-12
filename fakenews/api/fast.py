import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from fakenews.model import load_model
from fakenews.preprocess import preprocess_text

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Setting up application...")
app.state.vectorizer, app.state.model = None, None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        model: str,
        sentence: str
    ):
    """
    Make a single prediction.
    """
    logging.info("🧑‍💻 Making prediction...")

    X_preprocessed = preprocess_text(sentence)

    app.state.vectorizer, model = load_model(model=model)

    if app.state.vectorizer is None:
        app.state.vectorizer = TfidfVectorizer()
        X_vectorized = app.state.vectorizer.fit_transform([X_preprocessed])
        logging.info("✅ Model loaded.")
    else:
        logging.info("Using existing model.")
        X_vectorized = app.state.vectorizer.transform([X_preprocessed])

    y_pred = model.predict(X_vectorized).tolist()

    return {
        'sentence': sentence,
        'y_pred': y_pred[0],
        'y_proba': model.predict_proba(X_vectorized).tolist()[0]
    }


@app.get("/")
def root():
    return {
        'API': '200 [GET /predict?sentence=hello]'
    }
