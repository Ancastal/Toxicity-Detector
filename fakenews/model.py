from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from fakenews.preprocess import preprocess_text
import pickle
from google.cloud import storage
from datetime import datetime
from fakenews.params import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def initialize_model(vectorizer: object) -> tuple:
    """
    Initialize a text classification model.

    Args:
        vectorizer (object): A feature extraction vectorizer.

    Returns:
        tuple: A tuple containing the initialized vectorizer and the initialized classification model.
    """
    # If model already exists, load it
    model = None
    try:
        model = load_model()
        if model is not None:
            logging.info("Loaded model from local...")
            return vectorizer, model
    except:
        pass


    logging.info("Downloading dataset from huggingface...")
    dataset = load_dataset("mediabiasgroup/mbib-base", "hate-speech")
    df = dataset['train'].to_pandas()

    logging.info("Preprocessing dataset...")
    df = df.sample(frac=DATA_SIZE).reset_index(drop=True)

    df['text'] = df['text'].apply(preprocess_text)

    X_train = vectorizer.fit_transform(df['text'])
    y_train = df['label']

    model = LogisticRegression(max_iter=5000)
    logging.info("Training model...")
    model.fit(X_train, y_train)

    # Save model
    save_model(model)

    return vectorizer, model


def save_model(model: object, local=False) -> None:
    """
    Save a model to storage.

    Args:
        model (object): The model to save.
        local (bool): Whether to save the model locally or to the cloud.
    Returns:
        None
    """
    if MODEL_TARGET == 'local' or local:
        logging.info(f"Saved model to locally.")
        pickle.dump(model, open(MODEL_PATH, 'wb'))
    else:
        storage_client = storage.Client.from_service_account_json(CRED_PATH)
        logging.info(f"Saving local model and uploading on the cloud.")
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        bucket = storage_client.bucket('toxicity-classifier')
        blob = bucket.blob(MODEL_PATH)
        blob.upload_from_filename(MODEL_PATH)
        logging.info(f"Saved model on the cloud.")


def load_model() -> object:
    """
    Load a model from storage.

    Returns:
        object: The loaded model.
    """
    if MODEL_TARGET == 'local':
        with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket('toxicity-classifier')
        blob = bucket.blob(MODEL_NAME)
        logging.info(f"Loading model from {MODEL_NAME}")
        blob.download_to_filename(MODEL_NAME)
        model = pickle.load(open(MODEL_NAME, 'rb'))
        logging.info(f"Loaded model from the cloud.")
    return model
