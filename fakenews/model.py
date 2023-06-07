from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from preprocess import preprocess_text
import pickle
from google.cloud import storage
from datetime import datetime
from fakenews.params import *


def initialize_model(vectorizer: object) -> tuple:
    """
    Initialize a text classification model.

    Args:
        vectorizer (object): A feature extraction vectorizer.

    Returns:
        tuple: A tuple containing the initialized vectorizer and the initialized classification model.
    """
    # If model already exists, load it
    model = load_model(MODEL_NAME)
    if model:
        print("Loaded model from storage.")
        return vectorizer, model

    # Otherwise, train a new model

    # Load dataset
    try:
        df = pickle.load(open('raw_data/data.pkl', 'rb'))
    except FileNotFoundError:
        print("Downloading dataset...")
        dataset = load_dataset("mediabiasgroup/mbib-base", "hate-speech")
        df = dataset['train'].to_pandas()
        pickle.dump(df, open('raw_data/data.pkl', 'wb'))
        print("Done.")

    # Preprocess text
    df = df.sample(frac=DATA_SIZE).reset_index(drop=True)

    df['text'] = df['text'].apply(preprocess_text)

    X_train = vectorizer.fit_transform(df['text'])
    y_train = df['label']

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    save_model(model, 'raw_data/model.pkl')

    return vectorizer, model


def save_model(model: object, filename: str) -> None:
    """
    Save a model to storage.

    Args:
        model (object): The model to save.
        filename (str): The name of the file to save the model to.

    Returns:
        None
    """
    if MODEL_TARGET == 'local':
        pickle.dump(model, open(filename, 'wb'))
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket('toxicity-classifier')
        blob = bucket.blob(filename)
        blob.upload_from_filename(filename)


def load_model(filename: str) -> object:
    """
    Load a model from storage.

    Args:
        filename (str): The name of the file to load the model from.

    Returns:
        object: The loaded model.
    """
    if MODEL_TARGET == 'local':
        model = pickle.load(open(filename, 'rb'))
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket('toxicity-classifier')
        blob = bucket.blob(filename)
        blob.download_to_filename(filename)
        model = pickle.load(open(filename, 'rb'))
    return model
