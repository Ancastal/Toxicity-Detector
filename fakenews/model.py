from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from fakenews.preprocess import preprocess_text
import pickle
from google.cloud import storage
from datetime import datetime
from fakenews.params import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_bucket_files(bucket_name="toxicity-classifier"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if blob.name.startswith("toxicity-")]

def save_to_bucket(model_name, bucket_name="toxicity-classifier"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.upload_from_filename(model_name)

model_dict = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "naive_bayes": MultinomialNB,
    "svm": LinearSVC,
    "knn": KNeighborsClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
    "catboost": CatBoostClassifier,
    "mlp": MLPClassifier,
    "ada_boost": AdaBoostClassifier,
    # add more models here as needed
}

def load_model(model: str) -> tuple:
    logging.info("🛜 Loading model...")
    blobs = list_bucket_files()

    model_name = "toxicity-" + str(model) + ".pkl"

    if model_name in blobs:
        logging.info("🛜 Model found in bucket.")
        client = storage.Client()
        bucket = client.bucket("toxicity-classifier")
        blob = bucket.blob(model_name)
        blob.download_to_filename(model_name)
        loaded_vectorizer, loaded_model = pickle.load(open(model_name, "rb"))
        return loaded_vectorizer, loaded_model

    dataset = load_dataset("mediabiasgroup/mbib-base", "political-bias", keep_in_memory=True)
    df = dataset['train'].to_pandas()
    df['text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer().fit(df['text'])
    X_train = vectorizer.transform(df['text'])
    y_train = df['label']

    model_converted = model_dict.get(model)()
    model_converted.fit(X_train, y_train)

    save_model(vectorizer, model_converted, model_name)
    return vectorizer, model_converted

def save_model(vectorizer: object, model_converted: object, model_name: str, local=False) -> None:
    logging.info(f"Uploading on the cloud.")
    pickle.dump((vectorizer, model_converted), open(model_name, "wb"))  # pickle the actual model object
    save_to_bucket(model_name)
    logging.info(f"Saved model on the cloud.")
    logging.info(f"Saved model to locally.")
