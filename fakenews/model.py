# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from datasets import load_dataset
# from sklearn.feature_extraction.text import TfidfVectorizer
# from fakenews.preprocess import preprocess_text
import pickle
from google.cloud import storage
from datetime import datetime
from fakenews.params import *
from simpletransformers.classification import ClassificationModel
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.neural_network import MLPClassifier
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_bucket_files(bucket_name="toxicity-classifier"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs]

def save_to_bucket(model_name, bucket_name="toxicity-classifier"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.upload_from_filename(model_name)

def load_classification_model(model, model_path):
    model = ClassificationModel(model, model_path, use_cuda=False, num_labels=6, args={'overwrite_output_dir': True, 'output_dir': 'outputs/distilbert'})
    return model

def load_models() -> None:
    logging.info("🛜 Loading dataset...")
    blobs = list_bucket_files()
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    roberta_dir = os.path.join(parent_dir, "models", "roberta")

    try:
        loaded_model = load_classification_model("roberta", roberta_dir)
        return loaded_model
    except FileNotFoundError:
        loaded_model = None
        logging.info("🛜 Model not found locally.")
    except Exception as e:
        loaded_model = None
        logging.info(e)

    if loaded_model is None:
        logging.info("🛜 Model not found locally. Downloading from bucket...")
        logging.info(f"🛜 Downloading models from bucket...")

        os.makedirs(roberta_dir, exist_ok=True)

        for model_name in blobs:
            if model_name in blobs:
                logging.info(f"🛜 Model {model_name} found in bucket.")
                client = storage.Client()
                bucket = client.bucket("toxicity-classifier")
                blob = bucket.blob(model_name)

                blob.download_to_filename(os.path.join(roberta_dir, model_name))
                if model_name.endswith(".bin"):
                    logging.info(f"🛜 Model {model_name} downloaded from bucket.")
            else:
                logging.warning(f"🛜 Model {model_name} not found in bucket.")

        logging.info("🛜 Loading models...")

        assert os.path.exists(os.path.join(roberta_dir, "pytorch_model.bin"))

        loaded_model = load_classification_model("roberta", roberta_dir)

        assert loaded_model is not None

        logging.info("🛜 Models loaded.")

    return loaded_model


def save_model(vectorizer: object, model_converted: object, model_name: str, local=False) -> None:
    logging.info(f"Uploading on the cloud.")
    pickle.dump((vectorizer, model_converted), open(model_name, "wb"))  # pickle the actual model object
    save_to_bucket(model_name)
    logging.info(f"Saved model on the cloud.")
    logging.info(f"Saved model to locally.")
