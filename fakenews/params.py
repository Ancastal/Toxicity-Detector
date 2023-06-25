import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
API_URL = os.environ.get("API_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'fakenews', 'models', 'model.pkl')
CRED_PATH = os.path.join(BASE_DIR, "credentials.json")
DOCKER_LOCAL_PORT= os.environ.get("DOCKER_LOCAL_PORT")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GCR_IMAGE = os.environ.get("GCR_IMAGE")
# GCR_REGION = os.environ.get("GCR_REGION")
# GCR_MEMORY = os.environ.get("GCR_MEMORY")
