import streamlit as st
import requests
from google.cloud import storage

st.set_page_config(page_title="Toxicity Classifier", page_icon="🤖")

def list_bucket_files(bucket_name="toxicity-classifier"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if blob.name.startswith("model-")]

st.title("Toxicity Classifier")
# Select a model
st.subheader("Select a model")
files = list_bucket_files() + ["local"]
bucket = st.selectbox("Select a model:", files)
if bucket:
    st.write(f"Selected model: {bucket}")
sentence = st.text_input("Enter a sentence:")
button = st.button("Predict")
if button:
    st.write("You entered:", sentence)
    st.write("Predicting...")
    response = requests.get(f"http://127.0.0.1:8000/predict?sentence={sentence}&model={bucket}")
    toxic = response.json()["bias"]
    st.write(f"Prediction: {toxic}")
    st.write(f"Probabilities: {response.json()['probabilities']}")
