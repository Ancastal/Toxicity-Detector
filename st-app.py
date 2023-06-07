import streamlit as st
import requests

st.set_page_config(page_title="Toxicity Classifier", page_icon="🤖")
st.title("Toxicity Classifier")
sentence = st.text_input("Enter a sentence:")
button = st.button("Predict")
if button:
    st.write("You entered:", sentence)
    st.write("Predicting...")
    response = requests.get(f"https://toxicity-classifier-cwkm3of3qa-ew.a.run.app/predict?sentence={sentence}")
    st.write(response.json())
