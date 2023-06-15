import streamlit as st
import requests
import openai
import pandas as pd

st.set_page_config(page_title="Toxicity Detector", page_icon="🤖")

if st.button("📖 Reset"):
    response = requests.get(f"https://toxicity-classifier-cwkm3of3qa-ew.a.run.app/reset")
    st.write(response.json())

#st.title("🤖 Toxicity Detector")
st.image("https://nohatespeech.network/wp-content/uploads/2020/10/cropped-No-Hate-Speech-Logo-Transperent-1-1.png", width=200, caption="No Hate Speech")
st.divider()
st.write("Please, enter a sentence for **toxicity classification**.")

# Sentence input
sentence = st.text_input("📝 Enter a sentence:")

# Predict button
button = st.button("Predict")

if button:
    if sentence:
        st.write(f"🔄 Predicting...")
        response = requests.get(f"https://toxicity-classifier-cwkm3of3qa-ew.a.run.app/predict?sentence={sentence}")
        st.write(f"✅ Prediction made.")
        st.write(f"🤖 **{sentence}** is **{response.json()}**.")

        # openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        # messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": "Who won the world series in 2020?"},
        #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #         {"role": "user", "content": "Where was it played?"}
        #     ]
        # )

    else:
        st.write("⚠️ Please enter a sentence.")
