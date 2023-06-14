import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Toxicity Detector", page_icon="🤖")


st.title("🤖 Toxicity Detector")
st.divider()
st.image("https://nohatespeech.network/wp-content/uploads/2020/10/cropped-No-Hate-Speech-Logo-Transperent-1-1.png", width=600)
st.write("Please, enter a sentence for **toxicity classification**.")

# Sentence input
sentence = st.text_input("📝 Enter a sentence:")

# Predict button
button = st.button("Predict")

if button:
    if sentence:
        st.write(f"🔄 Predicting...")
        response = requests.get(f"https://toxicity-classifier-cwkm3of3qa-ew.a.run.app/predict?sentence={sentence}")
        toxic = response.json()
        pipe_fake = list(toxic[0].values())[0]
        pipe_gender = list(toxic[1].values())[0]
        pipe_hate = list(toxic[2].values())[0]
        pipe_political = list(toxic[3].values())[0]
        pipe_racial = list(toxic[4].values())[0]

        print(pipe_fake)


        st.markdown("### **🔮 Prediction:**")
        st.write(f"**Hate Speech:** `{'Yes' if pipe_hate == 1 else 'No'}`")
        st.write(f"**Political Bias:** `{'Yes' if pipe_political == 1 else 'No'}`")
        st.write(f"**Gender Bias:** `{'Yes' if pipe_gender == 1 else 'No'}`")
        st.write(f"**Racial Bias:** `{'Yes' if pipe_racial == 1 else 'No'}`")
        st.write(f"**Fake News:** `{'Yes' if pipe_fake == 1 else 'No'}`")

    else:
        st.write("⚠️ Please enter a sentence.")
