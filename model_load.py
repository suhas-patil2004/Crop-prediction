import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


model = joblib.load('random_forest_classifier.h5')

df = pd.read_csv('Crop_recommendation.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])


def predict_crop(N, P, K,temperature, humidity, pH, rainfall):
    input_data = np.array([[N, P, K,temperature, humidity, pH, rainfall]])
    predicted_label_encoded = model.predict(input_data)
    predicted_crop = label_encoder.inverse_transform(predicted_label_encoded)
    return predicted_crop[0]


st.title("Crop Recommendation System")


N = st.number_input("Nitrogen content (N)", min_value=0.0)
P = st.number_input("Phosphorus content (P)", min_value=0.0)
K = st.number_input("Potassium content (K)", min_value=0.0)
temperature = st.number_input("Temperature", min_value=0.0)
humidity = st.number_input("Humidity", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall", min_value=0.0)


if st.button("Predict"):
    result = predict_crop(N, P, K,temperature, humidity, pH, rainfall)
    st.success(f"The predicted crop type is: {result}")
