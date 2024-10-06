import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load model and data
model = joblib.load('random_forest_classifier.h5')  # Ensure this is the correct path
df = pd.read_csv('Crop_recommendation.csv')  # Ensure this is the correct path
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

# Define prediction function
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    predicted_label_encoded = model.predict(input_data)
    predicted_crop = label_encoder.inverse_transform(predicted_label_encoded)
    return predicted_crop[0]

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ("Home", "Predict", "Contact Us"))

# Home page
if options == "Home":
    st.title("Welcome to the Crop Recommendation System")
    st.write("""
        This application uses machine learning to recommend the best crops to plant based on soil and environmental conditions.
        Please navigate to the **Predict** page to make a prediction or to the **Contact Us** page for any inquiries.
    """)
    st.image("precisionagri.jpg", caption='Crop Image')  # Add an image if you have one

# Prediction page
elif options == "Predict":
    st.title("Crop Recommendation")
    st.write("Fill in the following values to get a crop recommendation:")

    N = st.number_input("Nitrogen content (N)", min_value=0.0)
    P = st.number_input("Phosphorus content (P)", min_value=0.0)
    K = st.number_input("Potassium content (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    if st.button("Predict"):
        result = predict_crop(N, P, K, temperature, humidity, pH, rainfall)
        st.success(f"The predicted crop type is: **{result}**")

# Contact Us page
elif options == "Contact Us":
    st.title("Contact Us")
    st.write("""
        For inquiries, feedback, or support, please reach out to us:
        - **Email**: support@example.com
        - **Phone**: +123 456 7890
        - **Follow us on social media**: [Twitter](https://twitter.com), [Facebook](https://facebook.com)
    """)

# Custom styling
st.markdown("""
<style>
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2C3E50; /* Dark background */
        color: white; /* Text color */
        border-radius: 10px; /* Rounded corners */
    }
    .css-1d391kg:hover {
        background-color: #34495E; /* Lighter background on hover */
        transition: background-color 0.3s ease; /* Smooth transition */
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border: None;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
