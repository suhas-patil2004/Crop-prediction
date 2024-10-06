# Crop Recommendation System

## Overview
The Crop Recommendation System is a web application built using Streamlit and machine learning algorithms. It provides users with crop recommendations based on soil and environmental conditions, helping farmers optimize their planting decisions for better yields.

## Features
- **User-Friendly Interface:** Simple UI for input parameters.
- **Real-Time Predictions:** Instant crop recommendations based on user inputs.
- **Data-Driven Insights:** Utilizes a Random Forest Classifier trained on historical crop data.

## Technologies Used
- **Python:** Backend logic.
- **Streamlit:** Web application framework.
- **Scikit-learn:** Machine learning library.
- **Pandas & NumPy:** Data manipulation.
- **Joblib:** Model saving/loading.

## Installation

### Prerequisites
- Python 3.6 or higher.
- Required packages.

### Steps
1. **Clone the Repository:**
   ```bash
   https://github.com/suhas-patil2004/Crop-prediction.git
   cd crop-recommendation-system
2. Install dependencies:
    ```bash
   pip install numpy pandas streamlit sklearn

3. Run the Application:
    ```bash
    streamlit run app.py
### Usage:
 - Home: Overview of the application and its purpose.
 - Predict: Input the following parameters to receive crop recommendations:
       1. Nitrogen content (N)
       2. Phosphorus content (P)
       3. Potassium content (K)
       4. Temperature (°C)
       5. Humidity (%)
       6. pH
       7. Rainfall (mm)
 - Contact Us: Reach out for inquiries, feedback, or support.

### Model Details :
The underlying model is a Random Forest Classifier, trained on a dataset containing various soil and crop data points. The model has been saved using Joblib for efficient loading during runtime.
