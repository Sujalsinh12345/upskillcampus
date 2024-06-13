import streamlit as st
import pandas as pd
import requests
import zipfile
import joblib
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Function to download file from GitHub
def download_file_from_github(repo_url, file_name):
    url = f"{repo_url}/raw/main/{file_name}"
    print(f"Downloading {file_name} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download file {file_name} from GitHub. Status code: {response.status_code}")

# Function to extract rf_model.pkl from rf_model.zip
def extract_rf_model_zip(zip_file_content):
    with zipfile.ZipFile(zip_file_content, 'r') as zip_ref:
        zip_ref.extractall('./temp')  # Extracts all files to ./temp directory
        return './temp/rf_model.pkl'  # Return the path to rf_model.pkl

# GitHub repository URL
repo_url = "https://github.com/Sujalsinh12345/upskillcampus"

# Streamlit app
st.title("Traffic Prediction App")
st.header("Make a Prediction")

try:
    # Download rf_model.zip from GitHub
    st.write("Downloading model file...")
    rf_model_zip_file = download_file_from_github(repo_url, "rf_model.zip")

    # Extract rf_model.pkl from rf_model.zip
    rf_model_pkl_path = extract_rf_model_zip(rf_model_zip_file)
    rf_model = joblib.load(rf_model_pkl_path)
    st.success("Model file extracted and loaded successfully!")

    # Download feature_columns.pkl from GitHub
    feature_columns_file = download_file_from_github(repo_url, "feature_columns.pkl")
    feature_columns = joblib.load(feature_columns_file)

    # Download scaler.pkl from GitHub
    scaler_file = download_file_from_github(repo_url, "scaler.pkl")
    scaler = joblib.load(scaler_file)

    # User inputs for prediction
    junction = st.selectbox("Select Junction", [1, 2, 3, 4])
    day = st.selectbox("Select Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour = st.slider("Select Hour", 0, 23, step=1)
    date = st.slider("Select Date", 1, 31, step=1)
    month = st.slider("Select Month", 1, 12, step=1)

    # Prepare input data for prediction
    input_data = {
        'Junction': [junction],
        'Day': [day],
        'Hours': [hour],
        'date': [date],
        'Month': [month]
    }
    df_input = pd.DataFrame(input_data)
    df_input_encoded = pd.get_dummies(df_input, columns=['Junction','Day','Hours','date','Month'])

    # Add missing columns
    for col in feature_columns:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # Ensure columns order
    df_input_encoded = df_input_encoded[feature_columns]

    # Scale the input data
    scaler.fit(df_input_encoded)
    scaled_input = scaler.transform(df_input_encoded)

    # Make the prediction
    prediction = rf_model.predict(scaled_input)

    # Display the prediction to the user
    st.header(f'Predicted number of vehicles: {round(prediction[0])}')

except Exception as e:
    st.error(f"Error: {e}")
