import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

# # Load the feature columns used during training
# # Assuming 'expected_columns' contains the expected feature columns
# expected_columns = ['Junction', 'Hours', 'date', 'Month', 'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday']

# # Standard scaler for feature scaling
# scaler = StandardScaler()

# Streamlit app
st.title("Traffic Prediction App")

st.header("Make a Prediction")

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

# # Add missing columns
for col in feature_columns:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0

# # Ensure columns order
df_input_encoded = df_input_encoded[feature_columns]

# Call the function to fit the scaler
# scaler.fit(df_input_encoded)

scaled_input=scaler.transform(df_input_encoded)


# Transform the input data using the fitted scaler

# Make the prediction
prediction = rf_model.predict(scaled_input)

# Display the prediction to the user
st.header(f'Predicted number of vehicles: {round(prediction[0])}')
