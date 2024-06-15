import streamlit as st
import pandas as pd
import requests

#API endpoints
API_URL = "http://127.0.0.1:5000/predict"

st.title('Diamond Price Predictor :diamond_shape_with_a_dot_inside:')
st.divider()

# Input features
carat = st.number_input('Carat', min_value=0.0, max_value=5.0, step=0.01)
cut = st.selectbox('Cut', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
x = st.number_input('Length (X)', min_value=0.0, max_value=10.0, step=0.01)

# Create dataframe from inputs
input_data = {
    'carat': carat,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'x': x
}

# Predict button
if st.button('Predict'):
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            predictions = response.json()
            st.write(predictions)
    except:
        print("Error. Please, Try Again or Contact Us")

