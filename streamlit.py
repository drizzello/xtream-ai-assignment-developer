import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title='Xtram_project', page_icon="💠", layout="centered", initial_sidebar_state="auto", menu_items=None)



#API endpoints
PREDICT_API_URL = "http://127.0.0.1:5000/predict"
SIMILAR_DIAMONDS_API_URL = "http://127.0.0.1:5000/get_similar_diamonds"

st.title('API Test :diamond_shape_with_a_dot_inside:')
st.divider()

st.header('Price Predictor endpoint: .../predict')
# Input features    
st.write("**All fiels are compulsary**")
carat = st.number_input('Carat', min_value=0.0, max_value=5.0, step=0.01)
cut = st.selectbox('Cut', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Depth', min_value=0.0, max_value=100.0, step=0.01)
table = st.number_input('Table', min_value=50.0, max_value=100.0, step=0.01)
x = st.number_input('Length (X)', min_value=0.0, max_value=10.0, step=0.01)
y = st.number_input('Length (Y)', min_value=0.0, max_value=10.0, step=0.01)
z = st.number_input('Length (Z)', min_value=0.0, max_value=10.0, step=0.01)



# Create dataframe from inputs
input_data = {
    'carat': carat,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z
}

# Predict button
if st.button('Predict'):
    try:
        response = requests.post(PREDICT_API_URL, json=input_data)
        if response.status_code == 200:
            predictions = response.json()
            st.write(predictions)
    except:
        print("Error. Please, Try Again or Contact Us")

st.divider()

# Similar Diamonds Finder
st.header('Find similar endpoint: .../predict')
st.write("Based on: Cut, Color, Clarity, weight")
n_samples = st.number_input('Number of similar diamonds to find', min_value=1, max_value=100, value=5, step=1)

# Find Similar Diamonds button
if st.button('Find Similar Diamonds'):
    try:
        similar_diamonds_input = {
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'weight': carat,
            'n': n_samples
        }
        response = requests.post(SIMILAR_DIAMONDS_API_URL, json=similar_diamonds_input)
        if response.status_code == 200:
            similar_diamonds = response.json()
            st.write(similar_diamonds)
        else:
            st.error("Error in fetching similar diamonds. Please try again.")
    except Exception as e:
        st.error(f"Error: {e}. Please try again or contact support.")

