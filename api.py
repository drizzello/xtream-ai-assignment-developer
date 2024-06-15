from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np


def preprocess_data(df):
    # Check if all required columns are present
    required_columns = ['x', 'cut', 'color', 'clarity', 'carat']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

    # Get dummy variables
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
    
    # Ensure all dummy variables from training are present
    training_columns = [
        'carat', 'x', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good',
        'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
        'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2'
    ]
    
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[training_columns]

    return df

app = Flask(__name__)

# Load the model and the training dataset
model = joblib.load('/Users/daviderizzello/Documents/Data_Science_Projects/xtream_project/xtream-ai-assignment-developer/models/diamond_model.pkl')

df = pd.read_csv('/Users/daviderizzello/Documents/Data_Science_Projects/xtream_project/xtream-ai-assignment-developer/data/diamonds.csv')
df_processed = preprocess_data(df)

# Endpoint to predict the value of a diamond
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    
    try:
        processed_data = preprocess_data(input_data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    prediction = model.predict(processed_data)
    price = float(np.exp(prediction)[0])
    return jsonify({'predicted_price': price})

# Endpoint to return similar diamond samples
@app.route('/similar', methods=['POST'])
def similar():
    data = request.json
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    weight = data['carat']
    n = int(data.get('n', 5))

    # Filter dataset based on cut, color, and clarity
    filtered_df = df[(df['cut'] == cut) & (df['color'] == color) & (df['clarity'] == clarity)]

    # Calculate the difference in weight
    filtered_df['weight_diff'] = (filtered_df['carat'] - weight).abs()

    # Get n samples with the smallest weight difference
    similar_samples = filtered_df.nsmallest(n, 'weight_diff')

    # Drop the weight_diff column before returning the result
    similar_samples = similar_samples.drop(columns=['weight_diff'])
    
    return similar_samples.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
