import logging
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
from models_registry import model_registry, dynamic_import
import sqlite3
import json



def get_project_root():
    """
    Get the project root directory.

    Returns:
    str: The absolute path to the project root directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, os.pardir))

def load_best_model(best_model_dir='models/best_model'):
    """
    Load the best trained model from the specified directory.

    Parameters:
    best_model_dir (str): Directory where the best model is stored.

    Returns:
    tuple: The best model and its metadata.
    """

    project_root = get_project_root()
    best_model_dir = os.path.join(project_root, best_model_dir)
    best_model_path = os.path.join(best_model_dir, 'best_model.pkl')

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model found in directory: {best_model_path}")

    with open(best_model_path, 'rb') as model_file:
        model, metadata = joblib.load(model_file)
        model_name = metadata.get('model_name', 'unknown')

    return model, model_name

def load_training_data(data_path='data/diamonds.csv'):
    """
    Load the training data from the specified path.

    Parameters:
    data_path (str): Path to the training data file.

    Returns:
    pd.DataFrame: Loaded training data.
    """
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found in directory: {data_path}")
    return pd.read_csv(data_path)

def init_db():
    """
    Initialize the database for logging API requests and responses.
    """
    conn = sqlite3.connect('api_requests.db')
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS requests
              (id INTEGER PRIMARY KEY AUTOINCREMENT,
              endpoint TEXT,
              request_data TEXT,
              response_data TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
              ''')
    conn.commit()
    conn.close()

def save_request_response(endpoint, request_data, response_data):
    """
    Save the API request and response to the database.

    Parameters:
    endpoint (str): The API endpoint that was called.
    request_data (dict): The request data sent to the API.
    response_data (dict): The response data returned by the API.
    """
    conn = sqlite3.connect('api_requests.db')
    c = conn.cursor()
    c.execute('INSERT INTO requests (endpoint, request_data, response_data) VALUES (?, ?, ?)', 
              (endpoint, json.dumps(request_data), json.dumps(response_data)))
    conn.commit()
    conn.close()

app = Flask(__name__)

# Initialize the database
init_db()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the value of a diamond based on input features.

    Returns:
    JSON: The predicted value or an error message.
    """

    try:
        data = request.get_json()

        # Convert the JSON data to a DataFrame
        df = pd.DataFrame([data])

        # Check which model to use
        best_model, model_name = load_best_model()

        df['price'] = 1

        if model_name in model_registry:
            model_details = model_registry[model_name]

            # Dynamically import the preprocessing function
            preprocess_function = dynamic_import(model_details['preprocess_module'], model_details['preprocess_function'])

            # Preprocess the data
            preprocessed_df = preprocess_function(df)

            # Drop the 'price' column
            preprocessed_df = preprocessed_df.drop(columns=['price'])

            # Make predictions
            predictions = best_model.predict(preprocessed_df)

            # Apply log transformation if required
            if model_details['log_transform']:
                predictions = np.exp(predictions)

        response = {'predictions': predictions.tolist(), 'model_name': model_name}
        save_request_response('/predict', data, response)
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        response = {'error': 'An error occurred during prediction.'}
        save_request_response('/predict', data, response)
        return jsonify(response), 500
        
# return n most similar diamonds of same cut, color, and clarity based on weight
@app.route('/get_similar_diamonds', methods=['POST'])
def get_similar_diamonds():
    """
    Return n most similar diamonds of the same cut, color, and clarity based on weight.

    Returns:
    JSON: The similar diamonds or an error message.
    """

    try:
        data = request.get_json()
        given_cut = data['cut']
        given_color = data['color']
        given_clarity = data['clarity']
        given_weight = data['weight']
        n = data.get('n', 5)  # Default to 5 samples if not specified

        # Load the training data
        training_data = load_training_data()

        # Filter based on cut, color, and clarity
        filtered_data = training_data[
            (training_data['cut'] == given_cut) &
            (training_data['color'] == given_color) &
            (training_data['clarity'] == given_clarity)
        ]

        if filtered_data.empty:
            return jsonify({'error': 'No matching diamonds found.'}), 404

        # Calculate the absolute difference in weight
        filtered_data['weight_diff'] = (filtered_data['carat'] - given_weight).abs()

        # Sort by weight difference and select the top n samples
        similar_diamonds = filtered_data.nsmallest(n, 'weight_diff')

        # Drop the weight_diff column before returning
        similar_diamonds = similar_diamonds.drop(columns=['weight_diff'])

        response = similar_diamonds.to_json(orient='records')
        save_request_response('/get_similar_diamonds', data, response)
        return response

    except Exception as e:
        logging.error(f"Error finding similar diamonds: {e}")
        response = {'error': str(e)}
        save_request_response('/get_similar_diamonds', data, response)
        return jsonify(response), 500



if __name__ == '__main__':
    app.run(debug=True)
