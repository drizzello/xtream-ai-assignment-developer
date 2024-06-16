import argparse
import pandas as pd
from data_preprocessing import load_data
from evaluate_model import evaluate_model
from save_model import save_model_and_metrics
from models_registry import model_registry


def main(model_type, data_url):
    """
    Main function to run the ML pipeline.

    Parameters:
    model_type (str): Type of the model (e.g., 'linear', 'xgboost').
    data_url (str): Path to the input data file.
    """
    try:
        # Load and preprocess data
        df = load_data(data_url)
        
        if model_type in model_registry:
            preprocess_fn = model_registry[model_type]['preprocess']
            train_fn = model_registry[model_type]['train']
            log_transform = model_registry[model_type]['log_transform']
            
            df = preprocess_fn(df)
            
            model, X_test, y_test = train_fn(df)
        else:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose a supported model type.")

        # Evaluate the model
        mae, r2 = evaluate_model(model, X_test, y_test, log_transform)

        # Save the model and metrics
        metrics = {'MAE': mae, 'R2': r2}
        save_model_and_metrics(model, metrics)
    
    except Exception as e:
        print("An error occurred: %s", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    parser.add_argument('data_url', type=str, help="Path to the input data file")
    parser.add_argument('model_type', choices=model_registry.keys(), help="Type of model to run")
    
    args = parser.parse_args()
    
    main(args.model_type, args.data_url)
