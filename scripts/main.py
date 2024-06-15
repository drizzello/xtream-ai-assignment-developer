import pandas as pd
from data_preprocessing import load_and_preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from save_model import save_model_and_metrics

def main():
    # Load and preprocess data
    data_url = "data/diamonds.csv"
    df = load_and_preprocess_data(data_url)

    # Train the model
    model, X_test, y_test = train_model(df)

    # Evaluate the model
    mae, r2 = evaluate_model(model, X_test, y_test)

    # Save the model and metrics
    metrics = {
        'MAE': mae,
        'R2': r2
    }
    save_model_and_metrics(model, metrics)

if __name__ == "__main__":
    main()
