import joblib
import json
from datetime import datetime
import os

def get_project_root():
   """
   Get the project root directory.
   """
   current_dir = os.path.dirname(os.path.abspath(__file__))
   return os.path.abspath(os.path.join(current_dir, os.pardir))

def save_model_and_metrics(model, metrics, all_models_dir='models', metrics_path='models/all_metrics.json',    best_model_dir='models/best_model'):
   project_root = get_project_root()
   all_models_dir = os.path.join(project_root, all_models_dir)
   metrics_path = os.path.join(project_root, metrics_path)
   best_model_dir = os.path.join(project_root, best_model_dir)

   # Add training date to metrics
   metrics['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   model_name = f"model_{metrics['training_date'].replace(' ', '_').replace(':', '-')}.pkl"

   # Create directories if they don't exist
   os.makedirs(all_models_dir, exist_ok=True)
   os.makedirs(best_model_dir, exist_ok=True)

   # Save the current model
   joblib.dump(model, os.path.join(all_models_dir, model_name))

   # Load existing metrics if the file exists and is not empty
   if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
      with open(metrics_path, 'r') as f:
         all_metrics = json.load(f)
   else:
      all_metrics = []

   # Append new metrics to all metrics
   all_metrics.append(metrics)

   # Save all metrics
   with open(metrics_path, 'w') as f:
      json.dump(all_metrics, f, indent=4)

    # Load existing best metrics if available
   best_metrics_path = os.path.join(best_model_dir, 'best_model_metrics.json')
   best_model_path = os.path.join(best_model_dir, 'best_model.pkl')
   is_best_model = False

   if os.path.exists(best_metrics_path) and os.path.getsize(best_metrics_path) > 0:
      with open(best_metrics_path, 'r') as f:
         best_metrics = json.load(f)
      if metrics['R2'] > best_metrics['R2']:
         is_best_model = True
   else:
      is_best_model = True

   # If this is the best model, update the best model and metrics
   if is_best_model:
      joblib.dump(model, best_model_path)
      with open(best_metrics_path, 'w') as f:
         json.dump(metrics, f, indent=4)
         print('New best model saved.')
   else:
      print('Model did not outperform the existing best model.')

   print('Model and metrics saved successfully.')
