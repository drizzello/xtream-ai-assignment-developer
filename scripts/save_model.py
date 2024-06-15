import joblib
import json
from datetime import datetime
import os

def save_model_and_metrics(model, metrics, model_path='models/diamond_model.pkl', metrics_path='models/model_metrics.json', best_metrics_path='models/best_model_metrics.json'):
   # Add training date to metrics
   metrics['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

   # Determine if this model is the best
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
      joblib.dump(model, model_path)
      with open(best_metrics_path, 'w') as f:
         json.dump(metrics, f, indent=4)
      print('New best model saved.')
   else:
      print('Model not saved as it did not outperform the existing best model.')
