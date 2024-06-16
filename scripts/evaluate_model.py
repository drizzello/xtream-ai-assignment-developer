from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test, log_transform):
   """
   Evaluate the given model on the test data.

   Parameters:
   model (sklearn model): Trained model.
   X_test (pd.DataFrame): Test features.
   y_test (pd.Series): Test target values.
   log_transform (bool): Whether to apply exponential transformation to predictions.

   Returns:
   tuple: MAE and R2 score of the model.
   """
   y_pred = model.predict(X_test)
    
   if log_transform:
      y_pred = np.exp(y_pred)  # Apply exponential transformation to reverse log scaling

   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)

   print(f'MAE: {mae}')
   print(f'R2: {r2}')

   return mae, r2
