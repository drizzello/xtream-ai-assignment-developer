from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
   # Make predictions
   y_pred = model.predict(X_test)
   y_pred = np.exp(y_pred)

   # Evaluate the model
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)

   print(f'MAE: {mae}')
   print(f'R2: {r2}')
    
   return mae, r2
