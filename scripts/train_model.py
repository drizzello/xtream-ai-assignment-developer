from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost
import numpy as np

def train_linear_model(df):
   """
   Train a linear regression model.

   Parameters:
   df (pd.DataFrame): The dataframe containing features and target variable.

   Returns:
   tuple: A tuple containing the trained model, test features, and test target variable.
   """
   
   # Split the data into features and target variable
   x = df.drop(columns='price')
   y = df.price

   # Split the data into training and testing sets
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   
   y_train = np.log(y_train)

   # Define and train the model
   model = LinearRegression()
   model.fit(x_train, y_train)
    
   return model, x_test, y_test

def train_xgboost_model(df):
   """
   Train an XGBoost regression model.

   Parameters:
   df (pd.DataFrame): The dataframe containing features and target variable.

   Returns:
   tuple: A tuple containing the trained model, test features, and test target variable.
   """

   # Split the data into features and target variable
   x = df.drop(columns='price')
   y = df.price

   # Split the data into training and testing sets
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   
   model = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
   model.fit(x_train, y_train)
    
   return model, x_test, y_test

