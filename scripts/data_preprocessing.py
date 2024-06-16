import pandas as pd

def load_data(file_path):
   """
   Load the dataset from a given file path.

   Parameters:
   file_path (str): The path to the CSV file.

   Returns:
   pd.DataFrame: Loaded dataframe.
   """

   df = pd.read_csv(file_path)
   
   return df

def linear_model_preprocess(df):
   """
   Preprocess the dataframe for linear regression.

   Parameters:
   df (pd.DataFrame): Raw dataframe.

   Returns:
   pd.DataFrame: Preprocessed dataframe for linear regression.
   """

   df = df[(df.x * df.y * df.z != 0) & (df.price > 0)]
   df = df.drop(columns=['depth', 'table', 'y', 'z'])
   df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)    
   
   return df

def xg_boost_preprocess(df):
   """
   Preprocess the dataframe for XGBoost model.

   Parameters:
   df (pd.DataFrame): Raw dataframe.

   Returns:
   pd.DataFrame: Preprocessed dataframe for XGBoost.
   """
   
   df = df[(df.x * df.y * df.z != 0) & (df.price > 0)]
   df['cut'] = pd.Categorical(df['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
   df['color'] = pd.Categorical(df['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
   df[f'clarity'] = pd.Categorical(df['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)

   return df
