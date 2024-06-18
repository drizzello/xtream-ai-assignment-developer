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

def preprocess_categorical_single_row(row): 
   """
   Preprocess a single row of categorical data.
   Used for API request call.

   Parameters:
   row (pd.Series): A row of the dataframe.

   Returns:
   pd.DataFrame: A dataframe with one-hot encoded categorical features and other columns.
   """

   cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
   color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
   clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
   dummy_columns = (
      ['cut_' + cat for cat in cut_categories[0:]] +
      ['color_' + cat for cat in color_categories[0:]] +
      ['clarity_' + cat for cat in clarity_categories[0:]]
   )
   dummy_template = pd.DataFrame(data=[[0] * len(dummy_columns)], columns=dummy_columns)
    
   if 'cut_' + row['cut'] in dummy_template.columns:
      dummy_template['cut_' + row['cut']] = 1
   if 'color_' + row['color'] in dummy_template.columns:
      dummy_template['color_' + row['color']] = 1
   if 'clarity_' + row['clarity'] in dummy_template.columns:
      dummy_template['clarity_' + row['clarity']] = 1
    
   non_dummy_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
   for col in non_dummy_cols:
      if col in row:
         dummy_template[col] = row[col]
    
   return dummy_template

def preprocess_categorical_multiple_rows(df):
   """
   Preprocess multiple rows of categorical data.

   Parameters:
   df (pd.DataFrame): The dataframe to preprocess.

   Returns:
   pd.DataFrame: A dataframe with one-hot encoded categorical features and other columns.
   """

   df = df.copy()  

   cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
   color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
   clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
   df['cut'] = pd.Categorical(df['cut'], categories=cut_categories)
   df['color'] = pd.Categorical(df['color'], categories=color_categories)
   df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_categories)
    
   df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=False)
    
   all_columns = (
      ['cut_' + cat for cat in cut_categories[1:]] +
      ['color_' + cat for cat in color_categories[1:]] +
      ['clarity_' + cat for cat in clarity_categories[1:]]
   )
   for col in all_columns:
      if col not in df.columns:
         df[col] = 0

   return df

def linear_model_preprocess(df):
   """
   Preprocess the dataframe for the linear regression model.

   Parameters:
   df (pd.DataFrame): The dataframe to preprocess.

   Returns:
   pd.DataFrame: The preprocessed dataframe.
   """


   df = df.copy()  
    
   if df.shape[0] == 1:
      processed_df = preprocess_categorical_single_row(df.iloc[0])
   else:
      processed_df = preprocess_categorical_multiple_rows(df)

   expected_order = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Fair', 'cut_Good', 'cut_Very Good', 'cut_Premium', 'cut_Ideal', 'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_SI2', 'clarity_SI1', 'clarity_VS2', 'clarity_VS1', 'clarity_VVS2', 'clarity_VVS1', 'clarity_IF', 'price']

   # Reorder the DataFrame columns to match the expected order
   processed_df = processed_df[expected_order]
   
   processed_df = processed_df[(df.x * df.y * df.z != 0) & (df.price > 0)]
   processed_df = processed_df.drop(columns=['depth', 'table', 'y', 'z'])

   return processed_df

def xg_boost_preprocess(df):
   """
   Preprocess the dataframe for the XGBoost model.

   Parameters:
   df (pd.DataFrame): The dataframe to preprocess.

   Returns:
   pd.DataFrame: The preprocessed dataframe.
   """

   df = df.copy()  
   
   expected_order = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Fair', 'cut_Good', 'cut_Very Good', 'cut_Premium', 'cut_Ideal', 'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_SI2', 'clarity_SI1', 'clarity_VS2', 'clarity_VS1', 'clarity_VVS2', 'clarity_VVS1', 'clarity_IF', 'price']

   if df.shape[0] == 1:
      processed_df = preprocess_categorical_single_row(df.iloc[0])
   else:
      processed_df = preprocess_categorical_multiple_rows(df)

   # Reorder the DataFrame columns to match the expected order
   processed_df = processed_df[expected_order]

   processed_df = processed_df[(df.x * df.y * df.z != 0) & (df.price > 0)]
    
   return processed_df
