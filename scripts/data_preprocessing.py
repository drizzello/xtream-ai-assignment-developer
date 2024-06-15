import pandas as pd

def load_and_preprocess_data(file_path):
   # Load the dataset
   df = pd.read_csv(file_path)
    
   #preprocessing steps from the notebook
   #removing negative prices and zero-dimensional stones
   df = df[(df.x * df.y * df.z != 0) & (df.price > 0)]
   
   #drop irrelevant columns for linear regression
   df = df.drop(columns=['depth', 'table', 'y', 'z'])

   #get dummy variables 
   df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)    

   

   return df

def preprocess_data(df):
    # Check if all required columns are present
    required_columns = ['x', 'cut', 'color', 'clarity']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

    # Preprocessing steps similar to load_and_preprocess_data but for in-memory data
    # Remove negative prices and zero-dimensional stones
    
    # Drop irrelevant columns for linear regression
    df = df.drop(columns=['depth', 'table', 'y', 'z'])

    # Get dummy variables 
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)    

    return df
