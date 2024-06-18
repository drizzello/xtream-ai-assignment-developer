import importlib

"""
Model Registry

This module defines the registry of models and their corresponding preprocessing,
training functions, and parameters. The registry allows for easy addition of new models
by mapping model names to their respective functions.

Functions:
   dynamic_import -- Dynamically imports a function from a given module.

Registry:
   model_registry -- Dictionary mapping model names to their preprocessing and training functions.
"""

model_registry = {
   'linear': {
      'preprocess_module': 'data_preprocessing',
      'preprocess_function': 'linear_model_preprocess',
      'train_module': 'train_model',
      'train_function': 'train_linear_model',
      'log_transform': True
   },
   'xgboost': {
      'preprocess_module': 'data_preprocessing',
      'preprocess_function': 'xg_boost_preprocess',
      'train_module': 'train_model',
      'train_function': 'train_xgboost_model',
      'log_transform': False
   }
   # Add more models here as needed
}

def dynamic_import(module_name, function_name):
   """
   Dynamically imports a function from a given module.

   Parameters:
   module_name (str): The name of the module from which to import the function.
   function_name (str): The name of the function to import.

   Returns:
   function: The imported function.
   """
   module = importlib.import_module(module_name)
   function = getattr(module, function_name)
   return function
