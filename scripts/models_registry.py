from data_preprocessing import linear_model_preprocess, xg_boost_preprocess
from train_model import train_linear_model, train_xgboost_model

"""
Model Registry

This module defines the registry of models and their corresponding preprocessing
training functions and parameters. The registry allows for easy addition of new models
by mapping model names to their respective functions.

Functions:
    linear_model_preprocess -- Preprocess data for the linear regression model
    train_linear_model -- Train the linear regression model
    xg_boost_preprocess -- Preprocess data for the XGBoost model
    train_xgboost_model -- Train the XGBoost model

Registry:
    model_registry -- Dictionary mapping model names to their preprocessing and training functions
"""

model_registry = {
    'linear': {
        'preprocess': linear_model_preprocess,
        'train': train_linear_model,
        'log_transform': True
    },
    'xgboost': {
        'preprocess': xg_boost_preprocess,
        'train': train_xgboost_model,
        'log_transform': False
    }
}
