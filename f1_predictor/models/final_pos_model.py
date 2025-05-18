import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import warnings
import os
from abc import ABC, abstractmethod # For abstract base class
from models.base_model import AbstractBasePredictor
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define data directory at the module level or pass to class
DATA_DIR_GLOBAL = "test_output/dataset"


class FinalPositionPredictor(AbstractBasePredictor):
    def __init__(self, data_dir=DATA_DIR_GLOBAL, model_params=None, test_size_ratio=0.20):
        super().__init__(data_dir=data_dir, test_size_ratio=test_size_ratio)
        self.model_params_config = model_params 
        self.model = None 

    def _get_default_model_params(self):
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 1e-2,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'random_state': 42,
            'eval_metric': ["rmse", "mae"] 
        }

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded or split. Call load_and_prepare_data() first.")

        current_params = self.model_params_config if self.model_params_config else self._get_default_model_params()
        
        fit_params = {}
        constructor_params = current_params.copy()

        if 'early_stopping_rounds' in constructor_params: 
            fit_params['early_stopping_rounds'] = constructor_params.pop('early_stopping_rounds')
        elif self.model_params_config and 'early_stopping_rounds' in self.model_params_config:
             fit_params['early_stopping_rounds'] = self.model_params_config['early_stopping_rounds']


        self.model = xgb.XGBRegressor(**constructor_params)
        print("\n--- Training XGBoost model ---")
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        self.model.fit(self.X_train, self.y_train,
                       eval_set=eval_set,
                       verbose=False,
                       **fit_params)
        print("Model training finished.")
        
        try:
            self.evals_result = self.model.evals_result()
        except AttributeError:
            print("Warning: model.evals_result() not found. Plotting training progress will be skipped.")
            self.evals_result = None
            
    def predict(self, X_data):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        return self.model.predict(X_data)

    def get_feature_importances(self):
        if self.model is None:
            print("Model not trained yet.")
            return None
        if self.X is None or self.X.empty : 
            print("Features (X) not available for importance. Ensure data is loaded.")
            return None
            
        try:
            importances = pd.Series(self.model.feature_importances_, index=self.X.columns)
            return importances.sort_values(ascending=False)
        except Exception as e:
            print(f"Could not retrieve feature importances: {e}")
            return None
