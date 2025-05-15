"""Lap time prediction model for F1 races."""

import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Optional

from f1_predictor.models.base_model import BaseModel


class LapTimeModel(BaseModel):
    """
    Model for predicting lap times or total race times for drivers.
    This model uses XGBoost to predict average lap time or total race time
    for each driver, which is then used to rank the drivers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the lap time prediction model.
        
        Args:
            config: Model-specific configuration.
        """
        super().__init__(config['model_path'], config)
        self.model_type = config.get('type', 'xgboost')
    
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ImportError: If required dependencies are not installed.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(self.model_path)
            except ImportError:
                raise ImportError("XGBoost is required for this model. Install with: pip install xgboost")
        elif self.model_type == 'neural_network':
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
            except ImportError:
                raise ImportError("TensorFlow is required for neural network model. Install with: pip install tensorflow")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict lap/total times for drivers.
        
        Args:
            features_df: DataFrame of features for prediction.
            
        Returns:
            DataFrame with predicted times for each driver.
        """
        if self.model is None:
            self.load_model()
        
        # Select only the features required by this model
        model_features = self.select_features(features_df)
        
        if self.model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(model_features)
            predictions = self.model.predict(dmatrix)
        elif self.model_type == 'neural_network':
            predictions = self.model.predict(model_features.values)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create a DataFrame with driver index and predicted time
        return pd.DataFrame({
            'predicted_time': predictions
        }, index=features_df.index)
    
    def train(self, train_features_df: pd.DataFrame, train_target_df: pd.DataFrame) -> None:
        """
        Train the lap time prediction model.
        
        Args:
            train_features_df: DataFrame of training features.
            train_target_df: DataFrame with target times.
        """
        # Select only the features required by this model
        model_features = self.select_features(train_features_df)
        
        if self.model_type == 'xgboost':
            import xgboost as xgb
            
            # Extract hyperparameters from config
            params = self.config.get('hyperparameters', {})
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(model_features, label=train_target_df['lap_time'])
            
            # Train model
            self.model = xgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 100))
            
        elif self.model_type == 'neural_network':
            import tensorflow as tf
            
            # Build a simple dense neural network
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(model_features.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
            ])
            
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
            
            # Train the model
            self.model.fit(
                model_features.values, 
                train_target_df['lap_time'].values,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10, 
                        restore_best_weights=True
                    )
                ]
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def save_model(self) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if self.model_type == 'xgboost':
            self.model.save_model(self.model_path)
        elif self.model_type == 'neural_network':
            self.model.save(self.model_path)
    
    def get_rank_output(self, predictions_df: pd.DataFrame, 
                         qualifying_positions: Optional[pd.Series] = None) -> pd.Series:
        """
        Transform predicted times into a ranking.
        
        Args:
            predictions_df: DataFrame with predicted times for each driver.
            qualifying_positions: Not used for this model, but included for API consistency.
            
        Returns:
            Series of ranks (1 is best, i.e., fastest predicted time).
        """
        # Lower predicted time is better (rank 1)
        return predictions_df['predicted_time'].rank(method='min') 