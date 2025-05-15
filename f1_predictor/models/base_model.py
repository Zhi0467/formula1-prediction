"""Abstract base model class for F1 prediction models."""

from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import Dict, Any, List


class BaseModel(ABC):
    """
    Abstract base class for all F1 prediction models.
    All model implementations should inherit from this class.
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            model_path: Path to save/load the model.
            config: Model-specific configuration.
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        pass
    
    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the model.
        
        Args:
            features_df: DataFrame of features for prediction.
            
        Returns:
            DataFrame with predictions.
        """
        pass
    
    @abstractmethod
    def train(self, train_features_df: pd.DataFrame, train_target_df: pd.DataFrame) -> None:
        """
        Train the model using provided features and target data.
        
        Args:
            train_features_df: DataFrame of training features.
            train_target_df: DataFrame of training targets.
        """
        pass
    
    @abstractmethod
    def save_model(self) -> None:
        """
        Save the trained model to disk.
        """
        pass
    
    @abstractmethod
    def get_rank_output(self, predictions_df: pd.DataFrame, 
                        qualifying_positions: pd.Series = None) -> pd.Series:
        """
        Transform model predictions into a ranking.
        
        Args:
            predictions_df: DataFrame with model predictions.
            qualifying_positions: Optional Series of qualifying positions.
            
        Returns:
            Series of ranks (1 is best).
        """
        pass
    
    def select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the features needed for this model from a larger feature set.
        
        Args:
            features_df: Complete feature DataFrame.
            
        Returns:
            DataFrame with only the required features.
        """
        features_to_use = self.config.get('features_to_use', [])
        
        # Check if all required features are available
        missing_features = [f for f in features_to_use if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for model: {missing_features}")
            
        return features_df[features_to_use] 