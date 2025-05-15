"""Ensemble combiner for aggregating model predictions."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class EnsembleCombiner:
    """
    Combines predictions from multiple models into a final ranking.
    Supports weighted averaging, Borda count, and Learning-to-Rank approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ensemble combiner.
        
        Args:
            config: Configuration dictionary from config.yaml.
        """
        self.config = config
        self.method = config['ensemble']['method']
        self.ltr_model = None
    
    def combine_ranks(self, model_ranks: Dict[str, pd.Series], 
                       expert_rank: Optional[pd.Series] = None) -> pd.Series:
        """
        Combine ranks from different models into a final ranking.
        
        Args:
            model_ranks: Dictionary mapping model names to rank Series.
            expert_rank: Optional expert-derived ranks.
            
        Returns:
            Series with final combined ranks.
        """
        if self.method == 'weighted_average':
            return self._weighted_average(model_ranks, expert_rank)
        elif self.method == 'borda_count':
            return self._borda_count(model_ranks, expert_rank)
        elif self.method == 'learning_to_rank':
            return self._learning_to_rank(model_ranks, expert_rank)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")
    
    def _weighted_average(self, model_ranks: Dict[str, pd.Series], 
                          expert_rank: Optional[pd.Series] = None) -> pd.Series:
        """
        Combine ranks using weighted average.
        
        Args:
            model_ranks: Dictionary mapping model names to rank Series.
            expert_rank: Optional expert-derived ranks.
            
        Returns:
            Series with final combined ranks.
        """
        # Build a DataFrame with all ranks
        combined_df = pd.DataFrame(model_ranks)
        
        # Add expert rank if provided
        if expert_rank is not None and self.config['ensemble']['weights'].get('expert_rank', 0) > 0:
            combined_df['expert_rank'] = expert_rank
        
        # Filter for weights that are actually present and non-zero
        active_weights = {k: v for k, v in self.config['ensemble']['weights'].items()
                          if k in combined_df.columns and v > 0}
        
        # If no active weights, return simple average
        if not active_weights:
            if combined_df.empty:
                return pd.Series(dtype='float64')
            return combined_df.mean(axis=1).rank(method='min')
        
        # Normalize weights to sum to 1
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
        
        # Calculate weighted average rank
        weighted_scores = pd.Series(0.0, index=combined_df.index)
        for model_name, weight in normalized_weights.items():
            weighted_scores += combined_df[model_name] * weight
        
        # Convert to ranks (lower score is better)
        return weighted_scores.rank(method='min')
    
    def _borda_count(self, model_ranks: Dict[str, pd.Series], 
                      expert_rank: Optional[pd.Series] = None) -> pd.Series:
        """
        Combine ranks using Borda count voting scheme.
        
        Args:
            model_ranks: Dictionary mapping model names to rank Series.
            expert_rank: Optional expert-derived ranks.
            
        Returns:
            Series with final combined ranks.
        """
        # Build a DataFrame with all ranks
        combined_df = pd.DataFrame(model_ranks)
        
        # Add expert rank if provided
        if expert_rank is not None:
            combined_df['expert_rank'] = expert_rank
        
        if combined_df.empty:
            return pd.Series(dtype='float64')
        
        # For each ranking list, assign points based on position
        # Number of points = (total_drivers - rank + 1)
        n_drivers = len(combined_df.index)
        
        # Convert ranks to points
        points_df = n_drivers + 1 - combined_df
        
        # Sum points across all ranking lists
        total_points = points_df.sum(axis=1)
        
        # Convert to ranks (higher points is better, rank 1)
        return total_points.rank(method='min', ascending=False)
    
    def _learning_to_rank(self, model_ranks: Dict[str, pd.Series], 
                           expert_rank: Optional[pd.Series] = None) -> pd.Series:
        """
        Combine ranks using a Learning-to-Rank model.
        
        Args:
            model_ranks: Dictionary mapping model names to rank Series.
            expert_rank: Optional expert-derived ranks.
            
        Returns:
            Series with final combined ranks.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required for learning-to-rank. Install with: pip install lightgbm")
        
        # Build features DataFrame from model ranks
        features_df = pd.DataFrame(model_ranks)
        
        # Add expert rank if provided
        if expert_rank is not None:
            features_df['expert_rank'] = expert_rank
        
        if features_df.empty:
            return pd.Series(dtype='float64')
        
        # Load the LTR model if not already loaded
        if self.ltr_model is None:
            model_path = self.config['ensemble']['ltr_model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"LTR model not found at {model_path}")
            self.ltr_model = lgb.Booster(model_file=model_path)
        
        # Make predictions
        scores = self.ltr_model.predict(features_df)
        
        # Convert to ranks (higher score is better, rank 1)
        return pd.Series(scores, index=features_df.index).rank(method='min', ascending=False)
    
    def train_ltr_model(self, train_ranks: List[Dict[str, pd.Series]], 
                         true_ranks: List[pd.Series]) -> None:
        """
        Train a Learning-to-Rank model for ensemble.
        
        Args:
            train_ranks: List of dictionaries mapping model names to rank Series
                         for multiple historical races.
            true_ranks: List of true race outcome rank Series.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required for learning-to-rank. Install with: pip install lightgbm")
        
        # Prepare training data
        x_train = []
        y_train = []
        query_lengths = []
        
        for race_ranks, race_true_ranks in zip(train_ranks, true_ranks):
            # Build features DataFrame from model ranks
            features_df = pd.DataFrame(race_ranks)
            
            if not features_df.empty:
                # Get feature values and true ranks
                x_train.append(features_df.values)
                y_train.append(race_true_ranks.values)
                query_lengths.append(len(features_df))
        
        if not x_train:
            raise ValueError("No training data provided")
        
        # Concatenate all race data
        x_train = np.vstack(x_train)
        y_train = np.concatenate(y_train)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            x_train, 
            label=y_train, 
            group=query_lengths,
            feature_name=list(train_ranks[0].keys())
        )
        
        # Extract parameters from config
        params = self.config['ensemble'].get('ltr_params', {})
        
        # Train model
        self.ltr_model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 100)
        )
        
        # Save model
        model_path = self.config['ensemble']['ltr_model_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.ltr_model.save_model(model_path) 