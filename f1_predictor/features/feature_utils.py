"""Utility functions for feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple


def calculate_rolling_average(
    df: pd.DataFrame, 
    value_col: str, 
    entity_col: str, 
    date_col: str,
    window: int = 3
) -> pd.Series:
    """
    Calculate rolling average of a value for each entity over a window of past events.
    
    Args:
        df: DataFrame containing the data.
        value_col: Column name of the value to average.
        entity_col: Column name of the entity (e.g., driver_id, team_id).
        date_col: Column name of the date or chronological order.
        window: Number of past events to include in the average.
        
    Returns:
        Series with rolling averages.
    """
    # Sort by date
    sorted_df = df.sort_values([entity_col, date_col])
    
    # Group by entity and calculate rolling average
    rolling_avg = sorted_df.groupby(entity_col)[value_col].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    
    return rolling_avg


def calculate_consistency(
    df: pd.DataFrame, 
    value_col: str, 
    entity_col: str, 
    date_col: str,
    window: int = 5,
    metric: str = 'std'
) -> pd.Series:
    """
    Calculate consistency (variation) of a value for each entity over a window of past events.
    
    Args:
        df: DataFrame containing the data.
        value_col: Column name of the value to analyze.
        entity_col: Column name of the entity (e.g., driver_id, team_id).
        date_col: Column name of the date or chronological order.
        window: Number of past events to include.
        metric: Metric of consistency ('std' for standard deviation, 'range' for max-min).
        
    Returns:
        Series with consistency values.
    """
    # Sort by date
    sorted_df = df.sort_values([entity_col, date_col])
    
    if metric == 'std':
        # Group by entity and calculate rolling standard deviation
        consistency = sorted_df.groupby(entity_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    elif metric == 'range':
        # Group by entity and calculate rolling range (max - min)
        consistency = sorted_df.groupby(entity_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).max() - x.rolling(window, min_periods=1).min()
        )
    
    # Replace NaN with 0 (for cases with less than 2 observations)
    consistency = consistency.fillna(0)
    
    return consistency


def one_hot_encode_categorical(
    df: pd.DataFrame, 
    categorical_cols: List[str],
    drop_original: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Args:
        df: DataFrame containing categorical columns.
        categorical_cols: List of categorical column names to encode.
        drop_original: Whether to drop the original columns.
        
    Returns:
        DataFrame with one-hot encoded columns.
    """
    result_df = df.copy()
    
    for col in categorical_cols:
        if col in result_df.columns:
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=False)
            
            # Concatenate with original DataFrame
            result_df = pd.concat([result_df, dummies], axis=1)
            
            # Drop original column if specified
            if drop_original:
                result_df = result_df.drop(columns=[col])
    
    return result_df


def calculate_time_delta(
    df: pd.DataFrame,
    time_col: str,
    reference_time_col: str = None,
    entity_col: str = None
) -> pd.Series:
    """
    Calculate time difference from a reference time.
    If reference_time_col is not provided, uses the minimum time in each group.
    
    Args:
        df: DataFrame containing time data.
        time_col: Column name of the time values.
        reference_time_col: Column name of reference time values.
        entity_col: Column name for grouping (e.g., race_id).
        
    Returns:
        Series with time deltas.
    """
    result_df = df.copy()
    
    if reference_time_col is not None:
        # Calculate delta from provided reference
        delta = result_df[time_col] - result_df[reference_time_col]
    elif entity_col is not None:
        # Calculate delta from minimum time in each group
        min_times = result_df.groupby(entity_col)[time_col].transform('min')
        delta = result_df[time_col] - min_times
    else:
        # Calculate delta from overall minimum
        min_time = result_df[time_col].min()
        delta = result_df[time_col] - min_time
    
    return delta


def calculate_position_changes(
    df: pd.DataFrame,
    start_pos_col: str,
    end_pos_col: str
) -> pd.Series:
    """
    Calculate position changes from start to end.
    
    Args:
        df: DataFrame containing position data.
        start_pos_col: Column name of starting positions.
        end_pos_col: Column name of ending positions.
        
    Returns:
        Series with position changes (negative for improvement).
    """
    # Positive values mean driver finished worse than started
    # Negative values mean driver improved
    return df[end_pos_col] - df[start_pos_col]


def calculate_historical_track_performance(
    df: pd.DataFrame,
    driver_col: str = 'driver_id',
    track_col: str = 'circuit_id',
    performance_col: str = 'position',
    date_col: str = 'race_date',
    better_is_lower: bool = True,
    aggregation: str = 'mean'
) -> pd.Series:
    """
    Calculate historical performance of drivers at specific tracks.
    
    Args:
        df: DataFrame containing historical race data.
        driver_col: Column name identifying drivers.
        track_col: Column name identifying tracks.
        performance_col: Column name with performance metric.
        date_col: Column name with chronological information.
        better_is_lower: Whether lower values are better (e.g., position).
        aggregation: Aggregation method ('mean', 'median', 'best', 'recent').
        
    Returns:
        Series with historical track performance.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Create driver-track groups
    driver_track_groups = df.groupby([driver_col, track_col])
    
    for (driver, track), group in driver_track_groups:
        # Get rows for current driver-track
        mask = (df[driver_col] == driver) & (df[track_col] == track)
        
        # Sort by date for temporal aggregations
        sorted_group = group.sort_values(date_col)
        
        if aggregation == 'mean':
            # Average performance
            value = group[performance_col].mean()
        elif aggregation == 'median':
            # Median performance
            value = group[performance_col].median()
        elif aggregation == 'best':
            # Best performance
            value = group[performance_col].min() if better_is_lower else group[performance_col].max()
        elif aggregation == 'recent':
            # Most recent performance
            if len(sorted_group) > 0:
                value = sorted_group.iloc[-1][performance_col]
            else:
                value = np.nan
        else:
            value = np.nan
        
        # Assign value to result Series
        result.loc[mask] = value
    
    return result


def create_track_type_features(
    df: pd.DataFrame,
    track_col: str = 'circuit_id',
    track_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Create features based on track types.
    
    Args:
        df: DataFrame containing track data.
        track_col: Column name identifying tracks.
        track_metadata: Dictionary mapping track IDs to metadata.
        
    Returns:
        DataFrame with additional track type features.
    """
    result_df = df.copy()
    
    if track_metadata is None:
        # Example track metadata - in a real implementation, this should be loaded from a file
        track_metadata = {
            'monaco': {
                'type': 'street',
                'downforce': 'high',
                'tire_wear': 'low',
                'avg_speed': 'low',
                'overtaking_difficulty': 'high'
            },
            'monza': {
                'type': 'traditional',
                'downforce': 'low',
                'tire_wear': 'medium',
                'avg_speed': 'high',
                'overtaking_difficulty': 'low'
            },
            # Add more tracks as needed
        }
    
    # Create features based on track metadata
    for track_id, metadata in track_metadata.items():
        # Find rows with this track
        mask = result_df[track_col] == track_id
        
        # Add metadata as features
        for attribute, value in metadata.items():
            feature_name = f'track_{attribute}'
            
            # For categorical attributes, create one-hot features
            if isinstance(value, str):
                # Create feature name with value (e.g., track_type_street)
                categorical_feature = f'{feature_name}_{value}'
                result_df.loc[mask, categorical_feature] = 1
                result_df.loc[~mask, categorical_feature] = 0
            else:
                # For numerical attributes
                result_df.loc[mask, feature_name] = value
                result_df.loc[~mask, feature_name] = np.nan
    
    return result_df 