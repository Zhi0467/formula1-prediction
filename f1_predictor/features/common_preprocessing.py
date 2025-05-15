"""Common preprocessing functions for cleaning F1 data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def standardize_driver_names(df: pd.DataFrame, name_col: str = 'driver_name') -> pd.DataFrame:
    """
    Standardize driver names to ensure consistency across different data sources.
    
    Args:
        df: DataFrame containing driver names.
        name_col: Column name containing driver names.
        
    Returns:
        DataFrame with standardized driver names.
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Common name variations and their standardized form
    # This list should be expanded based on the specific data sources and their conventions.
    name_mapping = {
        # Current/Recent Drivers (Examples from ~2020-2025)
        'Lewis Hamilton': 'Lewis Hamilton',
        'L. Hamilton': 'Lewis Hamilton',
        'HAM': 'Lewis Hamilton',

        'Max Verstappen': 'Max Verstappen',
        'M. Verstappen': 'Max Verstappen',
        'VER': 'Max Verstappen',

        'Charles Leclerc': 'Charles Leclerc',
        'C. Leclerc': 'Charles Leclerc',
        'LEC': 'Charles Leclerc',

        'George Russell': 'George Russell',
        'G. Russell': 'George Russell',
        'RUS': 'George Russell',

        'Sergio Pérez': 'Sergio Pérez', # With accent
        'Sergio Perez': 'Sergio Pérez', # Without accent
        'S. Pérez': 'Sergio Pérez',
        'S. Perez': 'Sergio Pérez',
        'PER': 'Sergio Pérez',
        'Checo Pérez': 'Sergio Pérez',
        'Checo Perez': 'Sergio Pérez',


        'Carlos Sainz Jr.': 'Carlos Sainz',
        'Carlos Sainz': 'Carlos Sainz',
        'C. Sainz Jr.': 'Carlos Sainz',
        'C. Sainz': 'Carlos Sainz',
        'SAI': 'Carlos Sainz',

        'Lando Norris': 'Lando Norris',
        'L. Norris': 'Lando Norris',
        'NOR': 'Lando Norris',

        'Fernando Alonso': 'Fernando Alonso',
        'F. Alonso': 'Fernando Alonso',
        'ALO': 'Fernando Alonso',

        'Esteban Ocon': 'Esteban Ocon',
        'E. Ocon': 'Esteban Ocon',
        'OCO': 'Esteban Ocon',

        'Pierre Gasly': 'Pierre Gasly',
        'P. Gasly': 'Pierre Gasly',
        'GAS': 'Pierre Gasly',

        'Valtteri Bottas': 'Valtteri Bottas',
        'V. Bottas': 'Valtteri Bottas',
        'BOT': 'Valtteri Bottas',

        'Zhou Guanyu': 'Zhou Guanyu',
        'Guanyu Zhou': 'Zhou Guanyu', # Common alternative order
        'G. Zhou': 'Zhou Guanyu',
        'ZHO': 'Zhou Guanyu',

        'Yuki Tsunoda': 'Yuki Tsunoda',
        'Y. Tsunoda': 'Yuki Tsunoda',
        'TSU': 'Yuki Tsunoda',

        'Kevin Magnussen': 'Kevin Magnussen',
        'K. Magnussen': 'Kevin Magnussen',
        'MAG': 'Kevin Magnussen',

        'Nico Hülkenberg': 'Nico Hülkenberg', # With umlaut
        'Nico Hulkenberg': 'Nico Hülkenberg', # Without umlaut
        'N. Hülkenberg': 'Nico Hülkenberg',
        'N. Hulkenberg': 'Nico Hülkenberg',
        'HUL': 'Nico Hülkenberg',

        'Lance Stroll': 'Lance Stroll',
        'L. Stroll': 'Lance Stroll',
        'STR': 'Lance Stroll',

        'Alexander Albon': 'Alexander Albon',
        'A. Albon': 'Alexander Albon',
        'ALB': 'Alexander Albon',

        'Oscar Piastri': 'Oscar Piastri',
        'O. Piastri': 'Oscar Piastri',
        'PIA': 'Oscar Piastri',

        'Daniel Ricciardo': 'Daniel Ricciardo',
        'D. Ricciardo': 'Daniel Ricciardo',
        'RIC': 'Daniel Ricciardo',

        'Logan Sargeant': 'Logan Sargeant',
        'L. Sargeant': 'Logan Sargeant',
        'SAR': 'Logan Sargeant',
        
        'Liam Lawson': 'Liam Lawson',
        'L. Lawson': 'Liam Lawson',
        'LAW': 'Liam Lawson',

        'Oliver Bearman': 'Oliver Bearman',
        'O. Bearman': 'Oliver Bearman',
        'BEA': 'Oliver Bearman', # Assuming BEA if he gets a code

        # Notable Past Drivers (Examples)
        'Sebastian Vettel': 'Sebastian Vettel',
        'S. Vettel': 'Sebastian Vettel',
        'VET': 'Sebastian Vettel',

        'Kimi Räikkönen': 'Kimi Räikkönen', # With umlaut
        'Kimi Raikkonen': 'Kimi Räikkönen', # Without umlaut
        'K. Räikkönen': 'Kimi Räikkönen',
        'K. Raikkonen': 'Kimi Räikkönen',
        'RAI': 'Kimi Räikkönen',

        'Mick Schumacher': 'Mick Schumacher',
        'M. Schumacher': 'Mick Schumacher',
        'MSC': 'Mick Schumacher',

        'Nicholas Latifi': 'Nicholas Latifi',
        'N. Latifi': 'Nicholas Latifi',
        'LAT': 'Nicholas Latifi',

        'Antonio Giovinazzi': 'Antonio Giovinazzi',
        'A. Giovinazzi': 'Antonio Giovinazzi',
        'GIO': 'Antonio Giovinazzi',

        'Nikita Mazepin': 'Nikita Mazepin',
        'N. Mazepin': 'Nikita Mazepin',
        'MAZ': 'Nikita Mazepin', # Official code

        'Romain Grosjean': 'Romain Grosjean',
        'R. Grosjean': 'Romain Grosjean',
        'GRO': 'Romain Grosjean',

        'Daniil Kvyat': 'Daniil Kvyat',
        'D. Kvyat': 'Daniil Kvyat',
        'KVY': 'Daniil Kvyat',

        'Nyck de Vries': 'Nyck de Vries',
        'N. de Vries': 'Nyck de Vries',
        'DEV': 'Nyck de Vries',

        'Robert Kubica': 'Robert Kubica',
        'R. Kubica': 'Robert Kubica',
        'KUB': 'Robert Kubica',

        # Add more mappings as identified from your specific datasets
    }
    
    # Apply standardization if the specified name column exists in the DataFrame
    if name_col in result_df.columns:
        # Define a helper function to apply the mapping
        # It returns the standardized name if found in mapping, otherwise the original name
        def standardize_name(name):
            if pd.isna(name): # Handle potential NaN values
                return name
            # Check both original and lowercased versions if needed, though current mapping is case-sensitive
            return name_mapping.get(name, name) # .get is safer than direct access
        
        result_df[name_col] = result_df[name_col].apply(standardize_name)
    else:
        print(f"Warning: Column '{name_col}' not found in DataFrame. No standardization applied.")
        
    return result_df

def standardize_team_names(df: pd.DataFrame, team_col: str = 'team_name') -> pd.DataFrame:
    """
    Standardize team names to ensure consistency across different data sources.
    
    Args:
        df: DataFrame containing team names.
        team_col: Column name containing team names.
        
    Returns:
        DataFrame with standardized team names.
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Common team name variations and their standardized form
    # This list should be expanded based on the specific data sources and their conventions.
    # Standardized names generally aim for the core constructor identity.
    team_mapping = {
        # Mercedes
        'Mercedes': 'Mercedes',
        'Mercedes AMG': 'Mercedes',
        'Mercedes-AMG Petronas F1 Team': 'Mercedes',
        'Mercedes AMG Petronas F1 Team': 'Mercedes',
        'Mercedes-AMG Petronas Formula One Team': 'Mercedes',

        # Red Bull Racing
        'Red Bull': 'Red Bull Racing',
        'Red Bull Racing': 'Red Bull Racing',
        'Oracle Red Bull Racing': 'Red Bull Racing',
        'Red Bull Racing Honda': 'Red Bull Racing', # Historical
        'Aston Martin Red Bull Racing': 'Red Bull Racing', # Historical

        # Ferrari
        'Ferrari': 'Ferrari',
        'Scuderia Ferrari': 'Ferrari',
        'Scuderia Ferrari HP': 'Ferrari', # Recent sponsor
        'Mission Winnow Ferrari': 'Ferrari', # Historical sponsor

        # McLaren
        'McLaren': 'McLaren',
        'McLaren F1 Team': 'McLaren',
        'McLaren Mercedes': 'McLaren', # Historical engine supplier
        'McLaren Honda': 'McLaren', # Historical engine supplier
        'McLaren Renault': 'McLaren', # Historical engine supplier

        # Aston Martin
        'Aston Martin': 'Aston Martin',
        'Aston Martin Aramco F1 Team': 'Aston Martin',
        'Aston Martin Cognizant F1 Team': 'Aston Martin', # Old sponsor
        'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
        'BWT Racing Point F1 Team': 'Racing Point', # Predecessor
        'Racing Point': 'Racing Point', # Predecessor

        # Alpine / Renault / Lotus
        'Alpine': 'Alpine',
        'Alpine F1 Team': 'Alpine',
        'BWT Alpine F1 Team': 'Alpine',
        'Renault': 'Renault', # Predecessor/historical
        'Renault F1 Team': 'Renault',
        'Lotus F1 Team': 'Lotus F1', # Predecessor
        'Lotus': 'Lotus F1',

        # Williams
        'Williams': 'Williams',
        'Williams Racing': 'Williams',
        'ROKiT Williams Racing': 'Williams', # Historical sponsor

        # RB (Visa Cash App RB) / AlphaTauri / Toro Rosso
        'RB': 'RB F1 Team',
        'VCARB': 'RB F1 Team',
        'Visa Cash App RB Formula One Team': 'RB F1 Team',
        'Racing Bulls': 'RB F1 Team', # Sometimes used
        'AlphaTauri': 'AlphaTauri', # Predecessor
        'Scuderia AlphaTauri': 'AlphaTauri',
        'Scuderia AlphaTauri Honda': 'AlphaTauri',
        'Toro Rosso': 'Toro Rosso', # Predecessor
        'Scuderia Toro Rosso': 'Toro Rosso',

        # Sauber / Kick Sauber / Alfa Romeo Racing
        'Sauber': 'Sauber',
        'Kick Sauber': 'Sauber', # Current branding
        'Stake F1 Team Kick Sauber': 'Sauber',
        'Stake F1 Team': 'Sauber',
        'Alfa Romeo': 'Alfa Romeo Racing', # Predecessor branding
        'Alfa Romeo Racing': 'Alfa Romeo Racing',
        'Alfa Romeo F1 Team Stake': 'Alfa Romeo Racing',
        'Alfa Romeo Racing Orlen': 'Alfa Romeo Racing',

        # Haas
        'Haas': 'Haas F1 Team',
        'Haas F1 Team': 'Haas F1 Team',
        'MoneyGram Haas F1 Team': 'Haas F1 Team',
        'Uralkali Haas F1 Team': 'Haas F1 Team', # Historical sponsor

        # Historical Teams (Examples)
        'Force India': 'Force India',
        'Sahara Force India F1 Team': 'Force India',
        'BWT Force India F1 Team': 'Force India',
        'Jordan': 'Jordan Grand Prix',
        'Minardi': 'Minardi',
        'Toyota': 'Toyota Racing',
        'BMW Sauber': 'BMW Sauber',
        'Caterham': 'Caterham F1 Team',
        'Marussia': 'Marussia F1 Team',
        'Manor Marussia F1 Team': 'Manor Racing',
        'Manor Racing': 'Manor Racing',
        'HRT': 'HRT F1 Team',
        'Hispania Racing F1 Team': 'HRT F1 Team',

        # Common abbreviations or alternative names
        'Mercedes GP': 'Mercedes',
        'Red Bull Honda': 'Red Bull Racing',
        'Scuderia Ferrari Marlboro': 'Ferrari', # Very old sponsor
        'West McLaren Mercedes': 'McLaren', # Very old sponsor
        
        # Add more mappings as identified from your specific datasets
    }
    
    # Apply standardization if the specified team column exists in the DataFrame
    if team_col in result_df.columns:
        # Define a helper function to apply the mapping
        # It returns the standardized name if found in mapping, otherwise the original name
        def standardize_team(name):
            if pd.isna(name): # Handle potential NaN values
                return name
            # Check both original and lowercased versions if needed, though current mapping is case-sensitive
            return team_mapping.get(name, name) # .get is safer than direct access
        
        result_df[team_col] = result_df[team_col].apply(standardize_team)
    else:
        print(f"Warning: Column '{team_col}' not found in DataFrame. No standardization applied.")
        
    return result_df

def handle_missing_values(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame according to configuration.
    
    Args:
        df: DataFrame with potential missing values.
        config: Configuration for handling missing values.
        
    Returns:
        DataFrame with handled missing values.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    if config is None:
        config = {}
    
    # Default strategies for different column types
    numeric_strategy = config.get('numeric_strategy', 'mean')
    categorical_strategy = config.get('categorical_strategy', 'mode')
    time_strategy = config.get('time_strategy', 'median')
    
    # Process numeric columns
    numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if result_df[col].isna().any():
            if numeric_strategy == 'mean':
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            elif numeric_strategy == 'median':
                result_df[col] = result_df[col].fillna(result_df[col].median())
            elif numeric_strategy == 'zero':
                result_df[col] = result_df[col].fillna(0)
    
    # Process categorical columns
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if result_df[col].isna().any():
            if categorical_strategy == 'mode':
                mode_value = result_df[col].mode()[0] if not result_df[col].mode().empty else "Unknown"
                result_df[col] = result_df[col].fillna(mode_value)
            elif categorical_strategy == 'unknown':
                result_df[col] = result_df[col].fillna("Unknown")
    
    # Special handling for time columns (if identified by names like 'time', 'lap_time', etc.)
    time_cols = [col for col in result_df.columns if any(time_word in col.lower() for time_word in ['time', 'duration'])]
    for col in time_cols:
        if col in result_df.columns and result_df[col].isna().any():
            if time_strategy == 'median':
                # For string time values, this is a simplification
                # A more sophisticated approach might convert to timedelta first
                median_value = result_df[col].median() if pd.api.types.is_numeric_dtype(result_df[col]) else None
                result_df[col] = result_df[col].fillna(median_value)
    
    return result_df


def convert_lap_times_to_seconds(df: pd.DataFrame, time_col: str = 'lap_time') -> pd.DataFrame:
    """
    Convert lap times in string format (e.g., '1:30.456') to seconds (float).
    
    Args:
        df: DataFrame containing lap times.
        time_col: Column name containing lap times.
        
    Returns:
        DataFrame with lap times converted to seconds.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    if time_col in result_df.columns and result_df[time_col].dtype == 'object':
        # Function to convert time string to seconds
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return np.nan
            
            # Handle different time formats
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # MM:SS.sss
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
                elif len(parts) == 3:  # HH:MM:SS.sss
                    hours, minutes, seconds = parts
                    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            else:
                # Just seconds
                return float(time_str)
        
        # Create a new column for seconds
        seconds_col = f"{time_col}_seconds"
        result_df[seconds_col] = result_df[time_col].apply(time_to_seconds)
    
    return result_df


def identify_outliers(df: pd.DataFrame, numeric_cols: List[str] = None, 
                     method: str = 'iqr', threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify outliers in numeric columns.
    
    Args:
        df: DataFrame to check for outliers.
        numeric_cols: List of numeric columns to check. If None, all numeric columns are used.
        method: Method for outlier detection ('iqr' or 'zscore').
        threshold: Threshold for outlier detection.
        
    Returns:
        Tuple of (DataFrame with outlier flags, DataFrame with only non-outlier rows).
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create outlier flag columns
    for col in numeric_cols:
        flag_col = f"{col}_is_outlier"
        
        if method == 'iqr':
            # IQR method
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            result_df[flag_col] = ((result_df[col] < lower_bound) | (result_df[col] > upper_bound))
        
        elif method == 'zscore':
            # Z-score method
            mean = result_df[col].mean()
            std = result_df[col].std()
            result_df[flag_col] = (abs(result_df[col] - mean) > threshold * std)
    
    # Create a DataFrame with only non-outlier rows
    outlier_flags = [f"{col}_is_outlier" for col in numeric_cols]
    non_outliers_df = result_df[~result_df[outlier_flags].any(axis=1)].drop(columns=outlier_flags)
    
    return result_df, non_outliers_df


def normalize_features(df: pd.DataFrame, numeric_cols: List[str] = None, 
                       method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numeric features.
    
    Args:
        df: DataFrame with features to normalize.
        numeric_cols: List of numeric columns to normalize. If None, all numeric columns are used.
        method: Normalization method ('standard', 'minmax', 'robust').
        
    Returns:
        DataFrame with normalized features.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for col in numeric_cols:
        if method == 'standard':
            # Standardization (z-score normalization)
            mean = result_df[col].mean()
            std = result_df[col].std()
            
            # To avoid division by zero
            if std == 0:
                result_df[f"{col}_normalized"] = 0
            else:
                result_df[f"{col}_normalized"] = (result_df[col] - mean) / std
        
        elif method == 'minmax':
            # Min-Max normalization
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            # To avoid division by zero
            if max_val == min_val:
                result_df[f"{col}_normalized"] = 0
            else:
                result_df[f"{col}_normalized"] = (result_df[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = result_df[col].median()
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # To avoid division by zero
            if IQR == 0:
                result_df[f"{col}_normalized"] = 0
            else:
                result_df[f"{col}_normalized"] = (result_df[col] - median) / IQR
    
    return result_df


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a standard preprocessing pipeline.
    
    Args:
        df: Raw DataFrame to preprocess.
        config: Configuration dictionary.
        
    Returns:
        Preprocessed DataFrame.
    """
    result_df = df.copy()
    
    # Extract preprocessing config for common steps
    # This assumes 'config' passed here is the 'feature_engineering' part of the main config
    # or contains a 'common' key directly.
    # Let's adjust to expect the full config and navigate down.
    main_feature_config = config.get('feature_engineering', {})
    preproc_config = main_feature_config.get('common', {})

    driver_name_col = 'driver_name' # Default, can be made configurable if needed
    team_name_col = 'team_name' # Default
    
    # Apply preprocessing steps based on configuration and column existence
    if preproc_config.get('standardize_driver_names', False) and driver_name_col in result_df.columns:
        result_df = standardize_driver_names(result_df, name_col=driver_name_col)
    
    if preproc_config.get('standardize_team_names', False) and team_name_col in result_df.columns:
        result_df = standardize_team_names(result_df, team_col=team_name_col)
    
    # Handle missing values
    # The missing_values config might be nested deeper, let's assume it's under 'common'
    missing_values_config = preproc_config.get('missing_values', {})
    result_df = handle_missing_values(result_df, missing_values_config)
    
    # Convert lap times if present (assuming 'finish_time' or other common names)
    # This could be made more specific if needed by passing a list of time columns from config
    time_cols_to_convert = [col for col in result_df.columns if 'time' in col.lower() or 'duration' in col.lower()]
    for time_col in time_cols_to_convert:
        if result_df[time_col].dtype == 'object': # Only try to convert if it's a string object
            result_df = convert_lap_times_to_seconds(result_df, time_col=time_col)
    
    # Additional preprocessing based on config
    if preproc_config.get('normalize_features', False):
        normalize_method = preproc_config.get('normalize_method', 'standard')
        # Pass numeric_cols explicitly if specified in config, otherwise auto-detect
        numeric_cols_to_normalize = preproc_config.get('numeric_cols_for_normalization', None)
        result_df = normalize_features(result_df, numeric_cols=numeric_cols_to_normalize, method=normalize_method)
    
    if preproc_config.get('remove_outliers', False):
        outlier_method = preproc_config.get('outlier_method', 'iqr')
        # Pass numeric_cols explicitly if specified in config, otherwise auto-detect
        numeric_cols_for_outliers = preproc_config.get('numeric_cols_for_outliers', None)
        _, result_df = identify_outliers(result_df, numeric_cols=numeric_cols_for_outliers, method=outlier_method)
    
    return result_df 