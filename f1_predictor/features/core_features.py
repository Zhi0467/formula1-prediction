"""Core feature generation for F1 predictor."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from f1_predictor.features.feature_utils import *


def compute_core_features(
    race_data: Dict[str, pd.DataFrame],
    historical_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute core features for all drivers in a race.
    
    Args:
        race_data: Dictionary of DataFrames for the current race:
            - 'qualifying': qualifying results
            - 'weather': weather forecast
            - 'circuit': circuit information
            - 'drivers': current drivers info
            - 'teams': current teams info
        historical_data: Dictionary of historical DataFrames:
            - 'race_results': past race results
            - 'qualifying_results': past qualifying results
            - 'driver_standings': past driver standings
            - 'constructor_standings': past constructor standings
        config: Configuration dictionary.
    
    Returns:
        DataFrame with core features for all drivers.
    """
    print("\n--- Debug: compute_core_features (Start) ---")
    print(f"Available keys in race_data: {list(race_data.keys())}")

    qualifying_df = race_data.get('qualifying', pd.DataFrame())
    drivers_info_df = race_data.get('drivers', pd.DataFrame())

    print(f"Debug: qualifying_df empty? {qualifying_df.empty}")
    if not qualifying_df.empty:
        print(f"Debug: qualifying_df columns: {qualifying_df.columns.tolist()}")
        if 'driver_id' in qualifying_df.columns:
            print(f"Debug: qualifying_df['driver_id'] raw unique: {qualifying_df['driver_id'].unique()}")
            # Filter out None or NaN from driver IDs used for index
            valid_q_driver_ids = [did for did in qualifying_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            print(f"Debug: qualifying_df['driver_id'] valid unique for index: {valid_q_driver_ids}")
        else:
            print("Debug: 'driver_id' MISSING in qualifying_df")

    print(f"Debug: drivers_info_df (from standings) empty? {drivers_info_df.empty}")
    if not drivers_info_df.empty:
        print(f"Debug: drivers_info_df columns: {drivers_info_df.columns.tolist()}")
        if 'driver_id' in drivers_info_df.columns:
            print(f"Debug: drivers_info_df['driver_id'] raw unique: {drivers_info_df['driver_id'].unique()}")
            valid_d_driver_ids = [did for did in drivers_info_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            print(f"Debug: drivers_info_df['driver_id'] valid unique for index: {valid_d_driver_ids}")
        else:
            print("Debug: 'driver_id' MISSING in drivers_info_df")
    
    feature_config = config.get('feature_engineering', {}).get('core_features', {})
    recent_k_races = feature_config.get('recent_k_races', 5)
    
    # Initialize features_df to None, to clearly track if it gets created
    initialized_features_df = None

    # Strategy 1: Try to initialize from qualifying_df
    if not qualifying_df.empty and 'driver_id' in qualifying_df.columns:
        print("Debug: Attempting to initialize features_df index from qualifying_df")
        unique_q_driver_ids = [did for did in qualifying_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
        print(f"Debug: Valid unique driver_ids from qualifying_df: {unique_q_driver_ids} (Count: {len(unique_q_driver_ids)})")
        if len(unique_q_driver_ids) > 0:
            try:
                temp_df = pd.DataFrame(index=unique_q_driver_ids)
                temp_df.index.name = 'driver_id'
                initialized_features_df = temp_df # Assign here
                print(f"Debug: temp_df from qualifying_df. Shape: {temp_df.shape}, Index: {temp_df.index.tolist()}, Is empty: {temp_df.empty}")
            except Exception as e:
                print(f"Debug: ERROR creating DataFrame from qualifying_df driver_ids: {e}")
        else:
            print("Debug: No valid unique driver_ids in qualifying_df to create index.")
            
    # Strategy 2: If not successful, try from drivers_info_df
    if initialized_features_df is None or len(initialized_features_df.index) == 0: # Check if previous attempt failed or resulted in empty
        if 'drivers' in race_data and not drivers_info_df.empty and 'driver_id' in drivers_info_df.columns:
            print("Debug: Initializing from qualifying failed or resulted in empty. Attempting from drivers_info_df.")
            unique_d_driver_ids = [did for did in drivers_info_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            print(f"Debug: Valid unique driver_ids from drivers_info_df: {unique_d_driver_ids} (Count: {len(unique_d_driver_ids)})")
            if len(unique_d_driver_ids) > 0:
                try:
                    temp_df = pd.DataFrame(index=unique_d_driver_ids)
                    temp_df.index.name = 'driver_id'
                    initialized_features_df = temp_df # Assign here
                    print(f"Debug: temp_df from drivers_info_df. Shape: {temp_df.shape}, Index: {temp_df.index.tolist()}, Is empty: {temp_df.empty}")
                except Exception as e:
                    print(f"Debug: ERROR creating DataFrame from drivers_info_df driver_ids: {e}")
            else:
                print("Debug: No valid unique driver_ids in drivers_info_df to create index.")
        else:
            print("Debug: drivers_info_df not suitable for fallback index creation.")


    # Final check if initialized_features_df is usable
    if initialized_features_df is None or len(initialized_features_df.index) == 0:
        print("Warning: features_df has no rows after initialization. Returning empty DataFrame.")
        print(f"Debug: Final state before return - initialized_features_df is None: {initialized_features_df is None}")
        if initialized_features_df is not None:
             print(f"Debug: Final state before return - initialized_features_df.shape: {initialized_features_df.shape}, .empty: {initialized_features_df.empty}, .index: {initialized_features_df.index.tolist()}")
        print("--- End Debug: compute_core_features (Empty features_df) ---\n")
        return pd.DataFrame()

    # If we reach here, initialized_features_df should be a DataFrame with an index
    features_df = initialized_features_df # Assign to the main variable used by subsequent code
    print(f"Debug: features_df successfully initialized. Shape: {features_df.shape}, Index: {features_df.index.tolist()}, Is empty: {features_df.empty}")

    if feature_config.get('include_qualifying_data', True) and 'qualifying' in race_data:
        features_df = _add_qualifying_features(features_df, race_data['qualifying'])
    
    if 'race_results' in historical_data and not historical_data['race_results'].empty:
        features_df = _add_historical_performance_features(
            features_df, 
            historical_data['race_results'], 
            recent_k_races
        )
    
    if 'driver_standings' in historical_data and not historical_data['driver_standings'].empty:
        current_constructor_standings = historical_data.get('constructor_standings', pd.DataFrame())
        features_df = _add_standings_features(
            features_df, 
            historical_data['driver_standings'], 
            current_constructor_standings # Pass even if empty, function handles it
        )
    
    current_circuit_id: Optional[str] = None
    if 'circuit' in race_data and not race_data['circuit'].empty:
        circuit_info_df = race_data['circuit']
        if 'circuit_id' in circuit_info_df.columns and not circuit_info_df.empty:
            # Assuming circuit_info_df has one row for the current race's circuit
            current_circuit_id = circuit_info_df['circuit_id'].iloc[0] 
            print(f"Debug compute_core_features: Extracted current_circuit_id: {current_circuit_id}")
        else:
            print("Debug compute_core_features: 'circuit_id' column missing or empty in race_data['circuit']")
    else:
        print("Debug compute_core_features: 'circuit' data missing or empty in race_data")

    if current_circuit_id is not None and 'race_results' in historical_data and not historical_data['race_results'].empty:
        features_df = _add_track_specific_features(
            features_df,
            historical_data['race_results'],
            current_circuit_id, # Pass the scalar string value
            recent_k_races
        )
    
    if 'circuit' in race_data and not race_data['circuit'].empty:
        features_df = _add_track_type_features(features_df, race_data['circuit'])
    
    if feature_config.get('include_weather_data', True) and 'weather' in race_data and not race_data['weather'].empty:
        features_df = _add_weather_features(features_df, race_data['weather'])
    
    domain_knowledge_config = config.get('domain_knowledge', {})
    if domain_knowledge_config: # Check if domain_knowledge config exists and is not empty
        features_df = _add_domain_knowledge_features(features_df, domain_knowledge_config) # Pass sub-config
    
    print(f"Debug: result_df columns before returning: {features_df.columns.tolist()}")
    print(f"--- End Debug: compute_core_features ---\n")
    return features_df

def _add_qualifying_features(features_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add qualifying-related features.
    
    Args:
        features_df: Core features DataFrame.
        qualifying_df: Qualifying results DataFrame.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()

    if qualifying_df.empty:
        # Add placeholder columns if qualifying_df is empty to maintain consistent feature set
        placeholder_cols = [
            'grid_position', 'qualifying_time_q1', 'qualifying_time_q2', 
            'qualifying_time_q3', 'gap_to_pole_seconds'
        ]
        for col in placeholder_cols:
            if col not in result_df.columns: # Check if already present from features_df init
                 result_df[col] = np.nan
        return result_df

    # Ensure driver_id is a column for merging if features_df.index.name is 'driver_id'
    if result_df.index.name == 'driver_id':
        result_df = result_df.reset_index()
    
    # Select only relevant columns and handle potential missing driver_id in qualifying_df
    q_cols_to_merge = ['driver_id', 'position', 'Q1', 'Q2', 'Q3']
    valid_q_cols = [col for col in q_cols_to_merge if col in qualifying_df.columns]
    
    if 'driver_id' not in valid_q_cols: # Cannot merge without driver_id
        if 'driver_id' in result_df.columns: # If result_df was reset_index()
            result_df = result_df.set_index('driver_id')
        # Add placeholder columns as above
        placeholder_cols = [
            'grid_position', 'qualifying_time_q1', 'qualifying_time_q2', 
            'qualifying_time_q3', 'gap_to_pole_seconds'
        ]
        for col in placeholder_cols:
            if col not in result_df.columns:
                 result_df[col] = np.nan
        return result_df

    qualifying_to_merge = qualifying_df[valid_q_cols].copy()

    # Merge qualifying data
    result_df = pd.merge(
        result_df, 
        qualifying_to_merge, 
        on='driver_id', # Merge on driver_id column
        how='left'
    )
    
    # Set index back to driver_id if it was reset
    if 'driver_id' in result_df.columns:
         result_df = result_df.set_index('driver_id')
    
    # Rename columns to be more descriptive
    rename_map = {
        'position': 'grid_position',
        'Q1': 'qualifying_time_q1',
        'Q2': 'qualifying_time_q2',
        'Q3': 'qualifying_time_q3'
    }
    # Only rename columns that actually exist after the merge
    actual_rename_map = {k: v for k, v in rename_map.items() if k in result_df.columns}
    result_df = result_df.rename(columns=actual_rename_map)
    
    # Calculate gap to pole position
    if 'qualifying_time_q3' in result_df.columns:
        # Robust time parsing function
        def parse_time(time_str):
            if pd.isna(time_str) or not isinstance(time_str, str) or time_str.strip() == '':
                return np.nan
            
            time_str = time_str.strip() # Remove leading/trailing whitespace

            if ':' in time_str:
                parts = time_str.split(':')
                try:
                    if len(parts) == 2:  # MM:SS.sss
                        minutes, seconds_str = parts
                        return float(minutes) * 60 + float(seconds_str)
                    elif len(parts) == 3:  # HH:MM:SS.sss
                        hours, minutes, seconds_str = parts
                        return float(hours) * 3600 + float(minutes) * 60 + float(seconds_str)
                    else: # Unrecognized colon format
                        return np.nan
                except ValueError: # If parts are not convertible to float
                    return np.nan
            else:
                # Assume it's just seconds or some other non-standard format
                try:
                    return float(time_str)
                except ValueError:
                    return np.nan # Not a valid number
        
        # Convert Q3 times to seconds
        # Create the column first to avoid issues if all values are NaN
        result_df['qualifying_time_q3_seconds'] = np.nan 
        # Apply parse_time only to non-null values to speed up and avoid errors
        q3_valid_times_idx = result_df['qualifying_time_q3'].notna()
        if q3_valid_times_idx.any():
            result_df.loc[q3_valid_times_idx, 'qualifying_time_q3_seconds'] = \
                result_df.loc[q3_valid_times_idx, 'qualifying_time_q3'].apply(parse_time)
        
        # Calculate gap to pole, ensuring pole_time is valid
        if 'qualifying_time_q3_seconds' in result_df.columns and result_df['qualifying_time_q3_seconds'].notna().any():
            pole_time = result_df['qualifying_time_q3_seconds'].min() # min() ignores NaNs by default
            if pd.notna(pole_time):
                 result_df['gap_to_pole_seconds'] = result_df['qualifying_time_q3_seconds'] - pole_time
            else:
                 result_df['gap_to_pole_seconds'] = np.nan
        else:
            result_df['gap_to_pole_seconds'] = np.nan
    else:
        # Ensure columns exist even if no Q3 data
        if 'qualifying_time_q3_seconds' not in result_df.columns:
            result_df['qualifying_time_q3_seconds'] = np.nan
        if 'gap_to_pole_seconds' not in result_df.columns:
            result_df['gap_to_pole_seconds'] = np.nan
            
    # Ensure all expected qualifying columns are present, fill with NaN if not
    expected_q_cols = [
        'grid_position', 'qualifying_time_q1', 'qualifying_time_q2', 
        'qualifying_time_q3', 'qualifying_time_q3_seconds', 'gap_to_pole_seconds'
    ]
    for col in expected_q_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
            
    return result_df


def _add_historical_performance_features(
    features_df: pd.DataFrame, 
    race_results_df: pd.DataFrame, 
    recent_k_races: int
) -> pd.DataFrame:
    """
    Add features based on historical race performance.
    
    Args:
        features_df: Core features DataFrame.
        race_results_df: Historical race results DataFrame.
        recent_k_races: Number of recent races to consider.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Ensure race_results_df is sorted by date
    if 'race_date' in race_results_df.columns:
        race_results_df = race_results_df.sort_values('race_date')
    
    # Get the most recent date for each driver
    if 'race_date' in race_results_df.columns:
        most_recent_dates = race_results_df.groupby('driver_id')['race_date'].max()
        
        # For each driver in features_df, get their recent races
        for driver_id in result_df.index:
            if driver_id in most_recent_dates:
                recent_date = most_recent_dates[driver_id]
                driver_races = race_results_df[race_results_df['driver_id'] == driver_id]
                
                # Get recent K races for this driver
                recent_races = driver_races.sort_values('race_date', ascending=False).head(recent_k_races)
                
                if not recent_races.empty:
                    # Calculate average finish position
                    result_df.loc[driver_id, 'avg_finish_position_recent'] = recent_races['position'].mean()
                    
                    # Calculate finishing consistency (standard deviation of positions)
                    result_df.loc[driver_id, 'finish_position_consistency'] = recent_races['position'].std()
                    
                    # Calculate average points
                    result_df.loc[driver_id, 'avg_points_recent'] = recent_races['points'].mean()
                    
                    # Calculate DNF rate
                    dnf_statuses = ['Accident', 'Mechanical', 'Retired', 'Disqualified']
                    dnf_count = sum(recent_races['status'].isin(dnf_statuses))
                    result_df.loc[driver_id, 'dnf_rate_recent'] = dnf_count / len(recent_races)
                    
                    # Calculate average position change (grid to finish)
                    if 'grid_position' in recent_races.columns:
                        position_changes = recent_races['position'] - recent_races['grid_position']
                        result_df.loc[driver_id, 'avg_position_change'] = position_changes.mean()
    
    return result_df


def _add_standings_features(
    features_df: pd.DataFrame, 
    driver_standings_df: pd.DataFrame,
    constructor_standings_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add features based on championship standings.
    
    Args:
        features_df: Core features DataFrame.
        driver_standings_df: Driver standings DataFrame.
        constructor_standings_df: Constructor standings DataFrame.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Get most recent driver standings
    if 'round' in driver_standings_df.columns:
        most_recent_round = driver_standings_df['round'].max()
        current_standings = driver_standings_df[driver_standings_df['round'] == most_recent_round]
        
        # Merge with features on driver_id
        standings_features = current_standings[['driver_id', 'position', 'points', 'wins']]
        standings_features = standings_features.rename(columns={
            'position': 'championship_position',
            'points': 'championship_points',
            'wins': 'season_wins'
        })
        
        # Merge driver standings
        result_df = pd.merge(
            result_df, 
            standings_features, 
            left_index=True, 
            right_on='driver_id', 
            how='left'
        ).set_index('driver_id')
    
    # Add constructor standings if available
    if constructor_standings_df is not None and 'round' in constructor_standings_df.columns:
        most_recent_round = constructor_standings_df['round'].max()
        current_team_standings = constructor_standings_df[constructor_standings_df['round'] == most_recent_round]
        
        # We need to map driver_id to constructor_id to get the team standings
        # This is typically available in the driver_standings_df
        if 'driver_id' in driver_standings_df.columns and 'constructor_ids' in driver_standings_df.columns:
            # Get the latest constructor for each driver
            most_recent_driver_info = driver_standings_df.sort_values('round', ascending=False)
            driver_to_constructor = {}
            
            for _, row in most_recent_driver_info.drop_duplicates('driver_id').iterrows():
                driver_to_constructor[row['driver_id']] = row['constructor_ids'][0] if isinstance(row['constructor_ids'], list) else row['constructor_ids']
            
            # Add team standings data to each driver
            for driver_id in result_df.index:
                if driver_id in driver_to_constructor:
                    constructor_id = driver_to_constructor[driver_id]
                    team_row = current_team_standings[current_team_standings['constructor_id'] == constructor_id]
                    
                    if not team_row.empty:
                        result_df.loc[driver_id, 'team_championship_position'] = team_row['position'].values[0]
                        result_df.loc[driver_id, 'team_championship_points'] = team_row['points'].values[0]
                        result_df.loc[driver_id, 'team_season_wins'] = team_row['wins'].values[0]
    
    return result_df


def _add_track_specific_features(
    features_df: pd.DataFrame, 
    race_results_df: pd.DataFrame,
    circuit_id: Optional[str],
    recent_k_races: int
) -> pd.DataFrame:
    """
    Add features specific to the current track.
    
    Args:
        features_df: Core features DataFrame.
        race_results_df: Historical race results DataFrame.
        circuit_id: ID of the current circuit. Can be None.
        recent_k_races: Number of recent races to consider.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    track_specific_cols = [
        'avg_finish_position_track', 'best_finish_position_track',
        'experience_level_track', 'avg_points_track'
    ]

    print(f"\n--- Debug: _add_track_specific_features ---")
    print(f"Input circuit_id type: {type(circuit_id)}, value: {circuit_id}")
    
    if race_results_df.empty:
        print("Debug: race_results_df is empty.")
    else:
        print(f"Debug: race_results_df shape: {race_results_df.shape}")
        if 'circuit_id' in race_results_df.columns:
            print(f"Debug: race_results_df['circuit_id'] dtype: {race_results_df['circuit_id'].dtype}")
            print(f"Debug: race_results_df['circuit_id'].head():\n{race_results_df['circuit_id'].head()}")
            print(f"Debug: race_results_df['circuit_id'].isna().sum(): {race_results_df['circuit_id'].isna().sum()} NaN values")
        else:
            print("Debug: 'circuit_id' column MISSING in race_results_df.")
        # print(f"Debug: race_results_df.index: {race_results_df.index}")


    if circuit_id is None or race_results_df.empty or 'circuit_id' not in race_results_df.columns:
        print("Debug: Condition met for early exit (circuit_id is None, race_results_df empty, or 'circuit_id' missing).")
        for col in track_specific_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan
        print(f"--- End Debug: _add_track_specific_features (early exit) ---\n")
        return result_df

    print(f"Debug: Attempting to filter race_results_df by circuit_id: '{circuit_id}'")
    try:
        # Corrected line: comparing with race_results_df['circuit_id']
        track_results = race_results_df[race_results_df['circuit_id'] == circuit_id].copy()
        print(f"Debug: Successfully filtered track_results. Shape: {track_results.shape}")
    except Exception as e:
        print(f"Debug: ERROR during filtering: {e}")
        # In case of error, fill with NaNs and return to prevent further issues
        for col in track_specific_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan
        print(f"--- End Debug: _add_track_specific_features (error during filter) ---\n")
        raise # Re-raise the exception to see the full traceback
    
    if not track_results.empty:
        if 'driver_id' not in track_results.columns:
            print("Debug: 'driver_id' column MISSING in track_results.")
            for col in track_specific_cols:
                if col not in result_df.columns:
                    result_df[col] = np.nan
            print(f"--- End Debug: _add_track_specific_features (no driver_id in track_results) ---\n")
            return result_df

        if 'race_date' not in track_results.columns:
            print("Debug: 'race_date' column MISSING in track_results. Calculating overall track stats.")
            for driver_id_idx in result_df.index:
                driver_track_results = track_results[track_results['driver_id'] == driver_id_idx]
                if not driver_track_results.empty:
                    if 'position' in driver_track_results.columns:
                        result_df.loc[driver_id_idx, 'avg_finish_position_track'] = driver_track_results['position'].mean()
                        result_df.loc[driver_id_idx, 'best_finish_position_track'] = driver_track_results['position'].min()
                    if 'points' in driver_track_results.columns:
                        result_df.loc[driver_id_idx, 'avg_points_track'] = driver_track_results['points'].mean()
                    result_df.loc[driver_id_idx, 'experience_level_track'] = len(driver_track_results)
        else: # race_date is present
            print("Debug: 'race_date' column PRESENT in track_results. Calculating recent K stats.")
            for driver_id_idx in result_df.index:
                driver_track_results = track_results[track_results['driver_id'] == driver_id_idx]
                if not driver_track_results.empty:
                    if 'position' not in driver_track_results.columns or 'points' not in driver_track_results.columns:
                        print(f"Debug: Missing 'position' or 'points' for driver {driver_id_idx} in track_results. Skipping.")
                        continue
                    recent_driver_track_results = driver_track_results.sort_values('race_date', ascending=False).head(recent_k_races)
                    result_df.loc[driver_id_idx, 'avg_finish_position_track'] = recent_driver_track_results['position'].mean()
                    result_df.loc[driver_id_idx, 'best_finish_position_track'] = recent_driver_track_results['position'].min()
                    result_df.loc[driver_id_idx, 'experience_level_track'] = len(recent_driver_track_results)
                    result_df.loc[driver_id_idx, 'avg_points_track'] = recent_driver_track_results['points'].mean()
            
        for col in track_specific_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(np.nan)
            else: # Should not happen if initialized correctly
                result_df[col] = np.nan
    else:
        print("Debug: track_results is empty after filtering.")
        for col in track_specific_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan
            else:
                result_df[col] = result_df[col].fillna(np.nan)

    for col in track_specific_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    print(f"Debug: result_df columns before returning: {result_df.columns.tolist()}")
    print(f"--- End Debug: _add_track_specific_features ---\n")
    return result_df


def _add_track_type_features(
    features_df: pd.DataFrame, 
    circuit_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add features based on track type.
    
    Args:
        features_df: Core features DataFrame.
        circuit_df: Circuit information DataFrame.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Ensure circuit_df is a DataFrame with a circuit_id column
    if isinstance(circuit_df, pd.DataFrame) and 'circuit_id' in circuit_df.columns:
        circuit_id = circuit_df['circuit_id'].iloc[0] if not circuit_df.empty else None
    else:
        # If circuit_df is a dictionary or other format, try to extract circuit_id
        circuit_id = circuit_df.get('circuit_id', None)
    
    if circuit_id is not None:
        # Use feature_utils function to add track type features
        # Create a DataFrame with circuit_id for each driver
        circuit_id_df = pd.DataFrame({'circuit_id': circuit_id}, index=result_df.index)
        
        # Add track type features - preserve existing columns
        track_type_features = create_track_type_features(circuit_id_df, 'circuit_id')
        # Merge with existing features instead of replacing
        result_df = pd.concat([result_df, track_type_features], axis=1)
    
    return result_df


def _add_weather_features(
    features_df: pd.DataFrame, 
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add weather-related features.
    
    Args:
        features_df: Core features DataFrame.
        weather_df: Weather forecast DataFrame.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Extract relevant weather metrics
    weather_metrics = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'precipitation_probability', 'precipitation_amount',
        'weather_condition', 'track_temperature'
    ]
    
    available_metrics = [metric for metric in weather_metrics if metric in weather_df.columns]
    
    if available_metrics:
        # Get the most recent weather forecast
        if 'forecast_time' in weather_df.columns:
            weather_df = weather_df.sort_values('forecast_time', ascending=False)
        
        most_recent_weather = weather_df.iloc[0]
        
        # Add each available metric to all drivers
        for metric in available_metrics:
            result_df[f'weather_{metric}'] = most_recent_weather[metric]
        
        # Create additional features for rain conditions
        if 'weather_condition' in available_metrics:
            rain_conditions = ['rain', 'showers', 'drizzle', 'thunderstorm']
            result_df['is_wet_race'] = most_recent_weather['weather_condition'].lower() in rain_conditions
        
        if 'precipitation_probability' in available_metrics:
            result_df['high_rain_probability'] = most_recent_weather['precipitation_probability'] > 50
    
    return result_df


def _add_domain_knowledge_features(
    features_df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Add features based on domain knowledge.
    
    Args:
        features_df: Core features DataFrame.
        config: Configuration dictionary.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Extract domain knowledge config
    domain_config = config.get('domain_knowledge', {})
    
    # Add flag for drivers unlikely to win points
    unlikely_points_drivers = domain_config.get('unlikely_to_win_points_drivers', [])
    if unlikely_points_drivers:
        result_df['is_unlikely_to_win_points'] = result_df.index.isin(unlikely_points_drivers)
    
    # Add team favoritism indicator
    team_favoritism = domain_config.get('team_favoritism_mapping', {})
    if team_favoritism:
        result_df['is_favored_by_team'] = result_df.index.isin(team_favoritism.values())
    
    return result_df 