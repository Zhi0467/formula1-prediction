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
            - 'race_results': past race results (general)
            - 'track_results': past race results (circuit-specific)
            - 'qualifying_results': past qualifying results
            - 'driver_standings': past driver standings
            - 'constructor_standings': past constructor standings
        config: Configuration dictionary.
    
    Returns:
        DataFrame with core features for all drivers.
    """
    print("\n--- Debug: compute_core_features (Start) ---")
    print(f"Available keys in race_data: {list(race_data.keys())}")
    print(f"Available keys in historical_data: {list(historical_data.keys())}")

    qualifying_df = race_data.get('qualifying', pd.DataFrame())
    drivers_info_df = race_data.get('drivers', pd.DataFrame())

    #print(f"Debug: qualifying_df empty? {qualifying_df.empty}")
    if not qualifying_df.empty:
        print(f"Debug: qualifying_df columns: {qualifying_df.columns.tolist()}")
        if 'driver_id' in qualifying_df.columns:
            print(f"Debug: qualifying_df['driver_id'] raw unique: {qualifying_df['driver_id'].unique()}")
            # Filter out None or NaN from driver IDs used for index
            valid_q_driver_ids = [did for did in qualifying_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            # print(f"Debug: qualifying_df['driver_id'] valid unique for index: {valid_q_driver_ids}")
        else:
            print("Debug: 'driver_id' MISSING in qualifying_df")

    # print(f"Debug: drivers_info_df (from standings) empty? {drivers_info_df.empty}")
    if not drivers_info_df.empty:
        # print(f"Debug: drivers_info_df columns: {drivers_info_df.columns.tolist()}")
        if 'driver_id' in drivers_info_df.columns:
            print(f"Debug: drivers_info_df['driver_id'] raw unique: {drivers_info_df['driver_id'].unique()}")
            valid_d_driver_ids = [did for did in drivers_info_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            #print(f"Debug: drivers_info_df['driver_id'] valid unique for index: {valid_d_driver_ids}")
        else:
            print("Debug: 'driver_id' MISSING in drivers_info_df")
    
    feature_config = config.get('feature_engineering', {}).get('core_features', {})
    recent_k_races = feature_config.get('recent_k_races', 5)
    
    # Initialize features_df to None, to clearly track if it gets created
    initialized_features_df = None

    # Strategy 1: Try to initialize from qualifying_df
    if not qualifying_df.empty and 'driver_id' in qualifying_df.columns:
        #print("Debug: Attempting to initialize features_df index from qualifying_df")
        unique_q_driver_ids = [did for did in qualifying_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
        #print(f"Debug: Valid unique driver_ids from qualifying_df: {unique_q_driver_ids} (Count: {len(unique_q_driver_ids)})")
        if len(unique_q_driver_ids) > 0:
            try:
                temp_df = pd.DataFrame(index=unique_q_driver_ids)
                temp_df.index.name = 'driver_id'
                initialized_features_df = temp_df # Assign here
                #print(f"Debug: temp_df from qualifying_df. Shape: {temp_df.shape}, Index: {temp_df.index.tolist()}, Is empty: {temp_df.empty}")
            except Exception as e:
                print(f"Debug: ERROR creating DataFrame from qualifying_df driver_ids: {e}")
        else:
            print("Debug: No valid unique driver_ids in qualifying_df to create index.")
            
    # Strategy 2: If not successful, try from drivers_info_df
    if initialized_features_df is None or len(initialized_features_df.index) == 0: # Check if previous attempt failed or resulted in empty
        if 'drivers' in race_data and not drivers_info_df.empty and 'driver_id' in drivers_info_df.columns:
            #print("Debug: Initializing from qualifying failed or resulted in empty. Attempting from drivers_info_df.")
            unique_d_driver_ids = [did for did in drivers_info_df['driver_id'].unique() if pd.notna(did) and str(did).strip() != '']
            #print(f"Debug: Valid unique driver_ids from drivers_info_df: {unique_d_driver_ids} (Count: {len(unique_d_driver_ids)})")
            if len(unique_d_driver_ids) > 0:
                try:
                    temp_df = pd.DataFrame(index=unique_d_driver_ids)
                    temp_df.index.name = 'driver_id'
                    initialized_features_df = temp_df # Assign here
                    #print(f"Debug: temp_df from drivers_info_df. Shape: {temp_df.shape}, Index: {temp_df.index.tolist()}, Is empty: {temp_df.empty}")
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
    # print(f"Debug: features_df successfully initialized. Shape: {features_df.shape}, Index: {features_df.index.tolist()}, Is empty: {features_df.empty}")

    # Get feature inclusion flags from config
    include_qualifying = feature_config.get('include_qualifying_features', True)
    include_historical = feature_config.get('include_historical_features', True)
    include_standings = feature_config.get('include_standings_features', True)
    include_track_specific = feature_config.get('include_track_specific_features', True)
    include_track_type = feature_config.get('include_track_type_features', False)
    include_weather = feature_config.get('include_weather_data', False) # Kept original weather flag name for compatibility
    include_domain_knowledge = feature_config.get('include_domain_knowledge_features', False)

    if include_qualifying and 'qualifying' in race_data:
        features_df = _add_qualifying_features(features_df, race_data['qualifying'])
    
    if include_historical and 'race_results' in historical_data and not historical_data['race_results'].empty:
        features_df = _add_historical_performance_features(
            features_df, 
            historical_data['race_results'], 
            recent_k_races
        )
    
    if include_standings and 'driver_standings' in historical_data and not historical_data['driver_standings'].empty:
        current_constructor_standings = historical_data.get('constructor_standings', pd.DataFrame())
        features_df = _add_standings_features(
            features_df, 
            historical_data['driver_standings'],
            current_constructor_standings,
            race_data  # Pass the race_data to _add_standings_features
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

    if include_track_specific:
        if 'track_results' in historical_data and not historical_data['track_results'].empty:
            features_df = _add_track_specific_features(
                features_df,
                historical_data['track_results'],
                None,  # No need to filter by circuit_id as data is already filtered
                recent_k_races
            )
        elif current_circuit_id is not None and 'race_results' in historical_data and not historical_data['race_results'].empty:
             # Fallback to filtering general historical data if circuit-specific data wasn't fetched
            features_df = _add_track_specific_features(
                features_df,
                historical_data['race_results'],
                current_circuit_id,
                recent_k_races
            )
        else:
            print("Warning: No data available to add track-specific features.")
            # Ensure columns are present with NaNs if no data was available
            placeholder_cols = [
                'avg_finish_position_track', 'best_finish_position_track',
                'experience_level_track', 'avg_points_track'
            ]
            for col in placeholder_cols:
                if col not in features_df.columns:
                     features_df[col] = np.nan


    if include_track_type and 'circuit' in race_data and not race_data['circuit'].empty:
        features_df = _add_track_type_features(features_df, race_data['circuit'])
    
    if include_weather and 'weather' in race_data and not race_data['weather'].empty:
        features_df = _add_weather_features(features_df, race_data['weather'])
    
    domain_knowledge_config = config.get('domain_knowledge', {})
    if include_domain_knowledge and domain_knowledge_config: # Check if domain_knowledge config exists and is not empty
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
        
        
        # Calculate gap to pole, ensuring pole_time is valid
        if 'qualifying_time_q3' in result_df.columns and result_df['qualifying_time_q3'].notna().any():
            pole_time = result_df['qualifying_time_q3'].min() # min() ignores NaNs by default
            if pd.notna(pole_time):
                 result_df['gap_to_pole_seconds'] = result_df['qualifying_time_q3'] - pole_time
            else:
                 result_df['gap_to_pole_seconds'] = np.nan
        else:
            result_df['gap_to_pole_seconds'] = np.nan
    else:
        # Ensure columns exist even if no Q3 data
        if 'qualifying_time_q3' not in result_df.columns:
            result_df['qualifying_time_q3'] = np.nan
        if 'gap_to_pole_seconds' not in result_df.columns:
            result_df['gap_to_pole_seconds'] = np.nan
            
    # Ensure all expected qualifying columns are present, fill with NaN if not
    expected_q_cols = [
        'grid_position', 'qualifying_time_q1', 'qualifying_time_q2', 
        'qualifying_time_q3', 'gap_to_pole_seconds'
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
    Add features based on historical race performance with improved date handling.
    """
    result_df = features_df.copy()
    
    # Debug information
    print("\n--- DEBUG: Adding historical performance features ---")
    print(f"Input features_df has {len(result_df)} drivers")
    print(f"Historical race_results_df has {len(race_results_df)} rows")
    
    # Create columns with NaN values for all drivers first
    history_cols = [
        'avg_finish_position_recent', 'finish_position_consistency',
        'avg_points_recent', 'dnf_rate_recent', 'avg_position_change'
    ]
    for col in history_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # If no historical data, return early
    if race_results_df.empty:
        print("No historical race results data available")
        return result_df
    
    # Ensure race_date is in datetime format for reliable sorting
    if 'race_date' in race_results_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(race_results_df['race_date']):
            try:
                race_results_df['race_date'] = pd.to_datetime(race_results_df['race_date'])
                print("Converted race_date to datetime format")
            except Exception as e:
                print(f"Error converting race_date to datetime: {e}")
                # Continue with string-based dates as fallback
    
    # Debug the historical race data
    print("\nAvailable race dates (most recent first):")
    if 'race_date' in race_results_df.columns:
        date_sorted = sorted(race_results_df['race_date'].unique(), reverse=True)
        print(date_sorted[:10])  # Show the 10 most recent races
    
    # Process for each driver
    for driver_id in result_df.index:
        driver_races = race_results_df[race_results_df['driver_id'] == driver_id]
        
        # Skip if no historical data found
        if driver_races.empty:
            print(f"No historical data for driver: {driver_id}")
            continue
        
        # Get recent K races using reliable date sorting
        if 'race_date' in driver_races.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(driver_races['race_date']):
                driver_races['race_date'] = pd.to_datetime(driver_races['race_date'])
                
            # Sort by date descending (newest first)
            recent_races = driver_races.sort_values('race_date', ascending=False).head(recent_k_races)
        else:
            # Fallback if no date column
            print(f"WARNING: No race_date column for {driver_id}, using last {recent_k_races} rows")
            recent_races = driver_races.iloc[-recent_k_races:]
        
        # Debug for key drivers
        if driver_id in ['max_verstappen', 'leclerc', 'perez']:
            print(f"\nDetailed analysis for {driver_id}:")
            print(f"Total historical races: {len(driver_races)}")
            print(f"Recent {recent_k_races} races:")
            if 'race_date' in recent_races.columns:
                display_df = recent_races[['race_name', 'race_date', 'position', 'points']]
                print(display_df.sort_values('race_date', ascending=False))
            
            if 'position' in recent_races.columns:
                positions = recent_races['position'].tolist()
                avg_pos = sum(positions) / len(positions) if positions else float('nan')
                print(f"Position values: {positions}")
                print(f"Average position: {avg_pos}")
        
        # Calculate metrics for each driver based on their recent races
        if not recent_races.empty:
            # Calculate average finish position
            if 'position' in recent_races.columns:
                if not recent_races['position'].isnull().all():  # Make sure we have valid positions
                    avg_pos = recent_races['position'].mean()
                    result_df.loc[driver_id, 'avg_finish_position_recent'] = avg_pos
                    if driver_id in ['max_verstappen', 'leclerc', 'hamilton']:
                        print(f"{driver_id} avg_position = {avg_pos} from values {recent_races['position'].tolist()}")
            
            # Calculate finishing consistency (standard deviation of positions)
            if 'position' in recent_races.columns and len(recent_races) > 1:
                result_df.loc[driver_id, 'finish_position_consistency'] = recent_races['position'].std()
            
            # Calculate average points
            if 'points' in recent_races.columns:
                result_df.loc[driver_id, 'avg_points_recent'] = recent_races['points'].mean()
            
            # Calculate DNF rate
            if 'status' in recent_races.columns:
                dnf_statuses = ['Accident', 'Mechanical', 'Retired', 'Disqualified']
                dnf_count = sum(recent_races['status'].isin(dnf_statuses))
                result_df.loc[driver_id, 'dnf_rate_recent'] = dnf_count / len(recent_races)
            
            # Calculate average position change (grid to finish)
            if 'grid_position' in recent_races.columns and 'position' in recent_races.columns:
                # Check for valid grid positions (not NaN)
                valid_positions = recent_races.dropna(subset=['grid_position', 'position'])
                if not valid_positions.empty:
                    position_changes = valid_positions['grid_position'] - valid_positions['position']
                    result_df.loc[driver_id, 'avg_position_change'] = position_changes.mean()
    
    # For drivers without history, set reasonable defaults for a rookie
    for col in history_cols:
        is_empty = result_df[col].isna()
        if is_empty.any():
            print(f"Setting default values for {is_empty.sum()} drivers missing {col}")
            # Sensible defaults for rookies as defined earlier...
    
    return result_df


def _add_standings_features(
    features_df: pd.DataFrame, 
    driver_standings_df: pd.DataFrame,
    constructor_standings_df: Optional[pd.DataFrame] = None,
    race_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Add features based on championship standings.
    
    Args:
        features_df: Core features DataFrame.
        driver_standings_df: Driver standings DataFrame.
        constructor_standings_df: Constructor standings DataFrame.
        race_data: Dictionary with data for the current race, used for rookie drivers.
        
    Returns:
        Updated features DataFrame.
    """
    result_df = features_df.copy()
    
    # Debug information
    print("\n--- DEBUG: Adding standings features ---")
    
    # Get most recent driver standings
    if 'round' in driver_standings_df.columns:
        most_recent_round = driver_standings_df['round'].max()
        current_standings = driver_standings_df[driver_standings_df['round'] == most_recent_round]
        
        # Debug the current standings
        print(f"Driver standings for round {most_recent_round}:")
        if not current_standings.empty and 'constructor_ids' in current_standings.columns:
            print(current_standings[['driver_id', 'position', 'points', 'wins', 'constructor_ids']].head())
        else:
            print(current_standings[['driver_id', 'position', 'points', 'wins']].head())
        
        # Select columns for merging, handling columns that might not exist
        columns_to_select = ['driver_id', 'position', 'points', 'wins']
        if 'constructor_ids' in current_standings.columns:
            columns_to_select.append('constructor_ids')
            
        standings_features = current_standings[columns_to_select]
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
    if constructor_standings_df is not None and not constructor_standings_df.empty and 'round' in constructor_standings_df.columns:
        most_recent_round = constructor_standings_df['round'].max()
        current_team_standings = constructor_standings_df[constructor_standings_df['round'] == most_recent_round]
        
        # Build a mapping of all drivers to their teams
        driver_to_constructor = {}
        
        # First from the standings data
        if 'constructor_ids' in result_df.columns:
            for driver_id, constructor_ids in result_df['constructor_ids'].items():
                if pd.notna(constructor_ids):
                    if isinstance(constructor_ids, list) and len(constructor_ids) > 0:
                        driver_to_constructor[driver_id] = constructor_ids[0]
                    elif isinstance(constructor_ids, str):
                        driver_to_constructor[driver_id] = constructor_ids
        
        # For rookies not in standings, use qualifying data to get their team
        if race_data is not None and 'qualifying' in race_data and not race_data['qualifying'].empty:
            quali_df = race_data['qualifying']
            if 'team_id' in quali_df.columns and 'driver_id' in quali_df.columns:
                for _, row in quali_df.iterrows():
                    if row['driver_id'] not in driver_to_constructor and pd.notna(row['team_id']):
                        driver_to_constructor[row['driver_id']] = row['team_id']
        
        # Add team standings data to each driver
        for driver_id in result_df.index:
            if driver_id in driver_to_constructor:
                constructor_id = driver_to_constructor[driver_id]
                team_row = current_team_standings[current_team_standings['constructor_id'] == constructor_id]
                
                if not team_row.empty:
                    result_df.loc[driver_id, 'team_championship_position'] = team_row['position'].values[0]
                    result_df.loc[driver_id, 'team_championship_points'] = team_row['points'].values[0]
                    result_df.loc[driver_id, 'team_season_wins'] = team_row['wins'].values[0]
            else:
                print(f"No constructor mapping found for {driver_id}")
    
    return result_df


def _add_track_specific_features(
    features_df: pd.DataFrame, 
    race_results_df: pd.DataFrame,
    circuit_id: Optional[str],
    recent_k_races: int
) -> pd.DataFrame:
    """
    Add features specific to the current track.
    """
    result_df = features_df.copy()
    
    # Initialize track-specific columns with NaN for all drivers
    track_specific_cols = [
        'avg_finish_position_track', 'best_finish_position_track',
        'experience_level_track', 'avg_points_track'
    ]
    
    for col in track_specific_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # Debug information
    print(f"\n--- DEBUG: Adding track-specific features ---")
    
    # Early exit if no race results
    if race_results_df.empty:
        print("Early exit: no race results data")
        # Set sensible default values for all drivers when no track data available
        for driver_id in result_df.index:
            result_df.loc[driver_id, 'experience_level_track'] = 0
            result_df.loc[driver_id, 'avg_finish_position_track'] = 10  # Mid-field assumption
            result_df.loc[driver_id, 'best_finish_position_track'] = 10
            result_df.loc[driver_id, 'avg_points_track'] = 0            # Conservative
        return result_df
    
    # If circuit_id is provided, filter results for this track
    # Otherwise assume the data is already filtered (from get_circuit_results)
    track_results = race_results_df
    if circuit_id is not None and 'circuit_id' in race_results_df.columns:
        track_results = race_results_df[race_results_df['circuit_id'] == circuit_id].copy()
        print(f"Filtered {len(track_results)} historical results for circuit {circuit_id}")
    else:
        print(f"Using pre-filtered track data with {len(track_results)} results")
    
    # Track driver experience
    experienced_drivers = set()
    
    if not track_results.empty and 'driver_id' in track_results.columns:
        # Process each driver's track history
        for driver_id in result_df.index:
            driver_track_results = track_results[track_results['driver_id'] == driver_id]
            
            # Skip if no track history for this driver
            if driver_track_results.empty:
                continue
                
            experienced_drivers.add(driver_id)
            
            # Sort by date if available
            if 'race_date' in driver_track_results.columns:
                driver_track_results = driver_track_results.sort_values('race_date', ascending=False)
                driver_track_results = driver_track_results.head(recent_k_races)
            
            # Calculate track-specific stats
            if 'position' in driver_track_results.columns:
                result_df.loc[driver_id, 'avg_finish_position_track'] = driver_track_results['position'].mean()
                result_df.loc[driver_id, 'best_finish_position_track'] = driver_track_results['position'].min()
            
            result_df.loc[driver_id, 'experience_level_track'] = len(driver_track_results)
            
            if 'points' in driver_track_results.columns:
                result_df.loc[driver_id, 'avg_points_track'] = driver_track_results['points'].mean()
    
    # Set sensible default values for rookie drivers with no track experience
    rookie_drivers = set(result_df.index) - experienced_drivers
    if rookie_drivers:
        print(f"Setting track defaults for rookies: {rookie_drivers}")
        for driver_id in rookie_drivers:
            result_df.loc[driver_id, 'experience_level_track'] = 0
            result_df.loc[driver_id, 'avg_finish_position_track'] = 15  # Conservative back of midfield
            result_df.loc[driver_id, 'best_finish_position_track'] = 15
            result_df.loc[driver_id, 'avg_points_track'] = 0
    
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