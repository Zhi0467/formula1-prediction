"""F1 data preparation and preprocessing pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from ..data_ingest.jolpica_client import JolpicaClient
from ..features.common_preprocessing import preprocess_data
from ..features.core_features import compute_core_features


class F1DataPreprocessor:
    """
    Comprehensive data preparation pipeline for F1 prediction.
    
    This class handles:
    1. Data fetching from the Jolpica API
    2. Data cleaning and standardization
    3. Feature engineering
    4. Preparing data for model input
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with settings for data fetching and preprocessing.
        """
        self.config = config
        
        # Initialize the API client
        client_config = config.get('data_ingest', {}).get('client', {})
        self.client = JolpicaClient(client_config)
        
        # Extract configuration parameters
        self.preprocessing_config = config.get('feature_engineering', {}).get('preprocessing', {})
        self.feature_config = config.get('feature_engineering', {})
        self.lookback_races = self.feature_config.get('historical_lookback_races', 5)
        self.lookback_years = self.feature_config.get('historical_lookback_years', 2)
        print(f"lookback years: {self.lookback_years}")
        print(f"lookback races: {self.lookback_races}")
    
    def prepare_data_for_race(self, 
                             season: Union[int, str], 
                             race: Union[int, str],
                             fetch_weather: bool = False) -> pd.DataFrame:
        """
        Prepare all data for a specific race.
        
        Args:
            season: F1 season (year)
            race: Race number
            fetch_weather: Whether to fetch weather data (requires external weather API)
            
        Returns:
            Feature DataFrame ready for model input
        """
        # Step 1: Collect all required data
        race_data = self._fetch_race_data(season, race, fetch_weather)
        historical_data = self._fetch_historical_data(season, race)
        
        # Step 2: Preprocess each dataset with common preprocessing
        race_data = self._preprocess_race_data(race_data)
        historical_data = self._preprocess_historical_data(historical_data)
        
        # Step 3: Generate core features
        features_df = compute_core_features(race_data, historical_data, self.config)
        
        # Step 4: Final cleanup to handle any remaining issues
        features_df = self._final_data_cleanup(features_df)
        labels_df = self._get_race_labels(season=season, race=race)
        
        return features_df, labels_df
    
    def _fetch_race_data(self, 
                        season: Union[int, str], 
                        race: Union[int, str],
                        fetch_weather: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data relevant to the current race.
        
        Args:
            season: F1 season (year)
            race: Race number
            fetch_weather: Whether to fetch weather data
            
        Returns:
            Dictionary of DataFrames with race data
        """
        race_data = {}
        
        # Fetch qualifying results
        qualifying_df = self.client.get_qualifying_results(season, race)
        if not qualifying_df.empty:
            race_data['qualifying'] = qualifying_df
        
        # Fetch circuit information
        circuits_df = self.client.get_circuits(season)
        if not circuits_df.empty:
            # Get specific circuit for this race
            if 'qualifying' in race_data and not qualifying_df.empty:
                circuit_name = qualifying_df['circuit_name'].iloc[0]
                circuit_df = circuits_df[circuits_df['circuit_name'] == circuit_name]
                if not circuit_df.empty:
                    race_data['circuit'] = circuit_df
        
        # Fetch driver information from standings
        driver_standings_df = self.client.get_driver_standings(season, race)
        if not driver_standings_df.empty:
            race_data['drivers'] = driver_standings_df
        
        # Fetch team information from standings
        team_standings_df = self.client.get_constructor_standings(season, race)
        if not team_standings_df.empty:
            race_data['teams'] = team_standings_df
        
        # Fetch weather data if requested (placeholder - would need external API)
        if fetch_weather:
            # This would be implemented with a weather API client
            # For now, create a simple placeholder
            weather_data = self._create_placeholder_weather()
            race_data['weather'] = weather_data
        
        return race_data
    
    def _fetch_historical_data(self, 
                              season: Union[int, str], 
                              race: Union[int, str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for feature engineering, optimized for API rate limits.
        
        Args:
            season: Current F1 season (year)
            race: Current race number
            
        Returns:
            Dictionary of DataFrames with historical data
        """
        historical_data = {}
        
        # Determine years to fetch based on lookback configuration
        current_year = int(season)
        lookback_years = list(range(current_year - self.lookback_years, current_year + 1))
        max_races_per_year = self.lookback_races  # Limit number of races to fetch per year to manage API calls
        
        print(f"\n--- DEBUG: Fetching historical data (optimized) ---")
        print(f"Current race: {season} Round {race}")
        print(f"Lookback years: {lookback_years}, max {max_races_per_year} races per year")
        
        # Collect race results from previous seasons and races
        all_race_results = []
        
        # Process years in reverse chronological order (most recent first)
        fetched_race_count = 0
        for year in reversed(lookback_years):
            if year == current_year:
                # For current year, get individual races before the current one
                if race == 'last':
                    race_df = self.client.get_race_results(year, "last")
                    race = int(race_df['round'].iloc[0])
                if int(race) > 1:
                    # Fetch races in reverse order (most recent first)
                    races_to_fetch = range(int(race)-1, max(0, int(race)-max_races_per_year-1), -1)
                    for r in races_to_fetch:
                        print(f"Fetching race {r} from {year} (current year)")
                        race_result_df = self.client.get_race_results(year, r)
                        fetched_race_count += 1
                        if not race_result_df.empty:
                            # Convert race_date to datetime
                            if 'race_date' in race_result_df.columns:
                                race_result_df['race_date'] = pd.to_datetime(race_result_df['race_date'])
                            all_race_results.append(race_result_df)
                            print(f"  Found {len(race_result_df)} results for {year} race {r}")
            elif year < current_year and fetched_race_count < self.lookback_races:
                print(f"Fetching most recent races from {year}")
                last_race_df = self.client.get_race_results(year, "last")
                
                if not last_race_df.empty:
                    # Get the last round number
                    if 'round' in last_race_df.columns:
                        # Check if any valid round values exist
                        if not last_race_df['round'].isnull().all():
                            last_round = int(last_race_df['round'].iloc[0])
                            print(f"  Last round for {year} was {last_round}")
                            
                            # Now fetch specific races, starting from the last round
                            races_to_fetch = range(last_round, max(0, last_round-max_races_per_year), -1)
                            for r in races_to_fetch:
                                print(f"  Fetching race {r} from {year}")
                                race_result_df = self.client.get_race_results(year, r)
                                if not race_result_df.empty:
                                    if 'race_date' in race_result_df.columns:
                                        race_result_df['race_date'] = pd.to_datetime(race_result_df['race_date'])
                                    all_race_results.append(race_result_df)
                                    print(f"    Found {len(race_result_df)} results")
                        else:
                            print(f"  No valid round numbers found in last_race_df for {year}")
                    else:
                        print(f"  'round' column not found in last_race_df for {year}")
                else:
                    # If we can't determine the last race, try fetching a few specific late-season races
                    # These are typically the last races in most F1 seasons
                    print("The Last round from the previous year is not fetched")
                    potential_last_races = [22, 21, 20, 19, 18]
                    races_fetched = 0
                    
                    for r in potential_last_races:
                        if races_fetched >= max_races_per_year:
                            break
                            
                        print(f"  Trying race {r} from {year}")
                        race_result_df = self.client.get_race_results(year, r)
                        if not race_result_df.empty:
                            if 'race_date' in race_result_df.columns:
                                race_result_df['race_date'] = pd.to_datetime(race_result_df['race_date'])
                            all_race_results.append(race_result_df)
                            races_fetched += 1
                            print(f"    Found {len(race_result_df)} results")
        
        # Combine all race results
        if all_race_results:
            combined_race_results = pd.concat(all_race_results, ignore_index=True)
            print(f"Combined race results: {len(combined_race_results)} rows")
            
            # Ensure correct datetime format and sort by date
            if 'race_date' in combined_race_results.columns:
                if not pd.api.types.is_datetime64_any_dtype(combined_race_results['race_date']):
                    combined_race_results['race_date'] = pd.to_datetime(combined_race_results['race_date'])
                
                date_range = f"{combined_race_results['race_date'].min()} to {combined_race_results['race_date'].max()}"
                print(f"Date range: {date_range}")
                
                # Sort by date ascending (oldest to newest)
                combined_race_results = combined_race_results.sort_values('race_date')
                
                # Print unique race dates for verification
                unique_dates = combined_race_results['race_date'].unique()
                print(f"Unique race dates ({len(unique_dates)}): {sorted(unique_dates)}")
            
            historical_data['race_results'] = combined_race_results
        
        # Fetch circuit-specific historical data for track performance features
        # First we need to get the current circuit_id from race information
        current_circuit_id = None
        
        # Try to get circuit_id from qualifying data first
        qualifying_df = self.client.get_qualifying_results(season, race)
        if not qualifying_df.empty and 'circuit_name' in qualifying_df.columns:
            circuit_name = qualifying_df['circuit_name'].iloc[0]
            
            # Get circuit ID from circuit information
            circuits_df = self.client.get_circuits(season)
            if not circuits_df.empty:
                circuit_df = circuits_df[circuits_df['circuit_name'] == circuit_name]
                if not circuit_df.empty and 'circuit_id' in circuit_df.columns:
                    current_circuit_id = circuit_df['circuit_id'].iloc[0]
                    print(f"Found circuit_id for current race: {current_circuit_id}")
        
        # If we found a circuit_id, fetch historical results for this specific track
        if current_circuit_id:
            print(f"Fetching historical results for circuit: {current_circuit_id}")
            all_circuit_results = []
            for year in reversed(lookback_years):
                track_results_df = self.client.get_circuit_results(season= year - 1, circuit_id = current_circuit_id)
                all_circuit_results.append(track_results_df)
                if not track_results_df.empty:
                    print(f"Found {len(track_results_df)} historical results for {current_circuit_id} at year {year - 1}")
            all_circuit_results_pd = pd.concat(all_circuit_results, ignore_index=True)
            historical_data['track_results'] = all_circuit_results_pd
        
        return historical_data
    
    def _preprocess_race_data(self, race_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply common preprocessing to race data.
        
        Args:
            race_data: Dictionary of raw race data DataFrames
            
        Returns:
            Dictionary of preprocessed race data DataFrames
        """
        preprocessed_data = {}
        
        for key, df in race_data.items():
            if not df.empty:
                # Convert qualifying times to seconds first if it's qualifying data
                if key == 'qualifying':
                    # Convert qualifying time strings to seconds
                    for col in ['Q1', 'Q2', 'Q3']:
                        if col in df.columns:
                            df[col] = df[col].apply(self._convert_time_to_seconds)
                
                # Apply standard preprocessing
                preprocessed_df = preprocess_data(df, self.config)
                preprocessed_data[key] = preprocessed_df
        
        return preprocessed_data
    
    def _convert_time_to_seconds(self, time_str):
        """
        Convert F1 time format (1:30.123) to seconds.
        
        Args:
            time_str: Time string to convert
            
        Returns:
            Seconds as float or NaN if invalid
        """
        if pd.isna(time_str) or not isinstance(time_str, str) or time_str.strip() == '':
            return np.nan
        
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # MM:SS.sss
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
                else:
                    return np.nan
            else:
                return float(time_str)
        except (ValueError, TypeError):
            return np.nan
    
    def _preprocess_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply common preprocessing to historical data.
        
        Args:
            historical_data: Dictionary of raw historical data DataFrames
            
        Returns:
            Dictionary of preprocessed historical data DataFrames
        """
        preprocessed_data = {}
        
        for key, df in historical_data.items():
            if not df.empty:
                # Apply standard preprocessing
                preprocessed_df = preprocess_data(df, self.config)
                preprocessed_data[key] = preprocessed_df
        
        return preprocessed_data
    
    def _prepare_model_input(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the final feature DataFrame for model input.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Model-ready DataFrame
        """
        # Create a copy to avoid modifying the original
        model_input_df = features_df.copy()
        
        # Handle categorical variables
        cat_encoding = self.preprocessing_config.get('categorical_encoding', 'one-hot')
        
        # Identify categorical columns
        categorical_cols = model_input_df.select_dtypes(include=['object', 'category']).columns
        
        if cat_encoding == 'one-hot':
            # Apply one-hot encoding
            model_input_df = pd.get_dummies(model_input_df, columns=categorical_cols, drop_first=True)
        elif cat_encoding == 'label':
            # Apply label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                model_input_df[col] = le.fit_transform(model_input_df[col])
        
        # Handle missing values
        model_input_df = model_input_df.fillna(0)
        
        # Ensure all values are numeric
        for col in model_input_df.columns:
            if not pd.api.types.is_numeric_dtype(model_input_df[col]):
                model_input_df[col] = pd.to_numeric(model_input_df[col], errors='coerce').fillna(0)
        
        # Feature scaling if configured
        if self.preprocessing_config.get('scale_features', False):
            numeric_cols = model_input_df.select_dtypes(include=['int64', 'float64']).columns
            
            # Choose scaling method
            scaling_method = self.preprocessing_config.get('scaling_method', 'standard')
            
            if scaling_method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                model_input_df[numeric_cols] = scaler.fit_transform(model_input_df[numeric_cols])
            elif scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                model_input_df[numeric_cols] = scaler.fit_transform(model_input_df[numeric_cols])
            elif scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                model_input_df[numeric_cols] = scaler.fit_transform(model_input_df[numeric_cols])
        
        return model_input_df
    
    def _create_placeholder_weather(self) -> pd.DataFrame:
        """
        Create placeholder weather data for demonstration purposes.
        
        Returns:
            DataFrame with placeholder weather data
        """
        weather_data = {
            'forecast_time': datetime.now(),
            'temperature': 25.0,  # Celsius
            'humidity': 60.0,  # Percentage
            'wind_speed': 10.0,  # km/h
            'wind_direction': 180.0,  # degrees
            'precipitation_probability': 20.0,  # Percentage
            'precipitation_amount': 0.0,  # mm
            'weather_condition': 'Partly Cloudy',
            'track_temperature': 35.0  # Celsius
        }
        
        return pd.DataFrame([weather_data])

    def _final_data_cleanup(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform final cleanup operations on the features DataFrame.
        
        Args:
            features_df: Features DataFrame to clean
            
        Returns:
            Cleaned features DataFrame
        """
        # Create a copy to avoid modifying the original
        cleaned_df = features_df.copy()
        
        # Fix qualifying times - ensure eliminated drivers have NaN for Q2/Q3
        if 'qualifying_time_q1' in cleaned_df.columns:
            # Drivers eliminated in Q1 (positions 16-20) should have NaN in Q2/Q3
            q1_eliminated = cleaned_df['grid_position'] >= 16
            if 'qualifying_time_q2' in cleaned_df.columns:
                cleaned_df.loc[q1_eliminated, 'qualifying_time_q2'] = np.nan
            if 'qualifying_time_q3' in cleaned_df.columns:
                cleaned_df.loc[q1_eliminated, 'qualifying_time_q3'] = np.nan
            
            # Drivers eliminated in Q2 (positions 11-15) should have NaN in Q3
            q2_eliminated = (cleaned_df['grid_position'] >= 11) & (cleaned_df['grid_position'] <= 15)
            if 'qualifying_time_q3' in cleaned_df.columns:
                cleaned_df.loc[q2_eliminated, 'qualifying_time_q3'] = np.nan
            
            # Only drivers in Q3 should have a gap_to_pole_seconds
            if 'gap_to_pole_seconds' in cleaned_df.columns:
                q3_participants = cleaned_df['grid_position'] <= 10
                cleaned_df.loc[~q3_participants, 'gap_to_pole_seconds'] = np.nan
        
        # Feature engineering - fill in missing data for rookies
        if 'championship_position' in cleaned_df.columns:
            # For rookies, assume they're at the back of the grid in their first race
            rookie_mask = cleaned_df['championship_position'].isna()
            # Set worst possible values for championship metrics
            max_position = cleaned_df['championship_position'].max()
            if pd.notna(max_position):
                cleaned_df.loc[rookie_mask, 'championship_position'] = max_position + 1
                cleaned_df.loc[rookie_mask, 'championship_points'] = 0
                cleaned_df.loc[rookie_mask, 'season_wins'] = 0
        
        # Add debugging info
        print(f"Data shape after cleanup: {cleaned_df.shape}")
        print(f"Columns with all NaN values: {cleaned_df.columns[cleaned_df.isna().all()].tolist()}")
        
        return cleaned_df

    def _get_race_labels(self, season: Union[int, str], race: Union[int, str]) -> pd.DataFrame:
        """
        Get race labels (final positions and times) for a specific race.
        
        Args:
            season: F1 season (year)
            race: Race number
            
        Returns:
            DataFrame with columns:
            - final_position: final race position (20 for DNFs)
            - race_time_millis: total race time in milliseconds (NaN for DNFs)
            - delta_pos: position change from grid to finish
        """
        # Fetch race results
        race_results = self.client.get_race_results(season, race)
        
        if race_results.empty:
            print(f"No race results found for {season} race {race}")
            return pd.DataFrame()
        
        # Initialize list to store processed results
        processed_results = []
        
        # Process each result
        for _, row in race_results.iterrows():
            # Get basic info
            driver_id = row.get('driver_id')
            status = row.get('status', '')
            grid_pos = row.get('grid_position')
            
            if status.lower() == 'finished' or status.lower() == 'lapped':
                final_position = row.get('position')
                # Use the directly extracted millisecond time from the client
                race_time = row.get('finish_time_sec')
            else:
                final_position = 20  # Set to last position for DNFs/non-finishers
                race_time = None
            
            result = {
                'driver_id': driver_id,
                'final_position': final_position,
                'final_time' : race_time,
                'grid_position': grid_pos
            }
            
            processed_results.append(result)
        
        # Create DataFrame
        labels_df = pd.DataFrame(processed_results)
        
        # Convert numeric columns
        numeric_cols = ['final_position', 'final_time', 'grid_position']
        for col in numeric_cols:
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(labels_df[col], errors='coerce')
        
        # Calculate position delta (grid to finish)
        if 'grid_position' in labels_df.columns and 'final_position' in labels_df.columns:
            # For DNFs (final_position = 20), delta_pos should be grid_position - 20
            labels_df['delta_pos'] = labels_df['grid_position'] - labels_df['final_position']
        
        # Drop grid_position as it's no longer needed
        if 'grid_position' in labels_df.columns:
            labels_df = labels_df.drop('grid_position', axis=1)
        
        # Set driver_id as index
        if 'driver_id' in labels_df.columns:
            labels_df = labels_df.set_index('driver_id')
        
        # Print debug info
        print(labels_df)
        
        return labels_df

# Example usage:
if __name__ == "__main__":
    # Example configuration
    config = {
        'data_ingest': {
            'client': {
                'api_key': None,
                'cache_ttl': 3600,
            }
        },
        'feature_engineering': {
            'historical_lookback_races': 5,
            'historical_lookback_years': 2,
            'core_features': {
                'recent_k_races': 5,
                'include_qualifying_data': True,
                'include_weather_data': True
            },
            'preprocessing': {
                'categorical_encoding': 'one-hot',
                'scale_features': True,
                'scaling_method': 'standard'
            },
            'common': {
                'standardize_driver_names': True,
                'standardize_team_names': True,
                'missing_values': {
                    'numeric_strategy': 'mean',
                    'categorical_strategy': 'mode'
                },
                'normalize_features': False,
                'normalize_method': 'standard',
                'remove_outliers': False
            }
        },
        'domain_knowledge': {
            'unlikely_to_win_points_drivers': ['sargeant'],
            'team_favoritism_mapping': {
                'red_bull': 'max_verstappen',
                'ferrari': 'charles_leclerc',
                'mercedes': 'lewis_hamilton'
            }
        }
    }
    
    # Initialize preprocessor
    preprocessor = F1DataPreprocessor(config)
    
    # Prepare data for a specific race
    try:
        features = preprocessor.prepare_data_for_race(season=2023, race=10)
        print(f"Successfully prepared data with {features.shape[1]} features for {features.shape[0]} drivers")
        print("Feature columns:")
        print(features.columns.tolist())
    except Exception as e:
        print(f"Error preparing data: {str(e)}")