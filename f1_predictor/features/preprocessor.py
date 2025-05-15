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
        
        # Step 4: Final preprocessing for model input
        # model_input_df = self._prepare_model_input(features_df)
        
        return features_df
    
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
        Fetch historical data for feature engineering.
        
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
        
        # Collect race results from previous seasons and races
        all_race_results = []
        
        for year in lookback_years:
            if year < current_year:
                # For previous years, get all races
                race_results_df = self.client.get_race_results(year)
                all_race_results.append(race_results_df)
            else:
                # For current year, get only races before the current one
                if int(race) > 1:
                    for r in range(1, int(race)):
                        race_result_df = self.client.get_race_results(year, r)
                        all_race_results.append(race_result_df)
        
        # Combine all race results
        if all_race_results:
            combined_race_results = pd.concat(all_race_results, ignore_index=True)
            historical_data['race_results'] = combined_race_results
        
        # Collect qualifying results
        all_qualifying_results = []
        
        for year in lookback_years:
            if year < current_year:
                # For previous years, get all qualifying sessions
                qualifying_results_df = self.client.get_qualifying_results(year)
                all_qualifying_results.append(qualifying_results_df)
            else:
                # For current year, get only qualifying sessions before the current race
                if int(race) > 1:
                    for r in range(1, int(race)):
                        qualifying_result_df = self.client.get_qualifying_results(year, r)
                        all_qualifying_results.append(qualifying_result_df)
        
        # Combine all qualifying results
        if all_qualifying_results:
            combined_qualifying_results = pd.concat(all_qualifying_results, ignore_index=True)
            historical_data['qualifying_results'] = combined_qualifying_results
        
        # Get most recent driver standings (from the previous race in current season)
        if int(race) > 1:
            driver_standings_df = self.client.get_driver_standings(season, int(race) - 1)
            if not driver_standings_df.empty:
                historical_data['driver_standings'] = driver_standings_df
        else:
            # If it's the first race, get the final standings from the previous season
            driver_standings_df = self.client.get_driver_standings(current_year - 1)
            if not driver_standings_df.empty:
                historical_data['driver_standings'] = driver_standings_df
        
        # Get most recent constructor standings
        if int(race) > 1:
            constructor_standings_df = self.client.get_constructor_standings(season, int(race) - 1)
            if not constructor_standings_df.empty:
                historical_data['constructor_standings'] = constructor_standings_df
        else:
            # If it's the first race, get the final standings from the previous season
            constructor_standings_df = self.client.get_constructor_standings(current_year - 1)
            if not constructor_standings_df.empty:
                historical_data['constructor_standings'] = constructor_standings_df
        
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
                # Apply standard preprocessing
                preprocessed_df = preprocess_data(df, self.config)
                preprocessed_data[key] = preprocessed_df
        
        return preprocessed_data
    
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