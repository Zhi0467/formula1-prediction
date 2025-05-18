"""
Main entry point for the F1 Prediction project.
Use this script to test various modules and functionalities.
"""
import sys
import os

# Add the project root to the Python path
# This allows importing modules from f1_predictor
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- EXAMPLE USAGE ---
# You can uncomment and modify the sections below to test different parts of your project.

def test_jolpica_client():
    """Example function to test the JolpicaClient."""
    print("Testing JolpicaClient...")
    from f1_predictor.data_ingest.jolpica_client import JolpicaClient
    
    # Basic config, replace with your actual config if needed
    client_config = {'api_key': None, 'cache_ttl': 3600}
    client = JolpicaClient(client_config)
    
    try:
        # Test fetching race results
        race_results = client.get_race_results(season=2023, race=1)
        if not race_results.empty:
            print(f"Successfully fetched {len(race_results)} race results for 2023, Race 1.")
            print(race_results.head())
        else:
            print("No race results found or an error occurred.")
        
        # Test fetching circuits
        circuits = client.get_circuits(season=2023)
        if not circuits.empty:
            print(f"\nSuccessfully fetched {len(circuits)} circuits for 2023.")
            print(circuits.head())
        else:
            print("No circuits found or an error occurred.")
            
    except Exception as e:
        print(f"An error occurred during JolpicaClient test: {e}")
    print("-" * 50)

def test_common_preprocessing():
    """Example function to test common_preprocessing functions."""
    print("Testing common_preprocessing...")
    from f1_predictor.data_ingest.jolpica_client import JolpicaClient
    from f1_predictor.features.common_preprocessing import preprocess_data
    
    client_config = {'api_key': None, 'cache_ttl': 3600}
    client = JolpicaClient(client_config)
    
    # Fetch some data to preprocess
    race_data = client.get_race_results(season=2023, race=1)
    
    if race_data.empty:
        print("Could not fetch data for preprocessing test.")
        return
        
    # Example preprocessing config
    # Ensure your actual config structure matches what preprocess_data expects
    config = {
        'feature_engineering': {
            'common': {
                'standardize_driver_names': True,
                'standardize_team_names': True,
                'missing_values': {
                    'numeric_strategy': 'mean',
                    'categorical_strategy': 'mode'
                },
                'normalize_features': True,
                'normalize_method': 'standard'
            }
        }
    }
    
    try:
        processed_data = preprocess_data(race_data, config)
        print(f"Successfully preprocessed data. Shape: {processed_data.shape}")
        print(processed_data.head())
    except Exception as e:
        print(f"An error occurred during common_preprocessing test: {e}")
    print("-" * 50)

def test_get_driver_standings_client():
    """Tests the get_driver_standings method of JolpicaClient."""
    print("Testing JolpicaClient.get_driver_standings()...")
    from f1_predictor.data_ingest.jolpica_client import JolpicaClient
    
    client_config = {'api_key': None, 'cache_ttl': 3600}
    client = JolpicaClient(client_config)
    
    test_cases = [
        {"season": 2023, "round": 1, "description": "2023 Season, Round 1"},
        {"season": 2023, "round": "last", "description": "2023 Season, Last Round"},
        {"season": 2022, "round": None, "description": "2022 Season, End of Season Standings"},
    ]
    
    for case in test_cases:
        print(f"\nFetching driver standings for: {case['description']}")
        try:
            standings_df = client.get_driver_standings(season=case["season"], round_num=case["round"])
            if not standings_df.empty:
                print(f"Successfully fetched {len(standings_df)} records.")
                print("Columns:", standings_df.columns.tolist())
                print(standings_df)
                # Specifically check for the 'position' column, which caused the error
                if 'position' not in standings_df.columns:
                    print("WARNING: 'position' column is missing!")
                else:
                    print("'position' column is present.")
            else:
                print("No driver standings found or an error occurred (empty DataFrame).")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 30)
    print("-" * 50)

def test_f1_data_preprocessor():
    """Example function to test the F1DataPreprocessor."""
    print("Testing F1DataPreprocessor...")
    from f1_predictor.features.preprocessor import F1DataPreprocessor
    import pandas as pd
    import os

    # Create output directory if it doesn't exist
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure you have a valid configuration structure
    config = {
        'data_ingest': {
            'client': {'api_key': None, 'cache_ttl': 3600}
        },
        'feature_engineering': {
            'historical_lookback_races': 5,
            'historical_lookback_years': 1, # Adjusted for faster testing
            'core_features': {
                'recent_k_races': 3, # Adjusted for faster testing
                'include_qualifying_data': True,
                'include_historical_features': True,
                'include_standings_features': True,
                'include_track_specific_features': True,
                'include_track_type_features': False,
                'include_weather_data': False,
                'include_domain_knowledge_features': False
            },
            'preprocessing': {
                'categorical_encoding': 'one-hot',
                'scale_features': True,
                'scaling_method': 'standard'
            },
            'common': { # This section is used by preprocess_data called internally
                'standardize_driver_names': True,
                'standardize_team_names': True,
                'missing_values': {
                    'numeric_strategy': 'mean',
                    'categorical_strategy': 'mode'
                }
            }
        },
        'domain_knowledge': {} # Add if needed
    }

    preprocessor = F1DataPreprocessor(config)
    try:
        season = 2024
        race = 'last'

        # Generate the processed features
        features_df, labels_df = preprocessor.prepare_data_for_race(season=season, race=race, fetch_weather=False)
        
        # Save processed features
        features_file = f"{output_dir}/processed_features_{season}_race{race}.csv"
        features_df.to_csv(features_file)
        
    except Exception as e:
        print(f"An error occurred during F1DataPreprocessor test: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 50)


if __name__ == "__main__":
    print("Starting F1 Predictor Test Runner...")
    
    # --- CHOOSE WHICH TEST TO RUN ---
    # Uncomment the test you want to execute.
    
    # test_jolpica_client()
    # test_common_preprocessing()
    # test_get_driver_standings_client() 
    # Test this specific function
    test_f1_data_preprocessor() 
    