#!/usr/bin/env python
"""
Script to train F1 prediction models.
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

from f1_predictor.utils.config_loader import load_config
from f1_predictor.utils.logging_setup import setup_logging
from f1_predictor.data_ingest.jolpica_client import JolpicaClient
from f1_predictor.features.common_preprocessing import preprocess_data
from f1_predictor.features.core_features import compute_core_features
from f1_predictor.models.lap_time_model import LapTimeModel
from f1_predictor.ensemble.combiner import EnsembleCombiner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train F1 prediction models')
    
    parser.add_argument('--seasons', required=True, help='Season range to use for training (e.g., 2018-2023)')
    parser.add_argument('--config', default=None, help='Path to custom config file')
    parser.add_argument('--output-dir', default=None, help='Path to save trained models')
    parser.add_argument('--data-cache', default=None, help='Path to cache downloaded data')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def load_historical_data(
    config: Dict[str, Any], 
    seasons: List[int], 
    cache_dir: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data for training.
    
    Args:
        config: Configuration dictionary.
        seasons: List of seasons to load.
        cache_dir: Directory to cache downloaded data.
        
    Returns:
        Dictionary of historical data DataFrames.
    """
    # Initialize data sources based on config
    jolpica_config = config.get('data_sources', {}).get('jolpica', {})
    if jolpica_config.get('enabled', True):
        jolpica_client = JolpicaClient(jolpica_config)
    else:
        raise ValueError("At least one data source must be enabled")
    
    # Initialize results
    historical_data = {
        'race_results': [],
        'qualifying_results': [],
        'driver_standings': [],
        'constructor_standings': [],
        'circuits': []
    }
    
    # Load data for each season
    for season in seasons:
        logging.info(f"Loading data for season {season}")
        
        # Check if cached data exists
        cached = False
        if cache_dir:
            season_cache = os.path.join(cache_dir, f"season_{season}.pkl")
            if os.path.exists(season_cache):
                try:
                    season_data = pd.read_pickle(season_cache)
                    for key in historical_data:
                        if key in season_data:
                            historical_data[key].append(season_data[key])
                    cached = True
                    logging.info(f"Loaded cached data for season {season}")
                except Exception as e:
                    logging.warning(f"Could not load cached data for season {season}: {e}")
                    cached = False
        
        # If not cached, fetch from API
        if not cached:
            # Get race results
            race_results = jolpica_client.get_race_results(season)
            if not race_results.empty:
                historical_data['race_results'].append(race_results)
            
            # Get qualifying results
            qualifying_results = jolpica_client.get_qualifying_results(season)
            if not qualifying_results.empty:
                historical_data['qualifying_results'].append(qualifying_results)
            
            # Get driver standings
            driver_standings = jolpica_client.get_driver_standings(season)
            if not driver_standings.empty:
                historical_data['driver_standings'].append(driver_standings)
            
            # Get constructor standings
            constructor_standings = jolpica_client.get_constructor_standings(season)
            if not constructor_standings.empty:
                historical_data['constructor_standings'].append(constructor_standings)
            
            # Get circuits
            circuits = jolpica_client.get_circuits(season)
            if not circuits.empty:
                historical_data['circuits'].append(circuits)
            
            # Cache data if cache_dir is provided
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                season_cache = os.path.join(cache_dir, f"season_{season}.pkl")
                
                try:
                    # Create a dictionary of DataFrames to cache
                    season_data = {
                        'race_results': race_results,
                        'qualifying_results': qualifying_results,
                        'driver_standings': driver_standings,
                        'constructor_standings': constructor_standings,
                        'circuits': circuits
                    }
                    
                    pd.to_pickle(season_data, season_cache)
                    logging.info(f"Cached data for season {season}")
                except Exception as e:
                    logging.warning(f"Could not cache data for season {season}: {e}")
    
    # Combine data across seasons
    combined_data = {}
    for key in historical_data:
        if historical_data[key]:
            combined_data[key] = pd.concat(historical_data[key], ignore_index=True)
        else:
            combined_data[key] = pd.DataFrame()
    
    return combined_data


def prepare_training_data(
    config: Dict[str, Any],
    historical_data: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Prepare training data for the models.
    
    Args:
        config: Configuration dictionary.
        historical_data: Dictionary of historical data DataFrames.
        
    Returns:
        Tuple of (features_df, targets_df).
    """
    # Preprocess historical data
    processed_data = {}
    for key in historical_data:
        if not historical_data[key].empty:
            processed_data[key] = preprocess_data(historical_data[key], config)
        else:
            processed_data[key] = pd.DataFrame()
    
    # Create features and targets for each race
    features_by_race = {}
    targets_by_race = {}
    
    # Group qualifying and race results by season and round
    if ('qualifying_results' in processed_data and 'race_results' in processed_data and
        not processed_data['qualifying_results'].empty and not processed_data['race_results'].empty):
        
        qualifying_df = processed_data['qualifying_results']
        race_results_df = processed_data['race_results']
        
        # Group by season and race
        race_groups = []
        
        # Check if we have appropriate columns for grouping
        if all(col in qualifying_df.columns for col in ['season', 'race_name']):
            quali_groups = qualifying_df.groupby(['season', 'race_name'])
            race_groups = race_results_df.groupby(['season', 'race_name'])
            
            # Process each race
            for (season, race_name), quali_group in quali_groups:
                if (season, race_name) in race_groups.groups:
                    race_group = race_groups.get_group((season, race_name))
                    
                    # Create race data dictionary for feature engineering
                    race_data = {
                        'qualifying': quali_group,
                        'circuit': processed_data['circuits'][
                            processed_data['circuits']['circuit_name'] == quali_group['circuit_name'].iloc[0]
                        ] if 'circuits' in processed_data else None
                    }
                    
                    # Create historical data up to this race
                    historical_mask = ((race_results_df['season'] < season) | 
                                       ((race_results_df['season'] == season) & 
                                        (race_results_df['race_name'] != race_name)))
                    historical_data_subset = {
                        'race_results': race_results_df[historical_mask],
                        'driver_standings': processed_data.get('driver_standings', pd.DataFrame()),
                        'constructor_standings': processed_data.get('constructor_standings', pd.DataFrame())
                    }
                    
                    # Engineer features for this race
                    try:
                        features = compute_core_features(race_data, historical_data_subset, config)
                        
                        if not features.empty:
                            # Store features for this race
                            race_key = f"{season}_{race_name}"
                            features_by_race[race_key] = features
                            
                            # Create targets (actual race results)
                            race_results = race_group.set_index('driver_id')['position']
                            targets_by_race[race_key] = race_results
                    except Exception as e:
                        logging.warning(f"Error computing features for {season} {race_name}: {e}")
    
    return features_by_race, targets_by_race


def train_models(
    config: Dict[str, Any],
    features_by_race: Dict[str, pd.DataFrame],
    targets_by_race: Dict[str, pd.DataFrame],
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Train prediction models.
    
    Args:
        config: Configuration dictionary.
        features_by_race: Dictionary mapping race keys to feature DataFrames.
        targets_by_race: Dictionary mapping race keys to target Series.
        output_dir: Directory to save trained models.
        
    Returns:
        Dictionary of trained models.
    """
    # Determine output directory
    if output_dir is None:
        output_dir = config.get('project_paths', {}).get('trained_models', 'trained_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Update model paths in config to use the specified output directory
    models_config = config.get('models', {})
    for model_name, model_config in models_config.items():
        if 'model_path' in model_config:
            model_filename = os.path.basename(model_config['model_path'])
            model_config['model_path'] = os.path.join(output_dir, model_filename)
    
    # Initialize dictionary to store trained models
    trained_models = {}
    
    # Train lap time model if enabled
    if 'lap_time_model' in models_config:
        try:
            logging.info("Training lap time prediction model...")
            
            # Prepare training data
            X_train = pd.concat([features_by_race[race_key] for race_key in features_by_race])
            
            # Create target: actual lap times or positions
            # For now, we'll use positions since we might not have lap times for all races
            y_train = pd.DataFrame(index=X_train.index)
            y_train['lap_time'] = pd.concat([targets_by_race[race_key] for race_key in targets_by_race])
            
            # Train the model
            lap_time_model = LapTimeModel(models_config['lap_time_model'])
            lap_time_model.train(X_train, y_train)
            
            # Save the model
            lap_time_model.save_model()
            
            # Store in trained models dictionary
            trained_models['lap_time_model'] = lap_time_model
            logging.info("Lap time model training completed")
            
        except Exception as e:
            logging.error(f"Error training lap time model: {e}")
    
    # Train other models here following a similar pattern
    # ...
    
    # Train ensemble model if using Learning-to-Rank
    if config.get('ensemble', {}).get('method') == 'learning_to_rank':
        try:
            logging.info("Training Learning-to-Rank ensemble model...")
            
            # Prepare data for LTR
            train_ranks = []
            true_ranks = []
            
            # For each race, get predictions from individual models and true ranks
            for race_key in features_by_race:
                if race_key in targets_by_race:
                    race_model_ranks = {}
                    
                    # Get predictions from each trained model
                    for model_name, model in trained_models.items():
                        predictions = model.predict(features_by_race[race_key])
                        ranks = model.get_rank_output(predictions)
                        race_model_ranks[f"{model_name}_rank"] = ranks
                    
                    # If we have predictions from at least one model, add to training data
                    if race_model_ranks:
                        train_ranks.append(race_model_ranks)
                        true_ranks.append(targets_by_race[race_key].rank())
            
            # Train LTR model if we have data
            if train_ranks and true_ranks:
                ensemble = EnsembleCombiner(config)
                ensemble.train_ltr_model(train_ranks, true_ranks)
                
                # Store in trained models dictionary
                trained_models['ensemble'] = ensemble
                logging.info("Ensemble model training completed")
            
        except Exception as e:
            logging.error(f"Error training ensemble model: {e}")
    
    return trained_models


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config, "INFO" if not args.verbose else "DEBUG")
    logger.info("Starting F1 prediction model training")
    
    try:
        # Parse seasons range
        seasons_range = args.seasons.split('-')
        if len(seasons_range) == 2:
            start_year, end_year = int(seasons_range[0]), int(seasons_range[1])
            seasons = list(range(start_year, end_year + 1))
        else:
            seasons = [int(args.seasons)]
        
        # Load historical data
        logger.info(f"Loading historical data for seasons: {', '.join(map(str, seasons))}")
        historical_data = load_historical_data(config, seasons, args.data_cache)
        
        # Prepare training data
        logger.info("Preparing training data...")
        features_by_race, targets_by_race = prepare_training_data(config, historical_data)
        
        if not features_by_race:
            logger.error("No valid training data could be prepared. Check your data sources and configuration.")
            return 1
        
        logger.info(f"Prepared training data for {len(features_by_race)} races")
        
        # Train models
        logger.info("Training prediction models...")
        trained_models = train_models(config, features_by_race, targets_by_race, args.output_dir)
        
        if not trained_models:
            logger.warning("No models were successfully trained.")
            return 1
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 