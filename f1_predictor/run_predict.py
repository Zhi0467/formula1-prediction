#!/usr/bin/env python
"""
Script to generate F1 race predictions.
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, Any

from f1_predictor.utils.config_loader import load_config
from f1_predictor.utils.logging_setup import setup_logging
from f1_predictor.data_ingest.jolpica_client import JolpicaClient
from f1_predictor.features.common_preprocessing import preprocess_data
from f1_predictor.features.core_features import compute_core_features
from f1_predictor.models.lap_time_model import LapTimeModel
from f1_predictor.ensemble.combiner import EnsembleCombiner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate F1 race predictions')
    
    parser.add_argument('--race', required=True, help='Race name or round number for prediction')
    parser.add_argument('--season', default='current', help='Season year (default: current)')
    parser.add_argument('--config', default=None, help='Path to custom config file')
    parser.add_argument('--expert-input', default=None, help='Path to expert predictions CSV')
    parser.add_argument('--output', default=None, help='Path to save predictions')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def load_data(config: Dict[str, Any], season: str, race: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load data for the specified race and historical data.
    
    Args:
        config: Configuration dictionary.
        season: Season year or 'current'.
        race: Race name or round number.
        
    Returns:
        Dictionary containing race data and historical data.
    """
    # Initialize data sources based on config
    jolpica_config = config.get('data_sources', {}).get('jolpica', {})
    if jolpica_config.get('enabled', True):
        jolpica_client = JolpicaClient(jolpica_config)
    else:
        raise ValueError("At least one data source must be enabled")
    
    # Get current race data
    race_data = {}
    
    # Get qualifying results
    race_data['qualifying'] = jolpica_client.get_qualifying_results(season, race)
    
    # Get circuit information
    if not race_data['qualifying'].empty:
        # Extract circuit ID from qualifying data
        circuit_id = race_data['qualifying']['circuit_id'].iloc[0] if 'circuit_id' in race_data['qualifying'].columns else None
        
        if circuit_id:
            # Get circuit details
            race_data['circuit'] = jolpica_client.get_circuits(season).loc[
                jolpica_client.get_circuits(season)['circuit_id'] == circuit_id
            ]
    
    # Get historical data
    historical_data = {}
    
    # Get race results for recent seasons
    current_year = pd.Timestamp.now().year
    start_year = current_year - 3  # Get data from the last 3 seasons
    
    race_results_list = []
    for year in range(start_year, current_year + 1):
        yearly_results = jolpica_client.get_race_results(year)
        if not yearly_results.empty:
            race_results_list.append(yearly_results)
    
    if race_results_list:
        historical_data['race_results'] = pd.concat(race_results_list, ignore_index=True)
    
    # Get latest driver standings
    historical_data['driver_standings'] = jolpica_client.get_driver_standings(season)
    
    # Get latest constructor standings
    historical_data['constructor_standings'] = jolpica_client.get_constructor_standings(season)
    
    return {
        'race_data': race_data,
        'historical_data': historical_data
    }


def predict_race(
    config: Dict[str, Any],
    race_data: Dict[str, pd.DataFrame],
    historical_data: Dict[str, pd.DataFrame],
    expert_predictions: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate predictions for a race.
    
    Args:
        config: Configuration dictionary.
        race_data: Dictionary of DataFrames for the current race.
        historical_data: Dictionary of historical DataFrames.
        expert_predictions: Optional DataFrame with expert predictions.
        
    Returns:
        DataFrame with predicted rankings for all drivers.
    """
    # Preprocess data
    for key in race_data:
        if isinstance(race_data[key], pd.DataFrame) and not race_data[key].empty:
            race_data[key] = preprocess_data(race_data[key], config)
    
    for key in historical_data:
        if isinstance(historical_data[key], pd.DataFrame) and not historical_data[key].empty:
            historical_data[key] = preprocess_data(historical_data[key], config)
    
    # Generate features
    features_df = compute_core_features(race_data, historical_data, config)
    
    # Initialize models based on config
    models_config = config.get('models', {})
    model_ranks = {}
    
    # Initialize and run lap time model if enabled
    if 'lap_time_model' in models_config:
        lap_time_model = LapTimeModel(models_config['lap_time_model'])
        lap_time_model.load_model()
        
        lap_time_predictions = lap_time_model.predict(features_df)
        lap_time_rank = lap_time_model.get_rank_output(lap_time_predictions)
        model_ranks['lap_time_rank'] = lap_time_rank
    
    # Initialize and run delta model if enabled
    if 'delta_model' in models_config:
        # Import here to avoid circular imports
        from f1_predictor.models.delta_model import DeltaModel
        
        delta_model = DeltaModel(models_config['delta_model'])
        delta_model.load_model()
        
        delta_predictions = delta_model.predict(features_df)
        
        # Get qualifying positions
        qualifying_positions = None
        if 'qualifying' in race_data and not race_data['qualifying'].empty:
            qualifying_positions = race_data['qualifying'].set_index('driver_id')['position']
        
        delta_rank = delta_model.get_rank_output(delta_predictions, qualifying_positions)
        model_ranks['delta_rank'] = delta_rank
    
    # Initialize and run H2H model if enabled
    if 'h2h_model' in models_config:
        # Import here to avoid circular imports
        from f1_predictor.models.h2h_model import H2HModel
        
        h2h_model = H2HModel(models_config['h2h_model'])
        h2h_model.load_model()
        
        h2h_predictions = h2h_model.predict(features_df)
        h2h_rank = h2h_model.get_rank_output(h2h_predictions)
        model_ranks['h2h_rank'] = h2h_rank
    
    # Initialize and run text model if enabled
    if 'text_model' in models_config and models_config['text_model'].get('enabled', False):
        # Import here to avoid circular imports
        from f1_predictor.models.text_model import TextModel
        
        text_model = TextModel(models_config['text_model'])
        text_model.load_model()
        
        text_predictions = text_model.predict(features_df)
        text_rank = text_model.get_rank_output(text_predictions)
        model_ranks['text_rank'] = text_rank
    
    # Extract expert rank if available
    expert_rank = None
    if expert_predictions is not None and not expert_predictions.empty:
        if 'rank' in expert_predictions.columns:
            expert_rank = expert_predictions.set_index('driver_id')['rank']
    
    # Combine models using ensemble
    ensemble_config = config.get('ensemble', {})
    ensemble = EnsembleCombiner(config)
    
    # Generate final predictions
    final_ranking = ensemble.combine_ranks(model_ranks, expert_rank)
    
    # Create a formatted results DataFrame
    results_df = pd.DataFrame(final_ranking).reset_index()
    results_df.columns = ['driver_id', 'predicted_rank']
    results_df = results_df.sort_values('predicted_rank')
    
    # Add driver names if available
    if 'qualifying' in race_data and 'driver_name' in race_data['qualifying'].columns:
        driver_names = race_data['qualifying'][['driver_id', 'driver_name']].drop_duplicates()
        results_df = pd.merge(results_df, driver_names, on='driver_id', how='left')
    
    # Add team names if available
    if 'qualifying' in race_data and 'team_name' in race_data['qualifying'].columns:
        team_names = race_data['qualifying'][['driver_id', 'team_name']].drop_duplicates()
        results_df = pd.merge(results_df, team_names, on='driver_id', how='left')
    
    return results_df


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config, "INFO" if not args.verbose else "DEBUG")
    logger.info(f"Generating predictions for {args.season} {args.race}")
    
    try:
        # Load race and historical data
        logger.info("Loading data...")
        data = load_data(config, args.season, args.race)
        
        # Load expert predictions if provided
        expert_predictions = None
        if args.expert_input and os.path.exists(args.expert_input):
            logger.info(f"Loading expert predictions from {args.expert_input}")
            expert_predictions = pd.read_csv(args.expert_input)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = predict_race(
            config,
            data['race_data'],
            data['historical_data'],
            expert_predictions
        )
        
        # Output predictions
        if args.output:
            # Save to file
            logger.info(f"Saving predictions to {args.output}")
            predictions.to_csv(args.output, index=False)
        else:
            # Print to console
            print("\nPredicted Race Ranking:")
            print("-----------------------")
            for i, (_, row) in enumerate(predictions.iterrows(), 1):
                driver_name = row.get('driver_name', row['driver_id'])
                team_name = row.get('team_name', '')
                print(f"{i}. {driver_name} ({team_name})")
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 