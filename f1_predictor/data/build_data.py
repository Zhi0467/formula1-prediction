"""
Script to build the complete F1 prediction dataset for model training.
This script processes all races in a given season to create a comprehensive dataset.
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
import logging
from pathlib import Path

from ..features.preprocessor import F1DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class F1DatasetBuilder:
    """
    Builds a complete dataset for F1 prediction model training.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "data/"):
        """
        Initialize the dataset builder.
        
        Args:
            config: Configuration dictionary for the preprocessor
            output_dir: Directory to save the processed datasets
        """
        self.config = config
        self.output_dir = output_dir
        self.preprocessor = F1DataPreprocessor(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage for all features and labels
        self.all_features = []
        self.all_labels = []
        self.race_metadata = []
    
    def build_season_dataset(self, season: int, start_race: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build dataset for a complete season or range of races.
        
        Args:
            season: F1 season (year)
            start_race: First race to process (default: 1)
            end_race: Last race to process (default: None, meaning process all races)
            
        Returns:
            Tuple of (features_df, labels_df) containing the complete dataset
        """
        logger.info(f"Building dataset for {season} season")
        
        # Get total number of races if end_race not specified
        end_race = 20
        last_race_df = self.preprocessor.client.get_race_results(season, "last")
        if not last_race_df.empty and 'round' in last_race_df.columns:
            end_race = int(last_race_df['round'].iloc[0])
        else:
            raise ValueError(f"Could not determine total races for {season}")
        
        logger.info(f"Processing races {start_race} to {end_race}")
        
        # Process each race
        for race in range(start_race, end_race + 1):
            try:
                logger.info(f"Processing {season} Race {race}")
                
                # Get features and labels for this race
                features_df, labels_df = self.preprocessor.prepare_data_for_race(
                    season=season,
                    race=race
                )
                
                if features_df.empty or labels_df.empty:
                    logger.warning(f"No data available for {season} Race {race}")
                    continue
                
                # Add race metadata
                features_df['season'] = season
                features_df['race'] = race
                labels_df['season'] = season
                labels_df['race'] = race
                
                # Store the data
                self.all_features.append(features_df)
                self.all_labels.append(labels_df)
                
                # Store metadata
                race_meta = {
                    'season': season,
                    'race': race,
                    'num_drivers': len(features_df),
                    'num_features': features_df.shape[1]
                }
                self.race_metadata.append(race_meta)
                
                logger.info(f"Successfully processed {season} Race {race}")
                logger.info(f"Features shape: {features_df.shape}")
                logger.info(f"Labels shape: {labels_df.shape}")
                
            except Exception as e:
                logger.error(f"Error processing {season} Race {race}: {str(e)}")
                continue
        
        # Combine all data
        if not self.all_features or not self.all_labels:
            raise ValueError("No data was successfully processed")
        
        combined_features = pd.concat(self.all_features, axis=0)
        combined_labels = pd.concat(self.all_labels, axis=0)
        
        # Save the datasets
        self._save_datasets(combined_features, combined_labels)
        
        return combined_features, combined_labels
    
    def build_multiple_seasons_dataset(self, start_season: int, end_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build dataset for multiple seasons in chronological order.
        
        Args:
            start_season: First season to process (inclusive)
            end_season: Last season to process (inclusive)
            
        Returns:
            Tuple of (features_df, labels_df) containing the complete dataset
        """
        logger.info(f"Building dataset for seasons {start_season} to {end_season}")
        
        # Initialize storage for all features and labels
        self.all_features = []
        self.all_labels = []
        self.race_metadata = []
        
        # Process each season in chronological order
        for season in range(start_season, end_season + 1):
            try:
                logger.info(f"\nProcessing season {season}")
                
                # Get total number of races for this season
                last_race_df = self.preprocessor.client.get_race_results(season, "last")
                if not last_race_df.empty and 'round' in last_race_df.columns:
                    end_race = int(last_race_df['round'].iloc[0])
                else:
                    logger.warning(f"Could not determine total races for {season}, skipping season")
                    continue
                
                # Process each race in the season
                for race in range(1, end_race + 1):
                    try:
                        logger.info(f"Processing {season} Race {race}")
                        
                        # Get features and labels for this race
                        features_df, labels_df = self.preprocessor.prepare_data_for_race(
                            season=season,
                            race=race
                        )
                        
                        if features_df.empty or labels_df.empty:
                            logger.warning(f"No data available for {season} Race {race}")
                            continue
                        
                        # Add race metadata
                        features_df['season'] = season
                        features_df['race'] = race
                        labels_df['season'] = season
                        labels_df['race'] = race
                        
                        # Store the data
                        self.all_features.append(features_df)
                        self.all_labels.append(labels_df)
                        
                        # Store metadata
                        race_meta = {
                            'season': season,
                            'race': race,
                            'num_drivers': len(features_df),
                            'num_features': features_df.shape[1],
                            'num_dnfs': (labels_df['final_position'] == 20).sum()
                        }
                        self.race_metadata.append(race_meta)
                        
                        logger.info(f"Successfully processed {season} Race {race}")
                        logger.info(f"Features shape: {features_df.shape}")
                        logger.info(f"Labels shape: {labels_df.shape}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {season} Race {race}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing season {season}: {str(e)}")
                continue
        
        # Combine all data
        if not self.all_features or not self.all_labels:
            raise ValueError("No data was successfully processed")
        
        combined_features = pd.concat(self.all_features, axis=0)
        combined_labels = pd.concat(self.all_labels, axis=0)
        
        # Save the datasets
        self._save_datasets(combined_features, combined_labels)
        
        return combined_features, combined_labels

    def _save_datasets(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Save the processed datasets and metadata.
        
        Args:
            features_df: Combined features DataFrame
            labels_df: Combined labels DataFrame
        """
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save features and labels
        features_file = os.path.join(self.output_dir, f"features_{timestamp}.csv")
        labels_file = os.path.join(self.output_dir, f"labels_{timestamp}.csv")
        
        features_df.to_csv(features_file)
        labels_df.to_csv(labels_file)
        
        # Save metadata
        metadata_df = pd.DataFrame(self.race_metadata)
        metadata_file = os.path.join(self.output_dir, f"metadata_{timestamp}.csv")
        metadata_df.to_csv(metadata_file)
        
        logger.info(f"Saved datasets to {self.output_dir}")
        logger.info(f"Features: {features_file}")
        logger.info(f"Labels: {labels_file}")
        logger.info(f"Metadata: {metadata_file}")
        
        # Save dataset statistics
        stats = {
            'total_races': len(self.race_metadata),
            'total_drivers': len(features_df),
            'total_features': features_df.shape[1],
            'total_dnfs': (labels_df['final_position'] == 20).sum(),
            'date_created': timestamp
        }
        
        stats_file = os.path.join(self.output_dir, f"dataset_stats_{timestamp}.txt")
        with open(stats_file, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved dataset statistics to {stats_file}")

