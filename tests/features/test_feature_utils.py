"""Tests for the feature utility functions."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from f1_predictor.features.feature_utils import (
    calculate_rolling_average,
    calculate_consistency,
    one_hot_encode_categorical,
    calculate_position_changes
)


class TestFeatureUtils(unittest.TestCase):
    """Test cases for the feature utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'driver_id': ['driver1', 'driver1', 'driver1', 'driver2', 'driver2', 'driver2'],
            'race_date': [
                datetime(2021, 3, 28), datetime(2021, 4, 18), datetime(2021, 5, 2),
                datetime(2021, 3, 28), datetime(2021, 4, 18), datetime(2021, 5, 2)
            ],
            'position': [3, 1, 2, 5, 7, 4],
            'grid_position': [5, 2, 1, 3, 6, 8],
            'points': [15, 25, 18, 10, 6, 12],
            'team': ['Team A', 'Team A', 'Team A', 'Team B', 'Team B', 'Team B']
        })
    
    def test_calculate_rolling_average(self):
        """Test calculating rolling averages."""
        # Calculate rolling average of positions for each driver
        rolling_avg = calculate_rolling_average(
            self.sample_df,
            value_col='position',
            entity_col='driver_id',
            date_col='race_date',
            window=2
        )
        
        # Expected values:
        # driver1: [3, (3+1)/2=2, (1+2)/2=1.5]
        # driver2: [5, (5+7)/2=6, (7+4)/2=5.5]
        expected = pd.Series([3.0, 2.0, 1.5, 5.0, 6.0, 5.5])
        
        # Check that the calculated values match expected values
        pd.testing.assert_series_equal(
            rolling_avg.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )
    
    def test_calculate_consistency(self):
        """Test calculating consistency (variation)."""
        # Calculate consistency (standard deviation) of positions
        consistency = calculate_consistency(
            self.sample_df,
            value_col='position',
            entity_col='driver_id',
            date_col='race_date',
            window=3,
            metric='std'
        )
        
        # For the first row of each driver, std of a single value is 0
        # For the second row, std of two values
        # For the third row, std of three values
        self.assertAlmostEqual(consistency.iloc[2], 1.0)  # std of [3, 1, 2]
        self.assertAlmostEqual(consistency.iloc[5], 1.528, places=3)  # std of [5, 7, 4]
    
    def test_one_hot_encode_categorical(self):
        """Test one-hot encoding of categorical variables."""
        # One-hot encode the 'team' column
        encoded_df = one_hot_encode_categorical(
            self.sample_df,
            categorical_cols=['team']
        )
        
        # Check that the original column is removed
        self.assertNotIn('team', encoded_df.columns)
        
        # Check that the new columns exist and have correct values
        self.assertIn('team_Team A', encoded_df.columns)
        self.assertIn('team_Team B', encoded_df.columns)
        
        # Check values for first driver (Team A)
        self.assertEqual(encoded_df.loc[0, 'team_Team A'], 1)
        self.assertEqual(encoded_df.loc[0, 'team_Team B'], 0)
        
        # Check values for second driver (Team B)
        self.assertEqual(encoded_df.loc[3, 'team_Team A'], 0)
        self.assertEqual(encoded_df.loc[3, 'team_Team B'], 1)
    
    def test_calculate_position_changes(self):
        """Test calculating position changes (grid to finish)."""
        # Calculate position changes
        changes = calculate_position_changes(
            self.sample_df,
            start_pos_col='grid_position',
            end_pos_col='position'
        )
        
        # Expected values:
        # driver1: 3-5=-2, 1-2=-1, 2-1=1
        # driver2: 5-3=2, 7-6=1, 4-8=-4
        expected = pd.Series([-2, -1, 1, 2, 1, -4])
        
        # Check that the calculated values match expected values
        pd.testing.assert_series_equal(
            changes.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )


if __name__ == '__main__':
    unittest.main() 