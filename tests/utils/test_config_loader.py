"""Tests for the config loader utility."""

import os
import unittest
import tempfile
import yaml

from f1_predictor.utils.config_loader import load_config, _validate_config


class TestConfigLoader(unittest.TestCase):
    """Test cases for the config loader utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Create a valid test config
        self.valid_config = {
            'project_paths': {
                'raw_data': 'data/raw/',
                'processed_data': 'data/processed/',
                'trained_models': 'trained_models/',
                'log_file': 'logs/f1_predictor.log'
            },
            'data_sources': {
                'jolpica': {'enabled': True}
            },
            'feature_engineering': {
                'common': {},
                'core_features': {'recent_k_races': 5}
            },
            'models': {
                'lap_time_model': {
                    'type': 'xgboost',
                    'model_path': 'trained_models/lap_time_xgb.json'
                }
            },
            'ensemble': {
                'method': 'weighted_average',
                'weights': {
                    'lap_time_rank': 1.0
                }
            },
            'evaluation': {
                'metrics': ['spearman_rank_correlation']
            }
        }
        
        # Write the valid config to a file
        self.config_path = os.path.join(self.test_dir.name, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.valid_config, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.test_dir.cleanup()
    
    def test_load_config(self):
        """Test loading a valid configuration file."""
        config = load_config(self.config_path)
        
        # Verify the loaded config matches the expected config
        self.assertEqual(config['project_paths']['raw_data'], 'data/raw/')
        self.assertEqual(config['models']['lap_time_model']['type'], 'xgboost')
        self.assertEqual(config['ensemble']['method'], 'weighted_average')
    
    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        # This should not raise an exception
        _validate_config(self.valid_config)
    
    def test_validate_config_missing_section(self):
        """Test validating a configuration with a missing required section."""
        # Create a config with a missing section
        invalid_config = self.valid_config.copy()
        del invalid_config['models']
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            _validate_config(invalid_config)
    
    def test_validate_config_invalid_ensemble(self):
        """Test validating a configuration with an invalid ensemble method."""
        # Create a config with an invalid ensemble method
        invalid_config = self.valid_config.copy()
        invalid_config['ensemble']['method'] = 'invalid_method'
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            _validate_config(invalid_config)
    
    def test_validate_config_missing_weights(self):
        """Test validating a configuration with missing weights for weighted average."""
        # Create a config with missing weights
        invalid_config = self.valid_config.copy()
        del invalid_config['ensemble']['weights']
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            _validate_config(invalid_config)


if __name__ == '__main__':
    unittest.main() 