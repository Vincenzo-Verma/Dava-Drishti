"""
Basic Tests for Forest Fire Prediction System
"""

import unittest
import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.utils.config import Config
from app.models.unet_model import UNetFirePredictionModel
from app.models.lstm_model import LSTMFireTemporalModel
from app.models.cellular_automata import CellularAutomataFireSpread

class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['DATABASE_URL'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.DATA_DIR
        Config.DATA_DIR = self.temp_dir
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        Config.DATA_DIR = self.original_data_dir
    
    def test_app_creation(self):
        """Test that the Flask app can be created"""
        self.assertIsNotNone(self.app)
        self.assertTrue(self.app.config['TESTING'])
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_unet_model_initialization(self):
        """Test U-Net model initialization"""
        model = UNetFirePredictionModel()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.config)
    
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMFireTemporalModel()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.config)
    
    def test_cellular_automata_initialization(self):
        """Test Cellular Automata initialization"""
        model = CellularAutomataFireSpread()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.config)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(Config.MODEL_CONFIG)
        self.assertIn('unet', Config.MODEL_CONFIG)
        self.assertIn('lstm', Config.MODEL_CONFIG)
        self.assertIn('cellular_automata', Config.MODEL_CONFIG)
    
    def test_api_endpoints_exist(self):
        """Test that API endpoints are registered"""
        with self.app.app_context():
            rules = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/api/v1/health', rules)
            self.assertIn('/api/v1/fire-risk/<region>', rules)
            self.assertIn('/api/v1/fire-simulation', rules)

class TestModelFunctionality(unittest.TestCase):
    """Model functionality tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.DATA_DIR
        Config.DATA_DIR = self.temp_dir
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        Config.DATA_DIR = self.original_data_dir
    
    def test_unet_model_building(self):
        """Test U-Net model building"""
        model = UNetFirePredictionModel()
        built_model = model.build_model()
        self.assertIsNotNone(built_model)
        self.assertEqual(built_model.input_shape, (None,) + model.config['input_shape'])
    
    def test_lstm_model_building(self):
        """Test LSTM model building"""
        model = LSTMFireTemporalModel()
        built_model = model.build_model()
        self.assertIsNotNone(built_model)
    
    def test_cellular_automata_simulation(self):
        """Test cellular automata simulation"""
        model = CellularAutomataFireSpread()
        result = model.simulate_fire_spread(
            ignition_point=(50, 50),
            wind_speed=10.0,
            wind_dir=45.0,
            duration_hours=2
        )
        self.assertIn('burned_area_km2', result)
        self.assertIn('final_grid', result)
        self.assertIn('burned_mask', result)
    
    def test_feature_processing(self):
        """Test feature processing"""
        from app.data.processors.feature_processor import FeatureProcessor
        
        processor = FeatureProcessor()
        region_bounds = [12.0, 13.0, 77.0, 78.0]  # Bangalore region
        date = '2024-01-15'
        
        # Test feature stack creation (with dummy data)
        feature_stack = processor.create_feature_stack(region_bounds, date)
        self.assertIsNotNone(feature_stack)
        self.assertEqual(len(feature_stack.shape), 3)  # height, width, features

if __name__ == '__main__':
    unittest.main() 