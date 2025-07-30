# ===========================================
# tests/test_tpms_module.py
# ===========================================

"""
Unit tests for indirect TPMS module
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator, SensorData

class TestTPMSDataProcessor(unittest.TestCase):
    """Test TPMSDataProcessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = TPMSDataProcessor(window_size=10, sampling_rate=100.0)
        self.simulator = TPMSDataSimulator()
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.window_size, 10)
        self.assertEqual(self.processor.sampling_rate, 100.0)
        self.assertFalse(self.processor.is_calibrated)
        self.assertEqual(len(self.processor.data_buffer), 0)
    
    def test_add_sensor_data(self):
        """Test adding sensor data"""
        data = self.simulator.generate_sensor_data([2.2, 2.2, 2.2, 2.2])
        
        # Should return False until window is full
        result = self.processor.add_sensor_data(data)
        self.assertFalse(result)
        
        # Fill the window
        for i in range(9):
            data = self.simulator.generate_sensor_data([2.2, 2.2, 2.2, 2.2])
            result = self.processor.add_sensor_data(data)
        
        # Should return True when window is full
        self.assertTrue(result)
        self.assertEqual(len(self.processor.data_buffer), 10)
    
    def test_feature_extraction(self):
        """Test feature extraction methods"""
        # Fill buffer with data
        for i in range(15):
            data = self.simulator.generate_sensor_data([2.2, 2.2, 2.2, 2.2], 60.0)
            self.processor.add_sensor_data(data)
        
        # Test wheel radius features
        wheel_features = self.processor.extract_wheel_radius_features()
        self.assertIsInstance(wheel_features, dict)
        self.assertGreater(len(wheel_features), 0)
        
        # Test vibration features
        vib_features = self.processor.extract_vibration_features()
        self.assertIsInstance(vib_features, dict)
        self.assertGreater(len(vib_features), 0)
        
        # Test fused features
        fused_features = self.processor.fuse_features()
        self.assertIsInstance(fused_features, dict)
        self.assertGreater(len(fused_features), 0)
        
        # Check for NaN values
        for key, value in fused_features.items():
            self.assertFalse(np.isnan(value), f"Feature {key} contains NaN")
            self.assertFalse(np.isinf(value), f"Feature {key} contains Inf")
    
    def test_calibration(self):
        """Test baseline calibration"""
        # Fill buffer with normal pressure data
        for i in range(15):
            data = self.simulator.generate_sensor_data([2.2, 2.2, 2.2, 2.2])
            self.processor.add_sensor_data(data)
        
        # Calibrate
        result = self.processor.calibrate_baseline()
        self.assertTrue(result)
        self.assertTrue(self.processor.is_calibrated)
        self.assertGreater(len(self.processor.baseline_params), 0)

class TestTPMSDataSimulator(unittest.TestCase):
    """Test TPMSDataSimulator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = TPMSDataSimulator()
    
    def test_data_generation(self):
        """Test sensor data generation"""
        pressures = [2.2, 2.2, 2.2, 2.2]
        data = self.simulator.generate_sensor_data(pressures, 60.0, 'straight')
        
        # Check data structure
        self.assertIsInstance(data, SensorData)
        self.assertEqual(len(data.wheel_speeds), 4)
        self.assertEqual(len(data.accelerations), 3)
        self.assertGreater(data.vehicle_speed, 0)
        self.assertIsInstance(data.gps_data, dict)
    
    def test_pressure_effects(self):
        """Test pressure effects on wheel speeds"""
        normal_pressures = [2.2, 2.2, 2.2, 2.2]
        low_pressures = [1.8, 2.2, 2.2, 2.2]
        
        normal_data = self.simulator.generate_sensor_data(normal_pressures, 60.0)
        low_data = self.simulator.generate_sensor_data(low_pressures, 60.0)
        
        # Low pressure should result in higher wheel speed (smaller radius)
        self.assertGreater(low_data.wheel_speeds[0], normal_data.wheel_speeds[0])

if __name__ == '__main__':
    unittest.main()
