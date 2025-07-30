# ===========================================
# tests/test_ml_module.py
# ===========================================

"""
Unit tests for ML TPMS module
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_tpms_module import TPMSMLEstimator, BayesianNeuralNetwork, LazyClassifierEnsemble
from ml_tpms_module import TPMSAlertLevel, TPMSPrediction

class TestTPMSMLEstimator(unittest.TestCase):
    """Test ML estimator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.estimator = TPMSMLEstimator()
    
    def test_initialization(self):
        """Test estimator initialization"""
        self.assertEqual(self.estimator.pressure_thresholds, (1.8, 1.5))
        self.assertEqual(self.estimator.confidence_threshold, 0.7)
        self.assertFalse(self.estimator.is_trained)
    
    def test_add_training_data(self):
        """Test adding training data"""
        features = {'feature1': 1.0, 'feature2': 2.0}
        pressures = [2.2, 2.2, 2.2, 2.2]
        
        self.estimator.add_training_data(features, pressures)
        
        self.assertEqual(len(self.estimator.training_features), 1)
        self.assertEqual(len(self.estimator.training_pressures), 1)
        self.assertEqual(len(self.estimator.training_labels), 1)
    
    def test_label_creation(self):
        """Test pressure label creation"""
        # Normal pressure
        normal_pressures = np.array([[2.2, 2.2, 2.2, 2.2]])
        labels = self.estimator._create_pressure_labels(normal_pressures)
        self.assertEqual(labels[0], 0)  # Normal
        
        # Low pressure
        low_pressures = np.array([[1.7, 2.2, 2.2, 2.2]])
        labels = self.estimator._create_pressure_labels(low_pressures)
        self.assertEqual(labels[0], 1)  # Low pressure
        
        # Critical pressure
        critical_pressures = np.array([[1.4, 2.2, 2.2, 2.2]])
        labels = self.estimator._create_pressure_labels(critical_pressures)
        self.assertEqual(labels[0], 2)  # Critical

class TestBayesianNeuralNetwork(unittest.TestCase):
    """Test Bayesian Neural Network"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bnn = BayesianNeuralNetwork(input_dim=10, hidden_units=[16, 8])
    
    def test_initialization(self):
        """Test BNN initialization"""
        self.assertEqual(self.bnn.input_dim, 10)
        self.assertEqual(self.bnn.hidden_units, [16, 8])
        self.assertFalse(self.bnn.is_trained)
    
    def test_model_compilation(self):
        """Test model compilation"""
        self.bnn.compile_model()
        self.assertIsNotNone(self.bnn.model)
    
    @unittest.skip("Requires TensorFlow - skip for quick tests")
    def test_training(self):
        """Test BNN training (skipped for quick tests)"""
        # Generate dummy data
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 4) + 2.0  # Pressure around 2.0
        
        history = self.bnn.train(X, y, epochs=5, batch_size=16)
        
        self.assertTrue(self.bnn.is_trained)
        self.assertIn('loss', history)

class TestLazyClassifierEnsemble(unittest.TestCase):
    """Test Lazy Classifier Ensemble"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ensemble = LazyClassifierEnsemble()
    
    def test_initialization(self):
        """Test ensemble initialization"""
        self.assertFalse(self.ensemble.is_trained)
        self.assertEqual(len(self.ensemble.classifiers), 3)
    
    def test_training(self):
        """Test ensemble training"""
        # Generate dummy data
        X = np.random.randn(50, 5)
        y = np.random.choice([0, 1, 2], size=50)
        
        self.ensemble.train(X, y)
        
        self.assertTrue(self.ensemble.is_trained)
    
    def test_prediction(self):
        """Test ensemble prediction"""
        # Generate and train on dummy data
        X_train = np.random.randn(50, 5)
        y_train = np.random.choice([0, 1, 2], size=50)
        self.ensemble.train(X_train, y_train)
        
        # Test prediction
        X_test = np.random.randn(10, 5)
        predictions = self.ensemble.predict(X_test)
        proba = self.ensemble.predict_proba(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertEqual(proba.shape, (10, 3))
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))

if __name__ == '__main__':
    unittest.main()
