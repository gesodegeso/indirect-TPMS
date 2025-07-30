"""
Basic usage example for Indirect TPMS system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator
from src.ml_tpms_module import TPMSMLEstimator, TPMSRealTimeEstimator

def basic_example():
    """Basic usage demonstration"""
    print("=== Basic Indirect TPMS Usage ===")
    
    # 1. Initialize components
    processor = TPMSDataProcessor(window_size=50, sampling_rate=100.0)
    simulator = TPMSDataSimulator()
    estimator = TPMSMLEstimator()
    
    # 2. Collect training data
    print("Collecting training data...")
    
    training_data = [
        # (pressures, speed, maneuver, samples)
        ([2.2, 2.2, 2.2, 2.2], 60, 'straight', 30),  # Normal
        ([1.9, 2.2, 2.2, 2.2], 50, 'straight', 20),  # Low pressure
        ([1.6, 2.2, 2.2, 2.2], 40, 'turn', 15),     # Very low pressure
    ]
    
    for pressures, speed, maneuver, n_samples in training_data:
        for i in range(n_samples):
            data = simulator.generate_sensor_data(pressures, speed, maneuver)
            processor.add_sensor_data(data)
            
            if len(processor.data_buffer) >= processor.window_size:
                features = processor.fuse_features()
                if features:
                    estimator.add_training_data(features, pressures)
    
    # 3. Train the model
    print(f"Training with {len(estimator.training_features)} samples...")
    if estimator.train_models():
        print("Training completed!")
        
        # 4. Real-time estimation
        rt_estimator = TPMSRealTimeEstimator(estimator)
        
        # 5. Test with new data
        test_pressures = [1.8, 2.2, 2.2, 2.2]  # Low pressure in front-left
        print(f"\nTesting with pressures: {test_pressures}")
        
        for i in range(10):
            test_data = simulator.generate_sensor_data(test_pressures, 55, 'straight')
            processor.add_sensor_data(test_data)
            
            if i % 3 == 0:  # Every 3rd iteration
                features = processor.fuse_features()
                if features:
                    prediction = rt_estimator.process_features(features)
                    print(f"Step {i}: Estimated pressures = {prediction.pressures.round(2)}, "
                          f"Confidence = {prediction.confidence:.2f}")
    
    else:
        print("Training failed!")

if __name__ == "__main__":
    basic_example()