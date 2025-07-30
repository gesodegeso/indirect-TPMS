"""
Indirect TPMS - Main Application Entry Point
Advanced Tire Pressure Monitoring System using Machine Learning
"""

import argparse
import sys
from typing import Optional
from indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator
from ml_tpms_module import TPMSMLEstimator, TPMSRealTimeEstimator

def run_demo():
    """Run basic demonstration of the TPMS system"""
    print("=== Indirect TPMS Demonstration ===")
    print("Initializing system components...")
    
    # Initialize components
    processor = TPMSDataProcessor(window_size=100, sampling_rate=100.0)
    simulator = TPMSDataSimulator()
    estimator = TPMSMLEstimator()
    
    print("Generating training data...")
    
    # Generate diverse training scenarios
    scenarios = [
        # (圧力設定, 車速, 運転パターン, サンプル数)
        ([2.2, 2.2, 2.2, 2.2], 60, 'straight', 40),  # Normal
        ([1.9, 2.2, 2.2, 2.2], 50, 'straight', 25),  # Mild low pressure
        ([1.6, 2.2, 2.2, 2.2], 40, 'turn', 15),     # Severe low pressure
        ([2.2, 1.8, 2.2, 2.2], 70, 'brake', 20),    # Different wheel
        ([1.8, 1.9, 2.2, 2.2], 55, 'straight', 15), # Multiple wheels
    ]
    
    for pressures, speed, maneuver, n_samples in scenarios:
        for i in range(n_samples):
            data = simulator.generate_sensor_data(pressures, speed, maneuver)
            processor.add_sensor_data(data)
            
            if len(processor.data_buffer) >= processor.window_size:
                features = processor.fuse_features()
                if features:
                    estimator.add_training_data(features, pressures)
    
    print(f"Training model with {len(estimator.training_features)} samples...")
    
    if estimator.train_models():
        print("Model training completed successfully!")
        
        # Real-time demonstration
        rt_estimator = TPMSRealTimeEstimator(estimator)
        
        print("\n=== Real-time Estimation Demo ===")
        test_scenarios = [
            ([2.2, 2.2, 2.2, 2.2], "Normal pressure"),
            ([1.7, 2.2, 2.2, 2.2], "Low pressure - Front Left"),
            ([2.2, 1.5, 2.2, 2.2], "Critical pressure - Front Right"),
        ]
        
        for pressures, description in test_scenarios:
            print(f"\nTesting: {description}")
            print(f"Actual pressures: {pressures}")
            
            # Generate test data
            for i in range(10):
                test_data = simulator.generate_sensor_data(pressures, 60, 'straight')
                processor.add_sensor_data(test_data)
                
                if i == 9:  # Last iteration
                    features = processor.fuse_features()
                    if features:
                        prediction = rt_estimator.process_features(features)
                        
                        print(f"Estimated pressures: {prediction.pressures.round(2)}")
                        print(f"Confidence: {prediction.confidence:.3f}")
                        print(f"Alert level: {prediction.alert_level.name}")
                        if prediction.affected_wheels:
                            print(f"Affected wheels: {prediction.affected_wheels}")
        
        # System status
        status = rt_estimator.get_system_status()
        print(f"\nSystem Status: {status}")
        
    else:
        print("Model training failed!")

def run_training_mode(data_path: Optional[str] = None):
    """Run system in training mode"""
    print("Training mode not yet implemented")
    print("Use demo mode for basic functionality")

def run_real_time_mode(model_path: Optional[str] = None):
    """Run system in real-time monitoring mode"""
    print("Real-time mode not yet implemented")
    print("This would connect to actual vehicle CAN bus")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Indirect TPMS - Advanced Tire Pressure Monitoring System"
    )
    
    parser.add_argument(
        "mode",
        choices=["demo", "train", "monitor"],
        help="Operation mode: demo (demonstration), train (model training), monitor (real-time)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data (for train mode)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model (for monitor mode)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            run_demo()
        elif args.mode == "train":
            run_training_mode(args.data_path)
        elif args.mode == "monitor":
            run_real_time_mode(args.model_path)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()