"""
Sample Dataset Generator for Indirect TPMS
Generates realistic sensor data for training and testing
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator
except ImportError:
    print("Warning: Could not import TPMS modules. Creating basic dataset only.")
    TPMSDataProcessor = None
    TPMSDataSimulator = None

class TPMSDatasetGenerator:
    """Generate comprehensive TPMS datasets"""
    
    def __init__(self, output_dir: str = "data/sample_data"):
        """
        Args:
            output_dir: Directory to save datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset configurations
        self.driving_scenarios = [
            # Scenario name, duration (samples), speed, maneuver, description
            ("city_driving", 500, 45, "mixed", "City traffic with stops and turns"),
            ("highway_cruising", 800, 80, "straight", "Highway constant speed"),
            ("parking_maneuvers", 300, 15, "turn", "Low speed parking operations"),
            ("emergency_braking", 200, 60, "brake", "Hard braking scenarios"),
            ("mountain_roads", 600, 50, "mixed", "Winding mountain roads"),
            ("stop_and_go", 400, 25, "mixed", "Traffic jam conditions"),
        ]
        
        self.pressure_conditions = [
            # Condition name, [FL, FR, RL, RR], severity, description
            ("normal", [2.2, 2.2, 2.2, 2.2], 0, "All tires properly inflated"),
            ("fl_low", [1.8, 2.2, 2.2, 2.2], 1, "Front left tire low pressure"),
            ("fr_low", [2.2, 1.8, 2.2, 2.2], 1, "Front right tire low pressure"),
            ("rl_low", [2.2, 2.2, 1.8, 2.2], 1, "Rear left tire low pressure"),
            ("rr_low", [2.2, 2.2, 2.2, 1.8], 1, "Rear right tire low pressure"),
            ("fl_critical", [1.5, 2.2, 2.2, 2.2], 2, "Front left tire critical pressure"),
            ("fr_critical", [2.2, 1.5, 2.2, 2.2], 2, "Front right tire critical pressure"),
            ("front_low", [1.8, 1.9, 2.2, 2.2], 1, "Both front tires low"),
            ("rear_low", [2.2, 2.2, 1.8, 1.9], 1, "Both rear tires low"),
            ("multiple_critical", [1.5, 2.2, 1.6, 2.2], 2, "Multiple tires critical"),
            ("gradual_deflation", [2.0, 2.2, 2.2, 2.2], 1, "Gradual pressure loss"),
            ("overinflated", [2.6, 2.2, 2.2, 2.2], 1, "Overinflated front left"),
        ]
    
    def generate_raw_sensor_data(self) -> pd.DataFrame:
        """Generate raw sensor data from various scenarios"""
        print("Generating raw sensor data...")
        
        if TPMSDataSimulator is None:
            return self._generate_basic_sensor_data()
        
        simulator = TPMSDataSimulator()
        all_data = []
        
        sample_id = 0
        for scenario_name, duration, base_speed, base_maneuver, description in self.driving_scenarios:
            for condition_name, pressures, severity, condition_desc in self.pressure_conditions:
                
                # Generate samples for this scenario-condition combination
                for i in range(duration):
                    # Add speed variation
                    speed_variation = np.random.normal(0, base_speed * 0.1)
                    current_speed = max(5, base_speed + speed_variation)
                    
                    # Vary maneuver based on scenario
                    if base_maneuver == "mixed":
                        maneuvers = ["straight", "turn", "brake"]
                        weights = [0.6, 0.3, 0.1] if "city" in scenario_name else [0.8, 0.15, 0.05]
                        current_maneuver = np.random.choice(maneuvers, p=weights)
                    else:
                        current_maneuver = base_maneuver
                    
                    # Add pressure variation (tire heating, measurement noise)
                    current_pressures = [
                        p + np.random.normal(0, 0.05) for p in pressures
                    ]
                    
                    # Generate sensor data
                    sensor_data = simulator.generate_sensor_data(
                        current_pressures, current_speed, current_maneuver
                    )
                    
                    # Create data record
                    record = {
                        'sample_id': sample_id,
                        'timestamp': sensor_data.timestamp,
                        'scenario': scenario_name,
                        'pressure_condition': condition_name,
                        'severity': severity,
                        'vehicle_speed': sensor_data.vehicle_speed,
                        'maneuver': current_maneuver,
                        'wheel_speed_fl': sensor_data.wheel_speeds[0],
                        'wheel_speed_fr': sensor_data.wheel_speeds[1],
                        'wheel_speed_rl': sensor_data.wheel_speeds[2],
                        'wheel_speed_rr': sensor_data.wheel_speeds[3],
                        'acceleration_x': sensor_data.accelerations[0],
                        'acceleration_y': sensor_data.accelerations[1],
                        'acceleration_z': sensor_data.accelerations[2],
                        'steering_angle': sensor_data.steering_angle,
                        'brake_pressure': sensor_data.brake_pressure,
                        'engine_load': sensor_data.engine_load,
                        'gps_latitude': sensor_data.gps_data['latitude'],
                        'gps_longitude': sensor_data.gps_data['longitude'],
                        'gps_altitude': sensor_data.gps_data['altitude'],
                        'gps_speed': sensor_data.gps_data['speed'],
                        'actual_pressure_fl': current_pressures[0],
                        'actual_pressure_fr': current_pressures[1],
                        'actual_pressure_rl': current_pressures[2],
                        'actual_pressure_rr': current_pressures[3],
                    }
                    
                    all_data.append(record)
                    sample_id += 1
                    
                    if sample_id % 1000 == 0:
                        print(f"Generated {sample_id} samples...")
        
        df = pd.DataFrame(all_data)
        print(f"Generated {len(df)} total samples")
        return df
    
    def _generate_basic_sensor_data(self) -> pd.DataFrame:
        """Generate basic sensor data without TPMS modules"""
        print("Generating basic synthetic sensor data...")
        
        np.random.seed(42)  # For reproducibility
        n_samples = 5000
        
        data = []
        for i in range(n_samples):
            # Random scenario and condition
            scenario = np.random.choice([s[0] for s in self.driving_scenarios])
            condition = np.random.choice([c[0] for c in self.pressure_conditions])
            
            # Basic sensor values
            vehicle_speed = np.random.uniform(10, 100)
            wheel_speeds = np.random.uniform(vehicle_speed * 0.95, vehicle_speed * 1.05, 4)
            accelerations = np.random.normal([0, 0, -9.81], [2, 2, 1])
            
            # Pressure values based on condition
            if "normal" in condition:
                pressures = [2.2, 2.2, 2.2, 2.2]
            elif "low" in condition:
                pressures = [1.8, 2.2, 2.2, 2.2] if "fl" in condition else [2.2, 2.2, 2.2, 2.2]
            elif "critical" in condition:
                pressures = [1.5, 2.2, 2.2, 2.2] if "fl" in condition else [2.2, 2.2, 2.2, 2.2]
            else:
                pressures = [2.0, 2.1, 2.2, 2.1]
            
            record = {
                'sample_id': i,
                'timestamp': i * 0.01,
                'scenario': scenario,
                'pressure_condition': condition,
                'severity': 0 if "normal" in condition else (2 if "critical" in condition else 1),
                'vehicle_speed': vehicle_speed,
                'maneuver': np.random.choice(['straight', 'turn', 'brake']),
                'wheel_speed_fl': wheel_speeds[0],
                'wheel_speed_fr': wheel_speeds[1],
                'wheel_speed_rl': wheel_speeds[2],
                'wheel_speed_rr': wheel_speeds[3],
                'acceleration_x': accelerations[0],
                'acceleration_y': accelerations[1],
                'acceleration_z': accelerations[2],
                'steering_angle': np.random.uniform(-30, 30),
                'brake_pressure': np.random.uniform(0, 10),
                'engine_load': np.random.uniform(20, 80),
                'gps_latitude': 35.0 + np.random.uniform(-0.1, 0.1),
                'gps_longitude': 139.0 + np.random.uniform(-0.1, 0.1),
                'gps_altitude': 50 + np.random.uniform(-20, 100),
                'gps_speed': vehicle_speed,
                'actual_pressure_fl': pressures[0] + np.random.normal(0, 0.05),
                'actual_pressure_fr': pressures[1] + np.random.normal(0, 0.05),
                'actual_pressure_rl': pressures[2] + np.random.normal(0, 0.05),
                'actual_pressure_rr': pressures[3] + np.random.normal(0, 0.05),
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_feature_dataset(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Generate feature-extracted dataset"""
        print("Generating feature dataset...")
        
        if TPMSDataProcessor is None:
            return self._generate_basic_features(raw_data)
        
        processor = TPMSDataProcessor(window_size=50, sampling_rate=100.0)
        feature_data = []
        
        # Group by scenario and condition for feature extraction
        for (scenario, condition), group in raw_data.groupby(['scenario', 'pressure_condition']):
            if len(group) < 50:
                continue
                
            # Reset processor for each group
            processor = TPMSDataProcessor(window_size=50, sampling_rate=100.0)
            
            # Add data to processor and extract features
            for _, row in group.iterrows():
                # Create sensor data object
                from indirect_tpms_module import SensorData
                sensor_data = SensorData(
                    timestamp=row['timestamp'],
                    wheel_speeds=np.array([row['wheel_speed_fl'], row['wheel_speed_fr'], 
                                         row['wheel_speed_rl'], row['wheel_speed_rr']]),
                    accelerations=np.array([row['acceleration_x'], row['acceleration_y'], 
                                          row['acceleration_z']]),
                    vehicle_speed=row['vehicle_speed'],
                    steering_angle=row['steering_angle'],
                    brake_pressure=row['brake_pressure'],
                    engine_load=row['engine_load'],
                    gps_data={'latitude': row['gps_latitude'], 'longitude': row['gps_longitude'],
                             'altitude': row['gps_altitude'], 'speed': row['gps_speed']}
                )
                
                processor.add_sensor_data(sensor_data)
                
                # Extract features when buffer is full
                if len(processor.data_buffer) >= processor.window_size:
                    features = processor.fuse_features()
                    
                    if features:
                        feature_record = {
                            'scenario': scenario,
                            'pressure_condition': condition,
                            'severity': row['severity'],
                            'actual_pressure_fl': row['actual_pressure_fl'],
                            'actual_pressure_fr': row['actual_pressure_fr'],
                            'actual_pressure_rl': row['actual_pressure_rl'],
                            'actual_pressure_rr': row['actual_pressure_rr'],
                        }
                        feature_record.update(features)
                        feature_data.append(feature_record)
            
            if len(feature_data) % 100 == 0:
                print(f"Processed {len(feature_data)} feature samples...")
        
        return pd.DataFrame(feature_data)
    
    def _generate_basic_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features without TPMS modules"""
        print("Generating basic feature dataset...")
        
        feature_data = []
        
        for _, row in raw_data.iterrows():
            # Basic statistical features
            wheel_speeds = [row['wheel_speed_fl'], row['wheel_speed_fr'], 
                           row['wheel_speed_rl'], row['wheel_speed_rr']]
            accelerations = [row['acceleration_x'], row['acceleration_y'], row['acceleration_z']]
            
            features = {
                'scenario': row['scenario'],
                'pressure_condition': row['pressure_condition'],
                'severity': row['severity'],
                'actual_pressure_fl': row['actual_pressure_fl'],
                'actual_pressure_fr': row['actual_pressure_fr'],
                'actual_pressure_rl': row['actual_pressure_rl'],
                'actual_pressure_rr': row['actual_pressure_rr'],
                
                # Wheel speed features
                'wheel_speed_mean': np.mean(wheel_speeds),
                'wheel_speed_std': np.std(wheel_speeds),
                'wheel_speed_range': np.max(wheel_speeds) - np.min(wheel_speeds),
                'fl_fr_speed_ratio': wheel_speeds[0] / wheel_speeds[1] if wheel_speeds[1] > 0 else 1,
                'rl_rr_speed_ratio': wheel_speeds[2] / wheel_speeds[3] if wheel_speeds[3] > 0 else 1,
                
                # Acceleration features
                'acceleration_magnitude': np.sqrt(sum(a**2 for a in accelerations)),
                'vertical_acceleration': row['acceleration_z'],
                'lateral_acceleration': row['acceleration_y'],
                'longitudinal_acceleration': row['acceleration_x'],
                
                # Vehicle dynamics features
                'speed_deviation': abs(row['vehicle_speed'] - np.mean(wheel_speeds)),
                'steering_activity': abs(row['steering_angle']),
                'brake_activity': row['brake_pressure'],
                'engine_load_normalized': row['engine_load'] / 100.0,
            }
            
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        # Stratified split by severity and condition
        np.random.seed(42)
        
        train_data = []
        val_data = []
        test_data = []
        
        for condition in df['pressure_condition'].unique():
            condition_data = df[df['pressure_condition'] == condition].copy()
            n_samples = len(condition_data)
            
            # Shuffle
            condition_data = condition_data.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split indices
            test_end = int(n_samples * test_size)
            val_end = test_end + int(n_samples * val_size)
            
            test_data.append(condition_data[:test_end])
            val_data.append(condition_data[test_end:val_end])
            train_data.append(condition_data[val_end:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        # Shuffle final datasets
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def generate_metadata(self) -> Dict:
        """Generate dataset metadata"""
        metadata = {
            "dataset_info": {
                "name": "Indirect TPMS Sample Dataset",
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "description": "Synthetic dataset for indirect tire pressure monitoring system development",
                "total_samples": 0,  # Will be updated
                "features_count": 0,  # Will be updated
            },
            "driving_scenarios": [
                {
                    "name": name,
                    "duration_samples": duration,
                    "base_speed_kmh": speed,
                    "maneuver_type": maneuver,
                    "description": description
                }
                for name, duration, speed, maneuver, description in self.driving_scenarios
            ],
            "pressure_conditions": [
                {
                    "name": name,
                    "pressures_bar": pressures,
                    "severity_level": severity,
                    "description": description
                }
                for name, pressures, severity, description in self.pressure_conditions
            ],
            "sensor_specifications": {
                "wheel_speed_sensors": {
                    "type": "ABS/ESP wheel speed sensors",
                    "sampling_rate_hz": 100,
                    "accuracy": "±0.1 rad/s"
                },
                "accelerometers": {
                    "type": "3-axis MEMS accelerometers",
                    "sampling_rate_hz": 100,
                    "range": "±16g",
                    "accuracy": "±0.1 m/s²"
                },
                "vehicle_dynamics": {
                    "steering_angle": "±180°",
                    "brake_pressure": "0-10 bar",
                    "engine_load": "0-100%"
                }
            },
            "target_variables": {
                "tire_pressures": {
                    "normal_range": "2.0-2.4 bar",
                    "low_threshold": "1.8 bar",
                    "critical_threshold": "1.5 bar"
                }
            }
        }
        
        return metadata
    
    def save_datasets(self):
        """Generate and save all datasets"""
        print("=== TPMS Dataset Generation ===")
        
        # Generate raw sensor data
        raw_data = self.generate_raw_sensor_data()
        raw_path = os.path.join(self.output_dir, "raw_sensor_data.csv")
        raw_data.to_csv(raw_path, index=False)
        print(f"Saved raw sensor data: {raw_path} ({len(raw_data)} samples)")
        
        # Generate feature dataset
        feature_data = self.generate_feature_dataset(raw_data)
        feature_path = os.path.join(self.output_dir, "feature_data.csv")
        feature_data.to_csv(feature_path, index=False)
        print(f"Saved feature data: {feature_path} ({len(feature_data)} samples)")
        
        # Create train/test splits for features
        if len(feature_data) > 0:
            train_df, val_df, test_df = self.create_train_test_split(feature_data)
            
            train_path = os.path.join(self.output_dir, "train_features.csv")
            val_path = os.path.join(self.output_dir, "val_features.csv")
            test_path = os.path.join(self.output_dir, "test_features.csv")
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"Saved training data: {train_path} ({len(train_df)} samples)")
            print(f"Saved validation data: {val_path} ({len(val_df)} samples)")
            print(f"Saved test data: {test_path} ({len(test_df)} samples)")
        
        # Generate and save metadata
        metadata = self.generate_metadata()
        metadata["dataset_info"]["total_samples"] = len(raw_data)
        metadata["dataset_info"]["features_count"] = len(feature_data.columns) if len(feature_data) > 0 else 0
        
        metadata_path = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        # Generate dataset summary
        self._generate_summary_report(raw_data, feature_data)
        
        print(f"\n=== Dataset Generation Complete ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Files generated: {len(os.listdir(self.output_dir))}")
    
    def _generate_summary_report(self, raw_data: pd.DataFrame, feature_data: pd.DataFrame):
        """Generate dataset summary report"""
        summary_path = os.path.join(self.output_dir, "dataset_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("TPMS Sample Dataset Summary Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Raw data summary
            f.write("Raw Sensor Data:\n")
            f.write(f"  Total samples: {len(raw_data)}\n")
            f.write(f"  Scenarios: {raw_data['scenario'].nunique()}\n")
            f.write(f"  Pressure conditions: {raw_data['pressure_condition'].nunique()}\n")
            f.write(f"  Severity distribution:\n")
            for severity, count in raw_data['severity'].value_counts().sort_index().items():
                severity_name = ["Normal", "Low Pressure", "Critical"][severity]
                f.write(f"    {severity_name}: {count} ({count/len(raw_data)*100:.1f}%)\n")
            f.write("\n")
            
            # Feature data summary
            if len(feature_data) > 0:
                f.write("Feature Data:\n")
                f.write(f"  Total samples: {len(feature_data)}\n")
                f.write(f"  Features: {len(feature_data.columns) - 7}\n")  # Excluding metadata columns
                f.write(f"  Scenarios: {feature_data['scenario'].nunique()}\n")
                f.write(f"  Conditions: {feature_data['pressure_condition'].nunique()}\n\n")
            
            # Scenario breakdown
            f.write("Scenario Breakdown:\n")
            for scenario, count in raw_data['scenario'].value_counts().items():
                f.write(f"  {scenario}: {count} samples\n")
            f.write("\n")
            
            # Condition breakdown
            f.write("Pressure Condition Breakdown:\n")
            for condition, count in raw_data['pressure_condition'].value_counts().items():
                f.write(f"  {condition}: {count} samples\n")
            f.write("\n")
            
            f.write("Files generated:\n")
            for filename in sorted(os.listdir(self.output_dir)):
                if filename.endswith(('.csv', '.json', '.txt')):
                    f.write(f"  {filename}\n")
        
        print(f"Saved summary report: {summary_path}")

def main():
    """Main dataset generation function"""
    print("TPMS Sample Dataset Generator")
    print("=" * 30)
    
    # Create output directory
    output_dir = "data/sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    generator = TPMSDatasetGenerator(output_dir)
    generator.save_datasets()
    
    print("\nDataset generation completed successfully!")
    print(f"Check the '{output_dir}' directory for generated files.")

if __name__ == "__main__":
    main()