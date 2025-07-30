"""
TPMS Sample Dataset Usage Examples
Demonstrates how to work with the generated datasets
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_datasets(data_dir="data/sample_data"):
    """Load all available datasets"""
    datasets = {}
    
    files_to_load = [
        ("raw", "raw_sensor_data.csv"),
        ("features", "feature_data.csv"),
        ("train", "train_features.csv"),
        ("val", "val_features.csv"),
        ("test", "test_features.csv")
    ]
    
    for name, filename in files_to_load:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            datasets[name] = pd.read_csv(filepath)
            print(f"Loaded {name}: {datasets[name].shape}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "dataset_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            datasets['metadata'] = json.load(f)
    
    return datasets

def explore_raw_data(raw_data):
    """Explore raw sensor data"""
    print("\n=== Raw Data Exploration ===")
    
    # Basic statistics
    print(f"Dataset shape: {raw_data.shape}")
    print(f"Time range: {raw_data['timestamp'].min():.1f} - {raw_data['timestamp'].max():.1f} seconds")
    print(f"Scenarios: {list(raw_data['scenario'].unique())}")
    print(f"Pressure conditions: {list(raw_data['pressure_condition'].unique())}")
    
    # Severity distribution
    print("\nSeverity Distribution:")
    severity_counts = raw_data['severity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        severity_name = ["Normal", "Low Pressure", "Critical"][severity]
        print(f"  {severity_name}: {count} ({count/len(raw_data)*100:.1f}%)")
    
    # Pressure statistics
    print(f"\nPressure Statistics:")
    pressure_cols = ['actual_pressure_fl', 'actual_pressure_fr', 'actual_pressure_rl', 'actual_pressure_rr']
    for col in pressure_cols:
        wheel = col.split('_')[-1].upper()
        print(f"  {wheel}: {raw_data[col].mean():.2f} ± {raw_data[col].std():.2f} bar")
    
    return raw_data

def visualize_data(raw_data):
    """Create visualizations of the dataset"""
    print("\n=== Data Visualization ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('TPMS Dataset Overview', fontsize=16)
    
    # 1. Pressure distribution by wheel
    ax = axes[0, 0]
    pressure_cols = ['actual_pressure_fl', 'actual_pressure_fr', 'actual_pressure_rl', 'actual_pressure_rr']
    raw_data[pressure_cols].boxplot(ax=ax)
    ax.set_title('Pressure Distribution by Wheel')
    ax.set_ylabel('Pressure (bar)')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Severity distribution
    ax = axes[0, 1]
    severity_counts = raw_data['severity'].value_counts().sort_index()
    severity_labels = ["Normal", "Low", "Critical"]
    ax.bar(severity_labels, severity_counts.values)
    ax.set_title('Severity Distribution')
    ax.set_ylabel('Count')
    
    # 3. Speed distribution by scenario
    ax = axes[0, 2]
    scenarios = raw_data['scenario'].unique()[:5]  # Top 5 scenarios
    for scenario in scenarios:
        data = raw_data[raw_data['scenario'] == scenario]['vehicle_speed']
        ax.hist(data, alpha=0.6, label=scenario, bins=20)
    ax.set_title('Speed Distribution by Scenario')
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 4. Wheel speed correlation
    ax = axes[1, 0]
    speed_cols = ['wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr']
    corr_matrix = raw_data[speed_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Wheel Speed Correlation')
    
    # 5. Acceleration magnitude over time
    ax = axes[1, 1]
    sample_data = raw_data.iloc[::100]  # Sample every 100th point
    acc_magnitude = np.sqrt(sample_data['acceleration_x']**2 + 
                           sample_data['acceleration_y']**2 + 
                           sample_data['acceleration_z']**2)
    ax.plot(sample_data['timestamp'], acc_magnitude)
    ax.set_title('Acceleration Magnitude Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    
    # 6. Pressure vs Speed relationship
    ax = axes[1, 2]
    # Sample data for clarity
    sample_data = raw_data.sample(n=min(1000, len(raw_data)))
    scatter = ax.scatter(sample_data['vehicle_speed'], sample_data['actual_pressure_fl'], 
                        c=sample_data['severity'], alpha=0.6, cmap='viridis')
    ax.set_title('FL Pressure vs Vehicle Speed')
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('FL Pressure (bar)')
    plt.colorbar(scatter, ax=ax, label='Severity')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_features(feature_data):
    """Analyze extracted features"""
    print("\n=== Feature Analysis ===")
    
    if feature_data is None or len(feature_data) == 0:
        print("No feature data available")
        return
    
    # Identify feature columns (exclude metadata)
    metadata_cols = ['scenario', 'pressure_condition', 'severity'] + \
                   [col for col in feature_data.columns if col.startswith('actual_pressure')]
    feature_cols = [col for col in feature_data.columns if col not in metadata_cols]
    
    print(f"Total features: {len(feature_cols)}")
    
    # Feature importance using Random Forest
    if len(feature_data) > 100:
        print("Calculating feature importance...")
        
        # Prepare data
        X = feature_data[feature_cols].fillna(0)
        y = feature_data['actual_pressure_fl']  # Use FL pressure as target
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

def train_baseline_model(datasets):
    """Train a baseline regression model"""
    print("\n=== Baseline Model Training ===")
    
    if 'train' not in datasets or 'test' not in datasets:
        print("Training/test data not available")
        return
    
    train_data = datasets['train']
    test_data = datasets['test']
    
    # Prepare features and targets
    metadata_cols = ['scenario', 'pressure_condition', 'severity'] + \
                   [col for col in train_data.columns if col.startswith('actual_pressure')]
    feature_cols = [col for col in train_data.columns if col not in metadata_cols]
    
    X_train = train_data[feature_cols].fillna(0)
    X_test = test_data[feature_cols].fillna(0)
    
    # Train separate models for each wheel
    wheels = ['fl', 'fr', 'rl', 'rr']
    results = {}
    
    for wheel in wheels:
        target_col = f'actual_pressure_{wheel}'
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[wheel] = {
            'mae': mae,
            'r2': r2,
            'model': model
        }
        
        print(f"{wheel.upper()} Wheel - MAE: {mae:.3f} bar, R²: {r2:.3f}")
    
    # Overall statistics
    all_maes = [results[wheel]['mae'] for wheel in wheels]
    all_r2s = [results[wheel]['r2'] for wheel in wheels]
    
    print(f"\nOverall Performance:")
    print(f"  Average MAE: {np.mean(all_maes):.3f} ± {np.std(all_maes):.3f} bar")
    print(f"  Average R²: {np.mean(all_r2s):.3f} ± {np.std(all_r2s):.3f}")
    
    return results

def demonstrate_anomaly_detection(datasets):
    """Demonstrate anomaly detection capabilities"""
    print("\n=== Anomaly Detection Demo ===")
    
    if 'test' not in datasets:
        print("Test data not available")
        return
    
    test_data = datasets['test']
    
    # Group by severity
    normal_data = test_data[test_data['severity'] == 0]
    low_pressure_data = test_data[test_data['severity'] == 1]
    critical_data = test_data[test_data['severity'] == 2]
    
    print(f"Normal samples: {len(normal_data)}")
    print(f"Low pressure samples: {len(low_pressure_data)}")
    print(f"Critical samples: {len(critical_data)}")
    
    # Analyze pressure distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (data, label) in enumerate([(normal_data, "Normal"), 
                                      (low_pressure_data, "Low Pressure"), 
                                      (critical_data, "Critical")]):
        if len(data) > 0:
            pressure_data = data[['actual_pressure_fl', 'actual_pressure_fr', 
                                'actual_pressure_rl', 'actual_pressure_rr']].values.flatten()
            axes[i].hist(pressure_data, bins=20, alpha=0.7)
            axes[i].set_title(f'{label} Condition')
            axes[i].set_xlabel('Pressure (bar)')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(x=1.8, color='orange', linestyle='--', label='Low threshold')
            axes[i].axvline(x=1.5, color='red', linestyle='--', label='Critical threshold')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demonstration function"""
    print("TPMS Dataset Analysis Demo")
    print("=" * 30)
    
    # Check if datasets exist
    data_dir = "data/sample_data"
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        print("Please run: python data/generate_sample_data.py")
        return
    
    # Load datasets
    datasets = load_datasets(data_dir)
    
    if not datasets:
        print("No datasets found. Generating sample data...")
        # Try to generate data
        try:
            sys.path.append('data')
            from generate_sample_data import TPMSDatasetGenerator
            generator = TPMSDatasetGenerator(data_dir)
            generator.save_datasets()
            datasets = load_datasets(data_dir)
        except ImportError:
            print("Could not generate datasets. Please run generate_sample_data.py manually.")
            return
    
    # Perform analysis
    if 'raw' in datasets:
        raw_data = explore_raw_data(datasets['raw'])
        visualize_data(raw_data)
    
    if 'features' in datasets:
        analyze_features(datasets['features'])
    
    if 'train' in datasets and 'test' in datasets:
        train_baseline_model(datasets)
        demonstrate_anomaly_detection(datasets)
    
    # Display metadata
    if 'metadata' in datasets:
        metadata = datasets['metadata']
        print(f"\n=== Dataset Metadata ===")
        print(f"Name: {metadata['dataset_info']['name']}")
        print(f"Version: {metadata['dataset_info']['version']}")
        print(f"Created: {metadata['dataset_info']['created']}")
        print(f"Total samples: {metadata['dataset_info']['total_samples']}")
    
    print("\n=== Analysis Complete ===")
    print("The datasets are ready for TPMS development!")

if __name__ == "__main__":
    main()