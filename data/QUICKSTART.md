# TPMS Dataset Quick Start Guide

この簡単なガイドで、5分以内にTPMSデータセットを使い始めることができます。

## 🚀 すぐに始める

### Step 1: サンプルデータの確認
```python
import pandas as pd

# 小さなサンプルデータを読み込み
sample_data = pd.read_csv("data/sample_data/sample_features_small.csv")
print(f"Sample data shape: {sample_data.shape}")
print(sample_data.head())
```

### Step 2: データ生成（フルデータセット）
```bash
# 完全なデータセットを生成
cd your-repo-directory
python data/generate_sample_data.py
```

### Step 3: データ分析の実行
```bash
# データ分析デモを実行
python examples/data_analysis_demo.py
```

## 📊 すぐ使えるコード例

### 基本的なデータ読み込み
```python
import pandas as pd
import numpy as np

# データ読み込み
train_data = pd.read_csv("data/sample_data/train_features.csv")
test_data = pd.read_csv("data/sample_data/test_features.csv")

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Features: {len(train_data.columns) - 7}")  # Exclude metadata columns
```

### 特徴量とターゲットの分離
```python
# メタデータ列を除外
metadata_cols = ['scenario', 'pressure_condition', 'severity', 
                'actual_pressure_fl', 'actual_pressure_fr', 
                'actual_pressure_rl', 'actual_pressure_rr']

feature_cols = [col for col in train_data.columns if col not in metadata_cols]

# 特徴量とターゲット
X_train = train_data[feature_cols]
y_train = train_data[['actual_pressure_fl', 'actual_pressure_fr', 
                     'actual_pressure_rl', 'actual_pressure_rr']]

X_test = test_data[feature_cols]
y_test = test_data[['actual_pressure_fl', 'actual_pressure_fr', 
                   'actual_pressure_rl', 'actual_pressure_rr']]

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target matrix shape: {y_train.shape}")
```

### 簡単な機械学習モデル
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# モデル訓練（前左輪の例）
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train['actual_pressure_fl'])

# 予測
y_pred = model.predict(X_test)

# 評価
mae = mean_absolute_error(y_test['actual_pressure_fl'], y_pred)
print(f"Mean Absolute Error: {mae:.3f} bar")
```

### データ可視化
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 圧力分布の確認
plt.figure(figsize=(12, 4))

# 正常 vs 異常の圧力分布
plt.subplot(1, 3, 1)
normal_data = train_data[train_data['severity'] == 0]
abnormal_data = train_data[train_data['severity'] > 0]

plt.hist(normal_data['actual_pressure_fl'], alpha=0.7, label='Normal', bins=20)
plt.hist(abnormal_data['actual_pressure_fl'], alpha=0.7, label='Abnormal', bins=20)
plt.xlabel('FL Pressure (bar)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Pressure Distribution')

# 車輪間の圧力相関
plt.subplot(1, 3, 2)
pressure_cols = ['actual_pressure_fl', 'actual_pressure_fr', 
                'actual_pressure_rl', 'actual_pressure_rr']
corr_matrix = train_data[pressure_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Wheel Pressure Correlation')

# 重要度分析
plt.subplot(1, 3, 3)
importance = model.feature_importances_
top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]

features, importances = zip(*top_features)
plt.barh(range(len(features)), importances)
plt.yticks(range(len(features)), features)
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')

plt.tight_layout()
plt.show()
```

## 🎯 典型的なタスク

### 1. 異常検知（分類）
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 分類用ターゲット作成
y_class_train = train_data['severity']
y_class_test = test_data['severity']

# 分類器訓練
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_class_train)

# 予測・評価
y_pred_class = classifier.predict(X_test)
print(classification_report(y_class_test, y_pred_class, 
                          target_names=['Normal', 'Low', 'Critical']))
```

### 2. 圧力回帰（回帰）
```python
# 全輪同時回帰
from sklearn.multioutput import MultiOutputRegressor

multi_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
multi_regressor.fit(X_train, y_train)

# 予測
y_pred_multi = multi_regressor.predict(X_test)

# 各輪のMAE
wheels = ['FL', 'FR', 'RL', 'RR']
for i, wheel in enumerate(wheels):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_multi[:, i])
    print(f"{wheel} Wheel MAE: {mae:.3f} bar")
```

### 3. 時系列分析
```python
# 生データからの時系列分析
raw_data = pd.read_csv("data/sample_data/raw_sensor_data.csv")

# 特定条件のデータ抽出
condition_data = raw_data[
    (raw_data['scenario'] == 'city_driving') & 
    (raw_data['pressure_condition'] == 'fl_low')
].head(100)

# 時系列プロット
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(condition_data['timestamp'], condition_data['actual_pressure_fl'], label='FL Pressure')
plt.plot(condition_data['timestamp'], condition_data['actual_pressure_fr'], label='FR Pressure')
plt.ylabel('Pressure (bar)')
plt.legend()
plt.title('Tire Pressure Over Time')

plt.subplot(2, 1, 2)
plt.plot(condition_data['timestamp'], condition_data['vehicle_speed'], label='Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.legend()
plt.title('Vehicle Speed Over Time')

plt.tight_layout()
plt.show()
```

## 📈 性能指標

### 期待される性能
- **圧力推定精度**: MAE < 0.1 bar
- **異常検知率**: > 95% (圧力低下15%以上)
- **偽陽性率**: < 5%
- **応答時間**: < 1分（圧力低下検出）

### モデル評価
```python
def evaluate_tpms_performance(y_true, y_pred, threshold=1.8):
    """TPMS性能評価"""
    # 圧力推定精度
    mae = mean_absolute_error(y_true, y_pred)
    
    # 異常検知性能
    true_anomaly = y_true < threshold
    pred_anomaly = y_pred < threshold
    
    tp = np.sum(true_anomaly & pred_anomaly)
    fp = np.sum(~true_anomaly & pred_anomaly)
    tn = np.sum(~true_anomaly & ~pred_anomaly)
    fn = np.sum(true_anomaly & ~pred_anomaly)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'mae': mae,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }

# 評価実行
performance = evaluate_tpms_performance(
    y_test['actual_pressure_fl'].values, 
    y_pred
)
print(f"Performance: {performance}")
```

## 🔧 トラブルシューティング

### よくある問題

**1. データが見つからない**
```bash
# データセット生成
python data/generate_sample_data.py
```

**2. 依存関係エラー**
```bash
# 必要パッケージのインストール
pip install -r requirements.txt
```

**3. メモリエラー**
```python
# 大きなデータセットの場合、チャンク読み込み
chunk_size = 1000
for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
    # Process chunk
    process_chunk(chunk)
```

## 📚 次のステップ

1. **詳細分析**: `examples/data_analysis_demo.py` を実行
2. **カスタムモデル**: 独自の機械学習手法を実装
3. **実データ適用**: 実車両データでの検証
4. **性能最適化**: ハイパーパラメータチューニング

---

**🎉 これで準備完了です！** データセットを使ってTPMS開発を始めましょう。

質問やフィードバックがあれば、Issueで気軽にお聞かせください。