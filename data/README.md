# TPMS Sample Datasets

このディレクトリには、間接タイヤ空気圧監視システム（Indirect TPMS）の開発・テスト・研究用のサンプルデータセットが含まれています。

## 📊 データセット概要

### データセット構成

| ファイル名 | 説明 | サンプル数 | 用途 |
|-----------|------|-----------|------|
| `raw_sensor_data.csv` | 生センサーデータ | ~25,000 | データ探索、前処理開発 |
| `feature_data.csv` | 特徴量抽出済みデータ | ~5,000 | 機械学習モデル開発 |
| `train_features.csv` | 訓練用特徴量データ | ~3,500 | モデル訓練 |
| `val_features.csv` | 検証用特徴量データ | ~750 | ハイパーパラメータ調整 |
| `test_features.csv` | テスト用特徴量データ | ~750 | 最終性能評価 |
| `dataset_metadata.json` | データセットメタデータ | - | 仕様・設定情報 |
| `dataset_summary.txt` | データセット統計情報 | - | 概要・統計 |

## 🚗 収集シナリオ

### 運転パターン
- **市街地運転** (`city_driving`): 停止・発進・旋回を含む市街地交通
- **高速道路** (`highway_cruising`): 一定速度での高速道路走行
- **駐車操作** (`parking_maneuvers`): 低速での駐車場内操作
- **緊急ブレーキ** (`emergency_braking`): 急制動シナリオ
- **山道走行** (`mountain_roads`): カーブの多い山間部道路
- **渋滞** (`stop_and_go`): ストップ&ゴーの渋滞状況

### タイヤ空気圧条件
- **正常** (`normal`): 全輪適正空気圧 (2.2 bar)
- **単輪低圧** (`fl_low`, `fr_low`, etc.): 1輪のみ低圧 (1.8 bar)
- **単輪重度低圧** (`fl_critical`, etc.): 1輪のみ重度低圧 (1.5 bar)
- **複数輪低圧** (`front_low`, `rear_low`): 複数輪同時低圧
- **徐々の空気漏れ** (`gradual_deflation`): 時間経過による圧力低下
- **過充填** (`overinflated`): 基準以上の空気圧 (2.6 bar)

## 📈 センサーデータ仕様

### 生センサーデータ (`raw_sensor_data.csv`)

| 列名 | データ型 | 単位 | 説明 |
|------|----------|------|------|
| `sample_id` | int | - | サンプル識別子 |
| `timestamp` | float | 秒 | タイムスタンプ |
| `scenario` | string | - | 運転シナリオ |
| `pressure_condition` | string | - | 空気圧状態 |
| `severity` | int | - | 異常度 (0:正常, 1:軽度, 2:重度) |
| `vehicle_speed` | float | km/h | 車両速度 |
| `maneuver` | string | - | 運転操作 (straight/turn/brake) |
| `wheel_speed_*` | float | rad/s | 各輪の回転速度 (FL/FR/RL/RR) |
| `acceleration_*` | float | m/s² | 3軸加速度 (x/y/z) |
| `steering_angle` | float | 度 | ステアリング角度 |
| `brake_pressure` | float | bar | ブレーキ圧力 |
| `engine_load` | float | % | エンジン負荷 |
| `gps_*` | float | 各種 | GPS情報 (緯度/経度/高度/速度) |
| `actual_pressure_*` | float | bar | 実際の空気圧 (FL/FR/RL/RR) |

### 特徴量データ (`feature_data.csv`)

抽出された特徴量（100+個）が含まれます：

#### 車輪半径特徴量
- `radius_mean_*`: 各輪の有効半径平均値
- `radius_std_*`: 各輪の有効半径標準偏差
- `radius_ratio_*_*`: 輪間半径比

#### 振動特徴量
- `acc_mean_*`: 加速度統計量 (x/y/z軸)
- `acc_rms_*`: 加速度実効値
- `acc_spectral_centroid_*`: スペクトル重心
- `acc_*_freq_energy_*`: 周波数帯域エネルギー

#### ヒストグラム特徴量
- `hist_entropy_*`: 各輪速度分布のエントロピー
- `hist_peak_*`: ヒストグラムピーク値
- `hist_asymmetry_*`: 分布の非対称性

#### ARMA特徴量
- `arma_ar_*_*`: 自己回帰パラメータ
- `arma_ma_*_*`: 移動平均パラメータ
- `arma_aic_*`: AIC情報量基準

#### 車両動特性特徴量
- `steering_*`: ステアリング関連統計量
- `brake_*`: ブレーキ使用パターン
- `engine_load_*`: エンジン負荷特性
- `speed_change_rate`: 速度変化率

## 🔧 データセット生成

### 自動生成
```bash
# データセットを自動生成
python data/generate_sample_data.py
```

### カスタム生成
```python
from data.generate_sample_data import TPMSDatasetGenerator

# カスタム設定でデータセット生成
generator = TPMSDatasetGenerator("custom_output_dir")
generator.save_datasets()
```

## 📚 使用例

### 基本的なデータ読み込み
```python
import pandas as pd

# 生センサーデータ読み込み
raw_data = pd.read_csv("data/sample_data/raw_sensor_data.csv")

# 特徴量データ読み込み
feature_data = pd.read_csv("data/sample_data/feature_data.csv")

# 訓練用データ読み込み
train_data = pd.read_csv("data/sample_data/train_features.csv")
```

### データ探索
```python
# データ概要確認
print(f"Raw data shape: {raw_data.shape}")
print(f"Scenarios: {raw_data['scenario'].unique()}")
print(f"Pressure conditions: {raw_data['pressure_condition'].unique()}")

# 空気圧分布確認
import matplotlib.pyplot as plt
raw_data['actual_pressure_fl'].hist(bins=50)
plt.title("Front Left Tire Pressure Distribution")
plt.xlabel("Pressure (bar)")
plt.show()
```

### 機械学習用データ準備
```python
# 特徴量とターゲット分離
feature_columns = [col for col in train_data.columns 
                  if not col.startswith(('scenario', 'pressure_condition', 'severity', 'actual_pressure'))]

X_train = train_data[feature_columns]
y_train = train_data[['actual_pressure_fl', 'actual_pressure_fr', 
                     'actual_pressure_rl', 'actual_pressure_rr']]

# 分類ラベル作成
labels = train_data['severity']
```

## 📋 データ品質情報

### データ完全性
- **欠損値**: なし（シミュレーションデータのため）
- **異常値**: 物理的制約内で生成
- **データ範囲**: 実車両仕様に基づく

### バランス
- **シナリオ**: 各運転パターンで均等なサンプル数
- **圧力条件**: 正常:異常 = 約1:3の比率
- **重要度**: 重度異常の検出性能を重視

### 制限事項
- シミュレーションデータ（実車データではない）
- 環境条件（気温、路面状況）は限定的
- 車両型式は単一モデルを想定

## 🎯 推奨用途

### 開発段階での使用
1. **アルゴリズム開発**: 新しい特徴量抽出手法の検証
2. **モデル比較**: 異なる機械学習手法の性能比較
3. **ハイパーパラメータ調整**: 最適パラメータの探索
4. **ベンチマーク**: システム性能の基準設定

### 研究での使用
1. **論文執筆**: 手法検証用のベンチマークデータ
2. **学会発表**: 比較実験用の標準データセット
3. **教育**: 機械学習・信号処理の教材

## ⚠️ 注意事項

1. **実用化前の検証**: 実車データでの追加検証が必須
2. **安全基準**: 自動車安全基準への適合確認が必要
3. **環境依存性**: 実際の環境条件での動作確認推奨
4. **更新**: 新しい車両技術への対応が必要な場合あり

## 📖 参考文献

- Honda Indirect TPMS methodology
- Latest TPMS research papers (2024-2025)
- Automotive sensor fusion techniques
- Bayesian neural network applications in automotive

---

データセットに関する質問やフィードバックは、Issueまたはディスカッションでお願いします。