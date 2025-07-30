# API Reference - Indirect TPMS

このドキュメントでは、Indirect TPMSシステムの全クラス・関数・メソッドの詳細なAPIリファレンスを提供します。

## 目次

1. [データ処理モジュール](#データ処理モジュール)
2. [機械学習モジュール](#機械学習モジュール)
3. [データ構造](#データ構造)
4. [列挙型](#列挙型)
5. [例外クラス](#例外クラス)

---

## データ処理モジュール

### SensorData

車両センサーデータを格納するデータクラス。

```python
@dataclass
class SensorData:
    timestamp: float
    wheel_speeds: np.ndarray
    accelerations: np.ndarray
    vehicle_speed: float
    steering_angle: float
    brake_pressure: float
    engine_load: float
    gps_data: Dict[str, float]
```

#### パラメータ

| パラメータ | 型 | 説明 |
|------------|----|----|
| `timestamp` | `float` | タイムスタンプ（秒） |
| `wheel_speeds` | `np.ndarray` | 4輪の角速度 [FL, FR, RL, RR] (rad/s) |
| `accelerations` | `np.ndarray` | 3軸加速度 [x, y, z] (m/s²) |
| `vehicle_speed` | `float` | 車両速度 (km/h) |
| `steering_angle` | `float` | ステアリング角度 (度) |
| `brake_pressure` | `float` | ブレーキ圧力 (bar) |
| `engine_load` | `float` | エンジン負荷 (%) |
| `gps_data` | `Dict[str, float]` | GPS情報 (latitude, longitude, altitude, speed) |

#### 例

```python
sensor_data = SensorData(
    timestamp=1234567890.0,
    wheel_speeds=np.array([45.2, 45.1, 45.3, 45.0]),
    accelerations=np.array([0.1, -0.2, -9.81]),
    vehicle_speed=60.0,
    steering_angle=5.2,
    brake_pressure=0.0,
    engine_load=45.5,
    gps_data={'latitude': 35.0, 'longitude': 139.0, 'altitude': 50.0, 'speed': 60.0}
)
```

---

### TPMSDataProcessor

センサーデータの処理と特徴量抽出を行うメインクラス。

```python
class TPMSDataProcessor:
    def __init__(self, window_size: int = 100, sampling_rate: float = 100.0)
```

#### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `window_size` | `int` | `100` | データウィンドウサイズ |
| `sampling_rate` | `float` | `100.0` | サンプリング周波数 (Hz) |

#### 主要メソッド

##### `add_sensor_data(data: SensorData) -> bool`

センサーデータをバッファに追加します。

**パラメータ:**
- `data`: センサーデータ

**戻り値:**
- `bool`: バッファが満杯の場合 `True`

**例:**
```python
processor = TPMSDataProcessor(window_size=50)
is_full = processor.add_sensor_data(sensor_data)
if is_full:
    print("データバッファが満杯になりました")
```

##### `extract_wheel_radius_features() -> Dict[str, np.ndarray]`

ホイール半径関連の特徴量を抽出します（Honda方式ベース）。

**戻り値:**
- `Dict[str, np.ndarray]`: 車輪半径特徴量辞書

**抽出される特徴量:**
- `radius_mean_{wheel}`: 各輪の有効半径平均値
- `radius_std_{wheel}`: 各輪の有効半径標準偏差
- `radius_var_{wheel}`: 各輪の有効半径分散
- `radius_change_rate_{wheel}`: 半径変化率
- `radius_ratio_{wheel1}_{wheel2}`: 輪間半径比

##### `extract_vibration_features() -> Dict[str, float]`

振動特徴量を抽出します（加速度センサーベース）。

**戻り値:**
- `Dict[str, float]`: 振動特徴量辞書

**抽出される特徴量:**
- `acc_mean_{axis}`: 各軸加速度平均値
- `acc_std_{axis}`: 各軸加速度標準偏差
- `acc_rms_{axis}`: 各軸加速度実効値
- `acc_skewness_{axis}`: 各軸加速度歪度
- `acc_kurtosis_{axis}`: 各軸加速度尖度
- `acc_peak_freq_{axis}`: ピーク周波数
- `acc_spectral_centroid_{axis}`: スペクトル重心
- `acc_{freq_band}_freq_energy_{axis}`: 周波数帯域エネルギー

##### `extract_histogram_features() -> Dict[str, float]`

ヒストグラム特徴量を抽出します。

**戻り値:**
- `Dict[str, float]`: ヒストグラム特徴量辞書

**抽出される特徴量:**
- `hist_entropy_{wheel}`: 各輪速度分布のエントロピー
- `hist_peak_{wheel}`: ヒストグラムピーク値
- `hist_peak_pos_{wheel}`: ピーク位置
- `hist_asymmetry_{wheel}`: 分布の非対称性

##### `extract_arma_features(order: Tuple[int, int] = (2, 1)) -> Dict[str, float]`

ARMA（自己回帰移動平均）特徴量を抽出します。

**パラメータ:**
- `order`: ARMA次数 (p, q)

**戻り値:**
- `Dict[str, float]`: ARMA特徴量辞書

**抽出される特徴量:**
- `arma_ar_{i}_{wheel}`: 自己回帰パラメータ
- `arma_ma_{i}_{wheel}`: 移動平均パラメータ
- `arma_aic_{wheel}`: AIC情報量基準
- `arma_bic_{wheel}`: BIC情報量基準

##### `extract_vehicle_dynamics_features() -> Dict[str, float]`

車両動特性特徴量を抽出します。

**戻り値:**
- `Dict[str, float]`: 車両動特性特徴量辞書

**抽出される特徴量:**
- `steering_mean`: ステアリング角度平均
- `steering_std`: ステアリング角度標準偏差
- `steering_change_rate`: ステアリング変化率
- `brake_mean`: ブレーキ圧力平均
- `brake_activation_ratio`: ブレーキ使用率
- `engine_load_mean`: エンジン負荷平均
- `engine_load_std`: エンジン負荷標準偏差
- `speed_change_rate`: 速度変化率
- `acceleration_magnitude`: 加速度ベクトル大きさ

##### `fuse_features() -> Dict[str, float]`

全特徴量を統合します。

**戻り値:**
- `Dict[str, float]`: 統合特徴量辞書

**例:**
```python
features = processor.fuse_features()
print(f"抽出された特徴量数: {len(features)}")
```

##### `calibrate_baseline() -> bool`

正常状態のベースライン特徴量を学習します。

**戻り値:**
- `bool`: 校正成功時 `True`

**例:**
```python
# 正常状態でデータ収集後
success = processor.calibrate_baseline()
if success:
    print("ベースライン校正完了")
```

##### `get_deviation_features() -> Dict[str, float]`

ベースラインからの偏差特徴量を計算します。

**戻り値:**
- `Dict[str, float]`: 偏差特徴量辞書

**前提条件:**
- `calibrate_baseline()` が事前に実行されている必要があります

---

### TPMSDataSimulator

TPMS開発用のシミュレーションデータ生成クラス。

```python
class TPMSDataSimulator:
    def __init__(self, nominal_pressure: float = 2.2)
```

#### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `nominal_pressure` | `float` | `2.2` | 正常空気圧 (bar) |

#### メソッド

##### `generate_sensor_data(pressures: List[float], vehicle_speed: float = 50.0, maneuver_type: str = 'straight') -> SensorData`

シミュレーション用センサーデータを生成します。

**パラメータ:**
- `pressures`: 4輪の空気圧 [FL, FR, RL, RR] (bar)
- `vehicle_speed`: 車速 (km/h)
- `maneuver_type`: 運転パターン ('straight', 'turn', 'brake')

**戻り値:**
- `SensorData`: 生成されたセンサーデータ

**例:**
```python
simulator = TPMSDataSimulator(nominal_pressure=2.2)
data = simulator.generate_sensor_data(
    pressures=[1.8, 2.2, 2.2, 2.2],  # 前左輪低圧
    vehicle_speed=60.0,
    maneuver_type='straight'
)
```

---

## 機械学習モジュール

### BayesianNeuralNetwork

不確実性を考慮したベイジアンニューラルネットワーク。

```python
class BayesianNeuralNetwork:
    def __init__(self, input_dim: int, hidden_units: List[int] = [64, 32, 16])
```

#### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `input_dim` | `int` | - | 入力特徴量次元数 |
| `hidden_units` | `List[int]` | `[64, 32, 16]` | 隠れ層のユニット数リスト |

#### メソッド

##### `compile_model()`

ベイジアンニューラルネットワークをコンパイルします。

##### `train(X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]`

モデルを訓練します。

**パラメータ:**
- `X`: 入力特徴量 [n_samples, n_features]
- `y`: 目標圧力値 [n_samples, 4]
- `validation_split`: 検証データ割合
- `epochs`: エポック数
- `batch_size`: バッチサイズ

**戻り値:**
- `Dict[str, List[float]]`: 訓練履歴

##### `predict_with_uncertainty(X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]`

不確実性付きで予測を行います。

**パラメータ:**
- `X`: 入力特徴量
- `n_samples`: モンテカルロサンプル数

**戻り値:**
- `Tuple[np.ndarray, np.ndarray]`: (予測平均値, 予測標準偏差)

**例:**
```python
bnn = BayesianNeuralNetwork(input_dim=100)
bnn.compile_model()
history = bnn.train(X_train, y_train, epochs=50)

mean_pred, std_pred = bnn.predict_with_uncertainty(X_test)
print(f"予測圧力: {mean_pred[0]} ± {std_pred[0]}")
```

---

### LazyClassifierEnsemble

K-NN系分類器のアンサンブル。

```python
class LazyClassifierEnsemble:
    def __init__(self)
```

#### メソッド

##### `train(X: np.ndarray, y: np.ndarray)`

分類器を訓練します。

**パラメータ:**
- `X`: 入力特徴量
- `y`: 分類ラベル (0: 正常, 1: 軽度異常, 2: 重度異常)

##### `predict_proba(X: np.ndarray) -> np.ndarray`

クラス確率を予測します（アンサンブル平均）。

**パラメータ:**
- `X`: 入力特徴量

**戻り値:**
- `np.ndarray`: クラス確率 [n_samples, 3]

##### `predict(X: np.ndarray) -> np.ndarray`

予測ラベルを返します。

**パラメータ:**
- `X`: 入力特徴量

**戻り値:**
- `np.ndarray`: 予測ラベル

---

### TPMSMLEstimator

統合TPMS機械学習推定器。

```python
class TPMSMLEstimator:
    def __init__(self, pressure_thresholds: Tuple[float, float] = (1.8, 1.5), confidence_threshold: float = 0.7)
```

#### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `pressure_thresholds` | `Tuple[float, float]` | `(1.8, 1.5)` | (軽度警告, 重度警告) 圧力閾値 (bar) |
| `confidence_threshold` | `float` | `0.7` | 信頼度閾値 |

#### メソッド

##### `add_training_data(features: Dict[str, float], pressures: List[float])`

訓練データを追加します。

**パラメータ:**
- `features`: 特徴量辞書
- `pressures`: 4輪の実際の空気圧 (bar)

##### `train_models(test_size: float = 0.2) -> bool`

全モデルを訓練します。

**パラメータ:**
- `test_size`: テストデータ割合

**戻り値:**
- `bool`: 訓練成功時 `True`

**例:**
```python
estimator = TPMSMLEstimator()

# 訓練データ追加
for features, pressures in training_data:
    estimator.add_training_data(features, pressures)

# モデル訓練
if estimator.train_models():
    print("訓練完了")
```

##### `predict(features: Dict[str, float]) -> TPMSPrediction`

リアルタイム推定を実行します。

**パラメータ:**
- `features`: 特徴量辞書

**戻り値:**
- `TPMSPrediction`: 推定結果

**例:**
```python
prediction = estimator.predict(features)
print(f"推定圧力: {prediction.pressures}")
print(f"警告レベル: {prediction.alert_level.name}")
```

##### `save_models(filepath: str)`

訓練済みモデルを保存します。

**パラメータ:**
- `filepath`: 保存先ファイルパス

##### `load_models(filepath: str)`

訓練済みモデルを読み込みます。

**パラメータ:**
- `filepath`: モデルファイルパス

---

### TPMSRealTimeEstimator

リアルタイムTPMS推定システム。

```python
class TPMSRealTimeEstimator:
    def __init__(self, estimator: TPMSMLEstimator)
```

#### コンストラクタ

| パラメータ | 型 | 説明 |
|------------|----|----|
| `estimator` | `TPMSMLEstimator` | 訓練済みML推定器 |

#### メソッド

##### `process_features(features: Dict[str, float]) -> TPMSPrediction`

特徴量を処理し推定を実行します。

**パラメータ:**
- `features`: 特徴量辞書

**戻り値:**
- `TPMSPrediction`: 推定結果

##### `get_system_status() -> Dict[str, Union[str, float, List]]`

システム状態を取得します。

**戻り値:**
- `Dict`: システム状態情報

**戻り値に含まれる情報:**
- `status`: 現在の状態
- `latest_pressures`: 最新の推定圧力
- `confidence`: 信頼度
- `recent_alerts_count`: 最近のアラート数
- `system_uptime`: システム稼働時間

**例:**
```python
rt_estimator = TPMSRealTimeEstimator(estimator)
prediction = rt_estimator.process_features(features)
status = rt_estimator.get_system_status()
print(f"システム状態: {status}")
```

---

## データ構造

### TPMSPrediction

TPMS推定結果を格納するデータクラス。

```python
@dataclass
class TPMSPrediction:
    pressures: np.ndarray
    uncertainties: np.ndarray
    confidence: float
    alert_level: TPMSAlertLevel
    affected_wheels: List[str]
    timestamp: float
```

#### フィールド

| フィールド | 型 | 説明 |
|------------|----|----|
| `pressures` | `np.ndarray` | 4輪の推定圧力 (bar) |
| `uncertainties` | `np.ndarray` | 推定不確実性 |
| `confidence` | `float` | 全体信頼度 [0-1] |
| `alert_level` | `TPMSAlertLevel` | 警告レベル |
| `affected_wheels` | `List[str]` | 異常検知された車輪 |
| `timestamp` | `float` | タイムスタンプ |

---

## 列挙型

### TPMSAlertLevel

TPMS警告レベルを定義する列挙型。

```python
class TPMSAlertLevel(Enum):
    NORMAL = 0
    LOW_PRESSURE = 1
    CRITICAL_PRESSURE = 2
    SENSOR_FAULT = 3
```

#### 値

| 値 | 説明 |
|----|----|
| `NORMAL` | 正常状態 |
| `LOW_PRESSURE` | 低圧警告 |
| `CRITICAL_PRESSURE` | 重大低圧警告 |
| `SENSOR_FAULT` | センサー故障 |

**例:**
```python
if prediction.alert_level == TPMSAlertLevel.CRITICAL_PRESSURE:
    print("緊急対応が必要です！")
```

---

## 例外クラス

### TPMSException

TPMS関連の基底例外クラス。

```python
class TPMSException(Exception):
    """TPMS関連の基底例外"""
    pass
```

### InsufficientDataException

データ不足例外。

```python
class InsufficientDataException(TPMSException):
    """訓練データが不足している場合の例外"""
    pass
```

### ModelNotTrainedException

未訓練モデル例外。

```python
class ModelNotTrainedException(TPMSException):
    """モデルが訓練されていない場合の例外"""
    pass
```

### SensorDataException

センサーデータ例外。

```python
class SensorDataException(TPMSException):
    """センサーデータに問題がある場合の例外"""
    pass
```

**例外処理の例:**
```python
try:
    prediction = estimator.predict(features)
except ModelNotTrainedException:
    print("モデルを先に訓練してください")
except SensorDataException as e:
    print(f"センサーデータエラー: {e}")
except TPMSException as e:
    print(f"TPMS エラー: {e}")
```

---

## 型ヒント

システム全体で使用される型ヒント定義。

```python
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# 型エイリアス
FeatureDict = Dict[str, float]
PressureArray = np.ndarray  # shape: (4,)
UncertaintyArray = np.ndarray  # shape: (4,)
WheelNames = List[str]  # ['FL', 'FR', 'RL', 'RR']
```

---

## 設定クラス

### TPMSConfig

システム設定を管理するクラス。

```python
class TPMSConfig:
    """TPMS システム設定"""
    
    # データ処理設定
    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_SAMPLING_RATE = 100.0
    
    # 圧力閾値
    LOW_PRESSURE_THRESHOLD = 1.8  # bar
    CRITICAL_PRESSURE_THRESHOLD = 1.5  # bar
    
    # ML設定
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    BNN_HIDDEN_UNITS = [64, 32, 16]
    BNN_EPOCHS = 100
    
    # アラート設定
    ALERT_HISTORY_SIZE = 1000
    
    @classmethod
    def from_file(cls, config_path: str):
        """設定ファイルから読み込み"""
        # YAML/JSON設定ファイル読み込み実装
        pass
    
    @classmethod
    def to_file(cls, config_path: str):
        """設定ファイルに保存"""
        # YAML/JSON設定ファイル保存実装
        pass
```

---

## ユーティリティ関数

### データ変換

```python
def pressure_to_label(pressure: float, 
                     low_threshold: float = 1.8, 
                     critical_threshold: float = 1.5) -> int:
    """圧力値をラベルに変換"""
    if pressure < critical_threshold:
        return 2  # Critical
    elif pressure < low_threshold:
        return 1  # Low
    else:
        return 0  # Normal

def normalize_features(features: FeatureDict) -> FeatureDict:
    """特徴量正規化"""
    # 実装はTPMSDataProcessorの内部メソッドを参照
    pass

def validate_sensor_data(data: SensorData) -> bool:
    """センサーデータの妥当性検証"""
    # 実装例
    if len(data.wheel_speeds) != 4:
        return False
    if len(data.accelerations) != 3:
        return False
    if data.vehicle_speed < 0:
        return False
    return True
```

---

## パフォーマンス指標

### メトリクス計算

```python
def calculate_tpms_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          threshold: float = 1.8) -> Dict[str, float]:
    """TPMS性能メトリクス計算"""
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # 回帰メトリクス
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 異常検知メトリクス
    true_anomaly = y_true < threshold
    pred_anomaly = y_pred < threshold
    
    tp = np.sum(true_anomaly & pred_anomaly)
    fp = np.sum(~true_anomaly & pred_anomaly)
    tn = np.sum(~true_anomaly & ~pred_anomaly)
    fn = np.sum(true_anomaly & ~pred_anomaly)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mae': mae,
        'r2': r2,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': (tp + tn) / (tp + fp + tn + fn)
    }
```

---

このAPIリファレンスを使用して、Indirect TPMSシステムの詳細な実装や拡張を行ってください。各メソッドの詳細な使用例については、`examples/` ディレクトリのサンプルコードも参照してください。