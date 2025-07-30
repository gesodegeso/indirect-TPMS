# Usage Manual - Indirect TPMS

このマニュアルでは、Indirect TPMSシステムの詳細な使用方法を説明します。

## 目次

1. [システム概要](#システム概要)
2. [基本的な使用方法](#基本的な使用方法)
3. [高度な設定](#高度な設定)
4. [リアルタイム監視](#リアルタイム監視)
5. [カスタマイズ](#カスタマイズ)
6. [トラブルシューティング](#トラブルシューティング)

## システム概要

Indirect TPMSは、直接的な圧力センサーを使用せずに、既存の車両センサーデータから機械学習によってタイヤ空気圧を推定するシステムです。

### 主要コンポーネント

- **TPMSDataProcessor**: センサーデータの処理と特徴量抽出
- **TPMSMLEstimator**: 機械学習による圧力推定
- **TPMSRealTimeEstimator**: リアルタイム監視システム

## 基本的な使用方法

### 1. システム初期化

```python
from src.indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator
from src.ml_tpms_module import TPMSMLEstimator, TPMSRealTimeEstimator

# 基本設定
processor = TPMSDataProcessor(
    window_size=100,        # データウィンドウサイズ
    sampling_rate=100.0     # サンプリング周波数 [Hz]
)

estimator = TPMSMLEstimator(
    pressure_thresholds=(1.8, 1.5),  # (警告, 重大警告) [bar]
    confidence_threshold=0.7          # 信頼度閾値
)
```

### 2. 訓練データの準備

#### シミュレーションデータを使用する場合

```python
simulator = TPMSDataSimulator(nominal_pressure=2.2)

# 様々な条件でデータ生成
training_scenarios = [
    ([2.2, 2.2, 2.2, 2.2], 60, 'straight', 50),  # 正常状態
    ([1.8, 2.2, 2.2, 2.2], 50, 'straight', 30),  # 低圧状態
    ([1.5, 2.2, 2.2, 2.2], 40, 'turn', 20),     # 重大低圧
]

for pressures, speed, maneuver, n_samples in training_scenarios:
    for i in range(n_samples):
        data = simulator.generate_sensor_data(pressures, speed, maneuver)
        processor.add_sensor_data(data)
        
        if len(processor.data_buffer) >= processor.window_size:
            features = processor.fuse_features()
            if features:
                estimator.add_training_data(features, pressures)
```

#### 実データを使用する場合

```python
import pandas as pd

# CSVファイルから実データ読み込み
real_data = pd.read_csv("path/to/real_sensor_data.csv")

for _, row in real_data.iterrows():
    # SensorDataオブジェクト作成
    from src.indirect_tpms_module import SensorData
    
    sensor_data = SensorData(
        timestamp=row['timestamp'],
        wheel_speeds=np.array([row['wheel_speed_fl'], row['wheel_speed_fr'], 
                              row['wheel_speed_rl'], row['wheel_speed_rr']]),
        accelerations=np.array([row['acc_x'], row['acc_y'], row['acc_z']]),
        vehicle_speed=row['vehicle_speed'],
        steering_angle=row['steering_angle'],
        brake_pressure=row['brake_pressure'],
        engine_load=row['engine_load'],
        gps_data={
            'latitude': row['gps_lat'],
            'longitude': row['gps_lon'],
            'altitude': row['gps_alt'],
            'speed': row['gps_speed']
        }
    )
    
    processor.add_sensor_data(sensor_data)
    
    if len(processor.data_buffer) >= processor.window_size:
        features = processor.fuse_features()
        actual_pressures = [row['pressure_fl'], row['pressure_fr'], 
                           row['pressure_rl'], row['pressure_rr']]
        
        if features:
            estimator.add_training_data(features, actual_pressures)
```

### 3. モデル訓練

```python
# モデル訓練実行
if estimator.train_models():
    print("訓練完了")
    
    # モデル保存
    estimator.save_models("tpms_model.pkl")
else:
    print("訓練失敗")
```

### 4. リアルタイム推定

```python
# リアルタイム推定器初期化
rt_estimator = TPMSRealTimeEstimator(estimator)

# リアルタイムデータ処理ループ
while True:
    # 新しいセンサーデータ取得（実際のCAN通信など）
    new_sensor_data = get_current_sensor_data()
    
    processor.add_sensor_data(new_sensor_data)
    
    # 特徴量抽出・推定
    features = processor.fuse_features()
    if features:
        prediction = rt_estimator.process_features(features)
        
        # 結果表示
        print(f"推定圧力: {prediction.pressures}")
        print(f"信頼度: {prediction.confidence:.2f}")
        print(f"警告レベル: {prediction.alert_level.name}")
        
        if prediction.affected_wheels:
            print(f"異常車輪: {prediction.affected_wheels}")
    
    time.sleep(0.1)  # 100ms間隔
```

## 高度な設定

### カスタム特徴量抽出

```python
class CustomTPMSProcessor(TPMSDataProcessor):
    """カスタム特徴量抽出器"""
    
    def extract_custom_features(self) -> Dict[str, float]:
        """独自の特徴量抽出"""
        if len(self.data_buffer) < self.window_size:
            return {}
        
        # 独自の処理ロジック
        wheel_speeds = np.array([d.wheel_speeds for d in self.data_buffer])
        
        features = {}
        # 例: 車輪速度の高次統計量
        for i in range(4):
            wheel_name = ['FL', 'FR', 'RL', 'RR'][i]
            speed_data = wheel_speeds[:, i]
            
            # 歪度・尖度
            features[f'skewness_{wheel_name}'] = stats.skew(speed_data)
            features[f'kurtosis_{wheel_name}'] = stats.kurtosis(speed_data)
            
            # 自己相関
            if len(speed_data) > 10:
                autocorr = np.corrcoef(speed_data[:-1], speed_data[1:])[0, 1]
                features[f'autocorr_{wheel_name}'] = autocorr if not np.isnan(autocorr) else 0
        
        return features
    
    def fuse_features(self) -> Dict[str, float]:
        """特徴量融合（独自特徴量を追加）"""
        # 基本特徴量
        base_features = super().fuse_features()
        
        # 独自特徴量
        custom_features = self.extract_custom_features()
        
        # 統合
        base_features.update(custom_features)
        return base_features

# 使用方法
custom_processor = CustomTPMSProcessor(window_size=100)
```

### ハイパーパラメータ調整

```python
from sklearn.model_selection import GridSearchCV

# BNNパラメータ調整例
class TunableTPMSEstimator(TPMSMLEstimator):
    """調整可能なTPMS推定器"""
    
    def tune_hyperparameters(self, X, y):
        """ハイパーパラメータ自動調整"""
        # Lazy分類器のパラメータ調整
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        best_score = 0
        best_params = {}
        
        # グリッドサーチ
        for n_neighbors in param_grid['n_neighbors']:
            for weights in param_grid['weights']:
                for algorithm in param_grid['algorithm']:
                    knn = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm
                    )
                    
                    # 分類ラベル作成
                    y_class = self._create_pressure_labels(y)
                    
                    # クロスバリデーション
                    scores = cross_val_score(knn, X, y_class, cv=5)
                    score = np.mean(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_neighbors': n_neighbors,
                            'weights': weights,
                            'algorithm': algorithm
                        }
        
        print(f"最適パラメータ: {best_params}")
        print(f"最高スコア: {best_score:.3f}")
        
        return best_params

# 使用例
tunable_estimator = TunableTPMSEstimator()
# ... 訓練データ準備 ...
best_params = tunable_estimator.tune_hyperparameters(X_train, y_train)
```

## リアルタイム監視

### CANバス統合

```python
import can  # python-can library

class CANTPMSInterface:
    """CAN通信インターフェース"""
    
    def __init__(self, can_interface='socketcan', channel='can0'):
        """CAN接続初期化"""
        self.bus = can.interface.Bus(interface=can_interface, channel=channel)
        self.processor = TPMSDataProcessor()
        self.estimator = None  # 訓練済みモデルをロード
        
    def load_trained_model(self, model_path):
        """訓練済みモデル読み込み"""
        self.estimator = TPMSMLEstimator()
        self.estimator.load_models(model_path)
        self.rt_estimator = TPMSRealTimeEstimator(self.estimator)
    
    def parse_can_message(self, message):
        """CANメッセージ解析"""
        sensor_data = None
        
        # メッセージID別処理
        if message.arbitration_id == 0x123:  # 車輪速度
            wheel_speeds = self._parse_wheel_speeds(message.data)
            # ... 他のデータと組み合わせてSensorDataオブジェクト作成
            
        elif message.arbitration_id == 0x124:  # 加速度
            accelerations = self._parse_accelerations(message.data)
            # ...
            
        # 完全なSensorDataが揃ったら返す
        return sensor_data
    
    def run_monitoring(self):
        """リアルタイム監視実行"""
        if not self.estimator:
            raise ValueError("訓練済みモデルが読み込まれていません")
        
        print("TPMS監視開始...")
        
        try:
            while True:
                # CANメッセージ受信
                message = self.bus.recv(timeout=1.0)
                
                if message:
                    sensor_data = self.parse_can_message(message)
                    
                    if sensor_data:
                        self.processor.add_sensor_data(sensor_data)
                        
                        # 推定実行
                        features = self.processor.fuse_features()
                        if features:
                            prediction = self.rt_estimator.process_features(features)
                            
                            # 異常時の処理
                            if prediction.alert_level != TPMSAlertLevel.NORMAL:
                                self._handle_alert(prediction)
                
        except KeyboardInterrupt:
            print("監視終了")
        finally:
            self.bus.shutdown()
    
    def _handle_alert(self, prediction):
        """アラート処理"""
        alert_msg = f"TPMS ALERT: {prediction.alert_level.name}"
        if prediction.affected_wheels:
            alert_msg += f" - Wheels: {', '.join(prediction.affected_wheels)}"
        
        print(alert_msg)
        
        # ログ記録、外部システム通知など
        self._log_alert(prediction)
        self._notify_external_system(prediction)

# 使用例
can_interface = CANTPMSInterface()
can_interface.load_trained_model("tpms_model.pkl")
can_interface.run_monitoring()
```

### Webダッシュボード

```python
from flask import Flask, render_template, jsonify
import threading
import time

class TPMSWebDashboard:
    """TPMS Webダッシュボード"""
    
    def __init__(self, rt_estimator):
        self.app = Flask(__name__)
        self.rt_estimator = rt_estimator
        self.latest_data = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """ルート設定"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/current_status')
        def current_status():
            """現在の状態をJSON形式で返す"""
            return jsonify(self.latest_data)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """最近のアラート履歴"""
            alerts = self.rt_estimator.alert_history[-10:]  # 最新10件
            return jsonify(alerts)
        
        @self.app.route('/api/system_status')
        def system_status():
            """システム状態"""
            status = self.rt_estimator.get_system_status()
            return jsonify(status)
    
    def update_data(self, prediction):
        """データ更新"""
        self.latest_data = {
            'timestamp': prediction.timestamp,
            'pressures': {
                'FL': float(prediction.pressures[0]),
                'FR': float(prediction.pressures[1]),
                'RL': float(prediction.pressures[2]),
                'RR': float(prediction.pressures[3])
            },
            'uncertainties': {
                'FL': float(prediction.uncertainties[0]),
                'FR': float(prediction.uncertainties[1]),
                'RL': float(prediction.uncertainties[2]),
                'RR': float(prediction.uncertainties[3])
            },
            'confidence': float(prediction.confidence),
            'alert_level': prediction.alert_level.name,
            'affected_wheels': prediction.affected_wheels
        }
    
    def run(self, host='0.0.0.0', port=5000):
        """ダッシュボード起動"""
        self.app.run(host=host, port=port, debug=False)

# 使用例
dashboard = TPMSWebDashboard(rt_estimator)

# バックグラウンドでダッシュボード起動
dashboard_thread = threading.Thread(target=dashboard.run)
dashboard_thread.daemon = True
dashboard_thread.start()

# メインループでデータ更新
while True:
    # ... データ取得・推定 ...
    prediction = rt_estimator.process_features(features)
    dashboard.update_data(prediction)
    time.sleep(0.1)
```

## カスタマイズ

### 独自の機械学習モデル

```python
import tensorflow as tf
from tensorflow import keras

class CustomTPMSModel:
    """カスタムTPMSモデル"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """独自のニューラルネットワーク構築"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # 4輪の圧力出力
            keras.layers.Dense(4, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X, y, epochs=100, validation_split=0.2):
        """モデル訓練"""
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """予測"""
        return self.model.predict(X)

# TPMSMLEstimatorに統合
class CustomTPMSMLEstimator(TPMSMLEstimator):
    """カスタムモデル対応推定器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_model = None
    
    def train_custom_model(self, X, y):
        """カスタムモデル訓練"""
        self.custom_model = CustomTPMSModel(input_dim=X.shape[1])
        history = self.custom_model.train(X, y)
        return history
    
    def predict_with_custom_model(self, features):
        """カスタムモデルで予測"""
        if self.custom_model is None:
            raise ValueError("カスタムモデルが訓練されていません")
        
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.feature_scaler.transform(feature_vector)
        
        pressures = self.custom_model.predict(X_scaled)[0]
        
        # 簡単な信頼度計算（実際はより複雑な手法を使用）
        confidence = 0.8  # プレースホルダー
        
        return pressures, confidence
```

### アラートシステム

```python
import smtplib
from email.mime.text import MIMEText
import requests

class TPMSAlertManager:
    """TPMS アラート管理システム"""
    
    def __init__(self, config):
        """
        Args:
            config: アラート設定辞書
            {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'your_email@gmail.com',
                    'password': 'your_password',
                    'recipients': ['admin@company.com']
                },
                'slack': {
                    'webhook_url': 'https://hooks.slack.com/...'
                },
                'sms': {
                    'api_key': 'your_sms_api_key',
                    'phone_numbers': ['+81-90-1234-5678']
                }
            }
        """
        self.config = config
    
    def send_alert(self, prediction):
        """アラート送信"""
        if prediction.alert_level == TPMSAlertLevel.NORMAL:
            return
        
        message = self._format_alert_message(prediction)
        
        # 重要度に応じて送信方法を決定
        if prediction.alert_level == TPMSAlertLevel.CRITICAL_PRESSURE:
            self._send_email(message)
            self._send_slack(message)
            self._send_sms(message)
        elif prediction.alert_level == TPMSAlertLevel.LOW_PRESSURE:
            self._send_email(message)
            self._send_slack(message)
        else:  # SENSOR_FAULT
            self._send_email(message)
    
    def _format_alert_message(self, prediction):
        """アラートメッセージ作成"""
        timestamp = datetime.fromtimestamp(prediction.timestamp)
        
        message = f"""
🚨 TPMS ALERT 🚨

時刻: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
警告レベル: {prediction.alert_level.name}
信頼度: {prediction.confidence:.2f}

推定圧力:
  前左 (FL): {prediction.pressures[0]:.2f} bar
  前右 (FR): {prediction.pressures[1]:.2f} bar
  後左 (RL): {prediction.pressures[2]:.2f} bar
  後右 (RR): {prediction.pressures[3]:.2f} bar

異常車輪: {', '.join(prediction.affected_wheels) if prediction.affected_wheels else 'なし'}

対応が必要です。
        """.strip()
        
        return message
    
    def _send_email(self, message):
        """メール送信"""
        if 'email' not in self.config:
            return
        
        email_config = self.config['email']
        
        try:
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = 'TPMS Alert - 緊急対応要'
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            print("メールアラート送信完了")
            
        except Exception as e:
            print(f"メール送信エラー: {e}")
    
    def _send_slack(self, message):
        """Slack通知"""
        if 'slack' not in self.config:
            return
        
        try:
            payload = {
                'text': message,
                'username': 'TPMS Bot',
                'icon_emoji': ':warning:'
            }
            
            response = requests.post(self.config['slack']['webhook_url'], json=payload)
            response.raise_for_status()
            
            print("Slackアラート送信完了")
            
        except Exception as e:
            print(f"Slack送信エラー: {e}")
    
    def _send_sms(self, message):
        """SMS送信（プレースホルダー）"""
        # 実際のSMS APIサービス（Twilio、AWS SNSなど）に応じて実装
        print(f"SMS送信: {message[:100]}...")

# 使用例
alert_config = {
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'tpms@company.com',
        'password': 'your_password',
        'recipients': ['admin@company.com', 'maintenance@company.com']
    },
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    }
}

alert_manager = TPMSAlertManager(alert_config)

# リアルタイム推定器に統合
class AlertEnabledRealTimeEstimator(TPMSRealTimeEstimator):
    """アラート機能付きリアルタイム推定器"""
    
    def __init__(self, estimator, alert_manager):
        super().__init__(estimator)
        self.alert_manager = alert_manager
    
    def _handle_alerts(self, prediction):
        """アラート処理をオーバーライド"""
        super()._handle_alerts(prediction)
        
        # 外部アラート送信
        self.alert_manager.send_alert(prediction)
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. 訓練データ不足
```python
# 症状: モデル訓練が失敗する
# 原因: 訓練サンプル数が不足

# 解決方法: データ拡張
def augment_training_data(processor, base_data, augmentation_factor=3):
    """訓練データ拡張"""
    augmented_data = []
    
    for data_point in base_data:
        augmented_data.append(data_point)  # 元データ
        
        # ノイズ追加バリエーション
        for i in range(augmentation_factor):
            noisy_data = data_point.copy()
            
            # 小さなノイズを追加
            noisy_data.wheel_speeds += np.random.normal(0, 0.01, 4)
            noisy_data.accelerations += np.random.normal(0, 0.1, 3)
            noisy_data.vehicle_speed += np.random.normal(0, 0.5)
            
            augmented_data.append(noisy_data)
    
    return augmented_data
```

#### 2. 特徴量の数値不安定性
```python
# 症状: NaN値やInf値が発生
# 解決方法: ロバスト前処理

def robust_feature_processing(features):
    """ロバストな特徴量処理"""
    processed_features = {}
    
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            processed_features[key] = 0.0  # デフォルト値
        elif abs(value) > 1e6:  # 異常に大きな値
            processed_features[key] = np.sign(value) * 1e6
        else:
            processed_features[key] = float(value)
    
    return processed_features
```

#### 3. リアルタイム性能問題
```python
# 症状: 処理遅延が発生
# 解決方法: 最適化

import cProfile
import time

def profile_tpms_performance():
    """TPMS性能プロファイリング"""
    
    def timed_function(func):
        """実行時間測定デコレータ"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__}: {(end_time - start_time) * 1000:.2f} ms")
            return result
        return wrapper
    
    # 重要な関数に適用
    TPMSDataProcessor.fuse_features = timed_function(TPMSDataProcessor.fuse_features)
    TPMSMLEstimator.predict = timed_function(TPMSMLEstimator.predict)

# 使用例
profile_tpms_performance()
```

#### 4. メモリ使用量増加
```python
# 症状: 長時間実行でメモリリーク
# 解決方法: メモリ管理

import gc
import psutil

class MemoryAwareTPMSProcessor(TPMSDataProcessor):
    """メモリ効率的なTPMSプロセッサ"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_check_interval = 1000
        self.sample_count = 0
    
    def add_sensor_data(self, data):
        """メモリチェック付きデータ追加"""
        result = super().add_sensor_data(data)
        
        self.sample_count += 1
        
        # 定期的にメモリチェック
        if self.sample_count % self.memory_check_interval == 0:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage > 500:  # 500MB以上
                print(f"メモリ使用量: {memory_usage:.1f} MB - ガベージコレクション実行")
                gc.collect()
        
        return result
```

### ログ設定

```python
import logging
import logging.handlers

def setup_tpms_logging():
    """TPMS用ログ設定"""
    
    # ロガー作成
    logger = logging.getLogger('tpms')
    logger.setLevel(logging.INFO)
    
    # フォーマット設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ファイルハンドラ（ローテーション付き）
    file_handler = logging.handlers.RotatingFileHandler(
        'tpms.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 使用例
logger = setup_tpms_logging()
logger.info("TPMS システム起動")
```

---

このマニュアルを参考に、Indirect TPMSシステムを効果的に活用してください。追加の質問やサポートが必要な場合は、プロジェクトのIssueページでお気軽にお尋ねください。