"""
機械学習ベース間接TPMS推定モジュール
最新研究に基づくベイジアンニューラルネットワークと統合分類器
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union
import pickle
import joblib
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TPMSAlertLevel(Enum):
    """TPMS警告レベル"""
    NORMAL = 0
    LOW_PRESSURE = 1
    CRITICAL_PRESSURE = 2
    SENSOR_FAULT = 3

@dataclass
class TPMSPrediction:
    """TPMS推定結果"""
    pressures: np.ndarray  # 4輪の推定圧力 [bar]
    uncertainties: np.ndarray  # 推定不確実性
    confidence: float  # 全体信頼度 [0-1]
    alert_level: TPMSAlertLevel
    affected_wheels: List[str]  # 異常検知された車輪
    timestamp: float

class BayesianNeuralNetwork:
    """ベイジアンニューラルネットワーク実装"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [64, 32, 16]):
        """
        Args:
            input_dim: 入力特徴量次元数
            hidden_units: 隠れ層のユニット数リスト
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.model = None
        self.is_trained = False
        
    def _build_model(self):
        """ベイジアンニューラルネットワーク構築"""
        # 変分推論のための prior/posterior 分布設定
        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            return tfp.distributions.Normal(
                loc=tf.zeros(shape, dtype=dtype), 
                scale=tf.ones(shape, dtype=dtype)
            )
        
        def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
            n = int(np.prod(shape))
            return tfp.distributions.MultivariateNormalDiag(
                loc=add_variable_fn(
                    name=name + '_loc',
                    shape=shape,
                    initializer='zeros',
                    trainable=trainable
                ),
                scale_diag=tfp.util.TransformedVariable(
                    add_variable_fn(
                        name=name + '_scale',
                        shape=shape,
                        initializer='ones',
                        trainable=trainable
                    ),
                    bijector=tfp.bijectors.Softplus()
                )
            )
        
        # モデル構築
        model = keras.Sequential()
        
        # 入力層
        model.add(keras.layers.Input(shape=(self.input_dim,)))
        
        # ベイジアン隠れ層
        for units in self.hidden_units:
            model.add(tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=1.0 / 1000,  # KL divergence weight
                activation='relu'
            ))
            model.add(keras.layers.Dropout(0.1))
        
        # 出力層（4輪の圧力推定）
        model.add(tfp.layers.DenseVariational(
            units=4,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1.0 / 1000,
            activation='linear'
        ))
        
        return model
    
    def negative_log_likelihood(self, y_true, y_pred):
        """負の対数尤度損失関数"""
        return -y_pred.log_prob(y_true)
    
    def compile_model(self):
        """モデルコンパイル"""
        if self.model is None:
            self.model = self._build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.negative_log_likelihood,
            metrics=['mae']
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, 
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, List[float]]:
        """
        モデル訓練
        
        Args:
            X: 入力特徴量 [n_samples, n_features]
            y: 目標圧力値 [n_samples, 4]
            validation_split: 検証データ割合
            epochs: エポック数
            batch_size: バッチサイズ
        
        Returns:
            訓練履歴
        """
        if self.model is None:
            self.compile_model()
        
        # Early stopping設定
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 訓練実行
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.is_trained = True
        return history.history
    
    def predict_with_uncertainty(self, X: np.ndarray, 
                                n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        不確実性付き予測
        
        Args:
            X: 入力特徴量
            n_samples: モンテカルロサンプル数
        
        Returns:
            (予測平均値, 予測標準偏差)
        """
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        # モンテカルロサンプリング
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # 平均と標準偏差計算
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

class LazyClassifierEnsemble:
    """Lazy-based分類器アンサンブル"""
    
    def __init__(self):
        """複数のLazy分類器を統合"""
        self.classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'knn_uniform': KNeighborsClassifier(n_neighbors=7, weights='uniform'),
            'knn_large': KNeighborsClassifier(n_neighbors=15, weights='distance')
        }
        self.is_trained = False
        self.scaler = StandardScaler()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        分類器訓練
        
        Args:
            X: 入力特徴量
            y: 分類ラベル (0: 正常, 1: 軽度異常, 2: 重度異常)
        """
        # データ正規化
        X_scaled = self.scaler.fit_transform(X)
        
        # 各分類器訓練
        for name, classifier in self.classifiers.items():
            classifier.fit(X_scaled, y)
        
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測（アンサンブル平均）"""
        if not self.is_trained:
            raise ValueError("分類器が訓練されていません")
        
        X_scaled = self.scaler.transform(X)
        
        # 各分類器の予測確率を平均
        proba_sum = np.zeros((X.shape[0], 3))  # 3クラス分類
        
        for classifier in self.classifiers.values():
            proba_sum += classifier.predict_proba(X_scaled)
        
        return proba_sum / len(self.classifiers)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測ラベル"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class TPMSMLEstimator:
    """統合TPMS機械学習推定器"""
    
    def __init__(self, 
                 pressure_thresholds: Tuple[float, float] = (1.8, 1.5),
                 confidence_threshold: float = 0.7):
        """
        Args:
            pressure_thresholds: (軽度警告, 重度警告) 圧力閾値 [bar]
            confidence_threshold: 信頼度閾値
        """
        self.pressure_thresholds = pressure_thresholds
        self.confidence_threshold = confidence_threshold
        
        # モデル初期化
        self.bnn = None
        self.lazy_ensemble = LazyClassifierEnsemble()
        self.feature_scaler = RobustScaler()
        self.is_trained = False
        
        # 訓練データ保存用
        self.training_features = []
        self.training_pressures = []
        self.training_labels = []
        
    def _create_pressure_labels(self, pressures: np.ndarray) -> np.ndarray:
        """圧力値から分類ラベル作成"""
        labels = np.zeros(pressures.shape[0])
        
        for i, press_set in enumerate(pressures):
            min_pressure = np.min(press_set)
            if min_pressure < self.pressure_thresholds[1]:
                labels[i] = 2  # 重度異常
            elif min_pressure < self.pressure_thresholds[0]:
                labels[i] = 1  # 軽度異常
            else:
                labels[i] = 0  # 正常
        
        return labels
    
    def add_training_data(self, features: Dict[str, float], 
                         pressures: List[float]):
        """訓練データ追加"""
        # 特徴量ベクトル化
        feature_vector = np.array(list(features.values()))
        pressure_array = np.array(pressures)
        
        self.training_features.append(feature_vector)
        self.training_pressures.append(pressure_array)
        
        # 分類ラベル作成
        label = self._create_pressure_labels(pressure_array.reshape(1, -1))[0]
        self.training_labels.append(label)
    
    def train_models(self, test_size: float = 0.2):
        """全モデル訓練"""
        if len(self.training_features) < 50:
            print(f"警告: 訓練データが不足しています ({len(self.training_features)} samples)")
            return False
        
        # データ配列化
        X = np.array(self.training_features)
        y_pressure = np.array(self.training_pressures)
        y_labels = np.array(self.training_labels)
        
        print(f"訓練データ: {X.shape[0]} samples, {X.shape[1]} features")
        
        # 特徴量正規化
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # データ分割
        X_train, X_test, y_p_train, y_p_test, y_l_train, y_l_test = train_test_split(
            X_scaled, y_pressure, y_labels, test_size=test_size, random_state=42
        )
        
        # BNN訓練
        print("ベイジアンニューラルネットワーク訓練中...")
        self.bnn = BayesianNeuralNetwork(input_dim=X.shape[1])
        history = self.bnn.train(X_train, y_p_train, epochs=50)
        
        # Lazy分類器訓練
        print("Lazy分類器アンサンブル訓練中...")
        self.lazy_ensemble.train(X_train, y_l_train)
        
        # 性能評価
        self._evaluate_models(X_test, y_p_test, y_l_test)
        
        self.is_trained = True
        return True
    
    def _evaluate_models(self, X_test: np.ndarray, 
                        y_p_test: np.ndarray, 
                        y_l_test: np.ndarray):
        """モデル性能評価"""
        # BNN評価
        y_pred_mean, y_pred_std = self.bnn.predict_with_uncertainty(X_test)
        pressure_mae = mean_absolute_error(y_p_test, y_pred_mean)
        
        # 分類器評価
        y_class_pred = self.lazy_ensemble.predict(X_test)
        class_accuracy = accuracy_score(y_l_test, y_class_pred)
        
        print(f"\nモデル性能評価:")
        print(f"圧力推定MAE: {pressure_mae:.3f} bar")
        print(f"分類精度: {class_accuracy:.3f}")
        print(f"平均不確実性: {np.mean(y_pred_std):.3f}")
    
    def predict(self, features: Dict[str, float]) -> TPMSPrediction:
        """リアルタイム推定"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        # 特徴量前処理
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # 欠損値・異常値処理
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, 
                                     posinf=1.0, neginf=-1.0)
        
        # 正規化
        X_scaled = self.feature_scaler.transform(feature_vector)
        
        # BNNによる圧力推定
        pressure_pred, pressure_uncertainty = self.bnn.predict_with_uncertainty(X_scaled)
        pressure_pred = pressure_pred[0]
        pressure_uncertainty = pressure_uncertainty[0]
        
        # 分類による異常検知
        class_proba = self.lazy_ensemble.predict_proba(X_scaled)[0]
        predicted_class = np.argmax(class_proba)
        
        # 信頼度計算
        confidence = float(np.max(class_proba)) * (1.0 - np.mean(pressure_uncertainty))
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # 警告レベル決定
        alert_level = TPMSAlertLevel.NORMAL
        affected_wheels = []
        
        if confidence < self.confidence_threshold:
            alert_level = TPMSAlertLevel.SENSOR_FAULT
        elif predicted_class == 2:
            alert_level = TPMSAlertLevel.CRITICAL_PRESSURE
        elif predicted_class == 1:
            alert_level = TPMSAlertLevel.LOW_PRESSURE
        
        # 異常車輪特定
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        for i, pressure in enumerate(pressure_pred):
            if pressure < self.pressure_thresholds[0]:
                affected_wheels.append(wheel_names[i])
        
        return TPMSPrediction(
            pressures=pressure_pred,
            uncertainties=pressure_uncertainty,
            confidence=confidence,
            alert_level=alert_level,
            affected_wheels=affected_wheels,
            timestamp=np.time.time()
        )
    
    def save_models(self, filepath: str):
        """モデル保存"""
        model_data = {
            'bnn_weights': self.bnn.model.get_weights() if self.bnn else None,
            'lazy_ensemble': self.lazy_ensemble,
            'feature_scaler': self.feature_scaler,
            'pressure_thresholds': self.pressure_thresholds,
            'confidence_threshold': self.confidence_threshold,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデル保存完了: {filepath}")
    
    def load_models(self, filepath: str):
        """モデル読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.lazy_ensemble = model_data['lazy_ensemble']
        self.feature_scaler = model_data['feature_scaler']
        self.pressure_thresholds = model_data['pressure_thresholds']
        self.confidence_threshold = model_data['confidence_threshold']
        self.is_trained = model_data['is_trained']
        
        # BNN復元
        if model_data['bnn_weights'] is not None:
            # 特徴量次元数を推定
            input_dim = self.feature_scaler.n_features_in_
            self.bnn = BayesianNeuralNetwork(input_dim=input_dim)
            self.bnn.compile_model()
            self.bnn.model.set_weights(model_data['bnn_weights'])
            self.bnn.is_trained = True
        
        print(f"モデル読み込み完了: {filepath}")

class TPMSRealTimeEstimator:
    """リアルタイムTPMS推定システム"""
    
    def __init__(self, estimator: TPMSMLEstimator):
        """
        Args:
            estimator: 訓練済みML推定器
        """
        self.estimator = estimator
        self.prediction_history = []
        self.alert_history = []
        
    def process_features(self, features: Dict[str, float]) -> TPMSPrediction:
        """特徴量処理と推定実行"""
        try:
            prediction = self.estimator.predict(features)
            
            # 履歴保存
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > 1000:  # 履歴サイズ制限
                self.prediction_history.pop(0)
            
            # アラート処理
            self._handle_alerts(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"推定エラー: {e}")
            # エラー時はセンサー故障として処理
            return TPMSPrediction(
                pressures=np.array([0.0, 0.0, 0.0, 0.0]),
                uncertainties=np.array([1.0, 1.0, 1.0, 1.0]),
                confidence=0.0,
                alert_level=TPMSAlertLevel.SENSOR_FAULT,
                affected_wheels=[],
                timestamp=np.time.time()
            )
    
    def _handle_alerts(self, prediction: TPMSPrediction):
        """アラート処理"""
        if prediction.alert_level != TPMSAlertLevel.NORMAL:
            alert_msg = self._generate_alert_message(prediction)
            print(f"[TPMS ALERT] {alert_msg}")
            
            self.alert_history.append({
                'timestamp': prediction.timestamp,
                'level': prediction.alert_level,
                'message': alert_msg,
                'affected_wheels': prediction.affected_wheels
            })
    
    def _generate_alert_message(self, prediction: TPMSPrediction) -> str:
        """アラートメッセージ生成"""
        if prediction.alert_level == TPMSAlertLevel.SENSOR_FAULT:
            return "センサー異常が検出されました"
        elif prediction.alert_level == TPMSAlertLevel.CRITICAL_PRESSURE:
            wheels = ', '.join(prediction.affected_wheels)
            return f"重大な空気圧低下: {wheels} (推定圧力: {prediction.pressures.min():.1f}bar)"
        elif prediction.alert_level == TPMSAlertLevel.LOW_PRESSURE:
            wheels = ', '.join(prediction.affected_wheels)
            return f"空気圧低下警告: {wheels} (推定圧力: {prediction.pressures.min():.1f}bar)"
        else:
            return "正常"
    
    def get_system_status(self) -> Dict[str, Union[str, float, List]]:
        """システム状態取得"""
        if not self.prediction_history:
            return {"status": "データなし"}
        
        latest = self.prediction_history[-1]
        recent_alerts = [a for a in self.alert_history 
                        if a['timestamp'] > np.time.time() - 3600]  # 1時間以内
        
        return {
            "status": latest.alert_level.name,
            "latest_pressures": latest.pressures.tolist(),
            "confidence": latest.confidence,
            "recent_alerts_count": len(recent_alerts),
            "system_uptime": len(self.prediction_history)
        }

# 使用例・統合テスト
if __name__ == "__main__":
    from indirect_tpms_module import TPMSDataProcessor, TPMSDataSimulator
    
    print("統合TPMS機械学習システム起動")
    
    # データ生成・収集
    simulator = TPMSDataSimulator()
    processor = TPMSDataProcessor()
    estimator = TPMSMLEstimator()
    
    print("訓練データ生成中...")
    
    # 様々な条件での訓練データ生成
    training_scenarios = [
        # (圧力設定, 車速, 運転パターン, サンプル数)
        ([2.2, 2.2, 2.2, 2.2], 60, 'straight', 50),  # 正常
        ([1.9, 2.2, 2.2, 2.2], 50, 'straight', 30),  # 軽度異常
        ([1.6, 2.2, 2.2, 2.2], 40, 'turn', 20),     # 重度異常
        ([2.2, 1.8, 2.2, 2.2], 70, 'brake', 25),    # 他輪異常
        ([1.8, 1.9, 2.2, 2.2], 55, 'straight', 15), # 複数輪異常
    ]
    
    for pressures, speed, maneuver, n_samples in training_scenarios:
        for i in range(n_samples):
            # センサーデータ生成
            data = simulator.generate_sensor_data(pressures, speed, maneuver)
            processor.add_sensor_data(data)
            
            # 特徴量抽出
            if len(processor.data_buffer) >= processor.window_size:
                features = processor.fuse_features()
                if features:  # 有効な特徴量が得られた場合
                    estimator.add_training_data(features, pressures)
    
    print(f"訓練データ収集完了: {len(estimator.training_features)} samples")
    
    # モデル訓練
    if estimator.train_models():
        print("モデル訓練完了")
        
        # リアルタイム推定テスト
        rt_estimator = TPMSRealTimeEstimator(estimator)
        
        print("\nリアルタイム推定テスト開始")
        test_pressures = [1.7, 2.2, 2.2, 2.2]  # 左前輪異常
        
        for i in range(20):
            # テストデータ生成
            test_data = simulator.generate_sensor_data(test_pressures, 65, 'straight')
            processor.add_sensor_data(test_data)
            
            # 特徴量抽出・推定
            features = processor.fuse_features()
            if features:
                prediction = rt_estimator.process_features(features)
                
                if i % 5 == 0:
                    print(f"\nStep {i}:")
                    print(f"  推定圧力: {prediction.pressures}")
                    print(f"  信頼度: {prediction.confidence:.3f}")
                    print(f"  警告レベル: {prediction.alert_level.name}")
                    if prediction.affected_wheels:
                        print(f"  異常車輪: {prediction.affected_wheels}")
        
        # システム状態確認
        status = rt_estimator.get_system_status()
        print(f"\nシステム状態: {status}")
        
        # モデル保存
        estimator.save_models("tpms_model.pkl")
        print("システム初期化・テスト完了")
    
    else:
        print("モデル訓練に失敗しました")