"""
間接タイヤ空気圧監視システム (Indirect TPMS) モジュール
最新研究に基づく統合センサーデータ処理システム
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.arima.model import ARIMA
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SensorData:
    """センサーデータ構造体"""
    timestamp: float
    wheel_speeds: np.ndarray  # 4輪の速度 [FL, FR, RL, RR]
    accelerations: np.ndarray  # 3軸加速度 [x, y, z]
    vehicle_speed: float
    steering_angle: float
    brake_pressure: float
    engine_load: float
    gps_data: Dict[str, float]  # latitude, longitude, altitude, speed

class TPMSDataProcessor:
    """TPMSデータ処理クラス"""
    
    def __init__(self, window_size: int = 100, sampling_rate: float = 100.0):
        """
        Args:
            window_size: データウィンドウサイズ
            sampling_rate: サンプリング周波数 [Hz]
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.data_buffer = deque(maxlen=window_size)
        self.baseline_params = {}
        self.is_calibrated = False
        
    def add_sensor_data(self, data: SensorData) -> bool:
        """センサーデータをバッファに追加"""
        self.data_buffer.append(data)
        return len(self.data_buffer) >= self.window_size
    
    def extract_wheel_radius_features(self) -> Dict[str, np.ndarray]:
        """ホイール半径関連特徴量抽出（Honda方式ベース）"""
        if len(self.data_buffer) < self.window_size:
            return {}
        
        # データ配列化
        wheel_speeds = np.array([d.wheel_speeds for d in self.data_buffer])
        vehicle_speeds = np.array([d.vehicle_speed for d in self.data_buffer])
        
        # 有効半径計算
        effective_radius = np.zeros((self.window_size, 4))
        for i in range(4):
            # ゼロ除算回避
            non_zero_mask = wheel_speeds[:, i] > 0.1
            if np.any(non_zero_mask):
                effective_radius[non_zero_mask, i] = (
                    vehicle_speeds[non_zero_mask] / 
                    (wheel_speeds[non_zero_mask, i] * 2 * np.pi) * 60
                )
        
        # 統計的特徴量
        features = {}
        for i in range(4):
            wheel_name = ['FL', 'FR', 'RL', 'RR'][i]
            valid_data = effective_radius[:, i][effective_radius[:, i] > 0]
            
            if len(valid_data) > 10:
                features[f'radius_mean_{wheel_name}'] = np.mean(valid_data)
                features[f'radius_std_{wheel_name}'] = np.std(valid_data)
                features[f'radius_var_{wheel_name}'] = np.var(valid_data)
                
                # 相対変化量
                if len(valid_data) > 1:
                    features[f'radius_change_rate_{wheel_name}'] = (
                        np.abs(np.diff(valid_data)).mean()
                    )
        
        # 輪間比較特徴量
        if len(effective_radius) > 0:
            mean_radius = np.mean(effective_radius, axis=0)
            for i in range(4):
                for j in range(i+1, 4):
                    wheel_i = ['FL', 'FR', 'RL', 'RR'][i]
                    wheel_j = ['FL', 'FR', 'RL', 'RR'][j]
                    if mean_radius[i] > 0 and mean_radius[j] > 0:
                        features[f'radius_ratio_{wheel_i}_{wheel_j}'] = (
                            mean_radius[i] / mean_radius[j]
                        )
        
        return features
    
    def extract_vibration_features(self) -> Dict[str, float]:
        """振動特徴量抽出（加速度センサーベース）"""
        if len(self.data_buffer) < self.window_size:
            return {}
        
        # 加速度データ抽出
        acc_data = np.array([d.accelerations for d in self.data_buffer])
        
        features = {}
        axis_names = ['x', 'y', 'z']
        
        for axis in range(3):
            signal_data = acc_data[:, axis]
            
            # 統計的特徴量
            features[f'acc_mean_{axis_names[axis]}'] = np.mean(signal_data)
            features[f'acc_std_{axis_names[axis]}'] = np.std(signal_data)
            features[f'acc_rms_{axis_names[axis]}'] = np.sqrt(np.mean(signal_data**2))
            features[f'acc_skewness_{axis_names[axis]}'] = stats.skew(signal_data)
            features[f'acc_kurtosis_{axis_names[axis]}'] = stats.kurtosis(signal_data)
            
            # 周波数領域特徴量
            try:
                freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
                fft_values = np.abs(fft(signal_data))
                
                # パワースペクトル密度
                psd = fft_values**2
                features[f'acc_peak_freq_{axis_names[axis]}'] = freqs[np.argmax(psd)]
                features[f'acc_spectral_centroid_{axis_names[axis]}'] = (
                    np.sum(freqs * psd) / np.sum(psd)
                )
                
                # 周波数帯域エネルギー
                low_freq_mask = (freqs >= 0) & (freqs <= 10)
                mid_freq_mask = (freqs > 10) & (freqs <= 50)
                high_freq_mask = freqs > 50
                
                if np.any(low_freq_mask):
                    features[f'acc_low_freq_energy_{axis_names[axis]}'] = (
                        np.sum(psd[low_freq_mask])
                    )
                if np.any(mid_freq_mask):
                    features[f'acc_mid_freq_energy_{axis_names[axis]}'] = (
                        np.sum(psd[mid_freq_mask])
                    )
                if np.any(high_freq_mask):
                    features[f'acc_high_freq_energy_{axis_names[axis]}'] = (
                        np.sum(psd[high_freq_mask])
                    )
            except Exception:
                pass
        
        return features
    
    def extract_histogram_features(self) -> Dict[str, float]:
        """ヒストグラム特徴量抽出"""
        if len(self.data_buffer) < self.window_size:
            return {}
        
        # 車輪速度データ
        wheel_speeds = np.array([d.wheel_speeds for d in self.data_buffer])
        
        features = {}
        for i in range(4):
            wheel_name = ['FL', 'FR', 'RL', 'RR'][i]
            speed_data = wheel_speeds[:, i]
            
            # ヒストグラム計算
            hist, bins = np.histogram(speed_data, bins=20, density=True)
            
            # ヒストグラム統計量
            features[f'hist_entropy_{wheel_name}'] = stats.entropy(hist + 1e-10)
            features[f'hist_peak_{wheel_name}'] = np.max(hist)
            features[f'hist_peak_pos_{wheel_name}'] = bins[np.argmax(hist)]
            
            # 分布の非対称性
            features[f'hist_asymmetry_{wheel_name}'] = np.sum(
                (bins[:-1] - np.mean(speed_data)) * hist
            )
        
        return features
    
    def extract_arma_features(self, order: Tuple[int, int] = (2, 1)) -> Dict[str, float]:
        """ARMA特徴量抽出"""
        if len(self.data_buffer) < max(30, order[0] + order[1] + 10):
            return {}
        
        wheel_speeds = np.array([d.wheel_speeds for d in self.data_buffer])
        
        features = {}
        for i in range(4):
            wheel_name = ['FL', 'FR', 'RL', 'RR'][i]
            try:
                # ARMA モデルフィッティング
                model = ARIMA(wheel_speeds[:, i], order=(order[0], 0, order[1]))
                fitted_model = model.fit()
                
                # ARパラメータ
                ar_params = fitted_model.arparams if hasattr(fitted_model, 'arparams') else []
                for j, param in enumerate(ar_params):
                    features[f'arma_ar_{j}_{wheel_name}'] = param
                
                # MAパラメータ
                ma_params = fitted_model.maparams if hasattr(fitted_model, 'maparams') else []
                for j, param in enumerate(ma_params):
                    features[f'arma_ma_{j}_{wheel_name}'] = param
                
                # モデル適合度
                features[f'arma_aic_{wheel_name}'] = fitted_model.aic
                features[f'arma_bic_{wheel_name}'] = fitted_model.bic
                
            except Exception:
                # モデルフィッティング失敗時はデフォルト値
                for j in range(order[0]):
                    features[f'arma_ar_{j}_{wheel_name}'] = 0.0
                for j in range(order[1]):
                    features[f'arma_ma_{j}_{wheel_name}'] = 0.0
                features[f'arma_aic_{wheel_name}'] = 0.0
                features[f'arma_bic_{wheel_name}'] = 0.0
        
        return features
    
    def extract_vehicle_dynamics_features(self) -> Dict[str, float]:
        """車両動特性特徴量抽出"""
        if len(self.data_buffer) < self.window_size:
            return {}
        
        # 車両データ抽出
        steering_angles = np.array([d.steering_angle for d in self.data_buffer])
        brake_pressures = np.array([d.brake_pressure for d in self.data_buffer])
        engine_loads = np.array([d.engine_load for d in self.data_buffer])
        vehicle_speeds = np.array([d.vehicle_speed for d in self.data_buffer])
        
        features = {}
        
        # ステアリング特徴量
        features['steering_mean'] = np.mean(steering_angles)
        features['steering_std'] = np.std(steering_angles)
        features['steering_change_rate'] = np.mean(np.abs(np.diff(steering_angles)))
        
        # ブレーキ特徴量
        features['brake_mean'] = np.mean(brake_pressures)
        features['brake_activation_ratio'] = np.mean(brake_pressures > 0.1)
        
        # エンジン負荷特徴量
        features['engine_load_mean'] = np.mean(engine_loads)
        features['engine_load_std'] = np.std(engine_loads)
        
        # 速度変化特徴量
        features['speed_change_rate'] = np.mean(np.abs(np.diff(vehicle_speeds)))
        features['acceleration_magnitude'] = np.std(np.diff(vehicle_speeds))
        
        return features
    
    def fuse_features(self) -> Dict[str, float]:
        """特徴量融合処理"""
        # 各特徴量抽出
        wheel_features = self.extract_wheel_radius_features()
        vibration_features = self.extract_vibration_features()
        histogram_features = self.extract_histogram_features()
        arma_features = self.extract_arma_features()
        dynamics_features = self.extract_vehicle_dynamics_features()
        
        # 特徴量統合
        fused_features = {}
        fused_features.update(wheel_features)
        fused_features.update(vibration_features)
        fused_features.update(histogram_features)
        fused_features.update(arma_features)
        fused_features.update(dynamics_features)
        
        # NaN値処理
        for key, value in fused_features.items():
            if np.isnan(value) or np.isinf(value):
                fused_features[key] = 0.0
        
        return fused_features
    
    def calibrate_baseline(self) -> bool:
        """ベースライン校正（正常空気圧時の特徴量学習）"""
        if len(self.data_buffer) < self.window_size:
            return False
        
        baseline_features = self.fuse_features()
        self.baseline_params = baseline_features.copy()
        self.is_calibrated = True
        
        print(f"ベースライン校正完了: {len(baseline_features)} 特徴量")
        return True
    
    def get_deviation_features(self) -> Dict[str, float]:
        """ベースラインからの偏差特徴量計算"""
        if not self.is_calibrated:
            return {}
        
        current_features = self.fuse_features()
        deviation_features = {}
        
        for key in current_features:
            if key in self.baseline_params:
                baseline_val = self.baseline_params[key]
                current_val = current_features[key]
                
                if baseline_val != 0:
                    deviation_features[f'dev_{key}'] = (
                        (current_val - baseline_val) / baseline_val
                    )
                else:
                    deviation_features[f'dev_{key}'] = current_val
        
        return deviation_features

class TPMSDataSimulator:
    """TPMS用データシミュレータ"""
    
    def __init__(self, nominal_pressure: float = 2.2):
        """
        Args:
            nominal_pressure: 正常空気圧 [bar]
        """
        self.nominal_pressure = nominal_pressure
        self.time = 0.0
        
    def generate_sensor_data(self, 
                           pressures: List[float],
                           vehicle_speed: float = 50.0,
                           maneuver_type: str = 'straight') -> SensorData:
        """
        シミュレーション用センサーデータ生成
        
        Args:
            pressures: 4輪の空気圧 [bar] [FL, FR, RL, RR]
            vehicle_speed: 車速 [km/h]
            maneuver_type: 運転パターン ('straight', 'turn', 'brake')
        """
        # 空気圧による有効半径変化（簡易モデル）
        radius_nominal = 0.32  # [m]
        radii = [radius_nominal * (p / self.nominal_pressure)**0.1 for p in pressures]
        
        # 車輪速度計算
        speed_ms = vehicle_speed / 3.6
        wheel_speeds = [speed_ms / r if r > 0 else 0 for r in radii]
        
        # ノイズ追加
        wheel_speeds = [ws + np.random.normal(0, 0.1) for ws in wheel_speeds]
        
        # 加速度データ（空気圧による振動特性変化を模擬）
        base_acc = np.array([0.0, 0.0, -9.81])
        
        # 空気圧低下による振動増加
        pressure_effect = sum([(self.nominal_pressure - p) / self.nominal_pressure 
                              for p in pressures]) / 4
        vibration_amplitude = 0.5 + pressure_effect * 2.0
        
        # 走行パターンによる加速度変化
        if maneuver_type == 'turn':
            base_acc[1] += np.sin(self.time * 0.5) * 2.0
        elif maneuver_type == 'brake':
            base_acc[0] -= 2.0
        
        # 振動成分追加
        vibration = np.array([
            np.sin(self.time * 20 + i) * vibration_amplitude * np.random.uniform(0.5, 1.5)
            for i in range(3)
        ])
        
        accelerations = base_acc + vibration
        
        # その他センサーデータ
        steering_angle = 0.0
        if maneuver_type == 'turn':
            steering_angle = np.sin(self.time * 0.3) * 15.0
        
        brake_pressure = 5.0 if maneuver_type == 'brake' else 0.0
        engine_load = 30.0 + np.random.normal(0, 5)
        
        self.time += 0.01  # 10ms step
        
        return SensorData(
            timestamp=self.time,
            wheel_speeds=np.array(wheel_speeds),
            accelerations=accelerations,
            vehicle_speed=vehicle_speed,
            steering_angle=steering_angle,
            brake_pressure=brake_pressure,
            engine_load=engine_load,
            gps_data={'latitude': 35.0, 'longitude': 139.0, 
                     'altitude': 50.0, 'speed': vehicle_speed}
        )

# 使用例
if __name__ == "__main__":
    # TPMSプロセッサ初期化
    tpms = TPMSDataProcessor(window_size=100, sampling_rate=100.0)
    simulator = TPMSDataSimulator()
    
    print("間接TPMS システム起動")
    print("正常空気圧でのベースライン校正中...")
    
    # 正常状態でのベースライン学習
    normal_pressures = [2.2, 2.2, 2.2, 2.2]  # 全輪正常圧力
    
    for i in range(150):
        data = simulator.generate_sensor_data(normal_pressures, 60.0, 'straight')
        if tpms.add_sensor_data(data):
            if i == 120:  # 十分なデータが蓄積された時点で校正
                tpms.calibrate_baseline()
    
    print("校正完了。圧力異常検知モード開始")
    
    # 異常状態テスト
    abnormal_pressures = [1.8, 2.2, 2.2, 2.2]  # 左前輪の圧力低下
    
    for i in range(50):
        data = simulator.generate_sensor_data(abnormal_pressures, 55.0, 'straight')
        tpms.add_sensor_data(data)
        
        # 偏差特徴量計算
        deviation_features = tpms.get_deviation_features()
        
        if i % 10 == 0 and deviation_features:
            print(f"\nStep {i}: 主要偏差特徴量")
            # 大きな偏差を持つ特徴量を表示
            for key, value in deviation_features.items():
                if abs(value) > 0.1:  # 10%以上の偏差
                    print(f"  {key}: {value:.3f}")