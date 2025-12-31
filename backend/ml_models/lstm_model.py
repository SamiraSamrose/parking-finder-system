import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict


class LSTMParkingModel:
    """LSTM model for time-series parking prediction"""
    
    def __init__(self, model_dir: str = "models", sequence_length: int = 24):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        Input shape: (samples, sequence_length, features)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'occupied') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale data for LSTM"""
        feature_cols = ['hour', 'day_of_week', 'temp_normalized', 
                       'traffic_normalized', 'weather_impact']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        df_sorted = df.sort_values('timestamp').copy()
        
        X_data = df_sorted[available_cols].fillna(0).values
        y_data = df_sorted[target_col].values
        
        X_scaled = self.scaler.fit_transform(X_data)
        
        return X_scaled, y_data
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM architecture
        - 64 LSTM units with return_sequences
        - Dropout 0.2
        - 32 LSTM units
        - Dense layers with sigmoid activation
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_col: str = 'occupied',
              validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model with early stopping
        
        Returns training history and metrics
        """
        X_scaled, y_data = self.prepare_data(df, target_col)
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_data)
        
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")
        
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        self.model = self.build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        y_pred = self.model.predict(X_val)
        
        metrics = {
            'train_mse': history.history['loss'][-1],
            'val_mse': history.history['val_loss'][-1],
            'test_mse': mean_squared_error(y_val, y_pred),
            'test_mae': mean_absolute_error(y_val, y_pred),
            'test_r2': r2_score(y_val, y_pred),
            'history': history.history
        }
        
        self.is_trained = True
        
        print(f"\nModel Performance:")
        print(f"  Train MSE: {metrics['train_mse']:.4f}")
        print(f"  Val MSE: {metrics['val_mse']:.4f}")
        print(f"  Test MSE: {metrics['test_mse']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, target_col: str = 'occupied') -> np.ndarray:
        """Predict parking occupancy for sequences"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled, _ = self.prepare_data(df, target_col)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
    def save(self, model_filename: str = "lstm_parking_model.h5",
             scaler_filename: str = "lstm_scaler.pkl"):
        """Save LSTM model and scaler"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_filename: str = "lstm_parking_model.h5",
             scaler_filename: str = "lstm_scaler.pkl"):
        """Load LSTM model and scaler"""
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")