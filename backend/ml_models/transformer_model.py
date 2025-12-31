import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     GlobalAveragePooling1D, MultiHeadAttention)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict


class TransformerParkingModel:
    """Transformer model for parking occupancy prediction"""
    
    def __init__(self, model_dir: str = "models", sequence_length: int = 24):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer
        Features: occupancy_rate, avg_duration_minutes, turnover_rate, revenue_per_hour, traffic_volume
        """
        feature_cols = ['occupancy_rate', 'turnover_rate', 'revenue_per_hour']
        
        if 'avg_duration_minutes' in df.columns:
            feature_cols.insert(1, 'avg_duration_minutes')
        
        if 'traffic_volume' in df.columns:
            feature_cols.append('traffic_volume')
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        df_sorted = df.sort_values('timestamp').copy() if 'timestamp' in df.columns else df.copy()
        
        X_data = df_sorted[available_cols].fillna(0).values
        
        X_scaled = self.scaler.fit_transform(X_data)
        
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(X_scaled[i + self.sequence_length, 0])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build Transformer architecture with multi-head attention
        - 2 attention layers with 4 heads each
        - Feed-forward networks with residual connections
        - Layer normalization
        - Global average pooling
        """
        inputs = Input(shape=input_shape)
        
        attention1 = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(inputs, inputs)
        attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)
        
        ff1 = Dense(128, activation='relu')(attention1)
        ff1 = Dropout(0.2)(ff1)
        ff1 = Dense(input_shape[1])(ff1)
        ff1 = LayerNormalization(epsilon=1e-6)(ff1 + attention1)
        
        attention2 = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(ff1, ff1)
        attention2 = LayerNormalization(epsilon=1e-6)(attention2 + ff1)
        
        ff2 = Dense(128, activation='relu')(attention2)
        ff2 = Dropout(0.2)(ff2)
        ff2 = Dense(input_shape[1])(ff2)
        ff2 = LayerNormalization(epsilon=1e-6)(ff2 + attention2)
        
        pooled = GlobalAveragePooling1D()(ff2)
        
        dense1 = Dense(64, activation='relu')(pooled)
        dense2 = Dense(32, activation='relu')(dense1)
        outputs = Dense(1, activation='sigmoid')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2,
              epochs: int = 50, batch_size: int = 64) -> Dict:
        """
        Train Transformer model with callbacks
        
        Returns training history and metrics
        """
        X_seq, y_seq = self.prepare_data(df)
        
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=42
        )
        
        self.model = self.build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': history.history['loss'][-1],
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'history': history.history
        }
        
        self.is_trained = True
        
        print(f"\nModel Performance:")
        print(f"  Train MSE: {metrics['train_mse']:.4f}")
        print(f"  Test MSE: {metrics['test_mse']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict parking occupancy"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_seq, _ = self.prepare_data(df)
        
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
    def save(self, model_filename: str = "transformer_parking_model.h5",
             scaler_filename: str = "transformer_scaler.pkl"):
        """Save Transformer model and scaler"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_filename: str = "transformer_parking_model.h5",
             scaler_filename: str = "transformer_scaler.pkl"):
        """Load Transformer model and scaler"""
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