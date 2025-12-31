import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict


class XGBoostParkingModel:
    """XGBoost model for parking occupancy prediction"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering pipeline
        - Cyclic encoding for temporal features
        - Label encoding for categorical features
        - Normalization for numerical features
        """
        df = df.copy()
        
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        categorical_cols = ['spot_type', 'zone', 'weather_main']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        if 'temp' in df.columns:
            df['temp_normalized'] = (df['temp'] - df['temp'].mean()) / (df['temp'].std() + 1e-8)
        
        if 'traffic_volume' in df.columns:
            max_traffic = df['traffic_volume'].max()
            df['traffic_normalized'] = df['traffic_volume'] / (max_traffic if max_traffic > 0 else 1)
        
        if 'hourly_rate' in df.columns:
            max_rate = df['hourly_rate'].max()
            df['rate_normalized'] = df['hourly_rate'] / (max_rate if max_rate > 0 else 1)
        
        return df
    
    def train(self, df: pd.DataFrame, target_col: str = 'occupied', 
              test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict:
        """
        Train XGBoost model with optional hyperparameter tuning
        
        Returns performance metrics
        """
        df_processed = self.prepare_features(df)
        
        feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                       'spot_type_encoded', 'zone_encoded', 'weather_main_encoded',
                       'temp_normalized', 'traffic_normalized', 'rate_normalized',
                       'is_weekend']
        
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
        self.feature_names = feature_cols
        
        X = df_processed[feature_cols].fillna(0)
        y = df_processed[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        self.is_trained = True
        
        print(f"Model Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict parking occupancy"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        df_processed = self.prepare_features(df)
        X = df_processed[self.feature_names].fillna(0)
        
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict parking occupancy probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        df_processed = self.prepare_features(df)
        X = df_processed[self.feature_names].fillna(0)
        
        return self.model.predict_proba(X)
    
    def save(self, filename: str = "xgboost_parking_model.pkl"):
        """Save model and encoders"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        filepath = os.path.join(self.model_dir, filename)
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filename: str = "xgboost_parking_model.pkl"):
        """Load model and encoders"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))