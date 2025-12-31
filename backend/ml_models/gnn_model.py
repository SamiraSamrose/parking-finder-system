import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial.distance import cdist
import networkx as nx
import joblib
import os
from typing import Tuple, Dict


class GNNParkingModel:
    """
    Graph Neural Network for spatial parking prediction
    Simulates graph convolution with dense layers
    """
    
    def __init__(self, model_dir: str = "models", distance_threshold: float = 0.01):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.distance_threshold = distance_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.adjacency_matrix = None
        self.graph = None
        self.is_trained = False
    
    def build_graph(self, df: pd.DataFrame) -> Tuple[np.ndarray, nx.Graph]:
        """
        Build spatial graph from parking locations
        Edges connect spots within distance threshold
        """
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")
        
        unique_locations = df[['latitude', 'longitude']].drop_duplicates().values
        
        distance_matrix = cdist(unique_locations, unique_locations, metric='euclidean')
        
        adjacency_matrix = (distance_matrix < self.distance_threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        
        G = nx.from_numpy_array(adjacency_matrix)
        
        avg_degree = np.sum(adjacency_matrix) / len(adjacency_matrix)
        
        print(f"Graph constructed:")
        print(f"  Nodes: {len(unique_locations)}")
        print(f"  Edges: {np.sum(adjacency_matrix) // 2}")
        print(f"  Average degree: {avg_degree:.2f}")
        
        self.adjacency_matrix = adjacency_matrix
        self.graph = G
        
        return adjacency_matrix, G
    
    def prepare_node_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare node features for GNN
        Features: occupancy_rate, traffic_volume, hour
        """
        feature_cols = []
        
        if 'occupancy_rate' in df.columns:
            feature_cols.append('occupancy_rate')
        elif 'occupied' in df.columns:
            df['occupancy_rate'] = df['occupied']
            feature_cols.append('occupancy_rate')
        
        if 'traffic_volume' in df.columns:
            max_traffic = df['traffic_volume'].max()
            df['traffic_normalized'] = df['traffic_volume'] / (max_traffic if max_traffic > 0 else 1)
            feature_cols.append('traffic_normalized')
        
        if 'hour' in df.columns:
            df['hour_normalized'] = df['hour'] / 24.0
            feature_cols.append('hour_normalized')
        
        X = df[feature_cols].fillna(0).values
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def build_model(self, input_dim: int) -> Sequential:
        """
        Build GNN architecture
        Simulates graph convolution with dense layers
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_col: str = 'occupied',
              test_size: float = 0.2, epochs: int = 30, batch_size: int = 32) -> Dict:
        """
        Train GNN model
        
        Returns training metrics
        """
        self.build_graph(df)
        
        X = self.prepare_node_features(df)
        y = df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model = self.build_model(input_dim=X.shape[1])
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': history.history['loss'][-1],
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'graph_metrics': {
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges),
                'avg_degree': np.sum(self.adjacency_matrix) / len(self.adjacency_matrix)
            }
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
        
        X = self.prepare_node_features(df)
        
        predictions = self.model.predict(X)
        
        return predictions.flatten()
    
    def save(self, model_filename: str = "gnn_parking_model.h5",
             scaler_filename: str = "gnn_scaler.pkl",
             graph_filename: str = "gnn_graph.pkl"):
        """Save GNN model, scaler, and graph"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        graph_path = os.path.join(self.model_dir, graph_filename)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'adjacency_matrix': self.adjacency_matrix,
            'graph': self.graph
        }, graph_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Graph saved to {graph_path}")
    
    def load(self, model_filename: str = "gnn_parking_model.h5",
             scaler_filename: str = "gnn_scaler.pkl",
             graph_filename: str = "gnn_graph.pkl"):
        """Load GNN model, scaler, and graph"""
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        graph_path = os.path.join(self.model_dir, graph_filename)
        
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        graph_data = joblib.load(graph_path)
        self.adjacency_matrix = graph_data['adjacency_matrix']
        self.graph = graph_data['graph']
        
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")