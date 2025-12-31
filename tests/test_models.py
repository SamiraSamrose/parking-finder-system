import unittest
import numpy as np
import pandas as pd
from backend.ml_models.xgboost_model import XGBoostParkingModel
from backend.ml_models.lstm_model import LSTMParkingModel
from backend.ml_models.transformer_model import TransformerParkingModel
from backend.ml_models.gnn_model import GNNParkingModel


class TestXGBoostModel(unittest.TestCase):
    """Test XGBoost parking model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = XGBoostParkingModel()
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'is_weekend': np.random.randint(0, 2, 100),
            'spot_type': np.random.choice(['free_street', 'paid_street'], 100),
            'zone': np.random.choice(['downtown', 'midtown'], 100),
            'weather_main': np.random.choice(['Clear', 'Rain'], 100),
            'temp': np.random.uniform(10, 30, 100),
            'traffic_volume': np.random.randint(1000, 7000, 100),
            'hourly_rate': np.random.uniform(2, 20, 100),
            'occupied': np.random.randint(0, 2, 100)
        })
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        processed = self.model.prepare_features(self.test_data)
        
        self.assertIn('hour_sin', processed.columns)
        self.assertIn('hour_cos', processed.columns)
        self.assertIn('day_sin', processed.columns)
        self.assertIn('day_cos', processed.columns)
        self.assertIn('temp_normalized', processed.columns)
    
    def test_model_training(self):
        """Test model training"""
        metrics = self.model.train(self.test_data)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_prediction(self):
        """Test model prediction"""
        self.model.train(self.test_data)
        
        predictions = self.model.predict(self.test_data.head(10))
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_prediction_probability(self):
        """Test probability predictions"""
        self.model.train(self.test_data)
        
        probabilities = self.model.predict_proba(self.test_data.head(10))
        
        self.assertEqual(probabilities.shape, (10, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
    
    def test_save_load(self):
        """Test model save and load"""
        self.model.train(self.test_data)
        
        self.model.save('test_xgboost.pkl')
        
        new_model = XGBoostParkingModel()
        new_model.load('test_xgboost.pkl')
        
        self.assertTrue(new_model.is_trained)


class TestLSTMModel(unittest.TestCase):
    """Test LSTM parking model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LSTMParkingModel(sequence_length=24)
        
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'hour': [t.hour for t in timestamps],
            'day_of_week': [t.dayofweek for t in timestamps],
            'temp_normalized': np.random.randn(100),
            'traffic_normalized': np.random.rand(100),
            'weather_impact': np.random.rand(100),
            'occupied': np.random.randint(0, 2, 100)
        })
    
    def test_sequence_creation(self):
        """Test sequence creation"""
        X_scaled, y_data = self.model.prepare_data(self.test_data)
        X_seq, y_seq = self.model.create_sequences(X_scaled, y_data)
        
        self.assertEqual(X_seq.shape[1], 24)
        self.assertEqual(len(X_seq), len(y_seq))
    
    def test_model_building(self):
        """Test model architecture"""
        model = self.model.build_model(input_shape=(24, 5))
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 7)
    
    def test_training(self):
        """Test LSTM training"""
        metrics = self.model.train(self.test_data, epochs=2, batch_size=8)
        
        self.assertIn('train_mse', metrics)
        self.assertIn('val_mse', metrics)
        self.assertIn('test_r2', metrics)


class TestTransformerModel(unittest.TestCase):
    """Test Transformer parking model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = TransformerParkingModel(sequence_length=24)
        
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'occupancy_rate': np.random.rand(100),
            'turnover_rate': np.random.rand(100),
            'revenue_per_hour': np.random.uniform(10, 50, 100)
        })
    
    def test_data_preparation(self):
        """Test data preparation"""
        X_seq, y_seq = self.model.prepare_data(self.test_data)
        
        self.assertEqual(len(X_seq.shape), 3)
        self.assertEqual(X_seq.shape[1], 24)
    
    def test_model_architecture(self):
        """Test transformer architecture"""
        model = self.model.build_model(input_shape=(24, 3))
        
        self.assertIsNotNone(model)
        self.assertGreater(len(model.layers), 5)


class TestGNNModel(unittest.TestCase):
    """Test GNN parking model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = GNNParkingModel()
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'latitude': 37.7749 + np.random.randn(50) * 0.01,
            'longitude': -122.4194 + np.random.randn(50) * 0.01,
            'occupied': np.random.randint(0, 2, 50),
            'traffic_volume': np.random.randint(1000, 7000, 50),
            'hour': np.random.randint(0, 24, 50)
        })
    
    def test_graph_construction(self):
        """Test spatial graph building"""
        adjacency, graph = self.model.build_graph(self.test_data)
        
        self.assertIsNotNone(adjacency)
        self.assertIsNotNone(graph)
        self.assertGreater(len(graph.nodes), 0)
    
    def test_node_features(self):
        """Test node feature preparation"""
        X = self.model.prepare_node_features(self.test_data)
        
        self.assertEqual(len(X), len(self.test_data))
        self.assertGreater(X.shape[1], 0)


if __name__ == '__main__':
    unittest.main()