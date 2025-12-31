from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List

from backend.data_loader import DataLoader
from backend.ml_models.xgboost_model import XGBoostParkingModel
from backend.ml_models.lstm_model import LSTMParkingModel
from backend.ml_models.transformer_model import TransformerParkingModel
from backend.ml_models.gnn_model import GNNParkingModel
from backend.services.parking_finder import ParkingFinderService
from backend.services.kafka_streaming import KafkaStreamingService
from backend.services.datadog_monitor import DatadogMonitorService
from backend.services.voice_interface import VoiceInterfaceService
from backend.services.google_integration import GoogleIntegrationService


class APIState:
    """Global state for API"""
    def __init__(self):
        self.data_loader = None
        self.datasets = {}
        self.models = {}
        self.services = {}
        self.initialized = False


state = APIState()


def initialize_system(config):
    """
    Initialize all system components
    Loads data, trains models, initializes services
    """
    print("Initializing parking finder system...")
    
    state.data_loader = DataLoader()
    state.datasets = state.data_loader.load_all_datasets()
    
    print("Training XGBoost model...")
    xgb_model = XGBoostParkingModel()
    xgb_metrics = xgb_model.train(state.datasets['parking'])
    state.models['xgboost'] = xgb_model
    
    print("Training LSTM model...")
    lstm_model = LSTMParkingModel()
    lstm_metrics = lstm_model.train(state.datasets['parking'])
    state.models['lstm'] = lstm_model
    
    print("Training Transformer model...")
    transformer_model = TransformerParkingModel()
    transformer_metrics = transformer_model.train(state.datasets['historical'])
    state.models['transformer'] = transformer_model
    
    print("Training GNN model...")
    gnn_model = GNNParkingModel()
    gnn_metrics = gnn_model.train(state.datasets['parking'])
    state.models['gnn'] = gnn_model
    
    print("Initializing services...")
    state.services['parking_finder'] = ParkingFinderService(ml_model=xgb_model)
    state.services['kafka'] = KafkaStreamingService(config)
    state.services['datadog'] = DatadogMonitorService(config)
    state.services['voice'] = VoiceInterfaceService(config)
    state.services['google'] = GoogleIntegrationService(config)
    
    state.initialized = True
    
    print("System initialization complete")
    
    return {
        'status': 'initialized',
        'models_trained': list(state.models.keys()),
        'services_loaded': list(state.services.keys()),
        'datasets_loaded': {k: len(v) for k, v in state.datasets.items()}
    }


def register_routes(app: Flask, config):
    """Register all API routes"""
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint
        Returns system status and uptime
        """
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'initialized': state.initialized,
            'version': '1.0.0'
        })
    
    @app.route('/api/initialize', methods=['POST'])
    def initialize():
        """
        Initialize system components
        Loads data and trains models
        """
        try:
            result = initialize_system(config)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/parking/search', methods=['POST'])
    def search_parking():
        """
        Search for optimal parking spots
        Request body: {latitude, longitude, max_distance, max_rate, spot_types, top_n}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized. Call /api/initialize first'}), 400
        
        data = request.get_json()
        
        user_lat = data.get('latitude', 37.7749)
        user_lon = data.get('longitude', -122.4194)
        max_distance = data.get('max_distance', 2.0)
        max_rate = data.get('max_rate', None)
        spot_types = data.get('spot_types', None)
        top_n = data.get('top_n', 10)
        
        finder = state.services['parking_finder']
        
        optimal_spots = finder.find_optimal_spots(
            state.datasets['parking'],
            user_lat, user_lon,
            max_distance_km=max_distance,
            top_n=top_n,
            spot_type_filter=spot_types,
            max_hourly_rate=max_rate
        )
        
        results = optimal_spots.to_dict('records')
        
        return jsonify({
            'total_results': len(results),
            'spots': results
        }), 200
    
    @app.route('/api/parking/recommendations', methods=['POST'])
    def get_recommendations():
        """
        Get real-time parking recommendations
        Request body: {latitude, longitude, duration_hours, preferences}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        
        user_lat = data.get('latitude', 37.7749)
        user_lon = data.get('longitude', -122.4194)
        duration = data.get('duration_hours', 2.0)
        preferences = data.get('preferences', {})
        
        finder = state.services['parking_finder']
        
        recommendations = finder.get_real_time_recommendations(
            state.datasets['parking'],
            user_lat, user_lon,
            duration_hours=duration,
            preferences=preferences
        )
        
        return jsonify({
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    @app.route('/api/parking/zones', methods=['GET'])
    def get_zone_statistics():
        """
        Get parking statistics by zone
        Returns occupancy rates, costs, and availability
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        finder = state.services['parking_finder']
        zone_stats = finder.get_zone_statistics(state.datasets['parking'])
        
        return jsonify({
            'zones': zone_stats,
            'total_zones': len(zone_stats)
        }), 200
    
    @app.route('/api/parking/patterns', methods=['GET'])
    def analyze_patterns():
        """
        Analyze historical parking patterns
        Returns peak hours, occupancy trends, revenue insights
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        finder = state.services['parking_finder']
        analysis = finder.analyze_parking_patterns(state.datasets['historical'])
        
        return jsonify(analysis), 200
    
    @app.route('/api/models/predict', methods=['POST'])
    def predict_occupancy():
        """
        Predict parking occupancy using ML models
        Request body: {model_type, features}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        model_type = data.get('model_type', 'xgboost')
        features = data.get('features', {})
        
        if model_type not in state.models:
            return jsonify({'error': f'Model {model_type} not found'}), 404
        
        model = state.models[model_type]
        
        feature_df = pd.DataFrame([features])
        
        if model_type == 'xgboost':
            predictions = model.predict_proba(feature_df)
            result = {
                'model': model_type,
                'occupied_probability': float(predictions[0][1]),
                'available_probability': float(predictions[0][0])
            }
        else:
            predictions = model.predict(feature_df)
            result = {
                'model': model_type,
                'prediction': float(predictions[0])
            }
        
        return jsonify(result), 200
    
    @app.route('/api/models/metrics', methods=['GET'])
    def get_model_metrics():
        """
        Get performance metrics for all models
        Returns accuracy, precision, recall, F1 scores
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        metrics = {}
        
        for model_name, model in state.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                metrics[model_name] = {
                    'trained': True,
                    'model_type': model.__class__.__name__
                }
            else:
                metrics[model_name] = {
                    'trained': False
                }
        
        return jsonify(metrics), 200
    
    @app.route('/api/kafka/produce', methods=['POST'])
    def produce_kafka_event():
        """
        Produce event to Kafka topic
        Request body: {topic, event_data}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        topic = data.get('topic')
        event_data = data.get('event_data', {})
        
        kafka = state.services['kafka']
        
        if not kafka.producer:
            kafka.initialize_producer()
        
        if topic == 'parking_events':
            kafka.produce_parking_event(event_data)
        elif topic == 'traffic_updates':
            kafka.produce_traffic_update(event_data)
        elif topic == 'predictions':
            kafka.produce_prediction(event_data)
        elif topic == 'alerts':
            kafka.produce_alert(event_data)
        else:
            return jsonify({'error': f'Unknown topic: {topic}'}), 400
        
        kafka.flush()
        
        return jsonify({
            'status': 'produced',
            'topic': topic
        }), 200
    
    @app.route('/api/kafka/simulate', methods=['POST'])
    def simulate_kafka_stream():
        """
        Simulate Kafka streaming data
        Request body: {stream_type, duration_seconds}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        stream_type = data.get('stream_type', 'magnetometer')
        duration = data.get('duration_seconds', 60)
        
        kafka = state.services['kafka']
        
        if not kafka.producer:
            kafka.initialize_producer()
        
        if stream_type == 'magnetometer':
            event_count = kafka.simulate_iot_magnetometer_stream(duration)
        elif stream_type == 'cctv':
            event_count = kafka.simulate_cctv_edge_stream(duration)
        elif stream_type == 'gps':
            event_count = kafka.simulate_user_gps_stream(duration)
        else:
            return jsonify({'error': f'Unknown stream type: {stream_type}'}), 400
        
        return jsonify({
            'status': 'complete',
            'stream_type': stream_type,
            'events_produced': event_count,
            'duration_seconds': duration
        }), 200
    
    @app.route('/api/monitoring/metrics', methods=['POST'])
    def send_monitoring_metric():
        """
        Send metric to Datadog
        Request body: {metric_name, value, tags, metric_type}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        metric_name = data.get('metric_name')
        value = data.get('value')
        tags = data.get('tags', [])
        metric_type = data.get('metric_type', 'gauge')
        
        datadog = state.services['datadog']
        datadog.send_metric(metric_name, value, tags=tags, metric_type=metric_type)
        
        return jsonify({
            'status': 'sent',
            'metric': metric_name
        }), 200
    
    @app.route('/api/monitoring/simulate', methods=['POST'])
    def simulate_monitoring():
        """
        Simulate monitoring data for dashboard
        Request body: {duration_minutes, samples_per_minute}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        duration = data.get('duration_minutes', 60)
        samples = data.get('samples_per_minute', 120)
        
        datadog = state.services['datadog']
        metrics_data = datadog.simulate_monitoring_data(duration, samples)
        
        return jsonify({
            'status': 'complete',
            'total_samples': len(metrics_data['timestamps']),
            'data': metrics_data
        }), 200
    
    @app.route('/api/monitoring/detect-hallucination', methods=['POST'])
    def detect_hallucination():
        """
        Detect LLM hallucination
        Request body: {suggested_spot_id, restricted_zones, restriction_status, restriction_reason}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        spot_id = data.get('suggested_spot_id')
        restricted_zones = data.get('restricted_zones', [])
        restriction_status = data.get('restriction_status')
        restriction_reason = data.get('restriction_reason')
        
        datadog = state.services['datadog']
        is_hallucination = datadog.detect_legal_hallucination(
            spot_id, restricted_zones, restriction_status, restriction_reason
        )
        
        return jsonify({
            'is_hallucination': is_hallucination,
            'spot_id': spot_id
        }), 200
    
    @app.route('/api/monitoring/detect-injection', methods=['POST'])
    def detect_injection():
        """
        Detect prompt injection attempts
        Request body: {user_input}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        user_input = data.get('user_input', '')
        
        datadog = state.services['datadog']
        detection_result = datadog.detect_prompt_injection(user_input)
        
        return jsonify(detection_result), 200
    
    @app.route('/api/voice/query', methods=['POST'])
    def voice_query():
        """
        Handle voice query with agent routing
        Request body: {user_input, synthesize}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        user_input = data.get('user_input', '')
        synthesize = data.get('synthesize', False)
        
        voice = state.services['voice']
        result = voice.handle_voice_query(user_input, synthesize=synthesize)
        
        return jsonify(result), 200
    
    @app.route('/api/voice/agents', methods=['GET'])
    def get_voice_agents():
        """
        Get information about all voice agents
        Returns agent capabilities and configurations
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        voice = state.services['voice']
        agents = voice.get_all_agents()
        
        return jsonify({
            'agents': agents,
            'total_agents': len(agents)
        }), 200
    
    @app.route('/api/voice/simulate', methods=['POST'])
    def simulate_voice_conversations():
        """
        Simulate voice conversations
        Request body: {num_queries}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        num_queries = data.get('num_queries', 10)
        
        voice = state.services['voice']
        summary = voice.simulate_voice_conversations(num_queries)
        
        return jsonify(summary), 200
    
    @app.route('/api/google/3d-tiles', methods=['GET'])
    def get_3d_tiles_config():
        """
        Get Google Maps 3D tiles configuration
        Returns API endpoints and rendering parameters
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        google = state.services['google']
        config = google.get_3d_tiles_config()
        
        return jsonify(config), 200
    
    @app.route('/api/google/curb-clearance', methods=['POST'])
    def calculate_curb_clearance():
        """
        Calculate curb-level clearance
        Request body: {spot_lat, spot_lon, vehicle_length, vehicle_width}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        spot_lat = data.get('spot_lat', 37.7749)
        spot_lon = data.get('spot_lon', -122.4194)
        vehicle_length = data.get('vehicle_length', 4.5)
        vehicle_width = data.get('vehicle_width', 2.0)
        
        google = state.services['google']
        clearance = google.calculate_curb_clearance(
            spot_lat, spot_lon, vehicle_length, vehicle_width
        )
        
        return jsonify(clearance), 200
    
    @app.route('/api/google/multi-agent-query', methods=['POST'])
    def multi_agent_query():
        """
        Execute multi-agent orchestration query
        Request body: {query, vehicle_length, duration_hours}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        query = data.get('query', 'Find parking for my truck')
        vehicle_length = data.get('vehicle_length', 15.0)
        duration = data.get('duration_hours', 2.0)
        
        google = state.services['google']
        result = google.multi_agent_query(query, vehicle_length, duration)
        
        return jsonify(result), 200
    
    @app.route('/api/google/bigquery', methods=['POST'])
    def execute_bigquery():
        """
        Execute BigQuery geospatial query
        Request body: {query_type, parameters}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        query_type = data.get('query_type', 'spatial_join')
        parameters = data.get('parameters', {})
        
        google = state.services['google']
        result = google.execute_bigquery_geospatial(query_type, parameters)
        
        return jsonify(result), 200
    
    @app.route('/api/google/calendar', methods=['POST'])
    def create_calendar_event():
        """
        Create Google Calendar parking event
        Request body: {location, spot_id, duration_hours, cost}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        location = data.get('location', 'Downtown')
        spot_id = data.get('spot_id', 'SPOT_00000')
        duration = data.get('duration_hours', 2.0)
        cost = data.get('cost', 10.0)
        
        google = state.services['google']
        result = google.create_calendar_event(location, spot_id, duration, cost)
        
        return jsonify(result), 200
    
    @app.route('/api/google/sheets', methods=['POST'])
    def log_to_sheets():
        """
        Log expense to Google Sheets
        Request body: {transaction}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        transaction = data.get('transaction', {})
        
        google = state.services['google']
        result = google.log_expense_to_sheets(transaction)
        
        return jsonify(result), 200
    
    @app.route('/api/google/automate', methods=['POST'])
    def automate_workflow():
        """
        Automate complete parking workflow
        Request body: {scenario}
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        scenario = data.get('scenario', 'business_meeting')
        
        google = state.services['google']
        result = google.automate_parking_workflow(scenario)
        
        return jsonify(result), 200
    
    @app.route('/api/datasets', methods=['GET'])
    def get_datasets():
        """
        Get information about loaded datasets
        Returns dataset sizes and column information
        """
        if not state.initialized:
            return jsonify({'error': 'System not initialized'}), 400
        
        dataset_info = {}
        
        for name, df in state.datasets.items():
            dataset_info[name] = {
                'rows': len(df),
                'columns': list(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        
        return jsonify(dataset_info), 200
    
    @app.route('/api/config', methods=['GET'])
    def get_system_config():
        """
        Get system configuration
        Returns configuration parameters
        """
        return jsonify(config.to_dict()), 200