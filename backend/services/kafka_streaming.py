import json
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
import threading
import numpy as np


class KafkaStreamingService:
    """
    Confluent Kafka streaming service
    Handles real-time parking events, traffic updates, and predictions
    """
    
    def __init__(self, config):
        self.config = config
        self.producer_config = {
            'bootstrap.servers': config.kafka.bootstrap_servers,
            'security.protocol': config.kafka.security_protocol,
            'sasl.mechanisms': config.kafka.sasl_mechanism,
            'sasl.username': config.kafka.api_key,
            'sasl.password': config.kafka.api_secret,
            'client.id': 'parking-finder-producer'
        }
        
        self.consumer_config = {
            'bootstrap.servers': config.kafka.bootstrap_servers,
            'security.protocol': config.kafka.security_protocol,
            'sasl.mechanisms': config.kafka.sasl_mechanism,
            'sasl.username': config.kafka.api_key,
            'sasl.password': config.kafka.api_secret,
            'group.id': 'parking-finder-consumer-group',
            'auto.offset.reset': 'earliest'
        }
        
        self.producer = None
        self.consumers = {}
        self.topics = config.kafka.topics
        self.running = False
        self.message_handlers = {}
    
    def initialize_producer(self):
        """Initialize Kafka producer"""
        try:
            self.producer = Producer(self.producer_config)
            print("Kafka producer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Kafka producer: {e}")
            self.producer = None
    
    def create_topics(self):
        """Create Kafka topics if they don't exist"""
        try:
            admin_client = AdminClient({
                'bootstrap.servers': self.producer_config['bootstrap.servers']
            })
            
            topic_list = [NewTopic(topic, num_partitions=3, replication_factor=3) 
                         for topic in self.topics.values()]
            
            fs = admin_client.create_topics(topic_list)
            
            for topic, f in fs.items():
                try:
                    f.result()
                    print(f"Topic {topic} created")
                except Exception as e:
                    print(f"Topic {topic} creation failed: {e}")
        except Exception as e:
            print(f"Topic creation error: {e}")
    
    def delivery_report(self, err, msg):
        """Callback for message delivery reports"""
        if err is not None:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def produce_parking_event(self, event_data: Dict):
        """
        Produce parking event to Kafka
        Event types: spot_occupied, spot_vacated, spot_reserved
        """
        if self.producer is None:
            print("Producer not initialized")
            return
        
        event_data['timestamp'] = event_data.get('timestamp', datetime.utcnow().isoformat())
        event_data['event_id'] = f"EVT_{int(time.time() * 1000)}"
        
        try:
            self.producer.produce(
                self.topics['parking_events'],
                key=event_data.get('spot_id', '').encode('utf-8'),
                value=json.dumps(event_data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Failed to produce parking event: {e}")
    
    def produce_traffic_update(self, traffic_data: Dict):
        """
        Produce traffic update to Kafka
        Includes traffic volume, speed, congestion level
        """
        if self.producer is None:
            print("Producer not initialized")
            return
        
        traffic_data['timestamp'] = traffic_data.get('timestamp', datetime.utcnow().isoformat())
        traffic_data['update_id'] = f"TRF_{int(time.time() * 1000)}"
        
        try:
            self.producer.produce(
                self.topics['traffic_updates'],
                key=traffic_data.get('zone', '').encode('utf-8'),
                value=json.dumps(traffic_data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Failed to produce traffic update: {e}")
    
    def produce_prediction(self, prediction_data: Dict):
        """
        Produce AI prediction to Kafka
        Includes occupancy forecast, confidence, horizon
        """
        if self.producer is None:
            print("Producer not initialized")
            return
        
        prediction_data['timestamp'] = prediction_data.get('timestamp', datetime.utcnow().isoformat())
        prediction_data['prediction_id'] = f"PRD_{int(time.time() * 1000)}"
        
        try:
            self.producer.produce(
                self.topics['predictions'],
                key=prediction_data.get('spot_id', '').encode('utf-8'),
                value=json.dumps(prediction_data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Failed to produce prediction: {e}")
    
    def produce_alert(self, alert_data: Dict):
        """
        Produce system alert to Kafka
        Alert types: high_occupancy, low_availability, system_error
        """
        if self.producer is None:
            print("Producer not initialized")
            return
        
        alert_data['timestamp'] = alert_data.get('timestamp', datetime.utcnow().isoformat())
        alert_data['alert_id'] = f"ALT_{int(time.time() * 1000)}"
        
        try:
            self.producer.produce(
                self.topics['alerts'],
                key=alert_data.get('alert_type', '').encode('utf-8'),
                value=json.dumps(alert_data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Failed to produce alert: {e}")
    
    def flush(self):
        """Flush pending messages"""
        if self.producer:
            self.producer.flush()
    
    def register_message_handler(self, topic: str, handler: Callable):
        """Register callback function for topic messages"""
        self.message_handlers[topic] = handler
    
    def consume_messages(self, topic: str, timeout: float = 1.0):
        """
        Consume messages from Kafka topic
        Runs in separate thread
        """
        try:
            consumer = Consumer(self.consumer_config)
            consumer.subscribe([topic])
            self.consumers[topic] = consumer
            
            print(f"Started consuming from topic: {topic}")
            
            while self.running:
                msg = consumer.poll(timeout=timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"Consumer error: {msg.error()}")
                        break
                
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    if topic in self.message_handlers:
                        self.message_handlers[topic](message_data)
                    else:
                        print(f"Received message from {topic}: {message_data}")
                
                except json.JSONDecodeError as e:
                    print(f"Failed to decode message: {e}")
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        except Exception as e:
            print(f"Consumer error for topic {topic}: {e}")
        finally:
            if topic in self.consumers:
                self.consumers[topic].close()
                del self.consumers[topic]
    
    def start_consuming(self, topics: List[str]):
        """Start consuming from multiple topics in separate threads"""
        self.running = True
        
        threads = []
        for topic in topics:
            thread = threading.Thread(target=self.consume_messages, args=(topic,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"Started consumer thread for topic: {topic}")
        
        return threads
    
    def stop_consuming(self):
        """Stop all consumers"""
        self.running = False
        
        for consumer in self.consumers.values():
            consumer.close()
        
        self.consumers.clear()
        print("All consumers stopped")
    
    def simulate_iot_magnetometer_stream(self, duration_seconds: int = 60, 
                                        frequency_seconds: int = 5,
                                        num_sensors: int = 100):
        """
        Simulate IoT magnetometer sensor stream
        Frequency: 5 seconds per sensor
        Payload: sensor_id, magnetic_field_strength, vehicle_detected, coordinates
        """
        print(f"Simulating magnetometer stream for {duration_seconds} seconds...")
        
        start_time = time.time()
        event_count = 0
        
        while time.time() - start_time < duration_seconds:
            for sensor_id in range(num_sensors):
                magnetic_strength = np.random.uniform(20, 80)
                vehicle_detected = magnetic_strength > 50
                
                event_data = {
                    'sensor_id': f"MAG_{sensor_id:04d}",
                    'magnetic_field_strength': float(magnetic_strength),
                    'vehicle_detected': bool(vehicle_detected),
                    'latitude': 37.7749 + np.random.uniform(-0.05, 0.05),
                    'longitude': -122.4194 + np.random.uniform(-0.05, 0.05),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.produce_parking_event(event_data)
                event_count += 1
            
            time.sleep(frequency_seconds)
        
        self.flush()
        print(f"Magnetometer stream complete. Produced {event_count} events")
        
        return event_count
    
    def simulate_cctv_edge_stream(self, duration_seconds: int = 60,
                                  frequency_seconds: int = 10,
                                  num_cameras: int = 20):
        """
        Simulate CCTV edge processing stream
        Frequency: 10 seconds per camera
        Payload: camera_id, detections (class, confidence, bbox), location
        """
        print(f"Simulating CCTV stream for {duration_seconds} seconds...")
        
        start_time = time.time()
        event_count = 0
        
        vehicle_classes = ['car', 'truck', 'motorcycle', 'van', 'bus']
        
        while time.time() - start_time < duration_seconds:
            for camera_id in range(num_cameras):
                num_detections = np.random.randint(0, 5)
                detections = []
                
                for _ in range(num_detections):
                    detection = {
                        'class': np.random.choice(vehicle_classes),
                        'confidence': float(np.random.uniform(0.7, 0.99)),
                        'bbox': {
                            'x': float(np.random.uniform(0, 800)),
                            'y': float(np.random.uniform(0, 600)),
                            'width': float(np.random.uniform(50, 200)),
                            'height': float(np.random.uniform(50, 150))
                        }
                    }
                    detections.append(detection)
                
                event_data = {
                    'camera_id': f"CAM_{camera_id:03d}",
                    'detections': detections,
                    'latitude': 37.7749 + np.random.uniform(-0.05, 0.05),
                    'longitude': -122.4194 + np.random.uniform(-0.05, 0.05),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.produce_parking_event(event_data)
                event_count += 1
            
            time.sleep(frequency_seconds)
        
        self.flush()
        print(f"CCTV stream complete. Produced {event_count} events")
        
        return event_count
    
    def simulate_user_gps_stream(self, duration_seconds: int = 60,
                                num_users: int = 50):
        """
        Simulate user GPS event stream
        Event-driven: search, arrival, departure, reservation
        Payload: user_id, event_type, spot_id, duration
        """
        print(f"Simulating user GPS stream for {duration_seconds} seconds...")
        
        start_time = time.time()
        event_count = 0
        
        event_types = ['search', 'arrival', 'departure', 'reservation']
        
        while time.time() - start_time < duration_seconds:
            num_events = np.random.randint(1, 10)
            
            for _ in range(num_events):
                event_data = {
                    'user_id': f"USR_{np.random.randint(1, num_users):04d}",
                    'event_type': np.random.choice(event_types),
                    'spot_id': f"SPOT_{np.random.randint(1, 1000):05d}",
                    'latitude': 37.7749 + np.random.uniform(-0.05, 0.05),
                    'longitude': -122.4194 + np.random.uniform(-0.05, 0.05),
                    'duration_minutes': int(np.random.uniform(30, 240)),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.produce_parking_event(event_data)
                event_count += 1
            
            time.sleep(1)
        
        self.flush()
        print(f"User GPS stream complete. Produced {event_count} events")
        
        return event_count
    
    def get_stream_statistics(self) -> Dict:
        """Get streaming statistics and throughput metrics"""
        return {
            'producer_initialized': self.producer is not None,
            'active_consumers': len(self.consumers),
            'configured_topics': list(self.topics.values()),
            'message_handlers': list(self.message_handlers.keys())
        }
