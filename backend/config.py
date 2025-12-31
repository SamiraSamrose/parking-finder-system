import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GoogleCloudConfig:
    """Google Cloud Platform configuration"""
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "parking-finder-ai")
    location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    gemini_model: str = "gemini-1.5-pro-001"
    gemini_flash_model: str = "gemini-2.5-flash"
    embedding_model: str = "textembedding-gecko@003"
    credentials_path: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


@dataclass
class DatadogConfig:
    """Datadog monitoring configuration"""
    api_key: str = os.getenv("DATADOG_API_KEY", "")
    app_key: str = os.getenv("DATADOG_APP_KEY", "")
    site: str = os.getenv("DATADOG_SITE", "datadoghq.com")
    service_name: str = "parking-finder"
    environment: str = "production"


@dataclass
class KafkaConfig:
    """Confluent Kafka configuration"""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_key: str = os.getenv("KAFKA_API_KEY", "")
    api_secret: str = os.getenv("KAFKA_API_SECRET", "")
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "PLAIN"
    
    topics = {
        "parking_events": "parking-events",
        "traffic_updates": "traffic-updates",
        "predictions": "predictions",
        "alerts": "alerts",
        "user_requests": "user-requests",
        "recommendations": "recommendations"
    }


@dataclass
class ElevenLabsConfig:
    """ElevenLabs voice API configuration"""
    api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    model: str = "eleven_multilingual_v2"
    stability: float = 0.75
    similarity_boost: float = 0.85
    style: float = 0.5


@dataclass
class ModelConfig:
    """ML model configuration"""
    refresh_interval_seconds: int = 30
    prediction_horizon_minutes: int = 60
    confidence_threshold: float = 0.75
    sequence_length: int = 24
    
    xgboost_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    lstm_params = {
        "units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32
    }
    
    transformer_params = {
        "num_heads": 4,
        "key_dim": 32,
        "ff_dim": 128,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 64
    }


@dataclass
class FlaskConfig:
    """Flask API configuration"""
    host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    cors_origins: str = "*"


class SystemConfig:
    """Main system configuration aggregator"""
    
    def __init__(self):
        self.google_cloud = GoogleCloudConfig()
        self.datadog = DatadogConfig()
        self.kafka = KafkaConfig()
        self.elevenlabs = ElevenLabsConfig()
        self.model = ModelConfig()
        self.flask = FlaskConfig()
    
    def validate(self) -> bool:
        """Validate critical configuration parameters"""
        critical_fields = [
            (self.google_cloud.project_id, "Google Cloud Project ID"),
            (self.datadog.api_key, "Datadog API Key"),
            (self.kafka.bootstrap_servers, "Kafka Bootstrap Servers")
        ]
        
        missing = [name for value, name in critical_fields if not value or value == ""]
        
        if missing:
            print(f"Warning: Missing configuration for: {', '.join(missing)}")
            return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "google_cloud": {
                "project_id": self.google_cloud.project_id,
                "location": self.google_cloud.location,
                "gemini_model": self.google_cloud.gemini_model
            },
            "datadog": {
                "service_name": self.datadog.service_name,
                "environment": self.datadog.environment
            },
            "kafka": {
                "topics": self.kafka.topics
            },
            "model": {
                "confidence_threshold": self.model.confidence_threshold,
                "prediction_horizon_minutes": self.model.prediction_horizon_minutes
            }
        }


config = SystemConfig()
