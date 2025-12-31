import time
from typing import Dict, List, Optional
from datetime import datetime
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.series import Series
from datadog_api_client.v1.model.point import Point
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
import hashlib


class DatadogMonitorService:
    """
    Datadog monitoring and observability service
    Tracks LLM latency, token usage, prediction accuracy, and security events
    """
    
    def __init__(self, config):
        self.config = config
        self.configuration = Configuration()
        self.configuration.api_key['apiKeyAuth'] = config.datadog.api_key
        self.configuration.api_key['appKeyAuth'] = config.datadog.app_key
        self.configuration.server_variables['site'] = config.datadog.site
        
        self.service_name = config.datadog.service_name
        self.environment = config.datadog.environment
        
        self.metrics_buffer = []
        self.logs_buffer = []
    
    def send_metric(self, metric_name: str, value: float, 
                   tags: Optional[List[str]] = None,
                   metric_type: str = 'gauge'):
        """
        Send metric to Datadog
        Metric types: gauge, count, rate, histogram
        """
        tags = tags or []
        tags.extend([
            f'service:{self.service_name}',
            f'env:{self.environment}'
        ])
        
        try:
            with ApiClient(self.configuration) as api_client:
                api_instance = MetricsApi(api_client)
                
                series = Series(
                    metric=f'parking.{metric_name}',
                    type=metric_type,
                    points=[Point([int(time.time()), value])],
                    tags=tags
                )
                
                payload = MetricsPayload(series=[series])
                
                api_instance.submit_metrics(body=payload)
                
                print(f"Metric sent: {metric_name} = {value} (tags: {tags})")
        
        except Exception as e:
            print(f"Failed to send metric {metric_name}: {e}")
    
    def track_llm_latency(self, latency_ms: float, model: str, 
                         endpoint: str, region: str = 'us-central1'):
        """
        Track LLM request latency
        Metric: llm.latency (histogram)
        """
        tags = [
            f'model:{model}',
            f'endpoint:{endpoint}',
            f'region:{region}'
        ]
        
        self.send_metric('llm.latency', latency_ms, tags=tags, metric_type='histogram')
    
    def track_llm_tokens(self, prompt_tokens: int, completion_tokens: int,
                        model: str, user_id: str, intent: str):
        """
        Track LLM token usage
        Metrics: llm.prompt.tokens, llm.completion.tokens
        """
        tags = [
            f'model:{model}',
            f'user_id:{user_id}',
            f'intent:{intent}'
        ]
        
        self.send_metric('llm.prompt.tokens', prompt_tokens, tags=tags, metric_type='gauge')
        self.send_metric('llm.completion.tokens', completion_tokens, tags=tags, metric_type='gauge')
        
        total_tokens = prompt_tokens + completion_tokens
        self.send_metric('llm.total.tokens', total_tokens, tags=tags, metric_type='gauge')
    
    def track_llm_request(self, success: bool, model: str, error_type: Optional[str] = None):
        """
        Track LLM request success/failure
        Metric: llm.requests (count)
        """
        tags = [
            f'model:{model}',
            f'success:{str(success).lower()}'
        ]
        
        if error_type:
            tags.append(f'error_type:{error_type}')
        
        self.send_metric('llm.requests', 1, tags=tags, metric_type='count')
    
    def track_parking_occupancy(self, zone: str, occupancy_rate: float):
        """
        Track parking occupancy rate by zone
        Metric: parking.occupancy.rate (gauge)
        """
        tags = [f'zone:{zone}']
        
        self.send_metric('occupancy.rate', occupancy_rate, tags=tags, metric_type='gauge')
    
    def track_prediction_accuracy(self, accuracy: float, model: str):
        """
        Track ML model prediction accuracy
        Metric: parking.prediction.accuracy (gauge)
        """
        tags = [f'model:{model}']
        
        self.send_metric('prediction.accuracy', accuracy, tags=tags, metric_type='gauge')
    
    def track_hallucination_score(self, score: float, model: str, query_type: str):
        """
        Track LLM hallucination detection score
        Score: 0-1, higher = more likely hallucination
        Metric: llm.hallucination.score (gauge)
        """
        tags = [
            f'model:{model}',
            f'query_type:{query_type}'
        ]
        
        self.send_metric('llm.hallucination.score', score, tags=tags, metric_type='gauge')
        
        if score > 0.7:
            self.create_incident('high_hallucination_score', {
                'score': score,
                'model': model,
                'query_type': query_type
            })
    
    def detect_legal_hallucination(self, suggested_spot_id: str, 
                                  restricted_zones: List[str],
                                  restriction_status: str,
                                  restriction_reason: str) -> bool:
        """
        Detect if LLM suggested a legally restricted parking spot
        Returns True if hallucination detected
        """
        is_hallucination = False
        
        for zone in restricted_zones:
            if zone in suggested_spot_id and restriction_status == "Temporarily Restricted":
                is_hallucination = True
                
                hallucination_data = {
                    'spot_id': suggested_spot_id,
                    'restricted_zone': zone,
                    'restriction_reason': restriction_reason,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.create_incident('legal_hallucination', hallucination_data)
                
                self.send_metric('llm.compliance.violation', 1, 
                               tags=[f'violation_type:legal_hallucination',
                                    f'severity:critical'],
                               metric_type='count')
                
                print(f"Legal hallucination detected: {suggested_spot_id} in restricted zone {zone}")
                break
        
        return is_hallucination
    
    def detect_prompt_injection(self, user_input: str) -> Dict:
        """
        Detect prompt injection attempts using pattern matching
        Patterns: ignore instructions, bypass restrictions, admin mode
        """
        injection_patterns = [
            'ignore previous instructions',
            'you are now in admin mode',
            'bypass handicapped verification',
            'override payment requirements',
            'disregard parking restrictions',
            'system prompt',
            'new instructions',
            'forget everything'
        ]
        
        user_input_lower = user_input.lower()
        detected_patterns = []
        
        for pattern in injection_patterns:
            if pattern in user_input_lower:
                detected_patterns.append(pattern)
        
        is_injection = len(detected_patterns) > 0
        
        if is_injection:
            self.send_metric('llm.security.prompt_injection', 1,
                           tags=[f'severity:high'],
                           metric_type='count')
            
            self.log_security_event('prompt_injection_attempt', {
                'user_input': user_input[:200],
                'detected_patterns': detected_patterns,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return {
            'is_injection': is_injection,
            'detected_patterns': detected_patterns,
            'confidence': min(len(detected_patterns) * 0.3, 1.0)
        }
    
    def create_incident(self, incident_type: str, incident_data: Dict):
        """
        Create Datadog incident for critical events
        Incident types: high_hallucination_score, legal_hallucination, security_breach
        """
        incident = {
            'incident_type': incident_type,
            'service': self.service_name,
            'environment': self.environment,
            'timestamp': datetime.utcnow().isoformat(),
            'data': incident_data
        }
        
        print(f"Incident created: {incident_type}")
        print(f"Details: {incident_data}")
        
        self.send_metric('incidents.created', 1,
                       tags=[f'incident_type:{incident_type}'],
                       metric_type='count')
    
    def log_security_event(self, event_type: str, event_data: Dict):
        """
        Log security event to Datadog
        Event types: prompt_injection, unauthorized_access, data_breach
        """
        log_entry = {
            'event_type': event_type,
            'service': self.service_name,
            'environment': self.environment,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high',
            'data': event_data
        }
        
        print(f"Security event logged: {event_type}")
        
        self.send_metric('security.events', 1,
                       tags=[f'event_type:{event_type}'],
                       metric_type='count')
    
    def track_geofence(self, city: str, zone: str, geofence_id: str):
        """
        Track geofence boundary crossing
        Metric: llm.geofence.id (gauge)
        """
        tags = [
            f'city:{city}',
            f'zone:{zone}'
        ]
        
        geofence_hash = int(hashlib.md5(geofence_id.encode()).hexdigest()[:8], 16)
        
        self.send_metric('llm.geofence.id', geofence_hash, tags=tags, metric_type='gauge')
    
    def simulate_monitoring_data(self, duration_minutes: int = 60, 
                                samples_per_minute: int = 120) -> Dict:
        """
        Simulate monitoring data for dashboard visualization
        Generates LLM latency, token usage, success rates, prediction accuracy
        """
        print(f"Simulating {duration_minutes} minutes of monitoring data...")
        
        total_samples = duration_minutes * samples_per_minute
        
        models = ['gemini-1.5-pro', 'gemini-2.5-flash', 'claude-sonnet']
        intents = ['find_parking', 'get_directions', 'check_availability', 'reserve_spot']
        
        metrics_data = {
            'timestamps': [],
            'latencies': [],
            'token_usage': [],
            'success_rates': [],
            'hallucination_scores': [],
            'models': [],
            'intents': []
        }
        
        for i in range(total_samples):
            timestamp = time.time() - (total_samples - i) * 0.5
            
            model = models[i % len(models)]
            intent = intents[i % len(intents)]
            
            if model == 'gemini-2.5-flash':
                latency = np.random.normal(200, 50)
            elif model == 'gemini-1.5-pro':
                latency = np.random.normal(300, 75)
            else:
                latency = np.random.normal(250, 60)
            
            latency = max(50, latency)
            
            prompt_tokens = int(np.random.normal(150, 40))
            completion_tokens = int(np.random.normal(80, 25))
            
            success = np.random.random() > 0.03
            
            hallucination_score = np.random.beta(2, 8)
            
            metrics_data['timestamps'].append(timestamp)
            metrics_data['latencies'].append(latency)
            metrics_data['token_usage'].append(prompt_tokens + completion_tokens)
            metrics_data['success_rates'].append(1 if success else 0)
            metrics_data['hallucination_scores'].append(hallucination_score)
            metrics_data['models'].append(model)
            metrics_data['intents'].append(intent)
        
        print(f"Generated {total_samples} monitoring data points")
        
        return metrics_data
    
    def get_monitoring_summary(self) -> Dict:
        """Get summary of monitoring metrics"""
        return {
            'service_name': self.service_name,
            'environment': self.environment,
            'datadog_site': self.configuration.server_variables['site'],
            'metrics_tracked': [
                'llm.latency',
                'llm.tokens',
                'llm.requests',
                'parking.occupancy.rate',
                'parking.prediction.accuracy',
                'llm.hallucination.score',
                'llm.compliance.violation',
                'llm.security.prompt_injection'
            ]
        }