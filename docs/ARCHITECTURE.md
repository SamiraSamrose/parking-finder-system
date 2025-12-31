# System Architecture

Comprehensive architecture documentation for the Parking Finder System.

## Overview

The Parking Finder System is built on a microservices-inspired architecture with modular components that can scale independently. The system combines real-time data processing, machine learning, cloud services, and monitoring into a cohesive platform.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                         Frontend Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Dashboard │  │Analytics │  │Monitoring│  │  Voice   │       │
│  │   UI     │  │    UI    │  │    UI    │  │Interface │       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └────┬─────┘       │
│        │             │             │            │             │
│        └─────────────┴─────────────┴────────────┘             │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
                       ┌─────▼─────┐
                       │Flask REST │
                       │    API    │
                       └─────┬─────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │  ML     │         │Business │         │External │
   │ Models  │         │ Logic   │         │Services │
   └─────────┘         └─────────┘         └─────────┘
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │XGBoost  │         │Parking  │         │ Kafka   │
   │  LSTM   │         │ Finder  │         │ Stream  │
   │Transform│         │ Service │         └─────────┘
   │   GNN   │         └─────────┘               │
   └─────────┘               │              ┌────▼────┐
        │              ┌────▼────┐          │Datadog  │
   ┌────▼────┐         │ Voice   │          │Monitor  │
   │  Data   │         │Interface│          └─────────┘
   │ Loader  │         └─────────┘               │
   └─────────┘               │              ┌────▼────┐
                        ┌────▼────┐         │ Google  │
                        │ Google  │         │  Cloud  │
                        │Services │         └─────────┘
                        └─────────┘
```

## Component Architecture

### 1. Frontend Layer

**Technology:** HTML5, CSS3, JavaScript (ES6+), Plotly.js

**Components:**
- Dashboard UI: Main interface for parking search and recommendations
- Analytics UI: Comprehensive data analysis and visualizations
- Monitoring UI: Real-time system monitoring and observability
- Voice Interface: Natural language query processing

**Responsibilities:**
- User interaction and input validation
- Data visualization and presentation
- Real-time updates via API polling
- Responsive design for multiple devices

**Communication:** RESTful API calls to backend

### 2. API Layer

**Technology:** Flask 3.0, Flask-CORS

**Components:**
- Route handlers for all endpoints
- Request validation and sanitization
- Response formatting and error handling
- CORS configuration for cross-origin requests

**Endpoints:**
- `/api/health` - System health checks
- `/api/initialize` - System initialization
- `/api/parking/*` - Parking operations
- `/api/models/*` - ML model operations
- `/api/monitoring/*` - Monitoring operations
- `/api/voice/*` - Voice interface
- `/api/google/*` - Google integration

**Design Patterns:**
- RESTful API design
- MVC pattern
- Dependency injection
- Factory pattern for service creation

### 3. Business Logic Layer

**Technology:** Python 3.8+

**Components:**

#### Parking Finder Service
- **Purpose:** Core parking detection and recommendation engine
- **Algorithm:** Composite scoring (50% availability + 30% distance + 20% cost)
- **Features:**
  - Haversine distance calculation
  - ML-based availability prediction
  - Zone statistics aggregation
  - Historical pattern analysis

#### Voice Interface Service
- **Purpose:** Natural language processing and agent routing
- **Agents:**
  - Parking Assistant: Spot finding and directions
  - Traffic Advisor: Route optimization
  - Payment Coordinator: Transaction handling
  - Support Specialist: Issue resolution
- **Features:**
  - Intent classification
  - Context management
  - Response generation
  - Speech synthesis

#### Google Integration Service
- **Purpose:** Google Cloud Platform services integration
- **Features:**
  - 3D Maps tiles configuration
  - Curb-level clearance calculation
  - Multi-agent orchestration
  - BigQuery geospatial queries
  - Calendar event creation
  - Sheets expense logging

#### Kafka Streaming Service
- **Purpose:** Real-time event processing
- **Features:**
  - IoT magnetometer stream processing
  - CCTV edge processing
  - User GPS event handling
  - Topic management
  - Producer/consumer implementation

#### Datadog Monitor Service
- **Purpose:** System monitoring and observability
- **Features:**
  - Metric tracking (latency, tokens, accuracy)
  - Hallucination detection
  - Prompt injection detection
  - Incident creation
  - Security event logging

### 4. Machine Learning Layer

**Technology:** TensorFlow 2.15, XGBoost 2.0, scikit-learn

**Models:**

#### XGBoost Model
- **Type:** Gradient Boosting Classifier
- **Input:** 11 engineered features
- **Output:** Binary classification (occupied/available)
- **Performance:** 89% accuracy, 0.89 F1 score
- **Training Time:** ~2 minutes
- **Use Case:** Primary occupancy prediction

#### LSTM Model
- **Type:** Recurrent Neural Network
- **Architecture:** 64→32 LSTM units
- **Input:** 24-hour sequence of 5 features
- **Output:** Occupancy probability
- **Performance:** 85% accuracy, R² 0.80
- **Use Case:** Time-series forecasting

#### Transformer Model
- **Type:** Attention-based Neural Network
- **Architecture:** 2 multi-head attention layers (4 heads each)
- **Input:** 24-hour sequence of 5 features
- **Output:** Occupancy rate prediction
- **Performance:** 91% accuracy, R² 0.90
- **Use Case:** Long-term pattern recognition

#### GNN Model
- **Type:** Graph Neural Network
- **Architecture:** Dense layers simulating graph convolution
- **Input:** Spatial node features
- **Output:** Node-level occupancy prediction
- **Performance:** 87% accuracy, R² 0.85
- **Use Case:** Spatial relationship modeling

**Feature Engineering:**
- Cyclic encoding for temporal features
- Label encoding for categorical features
- Normalization for numerical features
- Spatial feature extraction

### 5. Data Layer

**Technology:** Pandas, NumPy

**Data Sources:**

#### Metro Interstate Traffic Dataset
- **Source:** UCI ML Repository
- **Size:** 48,204 records
- **Features:** 8 columns (traffic, weather, temporal)
- **Update Frequency:** Batch loading

#### Synthetic Parking Data
- **Size:** 50,000 sensor records
- **Generation:** Correlated with traffic patterns
- **Formula:** occupancy = 0.3 + 0.5×traffic×time + 0.1×weather

#### Curb Regulations
- **Size:** 2,000 segments
- **Features:** Regulation type, time restrictions, vehicle types

#### Historical Patterns
- **Size:** 17,520 records (2 years hourly)
- **Aggregation:** By zone and hour
- **Metrics:** Occupancy rate, turnover, revenue

### 6. External Services Layer

**Services:**

#### Google Cloud Platform
- **Vertex AI:** Multi-agent orchestration
- **BigQuery:** Geospatial queries
- **Maps API:** 3D tiles and navigation
- **Calendar API:** Reservation management
- **Sheets API:** Expense tracking

#### Confluent Kafka
- **Topics:** 6 topics for different event types
- **Throughput:** 12,500 messages/second
- **Latency:** 20-80ms processing time

#### Datadog
- **Metrics:** 8 tracked metrics
- **Alerts:** 5 detection rules
- **Dashboard:** 8 widgets

#### ElevenLabs
- **API:** Voice synthesis
- **Models:** eleven_multilingual_v2
- **Latency:** 250-400ms per synthesis

## Data Flow

### Parking Search Flow

```
1. User Input (lat, lon, preferences)
   ↓
2. API Validation
   ↓
3. Parking Finder Service
   ├→ Calculate distances
   ├→ ML prediction (XGBoost)
   ├→ Composite scoring
   └→ Filter and rank
   ↓
4. Response Formatting
   ↓
5. Frontend Display
```

### Real-time Streaming Flow

```
1. IoT/CCTV/GPS Events
   ↓
2. Kafka Producer
   ↓
3. Kafka Topics
   ├→ parking-events
   ├→ traffic-updates
   └→ predictions
   ↓
4. Kafka Consumer
   ↓
5. Stream Processing
   ├→ Aggregation
   ├→ Enrichment
   └→ ML inference
   ↓
6. Database/Cache Update
   ↓
7. Dashboard Update
```

### ML Training Flow

```
1. Data Loading
   ├→ Traffic data
   ├→ Parking data
   └→ Historical data
   ↓
2. Feature Engineering
   ├→ Temporal features
   ├→ Spatial features
   └→ Contextual features
   ↓
3. Train/Test Split
   ↓
4. Model Training
   ├→ XGBoost (2 min)
   ├→ LSTM (5 min)
   ├→ Transformer (8 min)
   └→ GNN (4 min)
   ↓
5. Model Evaluation
   ├→ Accuracy metrics
   ├→ Feature importance
   └→ Error analysis
   ↓
6. Model Persistence
   ↓
7. API Integration
```

## Scalability Considerations

### Horizontal Scaling

**API Layer:**
- Multiple Flask instances behind load balancer
- Stateless design for easy replication
- Session management via external cache

**ML Layer:**
- Model serving via TensorFlow Serving
- Batch prediction for efficiency
- Model versioning and A/B testing

**Data Layer:**
- Database sharding by geographic region
- Read replicas for query distribution
- Caching layer (Redis/Memcached)

### Vertical Scaling

**Compute:**
- GPU acceleration for ML training
- Multi-core processing for data pipelines
- Memory optimization for large datasets

**Storage:**
- SSD for database storage
- Object storage for model artifacts
- CDN for static assets

### Performance Optimization

**Caching Strategy:**
- API response caching (5-minute TTL)
- Model prediction caching
- Database query result caching

**Async Processing:**
- Background jobs for model training
- Async API calls for external services
- Queue-based task processing

**Database Optimization:**
- Indexed queries on frequently accessed columns
- Materialized views for complex aggregations
- Connection pooling

## Security Architecture

### Authentication & Authorization
- API key authentication (production)
- Role-based access control
- JWT token management

### Data Security
- HTTPS/TLS for all communications
- Encryption at rest for sensitive data
- Secure credential storage (environment variables)

### Input Validation
- Request schema validation
- SQL injection prevention
- XSS protection

### Monitoring & Detection
- Hallucination detection for LLM outputs
- Prompt injection detection
- Anomaly detection for unusual patterns
- Security event logging

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python virtual environment
├── Local Flask server
├── Local data storage
└── Development credentials
```

### Production Environment
```
Cloud Infrastructure
├── Load Balancer
│   └── Multiple Flask instances
├── Application Servers
│   ├── ML models
│   └── Business logic
├── Data Layer
│   ├── PostgreSQL database
│   └── Redis cache
├── External Services
│   ├── Kafka cluster
│   ├── Datadog monitoring
│   └── Google Cloud Platform
└── Storage
    ├── Model artifacts (Cloud Storage)
    └── Static assets (CDN)
```

### Container Architecture
```
Docker Containers
├── web: Flask application
├── worker: Background jobs
├── ml: Model serving
└── nginx: Reverse proxy
```

## Monitoring & Observability

### Metrics Collected
- API latency (p50, p95, p99)
- Request rate and error rate
- ML model accuracy and latency
- Database query performance
- External service latency

### Logging
- Application logs (INFO, WARNING, ERROR)
- Access logs (request/response)
- Error logs with stack traces
- Audit logs for sensitive operations

### Alerting
- High latency alerts
- Error rate thresholds
- Model accuracy degradation
- Resource utilization alerts

### Dashboards
- System overview dashboard
- ML performance dashboard
- Business metrics dashboard
- Security monitoring dashboard

## Disaster Recovery

### Backup Strategy
- Database backups (hourly incremental, daily full)
- Model artifact backups
- Configuration backups
- Code repository backups

### Recovery Procedures
- Database point-in-time recovery
- Model rollback procedures
- Configuration rollback
- Failover to backup region

### Business Continuity
- Multi-region deployment
- Automated failover
- Data replication
- Regular disaster recovery drills

## Future Architecture Enhancements

### Planned Improvements
1. Microservices architecture with service mesh
2. Event-driven architecture with message queues
3. GraphQL API for flexible queries
4. Real-time WebSocket connections
5. Edge computing for low-latency predictions
6. Blockchain integration for decentralized marketplace
7. Mobile-first architecture with native apps
8. AI-powered auto-scaling

### Technology Roadmap
- Kubernetes orchestration
- Service mesh (Istio)
- Serverless functions
- Stream processing (Apache Flink)
- Graph database (Neo4j)
- Time-series database (InfluxDB)

