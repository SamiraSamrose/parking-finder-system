# Deployment Guide

Complete guide for deploying the Parking Finder System to various environments.

## Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- Cloud provider account (for cloud deployments)
- 8GB RAM minimum
- 10GB disk space

## Local Development Deployment

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/samirasamrose/parking-finder-system.git
cd parking-finder-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 3. Run Application

```bash
# Development mode
python run.py

# Access at http://localhost:5000
```

## Production Deployment

### Using Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 run:app

# With logging
gunicorn -w 4 -b 0.0.0.0:5000 \
  --access-logfile access.log \
  --error-logfile error.log \
  --log-level info \
  run:app
```

### Using Docker

**1. Create Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]
```

**2. Build and Run:**

```bash
# Build image
docker build -t parking-finder .

# Run container
docker run -d -p 5000:5000 --env-file .env parking-finder

# View logs
docker logs -f <container_id>
```

**3. Docker Compose:**

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

Run with Docker Compose:

```bash
docker-compose up -d
```

## Cloud Deployments

### Google Cloud Platform

**1. Cloud Run Deployment:**

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Deploy to Cloud Run
gcloud run deploy parking-finder \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300

# Set environment variables
gcloud run services update parking-finder \
  --update-env-vars GOOGLE_CLOUD_PROJECT_ID=project-id
```

**2. Google Kubernetes Engine:**

```bash
# Create cluster
gcloud container clusters create parking-finder-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --region us-central1

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Expose service
kubectl expose deployment parking-finder --type=LoadBalancer --port 80 --target-port 5000
```

### AWS Deployment

**1. Elastic Beanstalk:**

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB
eb init -p python-3.9 parking-finder --region us-east-1

# Create environment
eb create parking-finder-env

# Deploy
eb deploy

# Set environment variables
eb setenv GOOGLE_CLOUD_PROJECT_ID=project-id

# Open application
eb open
```

**2. ECS Fargate:**

Create `task-definition.json`:

```json
{
  "family": "parking-finder",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "parking-finder",
      "image": "your-ecr-repo/parking-finder:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ]
    }
  ]
}
```

Deploy:

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name parking-finder-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster parking-finder-cluster \
  --service-name parking-finder-service \
  --task-definition parking-finder \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Azure Deployment

**App Service:**

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name parking-finder-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name parking-finder-plan \
  --resource-group parking-finder-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group parking-finder-rg \
  --plan parking-finder-plan \
  --name parking-finder-app \
  --runtime "PYTHON|3.9"

# Deploy code
az webapp deployment source config-zip \
  --resource-group parking-finder-rg \
  --name parking-finder-app \
  --src parking-finder.zip

# Configure environment variables
az webapp config appsettings set \
  --resource-group parking-finder-rg \
  --name parking-finder-app \
  --settings FLASK_ENV=production
```

## Nginx Configuration

Create `nginx.conf`:

```nginx
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    include mime.types;
    default_type application/octet-stream;

    upstream parking_finder {
        server web:5000;
    }

    server {
        listen 80;
        server_name parking-finder.example.com;

        client_max_body_size 10M;

        location / {
            proxy_pass http://parking_finder;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        location /static {
            alias /app/frontend/static;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## SSL/TLS Configuration

### Using Let's Encrypt

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d parking-finder.example.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Manual SSL Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name parking-finder.example.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # ... rest of configuration
}

server {
    listen 80;
    server_name parking-finder.example.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring Setup

### Datadog Agent

```bash
# Install Datadog Agent
DD_API_KEY=your_api_key DD_SITE="datadoghq.com" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

# Configure APM
echo "apm_enabled: true" >> /etc/datadog-agent/datadog.yaml

# Restart agent
sudo systemctl restart datadog-agent
```

### Application Monitoring

Add to `run.py`:

```python
from ddtrace import tracer, patch_all

patch_all()

# Configure tracer
tracer.configure(
    hostname='parking-finder',
    port=8126,
)
```

## Database Setup (Optional)

### PostgreSQL

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb parking_finder

# Create user
sudo -u postgres createuser -P parking_user

# Grant privileges
sudo -u postgres psql
GRANT ALL PRIVILEGES ON DATABASE parking_finder TO parking_user;
```

### Redis Cache

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

## Performance Tuning

### Gunicorn Workers

```bash
# Calculate optimal workers
workers = (2 × CPU_cores) + 1

# For 4 CPU cores
gunicorn -w 9 -b 0.0.0.0:5000 run:app
```

### Python Optimization

```python
# requirements.txt additions
gevent==23.9.1
greenlet==3.0.1

# Use gevent worker
gunicorn -w 4 -k gevent --worker-connections 1000 -b 0.0.0.0:5000 run:app
```

### Memory Optimization

```bash
# Limit memory usage
gunicorn -w 4 --max-requests 1000 --max-requests-jitter 100 -b 0.0.0.0:5000 run:app
```

## Logging Configuration

### Application Logging

```python
# backend/config.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Centralized Logging

```bash
# Install Fluentd
curl -L https://toolbelt.treasuredata.com/sh/install-ubuntu-focal-td-agent4.sh | sh

# Configure Fluentd
sudo nano /etc/td-agent/td-agent.conf

# Start Fluentd
sudo systemctl start td-agent
```

## Backup Strategy

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump parking_finder > $BACKUP_DIR/db_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Delete old backups (keep 7 days)
find $BACKUP_DIR -type f -mtime +7 -delete
```

Schedule with cron:

```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

## Health Checks

### Application Health Check

```python
# Add to routes.py
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200
```

### Kubernetes Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /api/health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Troubleshooting

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

**Issue: Permission denied**
```bash
# Fix permissions
chmod +x run.py
chmod -R 755 data/ models/
```

**Issue: Out of memory**
```bash
# Reduce worker count
gunicorn -w 2 -b 0.0.0.0:5000 run:app

# Enable swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Maintenance

### Update Deployment

```bash
# Pull latest code
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart parking-finder
```

### Zero-Downtime Deployment

```bash
# Using Gunicorn
kill -HUP <gunicorn_master_pid>

# Using systemd
sudo systemctl reload parking-finder
```

## Security Checklist

- [ ] Environment variables configured securely
- [ ] HTTPS/TLS enabled
- [ ] Firewall rules configured
- [ ] API rate limiting enabled
- [ ] Input validation implemented
- [ ] SQL injection protection enabled
- [ ] XSS protection enabled
- [ ] CSRF protection enabled
- [ ] Security headers configured
- [ ] Regular security updates applied

## Post-Deployment Verification

```bash
# Check API health
curl https://parking-finder.example.com/api/health

# Test parking search
curl -X POST https://parking-finder.example.com/api/parking/search \
  -H "Content-Type: application/json" \
  -d '{"latitude": 37.7749, "longitude": -122.4194}'

# Check logs
tail -f /var/log/parking-finder/app.log

# Monitor resources
htop
