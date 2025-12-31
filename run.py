import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import config
from backend.api import create_app


def main():
    """
    Main function to start the parking finder system
    Initializes configuration and starts Flask server
    """
    print("=" * 70)
    print("PARKING FINDER SYSTEM")
    print("Intelligent Parking Space Detection with AI/ML Integration")
    print("=" * 70)
    print()
    
    if not config.validate():
        print("Warning: Some configuration parameters are missing.")
        print("The system will run in limited mode.")
        print("Please check .env file for proper configuration.")
        print()
    
    print(f"Starting Flask API server...")
    print(f"Host: {config.flask.host}")
    print(f"Port: {config.flask.port}")
    print(f"Debug: {config.flask.debug}")
    print()
    print("API Endpoints:")
    print("  - Health Check: GET /api/health")
    print("  - Initialize: POST /api/initialize")
    print("  - Search Parking: POST /api/parking/search")
    print("  - Recommendations: POST /api/parking/recommendations")
    print("  - Zone Statistics: GET /api/parking/zones")
    print("  - Patterns Analysis: GET /api/parking/patterns")
    print("  - Model Prediction: POST /api/models/predict")
    print("  - Voice Query: POST /api/voice/query")
    print("  - Google Integration: POST /api/google/*")
    print()
    print("Frontend URLs:")
    print(f"  - Dashboard: http://{config.flask.host}:{config.flask.port}/")
    print(f"  - Analytics: http://{config.flask.host}:{config.flask.port}/analytics")
    print(f"  - Monitoring: http://{config.flask.host}:{config.flask.port}/monitoring")
    print()
    
    app = create_app(config)
    
    @app.route('/')
    def index():
        """Serve main dashboard page"""
        return app.send_static_file('index.html') if os.path.exists('frontend/static/index.html') else \
               '<html><body><h1>Parking Finder System</h1><p>API is running. Use /api/* endpoints.</p></body></html>'
    
    @app.route('/analytics')
    def analytics():
        """Serve analytics page"""
        return app.send_static_file('analytics.html') if os.path.exists('frontend/static/analytics.html') else \
               '<html><body><h1>Analytics Dashboard</h1></body></html>'
    
    @app.route('/monitoring')
    def monitoring():
        """Serve monitoring page"""
        return app.send_static_file('monitoring.html') if os.path.exists('frontend/static/monitoring.html') else \
               '<html><body><h1>Monitoring Dashboard</h1></body></html>'
    
    try:
        app.run(
            host=config.flask.host,
            port=config.flask.port,
            debug=config.flask.debug
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
