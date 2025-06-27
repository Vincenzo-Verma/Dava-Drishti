"""
Configuration management for Forest Fire Prediction System
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/forest_fire_db')
    MONGODB_URL = os.environ.get('MONGODB_URL', 'mongodb://localhost:27017/forest_fire_ml')
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True
    CELERY_TASK_TRACK_STARTED = True
    CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
    
    # Cache configuration
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = 'data/uploads'
    ALLOWED_EXTENSIONS = {'tif', 'tiff', 'nc', 'csv', 'json', 'geojson'}
    
    # Data directories
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
    
    # Model configuration
    MODEL_CONFIG = {
        'unet': {
            'input_shape': (256, 256, 10),
            'filters': [64, 128, 256, 512],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        },
        'lstm': {
            'sequence_length': 30,
            'hidden_units': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50
        },
        'cellular_automata': {
            'grid_size': (1000, 1000),
            'cell_size': 30,  # meters
            'time_step': 3600,  # 1 hour in seconds
            'max_iterations': 72  # 3 days
        }
    }
    
    # Data sources configuration
    DATA_SOURCES = {
        'era5': {
            'api_url': 'https://cds.climate.copernicus.eu/api/v2',
            'variables': ['2m_temperature', '2m_relative_humidity', '10m_u_component_of_wind', 
                         '10m_v_component_of_wind', 'total_precipitation'],
            'resolution': 0.1,  # degrees
            'update_frequency': 'daily'
        },
        'viirs': {
            'api_url': 'https://firms.modaps.eosdis.nasa.gov/api/area/csv',
            'satellite': 'VIIRS',
            'update_frequency': 'hourly'
        },
        'dem': {
            'source': 'bhoonidhi',
            'resolution': 30,  # meters
            'update_frequency': 'static'
        },
        'lulc': {
            'source': 'bhuvan',
            'resolution': 30,  # meters
            'update_frequency': 'yearly'
        }
    }
    
    # API configuration
    API_RATE_LIMIT = '100 per minute'
    API_TIMEOUT = 30  # seconds
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(LOGS_DIR, 'app.log')
    
    # Security configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Monitoring configuration
    ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'true').lower() == 'true'
    METRICS_PORT = int(os.environ.get('METRICS_PORT', 9090))
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    MLFLOW_EXPERIMENT_NAME = 'forest_fire_prediction'
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        'min_accuracy': 0.90,
        'min_f1_score': 0.85,
        'max_api_response_time': 2.0,  # seconds
        'min_system_uptime': 0.995
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production-specific settings
    CACHE_DEFAULT_TIMEOUT = 600  # 10 minutes
    API_RATE_LIMIT = '1000 per hour'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'postgresql://test_user:test_pass@localhost:5432/test_db'
    MONGODB_URL = 'mongodb://localhost:27017/test_ml'
    REDIS_URL = 'redis://localhost:6379/1'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 