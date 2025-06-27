"""
Database initialization and connection management
"""

import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from pymongo import MongoClient
from app.utils.config import Config

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

class DatabaseManager:
    """Database connection and management class"""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_session = None
        self.mongo_client = None
        self.mongo_db = None
    
    def init_postgres(self, app):
        """Initialize PostgreSQL connection"""
        try:
            # Create engine
            self.postgres_engine = create_engine(
                app.config['DATABASE_URL'],
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            session_factory = sessionmaker(bind=self.postgres_engine)
            self.postgres_session = scoped_session(session_factory)
            
            # Test connection
            with self.postgres_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    def init_mongodb(self, app):
        """Initialize MongoDB connection"""
        try:
            # Create client
            self.mongo_client = MongoClient(app.config['MONGODB_URL'])
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            # Get database
            self.mongo_db = self.mongo_client.get_database()
            
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    def get_postgres_session(self):
        """Get PostgreSQL session"""
        return self.postgres_session()
    
    def get_mongodb_collection(self, collection_name):
        """Get MongoDB collection"""
        return self.mongo_db[collection_name]
    
    def close_connections(self):
        """Close all database connections"""
        if self.postgres_session:
            self.postgres_session.remove()
        
        if self.postgres_engine:
            self.postgres_engine.dispose()
        
        if self.mongo_client:
            self.mongo_client.close()
        
        logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

def init_database(app):
    """Initialize database connections"""
    try:
        db_manager.init_postgres(app)
        db_manager.init_mongodb(app)
        
        # Create tables if they don't exist
        create_tables()
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def create_tables():
    """Create database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(db_manager.postgres_engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise

def get_db_session():
    """Get database session for dependency injection"""
    session = db_manager.get_postgres_session()
    try:
        yield session
    finally:
        session.close()

# Database models
class FireEvent(Base):
    """Fire event database model"""
    __tablename__ = 'fire_events'
    
    id = Column(Integer, primary_key=True)
    event_date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    confidence = Column(Float)
    satellite = Column(String(50))
    brightness = Column(Float)
    scan = Column(Float)
    track = Column(Float)
    acq_date = Column(DateTime)
    acq_time = Column(String(10))
    satellite_time = Column(String(10))
    instrument = Column(String(50))
    confidence_level = Column(String(20))
    version = Column(String(10))
    bright_t31 = Column(Float)
    frp = Column(Float)
    daynight = Column(String(10))
    type = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WeatherData(Base):
    """Weather data database model"""
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature_2m = Column(Float)
    relative_humidity_2m = Column(Float)
    wind_u_10m = Column(Float)
    wind_v_10m = Column(Float)
    total_precipitation = Column(Float)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class TerrainData(Base):
    """Terrain data database model"""
    __tablename__ = 'terrain_data'
    
    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    slope = Column(Float)
    aspect = Column(Float)
    land_cover = Column(String(50))
    vegetation_type = Column(String(50))
    fuel_load = Column(Float)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionResult(Base):
    """Prediction result database model"""
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True)
    prediction_date = Column(DateTime, nullable=False)
    region = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    risk_level = Column(String(20))
    confidence = Column(Float)
    map_file_path = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SimulationResult(Base):
    """Simulation result database model"""
    __tablename__ = 'simulation_results'
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(String(100), unique=True, nullable=False)
    ignition_point_lat = Column(Float, nullable=False)
    ignition_point_lon = Column(Float, nullable=False)
    duration_hours = Column(Integer, nullable=False)
    affected_area_km2 = Column(Float)
    animation_file_path = Column(String(500))
    parameters = Column(JSON)
    status = Column(String(20), default='running')
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# MongoDB collections
class ModelMetadata:
    """Model metadata collection"""
    collection_name = 'model_metadata'
    
    @staticmethod
    def get_collection():
        return db_manager.get_mongodb_collection('model_metadata')
    
    @staticmethod
    def save_model_info(model_info):
        """Save model information"""
        collection = ModelMetadata.get_collection()
        return collection.insert_one(model_info)
    
    @staticmethod
    def get_model_info(model_name, version=None):
        """Get model information"""
        collection = ModelMetadata.get_collection()
        query = {'model_name': model_name}
        if version:
            query['version'] = version
        return collection.find_one(query, sort=[('created_at', -1)])

class TrainingHistory:
    """Training history collection"""
    collection_name = 'training_history'
    
    @staticmethod
    def get_collection():
        return db_manager.get_mongodb_collection('training_history')
    
    @staticmethod
    def save_training_record(record):
        """Save training record"""
        collection = TrainingHistory.get_collection()
        return collection.insert_one(record)
    
    @staticmethod
    def get_training_history(model_name, limit=100):
        """Get training history"""
        collection = TrainingHistory.get_collection()
        return list(collection.find(
            {'model_name': model_name},
            sort=[('created_at', -1)],
            limit=limit
        ))

class SystemMetrics:
    """System metrics collection"""
    collection_name = 'system_metrics'
    
    @staticmethod
    def get_collection():
        return db_manager.get_mongodb_collection('system_metrics')
    
    @staticmethod
    def save_metric(metric):
        """Save system metric"""
        collection = SystemMetrics.get_collection()
        return collection.insert_one(metric)
    
    @staticmethod
    def get_metrics(metric_type, start_time, end_time):
        """Get system metrics for a time range"""
        collection = SystemMetrics.get_collection()
        return list(collection.find({
            'metric_type': metric_type,
            'timestamp': {
                '$gte': start_time,
                '$lte': end_time
            }
        }, sort=[('timestamp', 1)])) 