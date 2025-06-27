"""
REST API Routes for Forest Fire Prediction System
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_caching import Cache
import numpy as np

from app.models.unet_model import UNetFirePredictionModel
from app.models.lstm_model import LSTMFireTemporalModel
from app.models.cellular_automata import CellularAutomataFireSpread
from app.utils.logging import performance_logger, security_logger
from app.utils.database import PredictionResult, SimulationResult, db_manager
from app.utils.config import Config

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__)

# Initialize cache
cache = Cache()

def init_cache(app):
    """Initialize cache with app"""
    cache.init_app(app)

# Initialize models
unet_model = None
lstm_model = None
ca_model = None

def init_models():
    """Initialize ML models"""
    global unet_model, lstm_model, ca_model
    try:
        unet_model = UNetFirePredictionModel()
        lstm_model = LSTMFireTemporalModel()
        ca_model = CellularAutomataFireSpread()
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'models_loaded': {
            'unet': unet_model is not None,
            'lstm': lstm_model is not None,
            'cellular_automata': ca_model is not None
        }
    })

@api_bp.route('/fire-risk/<region>', methods=['GET'])
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_fire_risk(region):
    """Get fire risk prediction for a specific region"""
    try:
        start_time = datetime.now()
        
        # Get parameters
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        bounds = request.args.get('bounds')  # Format: "min_lat,max_lat,min_lon,max_lon"
        
        if bounds:
            bounds = [float(x) for x in bounds.split(',')]
        else:
            # Default bounds for the region
            bounds = [12.0, 13.0, 77.0, 78.0]  # Example: Bangalore region
        
        # Generate fire risk prediction
        if unet_model is None:
            init_models()
        
        risk_result = unet_model.predict_fire_risk_map(
            region_bounds=tuple(bounds),
            date=date,
            output_path=f"data/processed/risk_map_{region}_{date}.tif"
        )
        
        # Save prediction result to database
        prediction_record = PredictionResult(
            prediction_date=datetime.strptime(date, '%Y-%m-%d'),
            region=region,
            model_type='unet',
            risk_level=risk_result['statistics']['risk_level'],
            confidence=0.92,  # Placeholder
            map_file_path=risk_result.get('output_path'),
            metadata=risk_result['statistics']
        )
        
        session = db_manager.get_postgres_session()
        session.add(prediction_record)
        session.commit()
        session.close()
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log performance
        performance_logger.log_api_request(
            f'/fire-risk/{region}',
            'GET',
            response_time,
            200
        )
        
        return jsonify({
            'region': region,
            'date': date,
            'risk_level': risk_result['statistics']['risk_level'],
            'confidence': 0.92,
            'statistics': risk_result['statistics'],
            'risk_map_url': f'/api/v1/maps/risk_map_{region}_{date}.tif',
            'prediction_time': response_time
        })
        
    except Exception as e:
        logger.error(f"Error in fire risk prediction: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/fire-simulation', methods=['POST'])
def run_fire_simulation():
    """Run fire spread simulation"""
    try:
        start_time = datetime.now()
        
        # Get request data
        data = request.get_json()
        ignition_point = data.get('ignition_point', {'lat': 12.9716, 'lon': 77.5946})
        duration_hours = data.get('duration_hours', 6)
        wind_conditions = data.get('wind_conditions', {'speed': 0.0, 'direction': 0.0})
        
        # Convert lat/lon to grid coordinates
        grid_x = int((ignition_point['lon'] - 77.0) * 1000)  # Approximate conversion
        grid_y = int((ignition_point['lat'] - 12.0) * 1000)
        
        # Run simulation
        if ca_model is None:
            init_models()
        
        simulation_result = ca_model.simulate_fire_spread(
            ignition_point=(grid_y, grid_x),
            wind_speed=wind_conditions['speed'],
            wind_dir=wind_conditions['direction'],
            duration_hours=duration_hours
        )
        
        # Generate animation
        animation_path = f"data/processed/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        ca_model.animate_fire_spread(simulation_result, animation_path)
        
        # Save simulation result to database
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        simulation_record = SimulationResult(
            simulation_id=simulation_id,
            ignition_point_lat=ignition_point['lat'],
            ignition_point_lon=ignition_point['lon'],
            duration_hours=duration_hours,
            affected_area_km2=simulation_result['burned_area_km2'],
            animation_file_path=animation_path,
            parameters=json.dumps(data),
            status='completed',
            completed_at=datetime.now()
        )
        
        session = db_manager.get_postgres_session()
        session.add(simulation_record)
        session.commit()
        session.close()
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log performance
        performance_logger.log_api_request(
            '/fire-simulation',
            'POST',
            response_time,
            200
        )
        
        return jsonify({
            'simulation_id': simulation_id,
            'affected_area_km2': simulation_result['burned_area_km2'],
            'animation_url': f'/api/v1/animations/{simulation_id}.gif',
            'completion_time': datetime.now().isoformat(),
            'simulation_time': response_time
        })
        
    except Exception as e:
        logger.error(f"Error in fire simulation: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/maps/<filename>', methods=['GET'])
def serve_map(filename):
    """Serve map files (GeoTIFF)"""
    try:
        file_path = os.path.join(Config.PROCESSED_DATA_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/tiff')
        else:
            return jsonify({'error': 'Map file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving map file: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/animations/<simulation_id>', methods=['GET'])
def serve_animation(simulation_id):
    """Serve animation files (GIF)"""
    try:
        # Get simulation record
        session = db_manager.get_postgres_session()
        simulation = session.query(SimulationResult).filter_by(simulation_id=simulation_id).first()
        session.close()
        
        if simulation and simulation.animation_file_path and os.path.exists(simulation.animation_file_path):
            return send_file(simulation.animation_file_path, mimetype='image/gif')
        else:
            return jsonify({'error': 'Animation file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving animation file: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """Get historical predictions"""
    try:
        region = request.args.get('region')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 100))
        
        session = db_manager.get_postgres_session()
        query = session.query(PredictionResult)
        
        if region:
            query = query.filter_by(region=region)
        if start_date:
            query = query.filter(PredictionResult.prediction_date >= start_date)
        if end_date:
            query = query.filter(PredictionResult.prediction_date <= end_date)
        
        predictions = query.order_by(PredictionResult.prediction_date.desc()).limit(limit).all()
        session.close()
        
        return jsonify({
            'predictions': [
                {
                    'id': p.id,
                    'region': p.region,
                    'prediction_date': p.prediction_date.isoformat(),
                    'risk_level': p.risk_level,
                    'confidence': p.confidence,
                    'model_type': p.model_type
                }
                for p in predictions
            ],
            'count': len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/simulations', methods=['GET'])
def get_simulations():
    """Get historical simulations"""
    try:
        status = request.args.get('status')
        limit = int(request.args.get('limit', 100))
        
        session = db_manager.get_postgres_session()
        query = session.query(SimulationResult)
        
        if status:
            query = query.filter_by(status=status)
        
        simulations = query.order_by(SimulationResult.created_at.desc()).limit(limit).all()
        session.close()
        
        return jsonify({
            'simulations': [
                {
                    'simulation_id': s.simulation_id,
                    'ignition_point': {'lat': s.ignition_point_lat, 'lon': s.ignition_point_lon},
                    'duration_hours': s.duration_hours,
                    'affected_area_km2': s.affected_area_km2,
                    'status': s.status,
                    'created_at': s.created_at.isoformat(),
                    'completed_at': s.completed_at.isoformat() if s.completed_at else None
                }
                for s in simulations
            ],
            'count': len(simulations)
        })
        
    except Exception as e:
        logger.error(f"Error getting simulations: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/status', methods=['GET'])
def get_model_status():
    """Get model status and performance metrics"""
    try:
        # Get latest model info from MongoDB
        unet_info = ModelMetadata.get_model_info('unet_fire_prediction')
        lstm_info = ModelMetadata.get_model_info('lstm_fire_temporal')
        
        return jsonify({
            'models': {
                'unet': {
                    'loaded': unet_model is not None,
                    'last_trained': unet_info.get('created_at').isoformat() if unet_info else None,
                    'version': unet_info.get('version') if unet_info else None
                },
                'lstm': {
                    'loaded': lstm_model is not None,
                    'last_trained': lstm_info.get('created_at').isoformat() if lstm_info else None,
                    'version': lstm_info.get('version') if lstm_info else None
                },
                'cellular_automata': {
                    'loaded': ca_model is not None,
                    'config': ca_model.config if ca_model else None
                }
            },
            'system_status': 'operational'
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/data/sources', methods=['GET'])
def get_data_sources():
    """Get available data sources and their status"""
    try:
        return jsonify({
            'data_sources': Config.DATA_SOURCES,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'details': str(error)}), 400

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500 