"""
Web Interface Routes for Forest Fire Prediction System
"""

import os
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils.config import Config

logger = logging.getLogger(__name__)

# Create web blueprint
web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', 
                         title='Forest Fire Prediction Dashboard',
                         current_time=datetime.now())

@web_bp.route('/dashboard')
def dashboard():
    """Interactive dashboard with maps and charts"""
    return render_template('dashboard.html',
                         title='Fire Prediction Dashboard',
                         mapbox_token=os.environ.get('MAPBOX_TOKEN', ''))

@web_bp.route('/prediction')
def prediction_page():
    """Fire risk prediction interface"""
    return render_template('prediction.html',
                         title='Fire Risk Prediction',
                         regions=get_available_regions())

@web_bp.route('/simulation')
def simulation_page():
    """Fire spread simulation interface"""
    return render_template('simulation.html',
                         title='Fire Spread Simulation')

@web_bp.route('/analysis')
def analysis_page():
    """Data analysis and visualization"""
    return render_template('analysis.html',
                         title='Data Analysis')

@web_bp.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html',
                         title='API Documentation')

def get_available_regions():
    """Get list of available regions"""
    return [
        {'id': 'bangalore', 'name': 'Bangalore Region', 'bounds': [12.0, 13.0, 77.0, 78.0]},
        {'id': 'mumbai', 'name': 'Mumbai Region', 'bounds': [18.0, 19.0, 72.0, 73.0]},
        {'id': 'delhi', 'name': 'Delhi Region', 'bounds': [28.0, 29.0, 76.0, 77.0]},
        {'id': 'chennai', 'name': 'Chennai Region', 'bounds': [12.0, 13.0, 79.0, 80.0]}
    ] 