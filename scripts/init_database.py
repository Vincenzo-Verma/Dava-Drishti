#!/usr/bin/env python3
"""
Database Initialization Script for Forest Fire Prediction System
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import Config
from app.utils.database import init_database, create_tables, db_manager
from app import create_app

def main():
    """Initialize database and create tables"""
    print("Initializing Forest Fire Prediction System Database...")
    
    # Create Flask app
    app = create_app()
    
    with app.app_context():
        try:
            # Initialize database connections
            init_database(app)
            print("✓ Database connections established")
            
            # Create tables
            create_tables()
            print("✓ Database tables created")
            
            # Create initial data directories
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
            os.makedirs(Config.MODELS_DIR, exist_ok=True)
            os.makedirs(Config.LOGS_DIR, exist_ok=True)
            print("✓ Data directories created")
            
            print("\nDatabase initialization completed successfully!")
            print(f"Database URL: {Config.DATABASE_URL}")
            print(f"MongoDB URL: {Config.MONGODB_URL}")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 