"""
Fire Detection Data Collector for Forest Fire Prediction System
Collects VIIRS active fire data from NASA FIRMS
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import csv
import io

from app.utils.config import Config
from app.utils.logging import data_logger
from app.utils.database import FireEvent, db_manager

logger = logging.getLogger(__name__)

class FireDataCollector:
    """Fire detection data collection from VIIRS"""
    
    def __init__(self):
        self.data_dir = os.path.join(Config.RAW_DATA_DIR, 'fire')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # VIIRS API configuration
        self.api_url = Config.DATA_SOURCES['viirs']['api_url']
        self.satellite = Config.DATA_SOURCES['viirs']['satellite']
    
    def collect_viirs_data(self, date: str, region_bounds: List[float]) -> Dict[str, Any]:
        """Collect VIIRS active fire data"""
        try:
            start_time = datetime.now()
            
            # VIIRS API request parameters
            params = {
                'satellite': self.satellite,
                'date': date,
                'area': f"{region_bounds[0]},{region_bounds[1]},{region_bounds[2]},{region_bounds[3]}",
                'type': 'csv'
            }
            
            # Make API request
            response = requests.get(self.api_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse CSV data
            csv_data = response.text
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Save raw data
            output_file = os.path.join(
                self.data_dir,
                f'viirs_{date}_{region_bounds[0]}_{region_bounds[1]}_{region_bounds[2]}_{region_bounds[3]}.csv'
            )
            
            df.to_csv(output_file, index=False)
            
            # Process and store data
            processed_data = self._process_viirs_data(df, date)
            
            # Log collection
            collection_time = (datetime.now() - start_time).total_seconds()
            data_logger.log_data_collection(
                'viirs',
                'SUCCESS',
                len(processed_data),
                None
            )
            
            logger.info(f"VIIRS data collected successfully for {date}")
            return {
                'source': 'viirs',
                'date': date,
                'file_path': output_file,
                'records_count': len(processed_data),
                'collection_time': collection_time
            }
            
        except Exception as e:
            logger.error(f"Failed to collect VIIRS data: {e}")
            data_logger.log_data_collection('viirs', 'FAILED', None, str(e))
            raise
    
    def _process_viirs_data(self, df: pd.DataFrame, date: str) -> List[Dict[str, Any]]:
        """Process VIIRS data and store in database"""
        try:
            processed_records = []
            
            for _, row in df.iterrows():
                # Parse acquisition date and time
                acq_date = datetime.strptime(row['acq_date'], '%Y-%m-%d')
                acq_time = row.get('acq_time', '00:00')
                
                fire_record = {
                    'event_date': acq_date,
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'confidence': float(row.get('confidence', 0)),
                    'satellite': row.get('satellite', 'VIIRS'),
                    'brightness': float(row.get('brightness', 0)),
                    'scan': float(row.get('scan', 0)),
                    'track': float(row.get('track', 0)),
                    'acq_date': acq_date,
                    'acq_time': acq_time,
                    'satellite_time': row.get('satellite_time', ''),
                    'instrument': row.get('instrument', 'VIIRS'),
                    'confidence_level': row.get('confidence_level', ''),
                    'version': row.get('version', ''),
                    'bright_t31': float(row.get('bright_t31', 0)),
                    'frp': float(row.get('frp', 0)),
                    'daynight': row.get('daynight', ''),
                    'type': row.get('type', '')
                }
                
                # Store in database
                self._store_fire_record(fire_record)
                processed_records.append(fire_record)
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Failed to process VIIRS data: {e}")
            raise
    
    def _store_fire_record(self, record: Dict[str, Any]):
        """Store fire record in database"""
        try:
            session = db_manager.get_postgres_session()
            
            # Check if record already exists
            existing = session.query(FireEvent).filter_by(
                event_date=record['event_date'],
                latitude=record['latitude'],
                longitude=record['longitude'],
                satellite=record['satellite']
            ).first()
            
            if not existing:
                fire_event = FireEvent(**record)
                session.add(fire_event)
                session.commit()
            
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to store fire record: {e}")
            session.rollback()
            session.close()
            raise
    
    def get_fire_data(self, start_date: str, end_date: str, region_bounds: List[float]) -> pd.DataFrame:
        """Get fire data for a date range and region"""
        try:
            session = db_manager.get_postgres_session()
            
            # Query fire data
            query = session.query(FireEvent).filter(
                FireEvent.event_date >= datetime.strptime(start_date, '%Y-%m-%d'),
                FireEvent.event_date <= datetime.strptime(end_date, '%Y-%m-%d'),
                FireEvent.latitude >= region_bounds[0],
                FireEvent.latitude <= region_bounds[1],
                FireEvent.longitude >= region_bounds[2],
                FireEvent.longitude <= region_bounds[3]
            )
            
            # Convert to DataFrame
            df = pd.read_sql(query.statement, session.bind)
            session.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get fire data: {e}")
            raise
    
    def collect_daily_data(self, date: str = None, regions: List[List[float]] = None):
        """Collect daily fire data for all regions"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if regions is None:
            regions = [
                [12.0, 13.0, 77.0, 78.0],  # Bangalore
                [18.0, 19.0, 72.0, 73.0],  # Mumbai
                [28.0, 29.0, 76.0, 77.0],  # Delhi
                [12.0, 13.0, 79.0, 80.0]   # Chennai
            ]
        
        results = []
        
        for region in regions:
            try:
                result = self.collect_viirs_data(date, region)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to collect fire data for region {region}: {e}")
        
        return results 