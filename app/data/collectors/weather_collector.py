"""
Weather Data Collector for Forest Fire Prediction System
Collects weather data from ERA5 and MOSDAC sources
"""

import os
import logging
import cdsapi
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
import json

from app.utils.config import Config
from app.utils.logging import data_logger
from app.utils.database import WeatherData, db_manager

logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Weather data collection from multiple sources"""
    
    def __init__(self):
        self.cds_client = None
        self.data_dir = os.path.join(Config.RAW_DATA_DIR, 'weather')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize CDS API client
        try:
            self.cds_client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
    
    def collect_era5_data(self, date: str, region_bounds: List[float]) -> Dict[str, Any]:
        """Collect ERA5 reanalysis data"""
        try:
            start_time = datetime.now()
            
            # Parse date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # ERA5 request parameters
            request_params = {
                'product_type': 'reanalysis',
                'variable': Config.DATA_SOURCES['era5']['variables'],
                'year': str(target_date.year),
                'month': str(target_date.month).zfill(2),
                'day': str(target_date.day).zfill(2),
                'time': [f'{hour:02d}:00' for hour in range(24)],
                'area': [
                    region_bounds[1],  # North
                    region_bounds[0],  # West
                    region_bounds[3],  # South
                    region_bounds[2]   # East
                ],
                'format': 'netcdf'
            }
            
            # Output file path
            output_file = os.path.join(
                self.data_dir,
                f'era5_{date}_{region_bounds[0]}_{region_bounds[1]}_{region_bounds[2]}_{region_bounds[3]}.nc'
            )
            
            # Download data
            if self.cds_client:
                self.cds_client.retrieve('reanalysis-era5-single-levels', request_params, output_file)
                
                # Process and store data
                processed_data = self._process_era5_data(output_file, date)
                
                # Log collection
                collection_time = (datetime.now() - start_time).total_seconds()
                data_logger.log_data_collection(
                    'era5',
                    'SUCCESS',
                    len(processed_data),
                    None
                )
                
                logger.info(f"ERA5 data collected successfully for {date}")
                return {
                    'source': 'era5',
                    'date': date,
                    'file_path': output_file,
                    'records_count': len(processed_data),
                    'collection_time': collection_time
                }
            else:
                raise Exception("CDS API client not available")
                
        except Exception as e:
            logger.error(f"Failed to collect ERA5 data: {e}")
            data_logger.log_data_collection('era5', 'FAILED', None, str(e))
            raise
    
    def collect_mosdac_data(self, date: str, region_bounds: List[float]) -> Dict[str, Any]:
        """Collect MOSDAC weather data"""
        try:
            start_time = datetime.now()
            
            # MOSDAC API endpoint (example)
            api_url = "https://mosdac.gov.in/api/weather"
            
            # Request parameters
            params = {
                'date': date,
                'lat_min': region_bounds[0],
                'lat_max': region_bounds[1],
                'lon_min': region_bounds[2],
                'lon_max': region_bounds[3]
            }
            
            # Make API request
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Save raw data
            output_file = os.path.join(
                self.data_dir,
                f'mosdac_{date}_{region_bounds[0]}_{region_bounds[1]}_{region_bounds[2]}_{region_bounds[3]}.json'
            )
            
            with open(output_file, 'w') as f:
                json.dump(data, f)
            
            # Process and store data
            processed_data = self._process_mosdac_data(data, date)
            
            # Log collection
            collection_time = (datetime.now() - start_time).total_seconds()
            data_logger.log_data_collection(
                'mosdac',
                'SUCCESS',
                len(processed_data),
                None
            )
            
            logger.info(f"MOSDAC data collected successfully for {date}")
            return {
                'source': 'mosdac',
                'date': date,
                'file_path': output_file,
                'records_count': len(processed_data),
                'collection_time': collection_time
            }
            
        except Exception as e:
            logger.error(f"Failed to collect MOSDAC data: {e}")
            data_logger.log_data_collection('mosdac', 'FAILED', None, str(e))
            raise
    
    def _process_era5_data(self, file_path: str, date: str) -> List[Dict[str, Any]]:
        """Process ERA5 NetCDF data and store in database"""
        try:
            # Load NetCDF data
            ds = xr.open_dataset(file_path)
            
            processed_records = []
            
            # Extract data for each time step and location
            for time_idx in range(len(ds.time)):
                time_data = ds.isel(time=time_idx)
                
                # Get coordinates
                lats = time_data.latitude.values
                lons = time_data.longitude.values
                
                for lat_idx, lat in enumerate(lats):
                    for lon_idx, lon in enumerate(lons):
                        # Extract weather variables
                        weather_record = {
                            'date': datetime.strptime(date, '%Y-%m-%d'),
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'temperature_2m': float(time_data['2m_temperature'].values[lat_idx, lon_idx]),
                            'relative_humidity_2m': float(time_data['2m_relative_humidity'].values[lat_idx, lon_idx]),
                            'wind_u_10m': float(time_data['10m_u_component_of_wind'].values[lat_idx, lon_idx]),
                            'wind_v_10m': float(time_data['10m_v_component_of_wind'].values[lat_idx, lon_idx]),
                            'total_precipitation': float(time_data['total_precipitation'].values[lat_idx, lon_idx]),
                            'source': 'era5'
                        }
                        
                        # Store in database
                        self._store_weather_record(weather_record)
                        processed_records.append(weather_record)
            
            ds.close()
            return processed_records
            
        except Exception as e:
            logger.error(f"Failed to process ERA5 data: {e}")
            raise
    
    def _process_mosdac_data(self, data: Dict[str, Any], date: str) -> List[Dict[str, Any]]:
        """Process MOSDAC JSON data and store in database"""
        try:
            processed_records = []
            
            # Process each weather station data
            for station_data in data.get('stations', []):
                weather_record = {
                    'date': datetime.strptime(date, '%Y-%m-%d'),
                    'latitude': float(station_data.get('lat', 0)),
                    'longitude': float(station_data.get('lon', 0)),
                    'temperature_2m': float(station_data.get('temperature', 0)),
                    'relative_humidity_2m': float(station_data.get('humidity', 0)),
                    'wind_u_10m': float(station_data.get('wind_u', 0)),
                    'wind_v_10m': float(station_data.get('wind_v', 0)),
                    'total_precipitation': float(station_data.get('precipitation', 0)),
                    'source': 'mosdac'
                }
                
                # Store in database
                self._store_weather_record(weather_record)
                processed_records.append(weather_record)
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Failed to process MOSDAC data: {e}")
            raise
    
    def _store_weather_record(self, record: Dict[str, Any]):
        """Store weather record in database"""
        try:
            session = db_manager.get_postgres_session()
            
            # Check if record already exists
            existing = session.query(WeatherData).filter_by(
                date=record['date'],
                latitude=record['latitude'],
                longitude=record['longitude'],
                source=record['source']
            ).first()
            
            if not existing:
                weather_data = WeatherData(**record)
                session.add(weather_data)
                session.commit()
            
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to store weather record: {e}")
            session.rollback()
            session.close()
            raise
    
    def get_weather_data(self, date: str, region_bounds: List[float]) -> pd.DataFrame:
        """Get weather data for a specific date and region"""
        try:
            session = db_manager.get_postgres_session()
            
            # Query weather data
            query = session.query(WeatherData).filter(
                WeatherData.date == datetime.strptime(date, '%Y-%m-%d'),
                WeatherData.latitude >= region_bounds[0],
                WeatherData.latitude <= region_bounds[1],
                WeatherData.longitude >= region_bounds[2],
                WeatherData.longitude <= region_bounds[3]
            )
            
            # Convert to DataFrame
            df = pd.read_sql(query.statement, session.bind)
            session.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get weather data: {e}")
            raise
    
    def collect_daily_data(self, date: str = None, regions: List[List[float]] = None):
        """Collect daily weather data for all regions"""
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
                # Collect ERA5 data
                era5_result = self.collect_era5_data(date, region)
                results.append(era5_result)
                
                # Collect MOSDAC data
                mosdac_result = self.collect_mosdac_data(date, region)
                results.append(mosdac_result)
                
            except Exception as e:
                logger.error(f"Failed to collect data for region {region}: {e}")
        
        return results 