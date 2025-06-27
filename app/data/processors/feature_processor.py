"""
Feature Processing for Forest Fire Prediction System
Data preprocessing, feature engineering, and validation
"""

import os
import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Point

from app.utils.config import Config
from app.utils.logging import data_logger
from app.utils.database import WeatherData, FireEvent, TerrainData, db_manager

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Feature processing and engineering for fire prediction"""
    
    def __init__(self):
        self.processed_dir = Config.PROCESSED_DATA_DIR
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize scalers
        self.weather_scaler = StandardScaler()
        self.terrain_scaler = MinMaxScaler()
        
        # Feature configuration
        self.feature_config = {
            'weather_features': ['temperature_2m', 'relative_humidity_2m', 'wind_u_10m', 'wind_v_10m', 'total_precipitation'],
            'terrain_features': ['elevation', 'slope', 'aspect'],
            'vegetation_features': ['ndvi', 'land_cover', 'fuel_load'],
            'temporal_features': ['day_of_year', 'month', 'season']
        }
    
    def create_feature_stack(self, region_bounds: List[float], date: str) -> np.ndarray:
        """Create multi-band feature stack for prediction"""
        try:
            start_time = datetime.now()
            
            # Get data dimensions
            height, width = self._calculate_grid_dimensions(region_bounds)
            
            # Initialize feature stack
            n_features = len(self.feature_config['weather_features']) + \
                        len(self.feature_config['terrain_features']) + \
                        len(self.feature_config['vegetation_features']) + \
                        len(self.feature_config['temporal_features'])
            
            feature_stack = np.zeros((height, width, n_features))
            
            # Add weather features
            weather_features = self._extract_weather_features(region_bounds, date)
            if weather_features is not None:
                feature_stack[:, :, :len(self.feature_config['weather_features'])] = weather_features
            
            # Add terrain features
            terrain_features = self._extract_terrain_features(region_bounds)
            if terrain_features is not None:
                start_idx = len(self.feature_config['weather_features'])
                end_idx = start_idx + len(self.feature_config['terrain_features'])
                feature_stack[:, :, start_idx:end_idx] = terrain_features
            
            # Add vegetation features
            vegetation_features = self._extract_vegetation_features(region_bounds, date)
            if vegetation_features is not None:
                start_idx = len(self.feature_config['weather_features']) + len(self.feature_config['terrain_features'])
                end_idx = start_idx + len(self.feature_config['vegetation_features'])
                feature_stack[:, :, start_idx:end_idx] = vegetation_features
            
            # Add temporal features
            temporal_features = self._extract_temporal_features(date, height, width)
            start_idx = len(self.feature_config['weather_features']) + len(self.feature_config['terrain_features']) + len(self.feature_config['vegetation_features'])
            feature_stack[:, :, start_idx:] = temporal_features
            
            # Normalize features
            feature_stack = self._normalize_features(feature_stack)
            
            # Save feature stack
            output_path = os.path.join(
                self.processed_dir,
                f'feature_stack_{date}_{region_bounds[0]}_{region_bounds[1]}_{region_bounds[2]}_{region_bounds[3]}.npy'
            )
            np.save(output_path, feature_stack)
            
            # Log processing
            processing_time = (datetime.now() - start_time).total_seconds()
            data_logger.log_data_processing(
                'feature_stack_creation',
                processing_time,
                f"{height}x{width}x{n_features}"
            )
            
            logger.info(f"Feature stack created successfully: {feature_stack.shape}")
            return feature_stack
            
        except Exception as e:
            logger.error(f"Failed to create feature stack: {e}")
            raise
    
    def _calculate_grid_dimensions(self, region_bounds: List[float]) -> Tuple[int, int]:
        """Calculate grid dimensions based on region bounds and resolution"""
        lat_range = region_bounds[1] - region_bounds[0]
        lon_range = region_bounds[3] - region_bounds[2]
        
        # Assuming 30m resolution (approximately 0.0003 degrees)
        resolution = 0.0003
        height = int(lat_range / resolution)
        width = int(lon_range / resolution)
        
        return height, width
    
    def _extract_weather_features(self, region_bounds: List[float], date: str) -> Optional[np.ndarray]:
        """Extract weather features for the region and date"""
        try:
            # Get weather data from database
            session = db_manager.get_postgres_session()
            query = session.query(WeatherData).filter(
                WeatherData.date == datetime.strptime(date, '%Y-%m-%d'),
                WeatherData.latitude >= region_bounds[0],
                WeatherData.latitude <= region_bounds[1],
                WeatherData.longitude >= region_bounds[2],
                WeatherData.longitude <= region_bounds[3]
            )
            
            weather_df = pd.read_sql(query.statement, session.bind)
            session.close()
            
            if weather_df.empty:
                logger.warning(f"No weather data found for {date}")
                return None
            
            # Interpolate to regular grid
            height, width = self._calculate_grid_dimensions(region_bounds)
            weather_features = np.zeros((height, width, len(self.feature_config['weather_features'])))
            
            # Create regular grid
            lat_grid = np.linspace(region_bounds[0], region_bounds[1], height)
            lon_grid = np.linspace(region_bounds[2], region_bounds[3], width)
            
            for i, feature in enumerate(self.feature_config['weather_features']):
                if feature in weather_df.columns:
                    # Interpolate using nearest neighbor
                    feature_values = weather_df[feature].values
                    lat_coords = weather_df['latitude'].values
                    lon_coords = weather_df['longitude'].values
                    
                    # Simple nearest neighbor interpolation
                    for h in range(height):
                        for w in range(width):
                            distances = np.sqrt((lat_coords - lat_grid[h])**2 + (lon_coords - lon_grid[w])**2)
                            nearest_idx = np.argmin(distances)
                            weather_features[h, w, i] = feature_values[nearest_idx]
            
            return weather_features
            
        except Exception as e:
            logger.error(f"Failed to extract weather features: {e}")
            return None
    
    def _extract_terrain_features(self, region_bounds: List[float]) -> Optional[np.ndarray]:
        """Extract terrain features for the region"""
        try:
            # Get terrain data from database
            session = db_manager.get_postgres_session()
            query = session.query(TerrainData).filter(
                TerrainData.latitude >= region_bounds[0],
                TerrainData.latitude <= region_bounds[1],
                TerrainData.longitude >= region_bounds[2],
                TerrainData.longitude <= region_bounds[3]
            )
            
            terrain_df = pd.read_sql(query.statement, session.bind)
            session.close()
            
            if terrain_df.empty:
                logger.warning("No terrain data found")
                return None
            
            # Interpolate to regular grid
            height, width = self._calculate_grid_dimensions(region_bounds)
            terrain_features = np.zeros((height, width, len(self.feature_config['terrain_features'])))
            
            # Create regular grid
            lat_grid = np.linspace(region_bounds[0], region_bounds[1], height)
            lon_grid = np.linspace(region_bounds[2], region_bounds[3], width)
            
            for i, feature in enumerate(self.feature_config['terrain_features']):
                if feature in terrain_df.columns:
                    feature_values = terrain_df[feature].values
                    lat_coords = terrain_df['latitude'].values
                    lon_coords = terrain_df['longitude'].values
                    
                    # Simple nearest neighbor interpolation
                    for h in range(height):
                        for w in range(width):
                            distances = np.sqrt((lat_coords - lat_grid[h])**2 + (lon_coords - lon_grid[w])**2)
                            nearest_idx = np.argmin(distances)
                            terrain_features[h, w, i] = feature_values[nearest_idx]
            
            return terrain_features
            
        except Exception as e:
            logger.error(f"Failed to extract terrain features: {e}")
            return None
    
    def _extract_vegetation_features(self, region_bounds: List[float], date: str) -> Optional[np.ndarray]:
        """Extract vegetation features for the region"""
        try:
            # For now, generate dummy vegetation data
            # In production, this would load from actual vegetation datasets
            height, width = self._calculate_grid_dimensions(region_bounds)
            vegetation_features = np.zeros((height, width, len(self.feature_config['vegetation_features'])))
            
            # Generate dummy NDVI (0.1 to 0.8)
            vegetation_features[:, :, 0] = np.random.uniform(0.1, 0.8, (height, width))
            
            # Generate dummy land cover (1-5 categories)
            vegetation_features[:, :, 1] = np.random.randint(1, 6, (height, width))
            
            # Generate dummy fuel load (0-10 tons/ha)
            vegetation_features[:, :, 2] = np.random.uniform(0, 10, (height, width))
            
            return vegetation_features
            
        except Exception as e:
            logger.error(f"Failed to extract vegetation features: {e}")
            return None
    
    def _extract_temporal_features(self, date: str, height: int, width: int) -> np.ndarray:
        """Extract temporal features"""
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            
            temporal_features = np.zeros((height, width, len(self.feature_config['temporal_features'])))
            
            # Day of year (1-365)
            temporal_features[:, :, 0] = date_obj.timetuple().tm_yday
            
            # Month (1-12)
            temporal_features[:, :, 1] = date_obj.month
            
            # Season (1-4: Winter, Spring, Summer, Fall)
            month = date_obj.month
            if month in [12, 1, 2]:
                season = 1  # Winter
            elif month in [3, 4, 5]:
                season = 2  # Spring
            elif month in [6, 7, 8]:
                season = 3  # Summer
            else:
                season = 4  # Fall
            
            temporal_features[:, :, 2] = season
            
            return temporal_features
            
        except Exception as e:
            logger.error(f"Failed to extract temporal features: {e}")
            return np.zeros((height, width, len(self.feature_config['temporal_features'])))
    
    def _normalize_features(self, feature_stack: np.ndarray) -> np.ndarray:
        """Normalize feature stack"""
        try:
            # Reshape for normalization
            original_shape = feature_stack.shape
            feature_stack_flat = feature_stack.reshape(-1, original_shape[-1])
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            feature_stack_flat = imputer.fit_transform(feature_stack_flat)
            
            # Normalize features
            scaler = StandardScaler()
            feature_stack_normalized = scaler.fit_transform(feature_stack_flat)
            
            # Reshape back
            feature_stack = feature_stack_normalized.reshape(original_shape)
            
            return feature_stack
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            return feature_stack
    
    def validate_data_quality(self, feature_stack: np.ndarray) -> Dict[str, Any]:
        """Validate data quality of feature stack"""
        try:
            validation_results = {
                'total_pixels': feature_stack.shape[0] * feature_stack.shape[1],
                'n_features': feature_stack.shape[2],
                'missing_values': np.isnan(feature_stack).sum(),
                'missing_percentage': (np.isnan(feature_stack).sum() / feature_stack.size) * 100,
                'feature_ranges': {},
                'feature_means': {},
                'feature_stds': {}
            }
            
            # Calculate statistics for each feature
            for i in range(feature_stack.shape[2]):
                feature_data = feature_stack[:, :, i]
                valid_data = feature_data[~np.isnan(feature_data)]
                
                if len(valid_data) > 0:
                    validation_results['feature_ranges'][f'feature_{i}'] = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data))
                    }
                    validation_results['feature_means'][f'feature_{i}'] = float(np.mean(valid_data))
                    validation_results['feature_stds'][f'feature_{i}'] = float(np.std(valid_data))
            
            # Log validation results
            data_logger.log_data_validation(
                'feature_stack',
                validation_results['missing_percentage'] < 5,  # Pass if < 5% missing
                int(validation_results['missing_values'])
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data quality: {e}")
            return {'error': str(e)}
    
    def create_training_dataset(self, start_date: str, end_date: str, region_bounds: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset with features and labels"""
        try:
            # Generate date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            date_range = pd.date_range(start_dt, end_dt, freq='D')
            
            features_list = []
            labels_list = []
            
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                
                # Create feature stack
                feature_stack = self.create_feature_stack(region_bounds, date_str)
                
                # Create labels (fire events)
                labels = self._create_fire_labels(region_bounds, date_str, feature_stack.shape[:2])
                
                # Add to lists
                features_list.append(feature_stack)
                labels_list.append(labels)
            
            # Stack all features and labels
            X = np.stack(features_list, axis=0)
            y = np.stack(labels_list, axis=0)
            
            logger.info(f"Training dataset created: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            raise
    
    def _create_fire_labels(self, region_bounds: List[float], date: str, shape: Tuple[int, int]) -> np.ndarray:
        """Create fire event labels for the region and date"""
        try:
            # Get fire events from database
            session = db_manager.get_postgres_session()
            query = session.query(FireEvent).filter(
                FireEvent.event_date == datetime.strptime(date, '%Y-%m-%d'),
                FireEvent.latitude >= region_bounds[0],
                FireEvent.latitude <= region_bounds[1],
                FireEvent.longitude >= region_bounds[2],
                FireEvent.longitude <= region_bounds[3]
            )
            
            fire_df = pd.read_sql(query.statement, session.bind)
            session.close()
            
            # Create binary labels
            labels = np.zeros(shape, dtype=np.uint8)
            
            if not fire_df.empty:
                # Convert fire coordinates to grid indices
                height, width = shape
                lat_grid = np.linspace(region_bounds[0], region_bounds[1], height)
                lon_grid = np.linspace(region_bounds[2], region_bounds[3], width)
                
                for _, fire in fire_df.iterrows():
                    # Find nearest grid cell
                    lat_idx = np.argmin(np.abs(lat_grid - fire['latitude']))
                    lon_idx = np.argmin(np.abs(lon_grid - fire['longitude']))
                    
                    # Mark as fire event
                    labels[lat_idx, lon_idx] = 1
                    
                    # Add some spatial spread (optional)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = lat_idx + di, lon_idx + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                labels[ni, nj] = 1
            
            return labels
            
        except Exception as e:
            logger.error(f"Failed to create fire labels: {e}")
            return np.zeros(shape, dtype=np.uint8) 