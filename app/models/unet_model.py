"""
U-Net Deep Learning Model for Forest Fire Prediction
Spatial binary classification model for fire/no-fire prediction
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.keras
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box

from app.utils.config import Config
from app.utils.logging import data_logger, performance_logger
from app.utils.database import ModelMetadata, TrainingHistory

logger = logging.getLogger(__name__)

class UNetFirePredictionModel:
    """
    U-Net model for forest fire prediction
    Input: Multi-band satellite imagery and environmental data
    Output: Binary fire probability map
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize U-Net model"""
        self.config = config or Config.MODEL_CONFIG['unet']
        self.model = None
        self.history = None
        self.model_path = os.path.join(Config.MODELS_DIR, 'unet_fire_prediction.h5')
        
        # Ensure models directory exists
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        logger.info(f"Initialized U-Net model with config: {self.config}")
    
    def build_model(self) -> Model:
        """Build U-Net architecture"""
        
        def conv_block(inputs, filters, kernel_size=3, dropout_rate=0.2):
            """Convolutional block with batch normalization and dropout"""
            x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            return x
        
        def encoder_block(inputs, filters, dropout_rate=0.2):
            """Encoder block with max pooling"""
            x = conv_block(inputs, filters, dropout_rate=dropout_rate)
            skip = x
            x = layers.MaxPooling2D((2, 2))(x)
            return x, skip
        
        def decoder_block(inputs, skip_features, filters, dropout_rate=0.2):
            """Decoder block with upsampling and skip connections"""
            x = layers.UpSampling2D((2, 2))(inputs)
            x = layers.Concatenate()([x, skip_features])
            x = conv_block(x, filters, dropout_rate=dropout_rate)
            return x
        
        # Input layer
        inputs = layers.Input(shape=self.config['input_shape'])
        
        # Encoder path
        encoder_outputs = []
        x = inputs
        
        for filters in self.config['filters']:
            x, skip = encoder_block(x, filters, self.config['dropout_rate'])
            encoder_outputs.append(skip)
        
        # Bridge
        x = conv_block(x, self.config['filters'][-1] * 2, dropout_rate=self.config['dropout_rate'])
        
        # Decoder path
        for i, filters in enumerate(reversed(self.config['filters'])):
            x = decoder_block(x, encoder_outputs[-(i+1)], filters, self.config['dropout_rate'])
        
        # Output layer
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=self._dice_loss,
            metrics=[
                'accuracy',
                self._dice_coefficient,
                self._iou_score,
                'precision',
                'recall'
            ]
        )
        
        logger.info("U-Net model built successfully")
        return self.model
    
    def _dice_loss(self, y_true, y_pred):
        """Dice loss for imbalanced binary classification"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def _dice_coefficient(self, y_true, y_pred):
        """Dice coefficient metric"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def _iou_score(self, y_true, y_pred):
        """Intersection over Union (IoU) metric"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def train(self, train_data, val_data, callbacks=None):
        """Train the U-Net model"""
        
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                ModelCheckpoint(
                    self.model_path,
                    monitor='val_dice_coefficient',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_dice_coefficient',
                    patience=15,
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"unet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params(self.config)
            
            # Training
            start_time = datetime.now()
            self.history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log metrics
            best_epoch = np.argmax(self.history.history['val_dice_coefficient'])
            best_metrics = {
                'best_val_dice_coefficient': self.history.history['val_dice_coefficient'][best_epoch],
                'best_val_accuracy': self.history.history['val_accuracy'][best_epoch],
                'best_val_iou_score': self.history.history['val_iou_score'][best_epoch],
                'training_time_seconds': training_time
            }
            
            mlflow.log_metrics(best_metrics)
            mlflow.keras.log_model(self.model, "unet_model")
            
            # Save training history
            training_record = {
                'model_name': 'unet_fire_prediction',
                'training_date': datetime.now(),
                'config': self.config,
                'metrics': best_metrics,
                'history': self.history.history
            }
            TrainingHistory.save_training_record(training_record)
            
            # Log performance
            performance_logger.log_model_prediction(
                'unet_training',
                training_time,
                best_metrics['best_val_accuracy']
            )
            
            logger.info(f"Training completed. Best validation dice coefficient: {best_metrics['best_val_dice_coefficient']:.4f}")
            
            return self.history
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Generate fire probability predictions"""
        
        if self.model is None:
            self.load_model()
        
        start_time = datetime.now()
        
        # Preprocess input data
        processed_data = self._preprocess_input(input_data)
        
        # Generate predictions
        predictions = self.model.predict(processed_data, verbose=0)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Log performance
        performance_logger.log_model_prediction('unet_prediction', prediction_time)
        
        return predictions
    
    def predict_fire_risk_map(self, region_bounds: Tuple[float, float, float, float], 
                            date: str, output_path: str = None) -> Dict[str, Any]:
        """Generate fire risk map for a specific region and date"""
        
        # Load and prepare input data
        input_data = self._load_region_data(region_bounds, date)
        
        # Generate predictions
        predictions = self.predict(input_data)
        
        # Post-process predictions
        risk_map = self._postprocess_predictions(predictions, region_bounds)
        
        # Save risk map
        if output_path:
            self._save_risk_map(risk_map, region_bounds, output_path)
        
        # Calculate risk statistics
        risk_stats = self._calculate_risk_statistics(risk_map)
        
        return {
            'risk_map': risk_map,
            'statistics': risk_stats,
            'region_bounds': region_bounds,
            'date': date,
            'output_path': output_path
        }
    
    def _preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input data for model prediction"""
        
        # Normalize data
        if input_data.max() > 1.0:
            input_data = input_data / 255.0
        
        # Ensure correct shape
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Pad if necessary
        target_shape = self.config['input_shape'][:2]
        if input_data.shape[1:3] != target_shape:
            input_data = self._pad_to_target_size(input_data, target_shape)
        
        return input_data
    
    def _pad_to_target_size(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Pad data to target size"""
        current_height, current_width = data.shape[1:3]
        target_height, target_width = target_size
        
        # Calculate padding
        pad_height = max(0, target_height - current_height)
        pad_width = max(0, target_width - current_width)
        
        # Pad symmetrically
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        return np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
    
    def _load_region_data(self, region_bounds: Tuple[float, float, float, float], 
                         date: str) -> np.ndarray:
        """Load and prepare region data for prediction"""
        
        # This would integrate with data collection modules
        # For now, return dummy data
        height, width = self.config['input_shape'][:2]
        channels = self.config['input_shape'][2]
        
        # Generate dummy multi-band data
        data = np.random.rand(height, width, channels)
        
        return data
    
    def _postprocess_predictions(self, predictions: np.ndarray, 
                               region_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Post-process model predictions"""
        
        # Apply threshold for binary classification
        threshold = 0.5
        binary_predictions = (predictions > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up predictions
        from scipy import ndimage
        
        # Remove small objects
        binary_predictions = ndimage.binary_opening(binary_predictions, structure=np.ones((3, 3)))
        
        # Fill holes
        binary_predictions = ndimage.binary_fill_holes(binary_predictions)
        
        return binary_predictions
    
    def _save_risk_map(self, risk_map: np.ndarray, region_bounds: Tuple[float, float, float, float], 
                      output_path: str):
        """Save risk map as GeoTIFF"""
        
        # Create transform from bounds
        transform = from_bounds(*region_bounds, risk_map.shape[1], risk_map.shape[0])
        
        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=risk_map.shape[0],
            width=risk_map.shape[1],
            count=1,
            dtype=risk_map.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(risk_map, 1)
        
        logger.info(f"Risk map saved to {output_path}")
    
    def _calculate_risk_statistics(self, risk_map: np.ndarray) -> Dict[str, float]:
        """Calculate risk statistics from prediction map"""
        
        total_pixels = risk_map.size
        fire_pixels = np.sum(risk_map)
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        # Calculate risk level
        if fire_percentage < 1:
            risk_level = 'low'
        elif fire_percentage < 5:
            risk_level = 'medium'
        elif fire_percentage < 15:
            risk_level = 'high'
        else:
            risk_level = 'extreme'
        
        return {
            'fire_pixels': int(fire_pixels),
            'total_pixels': int(total_pixels),
            'fire_percentage': float(fire_percentage),
            'risk_level': risk_level
        }
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = self.model_path
        
        if self.model is not None:
            self.model.save(filepath)
            
            # Save model metadata
            model_info = {
                'model_name': 'unet_fire_prediction',
                'version': '1.0.0',
                'created_at': datetime.now(),
                'config': self.config,
                'filepath': filepath,
                'input_shape': self.config['input_shape'],
                'output_shape': (1,)
            }
            ModelMetadata.save_model_info(model_info)
            
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("No model to save. Train the model first.")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = self.model_path
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(
                filepath,
                custom_objects={
                    '_dice_loss': self._dice_loss,
                    '_dice_coefficient': self._dice_coefficient,
                    '_iou_score': self._iou_score
                }
            )
            logger.info(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")
    
    def evaluate(self, test_data) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        
        if self.model is None:
            self.load_model()
        
        # Evaluate model
        results = self.model.evaluate(test_data, verbose=0)
        
        # Create metrics dictionary
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'dice_coefficient': results[2],
            'iou_score': results[3],
            'precision': results[4],
            'recall': results[5]
        }
        
        logger.info(f"Model evaluation results: {metrics}")
        
        return metrics 