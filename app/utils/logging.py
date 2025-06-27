"""
Logging configuration for Forest Fire Prediction System
"""

import os
import logging
import logging.handlers
from datetime import datetime
from app.utils.config import Config

def setup_logging():
    """Setup structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(Config.LOGS_DIR, 'error.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
        'Exception: %(exc_info)s\n'
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {Config.LOG_LEVEL}")
    logger.info(f"Log file: {Config.LOG_FILE}")

def get_logger(name):
    """Get a logger instance with the given name"""
    return logging.getLogger(name)

class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, logger_name='performance'):
        self.logger = get_logger(logger_name)
    
    def log_api_request(self, endpoint, method, response_time, status_code):
        """Log API request performance"""
        self.logger.info(
            f"API Request - {method} {endpoint} - "
            f"Response Time: {response_time:.3f}s - "
            f"Status: {status_code}"
        )
    
    def log_model_prediction(self, model_name, prediction_time, accuracy=None):
        """Log model prediction performance"""
        log_msg = f"Model Prediction - {model_name} - Time: {prediction_time:.3f}s"
        if accuracy is not None:
            log_msg += f" - Accuracy: {accuracy:.4f}"
        self.logger.info(log_msg)
    
    def log_data_processing(self, operation, processing_time, data_size=None):
        """Log data processing performance"""
        log_msg = f"Data Processing - {operation} - Time: {processing_time:.3f}s"
        if data_size is not None:
            log_msg += f" - Size: {data_size}"
        self.logger.info(log_msg)

class SecurityLogger:
    """Security event logging utility"""
    
    def __init__(self, logger_name='security'):
        self.logger = get_logger(logger_name)
    
    def log_login_attempt(self, username, ip_address, success):
        """Log login attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.warning(
            f"Login Attempt - User: {username} - IP: {ip_address} - Status: {status}"
        )
    
    def log_api_access(self, endpoint, method, ip_address, user_agent):
        """Log API access"""
        self.logger.info(
            f"API Access - {method} {endpoint} - IP: {ip_address} - "
            f"User-Agent: {user_agent}"
        )
    
    def log_suspicious_activity(self, activity_type, details, ip_address):
        """Log suspicious activities"""
        self.logger.error(
            f"Suspicious Activity - Type: {activity_type} - "
            f"Details: {details} - IP: {ip_address}"
        )

class DataLogger:
    """Data operation logging utility"""
    
    def __init__(self, logger_name='data'):
        self.logger = get_logger(logger_name)
    
    def log_data_collection(self, source, status, records_count=None, error=None):
        """Log data collection operations"""
        log_msg = f"Data Collection - Source: {source} - Status: {status}"
        if records_count is not None:
            log_msg += f" - Records: {records_count}"
        if error:
            log_msg += f" - Error: {error}"
        
        if status == 'SUCCESS':
            self.logger.info(log_msg)
        else:
            self.logger.error(log_msg)
    
    def log_data_validation(self, dataset_name, validation_result, issues_count=0):
        """Log data validation results"""
        status = "PASSED" if validation_result else "FAILED"
        log_msg = f"Data Validation - Dataset: {dataset_name} - Status: {status}"
        if issues_count > 0:
            log_msg += f" - Issues: {issues_count}"
        
        if validation_result:
            self.logger.info(log_msg)
        else:
            self.logger.error(log_msg)
    
    def log_model_training(self, model_name, epoch, loss, accuracy, validation_accuracy=None):
        """Log model training progress"""
        log_msg = f"Model Training - {model_name} - Epoch: {epoch} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}"
        if validation_accuracy is not None:
            log_msg += f" - Val Accuracy: {validation_accuracy:.4f}"
        self.logger.info(log_msg)

# Global logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
data_logger = DataLogger() 