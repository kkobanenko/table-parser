import os
import shutil
from datetime import datetime, timedelta
from typing import List
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger('file_handler')

class FileHandler:
    """Handle temporary file operations"""
    
    def __init__(self):
        self.settings = Settings()
        self.temp_dir = self.settings.TEMP_DIR
    
    def cleanup_old_files(self, hours: int = 24):
        """Remove temporary files older than specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def save_temp_file(self, content: bytes, filename: str) -> str:
        """Save content to temporary file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved temporary file: {temp_path}")
        return temp_path
    
    def get_temp_file_path(self, filename: str) -> str:
        """Get full path for temporary file"""
        return os.path.join(self.temp_dir, filename)
