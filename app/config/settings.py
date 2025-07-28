import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # Application Settings
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 20))
    MAX_FILES_PER_BATCH = int(os.getenv('MAX_FILES_PER_BATCH', 10))
    TEMP_DIR = os.getenv('TEMP_DIR', '/app/temp')
    LOG_DIR = os.getenv('LOG_DIR', '/app/logs')
    
    # OCR Settings
    OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'rus+eng')
    OCR_CONFIDENCE_THRESHOLD = int(os.getenv('OCR_CONFIDENCE_THRESHOLD', 60))
    
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(f"{self.TEMP_DIR}/screenshots", exist_ok=True)
