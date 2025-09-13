import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


class Settings:
    """
    Централизованный класс настроек приложения
    Загружает и предоставляет глобальные параметры конфигурации
    """

    # Базовые пути
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    TEMP_DIR: Path = BASE_DIR / 'temp'
    LOG_DIR: Path = BASE_DIR / 'logs'
    SCREENSHOTS_DIR: Path = BASE_DIR / 'screenshots'

    # Параметры парсинга
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', 50))
    MAX_FILES_PER_BATCH: int = int(os.getenv('MAX_FILES_PER_BATCH', 10))

    # OCR настройки
    OCR_LANGUAGES: str = os.getenv('OCR_LANGUAGES', 'rus+eng')
    OCR_CONFIDENCE_THRESHOLD: int = int(os.getenv('OCR_CONFIDENCE_THRESHOLD', 60))

    # Пути для хранения временных файлов
    TEMP_UPLOAD_DIR: Path = TEMP_DIR / 'uploads'
    TEMP_PROCESSING_DIR: Path = TEMP_DIR / 'processing'

    def __init__(self):
        """
        Инициализация директорий при создании экземпляра
        Гарантирует наличие всех необходимых директорий
        """
        self._create_directories()

    def _create_directories(self):
        """
        Создание всех необходимых директорий
        Метод вызывается при инициализации класса
        """
        try:
            dirs_to_create = [
                self.TEMP_DIR,
                self.LOG_DIR,
                self.SCREENSHOTS_DIR,
                self.TEMP_UPLOAD_DIR,
                self.TEMP_PROCESSING_DIR
            ]

            for directory in dirs_to_create:
                directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Если нет прав на создание директорий, используем временные
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / 'table_parser'
            self.TEMP_DIR = temp_dir
            self.LOG_DIR = temp_dir / 'logs'
            self.SCREENSHOTS_DIR = temp_dir / 'screenshots'
            self.TEMP_UPLOAD_DIR = temp_dir / 'uploads'
            self.TEMP_PROCESSING_DIR = temp_dir / 'processing'
            
            for directory in [self.TEMP_DIR, self.LOG_DIR, self.SCREENSHOTS_DIR, 
                            self.TEMP_UPLOAD_DIR, self.TEMP_PROCESSING_DIR]:
                directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_db_connection_params(cls) -> dict:
        """
        Получение параметров подключения к базе данных
        Используется для расширенной функциональности
        """
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'table_parser_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

    def get_logging_config(self) -> dict:
        """
        Конфигурация логирования

        Returns:
            dict: Настройки для библиотеки logging
        """
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'file_handler': {
                    'class': 'logging.FileHandler',
                    'filename': self.LOG_DIR / f'table_parser_{os.getpid()}.log',
                    'formatter': 'standard',
                    'level': 'INFO'
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard'
                }
            },
            'loggers': {
                '': {  # root logger
                    'handlers': ['file_handler', 'console'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }


# Создание глобального экземпляра настроек
settings = Settings()