import os
import shutil
from datetime import datetime, timedelta
from typing import List, Optional
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger('file_handler')


class FileHandler:
    """
    Утилита для работы с временными файлами и директориями
    """

    @staticmethod
    def cleanup_temp_files(
            directory: str = None,
            max_age_hours: int = 24
    ) -> List[str]:
        """
        Удаление устаревших временных файлов

        Args:
            directory (str, optional): Директория для очистки
            max_age_hours (int): Максимальный возраст файла в часах

        Returns:
            List[str]: Список удаленных файлов
        """
        directory = directory or settings.TEMP_DIR
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_files = []

        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)

                    try:
                        # Получение времени последнего изменения
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                        if mtime < cutoff_time:
                            os.unlink(file_path)
                            deleted_files.append(file_path)
                            logger.info(f"Удален файл: {file_path}")
                    except Exception as e:
                        logger.warning(f"Ошибка при удалении {file_path}: {e}")

        except Exception as e:
            logger.error(f"Ошибка очистки временных файлов: {e}")

        return deleted_files

    @staticmethod
    def save_uploaded_file(
            uploaded_file,
            directory: Optional[str] = None
    ) -> str:
        """
        Сохранение загруженного файла

        Args:
            uploaded_file: Загруженный файл
            directory (Optional[str]): Директория для сохранения

        Returns:
            str: Путь к сохраненному файлу
        """
        directory = directory or settings.TEMP_UPLOAD_DIR
        os.makedirs(directory, exist_ok=True)

        # Генерация уникального имени файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{uploaded_file.name}"
        filepath = os.path.join(directory, filename)

        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Файл сохранен: {filepath}")
        return filepath

    @staticmethod
    def get_file_size(filepath: str) -> int:
        """
        Получение размера файла в байтах

        Args:
            filepath (str): Путь к файлу

        Returns:
            int: Размер файла
        """
        return os.path.getsize(filepath)