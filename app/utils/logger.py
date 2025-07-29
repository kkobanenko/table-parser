import logging
import os
from datetime import datetime
from config.settings import settings


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера с файловым и консольным выводом

    Args:
        name (str): Имя логгера
        level (int): Уровень логирования

    Returns:
        logging.Logger: Сконфигурированный логгер
    """
    # Создание логгера
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Очистка существующих хендлеров
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Форматтеры
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Файловый логгер
    log_file_path = os.path.join(
        settings.LOG_DIR,
        f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Консольный логгер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Добавление хендлеров
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger