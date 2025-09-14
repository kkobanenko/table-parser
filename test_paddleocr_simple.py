#!/usr/bin/env python3
"""
Простой тест PaddleOCR для понимания структуры данных.
"""

import sys
from pathlib import Path
import logging
from pdf2image import convert_from_path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from parsers.paddleocr_parser import PaddleOCRParser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_paddleocr_simple():
    """Простой тест PaddleOCR."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔄 Тестируем PaddleOCR на файле: {test_file}")
    
    try:
        # Инициализация парсера
        parser = PaddleOCRParser(
            use_server_model=True,
            lang='ru'
        )
        
        logger.info("✅ PaddleOCR парсер инициализирован")
        
        # Конвертируем PDF в изображения
        images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=1)
        
        # Обрабатываем только первую страницу
        image = images[0]
        temp_image_path = "temp/paddleocr_test.png"
        Path("temp").mkdir(exist_ok=True)
        image.save(temp_image_path)
        
        try:
            # Извлечение таблиц
            tables = parser.extract_tables(temp_image_path)
            
            logger.info(f"📊 PaddleOCR извлек {len(tables)} таблиц")
            
            # Анализ структуры данных
            for i, table in enumerate(tables):
                logger.info(f"🔍 Таблица {i}: type={type(table)}")
                logger.info(f"🔍 Таблица {i}: dir={dir(table)}")
                
                if hasattr(table, '__len__'):
                    logger.info(f"🔍 Таблица {i}: len={len(table)}")
                    
                    if len(table) > 0:
                        logger.info(f"🔍 Таблица {i}: первый элемент type={type(table[0])}")
                        logger.info(f"🔍 Таблица {i}: первый элемент={table[0]}")
                        
                        if len(table) > 1:
                            logger.info(f"🔍 Таблица {i}: второй элемент={table[1]}")
                else:
                    logger.info(f"🔍 Таблица {i}: не имеет __len__")
        
        finally:
            # Удаляем временный файл
            if Path(temp_image_path).exists():
                Path(temp_image_path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования PaddleOCR: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Запуск простого тестирования PaddleOCR")
    
    success = test_paddleocr_simple()
    
    if success:
        logger.info("✅ Тестирование PaddleOCR завершено успешно")
    else:
        logger.error("❌ Тестирование PaddleOCR завершено с ошибками")
        sys.exit(1)
