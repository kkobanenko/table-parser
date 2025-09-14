#!/usr/bin/env python3
"""
Демо скрипт для тестирования отдельных компонентов пайплайна.

Этот скрипт позволяет тестировать каждый компонент пайплайна отдельно:
- PDF → Image конвертация
- Предобработка изображений
- Layout детекция
- OCR
- Детекция таблиц
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from pipeline.document_pipeline import DocumentPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_to_image():
    """Тестирует конвертацию PDF в изображения."""
    logger.info("🔄 Тестируем PDF → Image конвертацию")
    
    test_file = Path("input/Нанолек 04.03.25.pdf")
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    try:
        pipeline = DocumentPipeline()
        pages = pipeline._convert_pdf_to_images(str(test_file))
        
        logger.info(f"✅ Конвертировано {len(pages)} страниц")
        for i, page in enumerate(pages):
            if hasattr(page, 'shape'):
                logger.info(f"  Страница {i+1}: {page.shape}")
            else:
                logger.info(f"  Страница {i+1}: PIL Image")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка конвертации PDF: {e}")
        return False

def test_preprocessing():
    """Тестирует предобработку изображений."""
    logger.info("🔄 Тестируем предобработку изображений")
    
    test_file = Path("input/Нанолек 04.03.25.pdf")
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    try:
        pipeline = DocumentPipeline()
        pipeline.configure_pipeline({
            'enable_preprocessing': True,
            'enable_deskew': True,
            'enable_denoise': True,
            'enable_binarize': False
        })
        
        pages = pipeline._convert_pdf_to_images(str(test_file))
        processed_pages = pipeline._preprocess_images(pages)
        
        logger.info(f"✅ Обработано {len(processed_pages)} страниц")
        for i, page in enumerate(processed_pages):
            if isinstance(page, dict) and 'processed_image' in page:
                img = page['processed_image']
                if hasattr(img, 'shape'):
                    logger.info(f"  Страница {i+1}: {img.shape}")
                else:
                    logger.info(f"  Страница {i+1}: PIL Image")
            else:
                logger.info(f"  Страница {i+1}: обработана")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка предобработки: {e}")
        return False

def test_layout_detection():
    """Тестирует детекцию layout."""
    logger.info("🔄 Тестируем детекцию layout")
    
    test_file = Path("input/Нанолек 04.03.25.pdf")
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    try:
        pipeline = DocumentPipeline()
        pipeline.configure_pipeline({
            'enable_layout_detection': True,
            'layout_model': 'lp://EfficientDete/PubLayNet'
        })
        pipeline.initialize_components()
        
        pages = pipeline._convert_pdf_to_images(str(test_file))
        layout_regions = pipeline._detect_layout_regions(pages)
        
        logger.info(f"✅ Найдено {len(layout_regions)} layout зон")
        
        # Группируем по типам
        region_types = {}
        for region in layout_regions:
            region_type = region.get('type', 'unknown')
            region_types[region_type] = region_types.get(region_type, 0) + 1
        
        for region_type, count in region_types.items():
            logger.info(f"  {region_type}: {count} зон")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка детекции layout: {e}")
        return False

def test_ocr():
    """Тестирует OCR."""
    logger.info("🔄 Тестируем OCR")
    
    test_file = Path("input/Нанолек 04.03.25.pdf")
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    try:
        pipeline = DocumentPipeline()
        pipeline.configure_pipeline({
            'enable_ocr': True,
            'ocr_method': 'paddleocr'
        })
        pipeline.initialize_components()
        
        pages = pipeline._convert_pdf_to_images(str(test_file))
        
        # Конвертируем PIL Images в numpy arrays и выполняем OCR
        ocr_results = []
        for i, image in enumerate(pages):
            import numpy as np
            import cv2
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            text = pipeline._perform_ocr(img_array)
            ocr_results.append({
                'page_number': i + 1,
                'text': text
            })
        
        logger.info(f"✅ OCR выполнен на {len(ocr_results)} страницах")
        
        total_text_length = 0
        for i, result in enumerate(ocr_results):
            text_length = len(result.get('text', ''))
            total_text_length += text_length
            logger.info(f"  Страница {i+1}: {text_length} символов")
        
        logger.info(f"📝 Общая длина текста: {total_text_length} символов")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка OCR: {e}")
        return False

def test_table_detection():
    """Тестирует детекцию таблиц."""
    logger.info("🔄 Тестируем детекцию таблиц")
    
    test_file = Path("input/Нанолек 04.03.25.pdf")
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    try:
        pipeline = DocumentPipeline()
        pipeline.configure_pipeline({
            'enable_table_detection': True
        })
        pipeline.initialize_components()
        
        pages = pipeline._convert_pdf_to_images(str(test_file))
        ocr_results = pipeline._perform_ocr(pages)
        tables = pipeline._detect_tables(ocr_results)
        
        logger.info(f"✅ Найдено {len(tables)} таблиц")
        
        for i, table in enumerate(tables):
            if table:
                rows = len(table)
                cols = len(table[0]) if table else 0
                logger.info(f"  Таблица {i+1}: {rows} строк × {cols} колонок")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка детекции таблиц: {e}")
        return False

def main():
    """Главная функция с аргументами командной строки."""
    parser = argparse.ArgumentParser(description='Тестирование компонентов пайплайна')
    parser.add_argument('component', nargs='?', default='all',
                       choices=['all', 'pdf2image', 'preprocessing', 'layout', 'ocr', 'tables'],
                       help='Компонент для тестирования')
    
    args = parser.parse_args()
    
    logger.info("🚀 Запуск тестирования компонентов пайплайна")
    
    success = True
    
    if args.component == 'all' or args.component == 'pdf2image':
        success &= test_pdf_to_image()
    
    if args.component == 'all' or args.component == 'preprocessing':
        success &= test_preprocessing()
    
    if args.component == 'all' or args.component == 'layout':
        success &= test_layout_detection()
    
    if args.component == 'all' or args.component == 'ocr':
        success &= test_ocr()
    
    if args.component == 'all' or args.component == 'tables':
        success &= test_table_detection()
    
    if success:
        logger.info("✅ Все тесты завершены успешно")
    else:
        logger.error("❌ Некоторые тесты завершились с ошибками")
        sys.exit(1)

if __name__ == "__main__":
    main()
