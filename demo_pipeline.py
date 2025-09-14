#!/usr/bin/env python3
"""
Демо скрипт для тестирования комплексного пайплайна обработки документов.

Этот скрипт демонстрирует работу полного пайплайна:
PDF→Image→Preprocessing→Layout→OCR→Tables→Export
"""

import sys
import os
from pathlib import Path
import logging

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from pipeline.document_pipeline import DocumentPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Тестирует комплексный пайплайн на тестовом файле."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔄 Тестируем комплексный пайплайн на файле: {test_file}")
    
    try:
        # Инициализация пайплайна
        pipeline = DocumentPipeline()
        
        # Конфигурация пайплайна (все компоненты включены)
        config = {
            # Предобработка
            'enable_preprocessing': True,
            'enable_deskew': True,
            'enable_denoise': True,
            'enable_binarize': False,  # Отключаем бинаризацию для лучшего качества OCR
            
            # Layout детекция
            'enable_layout_detection': True,
            'layout_model': 'lp://EfficientDete/PubLayNet',
            
            # OCR
            'enable_ocr': True,
            'ocr_method': 'paddleocr',  # Лучший для русского языка
            
            # Детекция таблиц
            'enable_table_detection': True,
            
            # Экспорт
            'enable_export': True,
            'export_formats': ['json', 'excel', 'csv', 'txt']
        }
        
        # Конфигурируем пайплайн
        pipeline.configure_pipeline(config)
        
        # Инициализируем компоненты
        pipeline.initialize_components()
        
        logger.info("✅ Пайплайн инициализирован")
        
        # Обрабатываем документ
        results = pipeline.process_document(str(test_file))
        
        # Анализируем результаты
        logger.info("📊 Результаты обработки:")
        logger.info(f"  📄 Страниц обработано: {len(results['pages'])}")
        logger.info(f"  📊 Таблиц найдено: {len(results['tables'])}")
        logger.info(f"  🎯 Layout зон найдено: {len(results['layout_regions'])}")
        logger.info(f"  📝 Длина извлеченного текста: {len(results['text'])} символов")
        logger.info(f"  💾 Файлов экспорта: {len(results['export_files'])}")
        
        # Детальный анализ таблиц
        if results['tables']:
            logger.info("📋 Детали таблиц:")
            for i, table in enumerate(results['tables']):
                if table:
                    rows = len(table)
                    cols = len(table[0]) if table else 0
                    logger.info(f"  Таблица {i+1}: {rows} строк × {cols} колонок")
                    
                    # Показываем первые несколько строк
                    if rows > 0:
                        logger.info(f"    Первая строка: {table[0][:5]}...")  # Первые 5 колонок
                        if rows > 1:
                            logger.info(f"    Вторая строка: {table[1][:5]}...")
        
        # Анализ layout зон
        if results['layout_regions']:
            logger.info("🎯 Детали layout зон:")
            region_types = {}
            for region in results['layout_regions']:
                region_type = region.get('type', 'unknown')
                region_types[region_type] = region_types.get(region_type, 0) + 1
            
            for region_type, count in region_types.items():
                logger.info(f"  {region_type}: {count} зон")
        
        # Информация об экспорте
        if results['export_files']:
            logger.info("💾 Экспортированные файлы:")
            for file_path in results['export_files']:
                logger.info(f"  {file_path}")
        
        # Сравнение с тестовыми данными
        compare_with_test_data(results['tables'])
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования пайплайна: {e}")
        return False

def compare_with_test_data(tables):
    """Сравнивает результаты с тестовыми данными."""
    
    # Тестовые данные из .cursor/rules/file-testing.mdc
    expected_headers = ["", "Торговое наименование препарата", "МНН или химическое (групповое) наименование", 
                       "Форма выпуска", "Цена ЖНВЛП (без НДС)", "Цена за упаковку (без НДС)", 
                       "Цена за упаковку (с НДС)", "Производитель"]
    
    expected_first_row = ["1", "Вакцина для профилактики дифтерии (с уменьшенным содержанием антигена), коклюша (с уменьшенным содержанием антигена, бесклеточная) и столбняка, адсорбированная",
                         "суспензия для внутримышечного введения, 0.5 мл/доза, 0.5 мл - флаконы (1) - пачки картонные",
                         "-", "2122,26", "2334,49", "Санофи Пастер Лимитед, Канада"]
    
    logger.info("🔍 Сравнение с тестовыми данными:")
    logger.info(f"📝 Ожидаемые заголовки: {len(expected_headers)} колонок")
    logger.info(f"📄 Ожидаемая первая строка: {len(expected_first_row)} колонок")
    
    if not tables:
        logger.warning("⚠️ Таблицы не найдены")
        return
    
    # Ищем таблицу с наиболее подходящими заголовками
    best_match_score = 0
    best_table_idx = -1
    
    for i, table in enumerate(tables):
        if not table or len(table) == 0:
            continue
        
        # Проверяем заголовки
        headers = table[0] if len(table) > 0 else []
        header_score = calculate_match_score(headers, expected_headers)
        
        logger.info(f"📊 Таблица {i+1}: заголовки - {header_score:.1%} совпадение")
        
        if header_score > best_match_score:
            best_match_score = header_score
            best_table_idx = i
    
    if best_table_idx >= 0:
        logger.info(f"🏆 Лучшая таблица: {best_table_idx+1} с {best_match_score:.1%} совпадением заголовков")
        
        # Проверяем первую строку данных
        best_table = tables[best_table_idx]
        if len(best_table) > 1:
            first_row = best_table[1]  # Первая строка данных
            row_score = calculate_match_score(first_row, expected_first_row)
            logger.info(f"📄 Первая строка данных: {row_score:.1%} совпадение")
            
            # Общий счет
            total_score = (best_match_score + row_score) / 2
            logger.info(f"🎯 Общий счет: {total_score:.1%}")
            
            if total_score >= 0.5:
                logger.info("✅ Цель достигнута: 50%+ поячечное совпадение!")
            else:
                logger.warning(f"⚠️ Цель не достигнута: {total_score:.1%} < 50%")
    else:
        logger.warning("⚠️ Подходящие таблицы не найдены")

def calculate_match_score(actual, expected):
    """Вычисляет процент совпадения между фактическими и ожидаемыми данными."""
    if not actual or not expected:
        return 0.0
    
    matches = 0
    min_len = min(len(actual), len(expected))
    
    for i in range(min_len):
        if actual[i] and expected[i]:
            # Простое сравнение строк (можно улучшить)
            if actual[i].strip().lower() in expected[i].strip().lower() or \
               expected[i].strip().lower() in actual[i].strip().lower():
                matches += 1
    
    return matches / len(expected) if expected else 0.0

if __name__ == "__main__":
    logger.info("🚀 Запуск демо тестирования комплексного пайплайна")
    
    success = test_pipeline()
    
    if success:
        logger.info("✅ Тестирование пайплайна завершено успешно")
    else:
        logger.error("❌ Тестирование пайплайна завершено с ошибками")
        sys.exit(1)
