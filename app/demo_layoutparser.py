#!/usr/bin/env python3
"""
Демо скрипт для тестирования LayoutParser парсера.

Этот скрипт демонстрирует работу LayoutParser парсера на файле "Нанолек 04.03.25.pdf"
и сравнивает результаты с тестовыми данными.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from pdf2image import convert_from_path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from parsers.layoutparser_parser import LayoutParserParser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_layoutparser_parser():
    """Тестирует LayoutParser парсер на тестовом файле."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔄 Тестируем LayoutParser парсер на файле: {test_file}")
    
    try:
        # Инициализация парсера
        parser = LayoutParserParser(
            model_name='lp://EfficientDete/PubLayNet',
            confidence_threshold=0.8,
            ocr_agent='tesseract'
        )
        
        logger.info("✅ LayoutParser парсер инициализирован")
        
        # Конвертируем PDF в изображения для LayoutParser
        logger.info("📄 Конвертируем PDF в изображения...")
        images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=2)
        
        all_tables = []
        
        # Обрабатываем каждую страницу
        for page_idx, image in enumerate(images, start=1):
            logger.info(f"🔄 Обрабатываем страницу {page_idx}")
            
            # Сохраняем изображение во временный файл
            temp_image_path = f"temp/layoutparser_page_{page_idx}.png"
            Path("temp").mkdir(exist_ok=True)
            image.save(temp_image_path)
            
            try:
                # Извлечение таблиц со страницы
                tables = parser.extract_tables(temp_image_path)
                all_tables.extend(tables)
                
                logger.info(f"📊 Страница {page_idx}: найдено {len(tables)} таблиц")
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка обработки страницы {page_idx}: {e}")
            
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        logger.info(f"📊 LayoutParser извлек {len(all_tables)} таблиц всего")
        
        # Анализ результатов
        for i, table in enumerate(all_tables):
            logger.info(f"📋 Таблица {i+1}: {len(table)} строк × {len(table[0]) if table else 0} колонок")
            
            # Показываем первые несколько строк
            if table:
                logger.info("📄 Первые 3 строки:")
                for j, row in enumerate(table[:3]):
                    logger.info(f"  Строка {j+1}: {row[:5]}...")  # Показываем первые 5 колонок
        
        # Сравнение с тестовыми данными
        compare_with_test_data(all_tables)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования LayoutParser: {e}")
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
        if not table:
            continue
            
        # Проверяем заголовки
        headers = table[0] if table else []
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
    logger.info("🚀 Запуск демо тестирования LayoutParser парсера")
    
    success = test_layoutparser_parser()
    
    if success:
        logger.info("✅ Тестирование LayoutParser завершено успешно")
    else:
        logger.error("❌ Тестирование LayoutParser завершено с ошибками")
        sys.exit(1)
