#!/usr/bin/env python3
"""
Демо скрипт для тестирования DocTR (Mindee) парсера.

Этот скрипт демонстрирует работу DocTR парсера на файле "Нанолек 04.03.25.pdf"
и сравнивает результаты с тестовыми данными.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from parsers.doctr_parser import DocTRParser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_doctr_parser():
    """Тестирует DocTR парсер на тестовом файле."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔄 Тестируем DocTR парсер на файле: {test_file}")
    
    try:
        # Инициализация парсера
        parser = DocTRParser(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=False
        )
        
        logger.info("✅ DocTR парсер инициализирован")
        
        # Извлечение таблиц
        tables = parser.extract_tables(str(test_file))
        
        logger.info(f"📊 DocTR извлек {len(tables)} таблиц")
        
        # Анализ результатов
        for i, table in enumerate(tables):
            logger.info(f"📋 Таблица {i+1}: {len(table)} строк × {len(table[0]) if table else 0} колонок")
            
            # Показываем первые несколько строк
            if table:
                logger.info("📄 Первые 3 строки:")
                for j, row in enumerate(table[:3]):
                    logger.info(f"  Строка {j+1}: {row[:5]}...")  # Показываем первые 5 колонок
        
        # Сравнение с тестовыми данными
        compare_with_test_data(tables)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования DocTR: {e}")
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
    logger.info("🚀 Запуск демо тестирования DocTR парсера")
    
    success = test_doctr_parser()
    
    if success:
        logger.info("✅ Тестирование DocTR завершено успешно")
    else:
        logger.error("❌ Тестирование DocTR завершено с ошибками")
        sys.exit(1)
