#!/usr/bin/env python3
"""
Комбинированный демо скрипт для тестирования всех методов извлечения таблиц.

Этот скрипт сравнивает результаты различных методов на файле "Нанолек 04.03.25.pdf"
и определяет лучший метод для достижения 50%+ поячечного совпадения.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from pdf2image import convert_from_path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

# Импорты парсеров
from parsers.paddleocr_parser import PaddleOCRParser
from parsers.doctr_parser import DocTRParser
from parsers.layoutparser_parser import LayoutParserParser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_methods():
    """Тестирует все доступные методы извлечения таблиц."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔄 Тестируем все методы на файле: {test_file}")
    
    results = {}
    
    # 1. Тестируем PaddleOCR PP-OCRv5
    logger.info("\n" + "="*60)
    logger.info("🚀 ТЕСТИРОВАНИЕ PADDLEOCR PP-OCRV5")
    logger.info("="*60)
    
    try:
        paddleocr_parser = PaddleOCRParser(
            use_server_model=True,
            lang='ru'
        )
        
        # Конвертируем PDF в изображения
        images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=2)
        
        paddleocr_tables = []
        for page_idx, image in enumerate(images, start=1):
            temp_image_path = f"temp/paddleocr_page_{page_idx}.png"
            Path("temp").mkdir(exist_ok=True)
            image.save(temp_image_path)
            
            try:
                tables = paddleocr_parser.extract_tables(temp_image_path)
                paddleocr_tables.extend(tables)
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        results['PaddleOCR'] = paddleocr_tables
        logger.info(f"✅ PaddleOCR: найдено {len(paddleocr_tables)} таблиц")
        
    except Exception as e:
        logger.error(f"❌ PaddleOCR failed: {e}")
        results['PaddleOCR'] = []
    
    # 2. Тестируем DocTR (Mindee)
    logger.info("\n" + "="*60)
    logger.info("🚀 ТЕСТИРОВАНИЕ DOCTR (MINDEE)")
    logger.info("="*60)
    
    try:
        doctr_parser = DocTRParser(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=False
        )
        
        doctr_tables = doctr_parser.extract_tables(str(test_file))
        results['DocTR'] = doctr_tables
        logger.info(f"✅ DocTR: найдено {len(doctr_tables)} таблиц")
        
    except Exception as e:
        logger.error(f"❌ DocTR failed: {e}")
        results['DocTR'] = []
    
    # 3. Тестируем LayoutParser
    logger.info("\n" + "="*60)
    logger.info("🚀 ТЕСТИРОВАНИЕ LAYOUTPARSER")
    logger.info("="*60)
    
    try:
        layoutparser_parser = LayoutParserParser(
            model_name='lp://EfficientDete/PubLayNet',
            confidence_threshold=0.8,
            ocr_agent='tesseract'
        )
        
        # Конвертируем PDF в изображения
        images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=2)
        
        layoutparser_tables = []
        for page_idx, image in enumerate(images, start=1):
            temp_image_path = f"temp/layoutparser_page_{page_idx}.png"
            Path("temp").mkdir(exist_ok=True)
            image.save(temp_image_path)
            
            try:
                tables = layoutparser_parser.extract_tables(temp_image_path)
                layoutparser_tables.extend(tables)
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        results['LayoutParser'] = layoutparser_tables
        logger.info(f"✅ LayoutParser: найдено {len(layoutparser_tables)} таблиц")
        
    except Exception as e:
        logger.error(f"❌ LayoutParser failed: {e}")
        results['LayoutParser'] = []
    
    # Анализ результатов
    logger.info("\n" + "="*60)
    logger.info("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    logger.info("="*60)
    
    analyze_results(results)
    
    return True

def analyze_results(results):
    """Анализирует результаты всех методов."""
    
    # Тестовые данные
    expected_headers = ["", "Торговое наименование препарата", "МНН или химическое (групповое) наименование", 
                       "Форма выпуска", "Цена ЖНВЛП (без НДС)", "Цена за упаковку (без НДС)", 
                       "Цена за упаковку (с НДС)", "Производитель"]
    
    expected_first_row = ["1", "Вакцина для профилактики дифтерии (с уменьшенным содержанием антигена), коклюша (с уменьшенным содержанием антигена, бесклеточная) и столбняка, адсорбированная",
                         "суспензия для внутримышечного введения, 0.5 мл/доза, 0.5 мл - флаконы (1) - пачки картонные",
                         "-", "2122,26", "2334,49", "Санофи Пастер Лимитед, Канада"]
    
    method_scores = {}
    
    for method_name, tables in results.items():
        logger.info(f"\n🔍 Анализ метода: {method_name}")
        
        if not tables:
            logger.warning(f"⚠️ {method_name}: таблицы не найдены")
            method_scores[method_name] = 0.0
            continue
        
        best_score = 0.0
        best_table_idx = -1
        
        for i, table in enumerate(tables):
            logger.info(f"🔍 Таблица {i}: type={type(table)}, len={len(table) if hasattr(table, '__len__') else 'N/A'}")
            if not table or not hasattr(table, '__len__') or len(table) == 0:
                continue
            
            # Проверяем заголовки
            headers = table[0] if len(table) > 0 else []
            header_score = calculate_match_score(headers, expected_headers)
            
            # Проверяем первую строку данных
            row_score = 0.0
            if len(table) > 1:
                first_row = table[1]
                row_score = calculate_match_score(first_row, expected_first_row)
            
            # Общий счет для таблицы
            table_score = (header_score + row_score) / 2
            
            logger.info(f"  📋 Таблица {i+1}: заголовки {header_score:.1%}, строка {row_score:.1%}, общий {table_score:.1%}")
            
            if table_score > best_score:
                best_score = table_score
                best_table_idx = i
        
        method_scores[method_name] = best_score
        
        if best_score >= 0.5:
            logger.info(f"✅ {method_name}: ЦЕЛЬ ДОСТИГНУТА! {best_score:.1%} >= 50%")
        else:
            logger.info(f"⚠️ {method_name}: цель не достигнута {best_score:.1%} < 50%")
    
    # Итоговый рейтинг
    logger.info("\n" + "="*60)
    logger.info("🏆 ИТОГОВЫЙ РЕЙТИНГ МЕТОДОВ")
    logger.info("="*60)
    
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, score) in enumerate(sorted_methods, 1):
        status = "✅ ЦЕЛЬ ДОСТИГНУТА" if score >= 0.5 else "⚠️ Цель не достигнута"
        logger.info(f"{i}. {method}: {score:.1%} - {status}")
    
    # Рекомендации
    logger.info("\n" + "="*60)
    logger.info("💡 РЕКОМЕНДАЦИИ")
    logger.info("="*60)
    
    best_method = sorted_methods[0][0] if sorted_methods else "Нет данных"
    best_score = sorted_methods[0][1] if sorted_methods else 0.0
    
    if best_score >= 0.5:
        logger.info(f"🎯 Лучший метод: {best_method} ({best_score:.1%})")
        logger.info("✅ Рекомендуется использовать этот метод для достижения цели 50%+")
    else:
        logger.info(f"🎯 Лучший метод: {best_method} ({best_score:.1%})")
        logger.info("⚠️ Ни один метод не достиг цели 50%+")
        logger.info("💡 Рекомендуется:")
        logger.info("   1. Улучшить алгоритмы извлечения таблиц")
        logger.info("   2. Попробовать комбинацию методов")
        logger.info("   3. Настроить параметры парсеров")

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
    logger.info("🚀 Запуск комбинированного тестирования всех методов")
    
    success = test_all_methods()
    
    if success:
        logger.info("✅ Тестирование всех методов завершено успешно")
    else:
        logger.error("❌ Тестирование всех методов завершено с ошибками")
        sys.exit(1)
