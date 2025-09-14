#!/usr/bin/env python3
"""
Демо скрипт для сравнения качества OCR с удалением печатей и без.

Этот скрипт сравнивает:
1. OCR без предобработки
2. OCR с удалением печатей
3. OCR с полной предобработкой
4. Анализ поячечного совпадения с тестовыми данными
"""

import sys
import os
from pathlib import Path
import logging
import json

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from pipeline.document_pipeline import DocumentPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ocr_scenarios():
    """Тестирует различные сценарии OCR."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔍 Сравниваем качество OCR на файле: {test_file}")
    
    # Создаем папку temp для промежуточных файлов
    Path("temp").mkdir(exist_ok=True)
    
    # Сценарии тестирования
    scenarios = [
        {
            'name': 'Без предобработки',
            'config': {
                'enable_preprocessing': False,
                'enable_stamp_removal': False,
                'enable_deskew': False,
                'enable_denoise': False,
                'enable_binarize': False,
                'enable_layout_detection': False,
                'enable_ocr': True,
                'ocr_method': 'paddleocr',
                'enable_table_detection': True,
                'enable_export': False
            }
        },
        {
            'name': 'Только удаление печатей',
            'config': {
                'enable_preprocessing': True,
                'enable_stamp_removal': True,
                'enable_deskew': False,
                'enable_denoise': False,
                'enable_binarize': False,
                'enable_layout_detection': False,
                'enable_ocr': True,
                'ocr_method': 'paddleocr',
                'enable_table_detection': True,
                'enable_export': False,
                'stamp_removal_method': 'white_replacement',
                'blue_hsv_thresholds': {'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50},
                'red_hsv_thresholds': {'h_min': 0, 'h_max': 10, 'h_min2': 170, 'h_max2': 180, 's_min': 50, 'v_min': 50},
                'morphology_kernel_size': 5
            }
        },
        {
            'name': 'Полная предобработка',
            'config': {
                'enable_preprocessing': True,
                'enable_stamp_removal': True,
                'enable_deskew': True,
                'enable_denoise': True,
                'enable_binarize': False,  # Отключаем для лучшего качества OCR
                'enable_layout_detection': False,
                'enable_ocr': True,
                'ocr_method': 'paddleocr',
                'enable_table_detection': True,
                'enable_export': False,
                'stamp_removal_method': 'white_replacement',
                'blue_hsv_thresholds': {'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50},
                'red_hsv_thresholds': {'h_min': 0, 'h_max': 10, 'h_min2': 170, 'h_max2': 180, 's_min': 50, 'v_min': 50},
                'morphology_kernel_size': 5
            }
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"\n🔄 Тестируем сценарий: {scenario['name']}")
        
        try:
            # Инициализация пайплайна
            pipeline = DocumentPipeline()
            pipeline.configure_pipeline(scenario['config'])
            pipeline.initialize_components()
            
            # Обрабатываем документ
            result = pipeline.process_document(str(test_file))
            
            # Анализируем результаты
            scenario_result = {
                'name': scenario['name'],
                'config': scenario['config'],
                'pages_processed': len(result['pages']),
                'tables_found': len(result['tables']),
                'text_length': len(result['text']),
                'tables': result['tables']
            }
            
            results.append(scenario_result)
            
            logger.info(f"✅ {scenario['name']}:")
            logger.info(f"  📄 Страниц: {scenario_result['pages_processed']}")
            logger.info(f"  📊 Таблиц: {scenario_result['tables_found']}")
            logger.info(f"  📝 Текст: {scenario_result['text_length']} символов")
            
            # Детальный анализ таблиц
            if result['tables']:
                for i, table in enumerate(result['tables']):
                    if table:
                        rows = len(table)
                        cols = len(table[0]) if table else 0
                        logger.info(f"  Таблица {i+1}: {rows}×{cols}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в сценарии '{scenario['name']}': {e}")
            results.append({
                'name': scenario['name'],
                'error': str(e),
                'tables': []
            })
    
    # Сравниваем результаты с тестовыми данными
    compare_with_test_data(results)
    
    # Сохраняем результаты
    save_results(results)
    
    return True

def compare_with_test_data(results):
    """Сравнивает результаты с тестовыми данными."""
    
    # Тестовые данные из .cursor/rules/file-testing.mdc
    expected_headers = ["", "Торговое наименование препарата", "МНН или химическое (групповое) наименование", 
                       "Форма выпуска", "Цена ЖНВЛП (без НДС)", "Цена за упаковку (без НДС)", 
                       "Цена за упаковку (с НДС)", "Производитель"]
    
    expected_first_row = ["1", "Вакцина для профилактики дифтерии (с уменьшенным содержанием антигена), коклюша (с уменьшенным содержанием антигена, бесклеточная) и столбняка, адсорбированная",
                         "суспензия для внутримышечного введения, 0.5 мл/доза, 0.5 мл - флаконы (1) - пачки картонные",
                         "-", "2122,26", "2334,49", "Санофи Пастер Лимитед, Канада"]
    
    logger.info("\n🎯 Сравнение с тестовыми данными:")
    logger.info(f"📝 Ожидаемые заголовки: {len(expected_headers)} колонок")
    logger.info(f"📄 Ожидаемая первая строка: {len(expected_first_row)} колонок")
    
    best_score = 0
    best_scenario = None
    
    for result in results:
        if 'error' in result:
            logger.warning(f"⚠️ {result['name']}: ошибка - {result['error']}")
            continue
        
        logger.info(f"\n📊 Анализ сценария: {result['name']}")
        
        if not result['tables']:
            logger.warning("  ⚠️ Таблицы не найдены")
            continue
        
        # Ищем лучшую таблицу
        best_table_score = 0
        best_table_idx = -1
        
        for i, table in enumerate(result['tables']):
            if not table or len(table) == 0:
                continue
            
            # Проверяем заголовки
            headers = table[0] if len(table) > 0 and isinstance(table[0], list) else []
            header_score = calculate_match_score(headers, expected_headers)
            
            # Проверяем первую строку данных
            row_score = 0
            if len(table) > 1 and isinstance(table[1], list):
                first_row = table[1]
                row_score = calculate_match_score(first_row, expected_first_row)
            
            # Общий счет
            total_score = (header_score + row_score) / 2
            
            logger.info(f"  Таблица {i+1}: заголовки {header_score:.1%}, строка {row_score:.1%}, общий {total_score:.1%}")
            
            if total_score > best_table_score:
                best_table_score = total_score
                best_table_idx = i
        
        if best_table_score > best_score:
            best_score = best_table_score
            best_scenario = result['name']
        
        logger.info(f"  🏆 Лучшая таблица: {best_table_score:.1%}")
        
        if best_table_score >= 0.5:
            logger.info(f"  ✅ Цель достигнута: 50%+ поячечное совпадение!")
        else:
            logger.warning(f"  ⚠️ Цель не достигнута: {best_table_score:.1%} < 50%")
    
    logger.info(f"\n🎯 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    logger.info(f"🏆 Лучший сценарий: {best_scenario} ({best_score:.1%})")
    
    if best_score >= 0.5:
        logger.info("🎉 ЦЕЛЬ ДОСТИГНУТА: 50%+ поячечное совпадение!")
    else:
        logger.warning(f"⚠️ Цель не достигнута: {best_score:.1%} < 50%")

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

def save_results(results):
    """Сохраняет результаты в JSON файл."""
    output_path = "temp/ocr_comparison_results.json"
    
    # Подготавливаем данные для сохранения
    save_data = {
        'test_file': 'input/Нанолек 04.03.25.pdf',
        'timestamp': str(Path().cwd()),
        'scenarios': []
    }
    
    for result in results:
        scenario_data = {
            'name': result['name'],
            'pages_processed': result.get('pages_processed', 0),
            'tables_found': result.get('tables_found', 0),
            'text_length': result.get('text_length', 0),
            'error': result.get('error', None)
        }
        
        # Сохраняем только структуру таблиц, не весь контент
        if 'tables' in result:
            scenario_data['table_structures'] = []
            for table in result['tables']:
                if table:
                    scenario_data['table_structures'].append({
                        'rows': len(table),
                        'cols': len(table[0]) if table else 0,
                        'first_row_sample': table[0][:3] if table and len(table) > 0 else []
                    })
        
        save_data['scenarios'].append(scenario_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Результаты сохранены: {output_path}")

def main():
    """Главная функция."""
    logger.info("🚀 Запуск сравнения качества OCR")
    
    success = test_ocr_scenarios()
    
    if success:
        logger.info("✅ Сравнение OCR завершено успешно")
        logger.info("📁 Проверьте папку temp/ для промежуточных файлов и результатов")
    else:
        logger.error("❌ Сравнение OCR завершено с ошибками")
        sys.exit(1)

if __name__ == "__main__":
    main()
