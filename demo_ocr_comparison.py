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
    
    # Создаем папку output для промежуточных файлов
    Path("output").mkdir(exist_ok=True)
    
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
                'enable_binarize': True,  # Включаем бинаризацию для тестирования
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
            'name': 'Только бинаризация',
            'config': {
                'enable_preprocessing': True,
                'enable_stamp_removal': False,
                'enable_deskew': False,
                'enable_denoise': False,
                'enable_binarize': True,  # Только бинаризация
                'enable_layout_detection': False,
                'enable_ocr': True,
                'ocr_method': 'paddleocr',
                'enable_table_detection': True,
                'enable_export': False
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
                    if not table:
                        continue
                    rows, cols = 0, 0
                    try:
                        # Случай: словарь с ключом 'data' (pandas DataFrame или list[list])
                        if isinstance(table, dict):
                            data_obj = table.get('data')
                            if hasattr(data_obj, 'values') and not callable(getattr(data_obj, 'values', None)):
                                # pandas DataFrame
                                rows, cols = data_obj.shape
                            elif isinstance(data_obj, list):
                                rows = len(data_obj)
                                cols = len(data_obj[0]) if rows > 0 and isinstance(data_obj[0], list) else 0
                            else:
                                # Попытка универсальной оценки
                                rows = len(data_obj) if data_obj is not None else 0
                                cols = 0
                        # Случай: уже list[list]
                        elif isinstance(table, list):
                            rows = len(table)
                            cols = len(table[0]) if rows > 0 and isinstance(table[0], list) else 0
                        # Случай: pandas DataFrame напрямую
                        elif hasattr(table, 'values') and not callable(getattr(table, 'values', None)):
                            rows, cols = table.shape
                    except Exception:
                        rows, cols = 0, 0
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
    """Новое правило: страница 2, сравниваем первые две строки таблицы построчно.
    Итог записываем в ./temp/ocr_rule_page2_report.txt и логируем.
    """
    expected_row1 = [
        '13','Нейпомакс','Филграстим',
        'раствор для внутривенного и подкожного введения, 30 млн.МЕ/мл, 1.6 мл - флаконы (5) - упаковки ячейковые контурные (1) - пачки картонные',
        '5848,78','5848,78','6433,66',
        'Вл.Вып.к.Перв.Уп.Втор.УП.ПР.Открытое акционерное общество "Фармстандарт-Уфимский витаминный завод" (ОАО "Фармстандарт-УфаВИТА"), Россия (0274036993);'
    ]
    expected_row2 = [
        '14',
        'Пентаксим (вакцина для профилактики дифтерии и столбняка адсорбированная, коклюша ацеллюлярная, полиомиелита инактивированная и инфекции, вызываемой Haemophilus influenzae тип b конъюгированная)',
        'Вакцина для профилактики дифтерии, коклюша, полиомиелита, столбняка и инфекций, вызываемых Haemophilus influenzae типа b',
        'лиофилизат для приготовления суспензии для внутримышечного введения, 1 доза, 1 шт. - флакон (1) / в комплекте с суспензией для внутримышечного введения (шприц) 0.5 мл-1 шт., игла-1 шт. / - пачка картонная',
        '1363,00','1363,00','1499,30',
        'Санофи Пастер С.А. - Франция;Пр.,Перв.Уп.-Санофи Пастер С.А. Франция;Втор.Уп.,Вып.к.-ООО "Нанолек"- Россия'
    ]

    def norm(x):
        return str(x).strip().lower().replace('\xa0', ' ')

    def cell_match(a, b):
        a1, b1 = norm(a), norm(b)
        if not a1 or not b1:
            return False
        return a1 == b1 or a1 in b1 or b1 in a1

    def to_table_data(table):
        try:
            # Проверяем на pandas DataFrame
            if hasattr(table, 'values') and not callable(table.values) and hasattr(table, 'empty'):
                return table.values.tolist()
            if isinstance(table, list):
                return table
            if isinstance(table, dict):
                if 'data' in table:
                    return table['data']
                if 'rows' in table:
                    return table['rows']
                vals = list(table.values())
                if all(isinstance(v, list) for v in vals) and len({len(v) for v in vals}) == 1:
                    return [list(row) for row in zip(*vals)]
                return vals
        except Exception:
            return None
        return None

    lines = []
    Path('output').mkdir(exist_ok=True)

    for result in results:
        if 'error' in result:
            lines.append(f"Сценарий: {result['name']} | ошибка: {result['error']}")
            continue
        lines.append(f"Сценарий: {result['name']}")
        if not result.get('tables'):
            lines.append("  Таблицы не найдены")
            continue

        best = {'score': 0.0, 'row1_idx': -1, 'row2_idx': -1}

        for table in result['tables']:
            data = to_table_data(table)
            if data is None or not isinstance(data, list) or len(data) == 0:
                continue
            for i in range(len(data) - 1):
                r1 = data[i] if isinstance(data[i], list) else []
                r2 = data[i+1] if isinstance(data[i+1], list) else []
                L1 = min(len(r1), len(expected_row1))
                L2 = min(len(r2), len(expected_row2))
                if L1 == 0 or L2 == 0:
                    continue
                m1 = [cell_match(r1[j], expected_row1[j]) for j in range(L1)]
                m2 = [cell_match(r2[j], expected_row2[j]) for j in range(L2)]
                score = (sum(m1)/L1 + sum(m2)/L2) / 2 * 100.0
                if score > best['score']:
                    best = {
                        'score': score,
                        'row1_idx': i,
                        'row2_idx': i+1,
                        'r1': r1,
                        'r2': r2,
                        'm1': m1,
                        'm2': m2
                    }

        if best['row1_idx'] == -1:
            lines.append("  Подходящие строки не найдены")
            continue

        # Выводим в консоль сравниваемые строки
        print(f"\nСценарий: {result['name']}")
        print(f"  Совпадение: {best['score']:.1f}%")
        lines.append(f"  Совпадение: {best['score']:.1f}%")
        lines.append("  Строка 1 (exp vs got):")
        L1 = min(len(best['r1']), len(expected_row1))
        for j in range(L1):
            msg = f"    [{j}] exp='{expected_row1[j]}' | got='{best['r1'][j]}' | match={best['m1'][j]}"
            print(msg)
            lines.append(msg)
        lines.append("  Строка 2 (exp vs got):")
        L2 = min(len(best['r2']), len(expected_row2))
        for j in range(L2):
            msg = f"    [{j}] exp='{expected_row2[j]}' | got='{best['r2'][j]}' | match={best['m2'][j]}"
            print(msg)
            lines.append(msg)

    report = Path('output/ocr_rule_page2_report.txt')
    report.write_text("\n".join(lines), encoding='utf-8')
    logger.info(f"💾 Отчет сохранен: {report}")

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
    output_path = "output/ocr_comparison_results.json"
    
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
                if not table:
                    continue
                # Приводим таблицу к формату list[list[str]]
                table_data = None
                try:
                    if hasattr(table, 'values') and not callable(table.values):
                        # pandas DataFrame
                        table_data = table.values.tolist()
                    elif isinstance(table, list):
                        table_data = table
                    elif isinstance(table, dict):
                        if 'data' in table:
                            table_data = table['data']
                        elif 'rows' in table:
                            table_data = table['rows']
                        else:
                            # возможен словарь колонок -> списков
                            vals = list(table.values())
                            # если это столбцы одинаковой длины, транспонируем в строки
                            if all(isinstance(v, list) for v in vals) and len({len(v) for v in vals}) == 1:
                                table_data = [list(row) for row in zip(*vals)]
                            else:
                                table_data = vals
                except Exception:
                    table_data = None

                if not isinstance(table_data, list):
                    continue

                rows_count = len(table_data)
                cols_count = len(table_data[0]) if rows_count > 0 and isinstance(table_data[0], list) else 0
                first_row_sample = table_data[0][:3] if rows_count > 0 and isinstance(table_data[0], list) else []

                scenario_data['table_structures'].append({
                    'rows': rows_count,
                    'cols': cols_count,
                    'first_row_sample': first_row_sample
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
