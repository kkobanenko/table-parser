#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы PaddleOCR на тестовом файле.
"""

import sys
from pathlib import Path
sys.path.append('app')

from parsers.paddleocr_parser import PaddleOCRParser
from pdf2image import convert_from_path
import pandas as pd

def test_paddleocr():
    """Тестирует PaddleOCR на файле Нанолек 04.03.25.pdf"""
    
    print("🔬 Тестирование PaddleOCR PP-OCRv5")
    print("=" * 50)
    
    # Инициализируем парсер с русским языком
    parser = PaddleOCRParser(use_server_model=True, lang='ru')
    test_file = Path('input/Нанолек 04.03.25.pdf')
    
    if not test_file.exists():
        print(f"❌ Файл {test_file} не найден")
        return
    
    print(f"📄 Тестовый файл: {test_file.name}")
    
    # Конвертируем PDF в изображение
    print("🔄 Конвертируем PDF в изображение...")
    images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=1)
    
    if not images:
        print("❌ Не удалось конвертировать PDF в изображение")
        return
    
    temp_image = Path('/tmp/test_paddleocr.png')
    images[0].save(temp_image)
    print(f"✅ PDF конвертирован в изображение: {temp_image}")
    
    # Тестируем парсер
    print("🔄 Обрабатываем изображение через PaddleOCR...")
    tables = parser.extract_tables(str(temp_image))
    
    print(f"📊 Найдено таблиц: {len(tables)}")
    
    if tables:
        for i, table in enumerate(tables):
            print(f"\n📋 Таблица {i+1}:")
            print(f"   Размер: {table['data'].shape}")
            print(f"   Источник: {table['source']}")
            
            # Показываем первые несколько строк
            df = table['data']
            print(f"\n   Первые 5 строк:")
            print(df.head().to_string())
            
            # Проверяем наличие ожидаемых данных
            print(f"\n   Анализ содержимого:")
            
            # Ищем заголовки таблицы
            expected_headers = [
                "Торговое наименование препарата",
                "МНН или химическое (групповое) наименование", 
                "Форма выпуска",
                "Цена ЖНВЛП (без НДС)",
                "Цена за упаковку (без НДС)",
                "Цена за упаковку (с НДС)",
                "Производитель"
            ]
            
            found_headers = 0
            for header in expected_headers:
                # Ищем заголовок в данных
                for col in df.columns:
                    if header.lower() in str(df[col].iloc[0]).lower():
                        found_headers += 1
                        print(f"   ✅ Найден заголовок: {header}")
                        break
            
            print(f"   📊 Найдено заголовков: {found_headers}/{len(expected_headers)}")
            
            # Ищем первую строку данных
            expected_first_row = [
                "1",
                "Вакцина для профилактики дифтерии",
                "суспензия для внутримышечного введения",
                "-",
                "2122,26",
                "2334,49",
                "Санофи Пастер Лимитед"
            ]
            
            found_data = 0
            for expected_item in expected_first_row:
                for col in df.columns:
                    if expected_item.lower() in str(df[col].iloc[1]).lower():
                        found_data += 1
                        print(f"   ✅ Найдены данные: {expected_item}")
                        break
            
            print(f"   📊 Найдено элементов данных: {found_data}/{len(expected_first_row)}")
            
            # Вычисляем общий процент совпадения
            total_expected = len(expected_headers) + len(expected_first_row)
            total_found = found_headers + found_data
            match_percentage = (total_found / total_expected) * 100
            
            print(f"   🎯 Общий процент совпадения: {match_percentage:.1f}%")
            
            if match_percentage >= 50:
                print(f"   🎉 ТЕСТ ПРОЙДЕН! Достигнуто {match_percentage:.1f}% совпадения")
            else:
                print(f"   ❌ ТЕСТ НЕ ПРОЙДЕН. Требуется минимум 50% совпадения")
    
    # Удаляем временный файл
    temp_image.unlink(missing_ok=True)
    print(f"\n✅ Тестирование завершено")

if __name__ == "__main__":
    test_paddleocr()
