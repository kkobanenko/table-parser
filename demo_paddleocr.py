#!/usr/bin/env python3
"""
Демонстрация работы улучшенного PaddleOCR парсера.
"""

import sys
from pathlib import Path
sys.path.append('app')

from parsers.paddleocr_parser import PaddleOCRParser
from pdf2image import convert_from_path
import pandas as pd

def demo_paddleocr():
    """Демонстрирует работу PaddleOCR на файле Нанолек 04.03.25.pdf"""
    
    print("🚀 ДЕМОНСТРАЦИЯ PADDLEOCR PP-OCRV5")
    print("=" * 60)
    
    # Инициализируем парсер с русским языком
    print("🔄 Инициализация PaddleOCR с русской моделью...")
    parser = PaddleOCRParser(use_server_model=True, lang='ru')
    
    test_file = Path('input/Нанолек 04.03.25.pdf')
    
    if not test_file.exists():
        print(f"❌ Файл {test_file} не найден")
        return
    
    print(f"📄 Обрабатываем файл: {test_file.name}")
    
    # Конвертируем PDF в изображение с высоким DPI
    print("🔄 Конвертируем PDF в изображение (DPI=300)...")
    images = convert_from_path(str(test_file), dpi=300, first_page=1, last_page=1)
    
    if not images:
        print("❌ Не удалось конвертировать PDF в изображение")
        return
    
    temp_image = Path('/tmp/demo_paddleocr.png')
    images[0].save(temp_image)
    print(f"✅ PDF конвертирован в изображение: {temp_image}")
    
    # Обрабатываем через PaddleOCR
    print("🔄 Обрабатываем изображение через PaddleOCR PP-OCRv5...")
    print("⏳ Это может занять 1-2 минуты...")
    
    tables = parser.extract_tables(str(temp_image))
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
    print(f"   Найдено таблиц: {len(tables)}")
    
    if tables:
        for i, table in enumerate(tables):
            df = table['data']
            print(f"\n📋 ТАБЛИЦА {i+1}:")
            print(f"   Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
            print(f"   Источник: {table['source']}")
            
            # Показываем первые строки
            print(f"\n📝 ПЕРВЫЕ 10 СТРОК ТАБЛИЦЫ:")
            print(df.head(10).to_string())
            
            # Анализируем содержимое
            print(f"\n🔍 АНАЛИЗ СОДЕРЖИМОГО:")
            
            # Ищем ожидаемые данные
            expected_data = [
                "Торговое наименование препарата",
                "МНН или химическое наименование", 
                "Форма выпуска",
                "Цена ЖНВЛП",
                "Цена за упаковку",
                "Производитель",
                "Вакцина для профилактики",
                "дифтерии",
                "коклюша", 
                "столбняка",
                "суспензия",
                "внутримышечного введения",
                "2122,26",
                "2334,49",
                "Санофи Пастер"
            ]
            
            found_items = 0
            total_items = len(expected_data)
            
            for expected_item in expected_data:
                found = False
                for col in df.columns:
                    for row_idx in range(min(5, len(df))):  # Проверяем первые 5 строк
                        cell_text = str(df.iloc[row_idx, col]).lower()
                        if expected_item.lower() in cell_text:
                            found = True
                            found_items += 1
                            print(f"   ✅ Найдено: '{expected_item}' в ячейке [{row_idx}, {col}]")
                            break
                    if found:
                        break
            
            print(f"\n📊 СТАТИСТИКА РАСПОЗНАВАНИЯ:")
            print(f"   Найдено элементов: {found_items}/{total_items}")
            match_percentage = (found_items / total_items) * 100
            print(f"   Процент совпадения: {match_percentage:.1f}%")
            
            if match_percentage >= 50:
                print(f"   🎉 ЦЕЛЬ ДОСТИГНУТА! Процент совпадения ≥ 50%")
            else:
                print(f"   📈 Требуется улучшение для достижения 50%")
            
            # Показываем все уникальные значения для анализа
            print(f"\n📋 ВСЕ УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В ТАБЛИЦЕ:")
            all_values = set()
            for col in df.columns:
                for val in df[col].dropna():
                    if str(val).strip():
                        all_values.add(str(val).strip())
            
            for val in sorted(all_values)[:20]:  # Показываем первые 20
                print(f"   • {val}")
            
            if len(all_values) > 20:
                print(f"   ... и еще {len(all_values) - 20} значений")
    
    # Удаляем временный файл
    temp_image.unlink(missing_ok=True)
    
    print(f"\n✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)

if __name__ == "__main__":
    demo_paddleocr()
