#!/usr/bin/env python3
"""
Тестирование PaddleOCR PP-OCRv5 на всех файлах из папки input.
"""

import sys
from pathlib import Path
sys.path.append('app')

from parsers.paddleocr_parser import PaddleOCRParser
from pdf2image import convert_from_path
import pandas as pd

def test_paddleocr_on_file(file_path: Path):
    """Тестирует PaddleOCR на одном файле"""
    
    print(f"\n🚀 ТЕСТИРОВАНИЕ PADDLEOCR PP-OCRV5")
    print(f"📄 Файл: {file_path.name}")
    print("=" * 60)
    
    # Инициализируем парсер с русским языком
    parser = PaddleOCRParser(use_server_model=True, lang='ru')
    
    if not file_path.exists():
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем тип файла
    if file_path.suffix.lower() == '.pdf':
        # Конвертируем PDF в изображение с высоким DPI
        print("🔄 Конвертируем PDF в изображение (DPI=300)...")
        try:
            images = convert_from_path(str(file_path), dpi=300, first_page=1, last_page=1)
            
            if not images:
                print("❌ Не удалось конвертировать PDF в изображение")
                return False
            
            temp_image = Path('/tmp/test_paddleocr.png')
            images[0].save(temp_image)
            print(f"✅ PDF конвертирован в изображение: {temp_image}")
            
            # Обрабатываем через PaddleOCR
            print("🔄 Обрабатываем изображение через PaddleOCR PP-OCRv5...")
            print("⏳ Это может занять 1-2 минуты...")
            
            tables = parser.extract_tables(str(temp_image))
            
            # Удаляем временный файл
            temp_image.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ Ошибка обработки PDF: {e}")
            return False
            
    elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        # Прямая обработка изображения
        print("🔄 Обрабатываем изображение через PaddleOCR PP-OCRv5...")
        print("⏳ Это может занять 1-2 минуты...")
        
        try:
            tables = parser.extract_tables(str(file_path))
        except Exception as e:
            print(f"❌ Ошибка обработки изображения: {e}")
            return False
    else:
        print(f"⚠️ Неподдерживаемый тип файла: {file_path.suffix}")
        return False
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
    print(f"   Найдено таблиц: {len(tables)}")
    
    if tables:
        for i, table in enumerate(tables):
            df = table['data']
            print(f"\n📋 ТАБЛИЦА {i+1}:")
            print(f"   Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
            print(f"   Источник: {table['source']}")
            
            # Показываем первые строки
            print(f"\n📝 ПЕРВЫЕ 5 СТРОК ТАБЛИЦЫ:")
            print(df.head(5).to_string())
            
            # Анализируем содержимое
            print(f"\n🔍 АНАЛИЗ СОДЕРЖИМОГО:")
            
            # Показываем все уникальные значения для анализа
            print(f"\n📋 ВСЕ УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В ТАБЛИЦЕ:")
            all_values = set()
            for col in df.columns:
                for val in df[col].dropna():
                    if str(val).strip():
                        all_values.add(str(val).strip())
            
            for val in sorted(all_values)[:10]:  # Показываем первые 10
                print(f"   • {val}")
            
            if len(all_values) > 10:
                print(f"   ... и еще {len(all_values) - 10} значений")
        
        return True
    else:
        print("❌ Таблицы не найдены")
        return False

def main():
    """Основная функция для тестирования всех файлов"""
    
    input_dir = Path('input')
    
    if not input_dir.exists():
        print(f"❌ Папка {input_dir} не найдена")
        return
    
    print("🔬 ТЕСТИРОВАНИЕ PADDLEOCR PP-OCRV5 НА ВСЕХ ФАЙЛАХ")
    print("=" * 80)
    
    # Получаем все файлы из папки input
    files = list(input_dir.iterdir())
    files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']]
    
    if not files:
        print("❌ В папке input нет поддерживаемых файлов")
        return
    
    print(f"📁 Найдено {len(files)} файлов для тестирования:")
    for file in files:
        print(f"   • {file.name}")
    
    # Тестируем каждый файл
    successful_tests = 0
    total_tests = len(files)
    
    for file_path in files:
        try:
            success = test_paddleocr_on_file(file_path)
            if success:
                successful_tests += 1
                print(f"\n✅ ТЕСТ ПРОЙДЕН: {file_path.name}")
            else:
                print(f"\n❌ ТЕСТ НЕ ПРОЙДЕН: {file_path.name}")
        except Exception as e:
            print(f"\n❌ ОШИБКА ТЕСТА {file_path.name}: {e}")
    
    # Итоговая статистика
    print(f"\n🏁 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Всего файлов: {total_tests}")
    print(f"   Успешных тестов: {successful_tests}")
    print(f"   Неудачных тестов: {total_tests - successful_tests}")
    print(f"   Процент успеха: {(successful_tests / total_tests) * 100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    elif successful_tests > 0:
        print("📈 ЧАСТИЧНЫЙ УСПЕХ - некоторые тесты прошли")
    else:
        print("❌ ВСЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")

if __name__ == "__main__":
    main()
