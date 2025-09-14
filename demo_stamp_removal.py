#!/usr/bin/env python3
"""
Демо скрипт для тестирования удаления оттисков печатей.

Этот скрипт демонстрирует работу этапа удаления печатей в пайплайне:
- Цветовая сегментация в HSV пространстве
- Детекция синих и красных печатей
- Морфологические операции
- Замена печатей белым цветом или интерполяцией
"""

import sys
import os
from pathlib import Path
import logging
import cv2
import numpy as np
from PIL import Image

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "app"))

from pipeline.document_pipeline import DocumentPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stamp_removal():
    """Тестирует удаление оттисков печатей."""
    
    # Путь к тестовому файлу
    test_file = Path("input/Нанолек 04.03.25.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Тестовый файл не найден: {test_file}")
        return False
    
    logger.info(f"🔴 Тестируем удаление оттисков печатей на файле: {test_file}")
    
    try:
        # Инициализация пайплайна
        pipeline = DocumentPipeline()
        
        # Конфигурация пайплайна с удалением печатей
        config = {
            # Предобработка с удалением печатей
            'enable_preprocessing': True,
            'enable_stamp_removal': True,
            'enable_deskew': False,  # Отключаем для чистоты теста
            'enable_denoise': False,
            'enable_binarize': False,
            
            # Настройки удаления печатей
            'stamp_removal_method': 'white_replacement',
            'blue_hsv_thresholds': {
                'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50
            },
            'red_hsv_thresholds': {
                'h_min': 0, 'h_max': 10, 'h_min2': 170, 'h_max2': 180, 's_min': 50, 'v_min': 50
            },
            'morphology_kernel_size': 5,
            
            # Остальные компоненты отключены для чистоты теста
            'enable_layout_detection': False,
            'enable_ocr': False,
            'enable_table_detection': False,
            'enable_export': False
        }
        
        # Конфигурируем пайплайн
        pipeline.configure_pipeline(config)
        
        # Инициализируем компоненты
        pipeline.initialize_components()
        
        logger.info("✅ Пайплайн инициализирован")
        
        # Конвертируем PDF в изображения
        images = pipeline._convert_pdf_to_images(str(test_file))
        logger.info(f"📄 Конвертировано {len(images)} страниц")
        
        # Обрабатываем каждую страницу
        for i, image in enumerate(images):
            logger.info(f"🔄 Обрабатываем страницу {i + 1}")
            
            # Конвертируем PIL Image в numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Применяем удаление печатей
            processed_img = pipeline._remove_stamps(img_array)
            
            # Сохраняем результаты для визуального сравнения
            original_path = f"output/stamp_test_page_{i+1}_original.png"
            processed_path = f"output/stamp_test_page_{i+1}_processed.png"
            
            # Создаем папку output если её нет
            Path("output").mkdir(exist_ok=True)
            
            # Сохраняем оригинальное изображение
            cv2.imwrite(original_path, img_array)
            
            # Сохраняем обработанное изображение
            cv2.imwrite(processed_path, processed_img)
            
            logger.info(f"💾 Сохранено: {original_path}, {processed_path}")
            
            # Анализируем изменения
            analyze_stamp_removal(img_array, processed_img, i + 1)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования удаления печатей: {e}")
        return False

def analyze_stamp_removal(original: np.ndarray, processed: np.ndarray, page_num: int):
    """Анализирует результаты удаления печатей."""
    
    # Вычисляем разность изображений
    diff = cv2.absdiff(original, processed)
    
    # Конвертируем в grayscale для анализа
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Подсчитываем количество измененных пикселей
    changed_pixels = np.sum(diff_gray > 0)
    total_pixels = diff_gray.shape[0] * diff_gray.shape[1]
    change_percentage = (changed_pixels / total_pixels) * 100
    
    logger.info(f"📊 Страница {page_num}:")
    logger.info(f"  🔄 Изменено пикселей: {changed_pixels:,} ({change_percentage:.2f}%)")
    
    # Анализируем цветовые каналы
    for i, color in enumerate(['Blue', 'Green', 'Red']):
        channel_diff = np.sum(diff[:, :, i] > 0)
        channel_percentage = (channel_diff / total_pixels) * 100
        logger.info(f"  {color}: {channel_diff:,} пикселей ({channel_percentage:.2f}%)")
    
    # Сохраняем разность для визуального анализа
    diff_path = f"output/stamp_test_page_{page_num}_difference.png"
    cv2.imwrite(diff_path, diff)
    logger.info(f"  💾 Разность сохранена: {diff_path}")

def test_hsv_thresholds():
    """Тестирует различные HSV пороги для печатей."""
    
    logger.info("🎨 Тестируем различные HSV пороги")
    
    # Создаем тестовое изображение с цветными областями
    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Синие области
    test_img[50:100, 50:150] = [255, 0, 0]  # Синий
    test_img[50:100, 200:300] = [200, 50, 0]  # Синеватый
    
    # Красные области
    test_img[150:200, 50:150] = [0, 0, 255]  # Красный
    test_img[150:200, 200:300] = [0, 50, 200]  # Красноватый
    
    # Черный текст
    test_img[250:280, 50:350] = [0, 0, 0]  # Черный
    
    # Сохраняем тестовое изображение
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite("output/stamp_test_hsv_test_image.png", test_img)
    
    # Тестируем различные пороги
    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    
    # Синие пороги
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Красные пороги
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Объединяем маски
    combined_mask = cv2.bitwise_or(blue_mask, red_mask)
    
    # Сохраняем маски
    cv2.imwrite("output/stamp_test_hsv_blue_mask.png", blue_mask)
    cv2.imwrite("output/stamp_test_hsv_red_mask.png", red_mask)
    cv2.imwrite("output/stamp_test_hsv_combined_mask.png", combined_mask)
    
    logger.info("🎨 HSV тест завершен, маски сохранены в output/")

def main():
    """Главная функция."""
    logger.info("🚀 Запуск тестирования удаления оттисков печатей")
    
    # Создаем папку output
    Path("output").mkdir(exist_ok=True)
    
    # Тестируем HSV пороги
    test_hsv_thresholds()
    
    # Тестируем удаление печатей
    success = test_stamp_removal()
    
    if success:
        logger.info("✅ Тестирование удаления печатей завершено успешно")
        logger.info("📁 Проверьте папку output/ для визуального анализа результатов")
    else:
        logger.error("❌ Тестирование удаления печатей завершено с ошибками")
        sys.exit(1)

if __name__ == "__main__":
    main()
