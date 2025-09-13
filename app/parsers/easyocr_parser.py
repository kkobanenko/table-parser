"""
Парсер таблиц с использованием EasyOCR для получения точных координат текста
и анализа структуры таблицы на основе расположения слов.
"""

import cv2
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from utils.logger import setup_logger
from utils.image_cleaner import ImageCleaner

logger = setup_logger('easyocr_parser')


class EasyOCRParser:
    """
    Парсер таблиц на основе EasyOCR с анализом структуры по координатам текста.
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Инициализация EasyOCR парсера.
        
        Args:
            languages: Список языков для распознавания (по умолчанию ['en', 'ru'])
            gpu: Использовать GPU для ускорения (по умолчанию False)
        """
        self.languages = languages or ['en', 'ru']
        self.gpu = gpu
        self.image_cleaner = ImageCleaner()
        
        # Инициализируем EasyOCR reader
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("✅ EasyOCR инициализирован для языков: %s", self.languages)
        except Exception as e:
            logger.error("❌ Ошибка инициализации EasyOCR: %s", e)
            raise
    
    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из изображения с использованием EasyOCR.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с данными таблиц
        """
        try:
            # Читаем изображение
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            
            # Применяем очистку изображения
            cleaned_img = self._clean_image(img)
            
            # Сохраняем очищенное изображение в temp
            temp_path = f"temp/{Path(image_path).stem}_cleaned.png"
            cv2.imwrite(temp_path, cleaned_img)
            logger.info("💾 Очищенное изображение сохранено: %s", temp_path)
            
            # Получаем текст с координатами
            results = self.reader.readtext(cleaned_img, detail=1)
            
            if not results:
                logger.warning("⚠️ EasyOCR не нашел текст на изображении")
                return []
            
            # Анализируем структуру таблицы
            tables = self._analyze_table_structure(results, img.shape)
            
            logger.info("✅ EasyOCR извлек %d таблиц", len(tables))
            return tables
            
        except Exception as e:
            logger.error("❌ Ошибка EasyOCR парсинга: %s", e)
            return []
    
    def _clean_image(self, img: np.ndarray) -> np.ndarray:
        """
        Очищает изображение от помех для улучшения OCR.
        
        Args:
            img: Исходное изображение
            
        Returns:
            Очищенное изображение
        """
        # Конвертируем в серый
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Применяем CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Убираем шум
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Бинаризация для четкости текста
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _analyze_table_structure(self, ocr_results: List, img_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Анализирует структуру таблицы на основе координат распознанного текста.
        
        Args:
            ocr_results: Результаты EasyOCR с координатами
            img_shape: Размеры изображения (height, width)
            
        Returns:
            Список таблиц
        """
        if not ocr_results:
            return []
        
        # Конвертируем результаты в DataFrame для анализа
        data = []
        for result in ocr_results:
            bbox, text, confidence = result
            if confidence < 0.5:  # Фильтруем низкокачественные результаты
                continue
                
            # Вычисляем центр текста
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Вычисляем размеры текста
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            data.append({
                'text': text.strip(),
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'confidence': confidence,
                'bbox': bbox
            })
        
        if not data:
            return []
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return []
        
        # Группируем текст в строки и колонки
        table_data = self._group_text_into_table(df)
        
        if not table_data or len(table_data) < 2:
            logger.warning("⚠️ Не удалось определить структуру таблицы")
            return []
        
        # Создаем DataFrame таблицы
        # Определяем максимальное количество колонок
        max_cols = max(len(row) for row in table_data) if table_data else 0
        
        # Дополняем все строки до максимального количества колонок
        normalized_data = []
        for row in table_data:
            normalized_row = row + [''] * (max_cols - len(row))
            normalized_data.append(normalized_row)
        
        # Создаем DataFrame без заголовков
        table_df = pd.DataFrame(normalized_data)
        
        return [{
            'data': table_df,
            'sheet_name': 'easyocr_table',
            'source': 'easyocr',
            'cleaning_method': 'clahe_denoise_binary'
        }]
    
    def _group_text_into_table(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Группирует распознанный текст в структуру таблицы.
        
        Args:
            df: DataFrame с координатами текста
            
        Returns:
            Список строк таблицы
        """
        # Сортируем по Y-координате (сверху вниз)
        df_sorted = df.sort_values('center_y')
        
        # Определяем строки на основе кластеризации по Y-координате
        rows = self._cluster_rows(df_sorted)
        
        # Для каждой строки определяем колонки
        table_data = []
        for row_texts in rows:
            if row_texts.empty:
                continue
                
            # Сортируем тексты в строке по X-координате (слева направо)
            row_texts_sorted = row_texts.sort_values('center_x')
            
            # Группируем в колонки
            columns = self._cluster_columns(row_texts_sorted)
            
            # Объединяем тексты в колонках
            row_data = []
            for col_texts in columns:
                col_text = ' '.join(col_texts['text'].tolist())
                row_data.append(col_text)
            
            table_data.append(row_data)
        
        return table_data
    
    def _cluster_rows(self, df: pd.DataFrame, threshold: float = 20.0) -> List[pd.DataFrame]:
        """
        Кластеризует тексты по строкам на основе Y-координат.
        
        Args:
            df: DataFrame с координатами текста
            threshold: Порог для группировки строк
            
        Returns:
            Список DataFrame для каждой строки
        """
        if df.empty:
            return []
        
        # Простая кластеризация по Y-координате
        rows = []
        current_row = []
        last_y = None
        
        for _, row in df.iterrows():
            if last_y is None or abs(row['center_y'] - last_y) <= threshold:
                current_row.append(row)
                last_y = row['center_y']
            else:
                if current_row:
                    rows.append(pd.DataFrame(current_row))
                current_row = [row]
                last_y = row['center_y']
        
        if current_row:
            rows.append(pd.DataFrame(current_row))
        
        return rows
    
    def _cluster_columns(self, df: pd.DataFrame, threshold: float = 50.0) -> List[pd.DataFrame]:
        """
        Кластеризует тексты по колонкам на основе X-координат.
        
        Args:
            df: DataFrame с координатами текста
            threshold: Порог для группировки колонок
            
        Returns:
            Список DataFrame для каждой колонки
        """
        if df.empty:
            return []
        
        # Простая кластеризация по X-координате
        columns = []
        current_col = []
        last_x = None
        
        for _, row in df.iterrows():
            if last_x is None or abs(row['center_x'] - last_x) <= threshold:
                current_col.append(row)
                last_x = row['center_x']
            else:
                if current_col:
                    columns.append(pd.DataFrame(current_col))
                current_col = [row]
                last_x = row['center_x']
        
        if current_col:
            columns.append(pd.DataFrame(current_col))
        
        return columns
