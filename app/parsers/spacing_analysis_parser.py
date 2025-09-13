"""
Парсер таблиц на основе анализа пробелов и выравнивания текста.
Использует OCR для получения текста и анализирует его расположение для определения структуры таблицы.
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from utils.logger import setup_logger
from utils.image_cleaner import ImageCleaner

logger = setup_logger('spacing_analysis_parser')


class SpacingAnalysisParser:
    """
    Парсер таблиц на основе анализа пробелов и выравнивания текста.
    """
    
    def __init__(self, ocr_psm: int = 6, ocr_lang: str = 'eng+rus'):
        """
        Инициализация парсера.
        
        Args:
            ocr_psm: PSM режим для Tesseract
            ocr_lang: Языки для OCR
        """
        self.ocr_psm = ocr_psm
        self.ocr_lang = ocr_lang
        self.image_cleaner = ImageCleaner()
    
    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из изображения на основе анализа пробелов.
        
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
            temp_path = f"temp/{Path(image_path).stem}_spacing_cleaned.png"
            cv2.imwrite(temp_path, cleaned_img)
            logger.info("💾 Очищенное изображение для анализа пробелов сохранено: %s", temp_path)
            
            # Получаем детальную информацию о тексте
            text_data = self._get_detailed_text_data(cleaned_img)
            
            if text_data is None or text_data.empty:
                logger.warning("⚠️ Не удалось получить данные о тексте")
                return []
            
            # Анализируем структуру таблицы
            tables = self._analyze_spacing_structure(text_data, img.shape)
            
            logger.info("✅ SpacingAnalysis извлек %d таблиц", len(tables))
            return tables
            
        except Exception as e:
            logger.error("❌ Ошибка SpacingAnalysis парсинга: %s", e)
            return []
    
    def _clean_image(self, img: np.ndarray) -> np.ndarray:
        """
        Очищает изображение для улучшения OCR.
        
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
        
        # Бинаризация
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _get_detailed_text_data(self, img: np.ndarray) -> pd.DataFrame:
        """
        Получает детальную информацию о тексте с координатами.
        
        Args:
            img: Обработанное изображение
            
        Returns:
            DataFrame с данными о тексте
        """
        try:
            # Получаем детальные данные от Tesseract
            data = pytesseract.image_to_data(
                img, 
                config=f'--oem 3 --psm {self.ocr_psm}',
                lang=self.ocr_lang,
                output_type=pytesseract.Output.DATAFRAME
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Фильтруем пустые строки
            data = data.dropna(subset=['text'])
            data['text'] = data['text'].astype(str)
            data = data[data['text'].str.strip() != '']
            
            # Фильтруем по уверенности
            data = data[data['conf'] > 30]  # Минимальная уверенность 30%
            
            return data
            
        except Exception as e:
            logger.error("❌ Ошибка получения данных текста: %s", e)
            return pd.DataFrame()
    
    def _analyze_spacing_structure(self, text_data: pd.DataFrame, img_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Анализирует структуру таблицы на основе пробелов и выравнивания.
        
        Args:
            text_data: DataFrame с данными о тексте
            img_shape: Размеры изображения
            
        Returns:
            Список таблиц
        """
        if text_data is None or text_data.empty:
            return []
        
        # Группируем текст в строки
        rows = self._group_into_rows(text_data)
        
        if len(rows) < 2:
            logger.warning("⚠️ Недостаточно строк для таблицы")
            return []
        
        # Анализируем колонки на основе выравнивания
        table_data = self._analyze_columns(rows)
        
        if not table_data or len(table_data) < 2:
            logger.warning("⚠️ Не удалось определить колонки таблицы")
            return []
        
        # Создаем DataFrame таблицы
        table_df = pd.DataFrame(table_data[1:], columns=table_data[0])
        
        return [{
            'data': table_df,
            'sheet_name': 'spacing_analysis_table',
            'source': 'spacing_analysis',
            'cleaning_method': 'clahe_denoise_binary'
        }]
    
    def _group_into_rows(self, text_data: pd.DataFrame, threshold: float = 15.0) -> List[pd.DataFrame]:
        """
        Группирует текст в строки на основе Y-координат.
        
        Args:
            text_data: DataFrame с данными о тексте
            threshold: Порог для группировки строк
            
        Returns:
            Список DataFrame для каждой строки
        """
        # Сортируем по Y-координате
        sorted_data = text_data.sort_values('top')
        
        rows = []
        current_row = []
        last_y = None
        
        for _, row in sorted_data.iterrows():
            if last_y is None or abs(row['top'] - last_y) <= threshold:
                current_row.append(row)
                last_y = row['top']
            else:
                if current_row:
                    rows.append(pd.DataFrame(current_row))
                current_row = [row]
                last_y = row['top']
        
        if current_row:
            rows.append(pd.DataFrame(current_row))
        
        return rows
    
    def _analyze_columns(self, rows: List[pd.DataFrame]) -> List[List[str]]:
        """
        Анализирует колонки на основе выравнивания текста.
        
        Args:
            rows: Список строк таблицы
            
        Returns:
            Список строк таблицы с данными колонок
        """
        if not rows:
            return []
        
        # Определяем позиции колонок на основе анализа всех строк
        column_positions = self._detect_column_positions(rows)
        
        if not column_positions:
            logger.warning("⚠️ Не удалось определить позиции колонок")
            return []
        
        # Создаем таблицу на основе определенных позиций
        table_data = []
        
        for row_df in rows:
            row_data = []
            
            # Сортируем текст в строке по X-координате
            sorted_row = row_df.sort_values('left')
            
            # Группируем текст по колонкам
            for col_start, col_end in column_positions:
                col_texts = []
                
                for _, word in sorted_row.iterrows():
                    word_left = word['left']
                    word_right = word['left'] + word['width']
                    
                    # Проверяем, попадает ли слово в эту колонку
                    if (word_left >= col_start and word_right <= col_end) or \
                       (word_left < col_end and word_right > col_start):
                        col_texts.append(word['text'])
                
                # Объединяем тексты в колонке
                col_text = ' '.join(col_texts) if col_texts else ''
                row_data.append(col_text)
            
            table_data.append(row_data)
        
        return table_data
    
    def _detect_column_positions(self, rows: List[pd.DataFrame]) -> List[Tuple[float, float]]:
        """
        Определяет позиции колонок на основе анализа всех строк.
        
        Args:
            rows: Список строк таблицы
            
        Returns:
            Список кортежей (start, end) для каждой колонки
        """
        # Собираем все X-координаты из всех строк
        all_x_positions = []
        
        for row_df in rows:
            for _, word in row_df.iterrows():
                all_x_positions.extend([word['left'], word['left'] + word['width']])
        
        if not all_x_positions:
            return []
        
        # Сортируем позиции
        all_x_positions.sort()
        
        # Определяем границы колонок на основе кластеризации
        column_boundaries = self._cluster_x_positions(all_x_positions)
        
        # Создаем пары (start, end) для колонок
        column_positions = []
        for i in range(0, len(column_boundaries) - 1, 2):
            if i + 1 < len(column_boundaries):
                column_positions.append((column_boundaries[i], column_boundaries[i + 1]))
        
        return column_positions
    
    def _cluster_x_positions(self, positions: List[float], threshold: float = 20.0) -> List[float]:
        """
        Кластеризует X-координаты для определения границ колонок.
        
        Args:
            positions: Список X-координат
            threshold: Порог для кластеризации
            
        Returns:
            Список границ колонок
        """
        if not positions:
            return []
        
        # Простая кластеризация
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                clusters.append(current_cluster)
                current_cluster = [pos]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Возвращаем центры кластеров
        boundaries = []
        for cluster in clusters:
            if cluster:
                boundaries.append(sum(cluster) / len(cluster))
        
        return boundaries
