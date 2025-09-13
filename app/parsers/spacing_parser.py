import cv2
import numpy as np
import pandas as pd
import pytesseract
from typing import List, Dict, Tuple, Any, Optional
from utils.logger import setup_logger
from utils.image_cleaner import ImageCleaner

logger = setup_logger('spacing_parser')


class SpacingParser:
    """
    Парсер таблиц на основе анализа пробелов между словами.
    Определяет колонки по характерным промежуткам в тексте.
    """
    
    def __init__(
        self,
        ocr_psm: int = 6,
        ocr_lang: str = 'eng',
        min_gap_size: int = 20,
        max_gap_size: int = 200
    ):
        self.ocr_psm = ocr_psm
        self.ocr_lang = ocr_lang
        self.min_gap_size = min_gap_size
        self.max_gap_size = max_gap_size
        self.image_cleaner = ImageCleaner()
    
    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы, анализируя пробелы между словами.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Не удалось загрузить изображение: %s", image_path)
            return []
        
        tables: List[Dict[str, Any]] = []
        
        # Пробуем разные методы очистки
        cleaning_methods = ["adaptive", "strict", "color_filter"]
        
        for method in cleaning_methods:
            try:
                logger.info(f"Пробуем метод очистки: {method}")
                cleaned_img = self.image_cleaner.clean_image(img, method)
                
                # Получаем данные OCR
                ocr_data = self._get_ocr_data(cleaned_img)
                if ocr_data.empty:
                    continue
                
                # Анализируем пробелы
                table_data = self._analyze_spacing(ocr_data)
                if table_data:
                    df = pd.DataFrame(table_data)
                    tables.append({
                        "data": df,
                        "sheet_name": f"Spacing_{method}",
                        "source": "spacing_analysis",
                        "cleaning_method": method
                    })
                    logger.info(f"✅ Найдена таблица методом {method}: {df.shape}")
                
            except Exception as e:
                logger.warning(f"Ошибка при обработке методом {method}: {e}")
                continue
        
        return tables
    
    def _get_ocr_data(self, img: np.ndarray) -> pd.DataFrame:
        """
        Получает детальные данные OCR.
        """
        try:
            config = f'--oem 3 --psm {self.ocr_psm}'
            data = pytesseract.image_to_data(
                img,
                config=config,
                lang=self.ocr_lang,
                output_type=pytesseract.Output.DATAFRAME
            )
            
            if data.empty:
                return data
            
            # Фильтруем пустые тексты
            data = data.dropna(subset=["text"])
            data["text"] = data["text"].astype(str)
            data = data[data["text"].str.strip() != ""]
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при получении OCR данных: {e}")
            return pd.DataFrame()
    
    def _analyze_spacing(self, ocr_data: pd.DataFrame) -> List[List[str]]:
        """
        Анализирует пробелы между словами для определения табличной структуры.
        """
        if ocr_data.empty:
            return []
        
        # Группируем слова по строкам
        rows = self._group_words_by_rows(ocr_data)
        
        # Анализируем пробелы в каждой строке
        table_data = []
        for row in rows:
            # Определяем колонки на основе пробелов
            columns = self._detect_columns_by_spacing(row)
            table_data.append(columns)
        
        return table_data
    
    def _group_words_by_rows(self, ocr_data: pd.DataFrame) -> List[List[Dict]]:
        """
        Группирует слова по строкам.
        """
        rows = []
        current_row = []
        last_top = None
        row_tolerance = 15
        
        for _, word in ocr_data.iterrows():
            if last_top is None or abs(word["top"] - last_top) <= row_tolerance:
                current_row.append({
                    "text": word["text"],
                    "left": word["left"],
                    "top": word["top"],
                    "width": word["width"],
                    "height": word["height"]
                })
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [{
                    "text": word["text"],
                    "left": word["left"],
                    "top": word["top"],
                    "width": word["width"],
                    "height": word["height"]
                }]
            last_top = word["top"]
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _detect_columns_by_spacing(self, row: List[Dict]) -> List[str]:
        """
        Определяет колонки в строке на основе анализа пробелов.
        """
        if not row:
            return []
        
        # Сортируем слова по X-координате
        sorted_words = sorted(row, key=lambda x: x["left"])
        
        # Анализируем промежутки между словами
        columns = []
        current_column = []
        
        for i, word in enumerate(sorted_words):
            current_column.append(word["text"])
            
            # Проверяем промежуток до следующего слова
            if i < len(sorted_words) - 1:
                next_word = sorted_words[i + 1]
                gap = next_word["left"] - (word["left"] + word["width"])
                
                # Если промежуток достаточно большой, это граница колонки
                if gap >= self.min_gap_size:
                    # Объединяем слова в текущей колонке
                    column_text = " ".join(current_column)
                    columns.append(column_text)
                    current_column = []
        
        # Добавляем последнюю колонку
        if current_column:
            column_text = " ".join(current_column)
            columns.append(column_text)
        
        return columns
    
    def _find_common_column_positions(self, rows: List[List[Dict]]) -> List[int]:
        """
        Находит общие позиции колонок на основе анализа всех строк.
        """
        if not rows:
            return []
        
        # Собираем все промежутки между словами
        all_gaps = []
        for row in rows:
            sorted_words = sorted(row, key=lambda x: x["left"])
            for i in range(len(sorted_words) - 1):
                gap = sorted_words[i + 1]["left"] - (sorted_words[i]["left"] + sorted_words[i]["width"])
                if gap >= self.min_gap_size:
                    all_gaps.append(gap)
        
        if not all_gaps:
            return []
        
        # Находим наиболее частые размеры промежутков
        gap_counts = {}
        for gap in all_gaps:
            # Округляем до ближайших 10 пикселей
            rounded_gap = round(gap / 10) * 10
            gap_counts[rounded_gap] = gap_counts.get(rounded_gap, 0) + 1
        
        # Сортируем по частоте
        common_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем наиболее частые промежутки
        return [gap for gap, count in common_gaps[:5] if count > 1]
