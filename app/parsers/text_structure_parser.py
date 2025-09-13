import cv2
import numpy as np
import pandas as pd
import pytesseract
from typing import List, Dict, Tuple, Any, Optional
from utils.logger import setup_logger
from utils.image_cleaner import ImageCleaner

logger = setup_logger('text_structure_parser')


class TextStructureParser:
    """
    Парсер таблиц на основе анализа структуры текста.
    Использует координаты слов для определения табличной структуры.
    """
    
    def __init__(
        self,
        ocr_psm: int = 6,
        ocr_lang: str = 'eng',
        row_tolerance: int = 15,
        col_tolerance: int = 20
    ):
        self.ocr_psm = ocr_psm
        self.ocr_lang = ocr_lang
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance
        self.image_cleaner = ImageCleaner()
    
    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из изображения, анализируя структуру текста.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Не удалось загрузить изображение: %s", image_path)
            return []
        
        tables: List[Dict[str, Any]] = []
        
        # Пробуем разные методы очистки изображения
        cleaning_methods = ["adaptive", "strict", "color_filter"]
        
        for method in cleaning_methods:
            try:
                logger.info(f"Пробуем метод очистки: {method}")
                cleaned_img = self.image_cleaner.clean_image(img, method)
                
                # Получаем данные OCR
                ocr_data = self._get_ocr_data(cleaned_img)
                if ocr_data.empty:
                    continue
                
                # Анализируем структуру текста
                table_data = self._analyze_text_structure(ocr_data)
                if table_data:
                    df = pd.DataFrame(table_data)
                    tables.append({
                        "data": df,
                        "sheet_name": f"TextStructure_{method}",
                        "source": "text_structure",
                        "cleaning_method": method
                    })
                    logger.info(f"✅ Найдена таблица методом {method}: {df.shape}")
                
            except Exception as e:
                logger.warning(f"Ошибка при обработке методом {method}: {e}")
                continue
        
        return tables
    
    def _get_ocr_data(self, img: np.ndarray) -> pd.DataFrame:
        """
        Получает детальные данные OCR с координатами слов.
        """
        try:
            # Используем улучшенные параметры OCR для таблиц
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
    
    def _analyze_text_structure(self, ocr_data: pd.DataFrame) -> List[List[str]]:
        """
        Анализирует структуру текста для определения таблицы.
        """
        if ocr_data.empty:
            return []
        
        # Сортируем по координатам
        ocr_data = ocr_data.sort_values(["top", "left"])
        
        # Группируем слова по строкам
        rows = self._group_words_by_rows(ocr_data)
        
        # Анализируем структуру колонок
        table_data = self._analyze_columns(rows)
        
        return table_data
    
    def _group_words_by_rows(self, ocr_data: pd.DataFrame) -> List[List[Dict]]:
        """
        Группирует слова по строкам на основе Y-координат.
        """
        rows = []
        current_row = []
        last_top = None
        
        for _, word in ocr_data.iterrows():
            if last_top is None or abs(word["top"] - last_top) <= self.row_tolerance:
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
    
    def _analyze_columns(self, rows: List[List[Dict]]) -> List[List[str]]:
        """
        Анализирует структуру колонок в строках.
        """
        if not rows:
            return []
        
        # Определяем количество колонок на основе анализа всех строк
        column_positions = self._detect_column_positions(rows)
        
        if not column_positions:
            # Если не удалось определить колонки, возвращаем простую структуру
            return [[word["text"] for word in row] for row in rows]
        
        # Создаем таблицу с определенными колонками
        table_data = []
        for row in rows:
            table_row = [""] * len(column_positions)
            
            for word in row:
                # Определяем, в какую колонку попадает слово
                col_index = self._find_column_index(word["left"], column_positions)
                if col_index is not None:
                    if table_row[col_index]:
                        table_row[col_index] += " " + word["text"]
                    else:
                        table_row[col_index] = word["text"]
            
            table_data.append(table_row)
        
        return table_data
    
    def _detect_column_positions(self, rows: List[List[Dict]]) -> List[int]:
        """
        Определяет позиции колонок на основе анализа всех строк.
        """
        if not rows:
            return []
        
        # Собираем все X-координаты слов
        all_x_positions = []
        for row in rows:
            for word in row:
                all_x_positions.append(word["left"])
        
        if not all_x_positions:
            return []
        
        # Сортируем позиции
        all_x_positions.sort()
        
        # Кластеризуем позиции для определения колонок
        column_positions = []
        current_cluster = [all_x_positions[0]]
        
        for x in all_x_positions[1:]:
            if x - current_cluster[-1] <= self.col_tolerance:
                current_cluster.append(x)
            else:
                # Сохраняем среднюю позицию кластера
                column_positions.append(int(sum(current_cluster) / len(current_cluster)))
                current_cluster = [x]
        
        # Добавляем последний кластер
        if current_cluster:
            column_positions.append(int(sum(current_cluster) / len(current_cluster)))
        
        return sorted(column_positions)
    
    def _find_column_index(self, word_left: int, column_positions: List[int]) -> Optional[int]:
        """
        Определяет индекс колонки для слова на основе его X-координаты.
        """
        if not column_positions:
            return None
        
        # Находим ближайшую колонку
        min_distance = float('inf')
        closest_index = None
        
        for i, col_pos in enumerate(column_positions):
            distance = abs(word_left - col_pos)
            if distance < min_distance and distance <= self.col_tolerance:
                min_distance = distance
                closest_index = i
        
        return closest_index
