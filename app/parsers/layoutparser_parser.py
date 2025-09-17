"""
LayoutParser Parser для извлечения таблиц из документов.

LayoutParser - это библиотека для анализа структуры документов,
специализирующаяся на детекции layout элементов и извлечении таблиц.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import layoutparser as lp
    import cv2
    from PIL import Image
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    logger.warning("⚠️ LayoutParser не установлен. Установите: pip install layoutparser[ocr]")


class LayoutParserParser:
    """
    Парсер для извлечения таблиц с использованием LayoutParser.
    
    LayoutParser предоставляет возможности:
    - Детекции layout элементов (текст, таблицы, изображения)
    - Анализа структуры документа
    - Извлечения таблиц с высокой точностью
    - Интеграции с OCR движками
    """
    
    def __init__(self, 
                 model_name: str = 'lp://EfficientDete/PubLayNet',
                 confidence_threshold: float = 0.8,
                 ocr_agent: str = 'tesseract'):
        """
        Инициализация LayoutParser парсера.
        
        Args:
            model_name: Название модели для детекции layout
            confidence_threshold: Порог уверенности для детекции
            ocr_agent: OCR агент для извлечения текста
        """
        if not LAYOUTPARSER_AVAILABLE:
            raise ImportError("LayoutParser не установлен. Установите: pip install layoutparser[ocr]")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.ocr_agent = ocr_agent
        
        # Инициализация модели и OCR агента
        self.model = None
        self.ocr = None
        self._initialize_model()
        
        logger.info(f"✅ LayoutParser парсер инициализирован: {model_name}")
    
    def _initialize_model(self):
        """Инициализация модели LayoutParser и OCR агента."""
        try:
            # Упрощенная инициализация - используем только OCR
            self.model = None  # Пока не используем модель детекции
            
            # Инициализация OCR агента
            if self.ocr_agent == 'tesseract':
                self.ocr = lp.TesseractAgent()
            elif self.ocr_agent == 'gcv':
                self.ocr = lp.GCVAgent()
            else:
                logger.warning(f"⚠️ Неизвестный OCR агент: {self.ocr_agent}, используем Tesseract")
                self.ocr = lp.TesseractAgent()
            
            logger.info(f"🔍 OCR агент инициализирован: {self.ocr_agent}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации LayoutParser модели: {e}")
            raise
    
    def extract_tables(self, file_path: str):
        """
        Извлекает таблицы из документа.
        
        Args:
            file_path: Путь к файлу (PDF или изображение)
            
        Returns:
            Список таблиц, каждая таблица - список строк, каждая строка - список ячеек
        """
        if not self.ocr:
            logger.error("❌ OCR агент не инициализирован")
            return []
        
        try:
            logger.info(f"🔄 Обрабатываем файл через LayoutParser: {Path(file_path).name}")
            
            # Загрузка изображения
            image = self._load_image(file_path)
            if image is None:
                return []
            
            # Если модель не инициализирована, используем только OCR
            if self.model is None:
                logger.info("📄 Модель детекции не инициализирована, используем только OCR")
                tables, cells = self._extract_tables_ocr_only(image, return_cells=True)
                logger.info(f"✅ LayoutParser извлек {len(tables)} таблиц (OCR only)")
                return { 'tables': tables, 'cells': cells }
            
            # Детекция layout элементов
            layout = self.model.detect(image)
            logger.info(f"📄 Найдено {len(layout)} layout элементов")
            
            # Фильтрация табличных элементов
            table_elements = self._filter_table_elements(layout)
            logger.info(f"📊 Найдено {len(table_elements)} табличных элементов")
            
            # Извлечение таблиц
            tables = self._extract_tables_from_elements(image, table_elements)
            # Собираем ячейки (bbox) из элементов
            cells = []
            try:
                for row in table_elements:
                    # table_elements это список элементов, не по строкам; сохраним bbox каждого
                    x1, y1, x2, y2 = row.coordinates
                    cells.append({ 'bbox': [float(x1), float(y1), float(x2), float(y2)] })
            except Exception:
                pass
            logger.info(f"✅ LayoutParser извлек {len(tables)} таблиц")
            return { 'tables': tables, 'cells': cells }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки файла через LayoutParser: {e}")
            return []
    
    def _extract_tables_ocr_only(self, image: np.ndarray, return_cells: bool = False):
        """
        Извлекает таблицы используя только OCR без детекции layout.
        
        Args:
            image: Изображение для обработки
            
        Returns:
            Список таблиц
        """
        try:
            # Конвертируем в PIL Image для OCR
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Извлекаем текст через OCR
            ocr_result = self.ocr.detect(pil_image)
            
            if not ocr_result:
                logger.warning("⚠️ OCR не нашел текст")
                return []
            
            # Простая группировка текста в таблицы + сбор ячеек
            tables, cells = self._group_text_into_tables(ocr_result, return_cells=True)
            return (tables, cells) if return_cells else tables
            
        except Exception as e:
            logger.error(f"❌ Ошибка OCR-only извлечения: {e}")
            return []
    
    def _group_text_into_tables(self, ocr_result, return_cells: bool = False):
        """
        Группирует OCR результат в таблицы.
        
        Args:
            ocr_result: Результат OCR
            
        Returns:
            Список таблиц или (таблицы, ячейки)
        """
        try:
            # Простая реализация - группируем по строкам
            lines = []
            items = []
            
            for element in ocr_result:
                if hasattr(element, 'text') and hasattr(element, 'block'):
                    text = element.text.strip()
                    if text:
                        lines.append((text, element))
            
            if not lines:
                return ([] , []) if return_cells else []
            
            # Группируем по строкам по Y-координате
            # Подготовим элементы с координатами
            prepared = []
            for text, el in lines:
                try:
                    coords = el.block.coordinates
                    x1, y1, x2, y2 = coords
                    prepared.append({
                        'text': text,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'y': float(y1),
                        'x': float(x1)
                    })
                except Exception:
                    prepared.append({ 'text': text, 'bbox': [0,0,0,0], 'y': 0.0, 'x': 0.0 })

            # Сортируем по y, затем группируем с допуском
            prepared.sort(key=lambda it: it['y'])
            rows = []
            row_cells = []
            y_tol = 12.0
            current = []
            for it in prepared:
                if not current:
                    current = [it]
                else:
                    if abs(it['y'] - current[0]['y']) <= y_tol:
                        current.append(it)
                    else:
                        rows.append(current)
                        current = [it]
            if current:
                rows.append(current)

            tables = []
            cells = []
            for r in rows:
                # по X
                r.sort(key=lambda it: it['x'])
                row_texts = [it['text'] for it in r]
                tables.append(row_texts)
                for it in r:
                    cells.append({ 'bbox': it['bbox'] })

            # Одна таблица из всех строк
            tables_wrapped = [tables] if tables else []
            return (tables_wrapped, cells) if return_cells else tables_wrapped
            
        except Exception as e:
            logger.error(f"❌ Ошибка группировки текста: {e}")
            return []
    
    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Загружает изображение из файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Изображение в виде numpy массива или None
        """
        try:
            if file_path.lower().endswith('.pdf'):
                # Для PDF файлов нужно сначала конвертировать в изображение
                # Здесь используем простую загрузку через cv2
                # В реальном проекте лучше использовать pdf2image
                logger.warning("⚠️ PDF файлы требуют предварительной конвертации в изображения")
                return None
            
            # Загрузка изображения
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"❌ Не удалось загрузить изображение: {file_path}")
                return None
            
            # Конвертация из BGR в RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"📷 Изображение загружено: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки изображения: {e}")
            return None
    
    def _filter_table_elements(self, layout: lp.Layout) -> List[lp.TextBlock]:
        """
        Фильтрует табличные элементы из layout.
        
        Args:
            layout: Layout объект с детектированными элементами
            
        Returns:
            Список табличных элементов
        """
        table_elements = []
        
        try:
            for element in layout:
                # Проверяем, является ли элемент таблицей
                if self._is_table_element(element):
                    table_elements.append(element)
        
        except Exception as e:
            logger.error(f"❌ Ошибка фильтрации табличных элементов: {e}")
        
        return table_elements
    
    def _is_table_element(self, element) -> bool:
        """
        Проверяет, является ли элемент таблицей.
        
        Args:
            element: Layout элемент
            
        Returns:
            True, если элемент может быть таблицей
        """
        try:
            # Проверяем тип элемента
            element_type = getattr(element, 'type', None)
            if element_type == 'Table':
                return True
            
            # Проверяем размеры элемента
            coordinates = element.coordinates
            width = coordinates[2] - coordinates[0]
            height = coordinates[3] - coordinates[1]
            
            # Таблица должна иметь разумные размеры
            if width < 100 or height < 50:
                return False
            
            # Проверяем соотношение сторон
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки табличного элемента: {e}")
            return False
    
    def _extract_tables_from_elements(self, image: np.ndarray, table_elements: List) -> List[List[List[str]]]:
        """
        Извлекает таблицы из табличных элементов.
        
        Args:
            image: Исходное изображение
            table_elements: Список табличных элементов
            
        Returns:
            Список таблиц
        """
        tables = []
        
        try:
            for i, element in enumerate(table_elements):
                logger.info(f"📊 Обрабатываем табличный элемент {i + 1}")
                
                # Извлекаем таблицу из элемента
                table = self._extract_table_from_element(image, element)
                if table and len(table) > 1:  # Таблица должна иметь заголовки и данные
                    tables.append(table)
                    logger.info(f"📋 Создана таблица: {len(table)} строк × {len(table[0])} колонок")
        
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения таблиц из элементов: {e}")
        
        return tables
    
    def _extract_table_from_element(self, image: np.ndarray, element) -> List[List[str]]:
        """
        Извлекает таблицу из одного элемента.
        
        Args:
            image: Исходное изображение
            element: Табличный элемент
            
        Returns:
            Таблица в виде списка списков строк
        """
        try:
            # Обрезаем изображение по границам элемента
            cropped_image = element.crop_image(image)
            
            # Детектируем layout в обрезанном изображении
            cropped_layout = self.model.detect(cropped_image)
            
            # Группируем элементы по строкам
            rows = self._group_elements_into_rows(cropped_layout)
            
            # Создаем таблицу из строк
            table = self._create_table_from_rows(cropped_image, rows)
            
            return table
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения таблицы из элемента: {e}")
            return []
    
    def _group_elements_into_rows(self, layout: lp.Layout) -> List[List[lp.TextBlock]]:
        """
        Группирует элементы в строки.
        
        Args:
            layout: Layout объект
            
        Returns:
            Список строк, каждая строка - список элементов
        """
        rows = []
        
        try:
            # Сортируем элементы по Y-координате
            sorted_elements = sorted(layout, key=lambda e: e.coordinates[1])
            
            current_row = []
            current_y = None
            y_tolerance = 20  # Допустимое отклонение по Y для элементов одной строки
            
            for element in sorted_elements:
                y_coord = element.coordinates[1]
                
                if current_y is None or abs(y_coord - current_y) <= y_tolerance:
                    current_row.append(element)
                    current_y = y_coord if current_y is None else current_y
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [element]
                    current_y = y_coord
            
            if current_row:
                rows.append(current_row)
        
        except Exception as e:
            logger.error(f"❌ Ошибка группировки элементов в строки: {e}")
        
        return rows
    
    def _create_table_from_rows(self, image: np.ndarray, rows: List[List[lp.TextBlock]]) -> List[List[str]]:
        """
        Создает таблицу из строк элементов.
        
        Args:
            image: Изображение
            rows: Список строк элементов
            
        Returns:
            Таблица в виде списка списков строк
        """
        table = []
        
        try:
            for row_elements in rows:
                row = []
                
                # Сортируем элементы в строке по X-координате
                sorted_row_elements = sorted(row_elements, key=lambda e: e.coordinates[0])
                
                for element in sorted_row_elements:
                    # Извлекаем текст из элемента
                    text = self._extract_text_from_element(image, element)
                    if text:
                        row.append(text)
                
                if row:
                    table.append(row)
        
        except Exception as e:
            logger.error(f"❌ Ошибка создания таблицы из строк: {e}")
        
        return table
    
    def _extract_text_from_element(self, image: np.ndarray, element: lp.TextBlock) -> str:
        """
        Извлекает текст из элемента.
        
        Args:
            image: Изображение
            element: Текстовый элемент
            
        Returns:
            Извлеченный текст
        """
        try:
            # Обрезаем изображение по границам элемента
            cropped_image = element.crop_image(image)
            
            # Выполняем OCR
            text = self.ocr.detect(cropped_image)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения текста из элемента: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Возвращает информацию о используемой модели.
        
        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_type": f"LayoutParser_{self.model_name}",
            "layout_model": self.model_name,
            "ocr_agent": self.ocr_agent,
            "confidence_threshold": str(self.confidence_threshold),
            "description": f"LayoutParser model with {self.model_name} layout detection and {self.ocr_agent} OCR"
        }
