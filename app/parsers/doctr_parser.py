"""
DocTR (Mindee) Parser для извлечения таблиц из документов.

DocTR - это библиотека для OCR с поддержкой PyTorch и TensorFlow,
специализирующаяся на анализе документов и извлечении текста.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    logger.warning("⚠️ DocTR не установлен. Установите: pip install python-doctr[torch]")


class DocTRParser:
    """
    Парсер для извлечения таблиц с использованием DocTR (Mindee).
    
    DocTR предоставляет мощные возможности OCR с поддержкой:
    - Детекции текста
    - Распознавания текста
    - Анализа структуры документа
    - Извлечения таблиц
    """
    
    def __init__(self, 
                 det_arch: str = 'db_resnet50',
                 reco_arch: str = 'crnn_vgg16_bn',
                 pretrained: bool = True,
                 assume_straight_pages: bool = True,
                 preserve_aspect_ratio: bool = False):
        """
        Инициализация DocTR парсера.
        
        Args:
            det_arch: Архитектура модели детекции
            reco_arch: Архитектура модели распознавания
            pretrained: Использовать предобученные веса
            assume_straight_pages: Предполагать прямые страницы
            preserve_aspect_ratio: Сохранять соотношение сторон
        """
        if not DOCTR_AVAILABLE:
            raise ImportError("DocTR не установлен. Установите: pip install python-doctr[torch]")
        
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self.assume_straight_pages = assume_straight_pages
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Инициализация модели
        self.model = None
        self._initialize_model()
        
        logger.info(f"✅ DocTR парсер инициализирован: {det_arch} + {reco_arch}")
    
    def _initialize_model(self):
        """Инициализация модели DocTR."""
        try:
            self.model = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained,
                assume_straight_pages=self.assume_straight_pages,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                resolve_blocks=True,  # Группировать строки в блоки
                resolve_lines=True    # Группировать слова в строки
            )
            logger.info("🚀 DocTR модель загружена успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации DocTR модели: {e}")
            raise
    
    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Извлекает таблицы из документа.
        
        Args:
            file_path: Путь к файлу (PDF или изображение)
            
        Returns:
            Список таблиц, каждая таблица - список строк, каждая строка - список ячеек
        """
        if not self.model:
            logger.error("❌ Модель DocTR не инициализирована")
            return []
        
        try:
            logger.info(f"🔄 Обрабатываем файл через DocTR: {Path(file_path).name}")
            
            # Загрузка документа
            if file_path.lower().endswith('.pdf'):
                doc = DocumentFile.from_pdf(file_path)
            else:
                doc = DocumentFile.from_images(file_path)
            
            logger.info(f"📄 Загружен документ: {len(doc)} страниц")
            
            # Анализ документа
            result = self.model(doc)
            
            # Извлечение таблиц
            tables = self._extract_tables_from_result(result)
            
            logger.info(f"✅ DocTR извлек {len(tables)} таблиц")
            return tables
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки файла через DocTR: {e}")
            return []
    
    def _extract_tables_from_result(self, result) -> List[List[List[str]]]:
        """
        Извлекает таблицы из результата DocTR.
        
        Args:
            result: Результат анализа DocTR
            
        Returns:
            Список таблиц
        """
        tables = []
        
        try:
            # Экспорт результата в словарь
            result_dict = result.export()
            
            for page_idx, page in enumerate(result_dict['pages']):
                logger.info(f"📄 Обрабатываем страницу {page_idx + 1}")
                
                # Извлекаем таблицы со страницы
                page_tables = self._extract_tables_from_page(page)
                tables.extend(page_tables)
                
                logger.info(f"📊 Страница {page_idx + 1}: найдено {len(page_tables)} таблиц")
        
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения таблиц из результата DocTR: {e}")
        
        return tables
    
    def _extract_tables_from_page(self, page: Dict[str, Any]) -> List[List[List[str]]]:
        """
        Извлекает таблицы со страницы.
        
        Args:
            page: Данные страницы
            
        Returns:
            Список таблиц со страницы
        """
        tables = []
        
        try:
            # Получаем все блоки со страницы
            blocks = page.get('blocks', [])
            logger.info(f"🔍 Найдено {len(blocks)} блоков на странице")
            
            # Группируем блоки по строкам для поиска таблиц
            table_blocks = self._group_blocks_into_tables(blocks)
            
            # Создаем таблицы из сгруппированных блоков
            for table_blocks_group in table_blocks:
                table = self._create_table_from_blocks(table_blocks_group)
                if table and len(table) > 1:  # Таблица должна иметь заголовки и данные
                    tables.append(table)
                    logger.info(f"📋 Создана таблица: {len(table)} строк × {len(table[0])} колонок")
        
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения таблиц со страницы: {e}")
        
        return tables
    
    def _group_blocks_into_tables(self, blocks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Группирует блоки в таблицы на основе их расположения.
        
        Args:
            blocks: Список блоков
            
        Returns:
            Список групп блоков, каждая группа - потенциальная таблица
        """
        if not blocks:
            return []
        
        try:
            # Сортируем блоки по Y-координате
            def get_y_coord(block):
                geometry = block.get('geometry', [0, 0, 0, 0])
                if isinstance(geometry, tuple):
                    geometry = list(geometry)
                return geometry[1]
            
            sorted_blocks = sorted(blocks, key=get_y_coord)
        except Exception as e:
            logger.error(f"❌ Ошибка сортировки блоков: {e}")
            return []
        
        # Группируем блоки по строкам
        rows = []
        current_row = []
        current_y = None
        y_tolerance = 20  # Допустимое отклонение по Y для элементов одной строки
        
        try:
            for i, block in enumerate(sorted_blocks):
                geometry = block.get('geometry', [0, 0, 0, 0])
                # Преобразуем geometry в список, если это кортеж
                if isinstance(geometry, tuple):
                    geometry = list(geometry)
                
                # geometry - это список кортежей [(x1, y1), (x2, y2)]
                if isinstance(geometry, list) and len(geometry) >= 2:
                    if isinstance(geometry[1], tuple):
                        # Берем y-координату из второго кортежа
                        y_coord = geometry[1][1]
                    else:
                        y_coord = geometry[1]
                else:
                    y_coord = 0
                
                # Отладочная информация
                if i < 3:  # Показываем только первые 3 блока
                    logger.info(f"🔍 Блок {i}: geometry={geometry}, y_coord={y_coord}, type={type(y_coord)}")
                
                if current_y is None or abs(y_coord - float(current_y)) <= y_tolerance:
                    current_row.append(block)
                    current_y = float(y_coord) if current_y is None else float(current_y)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [block]
                    current_y = float(y_coord)
        except Exception as e:
            logger.error(f"❌ Ошибка группировки блоков по строкам: {e}")
            return []
        
        if current_row:
            rows.append(current_row)
        
        # Фильтруем строки, которые могут быть табличными
        table_rows = []
        for row in rows:
            if self._is_table_row(row):
                table_rows.append(row)
        
        # Группируем строки в таблицы
        tables = []
        if table_rows:
            # Простая группировка: все строки в одну таблицу
            # В будущем можно улучшить алгоритм группировки
            tables.append(table_rows)
        
        return tables
    
    def _is_table_row(self, row: List[Dict[str, Any]]) -> bool:
        """
        Проверяет, является ли строка табличной.
        
        Args:
            row: Строка блоков
            
        Returns:
            True, если строка может быть частью таблицы
        """
        if len(row) < 2:  # Таблица должна иметь минимум 2 колонки
            return False
        
        # Проверяем, что все блоки в строке содержат текст
        for block in row:
            lines = block.get('lines', [])
            if not lines:
                return False
            
            # Проверяем, что есть хотя бы одно слово
            has_text = False
            for line in lines:
                words = line.get('words', [])
                if words:
                    has_text = True
                    break
            
            if not has_text:
                return False
        
        return True
    
    def _create_table_from_blocks(self, table_blocks: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """
        Создает таблицу из блоков.
        
        Args:
            table_blocks: Группа строк блоков
            
        Returns:
            Таблица в виде списка списков строк
        """
        table = []
        
        try:
            for row_blocks in table_blocks:
                row = []
                
                # Сортируем блоки в строке по X-координате
                sorted_row_blocks = sorted(row_blocks, key=lambda b: b.get('geometry', [0, 0, 0, 0])[0])
                
                for block in sorted_row_blocks:
                    # Извлекаем текст из блока
                    block_text = self._extract_text_from_block(block)
                    if block_text:
                        row.append(block_text)
                
                if row:
                    table.append(row)
        
        except Exception as e:
            logger.error(f"❌ Ошибка создания таблицы из блоков: {e}")
        
        return table
    
    def _extract_text_from_block(self, block: Dict[str, Any]) -> str:
        """
        Извлекает текст из блока.
        
        Args:
            block: Блок с текстом
            
        Returns:
            Извлеченный текст
        """
        try:
            lines = block.get('lines', [])
            words = []
            
            for line in lines:
                line_words = line.get('words', [])
                for word in line_words:
                    word_text = word.get('value', '').strip()
                    if word_text:
                        words.append(word_text)
            
            return ' '.join(words)
        
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения текста из блока: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Возвращает информацию о используемой модели.
        
        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_type": f"DocTR_{self.det_arch}_{self.reco_arch}",
            "detection_arch": self.det_arch,
            "recognition_arch": self.reco_arch,
            "pretrained": str(self.pretrained),
            "description": f"DocTR OCR model with {self.det_arch} detection and {self.reco_arch} recognition"
        }
