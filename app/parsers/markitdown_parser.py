"""
Парсер таблиц с использованием библиотеки MarkItDown от Microsoft.
MarkItDown конвертирует различные форматы файлов в Markdown,
сохраняя структуру документов для анализа LLM.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from utils.logger import setup_logger

logger = setup_logger('markitdown_parser')


class MarkItDownParser:
    """
    Парсер таблиц с использованием библиотеки MarkItDown.
    Использует Microsoft MarkItDown для конвертации файлов в Markdown
    и извлечения табличных данных.
    """
    
    def __init__(self, enable_plugins: bool = False):
        """
        Инициализация парсера MarkItDown.
        
        Args:
            enable_plugins: Включить ли плагины MarkItDown для расширенной функциональности
        """
        self.enable_plugins = enable_plugins
        self.markitdown = None
        
        # Инициализируем MarkItDown только при первом использовании
        self._init_markitdown()
    
    def _init_markitdown(self):
        """
        Инициализирует библиотеку MarkItDown.
        """
        try:
            from markitdown import MarkItDown
            self.markitdown = MarkItDown(enable_plugins=self.enable_plugins)
            logger.info("✅ MarkItDown успешно инициализирован")
        except ImportError as e:
            logger.error(f"❌ Не удалось импортировать MarkItDown: {e}")
            logger.error("Установите библиотеку: pip install markitdown")
            self.markitdown = None
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации MarkItDown: {e}")
            self.markitdown = None
    
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из файла с использованием MarkItDown.
        
        Args:
            file_path: Путь к файлу для обработки
            
        Returns:
            Список словарей с данными таблиц
        """
        if self.markitdown is None:
            logger.warning("⚠️ MarkItDown недоступен, пропускаем парсинг")
            return []
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"❌ Файл не найден: {file_path}")
            return []
        
        tables: List[Dict[str, Any]] = []
        
        try:
            logger.info(f"🔄 Обрабатываем файл через MarkItDown: {file_path_obj.name}")
            
            # Конвертируем файл в Markdown
            result = self.markitdown.convert(str(file_path_obj))
            
            if not result or not result.text_content:
                logger.warning(f"⚠️ MarkItDown не смог извлечь содержимое из {file_path_obj.name}")
                return []
            
            # Логируем содержимое для отладки
            logger.info(f"📄 MarkItDown извлек содержимое ({len(result.text_content)} символов):")
            logger.info(f"Первые 500 символов: {result.text_content[:500]}")
            
            # Извлекаем таблицы из Markdown
            markdown_tables = self._extract_tables_from_markdown(result.text_content)
            
            for idx, table_data in enumerate(markdown_tables, start=1):
                if table_data and len(table_data) > 1:  # Минимум заголовок + 1 строка данных
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    
                    # Очищаем DataFrame от пустых значений
                    df = df.replace('', pd.NA).dropna(how='all', axis=0).dropna(how='all', axis=1)
                    
                    if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
                        tables.append({
                            "data": df,
                            "sheet_name": f"MarkItDown_{idx}",
                            "source": "markitdown",
                            "cleaning_method": "markdown_parsing"
                        })
                        logger.info(f"✅ Найдена таблица MarkItDown_{idx}: {df.shape}")
                    else:
                        logger.info(f"⏭️ Пропускаем пустую таблицу MarkItDown_{idx}")
                else:
                    logger.info(f"⏭️ Пропускаем таблицу MarkItDown_{idx} (недостаточно данных)")
            
            logger.info(f"🏁 MarkItDown обработка завершена: найдено {len(tables)} таблиц")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке файла {file_path_obj.name}: {e}")
            return []
        
        return tables
    
    def _extract_tables_from_markdown(self, markdown_content: str) -> List[List[List[str]]]:
        """
        Извлекает таблицы из Markdown контента.
        
        Args:
            markdown_content: Содержимое в формате Markdown
            
        Returns:
            Список таблиц, где каждая таблица представлена как список строк
        """
        tables = []
        
        # Паттерн для поиска таблиц в Markdown (GFM формат)
        table_pattern = r'\|(.+?)\|(?:\n\|(.+?)\|)*'
        
        # Находим все таблицы
        table_matches = re.finditer(table_pattern, markdown_content, re.MULTILINE | re.DOTALL)
        
        for match in table_matches:
            table_text = match.group(0)
            table_data = self._parse_markdown_table(table_text)
            if table_data:
                tables.append(table_data)
        
        # Дополнительный поиск таблиц с разделителями строк
        separator_pattern = r'\|[^|\n]+\|\n\|[-\s|:]+\|\n(\|[^|\n]+\|\n?)*'
        separator_matches = re.finditer(separator_pattern, markdown_content, re.MULTILINE)
        
        for match in separator_matches:
            table_text = match.group(0)
            table_data = self._parse_markdown_table(table_text)
            if table_data and table_data not in tables:  # Избегаем дублирования
                tables.append(table_data)
        
        return tables
    
    def _parse_markdown_table(self, table_text: str) -> Optional[List[List[str]]]:
        """
        Парсит отдельную таблицу из Markdown текста.
        
        Args:
            table_text: Текст таблицы в формате Markdown
            
        Returns:
            Список строк таблицы или None если таблица некорректная
        """
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:  # Минимум заголовок + разделитель
            return None
        
        table_data = []
        
        for line in lines:
            # Пропускаем строки-разделители (содержащие только |, -, :, пробелы)
            if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)*\|$', line):
                continue
            
            # Извлекаем ячейки из строки
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Убираем пустые элементы по краям
            
            if cells:  # Только если есть содержимое
                table_data.append(cells)
        
        # Проверяем, что все строки имеют одинаковое количество колонок
        if table_data:
            num_cols = len(table_data[0])
            if all(len(row) == num_cols for row in table_data):
                return table_data
            else:
                logger.warning(f"⚠️ Таблица имеет разное количество колонок в строках")
                return None
        
        return None
    
    def is_available(self) -> bool:
        """
        Проверяет доступность библиотеки MarkItDown.
        
        Returns:
            True если MarkItDown доступен, False иначе
        """
        return self.markitdown is not None
