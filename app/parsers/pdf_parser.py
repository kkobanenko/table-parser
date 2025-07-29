import os
import pdfplumber
import tabula
import pandas as pd
from typing import List, Dict, Optional
from config.pdf_parser_settings import PDFParserSettings
from utils.logger import setup_logger

logger = setup_logger('pdf_parser')


class PDFParser:
    """
    Продвинутый парсер для извлечения таблиц из PDF-документов
    Поддерживает различные стратегии и настройки извлечения
    """

    def __init__(self, settings: Optional[Dict] = None):
        """
        Инициализация парсера с настройками

        Args:
            settings (Optional[Dict]): Пользовательские настройки парсера
        """
        self.settings = settings or PDFParserSettings.load_settings()
        self.logger = logger

    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Основной метод извлечения таблиц из PDF

        Args:
            file_path (str): Путь к PDF файлу

        Returns:
            List[Dict]: Список извлеченных таблиц
        """
        all_tables = []

        # Обработка для каждого угла поворота
        for angle in self.settings.get('rotation_angles', [0]):
            rotated_tables = self._extract_tables_from_angle(file_path, angle)
            all_tables.extend(rotated_tables)

        return self._deduplicate_tables(all_tables)

    def _extract_tables_from_angle(self, file_path: str, angle: int) -> List[Dict]:
        """
        Извлечение таблиц для конкретного угла поворота

        Args:
            file_path (str): Путь к PDF файлу
            angle (int): Угол поворота

        Returns:
            List[Dict]: Список таблиц
        """
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Применение настроек парсера
                    page_tables = page.extract_tables(
                        line_margin=self.settings.get('line_margin', 0.5),
                        char_margin=self.settings.get('char_margin', 2.0),
                        vertical_strategy=self.settings.get('vertical_strategy', 'lines'),
                        horizontal_strategy=self.settings.get('horizontal_strategy', 'lines')
                    )

                    for table_idx, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])

                            tables.append({
                                'data': df,
                                'sheet_name': f'PDF_Table_p{page_num + 1}_t{table_idx + 1}_rot{angle}',
                                'source': 'pdfplumber',
                                'page': page_num + 1,
                                'rotation': angle,
                                'extraction_settings': self.settings
                            })

        except Exception as e:
            self.logger.error(f"Ошибка извлечения таблиц для угла {angle}: {e}")

        return tables

    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Удаление дубликатов таблиц

        Args:
            tables (List[Dict]): Список таблиц

        Returns:
            List[Dict]: Список уникальных таблиц
        """
        unique_tables = []
        seen_hashes = set()

        for table in tables:
            # Создание хеша для сравнения
            table_hash = hash(tuple(map(tuple, table['data'].values)))

            if table_hash not in seen_hashes:
                seen_hashes.add(table_hash)
                unique_tables.append(table)

        return unique_tables

    def fallback_table_extraction(self, file_path: str) -> List[Dict]:
        """
        Резервный метод извлечения таблиц с использованием Tabula

        Args:
            file_path (str): Путь к PDF файлу

        Returns:
            List[Dict]: Список таблиц
        """
        try:
            # Используем настройки Tabula из PDFParserSettings
            dfs = tabula.read_pdf(
                file_path,
                pages='all',
                multiple_tables=True,
                guess=self.settings.get('guess', True)
            )

            tables = []
            for idx, df in enumerate(dfs):
                if not df.empty:
                    tables.append({
                        'data': df,
                        'sheet_name': f'Tabula_Table_{idx + 1}',
                        'source': 'tabula'
                    })

            return tables

        except Exception as e:
            self.logger.error(f"Ошибка резервного извлечения таблиц: {e}")
            return []