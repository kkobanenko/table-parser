import os
import tempfile
from datetime import datetime
from typing import Dict, List
import pandas as pd
from io import BytesIO
import json
import time

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.csv_parser import CSVParser
from config.settings import settings
from processors.table_processor import TableProcessor
from utils.logger import setup_logger

logger = setup_logger('file_processor')


class FileProcessor:
    """
    Основной процессор для обработки файлов различных форматов
    Координирует извлечение и трансформацию таблиц
    """

    def __init__(self):
        """
        Инициализация процессора с парсерами для разных форматов
        """
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.doc': DOCXParser(),
            '.csv': CSVParser()
        }
        self.table_processor = TableProcessor()

    def process_file(self, uploaded_file) -> Dict:
        """
        Основной метод обработки загруженного файла

        Args:
            uploaded_file: Загруженный файл

        Returns:
            Dict: Результат обработки файла
        """
        start_time = time.time()

        # Проверка размера файла
        if len(uploaded_file.getvalue()) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Размер файла превышает {settings.MAX_FILE_SIZE_MB} МБ")

        # Временное сохранение файла
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            # Определение парсера по расширению
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            parser = self.parsers.get(file_ext)

            if not parser:
                raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")

            # Извлечение таблиц
            tables = parser.extract_tables(file_path)

            # Обработка таблиц
            processed_tables = []
            for table_data in tables:
                processed_table = self.table_processor.process_table(
                    table_data['data'],
                    table_data.get('sheet_name', 'Sheet')
                )
                processed_tables.append(processed_table)

            # Создание Excel файла
            output_bytes = self._create_excel(processed_tables, uploaded_file.name)

            # Генерация имени файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(uploaded_file.name)[0]
            output_filename = f"{base_name}_processed_{timestamp}.xlsx"

            processing_time = round(time.time() - start_time, 2)

            return {
                'data': output_bytes,
                'output_filename': output_filename,
                'tables_count': len(processed_tables),
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Ошибка обработки файла: {e}", exc_info=True)
            raise
        finally:
            # Удаление временного файла
            if os.path.exists(file_path):
                os.unlink(file_path)

    def _create_excel(self, tables: List[Dict], original_filename: str) -> bytes:
        """
        Создание Excel файла из обработанных таблиц

        Args:
            tables (List[Dict]): Список обработанных таблиц
            original_filename (str): Имя исходного файла

        Returns:
            bytes: Байты Excel файла
        """
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            used_sheet_names = set()

            for table_info in tables:
                df = table_info['dataframe']
                sheet_name = table_info['sheet_name']

                # Обеспечение уникальности имени листа
                base_name = sheet_name
                counter = 1
                while sheet_name in used_sheet_names:
                    sheet_name = f"{base_name}_{counter}"
                    counter += 1

                used_sheet_names.add(sheet_name)

                # Запись таблицы
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)
        return output.getvalue()