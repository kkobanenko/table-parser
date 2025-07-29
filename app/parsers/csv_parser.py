import pandas as pd
import chardet
from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger('csv_parser')


class CSVParser:
    """Парсер для CSV и текстовых файлов"""

    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Извлечение таблиц из CSV/текстового файла

        Args:
            file_path (str): Путь к файлу

        Returns:
            List[Dict]: Список извлеченных таблиц
        """
        tables = []

        try:
            # Определение кодировки
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']

            # Пробуем различные разделители
            delimiters = [',', ';', '\t', '|']

            for delimiter in delimiters:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        delimiter=delimiter,
                        on_bad_lines='skip'
                    )

                    if not df.empty and len(df.columns) > 1:
                        tables.append({
                            'data': df,
                            'sheet_name': 'CSV_Table',
                            'source': 'csv_parser',
                            'delimiter': delimiter
                        })
                        break

                except Exception as e:
                    logger.debug(f"Ошибка с разделителем {delimiter}: {e}")

        except Exception as e:
            logger.error(f"Ошибка парсинга CSV: {e}")

        return tables